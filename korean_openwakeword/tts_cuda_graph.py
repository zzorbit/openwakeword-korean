"""Batch-one CUDA graph adapter retaining official prompt/talker/codec APIs.

The predictor graph uses unchanged MIT upstream primitives, with explicit
float32 sampling to match installed HF GenerationMixin. Deployment requires
a source-bound parity benchmark, not a claimed performance number.
"""
from types import SimpleNamespace


def initialize_cache_layers(graph):
    """Installed Transformers 4.57 uses a one-argument cache initializer."""
    import inspect
    import torch
    inner=getattr(graph,'pred_model',None) or graph.model
    config=inner.config
    heads=getattr(config,'num_key_value_heads',config.num_attention_heads)
    dim=getattr(config,'head_dim',config.hidden_size//config.num_attention_heads)
    dummy=torch.zeros(1,heads,1,dim,dtype=graph.dtype,device=graph.device)
    for layer in graph.static_cache.layers:
        if not layer.is_initialized:
            if len(inspect.signature(layer.lazy_initialization).parameters)==1:
                layer.lazy_initialization(dummy)
            else:
                layer.lazy_initialization(dummy,dummy)


def validate_predictor_arguments(kwargs):
    expected = {'max_new_tokens':15, 'do_sample':True, 'top_k':50,
                'top_p':1.0, 'temperature':.9}
    if any(kwargs.get(k) != v for k,v in expected.items()):
        raise ValueError('Unqualified predictor sampling controls')
    if tuple(kwargs['inputs_embeds'].shape[:2]) != (1,2):
        raise ValueError('The qualified graph only supports batch-one two-token predictor input')


def enable_predictor_graph(model, exact_shapes=False):
    import torch
    from . import tts_cuda_predictor as module
    from .tts_cuda_sampling import sample_logits
    # The official HF loop casts next-token logits to float32 before sampling.
    module.sample_logits = lambda logits, **kw: sample_logits(logits.float(), **kw)
    predictor = model.model.talker.code_predictor
    class CompatiblePredictorGraph(module.PredictorGraph):
        def _init_cache_layers(self):initialize_cache_layers(self)
        def _build_attention_masks(self):
            if not exact_shapes:return super()._build_attention_masks()
            if self.has_sliding_layers:raise ValueError('Exact predictor needs full attention')
            self.prefill_attn={'full_attention':None}
            self.decode_attn=[{'full_attention':None} for _ in self.decode_cache_positions]
    graph = CompatiblePredictorGraph(predictor, predictor.model.config,
        model.model.config.talker_config.hidden_size, device='cuda:0',
        dtype=torch.bfloat16, do_sample=True, top_k=50, top_p=1.0, temperature=.9)
    if exact_shapes:
        original_update=graph.static_cache.update
        lengths={id(graph.prefill_cache_pos):2,
                 **{id(pos):i+3 for i,pos in enumerate(graph.decode_cache_positions)}}
        def update(key,value,layer_idx,cache_kwargs=None):
            keys,values=original_update(key,value,layer_idx,cache_kwargs)
            n=lengths[id(cache_kwargs['cache_position'])]
            return keys[:,:,:n].contiguous(),values[:,:,:n].contiguous()
        graph.static_cache.update=update
    with torch.inference_mode():
        graph.capture()
    original = predictor.generate
    def generate(**kwargs):
        validate_predictor_arguments(kwargs)
        return SimpleNamespace(sequences=graph.run(kwargs['inputs_embeds']).unsqueeze(0))
    predictor.generate = generate
    return {'graph':graph, 'original_generate':original,
            'backend':'official_HF_with_BF16_predictor_CUDA_graph_float32_sampling_v1',
            'exact_attention_shapes':exact_shapes}


def enable_exact_talker_graph(model,max_seq_len=512):
    """Lazy per-length graph buckets retain original attention tensor shapes.

    A shared pool is safe because buckets execute sequentially and copy their
    result into a persistent output buffer before any subsequent replay.
    """
    import torch
    from transformers import StaticCache
    from transformers.modeling_outputs import BaseModelOutputWithPast
    inner=model.model.talker.model;config=inner.config
    if config.sliding_window is not None:raise ValueError('Full attention required')
    original=inner.forward
    class ExactCache(StaticCache):
        logical_length=0
        @property
        def is_compileable(self):return False
        def get_mask_sizes(self,cache_position,layer_idx):return self.logical_length,0
        def update(self,key,value,layer_idx,cache_kwargs=None):
            keys,values=super().update(key,value,layer_idx,cache_kwargs)
            n=self.logical_length
            return keys[:,:,:n].contiguous(),values[:,:,:n].contiguous()
    cache=ExactCache(config=config,max_cache_len=max_seq_len)
    heads=getattr(config,'num_key_value_heads',config.num_attention_heads)
    dim=getattr(config,'head_dim',config.hidden_size//config.num_attention_heads)
    dummy=torch.zeros(1,heads,1,dim,dtype=torch.bfloat16,device='cuda:0')
    for layer in cache.layers:layer.lazy_initialization(dummy)
    inputs=torch.zeros(1,1,config.hidden_size,dtype=torch.bfloat16,device='cuda:0')
    outputs=torch.empty_like(inputs)
    positions=torch.zeros(3,1,1,dtype=torch.long,device='cuda:0')
    cache_pos=torch.zeros(1,dtype=torch.long,device='cuda:0')
    graphs={};pool=torch.cuda.graph_pool_handle();state={'prefill_pending':True}
    def body():
        out=original(inputs_embeds=inputs,attention_mask=None,
            position_ids=positions,past_key_values=cache,cache_position=cache_pos,
            use_cache=True,output_hidden_states=False,output_attentions=False)
        outputs.copy_(out.last_hidden_state)
    def forward(**kwargs):
        embedded=kwargs['inputs_embeds']
        if embedded.shape[1]>1:
            state['prefill_pending']=True
            return original(**kwargs)
        if embedded.shape[:2]!=(1,1):raise ValueError('Batch-one single-token required')
        position=int(kwargs['cache_position'][0].item())
        if not 0<=position<max_seq_len:raise ValueError('Graph capacity exceeded')
        if state['prefill_pending']:
            cache.reset()
            past=kwargs['past_key_values']
            for i,layer in enumerate(past.layers):
                indices=torch.arange(layer.keys.shape[2],device='cuda:0')
                # Populate without the exact-length return view; no result needed.
                StaticCache.update(cache,layer.keys,layer.values,i,{'cache_position':indices})
            state['prefill_pending']=False
        if kwargs.get('attention_mask') is not None and not bool(kwargs['attention_mask'].all()):
            raise ValueError('Unpadded batch-one prompt required')
        cache.logical_length=position+1
        inputs.copy_(embedded);positions.copy_(kwargs['position_ids']);cache_pos.copy_(kwargs['cache_position'])
        if position not in graphs:
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):body()
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,pool=pool):body()
            torch.cuda.current_stream().wait_stream(stream)
            graphs[position]=graph
        graphs[position].replay()
        hidden=outputs.clone()
        return BaseModelOutputWithPast(last_hidden_state=hidden,past_key_values=cache,hidden_states=(hidden,))
    inner.forward=forward
    return {'cache':cache,'graphs':graphs,'original_forward':original,
            'backend':'official_HF_exact_length_CUDA_graph_talker_v1'}

