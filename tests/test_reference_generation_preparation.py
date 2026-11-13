"""New reference generation publishes clean review inputs, never rewrites old voices."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import clone_reference as preparation, reference_candidates as candidates, tts
from korean_openwakeword.common import digest, identity, read_json, read_rows, write_json
from korean_openwakeword.project import create_project, load_project
from test_korean_library_reference_candidates import pool  # noqa: F401


def wave(seconds=2, rate=24000, frequency=211):
    return (.07 * np.sin(2 * np.pi * frequency * np.arange(round(seconds * rate)) / rate)).astype('float32')


def raw(root, values, rate=24000):
    path = root / 'raw.wav'
    sf.write(path, values, rate, subtype='FLOAT')
    return {'path': 'raw.wav', 'sha256': digest(path), 'generation': {'model': 'synthetic-fixture', 'seed': 17, 'text': '예문'}}


def state(root):
    return {path.relative_to(root).as_posix(): digest(path) for path in root.rglob('*') if path.is_file()}


@pytest.mark.parametrize('tail_seconds', [16, 36])
def test_generation_prepares_long_tail_even_when_raw_duration_exceeds_old_maximum(tmp_path, tail_seconds):
    values = np.concatenate((np.zeros(24000, dtype='float32'), wave(), np.full(tail_seconds * 24000, .00002, dtype='float32')))
    reference = raw(tmp_path, values)
    source_sha = reference['sha256']
    target = tmp_path / 'review.wav'
    metadata = preparation.prepare_generated_reference(tmp_path, reference, target)
    assert metadata['policy'] == preparation.GENERATED_POLICY
    assert metadata['prepared_duration_s'] == pytest.approx(2.4)
    assert metadata['leading_removed_s'] == pytest.approx(.8)
    assert metadata['trailing_removed_s'] > tail_seconds - .3
    output, rate = sf.read(target, dtype='float32')
    assert rate == 24000 and np.array_equal(output, values[metadata['start_frame']:metadata['end_frame']])
    assert digest(tmp_path / 'raw.wav') == source_sha
    assert read_json(tmp_path / metadata['raw_receipt']['path'])['generation'] == reference['generation']
    before = state(tmp_path)
    assert preparation.verify_generated_reference(tmp_path, metadata).samefile(target)
    assert preparation.prepare_generated_reference(tmp_path, reference, target) == metadata
    assert state(tmp_path) == before
    if tail_seconds > 30:
        with pytest.raises(ValueError, match='duration_outlier'):
            preparation.prepare_clone_reference(tmp_path, reference)  # Existing clone policy is unchanged.


def test_generation_keeps_normal_raw_bytes_and_internal_pauses_exact(tmp_path):
    values = np.concatenate((wave(.7), np.zeros(24000, dtype='float32'), wave(.8)))
    reference = raw(tmp_path, values)
    target = tmp_path / 'review.wav'
    metadata = preparation.prepare_generated_reference(tmp_path, reference, target)
    assert not metadata['trimmed']
    assert target.read_bytes() == (tmp_path / 'raw.wav').read_bytes()
    assert metadata['prepared']['sha256'] == metadata['source']['sha256']


@pytest.mark.parametrize('kind', ['silence', 'nonfinite', 'stereo', 'clipping', 'too_short', 'voiced_too_long'])
def test_generation_still_rejects_hard_invalid_and_unrepairable_outputs(tmp_path, kind):
    values = wave()
    if kind == 'silence': values[:] = 0
    elif kind == 'nonfinite': values[100] = np.nan
    elif kind == 'stereo': values = np.column_stack((values, values))
    elif kind == 'clipping': values[:] = 1.1
    elif kind == 'too_short': values = wave(.05)
    elif kind == 'voiced_too_long': values = wave(31)
    reference = raw(tmp_path, values)
    with pytest.raises(preparation.GeneratedReferenceQualityError):
        preparation.prepare_generated_reference(tmp_path, reference, tmp_path / 'review.wav')
    assert not (tmp_path / 'review.wav').exists()
    assert digest(tmp_path / 'raw.wav') == reference['sha256']
    assert (tmp_path / 'raw.wav.raw.json').exists()  # Rejected raw evidence remains.


@pytest.mark.parametrize('part', ['raw', 'raw_receipt', 'review', 'receipt', 'metadata'])
def test_new_reference_provenance_cannot_be_forged_or_silently_republished(tmp_path, part):
    reference = raw(tmp_path, np.concatenate((wave(), np.zeros(5 * 24000, dtype='float32'))))
    metadata = preparation.prepare_generated_reference(tmp_path, reference, tmp_path / 'review.wav')
    target = {'raw': metadata['source'], 'raw_receipt': metadata['raw_receipt'],
              'review': metadata['prepared'], 'receipt': metadata['derivation_receipt']}.get(part)
    if target:
        with (tmp_path / target['path']).open('ab') as stream: stream.write(b'changed')
    else:
        metadata['start_frame'] += 1
    before = state(tmp_path)
    with pytest.raises((ValueError, OSError)):
        preparation.verify_generated_reference(tmp_path, metadata)
    assert state(tmp_path) == before


def test_new_candidates_use_prepared_audio_and_resume_old_records_without_generation(pool):
    root, project, config, originals, saved, model = pool
    plan = candidates.plan_candidate_batch(project, saved, candidates.active_candidate_group(saved))
    candidates.register_candidate_batch(saved, plan)
    output = candidates.candidate_jobs(root) / plan['batch_id'] / 'audio'
    def long_output(**kwargs):
        model.calls.append(kwargs)
        return [np.concatenate((wave(1,16000,200+len(model.calls)*31), np.zeros(32*16000,dtype='float32')))],16000
    model.generate_voice_design=long_output
    rows=candidates.generate_candidate_batch(project,output,'job',plan,[],load_model=model.load)
    assert model.loads==1 and len(model.calls)==5
    for row in rows:
        assert row['reference_preparation']['prepared']['path']==row['path']
        assert row['technical_metrics']['duration_s'] < 2
        assert row['decoded_pcm_sha256']==row['reference_preparation']['technical_metrics']['decoded_pcm_sha256']
        assert row['seed'] in next(item for item in plan['candidates'] if item['candidate_id']==row['candidate_id'])['attempt_seeds']
    # Simulate a fully published legacy candidate before either check was recorded.
    first=rows[0]
    record_path=root/first['candidate_record']['path']
    record=read_json(record_path);record['candidate'].pop('reference_preparation')
    record['candidate'].pop('reference_timing_check')
    write_json(record_path,record)
    before=state(root)
    repeated=candidates.generate_candidate_batch(project,output,'job',plan,[],load_model=lambda *args:pytest.fail('Old records must not reload a model'))
    assert 'reference_preparation' not in repeated[0]
    assert 'reference_timing_check' not in repeated[0]
    assert len(repeated)==5 and len(model.calls)==5
    assert all(digest(root/name)==sha for name,sha in before.items() if not name.endswith('progress.json'))


@pytest.mark.parametrize('invalid_output', ['long_pause', 'overlong_utterance'])
def test_candidate_retries_invalid_timing_with_next_saved_seed(pool, monkeypatch, invalid_output):
    root, project, _, _, saved, model = pool
    plan = candidates.plan_candidate_batch(project, saved, candidates.active_candidate_group(saved))
    candidates.register_candidate_batch(saved, plan)
    output = candidates.candidate_jobs(root) / plan['batch_id'] / 'audio'
    rate = 24000
    rejected_wave = (np.concatenate((wave(2.7), np.full(16 * rate, .00002, dtype='float32'), wave(.9)))
                     if invalid_output == 'long_pause' else wave(17))
    normal_pause = np.concatenate((wave(1), np.zeros(round(.9 * rate), dtype='float32'), wave(1)))
    seeds = []
    monkeypatch.setattr(tts, '_seed', seeds.append)

    def generate(**kwargs):
        model.calls.append(kwargs)
        index = len(model.calls)
        values = rejected_wave if index == 1 else normal_pause if index == 2 else wave(frequency=211 + index)
        return [values], rate

    model.generate_voice_design = generate
    rows = candidates.generate_candidate_batch(project, output, 'job', plan, [], load_model=model.load)
    first, planned = rows[0], plan['candidates'][0]
    assert len(rows) == candidates.BATCH_SIZE and model.loads == 1 and len(model.calls) == 6
    assert seeds[:2] == planned['attempt_seeds'][:2]
    assert first['candidate_id'] == planned['candidate_id'] and first['attempt'] == 1
    assert first['seed'] == planned['attempt_seeds'][1]
    rejected = output / 'raw' / f"{planned['candidate_id']}_attempt0.wav"
    assert np.array_equal(sf.read(rejected, dtype='float32')[0], rejected_wave)
    assert read_json(rejected.with_name(rejected.name + '.raw.json'))['source']['sha256'] == digest(rejected)
    assert np.array_equal(sf.read(root / first['path'], dtype='float32')[0], normal_pause)
    assert first['reference_timing_check']['audible_span_s'] == pytest.approx(2.9)
    assert first['reference_timing_check']['longest_internal_pause_s'] == pytest.approx(.9)
    candidates.ingest_candidate_batch(root, saved, plan, rows)
    assert set(saved['candidates']) == {row['candidate_id'] for row in rows}


def test_candidate_timing_excludes_edges_and_allows_longer_sentences(tmp_path):
    rate = 24000
    source = tmp_path / 'reference.wav'
    sf.write(source, np.concatenate((np.zeros(6 * rate), wave(12), np.zeros(6 * rate))), rate, subtype='FLOAT')
    with pytest.raises(preparation.GeneratedReferenceQualityError, match='너무 깁니다'):
        candidates._check_candidate_timing(source, '오늘은 창밖으로 따뜻한 햇살이 들어옵니다.')
    longer_sentence = '오늘은 창밖으로 따뜻한 햇살이 들어옵니다. 천천히 숨을 고르고 자연스러운 목소리로 한국어 문장을 읽습니다.'
    metrics = candidates._check_candidate_timing(source, longer_sentence)
    assert metrics['audible_span_s'] == pytest.approx(12.)
    assert metrics['span_limit_s'] > metrics['audible_span_s']
    assert metrics['longest_internal_pause_s'] == 0.


def test_legacy_reference_journal_keeps_old_rows_contract_and_uses_cleanup_only_for_new_rows(tmp_path,monkeypatch):
    monkeypatch.setattr(tts,'_seed',lambda seed:None)
    project=create_project(tmp_path/'project','보라야','bora')
    root,config=load_project(project)
    families=[family for family in read_json(root/'voices.json')['families'] if family['split']=='train']
    output=root/'data/references/train';output.mkdir(parents=True)
    original=output/'original.wav';sf.write(original,wave(),24000,subtype='FLOAT')
    row={'sample_id':families[0]['family_id'],'voice_family_id':families[0]['family_id'],'path':original.relative_to(root).as_posix(),'sha256':digest(original),'text':config['tts']['reference_text'],'provenance':'qwen_synthetic_reference','human_reference_used':False,'split':'train'}
    journal=output/'references.jsonl'
    import json
    journal.write_text(json.dumps(row)+'\n',encoding='utf-8');old_prefix=journal.read_bytes()
    contract={'tts':config['tts'],'families':families,'text':config['tts']['reference_text'],'implementation_sha256':tts.LEGACY_CALL_IMPLEMENTATION}
    write_json(output/'contract.json',contract);old_contract=(output/'contract.json').read_bytes()
    class Model:
        calls=0
        def generate_voice_design(self,**kwargs):
            self.calls+=1
            return [np.concatenate((wave(),np.zeros(35*24000,dtype='float32')))],24000
    model=Model()
    result=tts.generate_references(project,'train',limit=2,model=model)
    rows=read_rows(result['review_queue'])
    assert model.calls==1 and rows[0]==row and len(rows)==2
    assert journal.read_bytes().startswith(old_prefix) and (output/'contract.json').read_bytes()==old_contract
    assert rows[1]['reference_preparation']['prepared_duration_s']<3
    before=state(root)
    tts.generate_references(project,'train',limit=2,model=model)
    assert model.calls==1 and state(root)==before
