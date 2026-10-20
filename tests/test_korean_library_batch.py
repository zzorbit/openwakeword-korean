"""New four-pool labels/exposure and historical configuration compatibility."""
from copy import deepcopy
import json

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, read_json, read_rows, write_json
from korean_openwakeword.project import (
    BATCH, LEGACY_BATCH, create_project, default_profile, load_project,
    training_batch, validate_project, workload_summary,
)
from korean_openwakeword.train import InputDelivery, PoolSampler, validate_training_scale
from korean_openwakeword.data import audit_project, prepare_windows, validate_admission, validate_hard_negative
from korean_openwakeword.features import build_features, verify_cache
from korean_openwakeword.review import apply_review
from test_korean_library import project


def test_default_targets_batch_and_actual_labels():
    config = default_profile()
    assert config['targets']['generic_hours'] == 7000
    assert config['targets']['korean_hours'] == 3000
    assert config['training']['updates'] == config['training']['schedule_horizon'] == 100000
    assert training_batch(config) == BATCH
    sizes = {r: 5 for r in BATCH}
    sampler = PoolSampler(sizes, 19, training_batch(config))
    pools = {role: np.full((5, 16, 96), i, np.float32) for i, role in enumerate(BATCH)}
    x, y = InputDelivery(pools, 'cpu', batch=training_batch(config)).batch(sampler.next_indices())
    assert tuple(x.shape) == (1224, 16, 96)
    assert y.sum().item() == 100
    assert (y[:1024] == 0).all() and (y[1024:1124] == 1).all() and (y[1124:] == 0).all()
    assert (x[1124:] == 3).all(), 'Hard Negative must come from its own pool'
    assert sampler.draws == BATCH
    summary = workload_summary(config)
    assert summary['batch_size'] == 1224
    assert summary['draws_per_pool_per_model']['positive'] == 10000000
    assert summary['draws_per_pool_per_model']['hard_negative'] == 10000000


def test_four_pool_sampler_restores_exactly_and_requires_hard_pool():
    sampler = PoolSampler({r: 9 for r in BATCH}, 22, BATCH)
    sampler.next_indices(); state = sampler.state(); expected = sampler.next_indices()
    sampler.restore(state); actual = sampler.next_indices()
    assert all(np.array_equal(expected[r], actual[r]) for r in BATCH)
    with pytest.raises(ValueError, match='nonempty'):
        PoolSampler({r: 9 for r in LEGACY_BATCH}, 22, BATCH)
    with pytest.raises(ValueError, match='nonempty'):
        PoolSampler({**{r: 9 for r in BATCH}, 'hard_negative': 0}, 22, BATCH)


def test_historical_config_retains_original_batch_and_exposure():
    config = default_profile()
    config['batch'] = deepcopy(LEGACY_BATCH)
    config['hard_negative_samples'] = 0
    config['training'].update(updates=50000, schedule_horizon=50000, checkpoints=[10000, 20000, 50000])
    config['manifests'].pop('hard_negative'); config['pool_features'].pop('hard_negative')
    before = deepcopy(config)
    validate_project(config)
    assert config == before
    assert workload_summary(config)['positive_draws_per_model'] == 2500000
    assert workload_summary(config)['batch_size'] == 1074
    sampler = PoolSampler({r: 7 for r in LEGACY_BATCH}, 42, training_batch(config))
    assert {r: len(v) for r, v in sampler.next_indices().items()} == LEGACY_BATCH


def test_historical_three_pool_project_can_still_train(project):
    from korean_openwakeword.train import train
    root, config = load_project(project)
    config['batch'] = deepcopy(LEGACY_BATCH)
    config['hard_negative_samples'] = 0
    config['manifests'].pop('hard_negative'); config['pool_features'].pop('hard_negative')
    config['training']['seeds'] = [42]
    write_json(project, config)
    receipt = train(project, root / 'runs/legacy_smoke', smoke=True)['members'][0]
    assert set(receipt['exposure']) == set(LEGACY_BATCH)
    assert receipt['exposure']['positive']['drawn'] == 200
    assert receipt['selected_workload']['batch_size'] == 1074


@pytest.mark.parametrize('change', [
    {'text': '보라야'}, {'text': '보 라 야 불 켜줘'}, {'text': ''},
    {'text': '보라'}, {'text': '보라가'}, {'text': '보라를'}, {'text': '보라는'},
    {'human_reference_used': True}, {'speech': False}, {'voice_family_id': ''},
])
def test_hard_negative_rejects_call_or_missing_synthetic_provenance(change):
    config = default_profile(); config['wakeword'] = '보라야'
    row = dict(text='보람아', speech=True, provenance='qwen_tts', human_reference_used=False, voice_family_id='train_voice_000')
    with pytest.raises(ValueError):
        validate_hard_negative({**row, **change}, config)


def test_direct_call_name_is_rejected_by_audit_windows_and_feature_resume(project):
    root, config = load_project(project)
    manifest = root / config['manifests']['hard_negative']
    rows = read_rows(manifest)
    rows[0]['text'] = '보라를'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    audit = audit_project(project, check_files=False)
    assert not audit['passed']
    assert any('complete call-name' in message for message in audit['errors'])
    with pytest.raises(ValueError, match='complete call-name'):
        validate_admission(root, rows, config, 'hard_negative', smoke=True)
    with pytest.raises(ValueError, match='complete call-name'):
        prepare_windows(project, 'hard_negative', root / 'data/negative/forbidden_windows.jsonl')
    # Even an already created feature cache cannot bypass the text guard.
    with pytest.raises(ValueError, match='complete call-name'):
        build_features(project, 'hard_negative', smoke=True, resume=True, frontend=StubFrontend())


class StubFrontend:
    identity = {'fixture': 'functional-only, not an audio-quality claim'}
    def embed(self, clips):
        return np.repeat(clips.astype(np.float32).reshape(len(clips), 16, 2000).mean(axis=2)[:, :, None], 96, axis=2)


@pytest.mark.parametrize('synthetic', [False, True])
def test_reviewed_hard_negative_windows_features_and_admission(tmp_path, synthetic):
    project = create_project(tmp_path / 'word', '보라야', 'bora')
    root, config = load_project(project)
    config['augmentation'].update(clean_probability=1, moderate_probability=0, rir_probability=0)
    write_json(project, config)
    path = root / 'data/negative/hard.wav'
    sf.write(path, .1 * np.sin(2 * np.pi * 230 * np.arange(48000) / 16000), 16000, subtype='PCM_16')
    row = dict(sample_id='hard1', source_id='fixture:hard1', split='train', path=str(path.relative_to(root)),
               sha256=digest(path), text='보람아', speech=True, category='competing_speech')
    if synthetic:
        row.update(provenance='qwen_tts', human_reference_used=False, voice_family_id='train_voice_000')
    else:
        row.update(provenance='public_dataset', source_url='https://example.org/test-fixture', license='test-only')
    queue = root / 'review/hard_queue.jsonl'
    queue.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
    record = apply_review(project, queue, {'hard1': {'decision': 'keep', 'listened': True}}, 'content')
    row['admission'] = record['admission']
    manifest = root / config['manifests']['hard_negative']
    manifest.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
    validate_admission(root, [row], config, 'hard_negative')
    assert audit_project(project)['passed']
    output = root / 'data/negative/hard_windows.jsonl'
    assert prepare_windows(project, 'hard_negative', output)['rows'] == 2
    config['manifests']['hard_negative'] = str(output.relative_to(root)); write_json(project, config)
    meta = build_features(project, 'hard_negative', frontend=StubFrontend())
    assert meta['role'] == 'hard_negative' and meta['feature_rows'] == 2
    assert meta['provenance'] == ('qwen_tts' if synthetic else 'public_dataset')
    assert meta['quality_review'] == 'accepted' and meta['rounds'] == 1
    assert verify_cache(root / config['pool_features']['hard_negative'])['feature_rows'] == 2
    build_features(project, 'hard_negative', frontend=StubFrontend(), resume=True)
    bad = deepcopy(read_rows(output)[0]); bad['split'] = 'final'
    with pytest.raises(ValueError, match='train split'):
        validate_admission(root, [bad], config, 'hard_negative')


def test_expanded_target_refuses_old_nominal_generic_scale():
    config = default_profile()
    meta = {'positive': {'original_count': 100000, 'rounds': 5, 'feature_rows': 500000},
            'generic_negative': {'legacy_generic_bridge': True, 'publisher_declared_hours': 2000},
            'korean_negative': {'unique_source_hours': 3000}}
    with pytest.raises(ValueError, match='7000h'):
        validate_training_scale(config, meta)


def test_cli_exposes_hard_negative_windows_and_features():
    from korean_openwakeword.cli import parser
    args = parser().parse_args(['features', '--project', 'project.json', '--role', 'hard_negative'])
    assert args.role == 'hard_negative'
    args = parser().parse_args(['windows', '--project', 'project.json', '--pool', 'hard_negative', '--output', 'windows.jsonl'])
    assert args.pool == 'hard_negative'
