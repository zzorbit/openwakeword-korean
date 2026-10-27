"""GPU-required release rebind preserves the previous CPU evaluation."""
from copy import deepcopy
from pathlib import Path

import pytest

from korean_openwakeword import studio
from korean_openwakeword.common import digest, identity, read_json, write_json
from test_korean_library_studio import manager, _paused_after_first_stage


def gpu_required_pair(manager, monkeypatch):
    view = _paused_after_first_stage(manager)
    state = manager._load(view['id'])
    current = deepcopy(manager._execution_contract)
    previous = deepcopy(state['execution_contract'])
    changed = ['evaluation_backend.py', 'gpu_runtime_install.py', 'studio_steps.py']
    for name in changed:
        previous['files'][name] = '0' * 64
    previous['computation_sha256'] = identity(previous['files'])
    previous['stages']['evaluation'] = '1' * 64
    state.update(index=next(i for i, step in enumerate(state['plan']) if step['id'] == 'evaluation'),
                 status='interrupted', auto_continue=False, job=None, execution_contract=previous)
    write_json(manager._path(view['id']), state)
    manager._implementation_id = 'audited-gpu-required-release'
    manager._execution_contract = current
    audit = {'source_stages_sha256': identity(previous['stages']),
             'target_computation_sha256': current['computation_sha256'],
             'target_stages_sha256': identity(current['stages']), 'changed_files': changed,
             'evidence_sha256': identity({'gpu_required': 'fixture'})}
    monkeypatch.setattr(studio, '_AUDITED_ACQUISITION_UPDATES', {previous['computation_sha256']: audit})
    return state, previous, current, audit


def test_gpu_required_migration_preserves_cpu_scores_and_workflow(manager, monkeypatch):
    state, previous, current, _ = gpu_required_pair(manager, monkeypatch)
    root = Path(state['project']).parent
    cpu = root / 'reports/studio_evaluation/_evaluation_work/frames.npy'
    cpu.parent.mkdir(parents=True)
    cpu.write_bytes(b'unchanged CPU score prefix')
    checkpoint = cpu.with_name('checkpoint.json')
    write_json(checkpoint, {'committed_sessions': 17, 'binding': 'preserved CPU binding'})
    before = {p: p.read_bytes() for p in (cpu, checkpoint, Path(state['project']))}
    workflow_before = manager._path(state['id']).read_bytes()
    migration = manager._migration_view(state)
    assert migration['state'] == 'available'
    assert 'CPU로 전환하지 않고' in migration['reason']
    manager.migrate(state['id'])
    after = manager._load(state['id'])
    for key in ('index', 'status', 'settings', 'plan', 'history', 'inputs', 'results', 'job', 'auto_continue'):
        assert after.get(key) == state.get(key)
    assert all(p.read_bytes() == value for p, value in before.items())
    receipt = read_json(after['migrations'][-1]['path'])
    assert Path(receipt['previous_workflow']['path']).read_bytes() == workflow_before
    assert digest(after['migrations'][-1]['path']) == after['migrations'][-1]['sha256']
    assert receipt['audited_acquisition_update']['source_computation_sha256'] == previous['computation_sha256']
    assert receipt['new_computation_sha256'] == current['computation_sha256']
    assert len(manager.workers.items) == 1


@pytest.mark.parametrize('extra', ['features.py', 'gpu_evaluation.py', 'train.py', 'resources/public_datasets.json'])
def test_gpu_required_migration_rejects_unrelated_computation(manager, monkeypatch, extra):
    state, _, current, audit = gpu_required_pair(manager, monkeypatch)
    current['files'][extra] = '2' * 64
    current['computation_sha256'] = identity(current['files'])
    audit['target_computation_sha256'] = current['computation_sha256']
    audit['changed_files'].append(extra)
    original = manager._path(state['id']).read_bytes()
    assert manager._migration_view(state)['state'] == 'blocked'
    with pytest.raises(ValueError):
        manager.migrate(state['id'])
    assert manager._path(state['id']).read_bytes() == original
