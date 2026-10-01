"""App jobs publish only complete bound evidence; subprocess lifetime is owned."""
import json
from pathlib import Path
import threading

import pytest

from sentinel_evc import physics_jobs as jobs
from test_ssh_experiment import _fake_result


def _fake_process(monkeypatch, *, return_code=3, ready=None, release=None):
    monkeypatch.setattr(jobs, 'physics_engine_status', lambda: {
        'available': True, 'version': '3.14-test', 'reason': None,
    })
    class Process:
        def __init__(self, command, **kwargs):
            self.returncode = None
            output = Path(command[command.index('--out') + 1])
            if return_code in (0, 3):
                complete = _fake_result(output)
                monkeypatch.setattr(jobs, '_expected_source_manifest_sha256', lambda root: complete['source_manifest_sha256'])
                # start already captured the fixture's expected digest below.
            if ready is not None:
                ready.set()
        def wait(self, timeout=None):
            if release is not None:
                assert release.wait(5)
            self.returncode = return_code
            return return_code
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = -15
            if release is not None:
                release.set()
        def kill(self):
            self.terminate()
    fixture_source = {'src/sentinel_evc/physics.py': '0' * 64}
    import hashlib
    from sentinel_evc.contracts import canonical_json
    digest = hashlib.sha256(canonical_json(fixture_source)).hexdigest()
    monkeypatch.setattr(jobs, '_expected_source_manifest_sha256', lambda root: digest)
    monkeypatch.setattr(jobs.subprocess, 'Popen', Process)


def test_completed_negative_exit_is_verified_and_unsigned_files_are_not_exported(tmp_path, monkeypatch):
    _fake_process(monkeypatch)
    manager = jobs.PhysicsJobs(tmp_path)
    try:
        record = manager.start({'seed': 7})
        manager._workers[record['id']].join(5)
        result = manager.get(record['id'])
        assert result['status'] == 'completed'
        assert result['verification']['ok']
        assert result['verification']['trial_count'] == 25
        directory = manager.directory(record['id']) / 'experiment'
        (directory / 'unsigned-local-note.txt').write_text('must not be exported')
        import zipfile
        with zipfile.ZipFile(manager.export(record['id'])) as archive:
            assert 'unsigned-local-note.txt' not in archive.namelist()
            assert 'COMPLETE.json' in archive.namelist()
        (directory / 'summary.json').write_text('{"forged":true}')
        assert manager.get(record['id'])['verification']['ok'] is False
        with pytest.raises(ValueError):
            manager.export(record['id'])
    finally:
        manager.close()


def test_partial_process_failure_has_no_success_or_export(tmp_path, monkeypatch):
    _fake_process(monkeypatch, return_code=2)
    manager = jobs.PhysicsJobs(tmp_path)
    try:
        record = manager.start({})
        manager._workers[record['id']].join(5)
        result = manager.get(record['id'])
        assert result['status'] == 'failed'
        assert result['summary'] is None
        with pytest.raises(ValueError):
            manager.trace(record['id'])
        with pytest.raises(ValueError):
            manager.export(record['id'])
    finally:
        manager.close()


def test_one_background_job_and_shutdown_owns_child(tmp_path, monkeypatch):
    ready, release = threading.Event(), threading.Event()
    _fake_process(monkeypatch, ready=ready, release=release)
    manager = jobs.PhysicsJobs(tmp_path)
    record = manager.start({})
    assert ready.wait(5)
    with pytest.raises(RuntimeError, match='运行中'):
        manager.start({})
    manager.close()
    assert not manager._workers[record['id']].is_alive()
    with pytest.raises(RuntimeError, match='关闭'):
        manager.start({})


def test_restart_marks_interrupted_and_rejects_path_identifiers(tmp_path):
    job_id = 'physics-job-' + 'a' * 32
    directory = tmp_path / job_id
    directory.mkdir()
    (directory / 'job.json').write_text(json.dumps({'id': job_id, 'status': 'running', 'created_at': '2026-10-01'}))
    manager = jobs.PhysicsJobs(tmp_path)
    try:
        assert manager.get(job_id)['status'] == 'interrupted'
        with pytest.raises(KeyError):
            manager.get('../private')
        with pytest.raises(ValueError):
            manager.start({'seed': True})
    finally:
        manager.close()


@pytest.mark.parametrize('value', [
    {'friction': '0.35'},
    {'friction': [0.35]},
    {'friction': True},
    {'friction': float('inf')},
])
def test_malformed_friction_is_input_error_before_engine_check(tmp_path, monkeypatch, value):
    monkeypatch.setattr(jobs, 'physics_engine_status', lambda: {
        'available': False, 'version': None, 'reason': 'engine unavailable',
    })
    manager = jobs.PhysicsJobs(tmp_path)
    try:
        with pytest.raises(jobs.InputError, match='friction'):
            manager.start(value)
    finally:
        manager.close()


def test_missing_optional_engine_is_reported_and_rejects_start(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs.importlib.util, 'find_spec', lambda name: None)
    manager = jobs.PhysicsJobs(tmp_path)
    try:
        status = manager.engine_status()
        assert status['available'] is False
        assert status['version'] is None
        assert 'install sentinel-evc-lab[physics]' in status['reason']
        with pytest.raises(jobs.InputError, match='not installed'):
            manager.start({})
    finally:
        manager.close()


def test_invalid_completed_evidence_is_persisted_for_list_and_restart(tmp_path, monkeypatch):
    _fake_process(monkeypatch)
    manager = jobs.PhysicsJobs(tmp_path)
    record = manager.start({'seed': 7})
    manager._workers[record['id']].join(5)
    experiment = manager.directory(record['id']) / 'experiment'
    (experiment / 'summary.json').write_text('{"forged":true}')
    invalid = manager.get(record['id'])
    assert invalid['status'] == 'failed' and invalid['verification'] == {'ok': False}
    assert manager.list()[0]['status'] == 'failed'
    manager.close()

    reopened = jobs.PhysicsJobs(tmp_path)
    try:
        persisted = reopened.get(record['id'])
        assert persisted['status'] == 'failed'
        assert persisted['verification'] == {'ok': False}
        assert reopened.list()[0]['status'] == 'failed'
    finally:
        reopened.close()
