"""Check that benchmark failures cannot be reported as successful timings."""
import json
from pathlib import Path
import subprocess

from scripts import run_model_comparison as comparison


def test_stage_timeout_kills_process_group_and_records_failure(tmp_path, monkeypatch):
    class Process:
        pid = 123
        attempts = 0

        def wait(self, timeout=None):
            self.attempts += 1
            if self.attempts == 1:
                raise subprocess.TimeoutExpired('worker', timeout)
            return -15

    killed = []
    monkeypatch.setattr(comparison.subprocess, 'Popen', lambda *a, **kw: Process())
    monkeypatch.setattr(comparison.os, 'killpg', lambda *args: killed.append(args))
    (tmp_path / 'logs').mkdir()
    result = comparison.stage(tmp_path, 'new', ['new', str(tmp_path)], 1, {})
    assert result['status'] == 'fail'
    assert result['exit_code'] == 124
    assert result['timed_out'] is True
    assert killed[0][0] == 123


def test_report_retains_failed_attempt_timing_and_pending_models(tmp_path):
    models = json.loads((comparison.REPO / 'benchmarks/model_comparison/models.json').read_text())
    path = tmp_path / 'cases/example/qwen32b/metadata.json'
    comparison.save(path, dict(status='fail', failed_stage='jax', total_wall_seconds=12.5,
                              stages={'jax': {'seconds': 4.5}}))
    comparison.report(tmp_path, models, [dict(name='example')])
    text = (tmp_path / 'comparison.md').read_text()
    assert 'fail at jax; 12.5' in text
    assert '| pending | pending |' in text
    with (tmp_path / 'comparison.csv').open() as handle:
        rows = list(comparison.csv.DictReader(handle))
    assert rows[0]['jax_seconds'] == '4.5'
    assert rows[0]['status'] == 'fail'
    assert rows[1]['total_wall_seconds'] == ''
