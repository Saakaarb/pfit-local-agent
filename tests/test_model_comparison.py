"""Check that benchmark failures cannot be reported as successful timings."""
import json
from pathlib import Path
import subprocess

import pytest

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


def test_run_case_excludes_benchmark_metadata_from_llm_context(tmp_path, monkeypatch):
    from local_agent.agent.session_init import _collect_session_context

    frozen = tmp_path / 'frozen_inputs/example'
    frozen.mkdir(parents=True)
    (frozen / 'user_info.txt').write_text('Fit dA/dt = -k A with A(0)=1.')
    (frozen / 'data.csv').write_text('time,A\n0,1\n1,0.5\n')
    model = json.loads((comparison.REPO / 'benchmarks/model_comparison/models.json').read_text())[0]
    case = dict(name='example', status='ready', expected_data_files=['data.csv'], expected_experiments=1)

    def fake_stage(directory, label, command, timeout, env):
        comparison.save(directory / 'active_stage.json', {'stage': label})
        (directory / 'logs/new.log').write_text('BENCHMARK_LOG_SENTINEL')
        context = _collect_session_context(Path(command[1]))
        assert 'Fit dA/dt' in context
        assert 'FILE: inputs/data.csv' in context
        assert 'metadata.json' not in context
        assert 'active_stage.json' not in context
        assert 'BENCHMARK_LOG_SENTINEL' not in context
        assert 'BENCHMARK_METADATA_SENTINEL' not in context
        return dict(status='fail', exit_code=1, seconds=1)

    monkeypatch.setattr(comparison, 'stage', fake_stage)
    comparison.run_case(tmp_path, case, model, {'marker': 'BENCHMARK_METADATA_SENTINEL'}, {})
    record = json.loads((tmp_path / 'cases/example/qwen32b/metadata.json').read_text())
    assert record['status'] == 'fail'
    assert record['failed_stage'] == 'new'
    assert 'error' not in record  # Do not swallow a context assertion in run_case.


def test_report_excludes_unselected_models(tmp_path):
    models = json.loads((comparison.REPO / 'benchmarks/model_comparison/models.json').read_text())[:2]
    comparison.report(tmp_path, models, [dict(name='example')])
    text = (tmp_path / 'comparison.md').read_text()
    assert 'qwen32b' in text and 'qwen14b' in text
    assert 'qwen3_coder_next' not in text


def test_external_smoke_applies_start_time_before_check_and_skips_curvature(tmp_path, monkeypatch):
    import yaml
    frozen = tmp_path / 'frozen_inputs/example'
    frozen.mkdir(parents=True)
    (frozen / 'user_info.txt').write_text('Initial conditions apply at time zero.')
    (frozen / 'data.csv').write_text('time,x\n0.25,1\n1,0.5\n')
    model = json.loads((comparison.REPO / 'benchmarks/model_comparison/models.json').read_text())[0]
    case = dict(name='example', status='ready', expected_data_files=['data.csv'],
                expected_experiments=1, sloppiness=False,
                pre_run_configuration={'gradient_opt.initial_time': 0})
    stages = []
    def fake_stage(directory, label, command, timeout, env):
        stages.append(label)
        config = Path(command[1]) / 'inputs/user_input.yaml'
        if label == 'new':
            config.write_text(yaml.safe_dump({'experiments': [{'data_file': 'data.csv'}]}))
        else:
            settings = yaml.safe_load(config.read_text())
            assert settings['gradient_opt']['initial_time'] == 0
            assert settings['gradient_opt']['num_iters'] == 5
            assert settings['population_opt']['population_size'] == 4
        if label == 'run':
            assert '--no-sloppiness' in command
            return dict(status='fail', exit_code=1, seconds=1)
        return dict(status='pass', exit_code=0, seconds=1)
    monkeypatch.setattr(comparison, 'stage', fake_stage)
    comparison.run_case(tmp_path, case, model, {}, {})
    assert stages == ['new', 'check', 'jax', 'run']
    result = json.loads((tmp_path / 'cases/example/qwen32b/metadata.json').read_text())
    assert result['settings']['sloppiness'] is False
    assert result['pre_run_configuration'] == case['pre_run_configuration']


@pytest.mark.parametrize('model,thinking', [('qwen3.8:27b', True), ('qwen2.5-coder:32b', None)])
def test_benchmark_thinking_mode_preserves_content_and_budget(tmp_path, monkeypatch, model, thinking):
    from local_agent.llm.ollama import OllamaClient
    import local_agent.cli.main as cli

    def fake_post(self, path, payload):
        assert payload.get('think') is thinking
        assert payload['options']['num_predict'] == 12000
        assert payload['options']['num_ctx'] == 32768
        return {'message': {'content': '{"ok":true}', 'thinking': 'reasoning' if thinking else ''},
                'done_reason': 'stop', 'eval_count': 12}

    def fake_main(arguments):
        assert OllamaClient(model).complete([]) == '{"ok":true}'
        return 0

    monkeypatch.setattr(OllamaClient, '_post_json', fake_post)
    monkeypatch.setattr(cli, 'main', fake_main)
    with pytest.raises(SystemExit) as outcome:
        comparison.instrumented_worker(tmp_path, 'new', [])
    assert outcome.value.code == 0
    metric = json.loads((tmp_path / 'llm_metrics.jsonl').read_text())
    assert metric['think'] == (True if thinking else 'model default')
    assert metric['thinking_characters'] == (9 if thinking else 0)
