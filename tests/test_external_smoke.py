"""Protect input parity and failed-attempt provenance in the external comparison."""
import json

import pytest

from scripts import run_external_smoke as external


def test_reuse_preserves_failed_attempt_and_rejects_changed_input(tmp_path):
    source, root = tmp_path / 'source', tmp_path / 'target'
    inputs = source / 'frozen_inputs/example'
    inputs.mkdir(parents=True)
    (inputs / 'user_info.txt').write_text('A scientific prompt.')
    model = {'key': 'qwen32b', 'model': 'qwen2.5-coder:32b'}
    case = dict(name='example', input_hashes=external.hashes(inputs), pre_run_configuration={})
    settings = dict(options=external.runner.OPTIONS, max_repair_attempts=5, request_timeout_seconds=600,
        gradient_iterations=5, population_size=4, population_iterations=1, gradient_optimizer='adam',
        numerical_backend='cpu', jax_x64=True, sloppiness=False)
    entry = dict(status='fail', failed_stage='new', settings=settings,
        input_audit={}, model_metadata=dict(model=model['model'], framework_commit='original'))
    directory = source / 'cases/example/qwen32b'
    external.runner.save(directory / 'metadata.json', entry)
    (directory / 'failure.log').write_text('Original failure, do not discard.')
    # Use a repository-relative source location in the import record.
    original_repo = external.runner.REPO
    external.runner.REPO = tmp_path
    try:
        assert external.reuse_case(source, root, case, model)
        copied = root / 'cases/example/qwen32b'
        assert json.loads((copied / 'metadata.json').read_text()) == entry
        assert (copied / 'failure.log').read_text() == 'Original failure, do not discard.'
        assert json.loads((copied / 'import_provenance.json').read_text())['original_framework_commit'] == 'original'
        (inputs / 'user_info.txt').write_text('Different science.')
        with pytest.raises(ValueError, match='inputs differ'):
            external.reuse_case(source, tmp_path / 'other', case, model)
    finally:
        external.runner.REPO = original_repo


def test_external_report_keeps_pending_models_and_has_no_old_cohort_claims(tmp_path):
    models = json.loads((external.runner.REPO / 'benchmarks/model_comparison/models.json').read_text())[:2]
    external.runner.save(tmp_path / 'cases/example/qwen32b/metadata.json',
        dict(status='fail', failed_stage='new', total_wall_seconds=9.5))
    rows = external.report(tmp_path, [dict(name='example')], models)
    assert [r['status'] for r in rows] == ['fail', 'pending']
    assert rows[0]['total_wall_seconds'] == 9.5
    text = (tmp_path / 'comparison.md').read_text()
    assert 'fail at new (9.5 s)' in text
    assert 'Robertson' not in text and 'qwen3_coder_next' not in text
