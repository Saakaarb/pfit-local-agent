"""Standalone sloppiness, adapted from pfit-claude deployed_branch (1b415d8)."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

from lib.utils.run_store import resolve_run, run_sources, run_config


def analyze_run(session, run=None, *, method="auto"):
    from lib.utils.helper_functions import CreatedClass, get_input_reader
    from lib.utils.run_artifacts import parameter_axes, load_restart_seed, scale_parameters
    from lib.utils.sloppiness import run_sloppiness_analysis
    output = resolve_run(session, run, require_seed=True)
    sources = run_sources(output, session)
    if sources == Path(session):
        raise ValueError("This old run has no snapshot; use a gradient-only restart to analyze a recorded model/data pair")
    reader = get_input_reader(run_config(output, session))
    reader.check_name_uniqueness()
    reader.output_dir = output
    script = sources / "generated" / "generated_script.py"
    spec = importlib.util.spec_from_file_location("pfit_analyzed_run", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from lib.utils.experiments import load_experiments
    problem = CreatedClass(experiments=load_experiments(sources, reader), input_reader=reader,
                           compute_loss_problem=module._compute_loss_problem,
                           write_problem_result=module._write_problem_result)
    lo, hi, logs = parameter_axes(reader)
    problem.set_min_limit(lo)
    problem.set_max_limit(hi)
    problem.set_is_logscale(logs)
    # Snapshot provides the historical order for old unnamed design points.
    physical = load_restart_seed(output, reader, allow_legacy=True)
    return run_sloppiness_analysis(module._compute_loss_problem, problem.constants_list,
                                   scale_parameters(physical, reader), reader.trainable_parameter_names,
                                   output, method=method)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--run")
    parser.add_argument("--method", choices=["auto", "ad", "finite-difference"], default="auto")
    args = parser.parse_args(argv)
    session = Path(args.session)
    if not session.is_dir():
        session = Path("sessions") / session
    result = analyze_run(session, args.run, method=args.method)
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
