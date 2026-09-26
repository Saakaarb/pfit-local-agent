"""Gradient-only entry point adapted from pfit-claude deployed_branch (1b415d8)."""
from pathlib import Path
import numpy as np
from lib.utils.run_store import resolve_run

def load_init_guess(session_dir: Path, seed_run=None) -> np.ndarray:
    """Load the starting design point for the gradient-only run.

    Reads the selected completed run's ``final_design_point.csv`` — the best
    point from a previous run (population search + any prior gradient refinement).
    This is what lets a user re-run *just* the gradient stage (e.g. with more
    iterations or a different optimizer) starting from where they left off.

    Args:
        session_dir (Path): Path to the session directory.

    Returns:
        numpy.ndarray: Initial guess in real parameter units, in YAML trainable order.
    """
    guess_path = resolve_run(session_dir, seed_run, require_seed=True) / "final_design_point.csv"
    if not guess_path.exists():
        raise FileNotFoundError(
            f"No initial guess found at {guess_path}\n"
            "Run the full fit (python fit_parameters.py <session>) at least once first, "
            "or place a final_design_point.csv in the session's outputs/ directory."
        )
    guess = np.atleast_1d(np.genfromtxt(guess_path, delimiter=","))
    print(f"Loaded initial guess from {guess_path}: {guess}")
    return guess


def run_driver(session_dir, input_reader, seed_run=None, **kwargs):
    from local_agent.core.fitting import run_driver as local_run_driver
    return local_run_driver(session_dir, input_reader, gradient_only=True, from_run=seed_run, **kwargs)


def main(argv=None):
    import argparse
    from local_agent.cli.main import main as cli_main
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--seed-run")
    parser.add_argument("--allow-legacy-seed", action="store_true")
    parser.add_argument("--no-sloppiness", action="store_true")
    parser.add_argument("--sloppiness-method", choices=["auto", "ad", "finite-difference"], default="auto")
    args = parser.parse_args(argv)
    session = Path(args.session)
    if not session.is_dir():
        session = Path("sessions") / session
    command = ["run", str(session), "gradient-only", "--sloppiness-method", args.sloppiness_method]
    if args.seed_run:
        command += ["--from-run", args.seed_run]
    if args.allow_legacy_seed:
        command += ["--allow-legacy-seed"]
    if args.no_sloppiness:
        command += ["--no-sloppiness"]
    return cli_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
