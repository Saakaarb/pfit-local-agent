import argparse
from pathlib import Path

from local_agent.agent.checks import check_session, write_check_report
from local_agent.agent.config import WorkflowConfig, load_config
from local_agent.agent.diagnostics import diagnose_run
from local_agent.agent.prompts import PromptRenderer
from local_agent.agent.session_init import init_session
from local_agent.agent.workflow import LocalWorkflow
from local_agent.llm.factory import create_llm_client


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "new":
            return _cmd_new(args)
        if args.command == "check":
            return _cmd_check(args)
        if args.command == "jax":
            return _cmd_jax(args)
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "diagnose":
            return _cmd_diagnose(args)
    except Exception as exc:
        print(f"error: {exc}")
        return 1

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pfit")
    subparsers = parser.add_subparsers(dest="command")

    new_session = subparsers.add_parser("new", help="draft a new fitting session with the local LLM")
    new_session.add_argument("session_dir", type=Path)
    new_session.add_argument("--overwrite", action="store_true")
    new_session.add_argument("--model")
    new_session.add_argument("--base-url")
    new_session.add_argument("--timeout-seconds", type=float)
    new_session.add_argument("--max-repair-attempts", type=int)
    new_session.add_argument("--temperature", type=float)
    new_session.add_argument("--max-tokens", type=int)
    new_session.add_argument("--fake-response-file", type=Path)
    new_session.add_argument("--debug", action="store_true")

    check = subparsers.add_parser(
        "check",
        help="check a session and write generated/user_input_check.txt",
    )
    check.add_argument("session_dir", type=Path)
    check.add_argument("--deterministic-only", action="store_true", help="validate without contacting Ollama")
    check.add_argument("--ready", action="store_true", help="also verify generated code is ready to run (no LLM)")
    check.add_argument("--model")
    check.add_argument("--base-url")
    check.add_argument("--timeout-seconds", type=float)
    check.add_argument("--temperature", type=float)
    check.add_argument("--max-tokens", type=int)
    check.add_argument("--fake-response-file", type=Path)
    check.add_argument("--debug", action="store_true")

    jax = subparsers.add_parser("jax", help="translate user model to generated_script.py")
    jax.add_argument("session_dir", type=Path)
    jax.add_argument("--model")
    jax.add_argument("--base-url")
    jax.add_argument("--timeout-seconds", type=float)
    jax.add_argument("--max-repair-attempts", type=int)
    jax.add_argument("--temperature", type=float)
    jax.add_argument("--max-tokens", type=int)
    jax.add_argument("--fake-response-file", type=Path)
    jax.add_argument("--debug", action="store_true")

    run = subparsers.add_parser("run", help="run fitting using existing generated code")
    run.add_argument("session_dir", type=Path)
    run.add_argument("mode", nargs="?", choices=["full", "gradient-only"], default="full")
    run.add_argument("--from-run", "--seed-run", dest="from_run", help="source run ID for gradient-only refinement")
    run.add_argument("--allow-legacy-seed", action="store_true", help="assert old unnamed CSV uses current YAML parameter order")
    run.add_argument("--no-sloppiness", action="store_true", help="skip post-fit curvature analysis")
    run.add_argument("--sloppiness-method", choices=["auto", "ad", "finite-difference"], default="auto")

    diagnose = subparsers.add_parser("diagnose", help="diagnose a completed fitting run")
    diagnose.add_argument("session_dir", type=Path)
    diagnose.add_argument("run_id", nargs="?")

    return parser


def _cmd_new(args: argparse.Namespace) -> int:
    config = load_config(Path.cwd(), args.session_dir)
    model = args.model or config.llm.model
    base_url = args.base_url or config.llm.base_url
    timeout_seconds = (
        args.timeout_seconds
        if args.timeout_seconds is not None
        else config.llm.timeout_seconds
    )
    workflow_config = WorkflowConfig(
        max_repair_attempts=(
            args.max_repair_attempts
            if args.max_repair_attempts is not None
            else config.workflow.max_repair_attempts
        ),
        temperature=(
            args.temperature
            if args.temperature is not None
            else config.workflow.temperature
        ),
        max_tokens=(
            args.max_tokens if args.max_tokens is not None else config.workflow.max_tokens
        ),
    )
    llm_client = create_llm_client(
        model,
        base_url,
        fake_response_file=args.fake_response_file,
        timeout_seconds=timeout_seconds,
        debug_stream=args.debug,
    )
    written = init_session(
        args.session_dir,
        llm_client,
        PromptRenderer(),
        workflow_config,
        overwrite=args.overwrite,
    )
    print(f"session initialized: {args.session_dir}")
    for path in written:
        print(f"written: {path}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    if args.deterministic_only or args.ready:
        from local_agent.agent.readiness import check_ready
        report = check_ready(args.session_dir) if args.ready else check_session(args.session_dir)
        report_path = write_check_report(args.session_dir, report)
        print(f"check report written: {report_path}")
        print(report.to_text())
        return 0 if report.passed else 1
    config = load_config(Path.cwd(), args.session_dir)
    model = args.model or config.llm.model
    base_url = args.base_url or config.llm.base_url
    timeout_seconds = (
        args.timeout_seconds
        if args.timeout_seconds is not None
        else config.llm.timeout_seconds
    )
    workflow_config = WorkflowConfig(
        max_repair_attempts=config.workflow.max_repair_attempts,
        temperature=(
            args.temperature
            if args.temperature is not None
            else config.workflow.temperature
        ),
        max_tokens=(
            args.max_tokens if args.max_tokens is not None else config.workflow.max_tokens
        ),
    )
    llm_client = create_llm_client(
        model,
        base_url,
        fake_response_file=args.fake_response_file,
        timeout_seconds=timeout_seconds,
        debug_stream=args.debug,
    )
    report = check_session(
        args.session_dir,
        llm_client,
        PromptRenderer(),
        workflow_config,
    )
    report_path = write_check_report(args.session_dir, report)
    print(f"check report written: {report_path}")
    if report.passed:
        print(f"validation passed: {args.session_dir}")
        for warning in report.warnings:
            print(f"warning: {warning}")
        for recommendation in report.recommendations:
            print(f"recommendation: {recommendation}")
        return 0
    for error in report.critical_errors:
        print(f"validation failed: {error}")
    return 1


def _cmd_jax(args: argparse.Namespace) -> int:
    config = load_config(Path.cwd(), args.session_dir)
    model = args.model or config.llm.model
    base_url = args.base_url or config.llm.base_url
    timeout_seconds = (
        args.timeout_seconds
        if args.timeout_seconds is not None
        else config.llm.timeout_seconds
    )
    workflow_config = WorkflowConfig(
        max_repair_attempts=(
            args.max_repair_attempts
            if args.max_repair_attempts is not None
            else config.workflow.max_repair_attempts
        ),
        temperature=(
            args.temperature
            if args.temperature is not None
            else config.workflow.temperature
        ),
        max_tokens=(
            args.max_tokens if args.max_tokens is not None else config.workflow.max_tokens
        ),
    )
    llm_client = create_llm_client(
        model,
        base_url,
        fake_response_file=args.fake_response_file,
        timeout_seconds=timeout_seconds,
        debug_stream=args.debug,
    )
    workflow = LocalWorkflow(
        llm_client,
        PromptRenderer(),
        workflow_config,
    )
    result = workflow.generate_script(args.session_dir)
    for event in result.events:
        print(f"{event.step}: {event.status}: {event.message}")
    return 0 if result.success else 1


def _cmd_run(args: argparse.Namespace) -> int:
    from local_agent.core.fitting import make_run_output_dir
    from fit_parameters import run_driver
    from local_agent.agent.validators import parse_input_yaml

    if args.mode != "gradient-only" and args.from_run:
        raise ValueError("--from-run requires gradient-only mode")
    if args.allow_legacy_seed and args.mode != "gradient-only":
        raise ValueError("--allow-legacy-seed requires gradient-only mode")
    input_file_path = args.session_dir / "inputs" / "user_input.yaml"
    input_reader = parse_input_yaml(input_file_path)
    output_dir = make_run_output_dir(args.session_dir)
    print(f"run directory: {output_dir}")
    run_driver(
        args.session_dir, input_reader, output_dir_override=output_dir,
        from_run=args.from_run, allow_legacy_seed=args.allow_legacy_seed,
        gradient_only=args.mode == "gradient-only",
        sloppiness=not args.no_sloppiness, sloppiness_method=args.sloppiness_method,
    )
    return 0


def _cmd_diagnose(args: argparse.Namespace) -> int:
    report_path = diagnose_run(args.session_dir, args.run_id)
    print(f"diagnosis written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
