from pathlib import Path
import json


def _read_optimizer_log(path: Path) -> dict[str, float | int] | None:
    rows = []
    if not path.exists():
        return None

    for line in path.read_text().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            rows.append((int(parts[0]), float(parts[1])))
        except ValueError:
            continue

    if not rows:
        return None
    losses = [loss for _, loss in rows]
    return {
        "iterations": len(rows),
        "first_loss": losses[0],
        "best_loss": min(losses),
        "final_loss": losses[-1],
    }


def _read_workflow_events(path: Path) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "attempted": 0}
    if not path.exists():
        return counts

    for line in path.read_text().splitlines():
        try:
            status = json.loads(line).get("status")
        except json.JSONDecodeError:
            continue
        if status in counts:
            counts[status] += 1
    return counts


def diagnose_run(session_dir: Path, run_id: str | None = None) -> Path:
    session_dir = Path(session_dir)
    output_dir = session_dir / "outputs"
    if run_id:
        output_dir = output_dir / run_id

    report_path = output_dir / "fit_diagnosis.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    final_design = output_dir / "final_design_point.csv"
    result_solution = output_dir / "result_solution.csv"
    pso_log = output_dir / "pso_fitting.log"
    de_log = output_dir / "de_fitting.log"
    node_log = output_dir / "NODE_fitting.log"
    error_file = output_dir / "fitting_error.txt"
    event_log = session_dir / "generated" / "agent_logs" / "workflow_events.jsonl"
    global_log = de_log if de_log.exists() else pso_log
    global_name = "DE" if de_log.exists() else "PSO"
    global_summary = _read_optimizer_log(global_log)
    node_summary = _read_optimizer_log(node_log)
    event_counts = _read_workflow_events(event_log)

    lines = [
        "Fit diagnosis",
        "",
        f"Session: {session_dir}",
        f"Run directory: {output_dir}",
        "",
        "Artifacts:",
        f"- final_design_point.csv: {'present' if final_design.exists() else 'missing'}",
        f"- result_solution.csv: {'present' if result_solution.exists() else 'missing'}",
        f"- pso_fitting.log: {'present' if pso_log.exists() else 'missing'}",
        f"- de_fitting.log: {'present' if de_log.exists() else 'missing'}",
        f"- NODE_fitting.log: {'present' if node_log.exists() else 'missing'}",
        f"- fitting_error.txt: {'present' if error_file.exists() else 'missing'}",
        "",
        "Optimizer summary:",
        _format_summary(global_name, global_summary),
        _format_summary("NODE", node_summary),
        "",
        "Generation workflow:",
        (
            f"- events: {event_counts['passed']} passed, "
            f"{event_counts['failed']} failed, {event_counts['attempted']} repair attempts"
            if event_log.exists()
            else "- workflow_events.jsonl: missing"
        ),
        "",
        "Recommendations:",
    ]

    if error_file.exists():
        lines.append("- Inspect fitting_error.txt before changing model assumptions.")
    if not final_design.exists():
        lines.append("- No final design point found; rerun fitting or inspect optimizer logs.")
    if final_design.exists() and result_solution.exists():
        lines.append("- Review fitted parameters and measured-versus-fitted trajectories.")
    if not pso_log.exists() and not de_log.exists():
        lines.append("- Global-search log missing; verify the PSO/DE stage ran.")
    if not node_log.exists():
        lines.append("- NODE log missing; verify the gradient-refinement stage ran.")
    if global_summary and global_summary["final_loss"] > global_summary["first_loss"]:
        lines.append("- Global-search loss worsened; review bounds and data/model consistency.")
    if node_summary and node_summary["final_loss"] > node_summary["first_loss"]:
        lines.append("- NODE loss worsened; inspect gradients, tolerances, and generated code.")

    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def _format_summary(name: str, summary: dict[str, float | int] | None) -> str:
    if summary is None:
        return f"- {name}: no parseable optimizer rows"
    return (
        f"- {name}: {summary['iterations']} iterations, "
        f"first loss {summary['first_loss']:.4E}, "
        f"best loss {summary['best_loss']:.4E}, "
        f"final loss {summary['final_loss']:.4E}"
    )
