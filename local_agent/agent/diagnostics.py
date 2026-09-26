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
    from lib.utils.run_store import output_root
    output_dir = output_root(session_dir)
    if run_id:
        output_dir = output_dir / run_id

    if run_id is None and not (output_dir / "final_design_point.csv").exists() and output_dir.exists():
        runs = sorted(path for path in output_dir.iterdir() if path.is_dir() and (path / "run_manifest.json").exists())
        if runs:
            output_dir = runs[-1]
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    gradient_only = manifest.get("mode") == "gradient-only"
    report_path = output_dir / "fit_diagnosis.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    final_design = output_dir / "final_design_point.csv"
    records = manifest.get("experiments", [])
    solution_paths = ([output_dir / record["output_file"] for record in records if record.get("output_file")]
                      if records else [output_dir / "result_solution.csv"])
    all_results_present = bool(solution_paths) and all(path.exists() for path in solution_paths)
    pso_log = output_dir / "pso_fitting.log"
    de_log = output_dir / "de_fitting.log"
    node_log = output_dir / "NODE_fitting.log"
    error_file = output_dir / "fitting_error.txt"
    sources = output_dir / "snapshot" if (output_dir / "snapshot").is_dir() else session_dir
    event_log = sources / "generated" / "agent_logs" / "workflow_events.jsonl"
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
        *[f"- {path.name}: {'present' if path.exists() else 'missing'}" for path in solution_paths],
        f"- pso_fitting.log: {'present' if pso_log.exists() else 'missing'}",
        f"- de_fitting.log: {'present' if de_log.exists() else 'missing'}",
        f"- NODE_fitting.log: {'present' if node_log.exists() else 'missing'}",
        f"- fitting_error.txt: {'present' if error_file.exists() else 'missing'}",
        "",
        "Optimizer summary:",
        "- Global search: intentionally skipped (gradient-only restart)" if gradient_only else _format_summary(global_name, global_summary),
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
    if final_design.exists() and all_results_present:
        lines.append("- Review fitted parameters and measured-versus-fitted trajectories.")
    if not gradient_only and not pso_log.exists() and not de_log.exists():
        lines.append("- Global-search log missing; verify the PSO/DE stage ran.")
    if not node_log.exists():
        lines.append("- NODE log missing; verify the gradient-refinement stage ran.")
    if global_summary and global_summary["final_loss"] > global_summary["first_loss"]:
        lines.append("- Global-search loss worsened; review bounds and data/model consistency.")
    if node_summary and node_summary["final_loss"] > node_summary["first_loss"]:
        lines.append("- NODE loss worsened; inspect gradients, tolerances, and generated code.")

    if records:
        lines.extend(["", f"Experiments: {len(records)}", f"Aggregation: {manifest.get('aggregation', 'unknown')}"])
        summary_path = output_dir / "fit_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        losses = summary.get("experiment_losses", [])
        for index, record in enumerate(records):
            loss = losses[index] if index < len(losses) else "not recorded"
            lines.append(f"- Experiment {record['index']}: {record['data_file']}; loss={loss}; output={record.get('output_file') or 'disabled'}")
        for path in solution_paths:
            if not path.exists():
                lines.append(f"- Missing experiment result: {path.name}; inspect run status and writeout errors.")

    if gradient_only:
        lines.append(f"- Restart source: {manifest.get('source_run')}")
    curvature_path = output_dir / "sloppiness.json"
    if curvature_path.exists():
        curvature = json.loads(curvature_path.read_text())
        lines.extend(["", "Sloppiness:", f"- status: {curvature['status']}"])
        if curvature["status"] == "ok":
            lines.extend([
                f"- method: {curvature['method']}", f"- {curvature['verdict']}",
                f"- weak modes: {curvature['weak_modes']}; negative modes: {curvature['negative_modes']}",
                f"- full report: {output_dir / 'sloppiness_report.txt'}",
            ])
            lines.extend(f"- {warning}" for warning in curvature["warnings"])
        else:
            lines.append(f"- {curvature.get('reason', '')}")
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
