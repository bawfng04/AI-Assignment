from __future__ import annotations

import csv
import json
import os
import statistics
from dataclasses import asdict

from .config import GAConfig, ProblemConfig, RunConfig
from .main import run_scheduler


def parse_seeds(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("Seed list is empty")
    return [int(part) for part in parts]


def write_summary_csv(path: str, rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "generations_run",
                "best_fitness",
                "best_total_penalty",
                "best_hard_violations",
                "best_soft_penalty",
                "json_path",
                "csv_path",
                "plot_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(
    path: str,
    rows: list[dict[str, object]],
    stats: dict[str, float],
) -> None:
    lines: list[str] = []
    lines.append("# Multi-seed GA Summary")
    lines.append("")
    lines.append("## Per-seed Results")
    lines.append("")
    lines.append("| Seed | Generations | Best Fitness | Best Total Penalty | Hard Violations | Soft Penalty |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {seed} | {generations_run} | {best_fitness:.8f} | {best_total_penalty:.4f} | {best_hard_violations} | {best_soft_penalty:.4f} |".format(
                **row
            )
        )

    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Runs | {int(stats['runs'])} |")
    lines.append(f"| Feasible runs (hard=0) | {int(stats['feasible_runs'])} |")
    lines.append(f"| Feasible rate | {stats['feasible_rate']:.2%} |")
    lines.append(f"| Avg soft penalty | {stats['avg_soft']:.4f} |")
    lines.append(f"| Std soft penalty | {stats['std_soft']:.4f} |")
    lines.append(f"| Min soft penalty | {stats['min_soft']:.4f} |")
    lines.append(f"| Max soft penalty | {stats['max_soft']:.4f} |")
    lines.append(f"| Avg generations run | {stats['avg_generations']:.2f} |")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_multi_seed_report(
    seeds: list[int],
    offerings: int,
    ga_config: GAConfig,
    output_dir: str,
    run_name: str,
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, str]]:
    os.makedirs(output_dir, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for seed in seeds:
        problem_config = ProblemConfig(seed=seed, num_offerings=offerings)
        run_config = RunConfig(
            output_dir=output_dir,
            run_name=run_name,
            print_schedule_table=False,
        )

        _, result, _, artifacts = run_scheduler(problem_config, ga_config, run_config)

        row = {
            "seed": seed,
            "generations_run": result.summary.generations_run,
            "best_fitness": result.summary.best_fitness,
            "best_total_penalty": result.summary.best_total_penalty,
            "best_hard_violations": result.summary.best_hard_violations,
            "best_soft_penalty": result.summary.best_soft_penalty,
            "json_path": artifacts.json_path,
            "csv_path": artifacts.csv_path,
            "plot_path": artifacts.plot_path,
        }
        summary_rows.append(row)

    soft_values = [float(row["best_soft_penalty"]) for row in summary_rows]
    hard_values = [int(row["best_hard_violations"]) for row in summary_rows]
    generation_values = [int(row["generations_run"]) for row in summary_rows]

    feasible_runs = sum(1 for value in hard_values if value == 0)
    stats = {
        "runs": float(len(summary_rows)),
        "feasible_runs": float(feasible_runs),
        "feasible_rate": feasible_runs / len(summary_rows),
        "avg_soft": statistics.mean(soft_values),
        "std_soft": statistics.pstdev(soft_values) if len(soft_values) > 1 else 0.0,
        "min_soft": min(soft_values),
        "max_soft": max(soft_values),
        "avg_generations": statistics.mean(generation_values),
    }

    summary_csv = os.path.join(output_dir, f"{run_name}_summary.csv")
    summary_md = os.path.join(output_dir, f"{run_name}_summary.md")
    summary_json = os.path.join(output_dir, f"{run_name}_summary_stats.json")

    write_summary_csv(summary_csv, summary_rows)
    write_summary_markdown(summary_md, summary_rows, stats)
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "stats": stats,
                "runs": summary_rows,
                "ga_config": asdict(ga_config),
                "seeds": seeds,
            },
            handle,
            indent=2,
        )

    paths = {
        "summary_csv": summary_csv,
        "summary_md": summary_md,
        "summary_json": summary_json,
    }
    return summary_rows, stats, paths
