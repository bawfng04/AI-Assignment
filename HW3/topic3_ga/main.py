from __future__ import annotations

import argparse
from collections.abc import Callable
import os

from .config import GAConfig, ProblemConfig, RunConfig
from .data_generation import generate_problem_data, validate_problem_data
from .encoding import chromosome_to_schedule_rows
from .export import (
    build_artifacts,
    ensure_output_dir,
    export_schedule_csv,
    export_schedule_json,
    format_schedule_table,
)
from .ga_engine import GeneticScheduler
from .models import GARunResult, ProblemData, RunArtifacts, ScheduleRow
from .models import GenerationMetrics
from .visualization import save_fitness_plot


def run_scheduler(
    problem_config: ProblemConfig,
    ga_config: GAConfig,
    run_config: RunConfig,
    generation_callback: Callable[[GenerationMetrics], None] | None = None,
) -> tuple[ProblemData, GARunResult, list[ScheduleRow], RunArtifacts]:
    data = generate_problem_data(problem_config)
    warnings = validate_problem_data(data)
    if warnings:
        for warning in warnings:
            print(f"[WARN] {warning}")

    scheduler = GeneticScheduler(problem_config, ga_config, seed=problem_config.seed)
    result = scheduler.run(data, generation_callback=generation_callback)

    rows = chromosome_to_schedule_rows(result.best_chromosome, data)

    output_dir = ensure_output_dir(
        os.path.join(run_config.output_dir, f"{run_config.run_name}_seed{problem_config.seed}")
    )
    json_path = export_schedule_json(
        output_dir,
        run_config.export_json_name,
        rows,
        result.summary,
        result.best_evaluation,
    )
    csv_path = export_schedule_csv(output_dir, run_config.export_csv_name, rows)
    plot_path = save_fitness_plot(
        result.generation_metrics,
        output_dir,
        run_config.export_plot_name,
    )
    artifacts = build_artifacts(output_dir, json_path, csv_path, plot_path)

    return data, result, rows, artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HW3 Topic 3 - Class Scheduling using Genetic Algorithm"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offerings", type=int, default=30)
    parser.add_argument("--population", type=int, default=240)
    parser.add_argument("--generations", type=int, default=420)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--crossover-rate", type=float, default=0.95)
    parser.add_argument("--elitism", type=int, default=10)
    parser.add_argument("--tournament-size", type=int, default=4)
    parser.add_argument("--no-improvement-patience", type=int, default=100)
    parser.add_argument("--feasible-streak-patience", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--run-name", type=str, default="topic3_ga")
    parser.add_argument("--no-print-table", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    problem_config = ProblemConfig(seed=args.seed, num_offerings=args.offerings)
    ga_config = GAConfig(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elitism_count=args.elitism,
        tournament_size=args.tournament_size,
        no_improvement_patience=args.no_improvement_patience,
        feasible_streak_patience=args.feasible_streak_patience,
    )
    run_config = RunConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        print_schedule_table=not args.no_print_table,
    )

    data, result, rows, artifacts = run_scheduler(problem_config, ga_config, run_config)

    print("=== HW3 Topic 3 - GA Scheduling Summary ===")
    print(f"Seed: {problem_config.seed}")
    print(f"Offerings: {len(data.offerings)}")
    print(f"Generations run: {result.summary.generations_run}")
    print(f"Best fitness: {result.summary.best_fitness:.8f}")
    print(f"Best total penalty: {result.summary.best_total_penalty:.4f}")
    print(f"Best hard violations: {result.summary.best_hard_violations}")
    print(f"Best soft penalty: {result.summary.best_soft_penalty:.4f}")

    if run_config.print_schedule_table:
        print("\n=== Final Schedule Table ===")
        print(format_schedule_table(rows))

    print("\n=== Artifacts ===")
    print(f"Output directory: {artifacts.output_dir}")
    print(f"JSON: {artifacts.json_path}")
    print(f"CSV: {artifacts.csv_path}")
    print(f"Fitness plot: {artifacts.plot_path}")


if __name__ == "__main__":
    main()
