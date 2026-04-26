from __future__ import annotations

import argparse
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from topic3_ga.config import GAConfig
from topic3_ga.multi_seed import parse_seeds, run_multi_seed_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Topic 3 GA for multiple seeds and export report tables"
    )
    parser.add_argument("--seeds", type=str, default="40,41,42,43,44")
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
    parser.add_argument("--run-name", type=str, default="topic3_multiseed")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
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

    summary_rows, stats, paths = run_multi_seed_report(
        seeds=seeds,
        offerings=args.offerings,
        ga_config=ga_config,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )

    for row in summary_rows:
        print(
            "[seed={}] hard={} soft={:.4f} fitness={:.8f}".format(
                row["seed"],
                row["best_hard_violations"],
                row["best_soft_penalty"],
                row["best_fitness"],
            )
        )

    print("\n=== Multi-seed summary ===")
    print(f"Runs: {len(summary_rows)}")
    print(f"Feasible rate: {stats['feasible_rate']:.2%}")
    print(f"Avg soft penalty: {stats['avg_soft']:.4f}")
    print(f"Std soft penalty: {stats['std_soft']:.4f}")
    print(f"Summary CSV: {paths['summary_csv']}")
    print(f"Summary MD: {paths['summary_md']}")
    print(f"Summary JSON: {paths['summary_json']}")


if __name__ == "__main__":
    main()
