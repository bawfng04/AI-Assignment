from __future__ import annotations

from dataclasses import dataclass, field


DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]


@dataclass(frozen=True)
class ProblemConfig:
    num_courses: int = 15
    num_sections: int = 10
    num_professors: int = 10
    num_days: int = 5
    num_timeslots_per_day: int = 6
    num_rooms: int = 30
    num_offerings: int = 30
    sessions_per_offering: int = 2
    max_courses_per_professor: int = 3
    seed: int = 42


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 240
    generations: int = 420
    tournament_size: int = 4
    crossover_rate: float = 0.95
    mutation_rate: float = 0.2
    elitism_count: int = 10
    hard_penalty_weight: float = 1000.0
    soft_penalty_weight: float = 1.0
    no_improvement_patience: int = 100
    feasible_streak_patience: int = 100


@dataclass(frozen=True)
class RunConfig:
    output_dir: str = "outputs"
    run_name: str = "topic3_ga"
    export_json_name: str = "best_schedule.json"
    export_csv_name: str = "best_schedule.csv"
    export_plot_name: str = "fitness_plot.png"
    print_schedule_table: bool = True


@dataclass
class ScheduleSummary:
    generations_run: int
    best_fitness: float
    best_total_penalty: float
    best_hard_violations: int
    best_soft_penalty: float
    history_best_fitness: list[float] = field(default_factory=list)
    history_avg_fitness: list[float] = field(default_factory=list)
    history_hard_violations: list[int] = field(default_factory=list)
    history_soft_penalty: list[float] = field(default_factory=list)
