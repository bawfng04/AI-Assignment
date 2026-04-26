from __future__ import annotations

from collections.abc import Callable
import random

from .config import GAConfig, ProblemConfig, ScheduleSummary
from .constraints import evaluate_chromosome
from .encoding import create_random_chromosome, random_day_pair, random_room_id, repair_chromosome
from .models import (
    Chromosome,
    EvaluationResult,
    GARunResult,
    GenerationMetrics,
    OfferingGene,
    ProblemData,
    SessionAssignment,
)
from .operators import mutate_chromosome, offering_uniform_crossover, tournament_selection


class GeneticScheduler:
    def __init__(
        self,
        problem_config: ProblemConfig,
        ga_config: GAConfig,
        seed: int | None = None,
    ) -> None:
        self.problem_config = problem_config
        self.ga_config = ga_config
        self.rng = random.Random(seed if seed is not None else problem_config.seed)

    def run(
        self,
        data: ProblemData,
        generation_callback: Callable[[GenerationMetrics], None] | None = None,
    ) -> GARunResult:
        population = self._initialize_population(data)

        best_chromosome: Chromosome | None = None
        best_eval: EvaluationResult | None = None
        metrics: list[GenerationMetrics] = []

        no_improvement = 0
        feasible_streak = 0

        for generation in range(self.ga_config.generations):
            evaluations = [
                evaluate_chromosome(chromosome, data, self.problem_config, self.ga_config)
                for chromosome in population
            ]

            ranked_indices = sorted(
                range(len(population)), key=lambda idx: evaluations[idx].total_penalty
            )
            best_idx = ranked_indices[0]
            generation_best_eval = evaluations[best_idx]
            generation_best = population[best_idx]

            avg_fitness = sum(ev.fitness for ev in evaluations) / len(evaluations)
            metrics.append(
                GenerationMetrics(
                    generation=generation,
                    best_fitness=generation_best_eval.fitness,
                    avg_fitness=avg_fitness,
                    best_hard_violations=generation_best_eval.hard.total,
                    best_soft_penalty=generation_best_eval.soft_penalty,
                )
            )
            if generation_callback is not None:
                generation_callback(metrics[-1])

            if best_eval is None or generation_best_eval.total_penalty < best_eval.total_penalty:
                best_eval = generation_best_eval
                best_chromosome = list(generation_best)
                no_improvement = 0
            else:
                no_improvement += 1

            if generation_best_eval.hard.total == 0:
                feasible_streak += 1
            else:
                feasible_streak = 0

            if (
                no_improvement >= self.ga_config.no_improvement_patience
                or feasible_streak >= self.ga_config.feasible_streak_patience
            ):
                break

            penalties = [ev.total_penalty for ev in evaluations]
            population = self._next_generation(population, penalties, data)

        if best_chromosome is None or best_eval is None:
            raise RuntimeError("GA finished without producing a valid candidate")

        summary = ScheduleSummary(
            generations_run=len(metrics),
            best_fitness=best_eval.fitness,
            best_total_penalty=best_eval.total_penalty,
            best_hard_violations=best_eval.hard.total,
            best_soft_penalty=best_eval.soft_penalty,
            history_best_fitness=[m.best_fitness for m in metrics],
            history_avg_fitness=[m.avg_fitness for m in metrics],
            history_hard_violations=[m.best_hard_violations for m in metrics],
            history_soft_penalty=[m.best_soft_penalty for m in metrics],
        )

        return GARunResult(
            best_chromosome=best_chromosome,
            best_evaluation=best_eval,
            summary=summary,
            generation_metrics=metrics,
        )

    def _initialize_population(self, data: ProblemData) -> list[Chromosome]:
        population: list[Chromosome] = []
        heuristic_count = max(1, int(self.ga_config.population_size * 0.7))

        for _ in range(heuristic_count):
            chromosome = self._create_heuristic_chromosome(data)
            population.append(repair_chromosome(chromosome, data, self.problem_config, self.rng))

        while len(population) < self.ga_config.population_size:
            chromosome = create_random_chromosome(data, self.problem_config, self.rng)
            population.append(repair_chromosome(chromosome, data, self.problem_config, self.rng))

        return population

    def _create_heuristic_chromosome(self, data: ProblemData) -> Chromosome:
        chromosome: Chromosome = []

        professor_time: set[tuple[int, int, int]] = set()
        section_time: set[tuple[int, int, int]] = set()
        room_time: set[tuple[int, int, int]] = set()

        for offering in data.offerings:
            best_gene: OfferingGene | None = None
            best_local_penalty: int | None = None

            for _ in range(24):
                day_a, day_b = random_day_pair(self.rng, self.problem_config.num_days)
                slot_a = self.rng.randrange(self.problem_config.num_timeslots_per_day)
                slot_b = self.rng.randrange(self.problem_config.num_timeslots_per_day)
                room_a = random_room_id(self.rng, data, offering.class_registration_size)
                room_b = random_room_id(self.rng, data, offering.class_registration_size)

                local_penalty = 0
                local_penalty += int((day_a, slot_a, offering.professor_id) in professor_time)
                local_penalty += int((day_b, slot_b, offering.professor_id) in professor_time)
                local_penalty += int((day_a, slot_a, offering.section_id) in section_time)
                local_penalty += int((day_b, slot_b, offering.section_id) in section_time)
                local_penalty += int((day_a, slot_a, room_a) in room_time)
                local_penalty += int((day_b, slot_b, room_b) in room_time)

                candidate = OfferingGene(
                    session_a=SessionAssignment(day=day_a, timeslot=slot_a, room_id=room_a),
                    session_b=SessionAssignment(day=day_b, timeslot=slot_b, room_id=room_b),
                )

                if best_local_penalty is None or local_penalty < best_local_penalty:
                    best_gene = candidate
                    best_local_penalty = local_penalty
                if local_penalty == 0:
                    break

            if best_gene is None:
                best_gene = OfferingGene(
                    session_a=SessionAssignment(
                        day=0,
                        timeslot=0,
                        room_id=random_room_id(self.rng, data, offering.class_registration_size),
                    ),
                    session_b=SessionAssignment(
                        day=2,
                        timeslot=0,
                        room_id=random_room_id(self.rng, data, offering.class_registration_size),
                    ),
                )

            chromosome.append(best_gene)
            professor_time.add(
                (best_gene.session_a.day, best_gene.session_a.timeslot, offering.professor_id)
            )
            professor_time.add(
                (best_gene.session_b.day, best_gene.session_b.timeslot, offering.professor_id)
            )
            section_time.add(
                (best_gene.session_a.day, best_gene.session_a.timeslot, offering.section_id)
            )
            section_time.add(
                (best_gene.session_b.day, best_gene.session_b.timeslot, offering.section_id)
            )
            room_time.add((best_gene.session_a.day, best_gene.session_a.timeslot, best_gene.session_a.room_id))
            room_time.add((best_gene.session_b.day, best_gene.session_b.timeslot, best_gene.session_b.room_id))

        return chromosome

    def _next_generation(
        self,
        population: list[Chromosome],
        penalties: list[float],
        data: ProblemData,
    ) -> list[Chromosome]:
        ranked = sorted(
            range(len(population)), key=lambda idx: penalties[idx]
        )

        next_population: list[Chromosome] = [
            list(population[idx]) for idx in ranked[: self.ga_config.elitism_count]
        ]

        while len(next_population) < self.ga_config.population_size:
            parent_a = tournament_selection(
                population,
                penalties,
                tournament_size=self.ga_config.tournament_size,
                rng=self.rng,
            )
            parent_b = tournament_selection(
                population,
                penalties,
                tournament_size=self.ga_config.tournament_size,
                rng=self.rng,
            )

            if self.rng.random() < self.ga_config.crossover_rate:
                child_a, child_b = offering_uniform_crossover(
                    parent_a,
                    parent_b,
                    data,
                    self.problem_config,
                    self.rng,
                )
            else:
                child_a, child_b = list(parent_a), list(parent_b)

            child_a = mutate_chromosome(
                child_a,
                data,
                self.problem_config,
                mutation_rate=self.ga_config.mutation_rate,
                rng=self.rng,
            )
            child_b = mutate_chromosome(
                child_b,
                data,
                self.problem_config,
                mutation_rate=self.ga_config.mutation_rate,
                rng=self.rng,
            )

            next_population.append(child_a)
            if len(next_population) < self.ga_config.population_size:
                next_population.append(child_b)

        return next_population
