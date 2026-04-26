from __future__ import annotations

import random

from .config import ProblemConfig
from .encoding import random_day_pair, random_room_id, repair_chromosome, repair_gene
from .models import Chromosome, OfferingGene, ProblemData, SessionAssignment


def tournament_selection(
    population: list[Chromosome],
    penalties: list[float],
    tournament_size: int,
    rng: random.Random,
) -> Chromosome:
    chosen_indices = [rng.randrange(len(population)) for _ in range(tournament_size)]
    best_idx = min(chosen_indices, key=lambda idx: penalties[idx])
    return list(population[best_idx])


def offering_uniform_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    data: ProblemData,
    config: ProblemConfig,
    rng: random.Random,
) -> tuple[Chromosome, Chromosome]:
    child_a: Chromosome = []
    child_b: Chromosome = []

    for index, (gene_a, gene_b) in enumerate(zip(parent_a, parent_b)):
        if rng.random() < 0.5:
            child_a.append(gene_a)
            child_b.append(gene_b)
        else:
            child_a.append(gene_b)
            child_b.append(gene_a)

    return (
        repair_chromosome(child_a, data, config, rng),
        repair_chromosome(child_b, data, config, rng),
    )


def mutate_chromosome(
    chromosome: Chromosome,
    data: ProblemData,
    config: ProblemConfig,
    mutation_rate: float,
    rng: random.Random,
) -> Chromosome:
    mutated = list(chromosome)

    for offering_index, gene in enumerate(mutated):
        if rng.random() >= mutation_rate:
            continue
        mutated[offering_index] = _mutate_gene(gene, offering_index, data, config, rng)

    if len(mutated) >= 2 and rng.random() < mutation_rate * 0.4:
        first_idx, second_idx = rng.sample(range(len(mutated)), 2)
        mutated[first_idx], mutated[second_idx] = _swap_one_session(
            mutated[first_idx], mutated[second_idx], rng
        )

    return repair_chromosome(mutated, data, config, rng)


def _mutate_gene(
    gene: OfferingGene,
    offering_index: int,
    data: ProblemData,
    config: ProblemConfig,
    rng: random.Random,
) -> OfferingGene:
    operation = rng.choice(["room", "slot", "day_pair", "mixed"])
    offering = data.offerings[offering_index]

    session_a = gene.session_a
    session_b = gene.session_b

    if operation in {"room", "mixed"}:
        if rng.random() < 0.5:
            session_a = SessionAssignment(
                day=session_a.day,
                timeslot=session_a.timeslot,
                room_id=random_room_id(rng, data, offering.class_registration_size),
            )
        else:
            session_b = SessionAssignment(
                day=session_b.day,
                timeslot=session_b.timeslot,
                room_id=random_room_id(rng, data, offering.class_registration_size),
            )

    if operation in {"slot", "mixed"}:
        if rng.random() < 0.5:
            session_a = SessionAssignment(
                day=session_a.day,
                timeslot=rng.randrange(config.num_timeslots_per_day),
                room_id=session_a.room_id,
            )
        else:
            session_b = SessionAssignment(
                day=session_b.day,
                timeslot=rng.randrange(config.num_timeslots_per_day),
                room_id=session_b.room_id,
            )

    if operation == "day_pair" or (operation == "mixed" and rng.random() < 0.6):
        day_a, day_b = random_day_pair(rng, config.num_days)
        session_a = SessionAssignment(
            day=day_a,
            timeslot=session_a.timeslot,
            room_id=session_a.room_id,
        )
        session_b = SessionAssignment(
            day=day_b,
            timeslot=session_b.timeslot,
            room_id=session_b.room_id,
        )

    updated = OfferingGene(session_a=session_a, session_b=session_b)
    return repair_gene(updated, offering_index, data, config, rng)


def _swap_one_session(
    first: OfferingGene,
    second: OfferingGene,
    rng: random.Random,
) -> tuple[OfferingGene, OfferingGene]:
    if rng.random() < 0.5:
        return (
            OfferingGene(session_a=second.session_a, session_b=first.session_b),
            OfferingGene(session_a=first.session_a, session_b=second.session_b),
        )

    return (
        OfferingGene(session_a=first.session_a, session_b=second.session_b),
        OfferingGene(session_a=second.session_a, session_b=first.session_b),
    )
