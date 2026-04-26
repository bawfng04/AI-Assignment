from __future__ import annotations

import random
from dataclasses import dataclass

from .config import ProblemConfig
from .models import (
    Chromosome,
    OfferingGene,
    ProblemData,
    ScheduleRow,
    SessionAssignment,
)


@dataclass(frozen=True)
class DecodedSession:
    offering_index: int
    session_index: int
    day: int
    timeslot: int
    room_id: int


def valid_day_pairs(num_days: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for first in range(num_days):
        for second in range(first + 1, num_days):
            if abs(second - first) >= 2:
                pairs.append((first, second))
    return pairs


def is_valid_day_pair(day_a: int, day_b: int) -> bool:
    return day_a != day_b and abs(day_a - day_b) >= 2


def random_day_pair(rng: random.Random, num_days: int) -> tuple[int, int]:
    pairs = valid_day_pairs(num_days)
    if not pairs:
        raise ValueError("No valid day pairs available for the current number of days")
    return rng.choice(pairs)


def random_room_id(
    rng: random.Random,
    data: ProblemData,
    class_registration_size: int,
) -> int:
    eligible = [
        room.room_id
        for room in data.rooms
        if room.is_available and room.room_size >= class_registration_size
    ]
    if not eligible:
        # Fallback keeps generation alive, hard constraint will penalize this.
        eligible = [room.room_id for room in data.rooms]
    return rng.choice(eligible)


def create_random_gene(
    offering_index: int,
    data: ProblemData,
    config: ProblemConfig,
    rng: random.Random,
) -> OfferingGene:
    offering = data.offerings[offering_index]
    day_a, day_b = random_day_pair(rng, config.num_days)

    session_a = SessionAssignment(
        day=day_a,
        timeslot=rng.randrange(config.num_timeslots_per_day),
        room_id=random_room_id(rng, data, offering.class_registration_size),
    )
    session_b = SessionAssignment(
        day=day_b,
        timeslot=rng.randrange(config.num_timeslots_per_day),
        room_id=random_room_id(rng, data, offering.class_registration_size),
    )
    return OfferingGene(session_a=session_a, session_b=session_b)


def create_random_chromosome(
    data: ProblemData,
    config: ProblemConfig,
    rng: random.Random,
) -> Chromosome:
    return [
        create_random_gene(i, data, config, rng)
        for i in range(len(data.offerings))
    ]


def repair_gene(
    gene: OfferingGene,
    offering_index: int,
    data: ProblemData,
    config: ProblemConfig,
    rng: random.Random,
) -> OfferingGene:
    offering = data.offerings[offering_index]
    session_a = gene.session_a
    session_b = gene.session_b

    if not is_valid_day_pair(session_a.day, session_b.day):
        day_a, day_b = random_day_pair(rng, config.num_days)
        session_a = SessionAssignment(
            day=day_a,
            timeslot=session_a.timeslot % config.num_timeslots_per_day,
            room_id=session_a.room_id,
        )
        session_b = SessionAssignment(
            day=day_b,
            timeslot=session_b.timeslot % config.num_timeslots_per_day,
            room_id=session_b.room_id,
        )

    room_ids = {room.room_id for room in data.rooms}
    if session_a.room_id not in room_ids:
        session_a = SessionAssignment(
            day=session_a.day,
            timeslot=session_a.timeslot % config.num_timeslots_per_day,
            room_id=random_room_id(rng, data, offering.class_registration_size),
        )
    if session_b.room_id not in room_ids:
        session_b = SessionAssignment(
            day=session_b.day,
            timeslot=session_b.timeslot % config.num_timeslots_per_day,
            room_id=random_room_id(rng, data, offering.class_registration_size),
        )

    compatible_rooms = {
        room.room_id
        for room in data.rooms
        if room.is_available and room.room_size >= offering.class_registration_size
    }
    if compatible_rooms and session_a.room_id not in compatible_rooms:
        session_a = SessionAssignment(
            day=session_a.day,
            timeslot=session_a.timeslot,
            room_id=rng.choice(sorted(compatible_rooms)),
        )
    if compatible_rooms and session_b.room_id not in compatible_rooms:
        session_b = SessionAssignment(
            day=session_b.day,
            timeslot=session_b.timeslot,
            room_id=rng.choice(sorted(compatible_rooms)),
        )

    session_a = SessionAssignment(
        day=session_a.day % config.num_days,
        timeslot=session_a.timeslot % config.num_timeslots_per_day,
        room_id=session_a.room_id,
    )
    session_b = SessionAssignment(
        day=session_b.day % config.num_days,
        timeslot=session_b.timeslot % config.num_timeslots_per_day,
        room_id=session_b.room_id,
    )
    return OfferingGene(session_a=session_a, session_b=session_b)


def repair_chromosome(
    chromosome: Chromosome,
    data: ProblemData,
    config: ProblemConfig,
    rng: random.Random,
) -> Chromosome:
    return [
        repair_gene(gene, i, data, config, rng)
        for i, gene in enumerate(chromosome)
    ]


def iter_decoded_sessions(chromosome: Chromosome):
    for offering_index, gene in enumerate(chromosome):
        yield DecodedSession(
            offering_index=offering_index,
            session_index=0,
            day=gene.session_a.day,
            timeslot=gene.session_a.timeslot,
            room_id=gene.session_a.room_id,
        )
        yield DecodedSession(
            offering_index=offering_index,
            session_index=1,
            day=gene.session_b.day,
            timeslot=gene.session_b.timeslot,
            room_id=gene.session_b.room_id,
        )


def chromosome_to_schedule_rows(
    chromosome: Chromosome,
    data: ProblemData,
) -> list[ScheduleRow]:
    rows: list[ScheduleRow] = []
    for offering_index, decoded in enumerate(iter_decoded_sessions(chromosome)):
        offering = data.offerings[decoded.offering_index]
        rows.append(
            ScheduleRow(
                offering_id=offering.offering_id,
                course_id=offering.course_id,
                section_id=offering.section_id,
                professor_id=offering.professor_id,
                session_index=decoded.session_index,
                day=decoded.day,
                timeslot=decoded.timeslot,
                room_id=decoded.room_id,
            )
        )
    return sorted(
        rows,
        key=lambda row: (row.day, row.timeslot, row.room_id, row.offering_id, row.session_index),
    )
