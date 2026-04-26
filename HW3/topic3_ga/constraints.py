from __future__ import annotations

from collections import defaultdict
import math

from .config import GAConfig, ProblemConfig
from .encoding import iter_decoded_sessions, is_valid_day_pair
from .models import Chromosome, EvaluationResult, HardConstraintBreakdown, ProblemData


def evaluate_hard_constraints(
    chromosome: Chromosome,
    data: ProblemData,
    config: ProblemConfig,
) -> HardConstraintBreakdown:
    hard = HardConstraintBreakdown(
        professor_course_overload=data.professor_course_overload_count
    )

    room_by_id = {room.room_id: room for room in data.rooms}
    professor_time_count: dict[tuple[int, int, int], int] = defaultdict(int)
    section_time_count: dict[tuple[int, int, int], int] = defaultdict(int)
    room_time_count: dict[tuple[int, int, int], int] = defaultdict(int)

    for offering_index, gene in enumerate(chromosome):
        offering = data.offerings[offering_index]

        if not is_valid_day_pair(gene.session_a.day, gene.session_b.day):
            hard.invalid_session_pairs += 1

        for session in (gene.session_a, gene.session_b):
            room = room_by_id.get(session.room_id)
            if room is None or not room.is_available:
                hard.room_unavailable += 1
            elif room.room_size < offering.class_registration_size:
                hard.room_capacity_mismatches += 1

            professor_time_count[(session.day, session.timeslot, offering.professor_id)] += 1
            section_time_count[(session.day, session.timeslot, offering.section_id)] += 1
            room_time_count[(session.day, session.timeslot, session.room_id)] += 1

    hard.professor_conflicts += sum(max(0, c - 1) for c in professor_time_count.values())
    hard.section_conflicts += sum(max(0, c - 1) for c in section_time_count.values())
    hard.room_conflicts += sum(max(0, c - 1) for c in room_time_count.values())

    return hard


def compute_soft_penalty(
    chromosome: Chromosome,
    data: ProblemData,
    config: ProblemConfig,
) -> float:
    slot_loads = [0 for _ in range(config.num_days * config.num_timeslots_per_day)]
    room_loads: dict[int, int] = defaultdict(int)
    section_day_slots: dict[tuple[int, int], list[int]] = defaultdict(list)

    for decoded in iter_decoded_sessions(chromosome):
        linear_slot = decoded.day * config.num_timeslots_per_day + decoded.timeslot
        if 0 <= linear_slot < len(slot_loads):
            slot_loads[linear_slot] += 1
        room_loads[decoded.room_id] += 1

        offering = data.offerings[decoded.offering_index]
        section_day_slots[(offering.section_id, decoded.day)].append(decoded.timeslot)

    if slot_loads:
        mean_slot = sum(slot_loads) / len(slot_loads)
        slot_var = sum((load - mean_slot) ** 2 for load in slot_loads) / len(slot_loads)
        slot_std = math.sqrt(slot_var)
    else:
        slot_std = 0.0

    room_load_values = list(room_loads.values())
    if room_load_values:
        mean_room = sum(room_load_values) / len(room_load_values)
        room_var = sum((load - mean_room) ** 2 for load in room_load_values) / len(
            room_load_values
        )
        room_std = math.sqrt(room_var)
    else:
        room_std = 0.0

    idle_gap_penalty = 0.0
    for slots in section_day_slots.values():
        ordered = sorted(slots)
        for idx in range(1, len(ordered)):
            gap = ordered[idx] - ordered[idx - 1]
            if gap > 1:
                idle_gap_penalty += float(gap - 1)

    return slot_std + room_std + idle_gap_penalty


def evaluate_chromosome(
    chromosome: Chromosome,
    data: ProblemData,
    problem_config: ProblemConfig,
    ga_config: GAConfig,
) -> EvaluationResult:
    hard = evaluate_hard_constraints(chromosome, data, problem_config)
    soft_penalty = compute_soft_penalty(chromosome, data, problem_config)
    total_penalty = (
        ga_config.hard_penalty_weight * hard.total
        + ga_config.soft_penalty_weight * soft_penalty
    )
    fitness = 1.0 / (1.0 + total_penalty)
    return EvaluationResult(
        hard=hard,
        soft_penalty=soft_penalty,
        total_penalty=total_penalty,
        fitness=fitness,
    )
