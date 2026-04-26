from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Room:
    room_id: int
    room_size: int
    is_available: bool = True


@dataclass(frozen=True)
class Offering:
    offering_id: int
    course_id: int
    section_id: int
    professor_id: int
    class_registration_size: int


@dataclass(frozen=True)
class SessionAssignment:
    day: int
    timeslot: int
    room_id: int


@dataclass(frozen=True)
class OfferingGene:
    session_a: SessionAssignment
    session_b: SessionAssignment


Chromosome = list[OfferingGene]


@dataclass(frozen=True)
class ProblemData:
    rooms: list[Room]
    offerings: list[Offering]
    professor_to_courses: dict[int, set[int]]
    professor_course_overload_count: int = 0


@dataclass
class HardConstraintBreakdown:
    invalid_session_pairs: int = 0
    room_capacity_mismatches: int = 0
    professor_conflicts: int = 0
    room_unavailable: int = 0
    section_conflicts: int = 0
    room_conflicts: int = 0
    professor_course_overload: int = 0

    @property
    def total(self) -> int:
        return (
            self.invalid_session_pairs
            + self.room_capacity_mismatches
            + self.professor_conflicts
            + self.room_unavailable
            + self.section_conflicts
            + self.room_conflicts
            + self.professor_course_overload
        )


@dataclass
class EvaluationResult:
    hard: HardConstraintBreakdown
    soft_penalty: float
    total_penalty: float
    fitness: float


@dataclass(frozen=True)
class ScheduleRow:
    offering_id: int
    course_id: int
    section_id: int
    professor_id: int
    session_index: int
    day: int
    timeslot: int
    room_id: int


@dataclass
class GenerationMetrics:
    generation: int
    best_fitness: float
    avg_fitness: float
    best_hard_violations: int
    best_soft_penalty: float


@dataclass
class RunArtifacts:
    output_dir: str
    json_path: str
    csv_path: str
    plot_path: str


@dataclass
class GARunResult:
    best_chromosome: Chromosome
    best_evaluation: EvaluationResult
    summary: object
    generation_metrics: list[GenerationMetrics] = field(default_factory=list)
