import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topic3_ga.config import ProblemConfig
from topic3_ga.constraints import evaluate_hard_constraints
from topic3_ga.models import Offering, OfferingGene, ProblemData, Room, SessionAssignment


class TestHardConstraints(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProblemConfig(num_offerings=2)
        self.rooms = [
            Room(room_id=1, room_size=0, is_available=True),
            Room(room_id=2, room_size=1, is_available=True),
            Room(room_id=3, room_size=1, is_available=False),
        ]
        self.offerings = [
            Offering(
                offering_id=1,
                course_id=1,
                section_id=1,
                professor_id=1,
                class_registration_size=1,
            ),
            Offering(
                offering_id=2,
                course_id=2,
                section_id=2,
                professor_id=2,
                class_registration_size=0,
            ),
        ]
        self.data = ProblemData(
            rooms=self.rooms,
            offerings=self.offerings,
            professor_to_courses={1: {1}, 2: {2}},
            professor_course_overload_count=0,
        )

    def _valid_chromosome(self):
        return [
            OfferingGene(
                session_a=SessionAssignment(day=0, timeslot=0, room_id=2),
                session_b=SessionAssignment(day=2, timeslot=1, room_id=2),
            ),
            OfferingGene(
                session_a=SessionAssignment(day=1, timeslot=2, room_id=1),
                session_b=SessionAssignment(day=3, timeslot=3, room_id=1),
            ),
        ]

    def test_valid_case_has_no_hard_violation(self):
        hard = evaluate_hard_constraints(self._valid_chromosome(), self.data, self.config)
        self.assertEqual(hard.total, 0)

    def test_invalid_session_pair_detected(self):
        chromosome = self._valid_chromosome()
        chromosome[0] = OfferingGene(
            session_a=SessionAssignment(day=1, timeslot=0, room_id=2),
            session_b=SessionAssignment(day=1, timeslot=1, room_id=2),
        )
        hard = evaluate_hard_constraints(chromosome, self.data, self.config)
        self.assertGreater(hard.invalid_session_pairs, 0)

    def test_room_capacity_mismatch_detected(self):
        chromosome = self._valid_chromosome()
        chromosome[0] = OfferingGene(
            session_a=SessionAssignment(day=0, timeslot=0, room_id=1),
            session_b=SessionAssignment(day=2, timeslot=1, room_id=2),
        )
        hard = evaluate_hard_constraints(chromosome, self.data, self.config)
        self.assertGreater(hard.room_capacity_mismatches, 0)

    def test_professor_conflict_detected(self):
        offerings = [self.offerings[0], Offering(2, 2, 2, 1, 0)]
        data = ProblemData(
            rooms=self.rooms,
            offerings=offerings,
            professor_to_courses={1: {1, 2}},
            professor_course_overload_count=0,
        )
        chromosome = self._valid_chromosome()
        chromosome[1] = OfferingGene(
            session_a=SessionAssignment(day=0, timeslot=0, room_id=1),
            session_b=SessionAssignment(day=3, timeslot=3, room_id=1),
        )
        hard = evaluate_hard_constraints(chromosome, data, self.config)
        self.assertGreater(hard.professor_conflicts, 0)

    def test_room_unavailable_detected(self):
        chromosome = self._valid_chromosome()
        chromosome[0] = OfferingGene(
            session_a=SessionAssignment(day=0, timeslot=0, room_id=3),
            session_b=SessionAssignment(day=2, timeslot=1, room_id=2),
        )
        hard = evaluate_hard_constraints(chromosome, self.data, self.config)
        self.assertGreater(hard.room_unavailable, 0)

    def test_section_conflict_detected(self):
        offerings = [self.offerings[0], Offering(2, 2, 1, 2, 0)]
        data = ProblemData(
            rooms=self.rooms,
            offerings=offerings,
            professor_to_courses={1: {1}, 2: {2}},
            professor_course_overload_count=0,
        )
        chromosome = self._valid_chromosome()
        chromosome[1] = OfferingGene(
            session_a=SessionAssignment(day=0, timeslot=0, room_id=1),
            session_b=SessionAssignment(day=3, timeslot=3, room_id=1),
        )
        hard = evaluate_hard_constraints(chromosome, data, self.config)
        self.assertGreater(hard.section_conflicts, 0)

    def test_room_conflict_detected(self):
        chromosome = self._valid_chromosome()
        chromosome[1] = OfferingGene(
            session_a=SessionAssignment(day=0, timeslot=0, room_id=2),
            session_b=SessionAssignment(day=3, timeslot=3, room_id=1),
        )
        hard = evaluate_hard_constraints(chromosome, self.data, self.config)
        self.assertGreater(hard.room_conflicts, 0)

    def test_professor_course_overload_detected(self):
        data = ProblemData(
            rooms=self.rooms,
            offerings=self.offerings,
            professor_to_courses={1: {1, 2, 3, 4}},
            professor_course_overload_count=2,
        )
        hard = evaluate_hard_constraints(self._valid_chromosome(), data, self.config)
        self.assertEqual(hard.professor_course_overload, 2)


if __name__ == "__main__":
    unittest.main()
