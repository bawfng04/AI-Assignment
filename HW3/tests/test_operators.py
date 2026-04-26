import os
import random
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topic3_ga.config import ProblemConfig
from topic3_ga.data_generation import generate_problem_data
from topic3_ga.encoding import create_random_chromosome, is_valid_day_pair
from topic3_ga.operators import mutate_chromosome, offering_uniform_crossover


class TestOperators(unittest.TestCase):
    def setUp(self) -> None:
        self.problem_config = ProblemConfig(seed=7, num_offerings=10)
        self.data = generate_problem_data(self.problem_config)
        self.rng = random.Random(99)

    def _assert_chromosome_structure(self, chromosome):
        self.assertEqual(len(chromosome), len(self.data.offerings))
        room_ids = {room.room_id for room in self.data.rooms}

        for gene in chromosome:
            self.assertTrue(is_valid_day_pair(gene.session_a.day, gene.session_b.day))
            self.assertTrue(0 <= gene.session_a.day < self.problem_config.num_days)
            self.assertTrue(0 <= gene.session_b.day < self.problem_config.num_days)
            self.assertTrue(
                0 <= gene.session_a.timeslot < self.problem_config.num_timeslots_per_day
            )
            self.assertTrue(
                0 <= gene.session_b.timeslot < self.problem_config.num_timeslots_per_day
            )
            self.assertIn(gene.session_a.room_id, room_ids)
            self.assertIn(gene.session_b.room_id, room_ids)

    def test_crossover_preserves_chromosome_shape(self):
        parent_a = create_random_chromosome(self.data, self.problem_config, self.rng)
        parent_b = create_random_chromosome(self.data, self.problem_config, self.rng)

        child_a, child_b = offering_uniform_crossover(
            parent_a,
            parent_b,
            self.data,
            self.problem_config,
            self.rng,
        )

        self._assert_chromosome_structure(child_a)
        self._assert_chromosome_structure(child_b)

    def test_mutation_preserves_chromosome_shape(self):
        chromosome = create_random_chromosome(self.data, self.problem_config, self.rng)
        mutated = mutate_chromosome(
            chromosome,
            self.data,
            self.problem_config,
            mutation_rate=0.8,
            rng=self.rng,
        )
        self._assert_chromosome_structure(mutated)


if __name__ == "__main__":
    unittest.main()
