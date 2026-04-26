import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topic3_ga.config import GAConfig, ProblemConfig, RunConfig
from topic3_ga.main import run_scheduler


class TestSmokeRun(unittest.TestCase):
    def test_end_to_end_creates_all_artifacts_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            problem_config = ProblemConfig(seed=42, num_offerings=20)
            ga_config = GAConfig(
                population_size=40,
                generations=60,
                no_improvement_patience=20,
                feasible_streak_patience=10,
            )
            run_config = RunConfig(
                output_dir=temp_dir,
                run_name="smoke",
                print_schedule_table=False,
            )

            _, result_one, rows_one, artifacts_one = run_scheduler(
                problem_config,
                ga_config,
                run_config,
            )
            _, result_two, rows_two, artifacts_two = run_scheduler(
                problem_config,
                ga_config,
                run_config,
            )

            self.assertGreater(len(rows_one), 0)
            self.assertEqual(len(rows_one), len(rows_two))
            self.assertEqual(
                result_one.summary.best_total_penalty,
                result_two.summary.best_total_penalty,
            )
            self.assertEqual(result_one.summary.best_fitness, result_two.summary.best_fitness)

            self.assertTrue(os.path.exists(artifacts_one.json_path))
            self.assertTrue(os.path.exists(artifacts_one.csv_path))
            self.assertTrue(os.path.exists(artifacts_one.plot_path))

            self.assertTrue(os.path.exists(artifacts_two.json_path))
            self.assertTrue(os.path.exists(artifacts_two.csv_path))
            self.assertTrue(os.path.exists(artifacts_two.plot_path))


if __name__ == "__main__":
    unittest.main()
