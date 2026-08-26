import json
import unittest
from pathlib import Path

from scripts.run_canonical import DATASETS, METHODS, SEEDS, build_command, load_freeze


ROOT = Path(__file__).resolve().parents[1]


class CanonicalRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze = load_freeze(ROOT / "experiments/documented_dev_selection_v1.json")

    def test_grid_contains_exactly_sixty_three_runs(self):
        keys = {
            (method, dataset, seed)
            for method in METHODS
            for dataset in DATASETS
            for seed in SEEDS
        }
        self.assertEqual(len(keys), 63)

    def test_every_command_uses_the_fixed_global_interface(self):
        for method in METHODS:
            command = build_command(self.freeze, method, "clinc150", 0, "python")
            self.assertIn("--classifier_protocol", command)
            index = command.index("--classifier_protocol")
            self.assertEqual(command[index + 1], "fixed_global")

    def test_selected_configuration_is_hash_locked(self):
        raw = json.loads(
            (ROOT / "experiments/documented_dev_selection_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(raw["selection_summary"]["runs"], 189)


if __name__ == "__main__":
    unittest.main()
