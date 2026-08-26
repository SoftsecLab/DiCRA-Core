import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


@unittest.skipIf(torch is None, "PyTorch is required for SLoRA numerical tests")
class SLoRANumericalTests(unittest.TestCase):
    def _model(self):
        from src.slora import SLoRAPreConsolidator

        class DummyPeftLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_layer = nn.Linear(4, 4, bias=False)
                self.lora_A = nn.ModuleDict(
                    {"default": nn.Linear(4, 2, bias=False)}
                )
                self.lora_B = nn.ModuleDict(
                    {"default": nn.Linear(2, 4, bias=False)}
                )
                self.scaling = {"default": 2.0}
                with torch.no_grad():
                    self.base_layer.weight.copy_(torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0])))
                    self.lora_A["default"].weight.copy_(
                        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
                    )
                    self.lora_B["default"].weight.copy_(
                        torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
                    )

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = nn.Module()
                self.bert.query = DummyPeftLinear()

        model = DummyModel()
        return model, SLoRAPreConsolidator

    def test_candidate_grid_is_unique_and_contains_full_rank(self):
        from src.slora import resolve_candidate_ranks

        self.assertEqual(resolve_candidate_ranks(16), (1, 3, 4, 6, 8, 9, 11, 12, 14, 16))
        with self.assertRaisesRegex(ValueError, "include the full rank"):
            resolve_candidate_ranks(16, [0.25, 0.5])

    def test_randomized_svd_does_not_advance_global_rng(self):
        from src.slora import randomized_rank_approximation

        update = torch.eye(4)
        torch.manual_seed(123)
        before = torch.random.get_rng_state().clone()
        first = randomized_rank_approximation(update, 2, seed=99)
        after = torch.random.get_rng_state().clone()
        second = randomized_rank_approximation(update, 2, seed=99)
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.allclose(first, second))

    def test_consolidation_merges_denoised_update_resets_adapter_and_audits(self):
        model, consolidator_type = self._model()
        with tempfile.TemporaryDirectory() as tmp:
            consolidator = consolidator_type(
                model,
                candidate_ratios=[0.5, 1.0],
                seed=7,
                output_dir=tmp,
            )
            reference_hash = consolidator.reference_sha256
            base_before = model.bert.query.base_layer.weight.detach().clone()
            report = consolidator.consolidate(model, task_id=0)
            audit = json.loads(
                (Path(tmp) / "slora_denoising.jsonl").read_text(encoding="utf-8")
            )

        self.assertEqual(consolidator.num_modules, 1)
        self.assertEqual(consolidator.reference_memory_bytes, 4 * 2 * 4)
        self.assertEqual(consolidator.reference_sha256, reference_hash)
        self.assertFalse(torch.equal(model.bert.query.base_layer.weight, base_before))
        self.assertTrue(
            torch.count_nonzero(model.bert.query.lora_B["default"].weight) == 0
        )
        self.assertGreater(
            torch.count_nonzero(model.bert.query.lora_A["default"].weight),
            0,
        )
        self.assertEqual(report, audit)
        self.assertIn(report["modules"][0]["chosen_rank"], {1, 2})


if __name__ == "__main__":
    unittest.main()
