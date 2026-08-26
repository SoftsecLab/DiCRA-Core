import unittest

import torch
import torch.nn as nn

from src.clora_regularizer import CLoRARegularizer


class FakeLoRALinear(nn.Module):
    def __init__(self, input_dim=5, output_dim=4, rank=2):
        super().__init__()
        self.lora_A = nn.ModuleDict(
            {"default": nn.Linear(input_dim, rank, bias=False)}
        )
        self.lora_B = nn.ModuleDict(
            {"default": nn.Linear(rank, output_dim, bias=False)}
        )


class FakePEFTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = FakeLoRALinear()
        self.value = FakeLoRALinear()


class CLoRARegularizerTests(unittest.TestCase):
    def test_matches_official_peft_tensor_formula(self):
        model = FakePEFTModel()
        regularizer = CLoRARegularizer(model, k=3, seed=7)

        expected = torch.zeros(())
        for record in regularizer._records:
            p_input = getattr(regularizer, record.p_input_name)
            p_output = getattr(regularizer, record.p_output_name)
            expected = expected + 0.5 * (
                (record.lora_a @ p_input).square().sum()
                + (record.lora_b.T @ p_output).square().sum()
            )
        torch.testing.assert_close(regularizer(), expected)

    def test_subspaces_are_frozen_and_hash_stable_after_optimizer_step(self):
        model = FakePEFTModel()
        regularizer = CLoRARegularizer(model, k=3, seed=7)
        initial_hash = regularizer.subspace_sha256()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        optimizer.zero_grad()
        regularizer().backward()
        optimizer.step()

        self.assertTrue(all(not buffer.requires_grad for buffer in regularizer.buffers()))
        self.assertEqual(initial_hash, regularizer.subspace_sha256())
        self.assertTrue(regularizer.audit_report()["subspaces_unchanged"])

    def test_initialization_does_not_advance_training_rng(self):
        model = FakePEFTModel()
        torch.manual_seed(123)
        expected = torch.rand(4)

        torch.manual_seed(123)
        CLoRARegularizer(model, k=3, seed=7)
        actual = torch.rand(4)

        torch.testing.assert_close(actual, expected)

    def test_rejects_wrong_target_scope(self):
        model = nn.Module()
        model.key = FakeLoRALinear()
        with self.assertRaisesRegex(ValueError, "target-module mismatch"):
            CLoRARegularizer(model, k=3, seed=7)


if __name__ == "__main__":
    unittest.main()
