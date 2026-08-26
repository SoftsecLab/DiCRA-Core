import copy
import math
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.config import build_consolidation_config
from src.consolidation import (
    consolidate_lora,
    resolve_merge_decay,
)
from src.run_config import ConsolidationConfig
from src.sleep import merge_and_reinit_lora


class FakeLoRALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(2, 2, bias=False)
        self.lora_A = nn.ModuleDict(
            {"default": nn.Linear(2, 1, bias=False)}
        )
        self.lora_B = nn.ModuleDict(
            {"default": nn.Linear(1, 2, bias=False)}
        )
        self.scaling = {"default": 0.5}
        with torch.no_grad():
            self.base_layer.weight.zero_()
            self.lora_A["default"].weight.copy_(
                torch.tensor([[1.0, 2.0]])
            )
            self.lora_B["default"].weight.copy_(
                torch.tensor([[3.0], [4.0]])
            )


class FakeLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = nn.Module()
        self.bert.adapter = FakeLoRALayer()


class NoLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = nn.Linear(2, 2, bias=False)


class ConsolidationContractTests(unittest.TestCase):
    def test_merge_matches_b_a_scaling_and_decay_exactly(self):
        model = FakeLoRAModel()
        layer = model.bert.adapter
        lora_a = layer.lora_A["default"].weight.detach().clone()
        lora_b = layer.lora_B["default"].weight.detach().clone()
        expected_base = (lora_b @ lora_a) * 0.5 * 0.25
        expected_a = torch.empty_like(lora_a)
        torch.manual_seed(123)
        nn.init.kaiming_uniform_(expected_a, a=math.sqrt(5))

        torch.manual_seed(123)
        merged_count, decay = consolidate_lora(
            model,
            ConsolidationConfig(merge_gamma=0.5),
            task_id=2,
        )

        self.assertEqual(merged_count, 1)
        self.assertEqual(decay, 0.25)
        torch.testing.assert_close(
            layer.base_layer.weight,
            expected_base,
        )
        torch.testing.assert_close(
            layer.lora_A["default"].weight,
            expected_a,
        )
        torch.testing.assert_close(
            layer.lora_B["default"].weight,
            torch.zeros_like(lora_b),
        )

    def test_max_and_affine_floor_decay_contracts_are_explicit(self):
        raw, max_floor = resolve_merge_decay(
            ConsolidationConfig(
                merge_gamma=0.5,
                merge_gamma_min=0.3,
                merge_decay_mode="max_floor",
            ),
            task_id=2,
        )
        _, affine_floor = resolve_merge_decay(
            ConsolidationConfig(
                merge_gamma=0.5,
                merge_gamma_min=0.2,
                merge_decay_mode="affine_floor",
            ),
            task_id=2,
        )

        self.assertEqual(raw, 0.25)
        self.assertEqual(max_floor, 0.3)
        self.assertEqual(affine_floor, 0.4)

    def test_fixed_seed_reinitialization_is_reproducible(self):
        first = FakeLoRAModel()
        second = copy.deepcopy(first)
        config = ConsolidationConfig(merge_gamma=0.75)

        torch.manual_seed(7)
        consolidate_lora(first, config, task_id=1)
        torch.manual_seed(7)
        consolidate_lora(second, config, task_id=1)

        for first_param, second_param in zip(
            first.parameters(),
            second.parameters(),
        ):
            torch.testing.assert_close(
                first_param,
                second_param,
            )

    def test_model_without_lora_modules_is_a_safe_noop(self):
        model = NoLoRAModel()
        before = model.bert.weight.detach().clone()

        merged_count, decay = consolidate_lora(
            model,
            ConsolidationConfig(merge_gamma=0.5),
            task_id=3,
        )

        self.assertEqual(merged_count, 0)
        self.assertEqual(decay, 0.125)
        torch.testing.assert_close(model.bert.weight, before)

    def test_sleep_facade_preserves_the_original_namespace_interface(self):
        first = FakeLoRAModel()
        second = copy.deepcopy(first)
        args = SimpleNamespace(
            merge_gamma=0.5,
            merge_gamma_min=0.3,
            merge_decay_mode="max_floor",
        )
        config = build_consolidation_config(args)

        torch.manual_seed(11)
        facade_result = merge_and_reinit_lora(
            first,
            args,
            task_id=2,
        )
        torch.manual_seed(11)
        typed_result = consolidate_lora(
            second,
            config,
            task_id=2,
        )

        self.assertEqual(facade_result, typed_result)
        for first_param, second_param in zip(
            first.parameters(),
            second.parameters(),
        ):
            torch.testing.assert_close(
                first_param,
                second_param,
            )

    def test_config_conversion_is_immutable_and_validated(self):
        config = build_consolidation_config(
            SimpleNamespace(
                merge_gamma=0.8,
                merge_gamma_min=0.1,
                merge_decay_mode="affine_floor",
            )
        )

        self.assertEqual(
            config,
            ConsolidationConfig(
                merge_gamma=0.8,
                merge_gamma_min=0.1,
                merge_decay_mode="affine_floor",
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            config.merge_gamma = 0.5
        with self.assertRaisesRegex(ValueError, "merge_gamma"):
            ConsolidationConfig(merge_gamma=-0.1)
        with self.assertRaisesRegex(ValueError, "merge_decay_mode"):
            ConsolidationConfig(merge_decay_mode="unknown")


if __name__ == "__main__":
    unittest.main()
