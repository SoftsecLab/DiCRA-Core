import json
import tempfile
import unittest
from pathlib import Path

from src.experiment_runner import ContinualExperimentRunner
from src.run_config import AlignmentConfig, ExperimentConfig


def make_config(
    data_root,
    *,
    num_tasks=1,
    num_classes=1,
    classifier_protocol="fixed_global",
    allow_missing_tasks=False,
):
    return ExperimentConfig(
        seed=0,
        data_root=str(data_root),
        num_tasks=num_tasks,
        num_classes=num_classes,
        max_length=32,
        batch_size=4,
        eval_batch_size=8,
        classifier_protocol=classifier_protocol,
        allow_missing_tasks=allow_missing_tasks,
        analyze_stages=False,
        log_rem_diagnostics=False,
        no_consolidation=False,
        merge_gamma=0.75,
        merge_gamma_min=0.0,
        merge_decay_mode="max_floor",
        use_sleep=False,
        matched_alignment_only=False,
        update_prototypes_without_sleep=False,
        save_checkpoints=False,
        audit_imprinting_agreement=False,
        min_imprinting_agreement=0.999,
        evaluate_train_accuracy=False,
        run_reference_diagnostics=False,
        run_prototype_staleness_diagnostics=False,
        num_centroids=1,
        prototype_std_scale=0.5,
        prototype_refresh_split="dev",
        prototype_refresh_protocol="heldout_dev",
        prototype_refresh_data_root=None,
        probe_epochs=1,
        probe_lr=1e-2,
        probe_weight_decay=0.0,
        probe_batch_size=4,
        probe_eval_batch_size=8,
        probe_max_train_examples_per_class=0,
        alignment=AlignmentConfig(alignment_method="none"),
    )


class FakeModel:
    def __init__(self):
        self.class_ids = [0]

    def eval(self):
        return self


class FakeTrainer:
    def __init__(self):
        self.train_calls = []

    def train_task(self, loader, task_id, prototype_memory):
        self.train_calls.append((loader, task_id, prototype_memory))

    def evaluate_global_and_seen(self, loader, seen_labels):
        return 0.75, 0.75

    def evaluate_global_seen_and_future(self, loader, seen_labels):
        global_accuracy, seen_accuracy = self.evaluate_global_and_seen(
            loader,
            seen_labels,
        )
        return {
            "global_accuracy": global_accuracy,
            "seen_accuracy": seen_accuracy,
            "pred_future_count": 0,
            "sample_count": 1,
        }

    def evaluate_ncm(self, loader, prototype_memory):
        return 0.5

    def bind_model(self, model):
        self.bound_model = model


class FakePrototypeMemory:
    pass


def write_task(data_root):
    task_dir = Path(data_root) / "task_0"
    task_dir.mkdir(parents=True)
    row = json.dumps({"text": "example", "label": 0}) + "\n"
    (task_dir / "train.json").write_text(row, encoding="utf-8")
    (task_dir / "test.json").write_text(row, encoding="utf-8")


class ContinualExperimentRunnerContractTests(unittest.TestCase):
    def test_dynamic_classifier_expands_before_loaders_are_constructed(self):
        events = []

        class DynamicModel:
            def __init__(self):
                self.class_ids = []

            def expand_classifier(self, current_labels):
                events.append("expand")
                added = sorted(current_labels)
                self.class_ids.extend(added)
                return added

            def eval(self):
                return self

        with tempfile.TemporaryDirectory() as data_root:
            task_dir = Path(data_root) / "task_0"
            task_dir.mkdir(parents=True)
            row = json.dumps({"text": "example", "label": 1}) + "\n"
            (task_dir / "train.json").write_text(row, encoding="utf-8")
            (task_dir / "test.json").write_text(row, encoding="utf-8")

            def make_loader(dataset, batch_size, shuffle):
                events.append(f"loader:{shuffle}")
                return dataset

            runner = ContinualExperimentRunner(
                config=make_config(
                    data_root,
                    num_classes=2,
                    classifier_protocol="dynamic_seen",
                ),
                task_order=[0],
                model=DynamicModel(),
                tokenizer=None,
                trainer=FakeTrainer(),
                prototype_memory=FakePrototypeMemory(),
                make_loader=make_loader,
                output_dir=data_root,
                device="cpu",
            )

            runner.run()

        self.assertEqual(events[:3], ["expand", "loader:True", "loader:False"])

    def test_single_task_flow_returns_the_existing_matrix_contract(self):
        with tempfile.TemporaryDirectory() as data_root:
            write_task(data_root)
            trainer = FakeTrainer()
            memory = FakePrototypeMemory()
            runner = ContinualExperimentRunner(
                config=make_config(data_root),
                task_order=[0],
                model=FakeModel(),
                tokenizer=None,
                trainer=trainer,
                prototype_memory=memory,
                make_loader=lambda dataset, batch_size, shuffle: dataset,
                output_dir=data_root,
                device="cpu",
            )

            state = runner.run()

        self.assertEqual(len(trainer.train_calls), 1)
        self.assertIs(trainer.train_calls[0][2], memory)
        self.assertEqual(state.test_matrix.tolist(), [[0.75]])
        self.assertEqual(state.seen_matrix.tolist(), [[0.75]])
        self.assertEqual(state.ncm_matrix.tolist(), [[0.5]])
        self.assertEqual(state.train_matrix.tolist(), [[0.0]])
        self.assertEqual(state.seen_labels_by_stage, [[0]])
        self.assertEqual(state.classifier_class_ids_by_stage, [[0]])
        self.assertEqual(state.classifier_output_dims_by_stage, [1])

    def test_optional_task_boundary_transform_runs_before_evaluation(self):
        events = []

        class OrderedTrainer(FakeTrainer):
            def train_task(self, loader, task_id, prototype_memory):
                events.append("train")

            def bind_model(self, model):
                events.append("bind")

            def evaluate_global_and_seen(self, loader, seen_labels):
                events.append("evaluate")
                return 0.75, 0.75

        with tempfile.TemporaryDirectory() as data_root:
            write_task(data_root)
            runner = ContinualExperimentRunner(
                config=make_config(data_root),
                task_order=[0],
                model=FakeModel(),
                tokenizer=None,
                trainer=OrderedTrainer(),
                prototype_memory=FakePrototypeMemory(),
                make_loader=lambda dataset, batch_size, shuffle: dataset,
                output_dir=data_root,
                device="cpu",
                task_boundary_transform=(
                    lambda model, task_id: events.append("transform")
                ),
            )
            runner.run()

        self.assertLess(events.index("train"), events.index("transform"))
        self.assertLess(events.index("transform"), events.index("evaluate"))
        self.assertIn("bind", events)

    def test_missing_task_policy_remains_explicit(self):
        with tempfile.TemporaryDirectory() as data_root:
            runner = ContinualExperimentRunner(
                config=make_config(data_root),
                task_order=[0],
                model=FakeModel(),
                tokenizer=None,
                trainer=FakeTrainer(),
                prototype_memory=FakePrototypeMemory(),
                make_loader=lambda dataset, batch_size, shuffle: dataset,
                output_dir=data_root,
                device="cpu",
            )
            with self.assertRaisesRegex(FileNotFoundError, "allow_missing_tasks"):
                runner.run()

    def test_task_order_must_match_the_typed_config(self):
        with tempfile.TemporaryDirectory() as data_root:
            with self.assertRaisesRegex(ValueError, "task_order length"):
                ContinualExperimentRunner(
                    config=make_config(data_root, num_tasks=1),
                    task_order=[],
                    model=FakeModel(),
                    tokenizer=None,
                    trainer=FakeTrainer(),
                    prototype_memory=FakePrototypeMemory(),
                    make_loader=lambda dataset, batch_size, shuffle: dataset,
                    output_dir=data_root,
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
