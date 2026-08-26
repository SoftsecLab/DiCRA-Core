from __future__ import annotations

from src.experiment_state import ExperimentState
from src.run_config import ExperimentConfig


class ClassifierProtocolCoordinator:


    def __init__(
        self,
        *,
        config: ExperimentConfig,
        state: ExperimentState,
    ):
        self.config = config
        self.state = state
        self.seen_labels: set[int] = set()

    def activate_task(
        self,
        model,
        current_labels: set[int],
    ) -> None:
        if self.config.classifier_protocol == "dynamic_seen":
            added_labels = model.expand_classifier(current_labels)
            if added_labels:
                print(
                    "[Classifier] expanded dynamic head: "
                    f"+{len(added_labels)} rows, "
                    f"total={len(model.class_ids)}"
                )

        self.seen_labels.update(current_labels)
        expected_classifier_labels = (
            self.seen_labels
            if self.config.classifier_protocol == "dynamic_seen"
            else set(range(self.config.num_classes))
        )
        if set(model.class_ids) != expected_classifier_labels:
            raise RuntimeError(
                "Classifier class-row mapping does not match the active "
                "protocol"
            )

        self.state.seen_labels_by_stage.append(
            sorted(self.seen_labels)
        )
        self.state.classifier_class_ids_by_stage.append(
            list(model.class_ids)
        )
        self.state.classifier_output_dims_by_stage.append(
            len(model.class_ids)
        )
