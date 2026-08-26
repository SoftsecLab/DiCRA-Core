from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


class RECAPEvaluator:


    def __init__(self, model, device, autocast_context: Callable):
        self.model = model
        self.device = device
        self._autocast_context = autocast_context

    def bind_model(self, model) -> None:


        self.model = model

    def inference_context(self):


        return self._autocast_context()

    @staticmethod
    def predict_from_logits(
        logits,
        candidate_labels=None,
        classifier_class_ids=None,
    ):


        if classifier_class_ids is None:
            classifier_class_ids = list(range(logits.shape[1]))
        classifier_class_ids = [
            int(label) for label in classifier_class_ids
        ]
        if len(classifier_class_ids) != logits.shape[1]:
            raise ValueError(
                "classifier_class_ids must match the number of logit columns"
            )
        class_to_row = {
            label: row for row, label in enumerate(classifier_class_ids)
        }
        class_id_tensor = torch.as_tensor(
            classifier_class_ids,
            device=logits.device,
            dtype=torch.long,
        )
        if candidate_labels is None:
            return class_id_tensor[torch.argmax(logits, dim=1)]

        labels = torch.as_tensor(
            candidate_labels,
            device=logits.device,
            dtype=torch.long,
        )
        if labels.ndim != 1 or labels.numel() == 0:
            raise ValueError(
                "candidate_labels must be a non-empty 1D sequence"
            )
        if labels.unique().numel() != labels.numel():
            raise ValueError("candidate_labels must not contain duplicates")

        try:
            rows = torch.as_tensor(
                [class_to_row[int(label)] for label in labels.tolist()],
                device=logits.device,
                dtype=torch.long,
            )
        except KeyError as exc:
            raise ValueError(
                f"candidate class {int(exc.args[0])} is not present in "
                "the classifier"
            ) from exc
        candidate_logits = logits.index_select(1, rows)
        return labels[torch.argmax(candidate_logits, dim=1)]

    def evaluate(self, loader):
        self.model.eval()
        preds, gts = [], []

        with torch.no_grad():
            for batch in loader:
                with self.inference_context():
                    logits = self.model(
                        batch["input_ids"].to(self.device),
                        batch["attention_mask"].to(self.device),
                    )
                preds.extend(
                    self.predict_from_logits(
                        logits,
                        classifier_class_ids=self.model.class_ids,
                    )
                    .cpu()
                    .numpy()
                )
                gts.extend(batch["labels"].cpu().numpy())

        return accuracy_score(gts, preds)

    def evaluate_global_and_seen(self, loader, seen_labels):


        report = self.evaluate_global_seen_and_future(loader, seen_labels)
        return report["global_accuracy"], report["seen_accuracy"]

    def evaluate_global_seen_and_future(self, loader, seen_labels):


        self.model.eval()
        global_preds, seen_preds, gts = [], [], []
        seen_label_set = {int(label) for label in seen_labels}
        if not seen_label_set:
            raise ValueError("seen_labels must not be empty")
        pred_future_count = 0

        with torch.no_grad():
            for batch in loader:
                with self.inference_context():
                    logits = self.model(
                        batch["input_ids"].to(self.device),
                        batch["attention_mask"].to(self.device),
                    )
                global_batch = self.predict_from_logits(
                    logits,
                    classifier_class_ids=self.model.class_ids,
                )
                seen_batch = self.predict_from_logits(
                    logits,
                    seen_labels,
                    classifier_class_ids=self.model.class_ids,
                )
                global_values = global_batch.cpu().tolist()
                global_preds.extend(global_values)
                seen_preds.extend(seen_batch.cpu().tolist())
                pred_future_count += sum(
                    int(prediction not in seen_label_set)
                    for prediction in global_values
                )
                gts.extend(batch["labels"].cpu().numpy())

        sample_count = len(gts)
        return {
            "global_accuracy": accuracy_score(gts, global_preds),
            "seen_accuracy": accuracy_score(gts, seen_preds),
            "pred_future_count": pred_future_count,
            "sample_count": sample_count,
            "pred_future_rate": (
                pred_future_count / sample_count if sample_count else 0.0
            ),
        }

    def evaluate_ncm(self, loader, prototype_memory):


        self.model.eval()
        if not prototype_memory.prototypes:
            return 0.0

        proto_labels = sorted(prototype_memory.prototypes.keys())
        center_chunks = []
        center_slices = []
        start = 0
        for label in proto_labels:
            proto = prototype_memory.prototypes[label]
            centers = proto.get("means")
            if centers is None:
                centers = proto["mean"].unsqueeze(0)
            centers = centers.to(self.device)
            end = start + centers.size(0)
            center_chunks.append(centers)
            center_slices.append((start, end))
            start = end

        proto_centers = F.normalize(
            torch.cat(center_chunks, dim=0),
            dim=1,
            eps=1e-8,
        )
        proto_label_tensor = torch.tensor(
            proto_labels,
            device=self.device,
        )

        preds, gts = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                with self.inference_context():
                    feats = self.model.get_features(
                        input_ids,
                        attention_mask,
                    )
                feats = F.normalize(feats, dim=1, eps=1e-8)
                center_sim = (
                    feats
                    @ proto_centers.to(dtype=feats.dtype).t()
                )
                class_sim = torch.stack(
                    [
                        center_sim[:, start:end].max(dim=1).values
                        for start, end in center_slices
                    ],
                    dim=1,
                )
                ncm_preds = proto_label_tensor[class_sim.argmax(dim=1)]
                preds.extend(ncm_preds.cpu().numpy())
                gts.extend(batch["labels"].cpu().numpy())

        return accuracy_score(gts, preds)

    def evaluate_classifier_ncm_agreement(
        self,
        loader,
        prototype_memory,
        seen_labels,
    ):


        self.model.eval()
        labels = sorted(int(label) for label in seen_labels)
        if not labels:
            return {"agreement": 0.0, "samples": 0}
        centers = torch.stack(
            [prototype_memory.class_mean(label) for label in labels],
            dim=0,
        ).to(self.device)
        centers = F.normalize(centers, dim=1, eps=1e-8)
        label_tensor = torch.tensor(
            labels,
            device=self.device,
            dtype=torch.long,
        )
        agreed = 0
        samples = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                with self.inference_context():
                    features = self.model.get_features(
                        input_ids,
                        attention_mask,
                    )
                    logits = self.model.classifier(features)
                classifier_preds = self.predict_from_logits(
                    logits,
                    candidate_labels=labels,
                    classifier_class_ids=self.model.class_ids,
                )
                normalized = F.normalize(features, dim=1, eps=1e-8)
                ncm_indices = (
                    normalized
                    @ centers.to(dtype=normalized.dtype).t()
                ).argmax(dim=1)
                ncm_preds = label_tensor[ncm_indices]
                agreed += int(
                    (classifier_preds == ncm_preds).sum().item()
                )
                samples += int(classifier_preds.numel())

        return {
            "agreement": agreed / samples if samples else 0.0,
            "samples": samples,
        }
