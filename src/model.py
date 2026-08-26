import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModel

from src.model_paths import get_bert_path as _resolve_model_path


def get_bert_path(model_id="bert-base-uncased"):
    return _resolve_model_path(model_id)


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.tensor(30.0))
        else:
            self.register_buffer("sigma", torch.tensor(30.0))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.in_features ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        out_norm = F.normalize(input, p=2, dim=1, eps=1e-8)
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-8)
        return self.sigma * F.linear(out_norm, w_norm)


class ClassifierLabelMixin:
    """Map dataset-global labels to classifier rows and grow the head safely."""

    def initialize_classifier_labels(self, class_ids=None):
        out_features = int(self.classifier.weight.shape[0])
        if class_ids is None:
            class_ids = list(range(out_features))
        class_ids = [int(label) for label in class_ids]
        if len(class_ids) != out_features:
            raise ValueError(
                f"classifier has {out_features} rows but received {len(class_ids)} class IDs"
            )
        if len(set(class_ids)) != len(class_ids):
            raise ValueError("classifier class IDs must be unique")
        if any(label < 0 for label in class_ids):
            raise ValueError("classifier class IDs must be non-negative")
        self.class_ids = class_ids
        self._class_to_row = {label: row for row, label in enumerate(class_ids)}
        self._identity_class_ids = class_ids == list(range(out_features))
        self._class_index_cache = {}

    def _class_index_tensor(self, device):
        key = str(device)
        lookup = self._class_index_cache.get(key)
        if lookup is None:
            lookup = torch.full(
                (max(self.class_ids) + 1,),
                -1,
                dtype=torch.long,
                device=device,
            )
            global_ids = torch.tensor(self.class_ids, dtype=torch.long, device=device)
            lookup[global_ids] = torch.arange(
                len(self.class_ids), dtype=torch.long, device=device
            )
            self._class_index_cache[key] = lookup
        return lookup

    def classifier_rows_for_global_labels(self, labels, device=None):
        if torch.is_tensor(labels):
            if device is None:
                device = labels.device
            label_tensor = labels.to(device=device, dtype=torch.long)
        else:
            label_tensor = torch.tensor(list(labels), dtype=torch.long, device=device)
        if label_tensor.numel() == 0:
            return label_tensor
        lookup = self._class_index_tensor(label_tensor.device)
        if label_tensor.min().item() < 0 or label_tensor.max().item() >= lookup.numel():
            raise ValueError("class is not present in the classifier")
        rows = lookup[label_tensor]
        if (rows < 0).any().item():
            raise ValueError("class is not present in the classifier")
        return rows

    def global_to_classifier_labels(self, labels):
        if self._identity_class_ids:
            return labels
        lookup = self._class_index_tensor(labels.device)
        return lookup[labels]

    def classifier_to_global_labels(self, rows):
        if self._identity_class_ids:
            return rows
        class_ids = torch.tensor(self.class_ids, dtype=torch.long, device=rows.device)
        return class_ids[rows]

    def expand_classifier(self, new_class_ids):
        new_class_ids = sorted(
            {int(label) for label in new_class_ids if int(label) not in self._class_to_row}
        )
        if not new_class_ids:
            return []

        old_classifier = self.classifier
        old_rows = int(old_classifier.weight.shape[0])
        new_rows = old_rows + len(new_class_ids)
        device = old_classifier.weight.device
        dtype = old_classifier.weight.dtype

        if isinstance(old_classifier, CosineLinear):
            sigma_trainable = isinstance(old_classifier.sigma, nn.Parameter)
            new_classifier = CosineLinear(
                old_classifier.in_features,
                new_rows,
                sigma=sigma_trainable,
            ).to(device=device, dtype=dtype)
            with torch.no_grad():
                new_classifier.weight[:old_rows].copy_(old_classifier.weight)
                new_classifier.sigma.copy_(old_classifier.sigma)
            new_classifier.weight.requires_grad_(old_classifier.weight.requires_grad)
            if isinstance(new_classifier.sigma, nn.Parameter):
                new_classifier.sigma.requires_grad_(old_classifier.sigma.requires_grad)
        elif isinstance(old_classifier, nn.Linear):
            new_classifier = nn.Linear(
                old_classifier.in_features,
                new_rows,
                bias=old_classifier.bias is not None,
            ).to(device=device, dtype=dtype)
            with torch.no_grad():
                new_classifier.weight[:old_rows].copy_(old_classifier.weight)
                if old_classifier.bias is not None:
                    new_classifier.bias[:old_rows].copy_(old_classifier.bias)
            new_classifier.weight.requires_grad_(old_classifier.weight.requires_grad)
            if old_classifier.bias is not None:
                new_classifier.bias.requires_grad_(old_classifier.bias.requires_grad)
        else:
            raise TypeError(
                "dynamic classifier expansion supports CosineLinear and nn.Linear only"
            )

        new_classifier.train(old_classifier.training)
        self.classifier = new_classifier
        self.initialize_classifier_labels(self.class_ids + new_class_ids)
        return new_class_ids


class RECAPBertClassifier(ClassifierLabelMixin, nn.Module):
    def __init__(
        self,
        bert_path,
        num_classes,
        use_lora=True,
        use_cosine=True,
        lora_rank=16,
        gradient_checkpointing=False,
        class_ids=None,
    ):
        super().__init__()
        print(f"[Model] Loading backbone: {bert_path}")
        self.config = AutoConfig.from_pretrained(bert_path)
        self.bert = AutoModel.from_pretrained(bert_path, config=self.config)

        if use_lora:
            print(f"[Model] Applying LoRA: rank={lora_rank}")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                lora_dropout=0.1,
            )
            self.bert = get_peft_model(self.bert, peft_config)
            self.bert.print_trainable_parameters()

        if gradient_checkpointing and hasattr(self.bert, "gradient_checkpointing_enable"):
            print("[Model] Enabling gradient checkpointing.")
            self.bert.gradient_checkpointing_enable()
            if hasattr(self.bert, "enable_input_require_grads"):
                self.bert.enable_input_require_grads()

        if not use_cosine:
            print("[Model] Ablation: using nn.Linear classifier.")
            self.classifier = nn.Linear(self.config.hidden_size, num_classes, bias=False)
        else:
            print("[Model] Using CosineLinear classifier.")
            self.classifier = CosineLinear(self.config.hidden_size, num_classes)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.initialize_classifier_labels(class_ids)

    def get_features(self, input_ids, attention_mask):
        """Extract CLS token features before dropout/classification."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, dict):
            return outputs["last_hidden_state"][:, 0, :]
        return outputs[0][:, 0, :]

    def forward(self, input_ids, attention_mask):
        pooled_output = self.get_features(input_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)
