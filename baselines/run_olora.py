import argparse
import gc
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import JSONLBatchCollator, JSONLDataset
from src.evaluation import summarize_evaluation_masking, summarize_final_results
from src.memory import PrototypeMemory
from src.model import ClassifierLabelMixin, CosineLinear, get_bert_path


def load_task_labels(path):
    labels = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.add(int(json.loads(line)["label"]))
    return labels


def task_path(data_root, task_id, split):
    return os.path.join(data_root, f"task_{task_id}", f"{split}.json")


class OfficialishOLoRALinear(nn.Module):


    def __init__(self, base_linear, rank=16, alpha=None, dropout=0.1):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("OfficialishOLoRALinear can only wrap nn.Linear.")

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank * 2)
        self.scaling = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)
        if base_linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)

        self.register_buffer("old_A", torch.empty(0, self.in_features))
        self.register_buffer("old_B", torch.empty(self.out_features, 0))

        self.new_A = nn.Linear(self.in_features, self.rank, bias=False)
        self.new_B = nn.Linear(self.rank, self.out_features, bias=False)
        self.reset_new_lora()

    @property
    def old_rank(self):
        return int(self.old_A.shape[0])

    def reset_new_lora(self):
        nn.init.kaiming_uniform_(self.new_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.new_B.weight)

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        dropped = self.dropout(x)

        if self.old_rank > 0:
            old_hidden = F.linear(dropped, self.old_A)
            result = result + F.linear(old_hidden, self.old_B) * self.scaling

        new_hidden = self.new_A(dropped)
        result = result + self.new_B(new_hidden) * self.scaling
        return result

    def orthogonal_loss(self):
        if self.old_rank == 0:
            return self.weight.new_tensor(0.0)
        return torch.abs(self.new_A.weight @ self.old_A.T).sum()

    def l2_loss(self):
        return torch.norm(self.new_A.weight, p=2) + torch.norm(self.new_B.weight, p=2)

    def accumulate_new_lora(self):
        new_A = self.new_A.weight.detach()
        new_B = self.new_B.weight.detach()
        self.old_A = torch.cat([self.old_A.to(new_A.device), new_A], dim=0)
        self.old_B = torch.cat([self.old_B.to(new_B.device), new_B], dim=1)
        self.reset_new_lora()


def module_name_matches(name, target_modules):
    leaf = name.split(".")[-1]
    return any(leaf == target or name.endswith(target) for target in target_modules)


def inject_officialish_olora(module, target_modules, rank, alpha, dropout):
    replaced = []
    for child_name, child in list(module.named_children()):
        full_child_name = child_name
        if isinstance(child, nn.Linear) and module_name_matches(full_child_name, target_modules):
            setattr(
                module,
                child_name,
                OfficialishOLoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
            )
            replaced.append(full_child_name)
        else:
            child_replaced = inject_officialish_olora(
                child, target_modules, rank=rank, alpha=alpha, dropout=dropout
            )
            replaced.extend(f"{child_name}.{name}" for name in child_replaced)
    return replaced


class OLoRABertClassifier(ClassifierLabelMixin, nn.Module):
    def __init__(
        self,
        model_path,
        num_classes,
        target_modules,
        lora_rank=16,
        lora_alpha=None,
        lora_dropout=0.1,
        use_cosine=True,
        class_ids=None,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        self.bert = AutoModel.from_pretrained(model_path, config=self.config)

        for param in self.bert.parameters():
            param.requires_grad = False

        replaced = inject_officialish_olora(
            self.bert,
            target_modules=target_modules,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        if not replaced:
            raise ValueError(
                f"No target modules matched {target_modules}. "
                "Try --target_modules query value dense."
            )

        self.olora_layers = [
            module for module in self.modules() if isinstance(module, OfficialishOLoRALinear)
        ]
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        if use_cosine:
            self.classifier = CosineLinear(self.config.hidden_size, num_classes)
        else:
            self.classifier = nn.Linear(self.config.hidden_size, num_classes, bias=False)
        self.initialize_classifier_labels(class_ids)

        print(f"[O-LoRA] Injected official-ish O-LoRA into {len(replaced)} linear modules.")
        print(f"[O-LoRA] Target modules: {target_modules}")

    def get_features(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, dict):
            return outputs["last_hidden_state"][:, 0, :]
        return outputs[0][:, 0, :]

    def forward(self, input_ids, attention_mask):
        features = self.dropout(self.get_features(input_ids, attention_mask))
        return self.classifier(features)

    def orthogonal_loss(self):
        losses = [layer.orthogonal_loss() for layer in self.olora_layers]
        return torch.stack(losses).sum() if losses else next(self.parameters()).new_tensor(0.0)

    def l2_loss(self):
        losses = [layer.l2_loss() for layer in self.olora_layers]
        return torch.stack(losses).sum() if losses else next(self.parameters()).new_tensor(0.0)

    def accumulate_new_lora(self):
        for layer in self.olora_layers:
            layer.accumulate_new_lora()

    def old_rank(self):
        return self.olora_layers[0].old_rank if self.olora_layers else 0


def print_trainable_parameter_summary(model):
    total = 0
    trainable = 0
    trainable_lora = 0
    trainable_classifier = 0

    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if not param.requires_grad:
            continue
        trainable += count
        if ".new_A." in name or ".new_B." in name:
            trainable_lora += count
        if "classifier" in name:
            trainable_classifier += count

    pct = 100.0 * trainable / max(total, 1)
    print(
        "[O-LoRA][Sanity] "
        f"trainable={trainable:,}/{total:,} ({pct:.4f}%), "
        f"new_lora={trainable_lora:,}, classifier={trainable_classifier:,}"
    )


class OLoRATrainer:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.current_labels = set()
        self.seen_labels = set()

    def set_label_context(self, current_labels, seen_labels):
        self.current_labels = set(current_labels)
        self.seen_labels = set(seen_labels)

    def _register_classifier_gradient_mask(self):
        if not self.args.freeze_old_classifier_rows:
            return None
        if not hasattr(self.model.classifier, "weight"):
            return None

        trainable_rows = torch.zeros(
            self.model.classifier.weight.shape[0],
            dtype=torch.float32,
            device=self.device,
        )
        for label in self.current_labels:
            row = self.model._class_to_row[int(label)]
            trainable_rows[row] = 1.0
        mask = trainable_rows[:, None]

        def hook(grad):
            return grad * mask

        return self.model.classifier.weight.register_hook(hook)

    def train_epoch(self, loader, optimizer, epoch_idx):
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_orth = 0.0
        total_l2 = 0.0
        seen_label_tensor = torch.tensor(
            sorted(self.seen_labels), dtype=torch.long, device=self.device
        )

        pbar = tqdm(loader, desc=f"  Epoch {epoch_idx + 1}/{self.args.epochs}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            if self.args.classifier_protocol == "dynamic_seen":
                ce_logits = logits
                ce_labels = self.model.global_to_classifier_labels(labels)
            elif self.args.global_interface:
                ce_logits = logits
                ce_labels = labels
            else:
                ce_logits = logits.new_full(logits.shape, float("-inf"))
                ce_logits[:, seen_label_tensor] = logits[:, seen_label_tensor]
                ce_labels = labels

            ce_loss = self.criterion(ce_logits, ce_labels)
            orth_loss = self.model.orthogonal_loss()
            l2_loss = self.model.l2_loss()
            loss = ce_loss + self.args.olora_lambda1 * orth_loss + self.args.olora_lambda2 * l2_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_ce += float(ce_loss.item())
            total_orth += float(orth_loss.item())
            total_l2 += float(l2_loss.item())
            pbar.set_postfix(
                {
                    "CE": f"{ce_loss.item():.3f}",
                    "Orth": f"{orth_loss.item():.3f}",
                }
            )

        n = max(len(loader), 1)
        print(
            f"    CE={total_ce / n:.4f} | Orth={total_orth / n:.4f} | "
            f"L2={total_l2 / n:.4f} | Total={total_loss / n:.4f}"
        )

    def train_task(self, loader, task_id):
        handle = self._register_classifier_gradient_mask()
        optimizer = torch.optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=self.args.lr,
            weight_decay=0.0,
        )

        print(
            f"[O-LoRA] Task {task_id} training: epochs={self.args.epochs}, "
            f"old_rank={self.model.old_rank()}, new_rank={self.args.lora_rank}, "
            f"lambda1={self.args.olora_lambda1}, lambda2={self.args.olora_lambda2}"
        )
        for epoch in range(self.args.epochs):
            self.train_epoch(loader, optimizer, epoch)

        if handle is not None:
            handle.remove()

        if not self.args.no_accumulate:
            self.model.accumulate_new_lora()
            print(
                f"[O-LoRA] Accumulated new LoRA into frozen old subspace. "
                f"old_rank={self.model.old_rank()}"
            )

        torch.cuda.empty_cache()
        gc.collect()

    def evaluate(self, loader, seen_labels=None):
        self.model.eval()
        preds = []
        gts = []
        seen_row_tensor = None
        seen_label_tensor = None
        if seen_labels is not None:
            seen_label_tensor = torch.tensor(
                sorted(seen_labels), dtype=torch.long, device=self.device
            )
            seen_row_tensor = self.model.classifier_rows_for_global_labels(
                seen_label_tensor,
                device=self.device,
            )

        with torch.no_grad():
            for batch in loader:
                logits = self.model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                if seen_label_tensor is not None:
                    candidate_logits = logits.index_select(1, seen_row_tensor)
                    pred_cols = candidate_logits.argmax(dim=1)
                    batch_preds = seen_label_tensor[pred_cols]
                else:
                    local_preds = torch.argmax(logits, dim=1)
                    batch_preds = self.model.classifier_to_global_labels(local_preds)
                preds.extend(batch_preds.cpu().tolist())
                gts.extend(batch["labels"].cpu().tolist())

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
        seen_label_tensor = torch.tensor(
            sorted(seen_labels), dtype=torch.long, device=self.device
        )
        seen_row_tensor = self.model.classifier_rows_for_global_labels(
            seen_label_tensor,
            device=self.device,
        )

        with torch.no_grad():
            for batch in loader:
                logits = self.model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                global_local = torch.argmax(logits, dim=1)
                global_batch = self.model.classifier_to_global_labels(global_local)
                seen_cols = logits.index_select(1, seen_row_tensor).argmax(dim=1)
                seen_batch = seen_label_tensor[seen_cols]
                global_preds.extend(global_batch.cpu().tolist())
                seen_preds.extend(seen_batch.cpu().tolist())
                pred_future_count += sum(
                    int(prediction not in seen_label_set)
                    for prediction in global_batch.cpu().tolist()
                )
                gts.extend(batch["labels"].cpu().tolist())

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

        proto_labels = sorted(prototype_memory.prototypes)
        centers = torch.stack(
            [prototype_memory.prototypes[label]["mean"] for label in proto_labels]
        ).to(self.device)
        centers = F.normalize(centers, dim=1, eps=1e-8)
        label_tensor = torch.tensor(proto_labels, device=self.device)
        preds, gts = [], []

        with torch.no_grad():
            for batch in loader:
                feats = self.model.get_features(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                feats = F.normalize(feats, dim=1, eps=1e-8)
                pred_indices = (feats @ centers.to(dtype=feats.dtype).t()).argmax(dim=1)
                preds.extend(label_tensor[pred_indices].cpu().tolist())
                gts.extend(batch["labels"].cpu().tolist())

        return accuracy_score(gts, preds)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Official-ish O-LoRA baseline for class-incremental text classification"
    )
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/clinc150")
    parser.add_argument("--model_id", type=str, default="bert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_tasks", type=int, default=15)
    parser.add_argument("--num_classes", type=int, default=150)
    parser.add_argument(
        "--classifier_protocol",
        choices=["fixed_global", "dynamic_seen"],
        default="fixed_global",
        help="Use a fixed full head or a true head that expands as classes arrive.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", nargs="+", default=["query", "value"])
    parser.add_argument("--olora_lambda1", type=float, default=0.5)
    parser.add_argument("--olora_lambda2", type=float, default=0.0)
    parser.add_argument("--no_cosine", action="store_true")
    parser.add_argument(
        "--no_accumulate",
        action="store_true",
        help="Disable official-style loranew -> lora accumulation; for sanity only.",
    )
    parser.add_argument(
        "--eval_all_classes",
        action="store_true",
        help="Evaluate all classifier outputs instead of seen-so-far classes.",
    )
    parser.add_argument(
        "--global_interface",
        action="store_true",
        help=(
            "Strict same-interface protocol: full global CE, fixed full classifier, "
            "no future-class masking, and all classifier rows trainable."
        ),
    )
    parser.add_argument(
        "--freeze_old_classifier_rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only update current-task classifier rows during training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.global_interface:
        if args.no_cosine:
            raise ValueError("--global_interface requires CosineLinear; remove --no_cosine.")
        if set(args.target_modules) != {"query", "value"}:
            raise ValueError(
                "--global_interface requires --target_modules query value to match RECAP."
            )
        args.eval_all_classes = True
        args.freeze_old_classifier_rows = False
    if args.classifier_protocol == "dynamic_seen":
        if args.global_interface:
            raise ValueError(
                "--classifier_protocol dynamic_seen is incompatible with --global_interface"
            )
        if args.no_cosine:
            raise ValueError("dynamic_seen requires CosineLinear; remove --no_cosine")
        if set(args.target_modules) != {"query", "value"}:
            raise ValueError(
                "dynamic_seen requires --target_modules query value to match RECAP"
            )
        args.eval_all_classes = False
        args.freeze_old_classifier_rows = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = get_bert_path(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    collator = JSONLBatchCollator(tokenizer, max_len=args.max_length)
    loader_kwargs = {"num_workers": args.num_workers}

    if args.classifier_protocol == "dynamic_seen":
        first_train_path = task_path(args.data_root, 0, "train")
        if not os.path.exists(first_train_path):
            raise FileNotFoundError(f"[O-LoRA] Missing train data: {first_train_path}")
        initial_class_ids = sorted(load_task_labels(first_train_path))
    else:
        initial_class_ids = list(range(args.num_classes))

    model = OLoRABertClassifier(
        model_path=model_path,
        num_classes=len(initial_class_ids),
        target_modules=args.target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_cosine=not args.no_cosine,
        class_ids=initial_class_ids,
    ).to(device)
    print_trainable_parameter_summary(model)
    print(
        "[O-LoRA][Protocol] officialish_accumulation="
        f"{not args.no_accumulate}, eval="
        f"{'all-classes' if args.eval_all_classes else 'seen-so-far'}, "
        f"freeze_old_classifier_rows={args.freeze_old_classifier_rows}, "
        f"global_interface={args.global_interface}, "
        f"classifier_protocol={args.classifier_protocol}"
    )

    trainer = OLoRATrainer(model, device, args)
    output_dir = os.path.join("outputs", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        config_payload = vars(args).copy()
        protocol_contract = {
            "backbone": args.model_id,
            "tokenizer": args.model_id,
            "classifier_type": "CosineLinear" if not args.no_cosine else "Linear",
            "fixed_global_num_classes": (
                args.num_classes if args.classifier_protocol == "fixed_global" else None
            ),
            "classifier_protocol": args.classifier_protocol,
            "classifier_output_policy": (
                "expand_on_class_arrival"
                if args.classifier_protocol == "dynamic_seen"
                else "fixed_full_dataset"
            ),
            "lora_rank": args.lora_rank,
            "effective_lora_alpha": (
                args.lora_alpha if args.lora_alpha is not None else args.lora_rank * 2
            ),
            "lora_target_modules": list(args.target_modules),
            "prediction_candidate_space": (
                "dynamic_seen"
                if args.classifier_protocol == "dynamic_seen"
                else (
                    "fixed_global_unmasked"
                    if args.global_interface
                    else ("all_classes" if args.eval_all_classes else "seen_so_far")
                )
            ),
            "training_candidate_space": (
                "dynamic_seen"
                if args.classifier_protocol == "dynamic_seen"
                else ("fixed_global_unmasked" if args.global_interface else "seen_so_far")
            ),
            "historical_real_text_storage": False,
            "sample_task_id_at_inference": False,
            "future_class_rows_present": args.classifier_protocol == "fixed_global",
        }
        config_payload.update(
            {
                "protocol_contract": protocol_contract,
                "prediction_candidate_space": protocol_contract[
                    "prediction_candidate_space"
                ],
                "training_candidate_space": protocol_contract[
                    "training_candidate_space"
                ],
                "historical_real_text_storage": False,
                "sample_task_id_at_inference": False,
                "diagnostic_only_prototype_memory": bool(
                    args.global_interface or args.classifier_protocol == "dynamic_seen"
                ),
                "prototype_memory_used_for_training": False,
                "prototype_memory_used_for_inference": False,
                "task_order": list(range(args.num_tasks)),
            }
        )
        json.dump(config_payload, f, indent=2)

    R = np.zeros((args.num_tasks, args.num_tasks))
    R_seen = np.zeros((args.num_tasks, args.num_tasks))
    R_ncm = np.zeros((args.num_tasks, args.num_tasks))
    R_pred_future_count = np.zeros(
        (args.num_tasks, args.num_tasks), dtype=np.int64
    )
    R_eval_sample_count = np.zeros(
        (args.num_tasks, args.num_tasks), dtype=np.int64
    )
    seen_labels = set()
    seen_labels_by_stage = []
    classifier_class_ids_by_stage = []
    classifier_output_dims_by_stage = []
    diagnostic_interface = bool(
        args.global_interface or args.classifier_protocol == "dynamic_seen"
    )
    prototype_memory = (
        PrototypeMemory(args.num_classes, model.config.hidden_size, device, num_centroids=1)
        if diagnostic_interface
        else None
    )

    for task_id in range(args.num_tasks):
        print(f"\n{'=' * 20} Task {task_id} {'=' * 20}")
        train_path = task_path(args.data_root, task_id, "train")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"[O-LoRA] Missing train data: {train_path}")

        current_labels = load_task_labels(train_path)
        if args.classifier_protocol == "dynamic_seen":
            added_labels = model.expand_classifier(current_labels)
            if added_labels:
                print(
                    "[O-LoRA][Classifier] expanded dynamic head: "
                    f"+{len(added_labels)} rows, total={len(model.class_ids)}"
                )
        seen_labels.update(current_labels)
        if args.classifier_protocol == "dynamic_seen" and set(model.class_ids) != seen_labels:
            raise RuntimeError("O-LoRA dynamic classifier mapping does not match seen labels")
        seen_labels_by_stage.append(sorted(seen_labels))
        classifier_class_ids_by_stage.append(list(model.class_ids))
        classifier_output_dims_by_stage.append(len(model.class_ids))
        trainer.set_label_context(current_labels=current_labels, seen_labels=seen_labels)
        print(
            f"[O-LoRA][Sanity] current_labels={len(current_labels)}, "
            f"seen_labels={len(seen_labels)}"
        )

        train_loader = DataLoader(
            JSONLDataset(train_path, max_len=args.max_length, encode_on_getitem=False),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collator,
            **loader_kwargs,
        )
        trainer.train_task(train_loader, task_id)

        if prototype_memory is not None:
            prototype_loader = DataLoader(
                JSONLDataset(train_path, max_len=args.max_length, encode_on_getitem=False),
                batch_size=args.eval_batch_size,
                shuffle=False,
                collate_fn=collator,
                **loader_kwargs,
            )
            prototype_memory.update_prototypes(model, prototype_loader, device)

        print(f"[O-LoRA] Evaluating Task 0 -> {task_id}")
        test_accs = []
        for eval_id in range(task_id + 1):
            test_path = task_path(args.data_root, eval_id, "test")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"[O-LoRA] Missing test data: {test_path}")
            test_loader = DataLoader(
                JSONLDataset(test_path, max_len=args.max_length, encode_on_getitem=False),
                batch_size=args.eval_batch_size,
                shuffle=False,
                collate_fn=collator,
                **loader_kwargs,
            )
            if diagnostic_interface:
                mask_audit = trainer.evaluate_global_seen_and_future(
                    test_loader,
                    seen_labels=seen_labels,
                )
                acc = mask_audit["global_accuracy"]
                acc_seen = mask_audit["seen_accuracy"]
                acc_ncm = trainer.evaluate_ncm(test_loader, prototype_memory)
                R_seen[task_id, eval_id] = acc_seen
                R_ncm[task_id, eval_id] = acc_ncm
                R_pred_future_count[task_id, eval_id] = int(
                    mask_audit["pred_future_count"]
                )
                R_eval_sample_count[task_id, eval_id] = int(
                    mask_audit["sample_count"]
                )
            else:
                acc = trainer.evaluate(
                    test_loader,
                    seen_labels=None if args.eval_all_classes else seen_labels,
                )
            test_accs.append(acc)
            R[task_id, eval_id] = acc

        if diagnostic_interface:
            stage_future = int(
                np.sum(R_pred_future_count[task_id, : task_id + 1])
            )
            stage_samples = int(
                np.sum(R_eval_sample_count[task_id, : task_id + 1])
            )
            stage_global = float(np.mean(R[task_id, : task_id + 1]))
            stage_seen = float(np.mean(R_seen[task_id, : task_id + 1]))
            pred_future = (
                stage_future / stage_samples if stage_samples else 0.0
            )
            print(
                "[Seen-mask Audit] "
                f"global={stage_global * 100:.2f}%, "
                f"seen={stage_seen * 100:.2f}%, "
                f"delta={(stage_seen - stage_global) * 100:+.2f}%, "
                f"PredFuture={pred_future * 100:.2f}%"
            )

        avg_acc = np.mean(test_accs) * 100
        current_acc = test_accs[-1] * 100 if test_accs else 0.0
        print(f"[O-LoRA] Task {task_id} Avg={avg_acc:.2f}%")
        print(f"[O-LoRA] Current Task Acc={current_acc:.2f}%")
        print(f"[O-LoRA] Acc List={[f'{acc * 100:.1f}' for acc in test_accs]}")

    if diagnostic_interface:
        results = summarize_final_results(
            R,
            R_seen,
            R_ncm,
            task_order=list(range(args.num_tasks)),
            num_tasks=args.num_tasks,
        )
        masking = summarize_evaluation_masking(
            R,
            R_seen,
            num_tasks=args.num_tasks,
            pred_future_count_matrix=R_pred_future_count,
            eval_sample_count_matrix=R_eval_sample_count,
        )
        results["evaluation_only_seen_masking"] = masking
        results["matrix_pred_future_count"] = R_pred_future_count.tolist()
        results["matrix_eval_sample_count"] = R_eval_sample_count.tolist()
        results["matrix_pred_future_rate"] = masking[
            "matrix_pred_future_rate"
        ]
        final_avg = results["final_avg"]
        avg_inc = results["avg_inc"]
        bwt = (
            results["bwt_seen"]
            if args.classifier_protocol == "dynamic_seen"
            else results["bwt_global"]
        )
    else:
        if args.num_tasks > 0:
            final_idx = args.num_tasks - 1
            final_avg = float(np.mean(R[final_idx, : args.num_tasks]))
            avg_inc = float(
                np.mean([np.mean(R[t, : t + 1]) for t in range(args.num_tasks)])
            )
            bwt = float(
                np.mean([R[final_idx, i] - R[i, i] for i in range(args.num_tasks - 1)])
                if args.num_tasks > 1
                else 0.0
            )
        else:
            final_avg = avg_inc = bwt = 0.0
        results = {
            "final_avg": final_avg,
            "avg_inc": avg_inc,
            "bwt": bwt,
            "matrix": R.tolist(),
        }

    results.update({
        "method": "O-LoRA",
        "variant": (
            "officialish_text_cil_dynamic_seen"
            if args.classifier_protocol == "dynamic_seen"
            else (
                "officialish_text_cil_global_interface"
                if args.global_interface
                else "officialish_text_cil"
            )
        ),
        "exp_name": args.exp_name,
        "seed": args.seed,
        "eval_protocol": (
            "dynamic-seen"
            if args.classifier_protocol == "dynamic_seen"
            else (
                "fixed-global-unmasked"
                if args.global_interface
                else ("all-classes" if args.eval_all_classes else "seen-so-far")
            )
        ),
        "training_protocol": (
            "dynamic-seen"
            if args.classifier_protocol == "dynamic_seen"
            else ("fixed-global-unmasked" if args.global_interface else "seen-so-far")
        ),
        "classifier_protocol": args.classifier_protocol,
        "global_interface": args.global_interface,
        "officialish_accumulation": not args.no_accumulate,
        "freeze_old_classifier_rows": args.freeze_old_classifier_rows,
        "target_modules": args.target_modules,
        "lora_rank": args.lora_rank,
        "olora_lambda1": args.olora_lambda1,
        "olora_lambda2": args.olora_lambda2,
        "historical_real_text_storage": False,
        "sample_task_id_at_inference": False,
        "diagnostic_only_prototype_memory": diagnostic_interface,
        "prototype_memory_used_for_training": False,
        "prototype_memory_used_for_inference": False,
        "seen_labels_by_stage": seen_labels_by_stage,
        "classifier_class_ids_by_stage": classifier_class_ids_by_stage,
        "classifier_output_dims_by_stage": classifier_output_dims_by_stage,
        "task_order": list(range(args.num_tasks)),
        "global_num_classes": args.num_classes,
        "protocol_contract": protocol_contract,
    })
    if args.classifier_protocol == "dynamic_seen":
        results.update(
            {
                "matrix_dynamic_seen": results["matrix_seen"],
                "final_avg_dynamic_seen": results["final_avg"],
                "avg_inc_dynamic_seen": results["avg_inc"],
                "bwt_dynamic_seen": results["bwt_seen"],
                "primary_bwt_key": "bwt_dynamic_seen",
                "legacy_global_fields_are_dynamic_aliases": True,
            }
        )
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[O-LoRA] Final Avg: {final_avg * 100:.2f}%")
    print(f"[O-LoRA] Avg Inc:   {avg_inc * 100:.2f}%")
    print(f"[O-LoRA] BWT:       {bwt * 100:.2f}%")
    if diagnostic_interface:
        print(f"[O-LoRA] NCM:       {results['final_avg_ncm'] * 100:.2f}%")
        print(f"[O-LoRA] BWT_seen:  {results['bwt_seen'] * 100:.2f}%")
        print(f"[O-LoRA] BWT_feat:  {results['bwt_features'] * 100:.2f}%")
        print(f"[O-LoRA] BWT_cls:   {results['bwt_classifier'] * 100:.2f}%")
    print(f"[O-LoRA] Results:   {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
