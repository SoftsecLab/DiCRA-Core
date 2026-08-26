"""Wake-phase optimization services for continual RECAP training."""

from __future__ import annotations

import copy
import gc
import math
from contextlib import nullcontext
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.clora_regularizer import CLoRARegularizer
from src.run_config import WakeConfig


def build_grad_scaler(device, precision: str):
    """Create a GradScaler without warning on new or breaking old PyTorch."""

    enabled = device.type == "cuda" and precision == "fp16"
    amp_namespace = getattr(torch, "amp", None)
    grad_scaler = getattr(amp_namespace, "GradScaler", None)
    if grad_scaler is not None:
        return grad_scaler("cuda", enabled=enabled)

    return torch.cuda.amp.GradScaler(enabled=enabled)


def freeze_backbone_layers(model, freeze_layers: int) -> list[nn.Parameter]:
    """Temporarily freeze embeddings and the first encoder layers."""

    frozen_params = []
    if freeze_layers <= 0:
        return frozen_params

    for param in model.bert.embeddings.parameters():
        if param.requires_grad:
            param.requires_grad = False
            frozen_params.append(param)

    encoder_layers = model.bert.encoder.layer
    for layer_idx in range(min(freeze_layers, len(encoder_layers))):
        for param in encoder_layers[layer_idx].parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_params.append(param)

    return frozen_params


def restore_frozen_parameters(
    frozen_params: Iterable[nn.Parameter],
) -> None:
    """Restore only parameters temporarily frozen by the Wake phase."""

    for param in frozen_params:
        param.requires_grad = True


def build_wake_optimizer(
    model,
    *,
    freeze_layers: int,
    base_lr: float,
    weight_decay: float,
    llrd_gamma: float,
):
    """Build AdamW with optional layer-wise learning-rate decay."""

    if llrd_gamma >= 1.0:
        return torch.optim.AdamW(
            filter(lambda param: param.requires_grad, model.parameters()),
            lr=base_lr,
            weight_decay=weight_decay,
        )

    encoder_layers = model.bert.encoder.layer
    top_layer = len(encoder_layers) - 1
    param_groups = []
    assigned_ids = set()

    for layer_idx in range(freeze_layers, len(encoder_layers)):
        depth_from_top = top_layer - layer_idx
        layer_lr = base_lr * (llrd_gamma ** depth_from_top)
        layer_params = [
            param
            for param in encoder_layers[layer_idx].parameters()
            if param.requires_grad
        ]
        if not layer_params:
            continue
        param_groups.append(
            {
                "params": layer_params,
                "lr": layer_lr,
                "weight_decay": weight_decay,
            }
        )
        assigned_ids.update(id(param) for param in layer_params)

    other_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in assigned_ids
    ]
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
        )

    return torch.optim.AdamW(param_groups)


def build_wake_scheduler(
    optimizer,
    *,
    batches_per_epoch: int,
    epochs: int,
    warmup_ratio: float,
    min_lr_ratio: float,
):
    """Build the per-batch warmup and cosine-decay scheduler."""

    if warmup_ratio <= 0:
        return None

    total_steps = batches_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return max(current_step / max(warmup_steps, 1), 1e-6)

        progress = (
            (current_step - warmup_steps)
            / max(total_steps - warmup_steps, 1)
        )
        return min_lr_ratio + (
            (1.0 - min_lr_ratio)
            * 0.5
            * (1.0 + math.cos(math.pi * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class WakeTrainer:
    """Own Wake-phase precision, optimization, replay and regularization."""

    def __init__(self, model, device, config: WakeConfig):
        if not isinstance(config, WakeConfig):
            raise TypeError("config must be a WakeConfig")

        self.model = model
        self.device = device
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.grad_accum_steps = config.grad_accum_steps
        self.feat_distill_beta = config.feat_distill_beta
        self.frozen_model = None
        self.clora_lambda = config.clora_lambda
        self.clora_regularizer = None
        self.clora_task_history = []
        self.last_clora_epoch_loss = None

        if self.clora_lambda > 0:
            self.clora_regularizer = CLoRARegularizer(
                model,
                k=config.clora_k,
                seed=config.seed,
            ).to(device)
            print(
                "[CLoRA] initialized fixed orthogonal subspaces: "
                f"modules={self.clora_regularizer.num_lora_modules}, "
                f"k={self.clora_regularizer.k}, "
                f"lambda={self.clora_lambda}"
            )

        precision = config.precision
        self.use_amp = (
            self.device.type == "cuda"
            and precision in {"fp16", "bf16"}
        )
        self.amp_dtype = (
            torch.float16
            if precision == "fp16"
            else torch.bfloat16
            if precision == "bf16"
            else None
        )
        self.scaler = build_grad_scaler(self.device, precision)
        self.trainable_params_cache = []
        self.bind_model(model)

    def bind_model(self, model) -> None:
        """Update the Wake target after another phase replaces the model."""

        self.model = model
        self.trainable_params_cache = [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]

    def autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def train_epoch(
        self,
        loader,
        optimizer,
        epoch_idx=0,
        scheduler=None,
        prototype_memory=None,
    ):
        self.model.train()
        total_loss = 0.0
        total_clora_loss = 0.0
        wake_replay_beta = self.config.wake_replay_beta

        pbar = tqdm(
            loader,
            desc=f"  Epoch {epoch_idx + 1}/{self.config.epochs}",
            leave=False,
        )
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            classifier_labels = self.model.global_to_classifier_labels(
                labels
            )

            with self.autocast_context():
                if self.frozen_model is not None:
                    raw_feat = self.model.get_features(
                        input_ids,
                        attention_mask,
                    )
                    logits = self.model.classifier(
                        self.model.dropout(raw_feat)
                    )
                    ce_loss = self.criterion(
                        logits,
                        classifier_labels,
                    )
                    with torch.no_grad():
                        old_feat = self.frozen_model.get_features(
                            input_ids,
                            attention_mask,
                        )
                    distill_loss = 1 - F.cosine_similarity(
                        raw_feat,
                        old_feat,
                        dim=1,
                    ).mean()
                    loss = (
                        ce_loss
                        + self.feat_distill_beta * distill_loss
                    )
                else:
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, classifier_labels)

            if (
                wake_replay_beta > 0
                and prototype_memory is not None
                and len(prototype_memory.prototypes) > 0
            ):
                with self.autocast_context():
                    proto_feats, proto_labels = (
                        prototype_memory.get_prototype_batch(
                            batch_size=32
                        )
                    )
                    if proto_feats is not None:
                        proto_feats = proto_feats.to(self.device)
                        proto_labels = proto_labels.to(self.device)
                        proto_logits = self.model.classifier(
                            self.model.dropout(proto_feats)
                        )
                        replay_labels = (
                            self.model.global_to_classifier_labels(
                                proto_labels
                            )
                        )
                        replay_loss = F.cross_entropy(
                            proto_logits,
                            replay_labels,
                        )
                        loss = loss + wake_replay_beta * replay_loss

            if self.clora_regularizer is not None:
                clora_loss = self.clora_regularizer()
                loss = loss + self.clora_lambda * clora_loss
                total_clora_loss += float(
                    clora_loss.detach().item()
                )

            total_loss += loss.item()
            loss_to_backward = loss / self.grad_accum_steps

            if self.scaler.is_enabled():
                self.scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            should_step = (
                batch_idx % self.grad_accum_steps == 0
                or batch_idx == len(loader)
            )
            if not should_step:
                pbar.set_postfix({"Loss": f"{loss.item():.3f}"})
                continue

            if self.scaler.is_enabled():
                self.scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                [
                    param
                    for _, param in self.trainable_params_cache
                    if param.grad is not None
                ],
                max_norm=1.0,
            )

            if self.scaler.is_enabled():
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

            pbar.set_postfix({"Loss": f"{loss.item():.3f}"})

        self.last_clora_epoch_loss = (
            total_clora_loss / max(len(loader), 1)
            if self.clora_regularizer is not None
            else None
        )
        return total_loss / max(len(loader), 1)

    def train_task(
        self,
        loader,
        task_id=0,
        prototype_memory=None,
    ) -> None:
        self.bind_model(self.model)
        freeze_layers = self.config.freeze_layers
        frozen_params = freeze_backbone_layers(
            self.model,
            freeze_layers,
        )

        llrd_gamma = self.config.llrd_gamma
        base_lr = self.config.lr
        weight_decay = self.config.weight_decay
        optimizer = build_wake_optimizer(
            self.model,
            freeze_layers=freeze_layers,
            base_lr=base_lr,
            weight_decay=weight_decay,
            llrd_gamma=llrd_gamma,
        )

        if self.feat_distill_beta > 0 and task_id > 0:
            self.frozen_model = copy.deepcopy(self.model)
            self.frozen_model.eval()
            for param in self.frozen_model.parameters():
                param.requires_grad = False
            distill_info = (
                f", feat_distill_beta={self.feat_distill_beta}"
            )
        else:
            self.frozen_model = None
            distill_info = ""

        freeze_info = (
            f", freeze_bottom={freeze_layers}"
            if freeze_layers > 0
            else ""
        )
        llrd_info = (
            f", llrd_gamma={llrd_gamma}"
            if llrd_gamma < 1.0
            else ""
        )

        warmup_ratio = self.config.warmup_ratio
        min_lr_ratio = self.config.min_lr_ratio
        scheduler = build_wake_scheduler(
            optimizer,
            batches_per_epoch=len(loader),
            epochs=self.config.epochs,
            warmup_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio,
        )
        schedule_info = (
            f", warmup={warmup_ratio}"
            if scheduler is not None
            else ""
        )

        wake_replay_beta = self.config.wake_replay_beta
        replay_info = (
            f", awake_replay_beta={wake_replay_beta}"
            if wake_replay_beta > 0 and task_id > 0
            else ""
        )
        clora_info = (
            f", CLoRA(k={self.clora_regularizer.k},"
            f"lambda={self.clora_lambda})"
            if self.clora_regularizer is not None
            else ""
        )

        print(
            f"[Wake] Train task: epochs={self.config.epochs}"
            f" (precision={self.config.precision}, "
            f"grad_accum={self.grad_accum_steps}{distill_info}"
            f"{freeze_info}{llrd_info}{schedule_info}{replay_info}"
            f"{clora_info})"
        )

        active_prototype_memory = (
            prototype_memory
            if task_id > 0 and wake_replay_beta > 0
            else None
        )

        try:
            clora_epoch_losses = []
            for epoch in range(self.config.epochs):
                self.train_epoch(
                    loader,
                    optimizer,
                    epoch_idx=epoch,
                    scheduler=scheduler,
                    prototype_memory=active_prototype_memory,
                )
                if self.last_clora_epoch_loss is not None:
                    clora_epoch_losses.append(
                        self.last_clora_epoch_loss
                    )
            if clora_epoch_losses:
                self.clora_task_history.append(
                    {
                        "task_id": int(task_id),
                        "epoch_regularization_loss": (
                            clora_epoch_losses
                        ),
                        "mean_regularization_loss": float(
                            sum(clora_epoch_losses)
                            / len(clora_epoch_losses)
                        ),
                    }
                )
        finally:
            self.frozen_model = None
            restore_frozen_parameters(frozen_params)
            torch.cuda.empty_cache()
            gc.collect()
