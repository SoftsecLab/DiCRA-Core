from src.config import build_sleep_config, build_wake_config
from src.evaluator import RECAPEvaluator
from src.sleep import merge_and_reinit_lora, sleep_phase
from src.wake_training import WakeTrainer


class RECAPTrainer:
    """Compatibility facade coordinating Wake, Sleep and evaluation."""

    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.wake_trainer = WakeTrainer(
            model,
            device,
            build_wake_config(args),
        )
        self.evaluator = RECAPEvaluator(
            self.model,
            self.device,
            self._autocast_context,
        )

    def bind_model(self, model) -> None:
        """Keep all phase services attached to one model instance."""

        self.model = model
        self.wake_trainer.bind_model(model)
        self.evaluator.bind_model(model)

    @property
    def clora_lambda(self):
        return self.wake_trainer.clora_lambda

    @property
    def clora_regularizer(self):
        return self.wake_trainer.clora_regularizer

    @property
    def clora_task_history(self):
        return self.wake_trainer.clora_task_history

    @property
    def last_clora_epoch_loss(self):
        return self.wake_trainer.last_clora_epoch_loss

    def _autocast_context(self):
        return self.wake_trainer.autocast_context()

    def train_epoch(
        self,
        loader,
        optimizer,
        epoch_idx=0,
        scheduler=None,
        prototype_memory=None,
    ):
        self.wake_trainer.bind_model(self.model)
        return self.wake_trainer.train_epoch(
            loader,
            optimizer,
            epoch_idx=epoch_idx,
            scheduler=scheduler,
            prototype_memory=prototype_memory,
        )

    def train_task(self, loader, task_id=0, prototype_memory=None):
        self.wake_trainer.bind_model(self.model)
        return self.wake_trainer.train_task(
            loader,
            task_id=task_id,
            prototype_memory=prototype_memory,
        )

    def sleep(
        self,
        tokenizer,
        prototype_memory,
        prototype_loader=None,
        task_id=0,
        output_dir=None,
        alignment_callback=None,
    ):
        if not self.args.use_sleep:
            return self.model

        print("[Sleep] consolidate current task")

        # Phase 0: merge LoRA updates into the shared backbone.
        if getattr(self.args, "no_consolidation", False):
            print("[Consolidation] skipped by --no_consolidation")
        else:
            merge_and_reinit_lora(self.model, self.args, task_id=task_id)

        self.bind_model(
            sleep_phase(
                self.model,
                tokenizer,
                self.device,
                build_sleep_config(self.args, task_id),
                prototype_memory,
                prototype_loader=prototype_loader,
                before_rem_callback=None,
                alignment_callback=alignment_callback,
                output_dir=output_dir,
            )
        )
        return self.model

    @staticmethod
    def _predict_from_logits(logits, candidate_labels=None, classifier_class_ids=None):
        """Return global predictions or predictions restricted to candidate labels."""
        return RECAPEvaluator.predict_from_logits(
            logits,
            candidate_labels=candidate_labels,
            classifier_class_ids=classifier_class_ids,
        )

    def _bound_evaluator(self):
        self.evaluator.bind_model(self.model)
        return self.evaluator

    def evaluate(self, loader):
        return self._bound_evaluator().evaluate(loader)

    def evaluate_global_and_seen(self, loader, seen_labels):
        """Evaluate full-output and observed-class predictions in one forward pass."""
        return self._bound_evaluator().evaluate_global_and_seen(
            loader,
            seen_labels,
        )

    def evaluate_global_seen_and_future(self, loader, seen_labels):
        """Also count global predictions assigned to unobserved classes."""
        return self._bound_evaluator().evaluate_global_seen_and_future(
            loader,
            seen_labels,
        )

    def evaluate_ncm(self, loader, prototype_memory):
        """Evaluate feature quality with nearest-centroid classification."""
        return self._bound_evaluator().evaluate_ncm(
            loader,
            prototype_memory,
        )

    def evaluate_classifier_ncm_agreement(
        self,
        loader,
        prototype_memory,
        seen_labels,
    ):
        """Compare cosine-head and single-mean NCM predictions sample by sample."""
        return self._bound_evaluator().evaluate_classifier_ncm_agreement(
            loader,
            prototype_memory,
            seen_labels,
        )
