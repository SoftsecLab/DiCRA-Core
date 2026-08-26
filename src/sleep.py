from src.alignment import (
    _synchronize_device,
    align_classifier,
    effective_alignment_method,
    finalize_alignment_report,
    imprint_classifier_from_prototypes,
    resolve_rem_budget,
    temporary_imprinted_classifier,
    write_alignment_budget_log,
)
from src.config import build_consolidation_config
from src.consolidation import consolidate_lora
from src.run_config import AlignmentConfig, ConsolidationConfig
from src.stabilization import (
    STABILIZATION_GROUPS,
    finalize_stabilization_audit,
    init_stabilization_audit,
    record_stabilization_change,
    stabilization_group,
    stabilize_model,
)


def merge_and_reinit_lora(model, args, task_id=0):


    config = (
        args
        if isinstance(args, ConsolidationConfig)
        else build_consolidation_config(args)
    )
    return consolidate_lora(
        model,
        config,
        task_id=task_id,
    )


def sleep_phase(
    model,
    tokenizer,
    device,
    config: AlignmentConfig,
    prototype_memory,
    prototype_loader=None,
    before_rem_callback=None,
    alignment_callback=None,
    output_dir=None,
):

    alignment_method = effective_alignment_method(config)
    print(f"[Sleep] Stabilization + Alignment ({alignment_method})")

    exclude_classifier_stabilization = config.exclude_classifier_stabilization
    audit_stabilization = config.audit_stabilization
    stabilization_audit = (
        init_stabilization_audit(
            model,
            task_id=config.task_id,
            exclude_classifier=exclude_classifier_stabilization,
        )
        if audit_stabilization
        else None
    )

    if prototype_loader is not None and prototype_memory is not None:
        prototype_memory.update_prototypes(model, prototype_loader, device)

    nrem_num_classes = len(prototype_memory.prototypes) if prototype_memory else 1

    stabilize_model(
        model,
        config,
        nrem_num_classes,
        output_dir=output_dir,
        audit=stabilization_audit,
    )

    if prototype_loader is not None and prototype_memory is not None:
        print("   [Post-NREM] refresh prototypes")
        prototype_memory.update_prototypes(model, prototype_loader, device)

    if before_rem_callback is not None:
        before_rem_callback()

    return align_classifier(
        model,
        prototype_memory,
        device,
        config,
        output_dir=output_dir,
        callback=alignment_callback,
    )
