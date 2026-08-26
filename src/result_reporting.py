from __future__ import annotations

import json
import os
from dataclasses import dataclass

from src.evaluation import (
    summarize_evaluation_masking,
    summarize_final_results,
)


@dataclass(frozen=True)
class ResultOptions:


    num_tasks: int
    num_classes: int
    alignment_method: str
    classifier_protocol: str
    audit_imprinting_agreement: bool
    evaluate_train_accuracy: bool
    run_reference_diagnostics: bool
    run_oracle_refit: bool
    log_rem_diagnostics: bool
    print_resource_summary: bool
    enabled_runtime_plugins: tuple[str, ...]

    @classmethod
    def from_args(cls, args, enabled_plugins) -> "ResultOptions":
        return cls(
            num_tasks=int(args.num_tasks),
            num_classes=int(args.num_classes),
            alignment_method=str(args.alignment_method),
            classifier_protocol=str(args.classifier_protocol),
            audit_imprinting_agreement=bool(
                args.audit_imprinting_agreement
            ),
            evaluate_train_accuracy=bool(
                args.evaluate_train_accuracy
            ),
            run_reference_diagnostics=bool(
                args.run_reference_diagnostics
            ),
            run_oracle_refit=bool(args.run_oracle_refit),
            log_rem_diagnostics=bool(args.log_rem_diagnostics),
            print_resource_summary=bool(args.print_resource_summary),
            enabled_runtime_plugins=tuple(enabled_plugins),
        )


def assemble_results(run_state, task_order, options: ResultOptions):


    results = summarize_final_results(
        test_matrix=run_state.test_matrix,
        seen_matrix=run_state.seen_matrix,
        ncm_matrix=run_state.ncm_matrix,
        task_order=task_order,
        num_tasks=options.num_tasks,
    )
    pred_future_count_matrix = getattr(
        run_state, "pred_future_count_matrix", None
    )
    eval_sample_count_matrix = getattr(
        run_state, "eval_sample_count_matrix", None
    )
    masking = summarize_evaluation_masking(
        global_matrix=run_state.test_matrix,
        seen_matrix=run_state.seen_matrix,
        num_tasks=options.num_tasks,
        pred_future_count_matrix=pred_future_count_matrix,
        eval_sample_count_matrix=eval_sample_count_matrix,
    )
    results["evaluation_only_seen_masking"] = masking
    if pred_future_count_matrix is not None:
        results["matrix_pred_future_count"] = (
            pred_future_count_matrix.tolist()
        )
        results["matrix_eval_sample_count"] = (
            eval_sample_count_matrix.tolist()
        )
        results["matrix_pred_future_rate"] = masking[
            "matrix_pred_future_rate"
        ]
    alignment_method = options.alignment_method
    results["alignment_method"] = alignment_method
    if alignment_method == "direct_ncm":
        results["primary_inference"] = "direct_ncm"
    elif alignment_method == "eval_only_imprinting":
        results["primary_inference"] = (
            "eval_only_imprinted_classifier"
        )
    else:
        results["primary_inference"] = "classifier"

    results["prototype_memory_required_at_inference"] = (
        alignment_method == "direct_ncm"
    )
    results[
        "prototype_memory_required_for_eval_head_construction"
    ] = alignment_method == "eval_only_imprinting"
    results["prototype_memory_lookup_per_prediction"] = (
        alignment_method == "direct_ncm"
    )
    results["persistent_classifier_modified_by_alignment"] = (
        alignment_method
        in {"gaussian", "mean_only", "weight_imprinting"}
    )
    results["classifier_optimized_during_alignment"] = (
        alignment_method in {"gaussian", "mean_only"}
    )
    results["alignment_uses_variance"] = (
        alignment_method == "gaussian"
    )
    results["imprinting_agreement_audited"] = (
        options.audit_imprinting_agreement
    )
    results["imprinting_agreement_by_stage"] = (
        run_state.imprinting_agreement_by_stage
    )
    results["eval_only_imprinting_by_stage"] = (
        run_state.eval_only_imprinting_by_stage
    )
    results["eval_only_imprinting_total_build_sec"] = sum(
        row["build_sec"]
        for row in run_state.eval_only_imprinting_by_stage
    )
    results["eval_only_imprinting_optimizer_steps"] = 0
    results["additional_persistent_parameters"] = 0
    results["enabled_runtime_plugins"] = list(
        options.enabled_runtime_plugins
    )

    if alignment_method == "direct_ncm":
        classifier_snapshot = {
            "final_avg": results["final_avg"],
            "avg_inc": results["avg_inc"],
            "bwt_global": results["bwt_global"],
            "bwt_seen": results["bwt_seen"],
            "bwt_classifier": results["bwt_classifier"],
            "matrix": results["matrix"],
            "matrix_seen": results["matrix_seen"],
        }
        results["diagnostic_classifier"] = classifier_snapshot
        results["final_avg"] = results["final_avg_ncm"]
        results["final_avg_seen"] = results["final_avg_ncm"]
        results["avg_inc"] = results["avg_inc_ncm"]
        results["avg_inc_seen"] = results["avg_inc_ncm"]
        results["bwt"] = results["bwt_features"]
        results["bwt_global"] = None
        results["bwt_seen"] = results["bwt_features"]
        results["bwt_classifier"] = None
        results["matrix"] = results["matrix_ncm"]
        results["matrix_seen"] = results["matrix_ncm"]
        results["bwt_direct_ncm"] = results["bwt_features"]

    results["train_accuracy_evaluated"] = (
        options.evaluate_train_accuracy
    )
    results["matrix_train"] = (
        run_state.train_matrix.tolist()
        if options.evaluate_train_accuracy
        else None
    )
    results["global_num_classes"] = options.num_classes
    results["seen_labels_by_stage"] = (
        run_state.seen_labels_by_stage
    )
    results["classifier_protocol"] = options.classifier_protocol
    results["classifier_class_ids_by_stage"] = (
        run_state.classifier_class_ids_by_stage
    )
    results["classifier_output_dims_by_stage"] = (
        run_state.classifier_output_dims_by_stage
    )

    if options.classifier_protocol == "dynamic_seen":
        results["matrix_dynamic_seen"] = results["matrix_seen"]
        results["final_avg_dynamic_seen"] = results["final_avg"]
        results["avg_inc_dynamic_seen"] = results["avg_inc"]
        results["bwt_dynamic_seen"] = results["bwt"]
        results["primary_bwt_key"] = "bwt_dynamic_seen"
        results["legacy_global_fields_are_dynamic_aliases"] = True
        results["prediction_protocol"] = {
            "primary": (
                "direct nearest-prototype prediction over observed classes"
                if alignment_method == "direct_ncm"
                else (
                    "temporary all-seen prototype-imprinted cosine "
                    "classifier; persistent Acquisition classifier is "
                    "restored after evaluation"
                    if alignment_method == "eval_only_imprinting"
                    else (
                        "dynamically expanded classifier; loss and "
                        "prediction over observed classes only"
                    )
                )
            ),
            "seen": (
                "all rows currently present in the dynamic classifier"
            ),
            "ncm": "prototype labels observed through each stage",
            "legacy_matrix_alias": (
                "matrix is identical to matrix_seen for compatibility; "
                "it is not a fixed-global full-class evaluation"
            ),
        }
    else:
        results["primary_bwt_key"] = "bwt_global"
        results["prediction_protocol"] = {
            "global": (
                "fixed full output space without future-class masking"
            ),
            "seen": (
                "classifier logits restricted to labels observed "
                "through each stage"
            ),
            "ncm": "prototype labels observed through each stage",
        }

    return results


def build_clora_report(trainer):


    if trainer.clora_regularizer is None:
        return None

    report = trainer.clora_regularizer.audit_report()
    if not report["subspaces_unchanged"]:
        raise RuntimeError("CLoRA fixed regularization subspaces changed")
    report["lambda"] = trainer.clora_lambda
    report["task_history"] = trainer.clora_task_history
    return report


def write_json_artifact(output_dir, filename, payload) -> None:
    with open(
        os.path.join(output_dir, filename),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=4)


def write_initial_clora_audit(output_dir, trainer) -> None:


    if trainer.clora_regularizer is None:
        return
    write_json_artifact(
        output_dir,
        "clora_regularizer.json",
        trainer.clora_regularizer.audit_report(),
    )


def _write_reference_diagnostics(
    *,
    results,
    run_state,
    task_order,
    options,
    output_dir,
    oracle_args,
    model,
    make_loader,
    device,
) -> None:
    if (
        not options.run_reference_diagnostics
        and not options.run_oracle_refit
    ):
        return

    from src.reference_diagnostics import (
        run_classifier_oracle_refit,
        summarize_probe_diagnostics,
        write_reference_diagnostics,
    )

    reference_payload = None
    if options.run_reference_diagnostics:
        reference_payload = summarize_probe_diagnostics(
            probe_matrix=run_state.probe_matrix,
            ncm_matrix=run_state.ncm_matrix,
            num_tasks=options.num_tasks,
        )
        reference_payload["task_order"] = task_order
        reference_payload["completed_tasks"] = options.num_tasks

    if options.run_oracle_refit:
        print("[Reference Diagnostics] Classifier-only oracle refit...")
        oracle_payload = run_classifier_oracle_refit(
            task_id=options.num_tasks - 1,
            args=oracle_args,
            task_order=task_order,
            model=model,
            make_loader=make_loader,
            device=device,
        )
        oracle_payload["bwt_classifier"] = float(
            results["bwt_classifier"]
        )
        oracle_payload["f_cls"] = float(
            -results["bwt_classifier"]
        )
        if reference_payload is None:
            reference_payload = {
                "task_order": task_order,
                "completed_tasks": options.num_tasks,
            }
        reference_payload["oracle_refit"] = oracle_payload
        print(
            "[Reference Diagnostics] "
            f"oracle_refit_avg="
            f"{oracle_payload['oracle_refit_final_avg'] * 100:.2f}%, "
            f"current_cls_avg="
            f"{oracle_payload['current_classifier_final_avg'] * 100:.2f}%, "
            f"repair_gain="
            f"{oracle_payload['oracle_repair_gain'] * 100:.2f}%, "
            f"F_cls={oracle_payload['f_cls'] * 100:.2f}%"
        )

    if reference_payload is not None:
        write_reference_diagnostics(output_dir, reference_payload)


def _write_process_diagnostics(
    diagnostics,
    output_dir,
    *,
    log_rem_diagnostics,
) -> None:
    if not diagnostics.tracked_stages:
        return

    write_json_artifact(
        output_dir,
        "process_analysis.json",
        diagnostics.process_payload(),
    )
    if log_rem_diagnostics:
        diagnostics.write_rem_json()
        diagnostics.print_rem_diagnostics_summary()


def print_final_results(
    results,
    task_order,
    options,
    *,
    output_dir,
    time_profiler,
) -> None:
    print("\n" + "=" * 80)
    print("[FINAL RESULTS]")
    print(f"Task order: {task_order}")
    print(f"Final Avg: {results['final_avg'] * 100:.2f}%")
    print(f"Final NCM: {results['final_avg_ncm'] * 100:.2f}%")
    print(f"Avg Inc: {results['avg_inc'] * 100:.2f}%")
    print(f"Avg Inc NCM: {results['avg_inc_ncm'] * 100:.2f}%")
    if options.alignment_method == "direct_ncm":
        print(f"BWT_direct_ncm: {results['bwt'] * 100:.2f}%")
    elif options.classifier_protocol == "dynamic_seen":
        print(
            f"BWT_dynamic_seen: "
            f"{results['bwt_dynamic_seen'] * 100:.2f}%"
        )
    else:
        print(f"BWT_global: {results['bwt_global'] * 100:.2f}%")
    print(f"BWT_seen: {results['bwt_seen'] * 100:.2f}%")
    print(f"BWT_feat: {results['bwt_features'] * 100:.2f}%")
    if results["bwt_classifier"] is None:
        print(
            "BWT_cls: N/A "
            "(Direct NCM does not use the learned classifier)"
        )
    else:
        print(f"BWT_cls: {results['bwt_classifier'] * 100:.2f}%")
    if options.print_resource_summary:
        from src.profiling import print_resource_summary

        print_resource_summary(
            output_dir,
            time_profiler=time_profiler,
        )
    print("=" * 80)


def finalize_experiment(
    *,
    run_state,
    task_order,
    options,
    output_dir,
    trainer,
    time_profiler,
    diagnostics,
    oracle_args,
    model,
    make_loader,
    device,
):


    if options.num_tasks <= 0:
        return None

    results = assemble_results(run_state, task_order, options)
    clora_report = build_clora_report(trainer)
    if clora_report is not None:
        results["clora"] = clora_report
        write_json_artifact(
            output_dir,
            "clora_regularizer.json",
            clora_report,
        )

    write_json_artifact(output_dir, "results.json", results)
    time_profiler.write(output_dir)
    _write_reference_diagnostics(
        results=results,
        run_state=run_state,
        task_order=task_order,
        options=options,
        output_dir=output_dir,
        oracle_args=oracle_args,
        model=model,
        make_loader=make_loader,
        device=device,
    )
    _write_process_diagnostics(
        diagnostics,
        output_dir,
        log_rem_diagnostics=options.log_rem_diagnostics,
    )
    print_final_results(
        results,
        task_order,
        options,
        output_dir=output_dir,
        time_profiler=time_profiler,
    )
    return results
