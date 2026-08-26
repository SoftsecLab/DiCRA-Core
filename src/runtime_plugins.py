OPTIONAL_RUNTIME_PLUGINS = {
    "train_accuracy": "evaluate_train_accuracy",
    "stage_analysis": "analyze_stages",
    "alignment_diagnostics": "log_rem_diagnostics",
    "checkpoint_writer": "save_checkpoints",
    "reference_probe": "run_reference_diagnostics",
    "oracle_refit": "run_oracle_refit",
    "prototype_staleness": "run_prototype_staleness_diagnostics",
    "imprinting_agreement": "audit_imprinting_agreement",
}


def enabled_runtime_plugins(args):


    return [
        plugin_id
        for plugin_id, argument in OPTIONAL_RUNTIME_PLUGINS.items()
        if bool(getattr(args, argument, False))
    ]
