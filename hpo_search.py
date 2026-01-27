import os
import sys
import json
import torch
from pathlib import Path

from transformers import TrainingArguments, set_seed, EarlyStoppingCallback

from model.ResNet import ResNetConfig, ResNetForImageClassification
from utils.loss_functions import LossFunctions
from utils.image_processor import CustomImageProcessor
from utils.utils import (
    collate_fn,
    compute_metrics,
    parse_HF_args,
    preprocess_local_folder_dataset,
    preprocess_local_csv_dataset,
    preprocess_hf_dataset,
    preprocess_kg_dataset,
)
from train import CustomTrainer


def main():
    set_seed(42)
    script_args = parse_HF_args()

    # --- Load dataset ONCE ---
    image_processor = CustomImageProcessor.from_pretrained(script_args.model)
    class_names = None

    if script_args.dataset_host == "huggingface":
        train_ds, val_ds, test_ds = preprocess_hf_dataset(script_args.dataset, script_args.model)
    elif script_args.dataset_host == "kaggle":
        train_ds, val_ds, test_ds = preprocess_kg_dataset(
            script_args.dataset, script_args.local_dataset_name, script_args.model
        )
    elif script_args.dataset_host == "local_folder":
        if script_args.local_dataset_format == "csv":
            train_ds, val_ds, test_ds, class_names = preprocess_local_csv_dataset(
                script_args.local_folder_path, script_args.model
            )
        else:
            train_ds, val_ds, test_ds, class_names = preprocess_local_folder_dataset(
                script_args.local_folder_path, script_args.model
            )
    else:
        raise ValueError(f"Unknown dataset_host: {script_args.dataset_host}")

    # --- Load pretrained weights ONCE ---
    pretrained_weights_path = os.path.abspath(script_args.weights)
    pretrained_weights = torch.load(pretrained_weights_path, map_location="cpu")
    filtered_weights = {k: v for k, v in pretrained_weights.items() if "classifier" not in k}

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- model_init: fresh model per trial ---
    def model_init(trial):
        config = ResNetConfig(num_labels=script_args.num_labels, depths=[3, 4, 6, 3])
        model = ResNetForImageClassification(config)
        missing, unexpected = model.load_state_dict(filtered_weights, strict=False)
        if missing:
            print(f"Missing keys: {missing}")

        # Freeze early stages when requested by the trial
        if trial is not None:
            num_frozen = trial.suggest_int("num_frozen_stages", 0, 3)
        else:
            num_frozen = 0

        if num_frozen >= 1:
            for param in model.resnet.embedder.parameters():
                param.requires_grad = False
        if num_frozen >= 2:
            for param in model.resnet.encoder.stages[0].parameters():
                param.requires_grad = False
        if num_frozen >= 3:
            for param in model.resnet.encoder.stages[1].parameters():
                param.requires_grad = False

        model.to(device)
        return model

    # --- Search space for TrainingArguments ---
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.01),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [8, 16, 32]
            ),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "lr_scheduler_type": trial.suggest_categorical(
                "lr_scheduler_type", ["linear", "cosine"]
            ),
        }

    # --- Objective ---
    def compute_objective(metrics):
        return metrics["eval_accuracy"]

    # --- Training arguments (high epoch ceiling; early stopping decides) ---
    training_args = TrainingArguments(
        output_dir="checkpoints/hpo_search",
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=30,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    # --- Build trainer with model_init (no model arg) ---
    trainer = CustomTrainer(
        loss_fxn=script_args.loss_function,
        cost_matrix=script_args.cost_matrix,
        model=None,
        model_init=model_init,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # --- Run HPO ---
    print("=" * 60)
    print("STARTING HYPERPARAMETER SEARCH (Optuna, 20 trials)")
    print("=" * 60)

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=20,
        compute_objective=compute_objective,
    )

    # --- Save results ---
    results_dir = Path("results/hpo_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "best_trial_number": best_trial.run_id,
        "best_eval_accuracy": best_trial.objective,
        "best_hyperparameters": best_trial.hyperparameters,
    }

    output_path = results_dir / "best_hyperparameters.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 60)
    print(f"Best trial: #{best_trial.run_id}")
    print(f"Best eval accuracy: {best_trial.objective:.4f}")
    print(f"Best hyperparameters:")
    for k, v in best_trial.hyperparameters.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
