"""
Real hyperparameter tuning via Optuna (TPE/Bayesian search by default).
Each trial actually trains a real model for --trial-epochs epochs and
scores it on the real validation set — no scoring heuristic, no lookup
table. Requires: pip install optuna

Usage:
    python -m src.tune --arch mobilenet_v1 --dataset cifar10 --resolution 32 \
        --trials 15 --trial-epochs 5 --out-dir runs/tune_mbv1_cifar10

Then take the best hyperparameters printed at the end and pass them to
train.py for the full run.
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from .data import get_dataloaders
from .models import build_model
from .train import build_optimizer, set_seed
from .attacks import evaluate_clean_accuracy


def objective_factory(args, channels, num_classes, mean, std):
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.7)  # reserved for arch extensions
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])

        set_seed(args.seed)
        train_loader, val_loader, _, _, _, _, _ = get_dataloaders(
            args.dataset, resolution=args.resolution, batch_size=batch_size,
            data_dir=args.data_dir, seed=args.seed)

        model = build_model(args.arch, channels, num_classes, args.resolution, mean, std).to(args.device)
        optimizer = build_optimizer(optimizer_name, model.parameters(), lr)

        for epoch in range(args.trial_epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(args.device), y.to(args.device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
            _, val_acc = evaluate_clean_accuracy(model, val_loader, args.device)
            trial.report(val_acc, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        _, final_val_acc = evaluate_clean_accuracy(model, val_loader, args.device)
        return final_val_acc
    return objective


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="mobilenet_v1")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--trial-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="runs/tune")
    args = p.parse_args()

    try:
        import optuna
    except ImportError:
        raise SystemExit("Optuna is required for real hyperparameter search: pip install optuna")

    from .data import DATASET_STATS
    stats = DATASET_STATS[args.dataset]

    os.makedirs(args.out_dir, exist_ok=True)
    study = optuna.create_study(direction="maximize",
                                  pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective_factory(args, stats["channels"], stats["classes"],
                                       stats["mean"], stats["std"]),
                    n_trials=args.trials)

    print("\nBest trial:")
    print(f"  value (val acc): {study.best_trial.value:.4f}")
    print(f"  params: {study.best_trial.params}")

    with open(os.path.join(args.out_dir, "best_params.json"), "w") as f:
        json.dump({"value": study.best_trial.value, "params": study.best_trial.params}, f, indent=2)
    print(f"Saved to {os.path.join(args.out_dir, 'best_params.json')}")


if __name__ == "__main__":
    main()
