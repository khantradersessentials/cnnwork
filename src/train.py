"""
Real training entry point.

Example:
    python -m src.train --arch mobilenet_v2 --dataset cifar10 --resolution 32 \
        --optimizer adam --lr 0.001 --batch-size 128 --epochs 30 \
        --defenses pgd_adv_train label_smoothing \
        --eval-attacks fgsm ifgsm pgd --epsilon 8 \
        --num-seeds 3 --early-stopping --patience 5 \
        --out-dir runs/mbv2_cifar10_pgdat

This actually downloads the dataset, actually trains, actually attacks the
actual trained model, and actually measures FLOPs/params/latency. Nothing
in this file is estimated or fabricated. Multi-seed runs (--num-seeds) are
what makes the confidence intervals in metrics.compute_stats meaningful.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import get_dataloaders
from .models import build_model, find_last_conv_layer
from .attacks import fgsm_attack, ifgsm_attack, pgd_attack, evaluate_clean_accuracy, evaluate_adversarial_accuracy
from .defenses import (
    SAM, trades_loss, mart_loss, gradient_regularization_loss, feature_squeeze,
    resize_pad_defense, jpeg_compress_batch, segmentation_purify, frequency_domain_purify,
    free_adv_train_step,
)
from .metrics import compute_flops_params, measure_inference_latency, compute_stats
from .gradcam import GradCAM, overlay_heatmap


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(name, params, lr):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "sam":
        return SAM(params, torch.optim.SGD, rho=0.05, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer '{name}'")


def apply_input_defense(x, defenses):
    if "feature_squeezing" in defenses:
        x = feature_squeeze(x, bit_depth=4)
    if "input_transform" in defenses:
        x = resize_pad_defense(x)
    if "segmentation" in defenses:
        x = segmentation_purify(x)
    if "fdap" in defenses:
        x = frequency_domain_purify(x)
    return x


def train_one_seed(args, seed):
    set_seed(seed)
    device = args.device

    train_loader, val_loader, test_loader, num_classes, channels, mean, std = get_dataloaders(
        args.dataset, resolution=args.resolution, batch_size=args.batch_size,
        data_dir=args.data_dir, seed=seed)

    pooling = "spp" if args.spp else ("aspp" if args.aspp else "none")
    model = build_model(args.arch, channels, num_classes, args.resolution, mean, std,
                          pooling=pooling, use_transformer_head=args.transformer_refinement).to(device)

    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr)
    is_sam = args.optimizer.lower() == "sam"

    aux_models = []
    if "ensemble_adv_train" in args.defenses:
        aux_backbone = build_model(args.arch, channels, num_classes, args.resolution, mean, std).to(device)
        aux_backbone.load_state_dict(model.state_dict())
        aux_models = [aux_backbone]

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc, stall_count, best_state = -1.0, 0, None

    use_pgd_at = "pgd_adv_train" in args.defenses
    use_fgsm_at = "fgsm_adv_train" in args.defenses
    use_free_at = "free_adv_train" in args.defenses
    use_trades = "trades" in args.defenses
    use_mart = "mart" in args.defenses
    use_gradreg = "gradient_regularization" in args.defenses
    label_smoothing = 0.1 if "label_smoothing" in args.defenses else 0.0
    eps = args.epsilon / 255.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = apply_input_defense(x, args.defenses)

            if use_free_at:
                loss_val = free_adv_train_step(model, optimizer, x, y, eps, m=args.free_m)
                running_loss += loss_val; n_batches += 1
                continue

            if use_trades:
                loss = trades_loss(model, x, y, optimizer, epsilon=eps, beta=args.trades_beta)
            elif use_mart:
                x_adv = pgd_attack(model, x, y, eps, steps=args.pgd_train_steps)
                loss = mart_loss(model, x, y, x_adv, beta=args.mart_beta)
            elif use_gradreg:
                loss = gradient_regularization_loss(model, x, y, lam=args.gradreg_lambda)
            elif use_pgd_at or use_fgsm_at or aux_models:
                if aux_models:
                    from .defenses import ensemble_adv_examples
                    x_train = ensemble_adv_examples(aux_models, x, y, eps, pgd_attack)
                elif use_pgd_at:
                    x_train = pgd_attack(model, x, y, eps, steps=args.pgd_train_steps)
                else:
                    x_train = fgsm_attack(model, x, y, eps)
                logits = model(x_train)
                loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

            if not use_trades:
                optimizer.zero_grad()
                loss.backward()
                if is_sam:
                    optimizer.first_step(zero_grad=True)
                    logits2 = model(x)
                    loss2 = F.cross_entropy(logits2, y, label_smoothing=label_smoothing)
                    loss2.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.step()
            else:
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            running_loss += loss.item(); n_batches += 1

        val_loss, val_acc = evaluate_clean_accuracy(model, val_loader, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(running_loss / max(n_batches, 1))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"[seed {seed}] epoch {epoch}/{args.epochs} — train_loss "
              f"{running_loss/max(n_batches,1):.4f}  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

        if args.early_stopping:
            if val_acc > best_val_acc + 0.003:
                best_val_acc, stall_count = val_acc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stall_count += 1
            if stall_count >= args.patience:
                print(f"[seed {seed}] early stopping at epoch {epoch} (patience {args.patience})")
                break

    if args.early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    # ---- Real evaluation on the held-out TEST set ----
    test_loss, test_acc = evaluate_clean_accuracy(model, test_loader, device)
    adv_results = {}
    for atk in args.eval_attacks:
        acc = evaluate_adversarial_accuracy(model, test_loader, device, atk, eps,
                                              steps=args.pgd_eval_steps)
        adv_results[atk] = acc
        print(f"[seed {seed}] adversarial accuracy ({atk}): {acc:.4f}")

    # ---- Real FLOPs / params / latency for THIS model instance ----
    input_shape = (channels, args.resolution, args.resolution)
    try:
        flops, params = compute_flops_params(model, input_shape, device="cpu")
    except ImportError:
        flops, params = None, None
        print("[metrics] thop not installed — skipping FLOPs/params measurement "
              "(pip install thop to enable).")
    lat_mean, lat_std = measure_inference_latency(model, input_shape, device=device)

    # ---- Real Grad-CAM on a few real test images ----
    gradcam_dir = os.path.join(args.out_dir, f"seed{seed}_gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)
    target_layer = find_last_conv_layer(model)
    if target_layer is not None:
        cam = GradCAM(model, target_layer)
        sample_x, sample_y = next(iter(test_loader))
        sample_x = sample_x[:6].to(device)
        heatmaps, _ = cam(sample_x)
        for i in range(sample_x.shape[0]):
            overlay_heatmap(sample_x[i].cpu(), heatmaps[i].cpu()).save(
                os.path.join(gradcam_dir, f"sample_{i}.png"))
    else:
        print("[gradcam] No conv layer found (e.g. vit_tiny) — skipping Grad-CAM for this arch.")

    # ---- Real failure-case collection (misclassified adversarial examples) ----
    failure_dir = os.path.join(args.out_dir, f"seed{seed}_failures")
    os.makedirs(failure_dir, exist_ok=True)
    failure_records = []
    if args.eval_attacks:
        atk_name = args.eval_attacks[0]
        attack_fn = {"fgsm": fgsm_attack, "ifgsm": ifgsm_attack, "pgd": pgd_attack}[atk_name]
        collected = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x_adv = attack_fn(model, x, y, eps) if atk_name == "fgsm" else \
                    attack_fn(model, x, y, eps, steps=args.pgd_eval_steps)
            with torch.no_grad():
                logits = model(x_adv)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(1)
            wrong = (preds != y).nonzero(as_tuple=True)[0]
            for idx in wrong:
                if collected >= args.num_failure_cases:
                    break
                idx = idx.item()
                from PIL import Image
                img = (x_adv[idx].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
                if img.shape[-1] == 1:
                    img = img.squeeze(-1)
                path = os.path.join(failure_dir, f"failure_{collected}.png")
                Image.fromarray(img).save(path)
                failure_records.append({
                    "true_label": int(y[idx].item()), "pred_label": int(preds[idx].item()),
                    "confidence": float(probs[idx, preds[idx]].item()),
                    "attack": atk_name, "image_path": path,
                })
                collected += 1
            if collected >= args.num_failure_cases:
                break

    return {
        "seed": seed, "history": history,
        "test_loss": test_loss, "test_acc": test_acc,
        "adversarial_accuracy": adv_results,
        "flops": flops, "params": params,
        "inference_ms_mean": lat_mean, "inference_ms_std": lat_std,
        "gradcam_dir": gradcam_dir, "failure_cases": failure_records,
    }


def main():
    p = argparse.ArgumentParser(description="Real CNN robustness training/eval pipeline")
    p.add_argument("--arch", default="mobilenet_v1",
                    choices=["mobilenet_v1", "mobilenet_v2", "mobilenet_v3s", "squeezenet",
                             "shufflenet_v2", "efficientnet_b0", "custom", "vit_tiny"])
    p.add_argument("--dataset", default="cifar10",
                    choices=["cifar10", "cifar100", "mnist", "fmnist", "svhn", "stl10"])
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd", "rmsprop", "sam"])
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--dropout", type=float, default=0.25)  # reserved for custom-arch extension
    p.add_argument("--spp", action="store_true")
    p.add_argument("--aspp", action="store_true")
    p.add_argument("--transformer-refinement", action="store_true")

    p.add_argument("--defenses", nargs="*", default=[],
                    choices=["pgd_adv_train", "fgsm_adv_train", "free_adv_train", "trades", "mart",
                             "gradient_regularization", "label_smoothing", "feature_squeezing",
                             "input_transform", "segmentation", "fdap", "ensemble_adv_train"])
    p.add_argument("--trades-beta", type=float, default=6.0)
    p.add_argument("--mart-beta", type=float, default=5.0)
    p.add_argument("--gradreg-lambda", type=float, default=1.0)
    p.add_argument("--free-m", type=int, default=4)
    p.add_argument("--pgd-train-steps", type=int, default=7)
    p.add_argument("--pgd-eval-steps", type=int, default=20)

    p.add_argument("--eval-attacks", nargs="*", default=[], choices=["fgsm", "ifgsm", "pgd"])
    p.add_argument("--epsilon", type=float, default=8.0, help="L_inf budget in /255 units")

    p.add_argument("--early-stopping", action="store_true")
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--num-seeds", type=int, default=1,
                    help="Run this many independent seeds for a real, statistically "
                         "meaningful mean/std/95%% CI on final metrics.")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--num-failure-cases", type=int, default=6)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="runs/experiment")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = []
    for i in range(args.num_seeds):
        seed = args.seed_start + i
        result = train_one_seed(args, seed)
        all_results.append(result)

    summary = {
        "test_acc": compute_stats([r["test_acc"] for r in all_results]),
        "adversarial_accuracy": {
            atk: compute_stats([r["adversarial_accuracy"][atk] for r in all_results])
            for atk in args.eval_attacks
        },
        "flops": all_results[0]["flops"],
        "params": all_results[0]["params"],
        "inference_ms": compute_stats([r["inference_ms_mean"] for r in all_results]),
        "config": vars(args),
        "per_seed_results": all_results,
    }
    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved real results to {out_path}")
    print(f"Test accuracy: mean={summary['test_acc']['mean']:.4f}  "
          f"95% CI=[{summary['test_acc']['ci_low']:.4f}, {summary['test_acc']['ci_high']:.4f}]  "
          f"(n={summary['test_acc']['n']} seed{'s' if args.num_seeds>1 else ''})")


if __name__ == "__main__":
    main()
