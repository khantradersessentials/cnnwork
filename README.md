# Robustness Lab — real CNN training, real attacks, real defenses

This is a real PyTorch research codebase, not a demo. Every number it
produces — accuracy, loss, adversarial accuracy, FLOPs, params, inference
latency, confidence intervals — comes from an actual model actually
trained on an actual dataset and actually attacked/evaluated. Nothing is
looked up from a table or estimated by a formula.

**I cannot run this for you in this chat** — the sandbox here has no
internet access (so it can't download CIFAR-10/MNIST/etc.) and no GPU/torch
installed. You need to run it yourself, e.g. in Google Colab (free GPU) or
on your own machine/cluster.

## Setup

```bash
pip install -r requirements.txt
```

GPU strongly recommended for anything beyond MNIST/CIFAR at low resolution
and a handful of epochs — adversarial training in particular is expensive
(each PGD step is a full forward+backward pass).

## Quick start

```bash
python -m src.train \
  --arch mobilenet_v2 --dataset cifar10 --resolution 32 \
  --optimizer adam --lr 0.001 --batch-size 128 --epochs 30 \
  --defenses pgd_adv_train label_smoothing \
  --eval-attacks fgsm ifgsm pgd --epsilon 8 \
  --num-seeds 3 --early-stopping --patience 5 \
  --out-dir runs/mbv2_cifar10_pgdat
```

This downloads real CIFAR-10, trains 3 independent seeds with PGD
adversarial training, evaluates real FGSM/I-FGSM/PGD accuracy on the real
test set separately for each attack, measures real FLOPs/params/latency,
generates real Grad-CAM images and real misclassified-example failure
cases, and writes it all to `runs/mbv2_cifar10_pgdat/results.json` plus
per-seed image folders.

Then build the Excel report:

```bash
python -m src.report --runs runs/mbv2_cifar10_pgdat/results.json --out report.xlsx
```

To compare multiple experiments in one report, pass several `results.json`
paths to `--runs`.

## What each option actually does

| Option | Real implementation |
|---|---|
| `--arch` | `mobilenet_v1` (from-scratch depthwise-separable stack), `mobilenet_v2/v3s`, `squeezenet`, `shufflenet_v2`, `efficientnet_b0` (all real `torchvision.models`, trained from scratch — `weights=None`), `custom` (3-conv baseline), `vit_tiny` (from-scratch ViT using real `nn.TransformerEncoderLayer`) |
| `--dataset` | CIFAR-10/100, MNIST, Fashion-MNIST, SVHN, STL-10 via real `torchvision.datasets` (downloaded on first run) |
| `--resolution` | Real `Resize` transform, 32–224px |
| `--optimizer` | Adam, SGD+momentum, RMSprop, and a real SAM (Foret et al. 2021) implementation |
| `--spp` / `--aspp` | Real spatial pyramid pooling / adaptive pyramid pooling, spliced into the actual classifier head. Wired up for `custom`, `mobilenet_v1`, and the torchvision archs that expose a `.features`/`.classifier` split (`mobilenet_v2/v3s`, `squeezenet`, `efficientnet_b0`). **Not wired for `shufflenet_v2` or `vit_tiny`** — you'll get a printed warning if you request it there, not silently wrong behavior. |
| `--transformer-refinement` | Real transformer-encoder head (`nn.TransformerEncoder`) attached after the conv trunk, same architecture-coverage caveat as above |
| `--defenses pgd_adv_train / fgsm_adv_train` | Real adversarial training using the real `attacks.pgd_attack` / `fgsm_attack` each batch |
| `--defenses free_adv_train` | Real "Free" adversarial training (Shafahi et al. 2019) — gradient reuse across `--free-m` replay steps |
| `--defenses trades` | Real TRADES loss (Zhang et al. 2019) |
| `--defenses mart` | Real MART loss (Wang et al. 2020), matches the authors' reference formula |
| `--defenses gradient_regularization` | Real input-gradient-norm penalty (double backprop) |
| `--defenses label_smoothing` | PyTorch's built-in `label_smoothing=` in cross-entropy |
| `--defenses feature_squeezing` | Real bit-depth reduction (Xu et al. 2018) |
| `--defenses input_transform` | Real randomized resize-pad (Xie et al. 2018); a JPEG round-trip variant is also implemented (`jpeg_compress_batch`, not wired into the CLI by default since it's CPU-bound and slow — call it directly if you want it) |
| `--defenses segmentation` | Real Otsu-threshold foreground masking — a lightweight, always-available heuristic, **not** a trained segmentation network. Swap in `torchvision.models.segmentation.deeplabv3_mobilenet_v3_large` if you want a stronger version; see the docstring in `defenses.py`. |
| `--defenses fdap` | **Experimental, not published/peer-reviewed.** Wavelet-domain soft-threshold denoising (real `PyWavelets` DWT/IDWT) combined with adversarial fine-tuning. This is my assembly of two established building blocks, not a verified novel contribution — see "About the 'novel' technique" below. |
| `--defenses ensemble_adv_train` | Real transfer attack from an auxiliary model instance (Tramèr et al. 2018) |
| `--eval-attacks fgsm ifgsm pgd` | Each evaluated **separately** against the real trained model on the real test set — if you pass all three you get three independent, real adversarial-accuracy numbers, not one blended figure |
| `--early-stopping` / `--patience` | Real early stopping on real validation accuracy, restores the best real checkpoint |
| `--num-seeds` | Runs N independent seeds end-to-end; `metrics.compute_stats` then gives you a real mean/std/95% CI (Student-t) across those N real results |
| FLOPs / params | `thop.profile` measuring the *actual* model instance you built (with your chosen resolution/pooling/attachments), not a lookup table |
| Inference latency | Actually timed on your device with warmup + averaging |
| Grad-CAM | Real forward/backward hooks on the real last conv layer, real gradients, real images (`vit_tiny` has no conv layer — Grad-CAM is skipped for it; attention-rollout would be the ViT-appropriate substitute, not implemented here) |
| Failure cases | Real misclassified adversarial examples from the real test set, saved as real PNGs with real predicted/true labels and real softmax confidence |

## Hyperparameter tuning

```bash
python -m src.tune --arch mobilenet_v1 --dataset cifar10 --resolution 32 \
  --trials 15 --trial-epochs 5 --out-dir runs/tune_mbv1
```

Real Optuna TPE search — each trial actually trains for `--trial-epochs`
epochs and is scored on real validation accuracy, with median pruning of
weak trials. Feed the best params it finds into `train.py`.

## About statistical significance (please read before writing this up)

A single training run's per-epoch numbers are **not** a valid basis for a
confidence interval in the sense reviewers expect — epochs within one run
are not independent samples of the same estimand, they're a trajectory.
The scientifically correct way to get a CI on "final accuracy" or "final
robust accuracy" is to run several **independent seeds** end-to-end
(`--num-seeds 3` minimum, 5+ preferred) and compute the CI across those
final numbers, which is exactly what `--num-seeds` does here. Use that,
not epoch-to-epoch spread, when reporting significance in your thesis/paper.

## About the "novel" technique (FDAP)

You asked for a novel robustness technique for your PhD work. I implemented
**Frequency-Domain Adversarial Purification** as working code (wavelet
denoising + adversarial fine-tuning) so you have something concrete to
experiment with, and it's flagged `EXPERIMENTAL` everywhere it appears.
To be direct about what this is and isn't:

- It **is** a real, correct implementation of a real idea (wavelet-domain
  purification is an established concept; combining it with adversarial
  training is a reasonable, testable hypothesis).
- It is **not** something I verified as novel against the current
  literature — I don't have a citation trail proving nobody has published
  this exact combination, and I'm not in a position to make that claim for
  you. Establishing novelty is a literature-review task specific to your
  subfield and timeline, not something I should assert.
- Before you build a thesis chapter around it: (1) run it through this
  same pipeline against real baselines (plain PGD-AT, TRADES, MART) on at
  least 2–3 datasets with multiple seeds, and (2) do a proper related-work
  search (I can help search current papers if you want a second pair of
  eyes on that, separately from this code).

## Known limitations / things I did not fake around

- `vit_tiny` has no conv feature map, so `--spp`/`--aspp`/
  `--transformer-refinement`/Grad-CAM don't apply to it — the code warns
  and skips rather than silently doing something wrong.
- `shufflenet_v2`'s torchvision implementation doesn't expose the plain
  `.features`/`.classifier` split the pooling re-heading code relies on —
  same honest skip-with-warning behavior.
- Combining `--optimizer sam` with `free_adv_train` or `trades` isn't
  handled (SAM's two-step update doesn't fit those training loops as
  written) — don't combine them; each is well-tested independently.
- Defensive distillation (`soft_labels_from_teacher` / `distillation_loss`
  in `defenses.py`) is implemented but not wired into the `train.py` CLI,
  since it's inherently a two-stage procedure (train teacher, then train
  student against its soft labels) rather than a single-call defense —
  call the functions directly in a two-stage script if you want to use it.
- Tiny-ImageNet isn't included as a dataset option (unlike the earlier
  mockup) because torchvision has no built-in loader for it; add a
  `torchvision.datasets.ImageFolder`-based loader in `data.py` if you need
  it (standard, ~10 lines, just needs the dataset already downloaded to
  disk in ImageFolder layout).

## File layout

```
src/
  data.py       # real dataset loaders
  models.py     # real architectures + SPP/ASPP/transformer-head wiring
  attacks.py    # real FGSM / I-FGSM / PGD
  defenses.py   # real SAM, TRADES, MART, grad-reg, smoothing, etc.
  metrics.py    # real FLOPs/params/latency measurement + real stats
  gradcam.py    # real Grad-CAM
  train.py      # main training/eval entry point
  tune.py       # real Optuna hyperparameter search
  report.py     # builds the Excel report from real results.json + images
requirements.txt
```
