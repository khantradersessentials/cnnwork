"""
Real defense implementations. Each is a faithful (if sometimes simplified)
implementation of a published technique — cited in each docstring — plus
one clearly-marked experimental technique of our own assembly (FDAP).

IMPORTANT: implementing a technique correctly is not the same as it being
*novel* or *validated*. Anything marked EXPERIMENTAL below should be run
through the same evaluation pipeline as everything else here and compared
against a real literature search before you claim novelty in a paper.
"""
import io
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta as beta_dist, norm as normal_dist


# ---------------------------------------------------------------------------
# SAM — Sharpness-Aware Minimization (Foret et al., ICLR 2021)
# ---------------------------------------------------------------------------
class SAM(torch.optim.Optimizer):
    """
    Wraps a base optimizer (e.g. SGD). Requires two forward/backward passes
    per step:
        loss = criterion(model(x), y); loss.backward()
        optimizer.first_step(zero_grad=True)
        criterion(model(x), y).backward()
        optimizer.second_step(zero_grad=True)
    """
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        norms = [p.grad.norm(p=2) for group in self.param_groups
                  for p in group["params"] if p.grad is not None]
        return torch.norm(torch.stack(norms), p=2) if norms else torch.tensor(0.0)


# ---------------------------------------------------------------------------
# TRADES (Zhang et al., ICML 2019)
# ---------------------------------------------------------------------------
def trades_loss(model, x_natural, y, optimizer, step_size=0.007, epsilon=8/255,
                 perturb_steps=10, beta=6.0):
    """Returns the scalar TRADES loss for one batch (call loss.backward() yourself)."""
    model.eval()
    batch_size = x_natural.size(0)
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural)
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                          F.softmax(model(x_natural), dim=1), reduction="sum")
        grad = torch.autograd.grad(kl, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon).clamp(0, 1)
    model.train()

    logits_natural = model(x_natural)
    logits_adv = model(x_adv)
    loss_natural = F.cross_entropy(logits_natural, y)
    loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1),
                            F.softmax(logits_natural, dim=1), reduction="batchmean")
    return loss_natural + beta * loss_robust


# ---------------------------------------------------------------------------
# MART (Wang et al., ICLR 2020) — faithful to the authors' reference code
# ---------------------------------------------------------------------------
def mart_loss(model, x_natural, y, x_adv, beta=5.0):
    """x_adv should already be generated (e.g. via attacks.pgd_attack)."""
    logits_adv = model(x_adv)
    logits_nat = model(x_natural)

    adv_probs = F.softmax(logits_adv, dim=1)
    top2 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(top2[:, -1] == y, top2[:, -2], top2[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits_nat, dim=1)
    true_probs = torch.gather(nat_probs, 1, y.unsqueeze(1)).squeeze(1)
    kl = torch.sum(F.kl_div(F.log_softmax(logits_adv, dim=1), nat_probs, reduction="none"), dim=1)
    loss_robust = torch.mean(kl * (1.0000001 - true_probs))

    return loss_adv + beta * loss_robust


# ---------------------------------------------------------------------------
# Gradient regularization (Ross & Doshi-Velez, AAAI 2018)
# ---------------------------------------------------------------------------
def gradient_regularization_loss(model, x, y, lam=1.0):
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    ce = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(ce, x, create_graph=True)[0]
    penalty = grad.flatten(1).norm(p=2, dim=1).pow(2).mean()
    return ce + lam * penalty


# ---------------------------------------------------------------------------
# Randomized smoothing (Cohen, Rosenfeld & Kolter, ICML 2019)
# ---------------------------------------------------------------------------
@torch.no_grad()
def smoothed_predict(model, x, sigma=0.25, n_samples=100, batch_size=100, num_classes=10):
    device = x.device
    counts = torch.zeros(num_classes, device=device)
    remaining = n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        x_rep = x.repeat(b, 1, 1, 1)
        noise = torch.randn_like(x_rep) * sigma
        preds = model((x_rep + noise).clamp(0, 1)).argmax(1)
        counts += torch.bincount(preds, minlength=num_classes).float()
        remaining -= b
    return counts.argmax().item(), counts


def certified_radius(n_a, n_total, sigma, alpha=0.001):
    """
    Clopper-Pearson lower confidence bound on p_A, then the certified L2
    radius r = sigma * Phi^-1(p_A_lower)  (Cohen et al., 2019, Eq. 3).
    Returns 0.0 if not certifiable at this confidence level.
    """
    if n_a == 0:
        return 0.0
    p_lower = beta_dist.ppf(alpha, n_a, n_total - n_a + 1) if n_a < n_total else 1.0 - alpha
    if p_lower <= 0.5:
        return 0.0
    return float(sigma * normal_dist.ppf(p_lower))


# ---------------------------------------------------------------------------
# Feature squeezing (Xu, Evans & Qi, NDSS 2018)
# ---------------------------------------------------------------------------
def feature_squeeze(x, bit_depth=4):
    levels = 2 ** bit_depth - 1
    return torch.round(x * levels) / levels


# ---------------------------------------------------------------------------
# Input transformation defense — JPEG round-trip / resize-pad
# ---------------------------------------------------------------------------
def jpeg_compress_batch(x, quality=75):
    """Real JPEG round-trip per image via PIL (CPU, slower — use sparingly)."""
    from PIL import Image
    import numpy as np
    device = x.device
    out = []
    for img in x.detach().cpu():
        arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
        pil_img = Image.fromarray(arr.squeeze() if arr.shape[-1] == 1 else arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recon = Image.open(buf).convert(pil_img.mode)
        recon_arr = np.array(recon).astype("float32") / 255.0
        if recon_arr.ndim == 2:
            recon_arr = recon_arr[:, :, None]
        out.append(torch.from_numpy(recon_arr).permute(2, 0, 1))
    return torch.stack(out).to(device)


def resize_pad_defense(x, shrink=0.9):
    """Randomized resize-then-pad (Xie et al., ICLR 2018)."""
    B, C, H, W = x.shape
    new_size = max(1, int(H * shrink))
    resized = F.interpolate(x, size=(new_size, new_size), mode="bilinear", align_corners=False)
    pad_total = H - new_size
    pad_left = torch.randint(0, pad_total + 1, (1,)).item() if pad_total > 0 else 0
    pad_right = pad_total - pad_left
    return F.pad(resized, (pad_left, pad_right, pad_left, pad_right))


# ---------------------------------------------------------------------------
# Defensive distillation (Papernot et al., 2016)
# ---------------------------------------------------------------------------
def soft_labels_from_teacher(teacher, x, temperature=20.0):
    teacher.eval()
    with torch.no_grad():
        return F.softmax(teacher(x) / temperature, dim=1)


def distillation_loss(student_logits, soft_labels, temperature=20.0):
    log_probs = F.log_softmax(student_logits / temperature, dim=1)
    return -(soft_labels * log_probs).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Segmentation-guided input purification — Otsu foreground mask
# (lightweight, always-available; swap in a real pretrained segmentation
#  network, e.g. torchvision's deeplabv3_mobilenet_v3_large, for a
#  stronger version if your compute budget allows)
# ---------------------------------------------------------------------------
def _otsu_threshold(gray_flat):
    hist = torch.histc(gray_flat, bins=256, min=0.0, max=1.0)
    total = gray_flat.numel()
    sum_all = torch.sum(hist * torch.arange(256, device=hist.device, dtype=hist.dtype))
    sum_b, w_b, best_thresh, best_var = 0.0, 0.0, 0, -1.0
    for t in range(256):
        w_b += hist[t].item()
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t].item()
        m_b = sum_b / w_b
        m_f = (sum_all.item() - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = t
    return best_thresh / 255.0


def segmentation_purify(x):
    """Zeroes out background pixels per-image using an Otsu-thresholded mask."""
    out = x.clone()
    for i in range(x.shape[0]):
        gray = x[i].mean(dim=0)
        thresh = _otsu_threshold(gray.flatten())
        mask = (gray > thresh).float().unsqueeze(0)
        out[i] = x[i] * mask
    return out


# ---------------------------------------------------------------------------
# EXPERIMENTAL — Frequency-Domain Adversarial Purification (FDAP)
# Assembled from established building blocks (wavelet-domain soft-
# thresholding denoising + adversarial fine-tuning). This is NOT a
# published, peer-reviewed technique — treat it as a starting point for
# your own research, not a citable baseline. Requires PyWavelets (pywt).
# ---------------------------------------------------------------------------
def frequency_domain_purify(x, wavelet="haar", level=1, threshold=0.04):
    import pywt
    import numpy as np
    device = x.device
    out = torch.empty_like(x)
    x_np = x.detach().cpu().numpy()
    for i in range(x_np.shape[0]):
        for c in range(x_np.shape[1]):
            coeffs = pywt.wavedec2(x_np[i, c], wavelet, level=level)
            coeffs = [coeffs[0]] + [
                tuple(pywt.threshold(d, threshold, mode="soft") for d in detail)
                for detail in coeffs[1:]
            ]
            recon = pywt.waverec2(coeffs, wavelet)
            recon = recon[: x_np.shape[2], : x_np.shape[3]]
            out[i, c] = torch.from_numpy(recon)
    return out.clamp(0, 1).to(device)


# ---------------------------------------------------------------------------
# Free adversarial training (Shafahi et al., NeurIPS 2019)
# ---------------------------------------------------------------------------
def free_adv_train_step(model, optimizer, x, y, eps, m=4):
    """
    Reuses the gradient across m mini-steps on the same batch, updating both
    the perturbation and the model weights each mini-step (roughly m times
    cheaper than PGD adversarial training for similar robustness).
    Returns the mean loss over the m mini-steps.
    """
    delta = torch.zeros_like(x, requires_grad=True)
    total_loss = 0.0
    for _ in range(m):
        x_adv = (x + delta).clamp(0, 1)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        grad = delta.grad.detach()
        optimizer.step()
        delta = (delta.detach() + eps * grad.sign()).clamp(-eps, eps)
        delta.requires_grad_(True)
        total_loss += loss.item()
    return total_loss / m


# ---------------------------------------------------------------------------
# Ensemble adversarial training (Tramèr et al., ICLR 2018)
# ---------------------------------------------------------------------------
def ensemble_adv_examples(aux_models, x, y, eps, attack_fn):
    """Craft perturbations on a randomly chosen held-out model, transfer to x."""
    import random
    aux = random.choice(aux_models)
    aux.eval()
    return attack_fn(aux, x, y, eps)
