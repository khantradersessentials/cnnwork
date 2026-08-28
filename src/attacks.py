"""
Real gradient-based adversarial attacks (L_inf).

All functions assume `model` accepts raw [0,1] pixel input (i.e. a
models.NormalizedModel instance, which normalizes internally) and expect
`x` already in [0,1]. Epsilon and step size are in true pixel units
(e.g. eps=8/255).
"""
import torch
import torch.nn.functional as F


def fgsm_attack(model, x, y, eps, loss_fn=None):
    """Single-step Fast Gradient Sign Method (Goodfellow et al., 2015)."""
    loss_fn = loss_fn or F.cross_entropy
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = loss_fn(logits, y)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = x + eps * grad.sign()
    return x_adv.clamp(0, 1).detach()


def ifgsm_attack(model, x, y, eps, alpha=None, steps=10, loss_fn=None):
    """
    Iterative FGSM / Basic Iterative Method (Kurakin et al., 2016).
    No random start; each step is clipped back into the epsilon L_inf ball
    around the original input.
    """
    loss_fn = loss_fn or F.cross_entropy
    alpha = alpha if alpha is not None else eps / max(steps, 1) * 1.25
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = loss_fn(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = x_adv.clamp(0, 1)
    return x_adv.detach()


def pgd_attack(model, x, y, eps, alpha=None, steps=20, random_start=True, loss_fn=None):
    """
    Projected Gradient Descent (Madry et al., 2018). Strongest of the three
    first-order attacks here due to the random start + more steps by default.
    """
    loss_fn = loss_fn or F.cross_entropy
    alpha = alpha if alpha is not None else eps / max(steps, 1) * 2.5
    x_orig = x.clone().detach()
    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0, 1).detach()
    else:
        x_adv = x_orig.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = loss_fn(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = x_adv.clamp(0, 1)
    return x_adv.detach()


ATTACKS = {
    "fgsm": fgsm_attack,
    "ifgsm": ifgsm_attack,
    "pgd": pgd_attack,
}


@torch.no_grad()
def evaluate_clean_accuracy(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def evaluate_adversarial_accuracy(model, loader, device, attack_name, eps, steps=20):
    """
    Real per-attack accuracy — runs the actual attack against the actual
    model on the actual held-out set. No estimation.
    """
    model.eval()
    correct, total = 0, 0
    attack_fn = ATTACKS[attack_name]
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if attack_name == "fgsm":
            x_adv = attack_fn(model, x, y, eps)
        else:
            x_adv = attack_fn(model, x, y, eps, steps=steps)
        with torch.no_grad():
            logits = model(x_adv)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total
