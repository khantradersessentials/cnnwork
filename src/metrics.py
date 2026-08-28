"""
Real measurement utilities — nothing here is looked up from a static table.
FLOPs/params are computed for the *actual* model instance you built (with
its actual resolution/pooling/attachments); latency is actually timed;
statistics are computed from actual repeated results.
"""
import time
import torch
import numpy as np
from scipy import stats


def compute_flops_params(model, input_shape, device="cpu"):
    """
    input_shape: (channels, H, W). Requires `pip install thop`.
    Returns (flops, params) as raw counts (not millions) — divide by 1e6
    yourself for MFLOPs/M-params in reports.
    """
    try:
        from thop import profile
    except ImportError as e:
        raise ImportError(
            "thop is required for real FLOPs/params measurement: pip install thop"
        ) from e
    model = model.to(device).eval()
    dummy = torch.randn(1, *input_shape, device=device)
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    return flops, params


def measure_inference_latency(model, input_shape, device="cpu", batch_size=1,
                               num_warmup=10, num_runs=100):
    """Actual wall-clock timing (ms/batch), not estimated."""
    model = model.to(device).eval()
    dummy = torch.randn(batch_size, *input_shape, device=device)
    with torch.no_grad():
        for _ in range(num_warmup):
            model(dummy)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            model(dummy)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
    times = np.array(times)
    return times.mean(), times.std()


def compute_stats(values, confidence=0.95):
    """
    Real mean / std / confidence interval across a list of actual results
    (e.g. final accuracy from N independent training runs with different
    seeds). Uses the Student-t distribution, which is the statistically
    correct choice for small sample sizes (few seeds) rather than a normal
    approximation.

    NOTE: computing a "confidence interval" across the epochs of a SINGLE
    training run (as one might be tempted to do) is not statistically
    meaningful — epochs are not independent samples of the same estimand.
    For a real CI on final accuracy/robustness, run several seeds
    (--num-seeds >= 3, ideally 5) and pass the list of final results here.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = arr.mean()
    std = arr.std(ddof=1) if n > 1 else 0.0
    if n > 1:
        margin = stats.t.ppf((1 + confidence) / 2, n - 1) * std / np.sqrt(n)
    else:
        margin = 0.0
    return {"mean": float(mean), "std": float(std), "n": n,
            "ci_low": float(mean - margin), "ci_high": float(mean + margin),
            "confidence": confidence}
