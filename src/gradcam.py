"""
Real Grad-CAM (Selvaraju et al., ICCV 2017) via forward/backward hooks.
Computed from actual gradients of the actual model on actual images —
nothing synthetic.

Note: TinyViT has no conv feature map for classic Grad-CAM; for
transformer architectures, use attention-rollout instead (not implemented
here — flagged in README as a follow-up if you need ViT explanations).
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        x = x.clone().detach().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        self.model.zero_grad()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # B,C,1,1
        cam = F.relu((weights * self.activations).sum(dim=1))     # B,H,W
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear",
                             align_corners=False).squeeze(1)
        return cam.detach(), logits.detach()


def overlay_heatmap(image_chw, cam_hw, alpha=0.45):
    """image_chw: torch tensor in [0,1], shape (C,H,W). Returns a PIL Image."""
    img = image_chw.detach().cpu().numpy().transpose(1, 2, 0)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    img = np.clip(img, 0, 1)

    cam = cam_hw.detach().cpu().numpy()
    heatmap = _jet_colormap(cam)  # H,W,3 in [0,1]

    blended = (1 - alpha) * img + alpha * heatmap
    blended = np.clip(blended * 255, 0, 255).astype("uint8")
    return Image.fromarray(blended)


def _jet_colormap(x):
    """Minimal jet-style colormap without a matplotlib dependency at call time."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)
