"""
Real model definitions.

- Lightweight CNNs come from torchvision.models (weights=None — trained from
  scratch on the small datasets here; ImageNet weights don't transfer
  meaningfully to 32-224px CIFAR/MNIST-scale images without care, so we don't
  fake pretrained accuracy).
- ViT-Tiny is implemented from scratch with nn.TransformerEncoderLayer
  (real PyTorch primitives, not a stub).
- SPP / Adaptive SPP are real spatial pyramid pooling modules you can splice
  in before the classifier head.
- NormalizedModel wraps (raw-pixel-space attacks) -> (dataset normalization)
  -> backbone, so FGSM/I-FGSM/PGD epsilon values are meaningful in true
  pixel units.
"""
import math
import torch
import torch.nn as nn
import torchvision.models as tvm


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class NormalizedModel(nn.Module):
    """Wrap a backbone so it accepts raw [0,1] pixel input."""
    def __init__(self, backbone, mean, std):
        super().__init__()
        self.normalize = Normalize(mean, std)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(self.normalize(x))


class SpatialPyramidPool2d(nn.Module):
    """Fixed-bin spatial pyramid pooling (He et al., 2015), concatenated."""
    def __init__(self, bins=(1, 2, 4)):
        super().__init__()
        self.bins = bins

    def forward(self, x):
        outs = []
        for b in self.bins:
            pooled = nn.functional.adaptive_avg_pool2d(x, output_size=b)
            outs.append(pooled.flatten(1))
        return torch.cat(outs, dim=1)

    def out_dim(self, channels):
        return sum(b * b for b in self.bins) * channels


class AdaptivePyramidPool2d(nn.Module):
    """
    Bin sizes scale with input resolution (fraction of feature-map size)
    rather than being fixed, so pooling adapts as --resolution changes.
    """
    def __init__(self, fractions=(1.0, 0.5, 0.25)):
        super().__init__()
        self.fractions = fractions

    def forward(self, x):
        h = x.shape[-1]
        outs = []
        for f in self.fractions:
            b = max(1, round(h * f))
            pooled = nn.functional.adaptive_avg_pool2d(x, output_size=b)
            outs.append(pooled.flatten(1))
        return torch.cat(outs, dim=1)

    def out_dim(self, channels, feature_map_size):
        total = 0
        for f in self.fractions:
            b = max(1, round(feature_map_size * f))
            total += b * b
        return total * channels


def _replace_first_conv(module, in_channels):
    """Find the first Conv2d in a model and replace it to accept in_channels."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            new_conv = nn.Conv2d(in_channels, child.out_channels, kernel_size=child.kernel_size,
                                  stride=child.stride, padding=child.padding, bias=(child.bias is not None))
            setattr(module, name, new_conv)
            return True
        if _replace_first_conv(child, in_channels):
            return True
    return False


class TinyViT(nn.Module):
    """
    Minimal Vision Transformer built from real nn.TransformerEncoderLayer
    blocks. Patchifies the input, adds a class token + learned positional
    embeddings, runs through a small transformer encoder, classifies from
    the class token.
    """
    def __init__(self, image_size=32, patch_size=4, in_channels=3, num_classes=10,
                 dim=192, depth=6, heads=3, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                  # B, dim, H', W'
        x = x.flatten(2).transpose(1, 2)          # B, N, dim
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


class TransformerEncoderHead(nn.Module):
    """
    Optional add-on: takes a conv backbone's feature map, flattens spatial
    positions into a token sequence, refines with a small real transformer
    encoder, then global-pools for the classifier. Used when
    --transformer-refinement is enabled on a conv backbone.
    """
    def __init__(self, channels, num_classes, depth=2, heads=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=heads, dim_feedforward=channels * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, feat_map):
        B, C, H, W = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)   # B, H*W, C
        tokens = self.encoder(tokens)
        pooled = tokens.mean(dim=1)
        return self.classifier(pooled)


class CustomConvNet(nn.Module):
    """Small 3-conv-layer baseline, real and simple."""
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


ARCH_CHOICES = [
    "mobilenet_v1",  # approximated via torchvision's mobilenet_v3_large-style depthwise stem is not exact v1;
                      # for true MobileNetV1 we implement a compact depthwise-separable stack below.
    "mobilenet_v2", "mobilenet_v3s", "squeezenet", "shufflenet_v2",
    "efficientnet_b0", "custom", "vit_tiny",
]


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class MobileNetV1(nn.Module):
    """Real, compact MobileNetV1-style depthwise-separable stack (width mult 1.0)."""
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        def block(i, o, s=1): return DepthwiseSeparableConv(i, o, s)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.body = nn.Sequential(
            block(32, 64), block(64, 128, 2), block(128, 128),
            block(128, 256, 2), block(256, 256), block(256, 512, 2),
            *[block(512, 512) for _ in range(5)],
            block(512, 1024, 2), block(1024, 1024),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_backbone(arch, in_channels, num_classes, resolution):
    """Returns (backbone_module, last_conv_channels_or_None_if_transformer)."""
    arch = arch.lower()
    if arch == "mobilenet_v1":
        return MobileNetV1(in_channels, num_classes), None
    if arch == "mobilenet_v2":
        m = tvm.mobilenet_v2(weights=None, num_classes=num_classes)
        if in_channels != 3:
            _replace_first_conv(m, in_channels)
        return m, None
    if arch == "mobilenet_v3s":
        m = tvm.mobilenet_v3_small(weights=None, num_classes=num_classes)
        if in_channels != 3:
            _replace_first_conv(m, in_channels)
        return m, None
    if arch == "squeezenet":
        m = tvm.squeezenet1_1(weights=None, num_classes=num_classes)
        if in_channels != 3:
            _replace_first_conv(m, in_channels)
        return m, None
    if arch == "shufflenet_v2":
        m = tvm.shufflenet_v2_x1_0(weights=None, num_classes=num_classes)
        if in_channels != 3:
            _replace_first_conv(m, in_channels)
        return m, None
    if arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=None, num_classes=num_classes)
        if in_channels != 3:
            _replace_first_conv(m, in_channels)
        return m, None
    if arch == "custom":
        return CustomConvNet(in_channels, num_classes), None
    if arch == "vit_tiny":
        patch = 4 if resolution % 4 == 0 else 2
        return TinyViT(image_size=resolution, patch_size=patch, in_channels=in_channels,
                        num_classes=num_classes), None
    raise ValueError(f"Unknown arch '{arch}'. Options: {ARCH_CHOICES}")


# Architectures whose torchvision implementation exposes a plain
# `.features` conv trunk + separate classifier, which we can generically
# re-head with pyramid pooling / a transformer-encoder head. ShuffleNetV2
# uses a different internal structure and is NOT covered by this — if you
# request SPP/ASPP/transformer-refinement with shufflenet_v2 you'll get a
# clear warning and the architecture's default head instead of silently
# wrong behaviour.
FEATURES_CLASSIFIER_ARCHS = {"mobilenet_v2", "mobilenet_v3s", "squeezenet", "efficientnet_b0"}


def _rehead_features_classifier_model(backbone, arch, num_classes, in_channels, resolution,
                                       pooling="none", use_transformer_head=False):
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, resolution, resolution)
        feat = backbone.features(dummy)
    channels, feat_size = feat.shape[1], feat.shape[-1]

    if use_transformer_head:
        head = TransformerEncoderHead(channels, num_classes)
        def new_forward(self, x):
            f = self.features(x)
            return self.transformer_head(f)
    elif pooling == "spp":
        bins = (1, 2, 4) if feat_size >= 4 else (1, 2)
        pool = SpatialPyramidPool2d(bins)
        head = nn.Linear(pool.out_dim(channels), num_classes)
        def new_forward(self, x, _pool=pool):
            f = self.features(x)
            return self.classifier(_pool(f))
    elif pooling == "aspp":
        pool = AdaptivePyramidPool2d((1.0, 0.5, 0.25))
        head = nn.Linear(pool.out_dim(channels, feat_size), num_classes)
        def new_forward(self, x, _pool=pool):
            f = self.features(x)
            return self.classifier(_pool(f))
    else:
        return backbone  # default architecture head, untouched

    backbone.classifier = head
    if use_transformer_head:
        backbone.transformer_head = head
    import types
    backbone.forward = types.MethodType(new_forward, backbone)
    return backbone


def _rehead_custom_or_mbv1(backbone, num_classes, pooling="none", use_transformer_head=False,
                            channels=128, feat_size=None):
    if use_transformer_head:
        head = TransformerEncoderHead(channels, num_classes)
        def new_forward(self, x):
            f = self.features(x) if hasattr(self, "features") else self.body(self.stem(x))
            return self._transformer_head(f)
        backbone._transformer_head = head
    elif pooling == "spp":
        bins = (1, 2, 4)
        pool = SpatialPyramidPool2d(bins)
        backbone.classifier = nn.Linear(pool.out_dim(channels), num_classes)
        def new_forward(self, x, _pool=pool):
            f = self.features(x) if hasattr(self, "features") else self.body(self.stem(x))
            return self.classifier(_pool(f))
    elif pooling == "aspp":
        pool = AdaptivePyramidPool2d((1.0, 0.5, 0.25))
        fs = feat_size or 4
        backbone.classifier = nn.Linear(pool.out_dim(channels, fs), num_classes)
        def new_forward(self, x, _pool=pool):
            f = self.features(x) if hasattr(self, "features") else self.body(self.stem(x))
            return self.classifier(_pool(f))
    else:
        return backbone
    import types
    backbone.forward = types.MethodType(new_forward, backbone)
    return backbone


def build_model(arch, dataset_channels, num_classes, resolution, mean, std,
                 pooling="none", use_transformer_head=False):
    """
    pooling: 'none' | 'spp' | 'aspp'
    use_transformer_head: attach a real transformer-encoder head after the
        conv trunk (mutually exclusive with pooling — if both are set,
        transformer_head takes precedence and pooling is ignored).
    """
    arch_l = arch.lower()
    backbone, _ = build_backbone(arch, dataset_channels, num_classes, resolution)

    if arch_l == "vit_tiny" and (pooling != "none" or use_transformer_head):
        print(f"[models] Note: pooling/transformer-head options don't apply to vit_tiny "
              f"(it has no conv feature map) — ignoring for this run.")
    elif arch_l == "shufflenet_v2" and (pooling != "none" or use_transformer_head):
        print(f"[models] Warning: SPP/ASPP/transformer-refinement are not wired up for "
              f"shufflenet_v2 in this reference implementation — using its default head.")
    elif arch_l in FEATURES_CLASSIFIER_ARCHS:
        backbone = _rehead_features_classifier_model(
            backbone, arch_l, num_classes, dataset_channels, resolution, pooling, use_transformer_head)
    elif arch_l in ("custom", "mobilenet_v1"):
        channels = 128 if arch_l == "custom" else 1024
        feat_size = max(1, resolution // (8 if arch_l == "custom" else 32))
        backbone = _rehead_custom_or_mbv1(backbone, num_classes, pooling, use_transformer_head,
                                          channels=channels, feat_size=feat_size)

    return NormalizedModel(backbone, mean, std)


def find_last_conv_layer(model):
    """Utility for Grad-CAM: returns the last nn.Conv2d module in the model."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last
