"""2D segmentation backbones: SegFormer and HRNetV2 wrappers.

See docs/sota.md Part A for why both are worth baselining: SegFormer is fast and
proven on UAVid; HRNetV2 keeps a full-resolution branch throughout the network,
which the literature credits with better thin-structure preservation -- relevant
given the powerline class is 1-5 px wide.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerWrapper(nn.Module):
    """Wraps HuggingFace's SegFormer for the project's class taxonomy."""

    def __init__(self, num_classes: int, variant: str = "nvidia/segformer-b2-finetuned-ade-512-512",
                 pretrained: bool = True):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            variant,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        ) if pretrained else SegformerForSemanticSegmentation(
            SegformerForSemanticSegmentation.config_class(num_labels=num_classes)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h, w = images.shape[-2:]
        logits = self.model(pixel_values=images).logits  # (B, C, H/4, W/4)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


class HRNetV2Wrapper(nn.Module):
    """HRNetV2 backbone (timm) + a lightweight per-pixel classification head.

    HRNet keeps parallel branches at multiple resolutions throughout the network
    instead of downsample-then-upsample; the classification head fuses all branch
    outputs at full resolution rather than only the coarsest one, preserving the
    thin-structure detail the powerline class needs (see docs/sota.md Part A).
    """

    def __init__(self, num_classes: int, variant: str = "hrnet_w32", pretrained: bool = True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            variant, pretrained=pretrained, features_only=True,
        )
        feat_channels = sum(f["num_chs"] for f in self.backbone.feature_info)
        self.head = nn.Sequential(
            nn.Conv2d(feat_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h, w = images.shape[-2:]
        feats = self.backbone(images)
        target_size = feats[0].shape[-2:]
        fused = torch.cat(
            [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False) for f in feats],
            dim=1,
        )
        logits = self.head(fused)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


def build_model(cfg: dict) -> nn.Module:
    """Factory: cfg is the "model2d" section of configs/default.json."""
    backbone = cfg["backbone"]
    num_classes = cfg["num_classes"]
    pretrained = cfg.get("pretrained", True)

    if backbone.startswith("segformer"):
        variant = {
            "segformer_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
            "segformer_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
            "segformer_b5": "nvidia/segformer-b5-finetuned-ade-640-640",
        }.get(backbone, backbone)
        return SegFormerWrapper(num_classes, variant, pretrained)
    elif backbone.startswith("hrnet"):
        return HRNetV2Wrapper(num_classes, backbone, pretrained)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
