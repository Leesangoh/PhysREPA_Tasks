"""V-JEPA 2 ViT-L loader with monkey-patched forward to capture raw residual stream.

Pattern adapted from /home/solee/pez/step2_extract.py (forward_resid_pre):
  layer 0  = patch_embed output (pre-block 0)
  layer k  = post-block (k-1) for k = 1..23
giving exactly 24 outputs for ViT-L (depth=24). No final LayerNorm applied.
"""

from __future__ import annotations

import sys
from typing import Any

import torch

from .io import load_common


_VJEPA_SRC_ADDED = False


def _ensure_paths() -> None:
    global _VJEPA_SRC_ADDED
    if _VJEPA_SRC_ADDED:
        return
    common = load_common()
    src = common["vjepa2"]["source_path"]
    root = "/home/solee/vjepa2"
    if root not in sys.path:
        sys.path.insert(0, root)
    if src not in sys.path:
        sys.path.insert(0, src)
    _VJEPA_SRC_ADDED = True


def _forward_resid_pre(self, x, masks=None):
    """Capture residual stream at depth-many positions:
    [patch_embed, post_block_0, post_block_1, ..., post_block_{depth-2}].
    """
    if masks is not None and not isinstance(masks, list):
        masks = [masks]

    if x.ndim == 4:
        _, _, height, width = x.shape
        tubelets = 1
    else:
        _, _, tubelets, height, width = x.shape
        tubelets = tubelets // self.tubelet_size

    h_patches = height // self.patch_size
    w_patches = width // self.patch_size
    if not self.handle_nonsquare_inputs:
        tubelets = h_patches = w_patches = None

    if not self.use_rope:
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = self.patch_embed(x)
        x = x + pos_embed
    else:
        x = self.patch_embed(x)

    if masks is not None:
        from src.masks.utils import apply_masks  # noqa
        x = apply_masks(x, masks)
        masks = torch.cat(masks, dim=0)

    outs = [x.clone()]
    for block in self.blocks[:-1]:
        x = block(
            x,
            mask=masks,
            attn_mask=None,
            T=tubelets,
            H_patches=h_patches,
            W_patches=w_patches,
        )
        outs.append(x.clone())

    return outs


def load_vjepa2_vit_l(device: str | torch.device = "cuda:0") -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load V-JEPA 2 ViT-L weights with monkey-patched forward."""
    _ensure_paths()
    common = load_common()
    vj = common["vjepa2"]

    import models.vision_transformer as vit_module  # type: ignore

    factory = getattr(vit_module, vj["factory"])
    model = factory(
        patch_size=vj["patch_size"],
        img_size=(vj["input_size"], vj["input_size"]),
        num_frames=64,                       # PEZ-style: factory wants this; window length is set at forward
        tubelet_size=vj["tubelet_size"],
        out_layers=list(range(vj["num_layers"])),
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
    )

    ckpt = torch.load(vj["weights_path"], map_location="cpu", weights_only=True)
    state = ckpt.get("target_encoder", ckpt)
    cleaned = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)

    model.__class__.forward = _forward_resid_pre

    model = model.to(device).eval()
    spec = {
        "factory": vj["factory"],
        "depth": vj["num_layers"],
        "embed_dim": vj["d_model"],
        "input_size": vj["input_size"],
        "patch_size": vj["patch_size"],
        "tubelet_size": vj["tubelet_size"],
    }
    return model, spec


# Frame preprocessing — ImageNet normalization; resize 384 → 256.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_frames(frames_uint8: "torch.Tensor") -> "torch.Tensor":
    """Input: [T, H, W, 3] uint8 on any device. Output: [3, T, 256, 256] float16 on same device."""
    import torch.nn.functional as F

    x = frames_uint8.permute(0, 3, 1, 2).float() / 255.0  # [T, 3, H, W]
    if x.shape[-1] != 256 or x.shape[-2] != 256:
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.permute(1, 0, 2, 3).contiguous().to(torch.float16)  # [3, T, 256, 256]
