"""Aesthetic quality scoring: CLIP image embedding -> LAION aesthetic head.

No Qt, no app dependencies, same rule as `tamis.recognition`'s own modules --
`tamis.quality.worker` is the seam where this gets wired into QThreadPool.

Why this model. Measured against five alternatives on a 455-photo unculled
folder (Tier-0 Laplacian/histogram statistics, NIMA, CLIP-IQA+, TOPIQ,
MUSIQ), the metrics split into two families: *technical* ones detect defects
but flatten out once a photo is merely fine, while *aesthetic* ones keep
discriminating. Over the 377 technically-fine photos in that folder the
spread was 0.50 standard deviations for a sharpness metric versus 0.91 for
this one -- which is the half of the problem a blur detector structurally
cannot solve. NIMA scored marginally better still (0.94) and is five times
cheaper, but the only weights readily available for it come from a toolbox
under a non-commercial licence, which cannot ship in a GPLv3 app. This
model's parts are MIT (CLIP) and Apache-2.0 (the aesthetic head).

Cost on an RTX 5070: ~80ms per photo end to end, of which only ~23ms is the
network -- the rest is JPEG decode, which the caller can share with whatever
else it already decodes.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from tamis.quality.store import MODEL_ID

logger = logging.getLogger(__name__)

# ViT-L/14 rather than the smaller ViT-B/32: B/32 is a 350MB download against
# 1.7GB and 8x cheaper to run, and discriminates among good photos just as
# well (0.90 vs 0.91 spread). But it ranks *differently* -- Spearman 0.77
# against L/14, which is a different opinion rather than a cheaper
# approximation of the same one. L/14 is what was evaluated by eye, so it is
# what ships. Switching is this constant plus the matching head below.
#
# The "-quickgelu" suffix is load-bearing, not decoration. OpenAI's CLIP was
# trained with QuickGELU activations, while open_clip's bare "ViT-L-14" config
# defaults to standard GELU -- so pairing that config with pretrained="openai"
# loads the right weights into the wrong architecture. open_clip warns and
# carries on, so it fails silently and produces plausible scores from a
# consistently different model: measured at Spearman 0.93 against the correct
# pairing, with 75% of scores shifted by more than 2 points out of 100.
# Any change here must also bump store.MODEL_ID.
CLIP_MODEL = "ViT-L-14-quickgelu"
CLIP_PRETRAINED = "openai"

# The "improved aesthetic predictor" MLP head (Apache-2.0), trained on AVA +
# SAC + LOGOS human ratings. Cached next to the app's other state rather than
# vendored, since it is fetched once and CLIP's own weights already download
# on first use.
HEAD_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor"
    "/raw/main/sac+logos+ava1-l14-linearMSE.pth"
)
HEAD_CACHE = Path.home() / ".tamis" / "aesthetic-head-l14.pth"
EMBED_DIM = 768

# Raw model output is roughly 1-10, but real photographs cluster tightly:
# 3.3-6.2 across a 455-photo folder. Mapping this window to 0-100 keeps the
# displayed score stable across folders (a percentile would shift every time
# the folder changes) while still using most of the range.
RAW_MIN, RAW_MAX = 3.0, 7.0


class _AestheticHead(nn.Module):
    """The published head's exact shape -- the indices matter, because the
    checkpoint's keys are positional (`layers.0`, `layers.2`, ...), so the
    dropouts have to stay even though they are inert in eval mode."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(EMBED_DIM, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16), nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


_model = None
_preprocess = None
_head = None
_device = None


def _ensure_head_downloaded() -> Path:
    if HEAD_CACHE.exists():
        return HEAD_CACHE
    HEAD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading aesthetic head to %s", HEAD_CACHE)
    # Via a temp file: an interrupted download must not leave a truncated
    # checkpoint that then fails to load forever after.
    temp = HEAD_CACHE.with_suffix(".partial")
    urllib.request.urlretrieve(HEAD_URL, temp)  # noqa: S310 - fixed https URL
    temp.replace(HEAD_CACHE)
    return HEAD_CACHE


def _allow_hub_downloads() -> None:
    """Re-enable huggingface_hub's network access, for the one-time download.

    The environment variable alone is not enough once huggingface_hub has been
    imported: it reads HF_HUB_OFFLINE into a module constant at import time,
    and its callers read that constant rather than the environment.
    """
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        from huggingface_hub import constants as hub_constants

        hub_constants.HF_HUB_OFFLINE = False
    except Exception:  # pragma: no cover - depends on huggingface_hub internals
        logger.warning("Could not re-enable huggingface_hub downloads")


def _load_clip() -> tuple:
    """Load CLIP from the local cache, reaching the network only if it is not
    cached yet.

    open_clip resolves pretrained="openai" to a HuggingFace repo rather than
    the original URL, and huggingface_hub re-checks the remote revision every
    time a model is loaded -- so a fully cached model still contacted
    huggingface.co on each start. No image data is involved, but "runs
    entirely locally" should be true rather than nearly true.

    Offline is only the *default*: a user who has set HF_HUB_OFFLINE
    themselves is left alone, in either direction.
    """
    chosen_by_user = "HF_HUB_OFFLINE" in os.environ
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import open_clip  # imported here so a missing extra fails at use, not import

    try:
        return open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    except Exception:
        if chosen_by_user:
            raise  # they asked for offline; do not quietly override them
        logger.info("CLIP weights are not cached yet; downloading once from HuggingFace")
        _allow_hub_downloads()
        return open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)


def _load() -> tuple:
    """Build the model on first use. Kept lazy for the same reason the
    recognition models are: importing this module must not pull ~2GB of
    weights into memory for a user who never opens the panel."""
    global _model, _preprocess, _head, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = _load_clip()
        head = _AestheticHead()
        head.load_state_dict(torch.load(_ensure_head_downloaded(), map_location="cpu", weights_only=True))
        _model = model.eval().to(_device)
        _preprocess = preprocess
        _head = head.eval().to(_device)
        logger.info("Aesthetic scorer ready on %s", _device)
    return _model, _preprocess, _head, _device


def model_id() -> str:
    """Identity of the scorer that produced a value, recorded in the sidecar
    so cached scores from a different model are recomputed rather than mixed
    with new ones. Kept in `store` so it can be read without importing torch."""
    return MODEL_ID


def to_display_score(raw: float) -> int:
    """Map a raw aesthetic value onto the 0-100 shown in the UI."""
    return int(round(float(np.clip((raw - RAW_MIN) / (RAW_MAX - RAW_MIN), 0.0, 1.0)) * 100))


def score_images(images: list[Image.Image]) -> list[int]:
    """Score a batch of PIL images, returning one 0-100 value each.

    Batched deliberately: measured per-image on this hardware, a forward pass
    costs ~23ms, but 0.34ms when 16 are submitted together. The caller is
    expected to accumulate a batch rather than call this per photo.
    """
    if not images:
        return []
    model, preprocess, head, device = _load()
    batch = torch.stack([preprocess(im.convert("RGB")) for im in images]).to(device)
    with torch.no_grad():
        features = model.encode_image(batch).float()
        # L2-normalise before the head: the published predictor was trained on
        # normalised CLIP embeddings, and skipping this silently shifts every
        # score.
        features = features / features.norm(dim=-1, keepdim=True)
        raw = head(features).squeeze(-1).cpu().numpy()
        if device == "cuda":
            # Same reasoning as the recognition models: release the peak so a
            # session's GPU footprint doesn't ratchet upward.
            torch.cuda.empty_cache()
    return [to_display_score(v) for v in np.atleast_1d(raw)]


def load_for_scoring(path: Path) -> Image.Image:
    """Decode `path` small enough for the model and no smaller.

    `draft()` scales in the JPEG DCT domain, so asking for a size near the
    model's 224px input makes the decode several times cheaper than a full
    one (~40ms versus ~157ms for a 10MP photo) with no effect on the score,
    since `preprocess` would resize to 224 regardless.
    """
    with Image.open(path) as image:
        image.draft("RGB", (448, 448))
        from PIL import ImageOps

        return ImageOps.exif_transpose(image).convert("RGB")
