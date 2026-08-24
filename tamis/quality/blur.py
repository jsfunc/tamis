"""Focus scoring: how little a deliberate re-blur changes the image.

No torch and no Qt -- this is arithmetic over a decoded image, and it rides
along on a decode the caller has already paid for.

Why not Laplacian variance. The obvious metric -- edge energy, the variance of
the image Laplacian -- measures how much fine texture a frame contains, and
that is not focus. It confounds the two whenever scene contrast varies, which
in a real folder it always does. Measured against hand-labelled photos, an
out-of-focus photograph of a projected slide (white text on near-black, so
enormous contrast across wide soft edges) scored *higher* than a correctly
focused close-up of a smooth cream sauce, which is almost featureless and has
almost no edge energy anywhere. Seven variations on edge energy were tried --
whole-frame, per-tile percentiles, contrast-normalised, FFT high-frequency
share, edge-ratio -- and every one of them got that pair the wrong way round.
The failure is structural, not a matter of calibration.

What this does instead (Crete et al. 2007, "The blur effect: perception and
estimation with a new no-reference perceptual blur metric"): blur the image
again on purpose and measure how much the neighbouring-pixel differences
change. A sharp image loses a great deal when blurred; an already-blurred one
has little left to lose. Because the result is a *ratio* of differences, scene
contrast cancels out of it -- which is exactly the confound that sank the
edge-energy family.

Aggregation answers "is anything in this frame in focus", not "is the frame
sharp on average", since culling cares about the subject and not the
background. But not by taking the sharpest region outright: `max` is
maximally optimistic, and a uniformly out-of-focus frame with strong wide
edges always has *some* region that looks structured. Measured on the same
labels, taking the best zone put that blurred slide at the 23rd-85th
percentile of the folder depending on zone size, against the 0th for the
percentile rule below. So: consider only tiles that contain edges at all (a
smooth surface has no opinion about focus), then let the 10th percentile of
those decide -- nearly the sharpest, but robust to one anomalous tile.

Resolution matters more than it looks. Score from the draft-decoded image,
never from a finished thumbnail: resampling is itself a low-pass filter, so it
erases the very differences being measured.

Known blind spot, stated plainly because it is real: a frame with almost no
texture anywhere -- an empty sky, a smooth surface, a night shot of the moon
-- gives this metric very little to work with, and it will report a low score
whether or not the photo is in focus. Read a low score as "worth looking at",
never as a verdict.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# The frame is cut into GRID x GRID tiles. At the ~500px draft decode this app
# scores from, 10 gives tiles of roughly 56x100px -- small enough that a
# subject occupying part of the frame owns tiles of its own rather than being
# averaged into the background, large enough that a tile's statistic is stable.
GRID = 10

# Only tiles above this percentile of gradient energy get a vote. Below it a
# tile is flat -- sky, a wall, a smooth surface -- and a re-blur changes
# nothing there for reasons that have nothing to do with focus.
_EDGE_PERCENTILE = 60.0

# Which of the voting tiles decides. 10 = "nearly the sharpest", chosen over a
# plain minimum so that a single anomalous tile cannot carry the whole frame.
_DECIDING_PERCENTILE = 10.0

# Radius of the deliberate re-blur, in pixels of the draft decode.
_REBLUR_RADIUS = 4

# The 0-100 window, as raw focus (1 - Crete blur). Chosen from the
# distribution over a 455-photo unculled folder (p1 0.723, median 0.895,
# p90 0.926): this leaves the labelled out-of-focus frames near the bottom
# without clamping the ordinary majority against either end.
#
# Deliberately absolute rather than a percentile within the folder, so the
# number means the same thing in every folder. A percentile would force 10% of
# any folder below 10 even when every photo in it is sharp.
RAW_MIN, RAW_MAX = 0.70, 0.95

# Below this a tile is too small for the differences to mean anything.
_MIN_TILE = 12


def _crete_blur(tile: np.ndarray) -> float | None:
    """How *blurred* `tile` is, in 0-1. None if it is too flat to tell.

    Compares neighbouring-pixel differences before and after a deliberate
    re-blur. `sum(d) - sum(max(0, d - d_blurred))` over `sum(d)` is a ratio,
    so multiplying the tile's contrast by any constant leaves it unchanged.
    """
    blurred = np.asarray(
        Image.fromarray(np.clip(tile, 0, 255).astype(np.uint8)).filter(
            ImageFilter.BoxBlur(_REBLUR_RADIUS)
        ),
        dtype=np.float32,
    )
    worst = None
    for axis in (0, 1):
        original = np.abs(np.diff(tile, axis=axis))
        after = np.abs(np.diff(blurred, axis=axis))
        total = float(original.sum())
        # A tile can be perfectly uniform along one axis and still informative
        # along the other -- a picket fence, a horizon, a window frame. Skip
        # the flat direction rather than discarding the tile, which is what an
        # earlier version did.
        if total < 1e-6:
            continue
        lost = float(np.maximum(0.0, original - after).sum())
        ratio = (total - lost) / total
        worst = ratio if worst is None else max(worst, ratio)
    return worst  # None only if flat in both directions: no evidence either way


def focus_measure(image: Image.Image) -> float:
    """Raw focus of `image` in roughly 0.66-0.99. Higher is sharper.

    See the module docstring for why it is aggregated this way rather than by
    taking the sharpest region.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    height, width = gray.shape
    if height < _MIN_TILE or width < _MIN_TILE:
        return 0.0
    scored: list[tuple[float, float]] = []
    for i in range(GRID):
        for j in range(GRID):
            tile = gray[
                i * height // GRID:(i + 1) * height // GRID,
                j * width // GRID:(j + 1) * width // GRID,
            ]
            if min(tile.shape) < _MIN_TILE:
                continue
            blur = _crete_blur(tile)
            if blur is None:
                continue
            dy, dx = np.gradient(tile)
            scored.append((float(np.hypot(dx, dy).mean()), blur))
    if not scored:
        return 0.0
    threshold = float(np.percentile([energy for energy, _ in scored], _EDGE_PERCENTILE))
    voting = [blur for energy, blur in scored if energy >= threshold]
    if not voting:  # every tile equally flat; let them all speak
        voting = [blur for _, blur in scored]
    return 1.0 - float(np.percentile(voting, _DECIDING_PERCENTILE))


def blur_score(image: Image.Image) -> int:
    """Focus of `image` as 0-100, where low means out of focus.

    Linear in the raw measure rather than log-scaled: unlike edge energy, this
    one already lives on a bounded, roughly uniform scale.
    """
    raw = focus_measure(image)
    if raw <= 0:
        return 0
    fraction = (raw - RAW_MIN) / (RAW_MAX - RAW_MIN)
    return int(round(float(np.clip(fraction, 0.0, 1.0)) * 100))
