"""Sharpness scoring: how few pixels a real edge takes to cross.

No torch and no Qt -- this is arithmetic over a decoded image, the same rule
`tamis.quality.scorer` follows.

What this measures. Defocus does not delete an edge, it widens it: blur
preserves an edge's amplitude and spreads it over more pixels. So the thing to
measure is the spread. One window slides along every row and column; its two
ends are plateaus and its middle is where an edge would sit:

    step  = mean(right plateau) - mean(left plateau)     how much it changes
    width = pixels for the profile to rise from 10% to 90% of that step

`step` is a difference of two means, never a sum of absolute differences. That
distinction is the whole design. Every earlier version of this file used a sum
of absolute differences somewhere -- the Crete 2007 re-blur ratio, and before
that Laplacian variance -- and film grain inflates such a sum without bound.
Measured on synthetic tiles, pure noise scored 0.911 against 0.800 for a
perfectly sharp noise-free edge: grain was the *maximum* of those metrics, not
sharpness. Averaging cancels grain instead of accumulating it.

Four tests decide whether a window contains an edge at all. Each exists because
a real photo scored wrongly without it, and each was found by printing the raw
sixteen-pixel profile of the window that set a tile's width:

  amplitude   the step must carry AMP_MIN grey levels, and beat SNR times the
              scatter inside its own plateaus. Removes grain with no noise
              model and no per-image calibration.
  contained   the profile must still be near the left plateau on entry and have
              reached the right one on exit. Without it a step that has already
              happened *inside* a plateau measures as an instant rise -- which
              is how a blurred corridor and a soft wall scored as sharp.
  monotone    it must climb without backtracking. Noise that merely touches the
              far level on one excursion is not an edge, however hard it
              touches it.
  no overshoot a bright rim before a dark region is a sharpening halo, which
              phone JPEGs apply heavily, and it fakes a fast transition.

A tile with fewer than MIN_EDGES measurable edges has no opinion and returns
None. One real edge crossing a tile lights up roughly a thousand windows, so
that threshold only discards statistical flukes. None means "no evidence", not
"blurred", and must never be recorded as zero -- a smooth wall and an
out-of-focus one are different answers.

Known blind spot, stated plainly. The score is the mean of the sharpest few
tiles anywhere in the frame, and nothing makes those tiles belong to the
subject. A motion-blurred portrait taken in front of a sunlit window scores
high on the window. Read a high score as "something here is in focus", not as
"this photo is good".

Resolution. Edge width is measured in pixels, so it scales with the decode.
Half resolution is deliberate: it is four times cheaper than native and
reorders almost nothing (rank correlation 0.955 over a 28-photo trial), while
quarter resolution leaves barely one pixel of usable range -- the whole span
from sharp to fully defocused compresses to 0.25-2px, and the ranking degrades
to 0.837 whichever way the clamp is set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageOps

# Window geometry, in pixels of the half-resolution decode.
PLATEAU = 4         # pixels averaged at each end of the window
SPAN = 8            # pixels in the middle: the widest edge measurable
WINDOW = 2 * PLATEAU + SPAN

AMP_MIN = 18.0      # grey levels a step must carry at minimum
SNR = 5.0           # ...and it must beat this many times the plateau scatter
LO, HI = 0.10, 0.90  # the rise is measured between these fractions of the step
ENTER = 0.25        # how near the plateaus the profile must enter and leave,
LEAVE = 0.75        # two-sided, so an overshooting halo is rejected too
BACKTRACK = 0.25    # how far the climb may reverse and still count

# Width at or beyond which a tile counts as fully defocused. Halved from the
# native-resolution value with the decode, since a width is a pixel count.
MAX_WIDTH = 5.0

TILE = 64           # tile side in decoded pixels
MIN_EDGES = 200     # fewer measurable edges and a tile has no opinion
QUANTILE = 25.0     # which of a tile's edge widths speaks for it
TOP_TILES = 5       # how many tiles decide the photo's score

# Scan lines per band. Bounds peak memory: the profile stack for a whole 20MP
# frame at once would be hundreds of megabytes.
_BAND = 512


def load_for_sharpness(path: Path) -> Image.Image:
    """Decode `path` to greyscale at half its stored size.

    `draft()` scales in the JPEG DCT domain, so this costs far less than a full
    decode. It is a no-op for other formats, hence the explicit fallback --
    without it a PNG would be measured at full resolution and score on a
    different scale from everything around it.
    """
    with Image.open(path) as image:
        full_width = image.width
        image.draft("L", (image.width // 2, image.height // 2))
        drafted_width = image.width
        gray = ImageOps.exif_transpose(image).convert("L")
    if drafted_width > full_width * 0.75:  # draft declined; do it ourselves
        gray = gray.resize(
            (max(1, gray.width // 2), max(1, gray.height // 2)), Image.Resampling.LANCZOS
        )
    return gray


def _tile_of_pixel(length: int, count: int) -> np.ndarray:
    """Which tile each pixel belongs to, matching `k*length//count` slicing."""
    bounds = (np.arange(count + 1) * length) // count
    return np.searchsorted(bounds, np.arange(length), side="right") - 1


def _scan(band: np.ndarray, along_tile: np.ndarray):
    """Edge widths along the rows of `band`.

    `along_tile` maps a position on a scan line to its tile, and is used both
    to place each edge and to drop windows straddling a tile boundary -- those
    would mix two tiles' content into one measurement.

    Returns (line index, centre position, width).
    """
    empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32))
    height, width = band.shape
    count = width - WINDOW + 1
    if count <= 0:
        return empty
    inside = along_tile[:count] == along_tile[WINDOW - 1:WINDOW - 1 + count]
    if not inside.any():
        return empty

    # A plateau is four samples, so its mean and variance come from four shifted
    # views. Prefix sums would need float64 to survive cancellation across a
    # whole row; this stays in float32, touches less memory, and is exact.
    taps = [band[:, i:width - (PLATEAU - 1 - i)] for i in range(PLATEAU)]
    mean = taps[0] + taps[1]
    for tap in taps[2:]:
        mean = mean + tap
    mean = mean * (1.0 / PLATEAU)
    variance = (taps[0] - mean) ** 2
    for tap in taps[1:]:
        variance = variance + (tap - mean) ** 2
    scatter = np.sqrt(variance * (1.0 / PLATEAU))

    left_mean, left_scatter = mean[:, :count], scatter[:, :count]
    right_mean = mean[:, WINDOW - PLATEAU:WINDOW - PLATEAU + count]
    right_scatter = scatter[:, WINDOW - PLATEAU:WINDOW - PLATEAU + count]

    step = right_mean - left_mean
    amplitude = np.abs(step)
    ok = (amplitude >= AMP_MIN) & (amplitude >= SNR * np.maximum(left_scatter, right_scatter))
    ok &= inside
    if not ok.any():
        return empty

    # Build the profile only where a step was found, and in two stages: the
    # containment test needs just the first and last sample and rejects most of
    # what survives above, so the rest is gathered for far fewer positions.
    line, column = np.nonzero(ok)
    step_ok = step[line, column]
    left_ok = left_mean[line, column]
    flat = line * width + column + PLATEAU
    pixels = band.reshape(-1)

    first = (pixels[flat] - left_ok) / step_ok
    last = (pixels[flat + SPAN] - left_ok) / step_ok
    near = (np.abs(first) <= ENTER) & (np.abs(last - 1.0) <= 1.0 - LEAVE)
    if not near.any():
        return empty
    line, column, flat = line[near], column[near], flat[near]
    step_ok, left_ok = step_ok[near], left_ok[near]

    # The window's samples are contiguous, so gather them as one strided block
    # rather than nine separate passes, and fold the running maximum, the
    # backtrack and both level crossings into a single sweep.
    block = sliding_window_view(pixels, SPAN + 1)[flat]
    profile = (block - left_ok[:, None]) / step_ok[:, None]

    run = profile[:, 0].copy()
    backtrack = np.zeros(run.size, np.float32)
    lo_pos = np.zeros(run.size, np.float32)
    hi_pos = np.zeros(run.size, np.float32)
    lo_done = run >= LO          # already past the level on entry -> position 0
    hi_done = run >= HI
    for k in range(1, SPAN + 1):
        previous = run.copy()
        np.maximum(run, profile[:, k], out=run)
        np.maximum(backtrack, run - profile[:, k], out=backtrack)
        for level, position, done in ((LO, lo_pos, lo_done), (HI, hi_pos, hi_done)):
            newly = ~done & (run >= level)
            if newly.any():
                fraction = np.clip(
                    (level - previous) / np.maximum(run - previous, 1e-6), 0.0, 1.0
                )
                np.copyto(position, (k - 1) + fraction, where=newly)
                done |= newly

    keep = backtrack <= BACKTRACK
    if not keep.any():
        return empty

    widths = np.maximum(hi_pos - lo_pos, 0.5)
    widths[~hi_done] = MAX_WIDTH  # never reached the far plateau: as wide as we can see
    line, column, widths = line[keep], column[keep], widths[keep]
    return (
        line.astype(np.int64),
        (column + WINDOW // 2).astype(np.int64),
        widths.astype(np.float32),
    )


def tile_sharpness(gray: Image.Image, tile: int = TILE) -> np.ndarray:
    """Sharpness of every tile, as a (rows, columns) array.

    NaN marks a tile that carried no measurable edge. That is "no evidence",
    which callers must not read as zero.
    """
    pixels = np.asarray(gray.convert("L"), dtype=np.float32)
    height, width = pixels.shape
    rows, columns = max(1, height // tile), max(1, width // tile)
    row_tile = _tile_of_pixel(height, rows)
    column_tile = _tile_of_pixel(width, columns)

    ids: list[np.ndarray] = []
    widths: list[np.ndarray] = []
    for top in range(0, height, _BAND):
        line, centre, found = _scan(pixels[top:min(top + _BAND, height)], column_tile)
        if found.size:
            ids.append(row_tile[top + line] * columns + column_tile[centre])
            widths.append(found)
    transposed = np.ascontiguousarray(pixels.T)
    for left in range(0, width, _BAND):
        line, centre, found = _scan(transposed[left:min(left + _BAND, width)], row_tile)
        if found.size:
            ids.append(row_tile[centre] * columns + column_tile[left + line])
            widths.append(found)

    grid = np.full(rows * columns, np.nan, np.float32)
    if not widths:
        return grid.reshape(rows, columns)

    all_ids = np.concatenate(ids)
    all_widths = np.concatenate(widths)
    order = np.lexsort((all_widths, all_ids))  # by tile, then by width
    all_ids, all_widths = all_ids[order], all_widths[order]

    every = np.arange(rows * columns)
    start = np.searchsorted(all_ids, every, side="left")
    counts = np.searchsorted(all_ids, every, side="right") - start
    enough = counts >= MIN_EDGES
    if enough.any():
        # numpy's linear-interpolation percentile, over each already-sorted run
        position = (counts[enough] - 1) * (QUANTILE / 100.0)
        lower = np.floor(position).astype(np.int64)
        fraction = (position - lower).astype(np.float32)
        base = start[enough]
        low = all_widths[base + lower]
        high = all_widths[base + np.minimum(lower + 1, counts[enough] - 1)]
        chosen = low + (high - low) * fraction
        grid[enough] = np.clip((MAX_WIDTH - chosen) / (MAX_WIDTH - 1.0), 0.0, 1.0)
    return grid.reshape(rows, columns)


def sharpness_score(gray: Image.Image) -> int | None:
    """Sharpness of `gray` as 0-100, or None when nothing in it can be measured.

    The mean of the sharpest few tiles rather than of all of them: culling cares
    whether the subject came out, and most of a frame is usually background. A
    median over the whole frame ranks a busy landscape above a portrait and, on
    the trial folder, separated good from bad barely better than chance.
    """
    grid = tile_sharpness(gray)
    measured = grid[~np.isnan(grid)]
    if measured.size < TOP_TILES:
        return None
    return int(round(float(np.sort(measured)[-TOP_TILES:].mean()) * 100))
