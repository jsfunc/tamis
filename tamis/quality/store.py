"""Per-folder cache of aesthetic scores, persisted beside the photos.

Mirrors `tamis.recognition.faces.FaceCatalog`'s sidecar convention: keyed by
filename, written atomically, and read defensively so a file from an older
version still loads. Scoring a photo costs a decode plus a network forward
pass, so it is done once and cached; the score slider then filters the cached
values with no model involved at all.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

from tamis.persistence import atomic_write_bytes

logger = logging.getLogger(__name__)

QUALITY_FILENAME = ".tamis_quality.json"

# Which scorer produced the cached values. Recorded in the sidecar and checked
# on load, so scores from a different model -- or a different raw->0-100
# mapping -- are recomputed rather than ranked alongside current ones.
#
# Bump this whenever scorer.CLIP_MODEL, CLIP_PRETRAINED, the aesthetic head, or
# RAW_MIN/RAW_MAX change. Scores are only ever compared against each other, so
# a mixed cache has no symptom beyond an ordering that is quietly wrong --
# there would be nothing to notice and nothing to debug from.
MODEL_ID = "clip-ViT-L-14-quickgelu-openai+laion-aesthetic-l14-mlp+range3.0-7.0+focus-crete-tile10-e60-d10-0.70-0.95"


class PhotoScores(NamedTuple):
    """What is known about one photo. `quality` is the aesthetic score,
    `blur` is sharpness -- both 0-100, both higher-is-better, and deliberately
    separate: a sharp photo can be dull and a lovely one can be soft."""

    quality: int
    blur: int


class QualityStore:
    def __init__(self) -> None:
        self.folder: Path | None = None
        self._scores: dict[str, PhotoScores] = {}
        # Bumped on every load(), for the same reason FaceCatalog has one: a
        # background worker started for the previous folder must not write its
        # results into this one's cache, which is keyed by filename only.
        self._generation = 0
        # Set only when the file exists but could not be read -- see
        # PersonGallery.load_error for why that distinction matters. Unlike
        # ratings, a lost score file costs only recomputation, so this is
        # advisory rather than a reason to block saving.
        self.load_error: str | None = None

    def load(self, folder: Path) -> None:
        self.folder = Path(folder)
        self._scores = {}
        self._generation += 1
        self.load_error = None
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            self.load_error = f"Could not read {path} ({exc}); scores will be recomputed."
            logger.warning("%s", self.load_error)
            return
        if not isinstance(data, dict):
            self.load_error = f"{path} is not in the expected format; scores will be recomputed."
            return
        if data.get("model") != MODEL_ID:
            # Recomputing costs one background pass; keeping these would mean
            # ordering photos by two models' opinions at once.
            logger.info("Discarding %s: scored by %r, not %r", path, data.get("model"), MODEL_ID)
            return
        scores = data.get("scores")
        if not isinstance(scores, dict):
            return
        self._scores = {
            str(name): PhotoScores(quality=int(entry["quality"]), blur=int(entry["blur"]))
            for name, entry in scores.items()
            if isinstance(entry, dict)
            and all(
                isinstance(entry.get(k), (int, float)) and 0 <= entry[k] <= 100
                for k in ("quality", "blur")
            )
        }

    def _state_path(self) -> Path:
        assert self.folder is not None
        return self.folder / QUALITY_FILENAME

    @property
    def generation(self) -> int:
        return self._generation

    def get(self, path: Path) -> PhotoScores | None:
        return self._scores.get(path.name)

    def has(self, path: Path) -> bool:
        return path.name in self._scores

    def set_many(self, scores: dict[str, PhotoScores], generation: int) -> bool:
        """Record a batch of results, unless the folder moved on while they
        were being computed. Returns whether anything was stored."""
        if generation != self._generation:
            return False
        self._scores.update(scores)
        return True

    def invalidate(self, path: Path) -> None:
        """Drop a photo's score -- its pixels changed (an overwrite save), so
        the cached value describes an image that no longer exists."""
        self._scores.pop(path.name, None)

    def prune_to(self, names: set[str]) -> None:
        """Forget scores for photos no longer in the folder, so a rename or a
        culling pass doesn't leave entries that a future file could inherit by
        reusing the same name."""
        self._scores = {name: value for name, value in self._scores.items() if name in names}

    def prepare_save(self) -> tuple[Path, dict] | None:
        """Snapshot for a background write, same split as FaceCatalog: cheap
        and synchronous here, serialisation and disk I/O on a worker."""
        if self.folder is None or not self._scores:
            return None
        return self._state_path(), {
            "model": MODEL_ID,
            "scores": {name: s._asdict() for name, s in self._scores.items()},
        }

    @staticmethod
    def write_payload(path: Path, data: dict) -> None:
        atomic_write_bytes(path, json.dumps(data, indent=2, sort_keys=True).encode("utf-8"))
