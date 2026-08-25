"""Qt seam for aesthetic scoring, mirroring `tamis.recognition.worker`.

The scoring code itself stays Qt-free; this is the only place it meets
QThreadPool.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal


class QualitySignals(QObject):
    # {filename: score}, generation the batch was requested under, error ("" if ok)
    finished = Signal(dict, int, str)


class QualityScoreWorker(QRunnable):
    """Decode and score one batch of photos.

    Batched rather than one-per-photo because the forward pass costs ~23ms
    alone but ~0.34ms per image when 16 are submitted together -- an
    almost 70x difference that a per-photo worker would throw away.

    Cancellable on the same terms as FaceDetectionWorker: checked once at the
    top of run(), which is enough for a queued-but-unstarted batch to exit
    immediately on a folder switch or a quit.
    """

    def __init__(self, paths: list[Path], generation: int) -> None:
        super().__init__()
        self.paths = paths
        self.generation = generation
        self.signals = QualitySignals()
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def run(self) -> None:
        if self._cancelled.is_set():
            self.signals.finished.emit({}, self.generation, "")
            return
        try:
            from tamis.quality.blur import load_for_sharpness, sharpness_score
            from tamis.quality.scorer import load_for_scoring, score_images
            from tamis.quality.store import PhotoScores

            images = []
            sharpness: list[int | None] = []
            kept: list[Path] = []
            for path in self.paths:
                try:
                    image = load_for_scoring(path)
                    # Sharpness needs its own, larger decode: edge width is
                    # measured in pixels, and the 224px-bound image the model
                    # wants has none left to measure. This is the expensive
                    # half of the batch -- roughly 550ms against 80ms for the
                    # aesthetic score -- and it is why they are not shared.
                    sharp = sharpness_score(load_for_sharpness(path))
                except Exception:
                    # One unreadable photo must not lose the whole batch; it
                    # simply stays unscored and can be retried later.
                    continue
                images.append(image)
                sharpness.append(sharp)
                kept.append(path)
            qualities = score_images(images)
            result = {
                path.name: PhotoScores(quality=quality, blur=blur)
                for path, quality, blur in zip(kept, qualities, sharpness)
            }
            error = ""
        except Exception as exc:
            result = {}
            error = str(exc)
        self.signals.finished.emit(result, self.generation, error)
