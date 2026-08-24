import json
import os
from pathlib import Path

import pytest

from tamis.quality.store import MODEL_ID, QUALITY_FILENAME, PhotoScores, QualityStore


def _store(tmp_path) -> QualityStore:
    store = QualityStore()
    store.load(tmp_path)
    return store


def test_scores_round_trip_through_the_sidecar(tmp_path):
    store = _store(tmp_path)
    store.set_many({"a.jpg": PhotoScores(42, 80), "b.jpg": PhotoScores(91, 30)}, store.generation)
    path, data = store.prepare_save()
    QualityStore.write_payload(path, data)

    reloaded = QualityStore()
    reloaded.load(tmp_path)

    assert reloaded.get(tmp_path / "a.jpg") == PhotoScores(42, 80)
    assert reloaded.get(tmp_path / "b.jpg") == PhotoScores(91, 30)
    assert reloaded.has(tmp_path / "a.jpg")
    assert not reloaded.has(tmp_path / "missing.jpg")


def test_results_from_a_previous_folder_are_refused(tmp_path):
    # Scoring is batched and runs in the background, so a batch can land after
    # the user has moved on. The cache is keyed by filename only, so accepting
    # it would attach one folder's score to a same-named file in another.
    store = _store(tmp_path)
    stale = store.generation
    store.load(tmp_path / "other")

    assert store.set_many({"a.jpg": PhotoScores(50, 50)}, stale) is False
    assert store.get(tmp_path / "a.jpg") is None


def test_prune_forgets_photos_no_longer_in_the_folder(tmp_path):
    # Otherwise a renamed-away entry could be inherited by a future file that
    # happens to reuse the name.
    store = _store(tmp_path)
    store.set_many({"a.jpg": PhotoScores(10, 10), "gone.jpg": PhotoScores(20, 20)}, store.generation)

    store.prune_to({"a.jpg"})

    assert store.get(tmp_path / "a.jpg").quality == 10
    assert store.get(tmp_path / "gone.jpg") is None


def test_invalidate_drops_one_score(tmp_path):
    # Used after an overwrite save: the score describes pixels that no longer
    # exist.
    store = _store(tmp_path)
    store.set_many({"a.jpg": PhotoScores(77, 77)}, store.generation)
    store.invalidate(tmp_path / "a.jpg")
    assert store.get(tmp_path / "a.jpg") is None


def test_an_unreadable_sidecar_is_reported_and_starts_empty(tmp_path):
    (tmp_path / QUALITY_FILENAME).write_text("{ not json")
    store = QualityStore()
    store.load(tmp_path)
    assert store.load_error is not None
    assert store.get(tmp_path / "a.jpg") is None


def test_out_of_range_and_malformed_entries_are_ignored(tmp_path):
    # Read defensively so a hand-edited file still loads.
    (tmp_path / QUALITY_FILENAME).write_text(
        json.dumps({
            "model": MODEL_ID,
            "scores": {
                "ok.jpg": {"quality": 55, "blur": 60},
                "high.jpg": {"quality": 150, "blur": 60},
                "low.jpg": {"quality": -3, "blur": 60},
                "text.jpg": {"quality": "nope", "blur": 60},
                "partial.jpg": {"quality": 40},
            },
        })
    )
    store = QualityStore()
    store.load(tmp_path)
    assert store.get(tmp_path / "ok.jpg") == PhotoScores(55, 60)
    for name in ("high.jpg", "low.jpg", "text.jpg", "partial.jpg"):
        assert store.get(tmp_path / name) is None


def test_prepare_save_is_none_when_there_is_nothing_to_write(tmp_path):
    store = _store(tmp_path)
    assert store.prepare_save() is None


def test_display_score_maps_the_useful_range_onto_0_100():
    pytest.importorskip("torch")
    from tamis.quality.scorer import RAW_MAX, RAW_MIN, to_display_score

    assert to_display_score(RAW_MIN) == 0
    assert to_display_score(RAW_MAX) == 100
    assert to_display_score((RAW_MIN + RAW_MAX) / 2) == 50
    # Clamped, so an unusually high or low raw value cannot leave 0-100.
    assert to_display_score(RAW_MIN - 5) == 0
    assert to_display_score(RAW_MAX + 5) == 100
    assert to_display_score(5.0) == 50


def test_a_cancelled_batch_does_no_work():
    pytest.importorskip("torch")
    from tamis.quality.worker import QualityScoreWorker

    worker = QualityScoreWorker([Path("/nonexistent/a.jpg")], generation=1)
    worker.cancel()
    received = []
    worker.signals.finished.connect(lambda scores, gen, err: received.append((scores, gen, err)))
    worker.run()

    assert received == [({}, 1, "")]
    assert worker.cancelled


def test_scores_from_a_different_model_are_discarded(tmp_path):
    """Mixing models would order photos by two opinions at once.

    Scores are only ever compared against each other, so this has no symptom:
    the ordering is simply wrong, with nothing to notice and nothing to debug
    from. The guard is what makes changing the model a safe edit.
    """
    (tmp_path / QUALITY_FILENAME).write_text(
        json.dumps({"model": "some-older-scorer", "scores": {"a.jpg": {"quality": 55, "blur": 60}}})
    )
    store = QualityStore()
    store.load(tmp_path)
    assert store.get(tmp_path / "a.jpg") is None


def test_a_sidecar_with_no_model_recorded_is_discarded(tmp_path):
    # An unlabelled file could have come from any scorer, so it is not
    # trustworthy enough to rank against current scores.
    (tmp_path / QUALITY_FILENAME).write_text(json.dumps({"a.jpg": 55, "b.jpg": 70}))
    store = QualityStore()
    store.load(tmp_path)
    assert store.get(tmp_path / "a.jpg") is None


def test_the_recorded_model_id_names_the_model_actually_used(tmp_path):
    # Guards against the two drifting apart, which is what would let stale
    # scores survive a model change.
    pytest.importorskip("torch")
    from tamis.quality import scorer

    assert scorer.CLIP_MODEL in MODEL_ID
    assert scorer.model_id() == MODEL_ID
    assert f"{scorer.RAW_MIN}-{scorer.RAW_MAX}" in MODEL_ID


def _gradient(width=200, height=200, period=4):
    """A sharp image: hard black/white stripes, maximum edge energy."""
    import numpy as np
    from PIL import Image

    a = np.zeros((height, width), dtype=np.uint8)
    a[:, ::period] = 255
    return Image.fromarray(a, mode="L").convert("RGB")


def test_a_blurred_image_scores_below_a_sharp_one():
    pytest.importorskip("numpy")
    from PIL import ImageFilter

    from tamis.quality.blur import blur_score

    sharp = _gradient()
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=4))

    assert blur_score(sharp) > blur_score(blurred)
    assert 0 <= blur_score(blurred) <= 100


def test_a_flat_image_scores_zero():
    # No edges at all -- the honest answer is "no evidence", which is also the
    # metric's known blind spot: a sharp photo of a blank wall looks like this.
    from PIL import Image

    from tamis.quality.blur import blur_score

    assert blur_score(Image.new("RGB", (100, 100), (128, 128, 128))) == 0


def test_the_score_is_clamped_to_0_100():
    from tamis.quality.blur import RAW_MAX, RAW_MIN, blur_score, focus_measure

    sharp = _gradient()
    assert RAW_MIN < RAW_MAX
    assert focus_measure(sharp) > 0
    assert 0 <= blur_score(sharp) <= 100


def test_focus_ignores_contrast():
    # The point of the metric: halving a scene's contrast must not read as
    # blur. Edge-energy metrics fail this by a factor of four.
    pytest.importorskip("numpy")
    import numpy as np
    from PIL import Image

    from tamis.quality.blur import focus_measure

    sharp = _gradient()
    faint = Image.fromarray(
        (np.asarray(sharp, dtype=np.float32) * 0.25 + 96).astype(np.uint8)
    )
    assert abs(focus_measure(sharp) - focus_measure(faint)) < 0.05


def test_a_tiny_image_does_not_crash():
    # The Laplacian needs a 3x3 neighbourhood.
    from PIL import Image

    from tamis.quality.blur import blur_score

    for size in ((1, 1), (2, 2), (1, 50)):
        assert blur_score(Image.new("RGB", size, (10, 20, 30))) == 0


def test_scores_are_stored_and_read_back_as_a_pair(tmp_path):
    store = _store(tmp_path)
    store.set_many({"a.jpg": PhotoScores(quality=62, blur=88)}, store.generation)
    path, data = store.prepare_save()
    QualityStore.write_payload(path, data)

    # Self-describing on disk, so the file can be read without the code.
    written = json.loads(path.read_text())
    assert written["scores"]["a.jpg"] == {"quality": 62, "blur": 88}

    reloaded = QualityStore()
    reloaded.load(tmp_path)
    assert reloaded.get(tmp_path / "a.jpg") == PhotoScores(62, 88)


def _patched_clip(monkeypatch, behaviour):
    """Replace open_clip's model builder and record HF_HUB_OFFLINE as it saw it."""
    open_clip = pytest.importorskip("open_clip")
    seen = []

    def fake(*args, **kwargs):
        seen.append(os.environ.get("HF_HUB_OFFLINE"))
        return behaviour(len(seen))

    monkeypatch.setattr(open_clip, "create_model_and_transforms", fake)
    # _allow_hub_downloads flips a module constant, which would otherwise leak
    # into later tests.
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub.constants, "HF_HUB_OFFLINE", hub.constants.HF_HUB_OFFLINE)
    return seen


def test_clip_loads_from_the_cache_without_touching_the_network(monkeypatch):
    """open_clip resolves pretrained="openai" to a HuggingFace repo, and
    huggingface_hub re-checks the remote revision on every load -- so a fully
    cached model still contacted huggingface.co each time the scorer started.
    No image data was involved, but "runs entirely locally" should be true
    rather than nearly true."""
    from tamis.quality import scorer

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    seen = _patched_clip(monkeypatch, lambda n: ("model", None, "preprocess"))

    scorer._load_clip()

    assert seen == ["1"]  # offline on the only attempt


def test_a_model_that_is_not_cached_yet_is_downloaded_once(monkeypatch):
    from tamis.quality import scorer

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    def behaviour(attempt):
        if attempt == 1:
            raise RuntimeError("not in the local cache")
        return ("model", None, "preprocess")

    seen = _patched_clip(monkeypatch, behaviour)

    scorer._load_clip()

    # Offline first, then network exactly once for the initial download.
    assert seen == ["1", "0"]


def test_an_explicit_offline_setting_is_not_overridden(monkeypatch):
    # Someone who set HF_HUB_OFFLINE deliberately must not have it flipped
    # back by a cache miss.
    from tamis.quality import scorer

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    seen = _patched_clip(monkeypatch, lambda n: (_ for _ in ()).throw(RuntimeError("not cached")))

    with pytest.raises(RuntimeError):
        scorer._load_clip()

    assert seen == ["1"]  # no retry, no download
