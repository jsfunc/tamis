# Tamis

Tamis is a small desktop app for quickly culling and lightly editing a folder of
photos: browse a shoot, mark shots as selected/rejected, star-rate them, crop
or rotate the keepers, and sort the results into folders — all from the
keyboard, without leaving a single window.

## Features

- **Browse & rate** — filmstrip of thumbnails, full-size preview, EXIF/GPS
  metadata panel. Mark each photo selected, rejected, or unrated, and give it
  a 0-5 star rating. Ratings and status persist alongside the folder (a
  `.tamis_state.json` sidecar) so you can close and resume later.
- **Edit** — rotate, flip, crop, and adjust brightness/contrast/saturation,
  with undo/redo. Save as a copy, overwrite the original, or save as a new
  file.
- **Apply culling** — move or copy selected/rejected photos into
  `selected/` and `rejected/` subfolders in one step.
- **Face recognition** *(installed by default, see [Installation](#installation))* —
  detect faces in a photo and suggest who they are, entirely offline (no
  cloud APIs, nothing ever leaves your machine). Suggestions are ranked by
  confidence, shown color-coded (green = likely, red = unlikely) rather than
  a hard yes/no cutoff — confirm one with a click and it strengthens future
  suggestions for that person. Includes manual add/remove of face boxes for
  anything the detector misses or gets wrong, a Manage People dialog
  (rename, merge duplicates, forget), gallery import/export to move your
  identities to another machine, and a progressive Search by Name that finds
  every photo of someone across the whole folder. See
  [docs/face_recognition.html](docs/face_recognition.html) for how the
  detection/recognition pipeline actually works.
- **Rename**
  - Rename the current photo to `<name><sequence number>.<ext>`.
  - Renumber an existing `<name><digits>.<ext>` sequence so numbering matches
    actual capture order.
  - Rename every photo in the folder to `pYYYYmmdd_hhmmss.ext` based on its
    capture date (from EXIF, falling back to file modification time).
- Supports JPEG, PNG, BMP, TIFF, WebP, and HEIC/HEIF.

## Installation

Requires Python 3.9+.

```bash
./install.sh
```

This checks your Python version, creates a `.venv` virtual environment, and
installs the dependencies (`PySide6`, `Pillow`, `pillow-heif`), **including
face recognition by default**. It auto-detects an NVIDIA GPU (`nvidia-smi`)
and installs the matching PyTorch build:

- **GPU present**: the standard (CUDA-enabled) wheels — faster detection/
  recognition, but a much larger download (~5.5GB total for the recognition
  dependencies, since it bundles NVIDIA's CUDA runtime libraries).
- **No GPU found**: the CPU-only wheels automatically (~1.7GB total) —
  detection/recognition still works, just slower.

Flags to override the default:

```bash
./install.sh --cpu             # force the CPU-only build even with a GPU present
./install.sh --no-recognition  # skip face recognition entirely (PySide6/Pillow only)
```

Without face recognition installed, the app runs normally with the Face
Recognition and Search by Name tabs simply absent. To add it later:

```bash
source .venv/bin/activate
pip install -r requirements-recognition.txt                                            # GPU-capable (Linux)
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-recognition.txt  # CPU-only
```

On **Windows**, the first command installs a CPU-only build — PyPI's Windows
`torch` wheel has no CUDA support. Use PyTorch's own index instead, matching
the CUDA version to your driver (see
[pytorch.org/get-started](https://pytorch.org/get-started/locally/)):

```powershell
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements-recognition.txt
```

Either way, `Help > About Tamis` reports whether recognition is enabled, and
`python -c "import torch; print(torch.cuda.is_available())"` confirms whether
it will actually use the GPU.

## Automatic quality scoring

With the optional quality extra installed, every photo gets two numbers under
its thumbnail: an **aesthetic score from 0 to 100** in bold, and a
**sharpness score** just after it in lighter type. A **vertical slider left of
the filmstrip** hides anything below a chosen aesthetic score, bracketed by a
button at each end that ranks the folder by one of the two numbers — `↓` above
for quality, `◎` below for sharpness. Either button toggles back to filename
order on a second click, and only one can be active, since there is a single
sort order. Both are also in the View menu. Nothing is
deleted — lowering the slider brings the photos straight back, and arrow-key
navigation skips whatever is hidden so the strip and the viewer agree.

```bash
source .venv/bin/activate
pip install -r requirements-quality.txt
```

Scores are computed once per folder in the background and cached in
`.tamis_quality.json` beside the photos, so reopening a folder is instant.
First use downloads ~1.7GB of CLIP weights (from HuggingFace, cached in
`~/.cache/huggingface`) plus a 3.7MB scoring head into `~/.tamis/`.

**After that it is genuinely offline** — no network request of any kind, which
was worth stating precisely because it was not true at first: `open_clip`
fetches those weights from HuggingFace, and `huggingface_hub` re-checks the
remote revision on *every* load, so a fully cached model still contacted
`huggingface.co` each time the scorer started. No photo or face data was ever
involved, but the request happened. Tamis now loads the model offline by
default and reaches the network only when the weights are genuinely missing.
Setting `HF_HUB_OFFLINE` yourself is respected in either direction.

The score comes from CLIP image embeddings fed to the LAION aesthetic
predictor. It was chosen over five alternatives measured on a 455-photo
unculled folder: cheap technical metrics (Laplacian variance, TOPIQ, MUSIQ)
detect blurred frames well but flatten out on photos that are simply fine,
while this one keeps discriminating — roughly twice the spread over the
technically-good photos. That is why both are shown: they answer different
questions, and neither subsumes the other. NIMA scored marginally better and is cheaper, but the only
readily available weights ship in a toolbox under a non-commercial licence
that cannot be distributed with GPLv3 software; this model's parts are MIT and
Apache-2.0.

The sharpness score measures **focus**, and does it by blurring the image
again on purpose: a sharp photo loses a great deal when blurred, an
already-blurred one has little left to lose (Crete et al. 2007). Because the
result is a ratio of pixel differences, scene contrast cancels out of it. That
matters more than it sounds — the obvious metric, the variance of the image
Laplacian, measures how much fine *texture* a frame holds, which is not the
same thing at all. Tested against hand-labelled photos it put an out-of-focus
picture of a projected slide (white text on black, so huge contrast across
soft wide edges) *above* a correctly focused close-up of a smooth sauce.
Seven variations on edge energy failed that pair the same way.

It is aggregated to answer "is anything in this frame in focus" rather than
"is the frame sharp on average", since culling cares about the subject and not
the background: only tiles containing edges get a vote, and the 10th
percentile of those decides — nearly the sharpest, but robust to one
anomalous tile. It runs on the same decode the aesthetic model uses, costing
about 11ms per photo and no extra file read.

Its blind spot is worth knowing: a frame with almost no texture anywhere — an
empty sky, a smooth surface — gives it little to work with, and it will report
a low score whether or not the photo is in focus.

Both are good ways to surface obvious rejects and to order a folder roughly.
Neither is a judgement of your taste — treat a low score as a suggestion to
look, not a verdict.

## Usage

```bash
source .venv/bin/activate
python main.py [folder]
```

Pass a folder to open it immediately, or use File > Open Folder from the app.
See Help > Keyboard Shortcuts inside the app for the full shortcut list
(navigation, rating, editing, renaming).

## Architecture

[docs/architecture.html](docs/architecture.html) (also reachable from the
app's `Help > Architecture Docs`) describes how the app is put together: the
layers and the dependency rules between them, the threading model (three
thread pools with different cancellation rules), the sidecar formats, and the invariants that the non-obvious code exists to
protect. Worth reading before adding a background task or a new persisted
field — most of those rules are there because something broke once.

For the recognition *algorithm* rather than the code structure, see
[docs/face_recognition.html](docs/face_recognition.html).

## Running tests

`pytest` is a dev-only dependency (not installed by `./install.sh`, so a
plain end-user install doesn't pull in a test runner):

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest
```

## Standalone executables

Before tagging a release, bump `__version__` in
[tamis/\_\_init\_\_.py](tamis/__init__.py) to match — it drives the window
title, `Help > About Tamis`, and `python main.py --version`, but nothing
derives it from the git tag automatically.

Pushing a tag like `v1.0.0` triggers [.github/workflows/release.yml](.github/workflows/release.yml),
which builds standalone executables for Linux, Windows, and macOS (Apple
Silicon) with [PyInstaller](https://pyinstaller.org/) and attaches them to a
GitHub Release. You can also run the workflow manually from the Actions tab.

These are unsigned builds, so Windows SmartScreen and macOS Gatekeeper will
warn on first run; you'll need to explicitly allow the app to run.

Each OS gets two release assets:

| Asset suffix | Contents | Approx. size |
| --- | --- | --- |
| *(none)* | No face recognition | 43–93MB |
| `-recognition-cpu` | Face recognition, CPU-only | 196–336MB |

### Why there's no GPU download

Packaged executables are CPU-only even where a CUDA build would be
possible. **If you have an NVIDIA GPU and want to use it, install from
source rather than downloading an executable** — `./install.sh` detects
`nvidia-smi` and installs the CUDA-enabled wheels automatically (see
[Installation](#installation) above). Nothing else needs configuring;
`tamis/recognition/detector.py` picks the device at import time.

Two reasons a prebuilt CUDA asset isn't offered:

- **It's too large to publish.** A CUDA-enabled PyInstaller build is ~2.6GB
  on Linux, and GitHub rejects release assets over 2GiB. Release 2.1.0 hit
  exactly this: the build itself succeeded, but the upload failed, which is
  why that release shipped without it.
- **It would be wrong on Windows.** CUDA wheels are only published on
  `download.pytorch.org`, not PyPI, and PyPI's Windows `torch` wheel is
  CPU-only — so a Windows asset labelled "GPU" would quietly contain CPU
  inference. (Release 2.1.0's Windows `-recognition-gpu` asset was exactly
  this: byte-for-byte the CPU build. Don't use it; take
  `-recognition-cpu`, or install from source.)

To build an executable locally:

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
# Optional, for a face-recognition-capable build -- otherwise the same lean
# build install.sh produces without recognition:
pip install -r requirements-recognition.txt                                            # GPU-capable (Linux)
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-recognition.txt  # CPU-only
pyinstaller tamis.spec
```

The executable is written to `dist/Tamis` (`dist/Tamis.exe` on Windows).
PyInstaller doesn't cross-compile, so this must be run on each target OS.

A CUDA-enabled build works locally and is the supported way to get a
GPU-capable executable — it just can't be published as a release asset (see
above). Expect ~2.6GB, and note that bundling CUDA runtime libraries into a
one-file executable is sensitive to the driver on the machine that runs it,
which is another reason installing from source is the smoother path for GPU
users.

## License

GPLv3 — see [LICENSE](LICENSE). Copyright (C) 2026 jsfunc.
