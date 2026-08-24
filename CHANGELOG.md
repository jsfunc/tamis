# Changelog

All notable changes to Tamis are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); dates are when the
work landed, not necessarily when a version was tagged.

## [Unreleased]

### Added

- **A sort-by-sharpness button (`◎`) below the score filter slider**, mirroring
  the sort-by-quality button (`↓`) above it. The slider and the two buttons
  together still occupy exactly one thumbnail cell, so the controls stay lined
  up with the strip they act on. The two buttons are mutually exclusive —
  there is one sort order, so lighting one unlights the other — and either
  returns to filename order when clicked again. Both orders are in the View
  menu as well, and both put not-yet-scored photos last rather than treating
  them as having scored zero, re-sorting as results arrive in the background.


## [2.5.0]

### Changed

- **The sharpness score now measures focus rather than fine detail, and is
  computed a completely different way.** The old score was the variance of
  the image Laplacian — edge energy — which measures how much fine texture a
  frame contains. That is not focus, and it confounds the two whenever scene
  contrast varies. Checked against hand-labelled photos, an out-of-focus shot
  of a projected slide (white text on near-black: enormous contrast across
  wide soft edges) scored *higher* than a correctly focused close-up of a
  smooth cream sauce, which is nearly featureless. Seven variations on edge
  energy — whole-frame, per-tile percentiles, contrast-normalised, FFT
  high-frequency share, edge-ratio, and two published no-reference metrics —
  all got that pair the wrong way round. The failure is structural, not a
  matter of calibration.

  The new score follows Crete et al. 2007: blur the image again deliberately
  and measure how much the neighbouring-pixel differences change. A sharp
  image loses a great deal; an already-blurred one has little left to lose.
  The result is a ratio, so scene contrast cancels out — which is exactly the
  confound that sank the edge-energy family. It is aggregated to answer "is
  anything in this frame in focus", by letting only tiles that contain edges
  vote and taking the 10th percentile of those: nearly the sharpest, but
  robust to one anomalous tile. Taking the sharpest zone outright was measured
  and is much worse — a uniformly out-of-focus frame with strong wide edges
  always has *some* region that looks structured, which put that blurred
  slide anywhere from the 23rd to the 85th percentile of the folder depending
  on zone size, against the 0th for the rule that shipped.

  Costs 11ms per photo against roughly 1ms before, on a decode already paid
  for, and next to ~80ms for the CLIP pass it rides along with.

  Cached scores from earlier versions are discarded and recomputed, since the
  number means something different now.

- Night photographs are no longer penalised for being mostly empty. A correctly
  exposed shot of the moon in a dark sky has almost no fine detail anywhere,
  so the old edge-energy score put it at the very bottom of the folder; it now
  scores in the middle or above.

### Fixed

- A tile that was perfectly uniform along one axis — a picket fence, a
  horizon, a window frame — was discarded entirely instead of being measured
  along the axis that did carry information.


## [2.4.1]

### Fixed

- **Quality scoring contacted HuggingFace on every start, even with the model
  fully cached.** `open_clip` resolves the CLIP weights to a HuggingFace repo
  rather than their original URL, and `huggingface_hub` re-checks the remote
  revision each time a model is loaded — so "everything runs locally after
  that" was not true. No photo, face or score data was ever sent; the request
  asked about a public model repo. The model is now loaded offline by
  default, reaching the network only when the weights are genuinely not
  cached yet, and an `HF_HUB_OFFLINE` you set yourself is respected in either
  direction. Verified by recording outbound connections: two per load before,
  zero now.

### Changed

- The filmstrip is 7px shorter: the padding around each cell was trimmed to
  what the contents actually need, giving the image that much more of the
  window. The cell size now comes from one function shared by the grid and
  the delegate, rather than the same formula written out twice — a mismatch
  between those two does not fail loudly, it paints the filename on top of
  the thumbnail.

## [2.4.0]

### Added

- **A sharpness score beside the quality score**, shown under each thumbnail
  in lighter type after the bold aesthetic score, and in the status bar.
  Variance of the Laplacian mapped to 0-100, computed from the same decode
  the aesthetic model already performs — about a millisecond per photo and no
  extra file read, so scoring a 455-photo folder still takes ~34s. It answers
  a question the aesthetic score cannot: on that folder the blurriest frames
  score 3, 5 and 16 for sharpness while their aesthetic scores (23, 29, 32)
  sit unremarkably mid-range. Cached scores from earlier versions are
  recomputed automatically, since the model identity changed.

- **Zoom now carries over to the next photo, and `Z` toggles 1:1.** Judging
  focus needs 1:1 — a 4000px photo fits the window at 18%, five image pixels
  per screen pixel, where a missed focus and a sharp shot look identical. But
  every navigation reset the view to fit, so comparing the same detail across
  a burst meant re-zooming and re-panning on every frame. The zoom level and
  position now persist, positioned by *relative* location so the next photo
  can be a different size or orientation. Zoom controls also appear in the
  keyboard shortcuts dialog, where they were previously undocumented.

## [2.3.2]

### Added

- `--version` now also reports which optional extras a build has, and the
  About dialog lists both features rather than only face recognition. Both
  features hide themselves when their dependencies are missing, so there was
  previously no way to tell a correctly-lean build from a packaged one that
  was meant to include them.

### Fixed

- **The downloadable builds had no quality scoring.** The release workflow
  installed the recognition extra but never the quality one, so `open_clip`
  was absent from the packaged executable and the feature switched itself
  off exactly as designed — no scores, no filter slider, no ranking button.
  It now ships in the `-recognition-cpu` assets, which grow by about 17MB.
- **CI had been failing since 2.3.0.** The test workflow did not install the
  quality extra either, so the twelve tests covering the feature failed
  instead of being skipped. They are now marked to skip when the extra is
  absent, and the workflow installs it so they actually run.

## [2.3.1]

### Added

- **Rank photos by quality score**, highest first — a toggle button above
  the filter slider beside the filmstrip, and a matching
  `View > Sort by Quality Score`. Clicking again returns to filename order.
  Photos not yet scored sort last rather than as zero, since scoring runs in
  the background and "not scored yet" is not "scored badly". The order
  settles once when a scoring pass finishes rather than reshuffling every
  sixteen photos.

### Fixed

- **Rebuilding the filmstrip could silently change which photo was
  displayed.** `clear()` does not simply drop to "no current row": as rows
  are removed Qt walks the current row along, emitting `currentRowChanged`
  with intermediate *valid* indices, which the window read as the user
  picking a photo. Every caller that rebuilt the strip happened to set the
  current photo immediately afterwards, so it stayed hidden until scoring
  began re-sorting the strip mid-browse — at which point the photo being
  inspected would change on its own. The rebuild no longer emits selection
  signals.
- **Sorting by quality score appeared to do nothing on a folder that had not
  been scored yet.** With no scores, every photo falls into the "unscored"
  bucket, so score order is identical to filename order and nothing moves;
  toggling the button off again before the background pass finished meant the
  order never settled, making the button look permanently broken. The order
  now fills in as each batch of results arrives, and choosing score order
  while scoring is still running says so in the status bar.
- **Quality scores were computed by the wrong model.** The scorer paired
  open_clip's `ViT-L-14` config with OpenAI's weights, which were trained
  with QuickGELU activations that config does not use — open_clip warns and
  proceeds, so it failed silently. Measured over 80 photos, the mismatch
  shifted 75% of scores by more than 2 points out of 100 (Spearman 0.93
  against the correct pairing). The sidecar now records which model produced
  its scores and discards any written by a different one, so caches from
  2.3.0 are recomputed rather than mixed with corrected values.

## [2.3.0]

### Added

- **Automatic aesthetic quality scoring** (optional; `requirements-quality.txt`).
  Each photo gets a 0-100 score, shown in bold beneath its thumbnail, and a
  vertical slider beside the filmstrip hides photos below a chosen score —
  hidden only, never deleted, with arrow-key navigation skipping what the
  filter hides so the strip and the viewer stay consistent. Scores are
  computed in the background, batched 16 at a time, on their own
  single-threaded pool, and cached per folder in `.tamis_quality.json` so
  reopening is instant. Uses CLIP image embeddings plus the LAION aesthetic
  predictor (MIT and Apache-2.0, GPLv3-compatible, entirely local); chosen
  over NIMA/TOPIQ/MUSIQ/CLIP-IQA after measuring all of them on a 455-photo
  unculled folder — the technical metrics detect defects but saturate on
  photos that are merely fine, while this one keeps ranking them.

- **Help > Architecture Docs**, opening
  [docs/architecture.html](docs/architecture.html) — how the app is put
  together: the layers and the dependency rules between them, the threading
  model, the sidecar formats, and the invariants the non-obvious code exists
  to protect. Bundled into the packaged executable alongside the face
  recognition docs.

### Changed

- Release notes are now taken from this changelog's section for the tag
  being released, rather than being a bare list of commit subjects. A tag
  whose version has no section here fails the release job instead of
  publishing an empty body.

## [2.2.0]

### Performance

- **Sorting a folder no longer re-reads every file.** Sorting by date (or by
  star rating, which breaks ties by date) opened each photo and parsed its
  EXIF on the UI thread, and kept nothing, so every sort paid it again --
  1215-2080ms of frozen window per sort on a 584-photo folder. Capture times
  are now memoized for the open folder.
- **Re-sorting no longer re-decodes every thumbnail.** Rebuilding the
  filmstrip discarded all its decoded thumbnails, so each sort re-read the
  whole folder from disk (565 decodes, 1941ms of redundant work). They are
  now kept across a re-sort and pruned to the photos actually present, so a
  folder switch still releases them. Together these take a repeat sort of a
  584-photo folder from ~1.9s frozen plus ~1.9s of background decoding to
  ~40ms and no decoding at all.
- **Marking and rating a photo no longer redraw the whole filmstrip.**
  `S`/`X`/`U`/`0`-`5` rebuilt a badged thumbnail for every photo in the
  folder, so the cost of the app's most repeated keystrokes scaled both with
  folder size and with how much of it was already marked (an unrated photo
  short-circuits the badge drawing, so the work grew as culling progressed).
  On a 584-photo folder: 5.8ms per keypress at the start of a pass rising to
  26.7ms once everything was marked, now a flat 0.037ms.
- **Face detection no longer starves the viewer.** With the Face Recognition
  tab open, every navigation queued a full detection (~320ms per uncached
  photo, serialized by the model lock) on the shared thread pool, where each
  worker held a thread while blocked. Browsing past 24 photos filled all 16
  shared threads, leaving no thread to decode the photo actually on screen.
  Detection now runs on a dedicated single-threaded pool, is debounced so
  photos skimmed past enqueue nothing, and queues already-visited photos
  behind the current one at lower priority to warm the cache. Measured over
  24 photos: displayed image 2323ms -> 707ms, faces 5986ms -> 782ms, and
  time-to-faces no longer grows with how far you browsed.

### Fixed

- The filmstrip thumbnail is now re-read after **Overwrite Original**; it
  previously kept showing the photo as it looked before the edit until the
  folder was reopened.
- Documented that PyPI's Windows `torch` wheel is CPU-only, so
  `pip install -r requirements-recognition.txt` there silently gives CPU
  inference rather than the CUDA build it does on Linux. This mislabelling
  is what made 2.1.0's Windows `-recognition-gpu` asset byte-for-byte
  identical to its `-recognition-cpu` one.
- **Closing the window during a Search by Name scan waited out the whole
  scan.** The scan runs on the shared thread pool and `closeEvent` blocks on
  `waitForDone()`, which only skips work that hadn't started -- so the window
  stayed up and unresponsive for however long the scan had left (5.7s for a
  20-photo folder of uncached photos, minutes for a real one). It is now
  cancelled on close, bounding the wait to a single photo: 0.2s, independent
  of folder size.
- **Confirming a face name could permanently duplicate its gallery sample.**
  Merging or forgetting a person only rewrote the face records of whichever
  folder happened to be open at the time, so records in every other folder
  kept naming a person the gallery no longer had. The labeling path read
  such a label as "not labeled yet", so re-confirming that face added a
  *second* copy of its sample instead of moving the existing one — and
  nothing ever removed the orphan. Measured on a real gallery: 83 of 242
  samples (34%) were redundant, and 7 were filed under two people at once,
  which made the same face vote for both and left `identify` breaking an
  exact tie arbitrarily. Fixed on four levels: merges now record a
  persistent redirect so a merged-away id keeps resolving in *every* folder;
  face labels are reconciled against the gallery whenever a folder is
  opened; a label naming a person who is genuinely gone is cleared rather
  than silently ignored; and `add_embedding`/`merge`/`import_from` no longer
  add a sample the person already has. Existing duplicate and
  claimed-by-two-people samples are dropped when the gallery loads.

### Changed

- **Face embeddings are stored ~18x more compactly**, cutting the work done
  on every face confirmation by about 85x. Both sidecars wrote each 512-d
  embedding as a JSON array of decimal floats — roughly 12,285 characters
  for 2,048 bytes of actual float32 data. They are now quantized to int8
  with a per-vector scale and base64'd into one fixed-width 688-character
  string (`tamis/recognition/codec.py`). On real data, `.tamis_faces.json`
  went from 4.88MB to 0.28MB and `people.json.gz` from 1.05MB to 0.08MB,
  and the serialization behind a single confirmed name dropped from ~315ms
  to ~4ms. Quantization is well below the model's own precision: of 313
  faces scored against 28 people, the suggested name changed for one, whose
  top two candidates were separated by 0.000418 and were already being
  ordered arbitrarily. Sidecars written by earlier versions still load
  unchanged, and are rewritten in the new encoding on the next save.
- Release assets are now CPU-only: the `-recognition-gpu` variants are no
  longer built or published. GPU users should install from source instead
  (`./install.sh` already detects an NVIDIA GPU and installs the CUDA
  wheels), which is documented in the README's "Why there's no GPU
  download". A CUDA build is still fully supported locally — it just can't
  be a release asset, since at ~2.6GB it exceeds GitHub's 2GiB limit for
  release assets. This is why 2.1.0 published without a Linux GPU build:
  every build leg succeeded, then the upload of that one asset failed.
- Release size budgets are now derived from real published v2.1.0 asset
  sizes (lean 200MB, cpu 700MB) instead of pre-CI estimates, and are kept
  below GitHub's 2GiB asset limit. The old GPU budget of 3500MB sat above
  that limit, so it passed an asset that could never actually publish.

## [2.1.0]

A large batch of work: the entire face-recognition feature (already usable,
behind an optional install) plus a wide-ranging correctness/architecture
pass, and a project rename.

### Added

- **Face recognition** (installed by default — see the README's
  Installation section for the `--cpu`/`--no-recognition` override flags):
  detect faces in a photo and suggest who they are, entirely offline (no
  cloud APIs). Suggestions are ranked by confidence and color-coded rather
  than a hard yes/no cutoff; confirming one strengthens future suggestions
  for that person. Includes manual add/remove of face boxes, a Manage
  People dialog (rename, merge duplicates, forget), gallery import/export,
  and a progressive Search by Name tab. See
  [docs/face_recognition.html](docs/face_recognition.html) for how the
  detection/embedding/identity-matching pipeline actually works.
- `install.sh` now installs face recognition by default, auto-detecting an
  NVIDIA GPU and picking the matching PyTorch build (CUDA-enabled or
  CPU-only) automatically.
- Packaged executables now come in three variants per OS: no recognition,
  CPU-only recognition, and GPU (CUDA) recognition — Linux and Windows get
  all three, macOS gets the first two (no CUDA support on Apple hardware).
- `--version`/`-v` CLI flag, a `Help > About Tamis` dialog, and a
  versioned window title — previously there was no way to tell which
  release a running copy actually was.
- Logging to `~/.tamis/tamis.log` plus a global exception handler, so an
  uncaught error leaves a diagnostic trail instead of vanishing silently
  (particularly useful for the packaged executable, which has no terminal).
- A CI workflow (`.github/workflows/tests.yml`) that runs the full test
  suite, including the optional recognition-dependent tests, on every push
  and PR to `main` — previously the only workflow was the tag-triggered
  release build, which never ran `pytest` at all.
- A pick/reject badge on filmstrip thumbnails, in addition to the existing
  background tint, so status reads without relying on color alone.
- A size-budget check in the release workflow, failing the build if a
  future dependency change accidentally bloats a packaged executable
  beyond what's expected for its recognition variant.

### Changed

- **Renamed from picSel to Tamis** — the old name collided with an
  unrelated commercial product in the same space (photo culling/editing
  with facial recognition). Package/import path, on-disk sidecar file names
  (`.tamis_state.json`, `.tamis_faces.json`, `~/.tamis/people.json.gz`),
  and the GitHub repository all changed accordingly; existing data migrates
  automatically from the old names with no user action needed.
- `main_window.py` split from a single ~1,700-line file into
  `tamis/controllers/` (`EditController`, `FaceRecognitionController`) plus
  several `tamis/views/` dialog and panel modules, for testability and to
  stop one file from doing six unrelated jobs at once.
- State files now write atomically (temp file + rename via
  `tamis/persistence.py`) instead of directly, so a crash or full disk
  mid-write can't corrupt them.
- Thumbnail generation now uses JPEG draft-mode decoding where safe
  (roughly halving folder-open time for large folders), and face-catalog/
  person-gallery saves moved off the UI thread (previously up to ~1s of
  felt lag on every face-name confirmation, since each save rewrote its
  entire file synchronously).
- Metadata (EXIF/GPS panel) now loads asynchronously instead of blocking
  the UI thread on every photo navigation.
- Dependency versions in `requirements*.txt` now have upper bounds, to
  stop a fresh install from silently pulling an untested next-major-version
  release.

### Fixed

- A cropped or rotated photo's saved EXIF no longer carries a stale
  embedded thumbnail depicting the pre-edit framing.
- Real camera metadata (capture date, GPS, exposure/lens info) is no
  longer silently stripped from saved photos on Pillow versions below
  11.1 — see `requirements.txt`'s comment on the `Pillow` lower bound for
  the full story.
- Several reload-ordering and race-condition bugs: stale ratings/face data
  surviving an Apply Culling move, a corrupted per-folder or central
  face-recognition data file silently resetting (and risking overwrite) on
  the next save instead of warning, cross-folder face-cache contamination
  on a fast folder switch, one corrupted photo aborting an entire folder
  search instead of being skipped, and a Manage People merge/forget
  silently truncating an in-flight search for the person it affected.
- Crop mode and Face Recognition "Edit Faces" mode are now mutually
  exclusive (both interpreted mouse drags on the shared image viewer).
- The image viewer no longer discards a manual zoom on every viewport
  resize (e.g. nudging the side-panel splitter), only on an actual
  double-click-to-fit.

## [2.0.1]

- Released cached GPU memory after each face-detection/embedding call.
- Documented the face-recognition feature in the README.

## [2.0.0]

- Added face detection and recognition as a new, optional feature.

## [1.0.0]

- Initial release: browse and rate a folder of photos (filmstrip, EXIF/GPS
  panel, select/reject/star-rating with a `.tamis_state.json` sidecar),
  rotate/flip/crop/adjust with undo-redo, Apply Culling to move or copy
  into `selected/`/`rejected/` subfolders, and three renaming modes
  (sequence, renumber-by-creation-order, rename-by-capture-date). JPEG,
  PNG, BMP, TIFF, WebP, and HEIC/HEIF.
