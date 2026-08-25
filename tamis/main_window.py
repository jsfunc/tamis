"""Main application window: layout, menus, shortcuts, and controller logic.

Editing and face-recognition state/behavior live in EditController and
FaceRecognitionController (tamis/controllers/) respectively -- MainWindow
constructs them, wires the handful of signals that cross between them (crop
mode and face-edit mode can't both be active; an overwrite save invalidates
that photo's cached face data), and owns everything else: the window layout,
menus/shortcuts, the photo library, and navigation between photos.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, QUrl
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSlider,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tamis import __version__
from tamis.controllers.edit_controller import EditController
from tamis.io_ops import (
    apply_culling,
    capture_time,
    find_sequence_groups,
    rename_by_creation_date,
    renumber_by_creation_time,
    rename_with_sequence,
)
from tamis.models import ImageLibrary, Status
from tamis.thumbnails import ImageLoadWorker, MetadataLoadWorker
from tamis.views.dialogs import ApplyCullingDialog, RenameDialog, RenumberDialog
from tamis.views.edit_panel import EditPanel
from tamis.views.image_viewer import ImageViewer
from tamis.views.metadata_panel import MetadataPanel
from tamis.views.thumbnail_list import ThumbnailList

# Face recognition depends on torch/facenet-pytorch, an optional heavy extra
# (see requirements-recognition.txt) not installed by the base ./install.sh --
# the whole feature degrades to "not available" rather than breaking the app
# for anyone who hasn't opted into it.
try:
    from tamis.controllers.face_recognition_controller import FaceRecognitionController
    from tamis.views.face_panel import FacePanel
    from tamis.views.search_panel import SearchPanel

    RECOGNITION_AVAILABLE = True
except ImportError:
    RECOGNITION_AVAILABLE = False

# Aesthetic scoring needs open_clip plus a ~1.7GB CLIP download (see
# requirements-quality.txt), a second optional extra on top of recognition.
# Same degradation rule: without it the filmstrip simply shows no scores and
# no filter slider.
try:
    from tamis.controllers.quality_controller import QualityController

    QUALITY_AVAILABLE = True
except ImportError:
    QUALITY_AVAILABLE = False

IMAGE_LOAD_PRIORITY = 10  # above the default (0) used by thumbnail workers

SHORTCUTS_TEXT = """\
Navigation
  Right / D       Next image
  Left  / A       Previous image

Zoom
  Z               Toggle 1:1 (actual pixels) / fit to window
  Mouse wheel     Zoom in/out
  Double-click    Fit to window
  (zoom and position carry over to the next photo, so you can
   compare focus across a burst at the same point)

Culling
  S / Up          Mark Selected
  X / Down        Mark Rejected
  U               Unmark
  1-5             Set star rating
  0               Clear rating

Editing
  E               Show Edit Image panel
  M               Show Image Information panel
  R               Rotate clockwise
  Shift+R         Rotate counter-clockwise
  H               Flip horizontal
  V               Flip vertical
  Ctrl+Z          Undo edit
  Ctrl+Shift+Z    Redo edit
  Ctrl+S          Save edited copy

Library
  Ctrl+Shift+A    Apply Culling (move/copy to folders)
  N               Rename with name + sequence number
  Ctrl+Shift+N    Renumber a sequence by creation time
  Ctrl+Shift+D    Rename all by creation date (pYYYYmmdd_hhmmss.ext)
"""

if RECOGNITION_AVAILABLE:
    SHORTCUTS_TEXT = SHORTCUTS_TEXT.replace(
        "  M               Show Image Information panel\n",
        "  M               Show Image Information panel\n  F               Show Face Recognition panel\n",
    )


def _bundled_resource_path(relative: str) -> Path:
    """Resolve a bundled data file's path, whether running from source or as
    a frozen PyInstaller executable -- tamis.spec's one-file build extracts
    its `datas` into a temp dir at runtime, referenced by `sys._MEIPASS`,
    rather than leaving them alongside the source tree.
    """
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base = Path(__file__).resolve().parent.parent  # repo root
    return base / relative


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Tamis {__version__}")
        self.resize(1280, 860)

        self.library = ImageLibrary()
        self._thread_pool = QThreadPool.globalInstance()
        self._sort_mode = "name"  # persists across folder switches within the session

        self._image_load_generation = 0
        self._pending_image_workers: list[ImageLoadWorker] = []

        self._metadata_load_generation = 0
        self._pending_metadata_workers: list[MetadataLoadWorker] = []

        # Memoized capture times for the open folder; see _capture_time.
        self._capture_times: dict[Path, float] = {}

        self.viewer = ImageViewer()
        self.thumbnail_list = ThumbnailList()
        self.thumbnail_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.edit_panel = EditPanel()
        self.metadata_panel = MetadataPanel()

        self.edit_ctl = EditController(self, self.library, self.viewer, self.edit_panel)

        # One panel visible at a time, selected by tab, rather than several
        # independently-toggleable docks -- Image Information is the default.
        self.side_tabs = QTabWidget()
        self.side_tabs.addTab(self.metadata_panel, "Image Information")
        self.side_tabs.addTab(self.edit_panel, "Edit Image")

        # Metadata/edit/faces sit beside the image only (not the full window
        # height like a dock would), so the thumbnail strip below spans the
        # full width.
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(self.viewer)
        top_splitter.addWidget(self.side_tabs)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 0)
        top_splitter.setSizes([900, 450])

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top_splitter)
        splitter.addWidget(self._build_filmstrip_row())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([700, 150])
        self.setCentralWidget(splitter)

        self.side_tabs.currentChanged.connect(self._on_side_tab_changed)

        if RECOGNITION_AVAILABLE:
            self.face_panel = FacePanel()
            self.side_tabs.addTab(self.face_panel, "Face Recognition")
            self.face_ctl = FaceRecognitionController(
                self, self.library, self.viewer, self.face_panel, self._thread_pool
            )
            # Crop mode and face-edit mode can't both be active -- both
            # interpret mouse drags on the shared ImageViewer -- so these two
            # connections deliberately go through MainWindow rather than
            # straight to either controller, to enforce that before
            # delegating to the one that's actually turning on.
            self.face_panel.edit_mode_toggled.connect(self._on_face_edit_mode_toggled)

            self.search_panel = SearchPanel(
                self.library,
                self.face_ctl.face_catalog,
                self.face_ctl.person_gallery,
                self._thread_pool,
                self.face_panel.threshold,
            )
            self.side_tabs.addTab(self.search_panel, "Search by Name")
            self.search_panel.photo_chosen.connect(self._on_search_photo_chosen)
            # A merge/forget in Manage People can remove a Person that an
            # in-flight search holds a reference to -- routed through
            # MainWindow (not FaceRecognitionController directly) since it's
            # the one place that knows about both controllers and SearchPanel.
            self.face_panel.manage_people_requested.connect(self._on_manage_people_requested)

        if QUALITY_AVAILABLE:
            self.quality_ctl = QualityController(self, self.library)
            self.quality_ctl.scores_updated.connect(self._on_scores_updated)
            self.quality_ctl.progress.connect(self._on_scoring_progress)
            self.quality_ctl.failed.connect(self._on_scoring_failed)

        self._build_menu()
        self._build_shortcuts()

        self.thumbnail_list.currentRowChanged.connect(self._on_thumbnail_selected)
        self.viewer.crop_selected.connect(self.edit_ctl.on_crop_selected)
        self.edit_panel.crop_mode_toggled.connect(self._on_crop_mode_toggled)
        self.edit_panel.save_copy_requested.connect(lambda: self._save_edit(mode="copy"))
        self.edit_panel.save_as_requested.connect(lambda: self._save_edit(mode="as"))
        self.edit_panel.save_overwrite_requested.connect(lambda: self._save_edit(mode="overwrite"))

        self._update_status_bar()

    def _build_filmstrip_row(self) -> QWidget:
        """The filmstrip, with the quality controls in a narrow column to its
        left: a sort-by-quality button, the filter slider, and a
        sort-by-sharpness button beneath it.

        The two buttons bracket the slider because they are the two orders the
        scores under each thumbnail support, and the slider filters on the
        first of them. The column is exactly one thumbnail cell tall, so the
        controls line up with the strip they act on and the slider's travel
        reads against the scores printed under each photo. Absent entirely
        when scoring isn't installed, rather than shown doing nothing.
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        if QUALITY_AVAILABLE:
            column = QWidget()
            # Matches a filmstrip item exactly (icon + filename + stars +
            # score), so button and slider together occupy the height of one
            # thumbnail rather than stretching with the widget.
            column.setFixedHeight(self.thumbnail_list.gridSize().height())
            column_layout = QVBoxLayout(column)
            column_layout.setContentsMargins(3, 2, 0, 2)
            column_layout.setSpacing(3)

            self.sort_by_score_button = QToolButton()
            self.sort_by_score_button.setText("\u2193")  # descending
            self.sort_by_score_button.setCheckable(True)
            self.sort_by_score_button.setToolTip(
                "Rank photos by quality score, highest first.\nClick again for filename order."
            )
            self.sort_by_score_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.sort_by_score_button.clicked.connect(self._on_sort_by_score_clicked)

            self.score_filter = QSlider(Qt.Orientation.Vertical)
            self.score_filter.setRange(0, 100)
            self.score_filter.setValue(0)
            self.score_filter.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._update_score_filter_tooltip(0)
            self.score_filter.valueChanged.connect(self._on_score_filter_changed)

            self.sort_by_sharpness_button = QToolButton()
            self.sort_by_sharpness_button.setText("\u25ce")  # a focusing reticle
            self.sort_by_sharpness_button.setCheckable(True)
            self.sort_by_sharpness_button.setToolTip(
                "Rank photos by sharpness, sharpest first.\nClick again for filename order."
            )
            self.sort_by_sharpness_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.sort_by_sharpness_button.clicked.connect(self._on_sort_by_sharpness_clicked)

            column_layout.addWidget(self.sort_by_score_button, 0, Qt.AlignmentFlag.AlignHCenter)
            column_layout.addWidget(self.score_filter, 1, Qt.AlignmentFlag.AlignHCenter)
            column_layout.addWidget(self.sort_by_sharpness_button, 0, Qt.AlignmentFlag.AlignHCenter)
            layout.addWidget(column, 0, Qt.AlignmentFlag.AlignTop)

        layout.addWidget(self.thumbnail_list, 1)
        return row

    def _on_sort_by_score_clicked(self) -> None:
        """Toggle between score order and the default alphabetical order.

        Clicking again returns to filename order rather than to whichever
        order happened to be active before. That keeps the button a genuine
        two-state toggle whose checked state always means "sorted by score";
        restoring an arbitrary previous mode would leave the unchecked state
        meaning several different things. A different order is still one click
        away in the View menu, which unchecks this.
        """
        self._set_sort_mode("name" if self._sort_mode == "score" else "score")

    def _on_sort_by_sharpness_clicked(self) -> None:
        """Toggle between sharpness order and the default alphabetical order.

        Same two-state contract as the quality button above the slider, and
        the two are mutually exclusive: picking one unchecks the other,
        because they are both faces of the single sort mode.
        """
        self._set_sort_mode("name" if self._sort_mode == "sharpness" else "sharpness")

    def _update_score_filter_tooltip(self, value: int) -> None:
        self.score_filter.setToolTip(
            f"Showing photos scoring {value} or above.\n"
            "Nothing is deleted -- lower this to bring them back."
            if value
            else "Hide photos below a quality score.\nNothing is deleted -- lower this to bring them back."
        )

    def _on_score_filter_changed(self, value: int) -> None:
        self._update_score_filter_tooltip(value)
        self.thumbnail_list.set_min_score(value)
        # The photo on screen may have just been filtered out; move to the
        # nearest one still visible rather than leaving the viewer showing
        # something the strip no longer offers.
        item = self.library.current_item
        if item is not None and self.thumbnail_list.is_filtered_out(item):
            nearest = self._nearest_visible_index(self.library.current_index)
            if nearest is not None:
                self.library.current_index = nearest
                self._show_current()
        self._update_status_bar()

    def _nearest_visible_index(self, start: int) -> int | None:
        """Closest index to `start` that the filter still shows, searching
        outward in both directions. None when the cutoff hides everything."""
        count = len(self.library.items)
        for offset in range(count):
            for index in (start + offset, start - offset):
                if 0 <= index < count and not self.thumbnail_list.is_filtered_out(self.library.items[index]):
                    return index
        return None

    def _on_scores_updated(self) -> None:
        self.thumbnail_list.set_scores(
            {item.path: self.quality_ctl.score_for(item.path)
             for item in self.library.items
             if self.quality_ctl.score_for(item.path) is not None}
        )
        # Re-sort as results arrive rather than only once the pass ends.
        # Waiting made asking for score order on a folder that hadn't been
        # scored yet look like it had done nothing at all: with no scores,
        # every photo sits in the "unscored" bucket and score order is
        # identical to filename order. Toggling off again before the pass
        # finished then meant it never appeared to work.
        if self._sort_mode in ("score", "sharpness"):
            self._resort_by_score()
        self._update_status_bar()

    def _resort_by_score(self) -> None:
        """Re-apply whichever score-derived order is active, as results land."""
        if not self.library.items:
            return
        self.library.sort_items(key=self._sort_key(self._sort_mode))
        self.thumbnail_list.set_items(self.library.items)
        self.thumbnail_list.select_index(self.library.current_index)

    def _on_scoring_failed(self, error: str) -> None:
        self.statusBar().showMessage(f"Quality scoring unavailable: {error}")

    def _on_scoring_progress(self, done: int, total: int) -> None:
        if total and done < total:
            self.statusBar().showMessage(f"Scoring photo quality: {done}/{total}...")
            return
        if total:
            # Ordering is kept up to date by _on_scores_updated as each batch
            # lands, so there is nothing to re-sort here.
            self._update_status_bar()

    # -- Menu / shortcuts --------------------------------------------------

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("Open Folder...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._choose_folder)
        file_menu.addAction(open_action)

        apply_action = QAction("Apply Culling...", self)
        apply_action.setShortcut("Ctrl+Shift+A")
        apply_action.triggered.connect(self._apply_culling)
        file_menu.addAction(apply_action)

        rename_action = QAction("Rename...", self)
        rename_action.setShortcut("N")
        rename_action.triggered.connect(self._rename_current)
        file_menu.addAction(rename_action)

        renumber_action = QAction("Renumber by Creation Time...", self)
        renumber_action.setShortcut("Ctrl+Shift+N")
        renumber_action.triggered.connect(self._renumber_by_creation_time)
        file_menu.addAction(renumber_action)

        rename_by_date_action = QAction("Rename All by Creation Date...", self)
        rename_by_date_action.setShortcut("Ctrl+Shift+D")
        rename_by_date_action.triggered.connect(self._rename_by_creation_date)
        file_menu.addAction(rename_by_date_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        sort_group = QActionGroup(self)
        sort_group.setExclusive(True)

        self.sort_by_name_action = QAction("Sort by Name", self)
        self.sort_by_name_action.setCheckable(True)
        self.sort_by_name_action.setChecked(True)
        self.sort_by_name_action.triggered.connect(lambda: self._set_sort_mode("name"))
        sort_group.addAction(self.sort_by_name_action)
        view_menu.addAction(self.sort_by_name_action)

        self.sort_by_date_action = QAction("Sort by Date Taken", self)
        self.sort_by_date_action.setCheckable(True)
        self.sort_by_date_action.triggered.connect(lambda: self._set_sort_mode("date"))
        sort_group.addAction(self.sort_by_date_action)
        view_menu.addAction(self.sort_by_date_action)

        self.sort_by_stars_action = QAction("Sort by Star Rating", self)
        self.sort_by_stars_action.setCheckable(True)
        self.sort_by_stars_action.triggered.connect(lambda: self._set_sort_mode("stars"))
        sort_group.addAction(self.sort_by_stars_action)
        view_menu.addAction(self.sort_by_stars_action)

        if QUALITY_AVAILABLE:
            self.sort_by_score_action = QAction("Sort by Quality Score", self)
            self.sort_by_score_action.setCheckable(True)
            self.sort_by_score_action.triggered.connect(lambda: self._set_sort_mode("score"))
            sort_group.addAction(self.sort_by_score_action)
            view_menu.addAction(self.sort_by_score_action)

            self.sort_by_sharpness_action = QAction("Sort by Sharpness", self)
            self.sort_by_sharpness_action.setCheckable(True)
            self.sort_by_sharpness_action.triggered.connect(
                lambda: self._set_sort_mode("sharpness")
            )
            sort_group.addAction(self.sort_by_sharpness_action)
            view_menu.addAction(self.sort_by_sharpness_action)

        help_menu = self.menuBar().addMenu("&Help")
        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_action)

        if RECOGNITION_AVAILABLE:
            face_docs_action = QAction("Face Recognition Docs", self)
            face_docs_action.triggered.connect(self._open_face_recognition_docs)
            help_menu.addAction(face_docs_action)

        if QUALITY_AVAILABLE:
            # Behind the flag, unlike the architecture doc: without the extra
            # there is no sharpness number on screen for it to explain.
            sharpness_docs_action = QAction("How Sharpness Is Scored", self)
            sharpness_docs_action.triggered.connect(self._open_sharpness_docs)
            help_menu.addAction(sharpness_docs_action)

        # Not behind RECOGNITION_AVAILABLE: this one describes the whole app,
        # so it's just as relevant to a build without the optional extra.
        architecture_action = QAction("Architecture Docs", self)
        architecture_action.triggered.connect(self._open_architecture_docs)
        help_menu.addAction(architecture_action)

        about_action = QAction("About Tamis", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_shortcuts(self) -> None:
        def add(sequence: str, handler) -> None:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(handler)

        add("Right", self._go_next)
        add("D", self._go_next)
        add("Left", self._go_prev)
        add("A", self._go_prev)

        add("S", lambda: self._set_status(Status.SELECTED))
        add("Up", lambda: self._set_status(Status.SELECTED))
        add("X", lambda: self._set_status(Status.REJECTED))
        add("Down", lambda: self._set_status(Status.REJECTED))
        add("U", lambda: self._set_status(Status.UNRATED))

        for rating in range(1, 6):
            add(str(rating), lambda r=rating: self._set_rating(r))
        add("0", lambda: self._set_rating(0))

        add("Z", self.viewer.toggle_actual_size)

        add("E", lambda: self.side_tabs.setCurrentWidget(self.edit_panel))
        add("M", lambda: self.side_tabs.setCurrentWidget(self.metadata_panel))
        if RECOGNITION_AVAILABLE:
            add("F", lambda: self.side_tabs.setCurrentWidget(self.face_panel))
        add("R", self.edit_panel.rotate_cw.emit)
        add("Shift+R", self.edit_panel.rotate_ccw.emit)
        add("H", self.edit_panel.flip_horizontal.emit)
        add("V", self.edit_panel.flip_vertical.emit)
        add("Ctrl+Z", self.edit_panel.undo_requested.emit)
        add("Ctrl+Shift+Z", self.edit_panel.redo_requested.emit)
        add("Ctrl+S", self.edit_panel.save_copy_requested.emit)

    def _show_shortcuts(self) -> None:
        QMessageBox.information(self, "Keyboard Shortcuts", SHORTCUTS_TEXT)

    def _open_docs(self, relative: str, title: str) -> None:
        """Open a bundled HTML doc in the user's browser.

        Anything opened this way must also be listed in tamis.spec's `datas`,
        or it will be missing from the frozen executable and this will report
        it as not found -- which is the whole reason for the exists() check:
        packaging is where a doc goes astray, not a source checkout.
        """
        path = _bundled_resource_path(relative)
        if not path.exists():
            QMessageBox.warning(self, title, f"Documentation file not found:\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_face_recognition_docs(self) -> None:
        self._open_docs("docs/face_recognition.html", "Face Recognition Docs")

    def _open_sharpness_docs(self) -> None:
        self._open_docs("docs/sharpness.html", "How Sharpness Is Scored")

    def _open_architecture_docs(self) -> None:
        self._open_docs("docs/architecture.html", "Architecture Docs")

    def _show_about(self) -> None:
        features = "<br>".join(
            f"{'Face recognition' if i == 0 else 'Quality scoring'}: "
            f"{'enabled' if available else 'not installed'}"
            for i, available in enumerate((RECOGNITION_AVAILABLE, QUALITY_AVAILABLE))
        )
        QMessageBox.about(
            self,
            "About Tamis",
            f"<b>Tamis</b> {__version__}<br><br>"
            "A small desktop app for culling and lightly editing a folder of photos.<br><br>"
            f"{features}<br><br>"
            'GPLv3 — <a href="https://github.com/jsfunc/tamis">github.com/jsfunc/tamis</a>',
        )

    # -- Folder / library ----------------------------------------------------

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder:
            self.open_folder(Path(folder))

    def open_folder(self, folder: Path) -> None:
        if not self._can_navigate_away():
            return
        if self.library.folder is not None:
            self._save_library_state()
        if RECOGNITION_AVAILABLE:
            self.face_ctl.save_before_switching_folder()

        self.edit_ctl.discard()
        self._capture_times.clear()  # a different folder's photos, and it is unbounded otherwise
        try:
            self.library.load(folder)
        except OSError as exc:
            QMessageBox.critical(self, "Open Folder Failed", f"Could not read {folder}:\n{exc}")
            return
        if self.library.load_error:
            QMessageBox.warning(self, "Ratings/Status", self.library.load_error)
        if RECOGNITION_AVAILABLE:
            self.face_ctl.load_folder(folder)

        if self._sort_mode != "name" and self.library.items:
            self.library.sort_items(key=self._sort_key(self._sort_mode))

        self.thumbnail_list.set_items(self.library.items)
        if QUALITY_AVAILABLE:
            self.quality_ctl.load_folder(folder)
            self._on_scores_updated()   # show whatever was cached from a previous visit
            self.quality_ctl.score_folder()
        self.setWindowTitle(f"Tamis {__version__} — {folder}")

        if self.library.items:
            self.library.current_index = 0
            self._show_current()
        else:
            self.viewer.set_image(QImage())
            self.metadata_panel.set_image(None)
            self.statusBar().showMessage(f"No supported images found in {folder}")

    # -- Persistence helper ---------------------------------------------
    # Every actual file operation elsewhere in this class (rename, save-as,
    # culling) is wrapped in try/except OSError with a message shown to the
    # user; this wraps the equivalent for the library's own state file
    # (ratings/status) so a folder on removable/network media going
    # unwritable mid-session reports a clear error instead of raising out of
    # whatever keyboard shortcut or signal handler happened to trigger the
    # save. (EditController/FaceRecognitionController have the equivalent
    # for their own state.)

    def _save_library_state(self) -> None:
        try:
            self.library.save_state()
        except OSError as exc:
            QMessageBox.warning(self, "Save Failed", f"Could not save photo ratings/status:\n{exc}")

    # -- Navigation ----------------------------------------------------

    def _can_navigate_away(self) -> bool:
        if self.edit_ctl.has_unsaved_edits():
            reply = QMessageBox.question(
                self,
                "Discard edits?",
                f"Discard unsaved edits to {self.edit_ctl.unsaved_edits_photo_name()}?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Discard:
                return False
            self.edit_ctl.discard()
        return True

    def _step(self, delta: int) -> None:
        """Move to the next/previous photo the filter actually shows.

        Skipping hidden photos rather than stopping on them keeps the arrow
        keys consistent with the filmstrip: a photo you cannot see in the
        strip should not be reachable by walking past it either.
        """
        if not self._can_navigate_away():
            return
        index = self.library.current_index + delta
        while 0 <= index < len(self.library.items):
            if not self.thumbnail_list.is_filtered_out(self.library.items[index]):
                self.library.current_index = index
                self._show_current()
                return
            index += delta

    def _go_next(self) -> None:
        self._step(1)

    def _go_prev(self) -> None:
        self._step(-1)

    def _on_thumbnail_selected(self, index: int) -> None:
        if index < 0 or index == self.library.current_index:
            return
        if not self._can_navigate_away():
            self.thumbnail_list.select_index(self.library.current_index)
            return
        self.library.current_index = index
        self._show_current()

    def _show_current(self) -> None:
        item = self.library.current_item
        if item is None:
            self.viewer.set_image(QImage())
            self.metadata_panel.set_image(None)
            self._update_status_bar()
            return

        # Metadata reflects the on-disk file regardless of any pending unsaved
        # edits, and doesn't need to wait on the (async) full-resolution decode
        # -- but it's read asynchronously too now, same reasoning as the image
        # itself: extract_metadata() does file I/O + IFD parsing per call, and
        # running that on the UI thread on every single navigation could
        # visibly stutter rapid next/prev browsing.
        self._load_metadata_async(item.path)

        if self.edit_ctl.edit_session is not None and self.edit_ctl.edit_session.source_path == item.path:
            self.edit_ctl.refresh_preview()
        else:
            self.edit_ctl.discard()
            self._load_image_async(item.path)

        if RECOGNITION_AVAILABLE:
            self.face_ctl.request_detection()

        self.thumbnail_list.select_index(self.library.current_index)
        self._update_status_bar()

    def _load_image_async(self, path: Path) -> None:
        self._image_load_generation += 1
        generation = self._image_load_generation
        worker = ImageLoadWorker(path)
        # Keep a reference until the worker actually finishes: it runs on a
        # background thread, and if this were the only Python reference to it,
        # a later navigation could let it (and its signals object) get garbage
        # collected mid-run, crashing the worker thread on emit().
        self._pending_image_workers.append(worker)
        worker.signals.finished.connect(
            lambda p, image, error, gen=generation, w=worker: self._on_image_loaded(gen, p, image, error, w)
        )
        # Higher priority than thumbnail workers (default 0, shared QThreadPool),
        # so the visible image doesn't wait behind a large folder's thumbnail queue.
        self._thread_pool.start(worker, IMAGE_LOAD_PRIORITY)

    def _on_image_loaded(
        self, generation: int, path: Path, qimage: QImage, error: str, worker: ImageLoadWorker
    ) -> None:
        if worker in self._pending_image_workers:
            self._pending_image_workers.remove(worker)
        if generation != self._image_load_generation:
            return
        if error:
            self.statusBar().showMessage(f"Failed to load {path.name}: {error}")
        self.viewer.set_image(qimage)  # also clears any face-box overlay from the previous photo
        self.edit_ctl.on_new_photo_loaded(qimage)
        if RECOGNITION_AVAILABLE:
            self.face_ctl.on_new_photo_loaded(qimage, path)

    def _load_metadata_async(self, path: Path) -> None:
        self._metadata_load_generation += 1
        generation = self._metadata_load_generation
        worker = MetadataLoadWorker(path)
        # Kept alive until it finishes, same reasoning as _pending_image_workers.
        self._pending_metadata_workers.append(worker)
        worker.signals.finished.connect(
            lambda p, sections, error, gen=generation, w=worker: self._on_metadata_loaded(gen, p, sections, error, w)
        )
        self._thread_pool.start(worker)

    def _on_metadata_loaded(
        self, generation: int, path: Path, sections: list, error: str, worker: MetadataLoadWorker
    ) -> None:
        if worker in self._pending_metadata_workers:
            self._pending_metadata_workers.remove(worker)
        if generation != self._metadata_load_generation:
            return  # user has navigated to a different photo since this was requested
        if error:
            self.statusBar().showMessage(f"Failed to load metadata for {path.name}: {error}")
        self.metadata_panel.set_sections(sections)

    def _update_status_bar(self) -> None:
        item = self.library.current_item
        if item is None:
            self.statusBar().showMessage("Open a folder to get started (File > Open Folder)")
            return
        counts = self.library.counts()
        position = f"{self.library.current_index + 1}/{len(self.library.items)}"
        rating = "*" * item.rating if item.rating else "-"
        message = (
            f"{item.name}  |  {position}  |  Status: {item.status.value}  |  Rating: {rating}"
            f"  |  Selected: {counts['selected']}  Rejected: {counts['rejected']}  Unrated: {counts['unrated']}"
        )
        if QUALITY_AVAILABLE:
            scores = self.quality_ctl.score_for(item.path)
            if scores is not None:
                sharpness = "not measurable" if scores.blur is None else scores.blur
                message += f"  |  Quality: {scores.quality}  Sharpness: {sharpness}"
            # The slider has no numeric label of its own -- it and the sort
            # button share one thumbnail's height -- so the active cutoff is
            # reported here, where it stays visible while browsing.
            minimum = self.thumbnail_list.min_score()
            if minimum:
                hidden = sum(1 for i in self.library.items if self.thumbnail_list.is_filtered_out(i))
                message += f"  |  Showing score >= {minimum} ({hidden} hidden)"
        self.statusBar().showMessage(message)

    # -- Sorting ----------------------------------------------------

    def _capture_time(self, item) -> float:
        """`capture_time`, memoized for this folder.

        Reading it opens the file and parses its EXIF -- ~2.7ms per photo, so
        sorting a 584-photo folder froze the window for 1.2-2.1s, and paid it
        again on every sort because nothing was kept. Keyed by path, so an
        entry left behind by a rename is simply never looked up again.
        """
        cached = self._capture_times.get(item.path)
        if cached is None:
            cached = capture_time(item.path)
            self._capture_times[item.path] = cached
        return cached

    def _sort_key(self, mode: str):
        if mode == "date":
            return self._capture_time
        if mode == "stars":
            # Highest rating first; break ties by capture time (earliest first).
            return lambda item: (-item.rating, self._capture_time(item))
        if mode == "score":
            return self._score_sort_key
        if mode == "sharpness":
            return self._sharpness_sort_key
        return lambda item: item.path

    def _score_sort_key(self, item):
        """Highest quality score first, unscored photos last.

        Unscored is deliberately not treated as zero: scoring runs in the
        background, so "not scored yet" is a different thing from "scored
        badly", and sinking a photo to the bottom for the former would
        reorder the folder again the moment its score arrived. The leading
        bucket keeps them apart; path breaks ties so the order is stable.
        """
        scores = self.quality_ctl.score_for(item.path) if QUALITY_AVAILABLE else None
        if scores is None:
            return (1, 0, item.path)
        return (0, -scores.quality, item.path)

    def _sharpness_sort_key(self, item):
        """Sharpest first, unscored photos last -- see `_score_sort_key` for
        why unscored is a bucket of its own rather than a zero.

        A photo the measure could not read at all -- no edge anywhere, so
        `blur` is None -- joins that same trailing bucket. Sorting it as zero
        would rank an empty sky below a genuinely out-of-focus frame, which is
        a claim the measure never made.
        """
        scores = self.quality_ctl.score_for(item.path) if QUALITY_AVAILABLE else None
        if scores is None or scores.blur is None:
            return (1, 0, item.path)
        return (0, -scores.blur, item.path)

    def _set_sort_mode(self, mode: str) -> None:
        if mode == self._sort_mode:
            return
        self._sort_mode = mode
        # Keep the View menu in sync even when called programmatically (e.g.
        # reapplying the preference in open_folder): a menu click checks the
        # action before triggered() fires, but a direct call here never
        # touches the action's checked state unless we set it ourselves.
        sort_actions = {
            "name": self.sort_by_name_action,
            "date": self.sort_by_date_action,
            "stars": self.sort_by_stars_action,
        }
        if QUALITY_AVAILABLE:
            sort_actions["score"] = self.sort_by_score_action
            sort_actions["sharpness"] = self.sort_by_sharpness_action
        sort_actions[mode].setChecked(True)
        if QUALITY_AVAILABLE:
            # The filmstrip buttons are a second face of the same choice, so
            # they have to follow a change made from the View menu too -- and
            # only one of them can be lit, since there is one sort mode.
            self.sort_by_score_button.setChecked(mode == "score")
            self.sort_by_sharpness_button.setChecked(mode == "sharpness")
        if self.library.items:
            self.library.sort_items(key=self._sort_key(mode))
            self.thumbnail_list.set_items(self.library.items)
            self.thumbnail_list.select_index(self.library.current_index)
            self._update_status_bar()
            if (
                QUALITY_AVAILABLE
                and mode in ("score", "sharpness")
                and self.quality_ctl.scoring_in_progress
            ):
                # Nothing will appear to move until scores exist, so say so
                # rather than leaving the click looking ignored.
                done, total = self.quality_ctl.scoring_progress
                order = "quality score" if mode == "score" else "sharpness"
                self.statusBar().showMessage(
                    f"Sorting by {order} — still scoring ({done}/{total}); "
                    "the order fills in as results arrive."
                )

    # -- Marking ----------------------------------------------------

    def _set_status(self, status: Status) -> None:
        if self.library.current_item is None:
            return
        self.library.set_status(self.library.current_index, status)
        # Only the current photo changed; refreshing the whole filmstrip made
        # this keystroke's cost scale with the folder -- see refresh_item.
        self.thumbnail_list.refresh_item(self.library.current_index)
        self._save_library_state()
        self._update_status_bar()

    def _set_rating(self, rating: int) -> None:
        if self.library.current_item is None:
            return
        self.library.set_rating(self.library.current_index, rating)
        # Only the current photo changed; refreshing the whole filmstrip made
        # this keystroke's cost scale with the folder -- see refresh_item.
        self.thumbnail_list.refresh_item(self.library.current_index)
        self._save_library_state()
        self._update_status_bar()

    # -- Rename ----------------------------------------------------

    def _rename_current(self) -> None:
        item = self.library.current_item
        if item is None:
            return
        if not self._can_navigate_away():
            return

        dialog = RenameDialog(list(self.library.renamed_names.keys()), self)
        if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.chosen_name:
            return

        name = dialog.chosen_name
        number = self.library.register_name_use(name)
        try:
            new_path = rename_with_sequence(item, name, number)
        except OSError as exc:
            QMessageBox.critical(self, "Rename Failed", str(exc))
            return

        self.edit_ctl.discard()
        self.thumbnail_list.refresh_item(self.library.current_index)  # only this photo's name changed
        self._save_library_state()
        self.statusBar().showMessage(f"Renamed to {new_path.name}")
        self._update_status_bar()

    def _renumber_by_creation_time(self) -> None:
        if not self.library.items:
            return
        if not self._can_navigate_away():
            return

        groups = find_sequence_groups(self.library.items)
        if not groups:
            QMessageBox.information(
                self,
                "Renumber by Creation Time",
                "No groups of same-basename sequenced images (e.g. toto001.jpg, "
                "toto002.jpg) were found in this folder.",
            )
            return

        dialog = RenumberDialog(groups, self)
        if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.chosen_names:
            return

        current_item = self.library.current_item
        total_renamed = 0
        summary_lines: list[str] = []
        all_errors: list[str] = []

        # Groups are disjoint (a filename matches exactly one basename), so
        # renumbering each one against the same items list in turn is safe —
        # earlier groups' renames can't start matching a later group's name.
        for name in dialog.chosen_names:
            report = renumber_by_creation_time(self.library.items, name)
            total_renamed += report.renamed
            if report.renamed:
                self.library.renamed_names[name] = max(self.library.renamed_names.get(name, 0), report.renamed)
            summary = f"{name}: {report.renamed} renamed"
            if report.errors:
                summary += f", {len(report.errors)} error(s)"
            summary_lines.append(summary)
            all_errors.extend(report.errors)

        if total_renamed:
            # Filenames changed (possibly the sort order too); re-sort in place
            # rather than reloading from disk, so status/rating stay attached
            # to the right ImageItem objects instead of being re-read by name.
            self.library.items.sort(key=lambda item: item.path)
            if current_item is not None:
                self.library.current_index = self.library.items.index(current_item)
            self.edit_ctl.discard()
            self.thumbnail_list.set_items(self.library.items)
            self._save_library_state()
            self._show_current()

        message = "\n".join(summary_lines)
        if all_errors:
            message += "\n\nErrors:\n" + "\n".join(all_errors)
        QMessageBox.information(self, "Renumber by Creation Time", message)

    def _rename_by_creation_date(self) -> None:
        if not self.library.items:
            return
        if not self._can_navigate_away():
            return

        confirm = QMessageBox.question(
            self,
            "Rename All by Creation Date",
            f"Rename all {len(self.library.items)} image(s) in this folder to "
            "pYYYYmmdd_hhmmss.ext, based on each photo's creation date?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        current_item = self.library.current_item
        report = rename_by_creation_date(self.library.items)

        if report.renamed:
            # Filenames (and their sort order) changed; re-sort in place rather
            # than reloading from disk, so status/rating stay attached to the
            # right ImageItem objects instead of being re-read by name.
            self.library.items.sort(key=lambda item: item.path)
            if current_item is not None:
                self.library.current_index = self.library.items.index(current_item)
            self.edit_ctl.discard()
            self.thumbnail_list.set_items(self.library.items)
            self._save_library_state()
            self._show_current()

        message = f"{report.renamed} renamed"
        if report.errors:
            message += f", {len(report.errors)} error(s):\n" + "\n".join(report.errors)
        QMessageBox.information(self, "Rename All by Creation Date", message)

    # -- Editing / face recognition cross-feature glue -----------------------
    # Everything else for these two features lives in EditController and
    # FaceRecognitionController -- these three methods exist only because
    # Crop mode and Edit Faces mode can't both be active (both interpret
    # mouse drags on the shared ImageViewer), and an overwrite save needs to
    # invalidate that photo's now-stale cached face data.

    def _on_side_tab_changed(self, _index: int) -> None:
        """The side panel is a single tab widget now (Image Information /
        Edit Image / Face Recognition / Search by Name), exactly one visible
        at a time -- redo whatever used to happen when a dock was
        independently toggled on."""
        current = self.side_tabs.currentWidget()
        if current is not self.edit_panel:
            self.edit_ctl.on_tab_deactivated()
        if RECOGNITION_AVAILABLE and current is not self.face_panel:
            self.face_ctl.on_tab_deactivated()
        if current is self.edit_panel:
            self.edit_ctl.on_tab_activated()
        elif RECOGNITION_AVAILABLE and current is self.face_panel:
            self.face_ctl.on_tab_activated()
        elif RECOGNITION_AVAILABLE and current is self.search_panel:
            self.search_panel.refresh_people()

    def _on_crop_mode_toggled(self, enabled: bool) -> None:
        if enabled and RECOGNITION_AVAILABLE:
            self.face_ctl.exit_face_edit_mode()
        self.edit_ctl.set_crop_mode(enabled)

    def _on_face_edit_mode_toggled(self, enabled: bool) -> None:
        if enabled:
            self.edit_ctl.exit_crop_mode()
        self.face_ctl.set_face_edit_mode(enabled)

    def _on_manage_people_requested(self) -> None:
        removed_ids = self.face_ctl.show_manage_people_dialog()
        if removed_ids:
            self.search_panel.cancel_if_targeting(removed_ids)

    def _on_search_photo_chosen(self, path: Path) -> None:
        index = next((i for i, item in enumerate(self.library.items) if item.path == path), None)
        if index is None or not self._can_navigate_away():
            return
        self.library.current_index = index
        self._show_current()

    def _save_edit(self, mode: str) -> None:
        saved_path = self.edit_ctl.save(mode)
        if saved_path is not None and mode == "overwrite":
            # The filmstrip still shows this photo decoded from its pre-edit
            # pixels; re-read it so the thumbnail matches the file.
            self.thumbnail_list.reload_item(self.library.current_index)
            # Its capture time may have moved too, if it had no EXIF date and
            # was therefore being sorted by file mtime.
            self._capture_times.pop(saved_path, None)
            if QUALITY_AVAILABLE:
                # The score described the pre-edit pixels.
                self.quality_ctl.invalidate(saved_path)
                self.quality_ctl.score_folder()
        if saved_path is not None and mode == "overwrite" and RECOGNITION_AVAILABLE:
            # The old cached boxes/embeddings were computed against the
            # pre-edit pixel geometry (rotation/flip/crop) and are now
            # wrong -- drop them so the next detection request re-processes
            # the actual current file.
            self.face_ctl.invalidate_and_maybe_redetect(saved_path)

    # -- Apply culling ----------------------------------------------------

    def _apply_culling(self) -> None:
        if self.library.folder is None:
            return
        if not self._can_navigate_away():
            return

        counts = self.library.counts()
        if counts["selected"] == 0 and counts["rejected"] == 0:
            QMessageBox.information(self, "Apply Culling", "No images are marked as selected or rejected.")
            return

        dialog = ApplyCullingDialog(counts, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        mode, selected_dir, rejected_dir = dialog.values()

        confirm = QMessageBox.question(
            self,
            "Apply Culling",
            f"{mode.capitalize()} {counts['selected']} selected and {counts['rejected']} rejected "
            f"image(s) into '{selected_dir}/' and '{rejected_dir}/'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        # Snapshot which items are candidates to move, and their pre-move
        # path, so we can tell afterward which ones actually moved (vs.
        # errored partway and stayed put) -- needed below to invalidate just
        # their cached face data, not everything.
        moved_candidates = [
            (item, item.path) for item in self.library.items if item.status in (Status.SELECTED, Status.REJECTED)
        ]

        report = apply_culling(self.library, mode=mode, selected_dir=selected_dir, rejected_dir=rejected_dir)

        if RECOGNITION_AVAILABLE and mode == "move":
            # Drop cached face data for anything that actually left this
            # folder *before* it can be saved -- otherwise the save below
            # would write those now-orphaned entries into this folder's
            # .tamis_faces.json, ready to misattribute results if a future
            # photo reuses the same filename here (the same class of bug as
            # the ratings/status one this method already guards against by
            # saving only after the reload below).
            for item, original_path in moved_candidates:
                self.face_ctl.invalidate_moved(item, original_path)

        if mode == "copy":
            # Nothing moved out of this folder -- self.library.items still
            # accurately reflects what's here, so this is safe to save now.
            self._save_library_state()

        message = f"Moved {report.moved_selected} selected and {report.moved_rejected} rejected image(s)."
        if report.errors:
            message += "\n\nErrors:\n" + "\n".join(report.errors)
        QMessageBox.information(self, "Apply Culling", message)

        if mode == "move":
            folder = self.library.folder
            previous_name = self.library.current_item.name if self.library.current_item else None
            previous_index = self.library.current_index
            self.edit_ctl.discard()
            try:
                self.library.load(folder)
            except OSError as exc:
                QMessageBox.critical(self, "Apply Culling", f"Photos were moved, but reloading {folder} failed:\n{exc}")
                return
            if self.library.load_error:
                QMessageBox.warning(self, "Ratings/Status", self.library.load_error)
            # Only now, after the reload -- self.library.items reflects what's
            # actually still in this folder, so the rewritten state file
            # can't include stale entries for photos that just moved out
            # (save_state() rewrites the whole file from self.items, it
            # doesn't merge, so this naturally scrubs them).
            self._save_library_state()
            if RECOGNITION_AVAILABLE:
                # Reload rather than keep the in-memory cache: some cached
                # entries now refer to photos that just moved to selected/
                # rejected/, and stale entries could otherwise misattribute
                # results if a moved photo's filename gets reused later --
                # already invalidated above, so this save is now clean too.
                self.face_ctl.save_face_catalog()
                self.face_ctl.load_folder(folder)
            if self._sort_mode != "name" and self.library.items:
                self.library.sort_items(key=self._sort_key(self._sort_mode))
            self.thumbnail_list.set_items(self.library.items)
            if self.library.items:
                # The previously current item may itself have been moved out by
                # this culling pass, so it won't be found by name; fall back to
                # the closest valid index to where it used to be.
                match = next(
                    (i for i, item in enumerate(self.library.items) if item.name == previous_name), None
                )
                self.library.current_index = (
                    match if match is not None else min(previous_index, len(self.library.items) - 1)
                )
                self._show_current()
            else:
                self.viewer.set_image(QImage())
                self.metadata_panel.set_image(None)
                self._update_status_bar()

    # -- Lifecycle ----------------------------------------------------

    def closeEvent(self, event) -> None:
        self._save_library_state()
        if RECOGNITION_AVAILABLE:
            # Abandon queued/running detection before anything blocks: a
            # browse can leave a queue of speculative warming jobs behind, and
            # waiting them out would hold the window open for seconds with no
            # explanation. Cancelled workers exit as soon as they are picked
            # up, so the wait below is bounded by one photo's detection.
            self.face_ctl.cancel_detection_work()
            self.face_ctl.wait_for_detection_to_stop()
            # A Search by Name scan runs on the shared pool and only checks
            # for cancellation between photos, so without this the
            # waitForDone() below sits through the rest of the whole folder --
            # the window stays up, unresponsive, for as long as the scan had
            # left to run.
            self.search_panel.cancel_search()
            self.face_ctl.save_face_catalog()
            # face_ctl's saves run on their own background pool now (see
            # FaceRecognitionController._save_thread_pool) -- wait for it
            # explicitly, since the shared _thread_pool.clear() below would
            # otherwise have no effect on it either way (it's a separate
            # pool), and a queued-but-not-yet-run save must never be dropped.
            self.face_ctl.wait_for_pending_saves()
        if QUALITY_AVAILABLE:
            # Cancel before waiting, same reasoning as detection: a folder's
            # worth of queued scoring must not hold the window open.
            self.quality_ctl.cancel()
            self.quality_ctl.save()
            self.quality_ctl.wait_for_idle()
        # Thumbnail and image-load workers run on the shared global thread pool.
        # If any are still running when Qt starts tearing down, they crash trying
        # to emit `finished` on a signals object whose C++ side is already gone.
        # Cancel anything not yet started and block until in-flight tasks finish.
        self._thread_pool.clear()
        self._thread_pool.waitForDone()
        super().closeEvent(event)
