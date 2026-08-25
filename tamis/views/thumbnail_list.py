"""Horizontal filmstrip of thumbnails with async loading and status badges."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, QRectF, QSize, Qt, QThreadPool
from PySide6.QtGui import QFont, QFontMetrics, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QStyledItemDelegate,
)

from tamis.models.image_item import ImageItem, Status
from tamis.thumbnails import ThumbnailWorker
from tamis.views.theme import (
    BADGE_TEXT_COLOR,
    NEUTRAL_TINT,
    REJECTED_BADGE_COLOR,
    REJECTED_TINT,
    SELECTED_BADGE_COLOR,
    SELECTED_TINT,
)

ICON_SIZE = QSize(120, 120)

# The un-badged thumbnail pixmap, cached per item so a status change can
# redraw the badge without re-decoding the image from disk.
_RAW_PIXMAP_ROLE = Qt.ItemDataRole.UserRole + 1

BADGE_DIAMETER = 22

# (quality, blur) as a PhotoScores, stored per item so the delegate can draw
# them without reaching back into a controller during paint.
_SCORE_ROLE = Qt.ItemDataRole.UserRole + 2

# Vertical layout of one cell: padding, the icon, a gap, then three text lines
# (filename, stars, scores). Kept tight, because the filmstrip competes with
# the image for window height and the padding was doing nothing but consuming
# it.
_TEXT_LINES = 3
_TOP_PADDING = 2
_ICON_TEXT_GAP = 2
_BOTTOM_PADDING = 1


def _cell_size(font) -> QSize:
    """The size of one filmstrip cell.

    The single source of truth for both the delegate's sizeHint and the
    widget's grid: when those two disagree the delegate lays text out in a
    rect too short for it and the filename is painted over the thumbnail.
    """
    line = QFontMetrics(font).height()
    return QSize(
        ICON_SIZE.width() + 20,
        _TOP_PADDING + ICON_SIZE.height() + _ICON_TEXT_GAP + _TEXT_LINES * line + _BOTTOM_PADDING,
    )


def _badged_pixmap(pixmap: QPixmap, status: Status) -> QPixmap:
    """A copy of `pixmap` with a small check/cross badge in the top-left
    corner for selected/rejected status -- the background tint alone isn't
    reliably distinguishable for colorblind users (red/green is the most
    common form of color vision deficiency), so this adds a shape cue that
    doesn't depend on color to read."""
    if status is Status.UNRATED:
        return pixmap
    color, glyph = (SELECTED_BADGE_COLOR, "✓") if status is Status.SELECTED else (REJECTED_BADGE_COLOR, "✕")

    badged = QPixmap(pixmap)
    painter = QPainter(badged)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    circle = QRectF(3, 3, BADGE_DIAMETER, BADGE_DIAMETER)
    painter.setPen(BADGE_TEXT_COLOR)
    painter.setBrush(color)
    painter.drawEllipse(circle)
    font = painter.font()
    font.setPixelSize(int(BADGE_DIAMETER * 0.65))
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(circle, Qt.AlignmentFlag.AlignCenter, glyph)
    painter.end()
    return badged


class _ThumbnailDelegate(QStyledItemDelegate):
    """Paints an item as icon / filename / stars / score, with only the score
    in bold.

    A delegate rather than multi-line item text, because a QListWidgetItem
    carries a single font for its whole label -- there is no way to bold one
    line of it. Painting the lines separately is the only way to give the
    score its own weight.
    """

    def sizeHint(self, option, index) -> QSize:
        """Reserve the icon plus three text lines.

        Required, not cosmetic: without it QStyledItemDelegate sizes the item
        for a single line of display text, so paint() lays out three lines in
        a rect too short for them and the filename lands on top of the
        thumbnail.
        """
        return _cell_size(option.font)

    def paint(self, painter, option, index) -> None:
        self.initStyleOption(option, index)
        painter.save()

        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        if selected:
            painter.fillRect(option.rect, option.palette.highlight())
        else:
            brush = index.data(Qt.ItemDataRole.BackgroundRole)
            if brush is not None:
                painter.fillRect(option.rect, brush)

        metrics = QFontMetrics(option.font)
        line = metrics.height()
        rect = option.rect
        text_top = rect.y() + _TOP_PADDING + ICON_SIZE.height() + _ICON_TEXT_GAP

        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if icon is not None:
            pixmap = icon.pixmap(ICON_SIZE)
            # Centred within the icon band, so portrait and landscape
            # thumbnails share a baseline.
            x = rect.x() + (rect.width() - pixmap.width()) // 2
            y = rect.y() + _TOP_PADDING + (ICON_SIZE.height() - pixmap.height()) // 2
            painter.drawPixmap(x, y, pixmap)

        painter.setPen(
            option.palette.highlightedText().color() if selected else option.palette.text().color()
        )
        item = index.data(Qt.ItemDataRole.UserRole)

        name_rect = rect.adjusted(3, 0, -3, 0)
        name_rect.setTop(text_top)
        name_rect.setHeight(line)
        painter.drawText(
            name_rect,
            Qt.AlignmentFlag.AlignCenter,
            metrics.elidedText(
                index.data(Qt.ItemDataRole.DisplayRole) or "",
                Qt.TextElideMode.ElideMiddle,
                name_rect.width(),
            ),
        )

        stars_rect = name_rect.translated(0, line)
        if item is not None and item.rating:
            painter.drawText(stars_rect, Qt.AlignmentFlag.AlignCenter, "\u2605" * item.rating)

        scores = index.data(_SCORE_ROLE)
        if scores is not None:
            self._draw_scores(painter, stars_rect.translated(0, line), option.font, scores)

        painter.restore()

    @staticmethod
    def _draw_scores(painter, rect, base_font, scores) -> None:
        """Quality in bold, sharpness just after it in regular weight.

        Two weights rather than a separator, because the pair is read at a
        glance across a whole strip: the bold number is the one being ranked
        and filtered on, and the lighter one qualifies it.
        """
        bold = QFont(base_font)
        bold.setBold(True)
        # An en dash where sharpness could not be measured at all: a blank
        # would read as "still scoring" and a 0 as "out of focus", and it is
        # neither.
        quality = str(scores.quality)
        blur = "–" if scores.blur is None else str(scores.blur)
        gap = QFontMetrics(base_font).horizontalAdvance("  ")
        quality_width = QFontMetrics(bold).horizontalAdvance(quality)
        blur_width = QFontMetrics(base_font).horizontalAdvance(blur)

        x = rect.center().x() - (quality_width + gap + blur_width) // 2
        painter.setFont(bold)
        painter.drawText(
            QRect(x, rect.y(), quality_width, rect.height()), Qt.AlignmentFlag.AlignCenter, quality
        )
        painter.setFont(base_font)
        painter.drawText(
            QRect(x + quality_width + gap, rect.y(), blur_width, rect.height()),
            Qt.AlignmentFlag.AlignCenter,
            blur,
        )


class ThumbnailList(QListWidget):
    """Single-row filmstrip; `currentRowChanged` reflects the selected image's index."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(False)
        self.setMovement(QListWidget.Movement.Static)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setIconSize(ICON_SIZE)
        # Thumbnails load asynchronously, so an item's size hint changes once its icon
        # arrives (no icon -> icon). setUniformItemSizes(True) would cache the smaller
        # pre-icon layout rect and never grow it, squashing the thumbnail into a sliver.
        self.setUniformItemSizes(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.setItemDelegate(_ThumbnailDelegate(self))

        item_size = _cell_size(self.font())
        self.setGridSize(item_size)
        scrollbar_height = self.horizontalScrollBar().sizeHint().height()
        self.setFixedHeight(item_size.height() + scrollbar_height + 2 * self.frameWidth() + 2)

        self._thread_pool = QThreadPool.globalInstance()
        self._pending_workers: list[ThumbnailWorker] = []
        self._generation = 0
        # Decoded thumbnails, kept across set_items so a re-sort doesn't
        # re-decode the folder. Pruned to the current photos on every
        # set_items, so it stays bounded by folder size (~57KB each).
        self._pixmap_cache: dict[Path, QPixmap] = {}
        # Aesthetic scores by path, and the slider's current cutoff. Filtering
        # only hides rows -- nothing is removed from the library, so a photo
        # below the cutoff is still there when the slider comes back down.
        self._scores: dict[Path, object] = {}  # path -> PhotoScores
        self._min_score = 0

    def set_items(self, items: list[ImageItem]) -> None:
        self._generation += 1
        generation = self._generation
        # Re-sorting calls this with the same photos in a different order, and
        # rebuilding the list drops every decoded thumbnail -- so every sort
        # used to re-decode the whole folder from disk (measured: 565 decodes,
        # 1941ms, to reproduce work already done). Keep what we already have,
        # pruned to the photos actually present, which also releases the old
        # folder's thumbnails on a folder switch rather than growing forever.
        incoming = {item.path for item in items}
        self._pixmap_cache = {
            path: pixmap for path, pixmap in self._pixmap_cache.items() if path in incoming
        }
        # Don't clear _pending_workers here: workers from the previous folder may
        # still be running on the thread pool. Dropping their only Python reference
        # would let their `signals` QObject get collected mid-flight, crashing the
        # worker thread when it later tries to emit. Stale results are already
        # discarded by the generation check in _on_thumbnail_ready, which also
        # removes each worker from this list once it actually completes.
        # Signals stay blocked for the whole rebuild. clear() does not simply
        # drop to "no current row": as rows are removed Qt walks the current
        # row along, emitting currentRowChanged with intermediate *valid*
        # indices. The owning window reads that signal as the user picking a
        # photo, so an unblocked rebuild silently navigates the library --
        # visible as the displayed photo changing on its own while the
        # filmstrip is re-sorted underneath it. Hiding rows in _apply_filter
        # moves the current row for the same reason, so it is covered too.
        was_blocked = self.signalsBlocked()
        self.blockSignals(True)
        try:
            self.clear()
            for item in items:
                list_item = QListWidgetItem(item.name)
                list_item.setData(Qt.ItemDataRole.UserRole, item)
                list_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                score = self._scores.get(item.path)
                if score is not None:
                    list_item.setData(_SCORE_ROLE, score)
                self.addItem(list_item)
                self._request_thumbnail(list_item, item, generation)
            self.refresh_badges()
            self._apply_filter()
        finally:
            self.blockSignals(was_blocked)

    def _request_thumbnail(self, list_item: QListWidgetItem, item: ImageItem, generation: int) -> None:
        cached = self._pixmap_cache.get(item.path)
        if cached is not None:
            # Only the raw pixmap: set_items ends with refresh_badges(), which
            # draws the badge for every row. Compositing it here as well would
            # do that twice for the whole folder on every re-sort.
            list_item.setData(_RAW_PIXMAP_ROLE, cached)
            return
        worker = ThumbnailWorker(item.path)
        self._pending_workers.append(worker)
        worker.signals.finished.connect(
            lambda path, image, error, li=list_item, w=worker, gen=generation: self._on_thumbnail_ready(
                li, image, error, w, gen
            )
        )
        self._thread_pool.start(worker)

    def _on_thumbnail_ready(
        self, list_item: QListWidgetItem, image: QImage, error: str, worker: ThumbnailWorker, generation: int
    ) -> None:
        if worker in self._pending_workers:
            self._pending_workers.remove(worker)
        if generation != self._generation:
            return
        if image.isNull():
            list_item.setToolTip(f"Failed to load thumbnail: {error}" if error else "Failed to load thumbnail")
            return
        pixmap = QPixmap.fromImage(image)
        img_item: ImageItem = list_item.data(Qt.ItemDataRole.UserRole)
        self._pixmap_cache[img_item.path] = pixmap
        self._apply_pixmap(list_item, pixmap)

    def _apply_pixmap(self, list_item: QListWidgetItem, pixmap: QPixmap) -> None:
        list_item.setData(_RAW_PIXMAP_ROLE, pixmap)
        img_item: ImageItem = list_item.data(Qt.ItemDataRole.UserRole)
        list_item.setIcon(QIcon(_badged_pixmap(pixmap, img_item.status)))

    def reload_item(self, index: int) -> None:
        """Re-decode one row's thumbnail because its file changed on disk.

        Needed after an overwrite save: the displayed thumbnail was decoded
        from the pre-edit pixels. That was already stale before thumbnails
        were cached (nothing re-read the file until the list was rebuilt);
        caching would have made it stale until a folder switch, so the
        invalidation is wired up rather than left implicit.
        """
        list_item = self.item(index)
        if list_item is None:
            return
        img_item: ImageItem = list_item.data(Qt.ItemDataRole.UserRole)
        self._pixmap_cache.pop(img_item.path, None)
        self._request_thumbnail(list_item, img_item, self._generation)

    def set_scores(self, scores: dict) -> None:
        """Attach aesthetic scores (path -> 0-100) and redraw. Called as
        background scoring produces results, so it must tolerate being handed
        a partial map many times over."""
        self._scores.update(scores)
        for i in range(self.count()):
            list_item = self.item(i)
            img_item: ImageItem = list_item.data(Qt.ItemDataRole.UserRole)
            score = self._scores.get(img_item.path)
            if score is not None:
                list_item.setData(_SCORE_ROLE, score)
        self._apply_filter()
        self.viewport().update()

    def score_for(self, item: ImageItem) -> int | None:
        return self._scores.get(item.path)

    def set_min_score(self, minimum: int) -> None:
        """Hide photos scoring below `minimum`. Hidden, never removed: the
        slider is a view filter, and dropping items would lose the library's
        own ordering and selection."""
        self._min_score = minimum
        self._apply_filter()

    def min_score(self) -> int:
        return self._min_score

    def is_filtered_out(self, item: ImageItem) -> bool:
        """Whether `item` is hidden by the current cutoff. An unscored photo
        is never hidden -- scoring runs in the background, and making photos
        vanish as results trickle in would be worse than showing them."""
        if self._min_score <= 0:
            return False
        scores = self._scores.get(item.path)
        return scores is not None and scores.quality < self._min_score

    def _apply_filter(self) -> None:
        for i in range(self.count()):
            list_item = self.item(i)
            img_item: ImageItem = list_item.data(Qt.ItemDataRole.UserRole)
            list_item.setHidden(self.is_filtered_out(img_item))

    def refresh_item(self, index: int) -> None:
        """Redraw one row's label, tint and badge.

        Marking and rating change exactly one photo, and used to go through
        `refresh_badges()`, which recomposites a badged pixmap for every
        thumbnail in the folder. That made the cost of the app's most repeated
        keystrokes scale with the size of the folder *and* with how much of it
        had already been marked -- `_badged_pixmap` returns immediately for an
        unrated photo, so the work grew as culling progressed. Measured on a
        584-photo folder with every thumbnail decoded: 5.9ms per keypress at
        the start of a pass, 26.4ms once everything was marked, for a redraw
        of one row.
        """
        list_item = self.item(index)
        if list_item is None:
            return
        img_item: ImageItem = list_item.data(Qt.ItemDataRole.UserRole)
        # Filename only. The star rating and the aesthetic score are drawn by
        # _ThumbnailDelegate on their own lines -- including them here too
        # renders them twice: once as part of this multi-line string, clipped
        # into the single-line name row, and once by the delegate.
        list_item.setText(img_item.name)
        if img_item.status is Status.SELECTED:
            list_item.setBackground(SELECTED_TINT)
        elif img_item.status is Status.REJECTED:
            list_item.setBackground(REJECTED_TINT)
        else:
            list_item.setBackground(NEUTRAL_TINT)
        raw_pixmap = list_item.data(_RAW_PIXMAP_ROLE)
        if raw_pixmap is not None:
            list_item.setIcon(QIcon(_badged_pixmap(raw_pixmap, img_item.status)))

    def refresh_badges(self) -> None:
        """Redraw every row. Only needed when the whole list changed (a new
        folder, or a re-sort); a single photo's change should use
        `refresh_item`."""
        for i in range(self.count()):
            self.refresh_item(i)

    def select_index(self, index: int) -> None:
        if 0 <= index < self.count() and self.currentRow() != index:
            self.blockSignals(True)
            self.setCurrentRow(index)
            self.blockSignals(False)
        if 0 <= index < self.count():
            self.scrollToItem(self.item(index), QAbstractItemView.ScrollHint.EnsureVisible)
