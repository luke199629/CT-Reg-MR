"""PySide6 GUI for viewing CT/MR overlays and applying DICOM rigid registrations."""
from __future__ import annotations

import os
import sys

import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QAction, QColor, QFont, QImage, QPainter, QPen, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox,
    QFileDialog, QLineEdit,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMainWindow,
    QMessageBox, QProgressDialog, QPushButton, QSlider, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QScrollArea, QSizePolicy,
)

from .dicom_io import Registration, Volume, load_registration, load_volume, scan_folder
from .export import derived_volume, resample_to, write_dicom_series, write_raw
from .geometry import VIEWS, SLICE_AXIS, DisplayGrid, rigid_params, sample_plane, sample_point

CT_PRESETS = {  # name: (level, width)
    "Soft tissue": (40, 400), "Bone": (500, 2000), "Lung": (-600, 1500),
    "Brain": (40, 80), "Abdomen": (60, 350),
}
ORIENT = {  # view: (left, right, top, bottom) labels, radiological convention
    "axial": ("R", "L", "A", "P"),
    "coronal": ("R", "L", "S", "I"),
    "sagittal": ("A", "P", "S", "I"),
}
CROSS_COLORS = {"axial": QColor(80, 160, 255), "coronal": QColor(80, 220, 120), "sagittal": QColor(255, 170, 60)}


# =========================================================================== slice view
class SliceView(QWidget):
    sliceDelta = Signal(str, int)
    clicked = Signal(str, float, float)       # view, row, col (image pixel coords)
    hovered = Signal(str, float, float)

    def __init__(self, view: str, parent=None):
        super().__init__(parent)
        self.view = view
        self.image: QImage | None = None
        self.pixel_mm = (1.0, 1.0)
        self.cross = None                     # (row, col)
        self.show_cross = True
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._drag = None
        self._pan_start = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(200, 200)

    def set_image(self, img: QImage, pixel_mm):
        self.image, self.pixel_mm = img, pixel_mm
        self.update()

    # ---- geometry
    def _target(self) -> QRectF:
        if self.image is None:
            return QRectF()
        w, h = self.image.width(), self.image.height()
        pw, ph = w * self.pixel_mm[1], h * self.pixel_mm[0]
        s = min(self.width() / pw, self.height() / ph) * self.zoom
        tw, th = pw * s, ph * s
        return QRectF((self.width() - tw) / 2 + self.pan.x(), (self.height() - th) / 2 + self.pan.y(), tw, th)

    def _to_image(self, pos):
        t = self._target()
        if t.isEmpty():
            return None
        c = (pos.x() - t.left()) / t.width() * self.image.width() - 0.5
        r = (pos.y() - t.top()) / t.height() * self.image.height() - 0.5
        return r, c

    def _to_widget(self, r, c):
        t = self._target()
        return (t.left() + (c + 0.5) / self.image.width() * t.width(),
                t.top() + (r + 0.5) / self.image.height() * t.height())

    # ---- painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        if self.image is None:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignCenter, f"{self.view.capitalize()}\n(no data)")
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom < 3)
        p.drawImage(self._target(), self.image)
        if self.show_cross and self.cross is not None:
            x, y = self._to_widget(*self.cross)
            others = [v for v in VIEWS if v != self.view]
            # vertical line = the plane of the view whose slice axis is this view's column axis
            col_view = {"axial": "sagittal", "coronal": "sagittal", "sagittal": "coronal"}[self.view]
            row_view = [v for v in others if v != col_view][0]
            p.setPen(QPen(CROSS_COLORS[col_view], 1))
            p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            p.setPen(QPen(CROSS_COLORS[row_view], 1))
            p.drawLine(QPointF(0, y), QPointF(self.width(), y))
        p.setPen(QColor(230, 230, 60))
        f = QFont(); f.setBold(True); p.setFont(f)
        l, r, t, b = ORIENT[self.view]
        m = 6
        p.drawText(QRectF(m, 0, 30, self.height()), Qt.AlignVCenter | Qt.AlignLeft, l)
        p.drawText(QRectF(self.width() - 30 - m, 0, 30, self.height()), Qt.AlignVCenter | Qt.AlignRight, r)
        p.drawText(QRectF(0, m, self.width(), 20), Qt.AlignHCenter | Qt.AlignTop, t)
        p.drawText(QRectF(0, self.height() - 20 - m, self.width(), 20), Qt.AlignHCenter | Qt.AlignBottom, b)

    # ---- interaction
    def wheelEvent(self, e):
        steps = e.angleDelta().y() // 120 or (1 if e.angleDelta().y() > 0 else -1 if e.angleDelta().y() < 0 else 0)
        if e.modifiers() & Qt.ControlModifier:
            self.zoom = float(np.clip(self.zoom * (1.15 ** steps), 0.25, 20))
            self.update()
        elif steps:
            self.sliceDelta.emit(self.view, int(steps))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = True
            self._emit_click(e.position())
        elif e.button() in (Qt.MiddleButton, Qt.RightButton):
            self._pan_start = (e.position(), QPointF(self.pan))

    def mouseMoveEvent(self, e):
        rc = self._to_image(e.position())
        if rc is not None:
            self.hovered.emit(self.view, *rc)
        if self._drag:
            self._emit_click(e.position())
        elif self._pan_start is not None:
            start, pan0 = self._pan_start
            self.pan = pan0 + (e.position() - start)
            self.update()

    def mouseReleaseEvent(self, _):
        self._drag = None
        self._pan_start = None

    def mouseDoubleClickEvent(self, e):
        if e.button() in (Qt.MiddleButton, Qt.RightButton):
            self.zoom, self.pan = 1.0, QPointF(0, 0)
            self.update()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Up, Qt.Key_PageUp):
            self.sliceDelta.emit(self.view, 1)
        elif e.key() in (Qt.Key_Down, Qt.Key_PageDown):
            self.sliceDelta.emit(self.view, -1)
        else:
            super().keyPressEvent(e)

    def _emit_click(self, pos):
        rc = self._to_image(pos)
        if rc is not None:
            self.clicked.emit(self.view, *rc)


class ViewPanel(QWidget):
    def __init__(self, view: str, parent=None):
        super().__init__(parent)
        self.view = view
        self.title = QLabel(view.capitalize())
        self.title.setStyleSheet(f"color: {CROSS_COLORS[view].name()}; font-weight: bold;")
        self.canvas = SliceView(view)
        self.slider = QSlider(Qt.Horizontal)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.title)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.slider)


class CoordPanel(QWidget):
    """Top-right cell: crosshair / mouse coordinates and a 'go to' entry."""
    goRequested = Signal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        title = QLabel("Coordinates")
        title.setStyleSheet("font-weight: bold;")
        lay.addWidget(title)
        self.cross_lbl = QLabel("—")
        self.hover_lbl = QLabel("—")
        for lbl in (self.cross_lbl, self.hover_lbl):
            lbl.setTextFormat(Qt.RichText)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(self.cross_lbl)
        lay.addWidget(self.hover_lbl)
        lay.addStretch(1)

        go = QGroupBox("Go to (fixed-image patient coords, mm)")
        gl = QHBoxLayout(go)
        self.go_xyz = []
        for name in "xyz":
            s = QDoubleSpinBox(); s.setRange(-5000, 5000); s.setDecimals(1); s.setPrefix(f"{name} ")
            self.go_xyz.append(s); gl.addWidget(s)
        b = QPushButton("Go")
        b.clicked.connect(lambda: self.goRequested.emit(*(s.value() for s in self.go_xyz)))
        gl.addWidget(b)
        lay.addWidget(go)

    @staticmethod
    def table(title, rows):
        cells = "".join(f"<tr><td style='padding-right:10px'>{k}</td><td>{v}</td></tr>" for k, v in rows)
        return f"<b>{title}</b><table style='font-family: Consolas, monospace;'>{cells}</table>"


def _fmt(p):
    return f"({p[0]:8.1f}, {p[1]:8.1f}, {p[2]:8.1f})"


class ExportDialog(QDialog):
    """Choose what to export (original / registered), formats (DICOM, RAW) and destination."""

    def __init__(self, fixed: str, moving: str | None, parent=None, start_dir=""):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.fixed, self.moving = fixed, moving
        lay = QVBoxLayout(self)
        form = QFormLayout()

        self.what = QComboBox()
        if moving:
            self.what.addItem(f"Registered {moving}  (resampled onto {fixed} grid)", "registered")
            self.what.addItem(f"{fixed} + registered {moving}", "both")
        self.what.addItem(f"{fixed}  (fixed image, original)", "fixed")
        if moving:
            self.what.addItem(f"{moving}  (original grid, no registration)", "moving")
        form.addRow("Image:", self.what)

        fmt = QHBoxLayout()
        self.dicom_cb = QCheckBox("DICOM series"); self.dicom_cb.setChecked(True)
        self.raw_cb = QCheckBox("RAW (.raw + .mhd header)"); self.raw_cb.setChecked(True)
        fmt.addWidget(self.dicom_cb); fmt.addWidget(self.raw_cb); fmt.addStretch(1)
        form.addRow("Format:", fmt)

        self.raw_type = QComboBox(); self.raw_type.addItems(["float32", "int16", "uint16"])
        self.raw_cb.toggled.connect(self.raw_type.setEnabled)
        form.addRow("RAW data type:", self.raw_type)

        self.prefix = QLineEdit(f"{moving}_reg_to_{fixed}" if moving else fixed)
        form.addRow("Name prefix:", self.prefix)

        row = QHBoxLayout()
        self.folder = QLineEdit(start_dir)
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse)
        row.addWidget(self.folder, 1); row.addWidget(b)
        form.addRow("Output folder:", row)
        lay.addLayout(form)

        note = QLabel("<small>Registered images use the matrix currently in effect (direction, inversion "
                      "and the Apply checkbox) and keep the fixed image's geometry and Frame of "
                      "Reference. The matrix used is saved alongside as a .txt file.</small>")
        note.setWordWrap(True)
        lay.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(560, 0)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.folder.text())
        if d:
            self.folder.setText(d)

    def _accept(self):
        if not (self.dicom_cb.isChecked() or self.raw_cb.isChecked()):
            QMessageBox.warning(self, "Export", "Choose at least one format."); return
        if not self.folder.text().strip():
            QMessageBox.warning(self, "Export", "Choose an output folder."); return
        self.accept()


# =========================================================================== image composition
def _unit(img, level, width):
    if img is None:
        return None
    lo = level - width / 2.0
    u = (img - lo) / max(width, 1e-6)
    return np.nan_to_num(np.clip(u, 0, 1), nan=0.0)


def _hot(u):
    return np.stack([np.clip(3 * u, 0, 1), np.clip(3 * u - 1, 0, 1), np.clip(3 * u - 2, 0, 1)], -1)


def compose(ct, mr, alpha, mode, checker_px=32):
    """ct/mr: unit images (or None). alpha: 0 -> only CT, 1 -> only MR."""
    if ct is None and mr is None:
        return None
    if ct is None:
        ct, alpha = np.zeros_like(mr), 1.0
    if mr is None:
        mr, alpha = np.zeros_like(ct), 0.0
    if mode == "Checkerboard":
        h, w = ct.shape
        yy, xx = np.mgrid[0:h, 0:w]
        mask = ((yy // checker_px + xx // checker_px) % 2).astype(bool)
        g = np.where(mask, mr, ct)
        rgb = np.repeat(g[..., None], 3, -1)
    elif mode == "Color fusion (MR hot)":
        rgb = (1 - alpha) * np.repeat(ct[..., None], 3, -1) + alpha * _hot(mr)
    elif mode == "Magenta / green":
        # CT in green, MR in magenta; aligned structures turn gray/white.
        rgb = np.stack([mr * alpha * 2, ct * (1 - alpha) * 2, mr * alpha * 2], -1)
    else:  # Gray blend
        g = (1 - alpha) * ct + alpha * mr
        rgb = np.repeat(g[..., None], 3, -1)
    return np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def to_qimage(rgb: np.ndarray) -> QImage:
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


# =========================================================================== main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT ⟷ MR Registration Viewer")
        self.resize(1600, 900)
        self.ct: Volume | None = None
        self.mr: Volume | None = None
        self.reg: Registration | None = None
        self.matrix = np.eye(4)          # as shown in the table
        self.file_matrix: np.ndarray | None = None
        self.grid: DisplayGrid | None = None
        self.grid_ref: str | None = None
        self.cursor = None               # grid index (float ix, iy, iz)
        self._updating_table = False

        self._build_ui()
        self._build_menu()
        self._refresh_info()

    # ------------------------------------------------------------------ UI construction
    def _build_ui(self):
        self.panels = {v: ViewPanel(v) for v in VIEWS}
        self.coords = CoordPanel()
        self.coords.goRequested.connect(self._go_to_world)
        # 2x2: axial | coordinates / coronal | sagittal  (nested splitters so cells can be resized)
        top, bottom = QSplitter(Qt.Horizontal), QSplitter(Qt.Horizontal)
        top.addWidget(self.panels["axial"]); top.addWidget(self.coords)
        bottom.addWidget(self.panels["coronal"]); bottom.addWidget(self.panels["sagittal"])
        split = QSplitter(Qt.Vertical)
        split.addWidget(top); split.addWidget(bottom)
        self._splitters = (top, bottom, split)
        for sp in self._splitters:
            for i in range(2):
                sp.setStretchFactor(i, 1)
        self.coords.setSizePolicy(self.panels["axial"].sizePolicy())
        top.splitterMoved.connect(lambda pos, i: bottom.setSizes(top.sizes()))
        bottom.splitterMoved.connect(lambda pos, i: top.setSizes(bottom.sizes()))
        for v in VIEWS:
            p = self.panels[v]
            p.slider.valueChanged.connect(lambda val, v=v: self._on_slider(v, val))
            p.canvas.sliceDelta.connect(self._on_slice_delta)
            p.canvas.clicked.connect(self._on_click)
            p.canvas.hovered.connect(self._on_hover)

        # Blend slider bar under the views
        self.blend = QSlider(Qt.Horizontal)
        self.blend.setRange(0, 100)
        self.blend.setValue(50)
        self.blend.valueChanged.connect(self._on_blend)
        self.blend_lbl = QLabel()
        bar = QHBoxLayout()
        ct_l, mr_l = QLabel("<b>CT</b>"), QLabel("<b>MR</b>")
        bar.addWidget(ct_l); bar.addWidget(self.blend, 1); bar.addWidget(mr_l); bar.addWidget(self.blend_lbl)
        self.mode = QComboBox()
        self.mode.addItems(["Gray blend", "Color fusion (MR hot)", "Magenta / green", "Checkerboard"])
        self.mode.currentTextChanged.connect(lambda _: self.render_all())
        self.checker = QSpinBox(); self.checker.setRange(4, 256); self.checker.setValue(32)
        self.checker.setSuffix(" px"); self.checker.setToolTip("Checkerboard tile size")
        self.checker.valueChanged.connect(lambda _: self.render_all())
        bar.addSpacing(20); bar.addWidget(QLabel("Overlay:")); bar.addWidget(self.mode); bar.addWidget(self.checker)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addWidget(split, 1)
        lay.addLayout(bar)
        self.setCentralWidget(central)

        # ---- control dock
        dock = QDockWidget("Controls", self)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        body = QWidget()
        dl = QVBoxLayout(body)

        g = QGroupBox("Data")
        gl = QVBoxLayout(g)
        row = QHBoxLayout()
        for text, slot in (("Load CT…", self.load_ct), ("Load MR…", self.load_mr), ("Load REG…", self.load_reg)):
            b = QPushButton(text); b.clicked.connect(slot); row.addWidget(b)
        gl.addLayout(row)
        b = QPushButton("Load folder (auto-detect CT, MR, REG)…"); b.clicked.connect(self.load_folder)
        gl.addWidget(b)
        b = QPushButton("Export (DICOM / RAW)…"); b.clicked.connect(self.export_dialog)
        gl.addWidget(b)
        self.info = QLabel(); self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        gl.addWidget(self.info)
        dl.addWidget(g)

        g = QGroupBox("Registration")
        fl = QVBoxLayout(g)
        self.apply_cb = QCheckBox("Apply registration")
        self.apply_cb.toggled.connect(lambda _: self.render_all())
        fl.addWidget(self.apply_cb)
        form = QFormLayout()
        self.direction = QComboBox()
        self.direction.addItems(["MR → CT", "CT → MR"])
        self.direction.setToolTip("Which coordinates the matrix maps points FROM and TO.\n"
                                  "DICOM REG convention: registered (moving) FoR → reference FoR.")
        self.direction.currentIndexChanged.connect(lambda _: (self._refresh_matrix_info(), self.render_all()))
        form.addRow("Matrix maps:", self.direction)
        self.reference = QComboBox()
        self.reference.addItems(["CT  (MR resampled onto CT)", "MR  (CT resampled onto MR)"])
        self.reference.currentIndexChanged.connect(lambda _: self._rebuild_grid())
        form.addRow("Fixed image:", self.reference)
        fl.addLayout(form)

        self.table = QTableWidget(4, 4)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().hide(); self.table.verticalHeader().hide()
        self.table.setFixedHeight(130)
        self.table.itemChanged.connect(self._on_table_edit)
        fl.addWidget(self.table)
        row = QHBoxLayout()
        for text, slot in (("Invert matrix", self.invert_matrix), ("Reset to file", self.reset_matrix),
                           ("Identity", self.identity_matrix)):
            b = QPushButton(text); b.clicked.connect(slot); row.addWidget(b)
        fl.addLayout(row)
        row = QHBoxLayout()
        b = QPushButton("Save matrix…"); b.clicked.connect(self.save_matrix); row.addWidget(b)
        b = QPushButton("Load matrix…"); b.clicked.connect(self.load_matrix_txt); row.addWidget(b)
        fl.addLayout(row)
        self.matrix_info = QLabel(); self.matrix_info.setWordWrap(True)
        fl.addWidget(self.matrix_info)
        dl.addWidget(g)

        g = QGroupBox("Window / level")
        form = QFormLayout(g)
        self.ct_preset = QComboBox(); self.ct_preset.addItems(list(CT_PRESETS))
        self.ct_preset.currentTextChanged.connect(self._apply_ct_preset)
        form.addRow("CT preset:", self.ct_preset)
        self.ct_level, self.ct_width = self._spin(-2000, 4000, 40), self._spin(1, 8000, 400)
        form.addRow("CT level:", self.ct_level); form.addRow("CT width:", self.ct_width)
        self.mr_level, self.mr_width = self._spin(-1e5, 1e5, 500), self._spin(1, 2e5, 1000)
        form.addRow("MR level:", self.mr_level); form.addRow("MR width:", self.mr_width)
        b = QPushButton("Auto MR window"); b.clicked.connect(self._auto_mr_window)
        form.addRow(b)
        self.cross_cb = QCheckBox("Show crosshair"); self.cross_cb.setChecked(True)
        self.cross_cb.toggled.connect(self._on_cross_toggle)
        form.addRow(self.cross_cb)
        dl.addWidget(g)

        help_lbl = QLabel(
            "<small><b>Mouse:</b> wheel = scroll slices · left click/drag = move crosshair · "
            "Ctrl+wheel = zoom · right/middle drag = pan · right double-click = reset view<br>"
            "<b>Keys:</b> ↑/↓ = slice · Space = swap CT/MR emphasis · R = toggle registration · I = invert</small>")
        help_lbl.setWordWrap(True)
        dl.addWidget(help_lbl)
        dl.addStretch(1)

        scroll = QScrollArea(); scroll.setWidget(body); scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self._write_table()
        self._on_blend(self.blend.value())

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_sized", False):
            self._sized = True
            top, bottom, vert = self._splitters
            half = (top.width() + bottom.width()) // 4 or 400
            top.setSizes([half, half]); bottom.setSizes([half, half])
            h = vert.height() // 2 or 400
            vert.setSizes([h, h])

    def _spin(self, lo, hi, val):
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(1); s.setValue(val)
        s.setKeyboardTracking(False)
        s.valueChanged.connect(lambda _: self.render_all())
        return s

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, slot, key in (("Load CT series…", self.load_ct, "Ctrl+1"), ("Load MR series…", self.load_mr, "Ctrl+2"),
                                ("Load registration (REG)…", self.load_reg, "Ctrl+3"),
                                ("Load folder (auto-detect)…", self.load_folder, "Ctrl+O")):
            a = QAction(text, self); a.setShortcut(QKeySequence(key)); a.triggered.connect(slot); m.addAction(a)
        m.addSeparator()
        a = QAction("Export (DICOM / RAW)…", self); a.setShortcut(QKeySequence("Ctrl+E"))
        a.triggered.connect(self.export_dialog); m.addAction(a)
        m.addSeparator()
        a = QAction("Quit", self); a.setShortcut(QKeySequence("Ctrl+Q")); a.triggered.connect(self.close); m.addAction(a)
        for key, slot in (("Space", self._swap_blend), ("R", self.apply_cb.toggle), ("I", self.invert_matrix)):
            a = QAction(self); a.setShortcut(QKeySequence(key)); a.triggered.connect(slot); self.addAction(a)

    # ------------------------------------------------------------------ loading
    def _progress(self, title):
        dlg = QProgressDialog(title, "Cancel", 0, 100, self)
        dlg.setWindowModality(Qt.WindowModal); dlg.setMinimumDuration(300)

        def cb(done, total, msg):
            dlg.setLabelText(msg); dlg.setMaximum(max(total, 1)); dlg.setValue(done)
            QApplication.processEvents()
            return not dlg.wasCanceled()
        return dlg, cb

    def _scan(self, folder):
        dlg, cb = self._progress("Scanning folder…")
        try:
            return scan_folder(folder, cb)
        finally:
            dlg.close()

    def _pick_series(self, series, modality):
        cands = [s for s in series.values() if s.modality == modality] or list(series.values())
        if not cands:
            return None
        cands.sort(key=lambda s: -len(s.files))
        if len(cands) == 1:
            return cands[0]
        labels = [s.label() for s in cands]
        choice, ok = QInputDialog.getItem(self, f"Choose {modality} series", "Series:", labels, 0, False)
        return cands[labels.index(choice)] if ok else None

    def _load_series(self, info):
        dlg, cb = self._progress(f"Loading {info.modality}…")
        try:
            return load_volume(info.files, cb)
        finally:
            dlg.close()

    def _load_modality(self, modality, folder=None, series=None):
        if series is None:
            folder = folder or QFileDialog.getExistingDirectory(self, f"Select {modality} DICOM folder")
            if not folder:
                return
            series, _ = self._scan(folder)
        info = self._pick_series(series, modality)
        if info is None:
            QMessageBox.warning(self, "No images", f"No DICOM image series found for {modality}.")
            return
        try:
            vol = self._load_series(info)
        except Exception as ex:
            QMessageBox.critical(self, "Load failed", str(ex)); return
        if vol.warnings:
            self.statusBar().showMessage(" | ".join(vol.warnings), 15000)
        if modality == "CT":
            self.ct = vol
        else:
            self.mr = vol
            self._auto_mr_window()
        self._after_data_change()

    def load_ct(self, *_):
        self._load_modality("CT")

    def load_mr(self, *_):
        self._load_modality("MR")

    def load_reg(self, *_, path=None):
        path = path or QFileDialog.getOpenFileName(self, "Select DICOM registration file", "",
                                                   "DICOM (*.dcm *);;All files (*)")[0]
        if not path:
            return
        try:
            self.reg = load_registration(path)
        except Exception as ex:
            QMessageBox.critical(self, "Registration load failed", str(ex)); return
        if self.reg.warnings:
            QMessageBox.information(self, "Registration", "\n".join(self.reg.warnings))
        self._auto_matrix()
        self.apply_cb.setChecked(True)
        self._after_data_change()

    def load_folder(self, *_, folder=None):
        folder = folder or QFileDialog.getExistingDirectory(self, "Select folder containing CT, MR and REG")
        if not folder:
            return
        series, regs = self._scan(folder)
        if any(s.modality == "CT" for s in series.values()):
            self._load_modality("CT", series=series)
        if any(s.modality == "MR" for s in series.values()):
            self._load_modality("MR", series=series)
        if regs:
            path = regs[0]
            if len(regs) > 1:
                names = [os.path.relpath(r, folder) for r in regs]
                choice, ok = QInputDialog.getItem(self, "Choose registration", "REG file:", names, 0, False)
                if not ok:
                    return
                path = regs[names.index(choice)]
            self.load_reg(path=path)

    # ------------------------------------------------------------------ matrix handling
    def _auto_matrix(self):
        """Pick the matrix from the REG file, matched to the loaded CT/MR by Frame of Reference."""
        if self.reg is None:
            return
        self._match_msg = ""
        m = None
        if self.ct is not None and self.mr is not None:
            m = self.reg.matrix_from_to(self.mr.frame_of_reference_uid, self.ct.frame_of_reference_uid)
        if m is not None:
            self._match_msg = "Matched to CT/MR by Frame of Reference UID."
        else:
            m = self.reg.primary_matrix()
            self._match_msg = ("⚠ Could not match REG Frame of Reference UIDs to the loaded CT/MR"
                               if self.ct is not None and self.mr is not None else
                               "Load both CT and MR to match by Frame of Reference.")
            self._match_msg += " Using the first non-identity matrix, assumed MR → CT."
        self.file_matrix = m.copy()
        self.matrix = m.copy()
        self.direction.setCurrentIndex(0)  # MR → CT
        self._write_table()

    def t_ct2mr(self) -> np.ndarray:
        """Transform from CT patient coords to MR patient coords currently in effect."""
        if not self.apply_cb.isChecked():
            return np.eye(4)
        try:
            return self.matrix.copy() if self.direction.currentIndex() == 1 else np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError:
            return np.eye(4)

    def transforms(self):
        """(T_ref->CT, T_ref->MR) for the current reference."""
        T = self.t_ct2mr()
        if self.grid_ref == "CT":
            return np.eye(4), T
        return np.linalg.inv(T), np.eye(4)

    def _write_table(self):
        self._updating_table = True
        for r in range(4):
            for c in range(4):
                it = QTableWidgetItem(f"{self.matrix[r, c]:.6g}")
                it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, it)
        self._updating_table = False
        self._refresh_matrix_info()

    def _on_table_edit(self, item):
        if self._updating_table:
            return
        try:
            self.matrix[item.row(), item.column()] = float(item.text())
        except ValueError:
            self._write_table(); return
        self._refresh_matrix_info()
        self.render_all()

    def invert_matrix(self):
        try:
            self.matrix = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError:
            QMessageBox.warning(self, "Invert", "Matrix is singular."); return
        self._write_table()
        self.statusBar().showMessage("Matrix inverted (direction label unchanged).", 4000)
        self.render_all()

    def reset_matrix(self):
        if self.file_matrix is not None:
            self.matrix = self.file_matrix.copy()
            self.direction.setCurrentIndex(0)
            self._write_table(); self.render_all()

    def identity_matrix(self):
        self.matrix = np.eye(4); self._write_table(); self.render_all()

    def save_matrix(self):
        path = QFileDialog.getSaveFileName(self, "Save matrix", "matrix.txt", "Text (*.txt)")[0]
        if path:
            header = f"4x4 matrix, maps points {self.direction.currentText()} (DICOM patient coords, mm)"
            np.savetxt(path, self.matrix, fmt="%.10g", header=header, encoding="utf-8")

    def load_matrix_txt(self):
        path = QFileDialog.getOpenFileName(self, "Load matrix", "", "Text (*.txt);;All files (*)")[0]
        if not path:
            return
        try:
            m = np.loadtxt(path).reshape(4, 4)
        except Exception as ex:
            QMessageBox.critical(self, "Load matrix", str(ex)); return
        self.matrix = m; self._write_table(); self.apply_cb.setChecked(True); self.render_all()

    def _refresh_matrix_info(self):
        t, ang, det = rigid_params(self.matrix)
        txt = (f"Translation: ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}) mm<br>"
               f"Rotation x/y/z: ({ang[0]:.2f}°, {ang[1]:.2f}°, {ang[2]:.2f}°)  det={det:.4f}")
        if abs(det - 1) > 1e-3:
            txt += " <span style='color:#c80'>(not purely rigid)</span>"
        if self.reg is not None:
            txt = f"<i>{getattr(self, '_match_msg', '')}</i><br>" + txt
        self.matrix_info.setText(txt)

    # ------------------------------------------------------------------ state changes
    def _after_data_change(self):
        if self.reg is not None and self.file_matrix is None:
            self._auto_matrix()
        elif self.reg is not None and self.ct is not None and self.mr is not None and \
                not getattr(self, "_match_msg", "").startswith("Matched"):
            self._auto_matrix()
        self._refresh_info()
        self._rebuild_grid()

    def _refresh_info(self):
        parts = []
        for name, v in (("CT", self.ct), ("MR", self.mr)):
            parts.append(f"<b>{name}:</b> " + (v.summary().replace("\n", "<br>") if v else "not loaded"))
        if self.reg:
            fr = ", ".join("…" + it.frame_of_reference_uid[-8:] for it in self.reg.items)
            parts.append(f"<b>REG:</b> {os.path.basename(self.reg.path)}<br>"
                         f"ref FoR …{self.reg.frame_of_reference_uid[-8:]}; items: {fr}")
        else:
            parts.append("<b>REG:</b> not loaded")
        self.info.setText("<br>".join(parts))

    def _rebuild_grid(self):
        want = "CT" if self.reference.currentIndex() == 0 else "MR"
        vol = {"CT": self.ct, "MR": self.mr}[want]
        if vol is None:
            want = "MR" if want == "CT" else "CT"
            vol = {"CT": self.ct, "MR": self.mr}[want]
        if vol is None:
            return
        # Keep the cursor at the same physical location when switching reference.
        old_world = None
        if self.grid is not None and self.cursor is not None:
            old_world = np.append(self.grid.world(self.cursor), 1.0)
            if self.grid_ref != want:
                T = self.t_ct2mr()
                old_world = (T if want == "MR" else np.linalg.inv(T)) @ old_world
        self.grid, self.grid_ref = DisplayGrid(vol), want
        if old_world is not None:
            self.cursor = np.clip(self.grid.index(old_world), 0, self.grid.shape - 1)
        else:
            self.cursor = (self.grid.shape - 1) / 2.0
        for v, p in self.panels.items():
            p.slider.blockSignals(True)
            p.slider.setRange(0, self.grid.n_slices(v) - 1)
            p.slider.blockSignals(False)
        self.render_all()

    # ------------------------------------------------------------------ rendering
    def render_all(self, views=VIEWS):
        if self.grid is None:
            return
        T_ct, T_mr = self.transforms()
        a = self.blend.value() / 100.0
        mode = self.mode.currentText()
        self.checker.setVisible(mode == "Checkerboard")
        c = self.grid.clamp(self.cursor)
        for v in views:
            s, r, col = self.grid.view_coords(v, c)
            plane = self.grid.plane(v, s)
            ct = mr = None
            if self.ct is not None:
                ct = _unit(sample_plane(self.ct, T_ct, *plane), self.ct_level.value(), self.ct_width.value())
            if self.mr is not None:
                mr = _unit(sample_plane(self.mr, T_mr, *plane), self.mr_level.value(), self.mr_width.value())
            rgb = compose(ct, mr, a, mode, self.checker.value())
            p = self.panels[v]
            p.canvas.cross = self.grid.view_coords(v, self.cursor)[1:]
            p.canvas.set_image(to_qimage(rgb), self.grid.pixel_mm(v))
            p.slider.blockSignals(True); p.slider.setValue(int(s)); p.slider.blockSignals(False)
            p.title.setText(f"{v.capitalize()}   {s + 1} / {self.grid.n_slices(v)}"
                            f"   (fixed: {self.grid_ref})")
        self._update_cross_info()

    # ------------------------------------------------------------------ event handlers
    def _on_slider(self, view, val):
        if self.grid is None:
            return
        self.cursor[SLICE_AXIS[view]] = val
        self.render_all()

    def _on_slice_delta(self, view, d):
        if self.grid is None:
            return
        ax = SLICE_AXIS[view]
        # Axial: wheel-up moves superior (+z). Coronal/sagittal slider index runs along +y/+x.
        self.cursor[ax] = np.clip(round(self.cursor[ax]) + d, 0, self.grid.shape[ax] - 1)
        self.render_all()

    def _on_click(self, view, r, c):
        if self.grid is None:
            return
        s = self.grid.view_coords(view, self.grid.clamp(self.cursor))[0]
        ijk = self.grid.from_view(view, s, r, c)
        self.cursor = np.clip(ijk, 0, self.grid.shape - 1)
        self.render_all()
        self._on_hover(view, r, c)

    # ------------------------------------------------------------------ export
    def export_dialog(self, *_):
        if self.grid is None:
            QMessageBox.information(self, "Export", "Load an image first."); return
        vols = {"CT": self.ct, "MR": self.mr}
        fixed_name = self.grid_ref
        moving_name = "MR" if fixed_name == "CT" else "CT"
        if vols[moving_name] is None:
            moving_name = None
        start = getattr(self, "_last_export_dir", "")
        if not start and vols[fixed_name].source_files:
            start = os.path.dirname(os.path.dirname(vols[fixed_name].source_files[0]))
        dlg = ExportDialog(fixed_name, moving_name, self, start)
        if dlg.exec() != QDialog.Accepted:
            return
        self._last_export_dir = out = dlg.folder.text().strip()
        try:
            written = self._do_export(dlg.what.currentData(), fixed_name, moving_name, out,
                                      dlg.prefix.text().strip() or "export",
                                      dlg.dicom_cb.isChecked(), dlg.raw_cb.isChecked(),
                                      dlg.raw_type.currentText())
        except Exception as ex:
            QMessageBox.critical(self, "Export failed", str(ex)); return
        QMessageBox.information(self, "Export", "Exported to " + out + ":\n\n" + "\n".join(written))

    def _do_export(self, what, fixed_name, moving_name, out, prefix, dicom, raw, raw_type):
        vols = {"CT": self.ct, "MR": self.mr}
        fixed = vols[fixed_name]
        windows = {"CT": (self.ct_level.value(), self.ct_width.value()),
                   "MR": (self.mr_level.value(), self.mr_width.value())}
        os.makedirs(out, exist_ok=True)
        jobs = []  # (volume, name, description, derivation)
        if what in ("fixed", "both"):
            jobs.append((fixed, f"{prefix}_{fixed_name}", fixed.description or fixed_name, ""))
        if what == "moving":
            mv = vols[moving_name]
            jobs.append((mv, f"{prefix}_{moving_name}", mv.description or moving_name, ""))
        written = []
        if what in ("registered", "both"):
            moving = vols[moving_name]
            T_ct, T_mr = self.transforms()
            T = T_mr if moving_name == "MR" else T_ct
            dlg, cb = self._progress(f"Resampling {moving_name} onto {fixed_name}…")
            try:
                arr = resample_to(moving, fixed, T, progress=cb)
            finally:
                dlg.close()
            desc = f"{moving.description or moving_name} reg to {fixed_name}"
            applied = self.apply_cb.isChecked()
            deriv = (f"Resampled onto {fixed_name} grid; {fixed_name}->{moving_name} patient-coordinate "
                     f"matrix: " + " ".join(f"{v:.6g}" for v in T.ravel()) if applied else
                     f"Resampled onto {fixed_name} grid without registration (identity).")
            jobs.append((derived_volume(arr, fixed, moving, desc), f"{prefix}_{moving_name}_registered",
                         desc, deriv))
            mpath = os.path.join(out, f"{prefix}_matrix.txt")
            np.savetxt(mpath, T, fmt="%.10g", encoding="utf-8", header=(
                f"Matrix used for export: maps {fixed_name} (fixed) patient coords -> {moving_name} "
                f"(moving) patient coords, mm, LPS.\nRegistration applied: {applied}. "
                f"Table matrix maps {self.direction.currentText()}:\n" +
                "\n".join(" ".join(f"{v:.10g}" for v in r) for r in self.matrix)))
            written.append(mpath)
        for n, (vol, name, desc, deriv) in enumerate(jobs):
            if raw:
                written += write_raw(vol, os.path.join(out, name + ".mhd"), raw_type)
            if dicom:
                d = os.path.join(out, name + "_DICOM")
                # Patient/study tags from the series whose Frame of Reference the output is in.
                tpl_vol = fixed if vol.frame_of_reference_uid == fixed.frame_of_reference_uid else vol
                template = tpl_vol.source_files[0] if tpl_vol.source_files else None
                dlg, cb = self._progress(f"Writing {name}…")
                try:
                    files = write_dicom_series(
                        vol, d, template, desc,
                        window=windows.get(vol.modality), series_number=9001 + n,
                        derivation=deriv, progress=cb)
                finally:
                    dlg.close()
                written.append(f"{d}  ({len(files)} files)")
        return written

    def _point_rows(self, w):
        """Coordinate/value rows for a point w given in fixed-image patient coords."""
        T_ct, T_mr = self.transforms()
        rows = [(f"Fixed ({self.grid_ref}) mm", _fmt(w))]
        for name, vol, T, unit in (("CT", self.ct, T_ct, " HU"), ("MR", self.mr, T_mr, "")):
            if vol is None:
                continue
            p = (T @ np.append(w, 1.0))[:3]
            vox = (np.linalg.inv(vol.affine) @ np.append(p, 1.0))[:3]
            val = sample_point(vol, T, w)
            rows += [(f"{name} patient mm", _fmt(p)),
                     (f"{name} voxel i,j,k", _fmt(vox)),
                     (f"{name} value", "outside" if np.isnan(val) else f"{val:.1f}{unit}")]
        return rows

    def _update_cross_info(self):
        if self.grid is None:
            return
        w = self.grid.world(self.cursor)
        self.coords.cross_lbl.setText(CoordPanel.table("Crosshair", self._point_rows(w)))
        for s, v in zip(self.coords.go_xyz, w):
            s.blockSignals(True); s.setValue(float(v)); s.blockSignals(False)

    def _go_to_world(self, x, y, z):
        if self.grid is None:
            return
        self.cursor = np.clip(self.grid.index([x, y, z]), 0, self.grid.shape - 1)
        self.render_all()

    def _on_hover(self, view, r, c):
        if self.grid is None:
            return
        s = self.grid.view_coords(view, self.grid.clamp(self.cursor))[0]
        ijk = self.grid.from_view(view, s, r, c)
        w = self.grid.world(ijk)
        rows = self._point_rows(w)
        self.coords.hover_lbl.setText(CoordPanel.table(f"Mouse ({view})", rows))
        self.statusBar().showMessage("   ".join(f"{k}: {v.strip()}" for k, v in rows if "voxel" not in k))

    def _on_blend(self, val):
        self.blend_lbl.setText(f"{100 - val}% CT / {val}% MR")
        self.render_all()

    def _swap_blend(self):
        self.blend.setValue(100 - self.blend.value())

    def _on_cross_toggle(self, on):
        for p in self.panels.values():
            p.canvas.show_cross = on
            p.canvas.update()

    def _apply_ct_preset(self, name):
        lvl, wid = CT_PRESETS[name]
        for s, v in ((self.ct_level, lvl), (self.ct_width, wid)):
            s.blockSignals(True); s.setValue(v); s.blockSignals(False)
        self.render_all()

    def _auto_mr_window(self):
        if self.mr is None:
            return
        a = self.mr.array
        sub = a[:: max(1, a.shape[0] // 20), ::4, ::4].ravel()
        sub = sub[np.isfinite(sub)]
        lo, hi = np.percentile(sub, [1, 99.5]) if sub.size else (0, 1)
        for s, v in ((self.mr_level, (lo + hi) / 2), (self.mr_width, max(hi - lo, 1))):
            s.blockSignals(True); s.setValue(float(v)); s.blockSignals(False)
        self.render_all()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="CT/MR DICOM registration viewer")
    ap.add_argument("--ct", help="CT DICOM folder")
    ap.add_argument("--mr", help="MR DICOM folder")
    ap.add_argument("--reg", help="DICOM spatial registration file")
    ap.add_argument("--folder", help="Folder containing CT, MR and REG (auto-detect)")
    args = ap.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    if args.folder:
        w.load_folder(folder=args.folder)
    if args.ct:
        w._load_modality("CT", folder=args.ct)
    if args.mr:
        w._load_modality("MR", folder=args.mr)
    if args.reg:
        w.load_reg(path=args.reg)
    return app.exec()
