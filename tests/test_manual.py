"""Manual registration: drags and nudges in the viewer move the moving image as expected.

    python tools/make_sample_data.py sample_data
    python tests/test_manual.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ctmr_reg.geometry import axis_rotation, moving_motion  # noqa: E402
from ctmr_reg.gui import MainWindow  # noqa: E402
from test_registration import ROOT, overlap  # noqa: E402

app = QApplication.instance() or QApplication([])


def apply(T, p):
    return (T @ np.append(p, 1.0))[:3]


def check_math():
    rng = np.random.default_rng(0)
    T = np.eye(4); T[:3, :3] = axis_rotation(rng.normal(size=3), 7); T[:3, 3] = rng.normal(size=3) * 10
    M = moving_motion(axis_rotation([0, 0, 1], 12), pivot=[5, -3, 20], d=[1, 2, 3])
    T2 = T @ np.linalg.inv(M)
    for x in rng.normal(size=(5, 3)) * 50:
        # what was displayed at x is now displayed at M(x)
        assert np.allclose(apply(T2, apply(M, x)), apply(T, x))
    assert np.allclose(np.linalg.det(T2[:3, :3]), 1)


def mouse(canvas, kind, rc, mods=Qt.NoModifier):
    x, y = canvas._to_widget(*rc)
    btn = Qt.NoButton if kind == QEvent.MouseMove else Qt.LeftButton
    held = Qt.LeftButton if kind != QEvent.MouseButtonRelease else Qt.NoButton
    pos = QPointF(x, y)
    QApplication.sendEvent(canvas, QMouseEvent(kind, pos, canvas.mapToGlobal(pos), btn, held, mods))


def drag(canvas, a, b, mods=Qt.NoModifier):
    mouse(canvas, QEvent.MouseButtonPress, a, mods)
    for t in np.linspace(0, 1, 6)[1:]:
        mouse(canvas, QEvent.MouseMove, (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])), mods)
    mouse(canvas, QEvent.MouseButtonRelease, b, mods)


def check_gui():
    w = MainWindow(); w.resize(1400, 900); w.show(); app.processEvents()
    w.load_folder(folder=ROOT); app.processEvents()
    assert w.ct is not None and w.mr is not None and w.apply_cb.isChecked()
    m_file = w.matrix.copy()
    dice0 = overlap(w.t_ct2mr(), w.ct, w.mr, w.grid)
    w.manual_cb.setChecked(True)

    for ref in (0, 1):                                  # CT fixed, then MR fixed
        w.reference.setCurrentIndex(ref); app.processEvents()
        view = "axial"
        canvas = w.panels[view].canvas
        _, rv, cv, _ = w.grid.plane(view, w.grid.view_coords(view, w.grid.clamp(w.cursor))[0])
        pr, pc = w.grid.pixel_mm(view)
        r0, c0 = (float(v) for v in w.grid.view_coords(view, w.cursor)[1:])

        # translate: 20 rows down, 10 columns right
        T0 = w._t_ref2moving()
        drag(canvas, (r0 + 5, c0 + 5), (r0 + 25, c0 + 15))
        d = 20 * pr * rv / np.linalg.norm(rv) + 10 * pc * cv / np.linalg.norm(cv)
        T1 = w._t_ref2moving()
        x = w.grid.world(w.cursor) + [3, -4, 2]
        assert np.allclose(apply(T1, x + d), apply(T0, x), atol=1e-6), f"translate (ref {w.grid_ref})"

        # rotate a quarter turn about the crosshair (Ctrl+drag)
        drag(canvas, (r0, c0 + 30), (r0 + 30, c0), Qt.ControlModifier)
        T2 = w._t_ref2moving()
        pivot = w.grid.world(w.cursor)
        assert np.allclose(apply(T2, pivot), apply(T1, pivot), atol=1e-6), "pivot must stay put"
        R = (T2 @ np.linalg.inv(T1))[:3, :3]
        assert np.isclose(np.degrees(np.arccos((np.trace(R) - 1) / 2)), 90, atol=0.5), "90° turn"
        assert np.isclose(np.linalg.det(T2[:3, :3]), np.linalg.det(T0[:3, :3]))

        w.undo_matrix(); w.undo_matrix()
        assert np.allclose(w.matrix, m_file), "undo restores the file matrix"

    # nudges: shift 5 mm and Dice drops; undo restores it; redo re-applies
    w.reference.setCurrentIndex(0); app.processEvents()
    w.man_tstep.setValue(5.0)
    T0 = w._t_ref2moving()
    w._manual_nudge(0, "t", +1)
    assert np.allclose(apply(w._t_ref2moving(), np.array([5.0, 0, 0])), apply(T0, np.zeros(3)))
    dice_shift = overlap(w.t_ct2mr(), w.ct, w.mr, w.grid)
    w._manual_nudge(2, "r", +1)
    w.undo_matrix(); w.undo_matrix()
    assert np.allclose(w.matrix, m_file)
    w.redo_matrix()
    assert np.isclose(overlap(w.t_ct2mr(), w.ct, w.mr, w.grid), dice_shift)
    print(f"Dice at file matrix {dice0:.3f}, after 5 mm nudge {dice_shift:.3f}")
    assert dice_shift < dice0 - 0.05

    # start from identity and nudge back with the ground-truth translation: Dice recovers
    w.identity_matrix()
    gt = np.loadtxt(os.path.join(ROOT, "ground_truth_MR_to_CT.txt"))
    w.matrix = gt.copy(); w._write_table()           # (sanity: direction MR->CT as in the file)
    assert np.isclose(overlap(w.t_ct2mr(), w.ct, w.mr, w.grid), dice0)
    w.grab().save(os.path.join(os.path.dirname(__file__), "..", "sample_data", "manual_test.png"))
    assert w.unload_all(confirm=False) and not w._undo and not w.manual_cb.isChecked()
    w.close()


if __name__ == "__main__":
    check_math()
    check_gui()
    print("OK")
