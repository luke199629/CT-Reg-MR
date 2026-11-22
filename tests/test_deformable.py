"""Deformable registration check against a known non-rigid displacement.

    python tools/make_sample_data.py sample_data_deform --deform
    python tests/test_deformable.py [method ...]      # elastix convexadam multigradicon

The MR content carries a smooth bump (up to (8, 4, 0) mm around CT point (40,-20,30)) on top
of the rigid REG matrix, so the true fixed->moving displacement u(x) is known. Each method
must reduce the displacement error where the bump is (|u_true| > 2 mm) below rigid-only,
keep the mean error over the body modest, and not fold.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from make_sample_data import bump  # noqa: E402
from ctmr_reg.deformable import DEFAULT_PARAMS, method_available, register  # noqa: E402
from ctmr_reg.dicom_io import load_registration, load_volume, scan_folder  # noqa: E402

ROOT = os.path.join(HERE, "..", "sample_data_deform")


def true_dvf(grid):
    nk, nj, ni = grid.shape
    kk, jj, ii = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
    A = grid.affine
    X = np.stack([A[c, 0] * ii + A[c, 1] * jj + A[c, 2] * kk + A[c, 3] for c in range(3)], -1)
    u = np.zeros_like(X)
    for _ in range(30):                     # solve x + u + b(x + u) = x
        u = -bump(X + u)
    body = (X[..., 0] / 115) ** 2 + (X[..., 1] / 85) ** 2 + (X[..., 2] / 105) ** 2 < 1
    return u, body


def main(methods):
    series, regs = scan_folder(ROOT)
    by_mod = {s.modality: s for s in series.values()}
    ct, mr = load_volume(by_mod["CT"].files), load_volume(by_mod["MR"].files)
    reg = load_registration(regs[0])
    T = np.linalg.inv(reg.matrix_from_to(mr.frame_of_reference_uid, ct.frame_of_reference_uid))
    ok = True
    for m in methods:
        avail, why = method_available(m)
        if not avail:
            print(f"{m}: skipped ({why})"); continue
        r = register(m, ct, mr, "CT", "MR", T, DEFAULT_PARAMS[m], spacing_mm=2.0)
        u, body = true_dvf(r.grid)
        near = body & (np.linalg.norm(u, axis=-1) > 2)
        err = np.linalg.norm(r.dvf - u, axis=-1)
        rigid_near = np.linalg.norm(u, axis=-1)[near].mean()
        s = r.stats()
        print(f"{m:14s} error where bump >2 mm: {err[near].mean():.2f} mm (rigid only {rigid_near:.2f}); "
              f"body mean {err[body].mean():.2f} mm; folding {s['folding_pct']:.3f}%; {r.seconds:.0f} s")
        ok &= err[near].mean() < rigid_near and err[body].mean() < 3.5 and s["folding_pct"] < 0.5
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["elastix", "convexadam", "multigradicon"]))
