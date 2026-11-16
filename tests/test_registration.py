"""End-to-end check on the synthetic data: the REG matrix must align MR to CT.

    python tools/make_sample_data.py sample_data
    python tests/test_registration.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ctmr_reg.dicom_io import load_registration, load_volume, scan_folder  # noqa: E402
from ctmr_reg.geometry import DisplayGrid, sample_plane  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "sample_data")


def overlap(T_ct2mr, ct, mr, grid):
    """Dice of 'bright sphere' masks (CT 300 HU sphere vs MR 1400 sphere) over all axial slices."""
    inter = a_sum = b_sum = 0
    for s in range(grid.n_slices("axial")):
        plane = grid.plane("axial", s)
        c = sample_plane(ct, np.eye(4), *plane)
        m = sample_plane(mr, T_ct2mr, *plane)
        a = (c > 250) & (c < 350)
        b = (m > 1200) & (m < 1600)
        inter += (a & b).sum(); a_sum += a.sum(); b_sum += b.sum()
    return 2 * inter / max(a_sum + b_sum, 1)


def main():
    series, regs = scan_folder(ROOT)
    by_mod = {s.modality: s for s in series.values()}
    ct, mr = load_volume(by_mod["CT"].files), load_volume(by_mod["MR"].files)
    reg = load_registration(regs[0])
    gt = np.loadtxt(os.path.join(ROOT, "ground_truth_MR_to_CT.txt"))

    m = reg.matrix_from_to(mr.frame_of_reference_uid, ct.frame_of_reference_uid)
    assert m is not None, "FoR matching failed"
    assert np.allclose(m, gt), "matrix != ground truth"
    assert np.allclose(reg.matrix_from_to(ct.frame_of_reference_uid, mr.frame_of_reference_uid), np.linalg.inv(gt))
    print("CT", ct.array.shape, ct.spacing, "MR", mr.array.shape, mr.spacing)

    grid = DisplayGrid(ct)
    T_ct2mr = np.linalg.inv(m)          # matrix maps MR->CT, so CT->MR is its inverse
    d_good = overlap(T_ct2mr, ct, mr, grid)
    d_none = overlap(np.eye(4), ct, mr, grid)
    d_wrong = overlap(m, ct, mr, grid)  # direction flipped
    print(f"Dice registered={d_good:.3f}  unregistered={d_none:.3f}  inverted={d_wrong:.3f}")
    assert d_good > 0.85 and d_none < 0.6 and d_wrong < 0.6
    print("OK")


if __name__ == "__main__":
    main()
