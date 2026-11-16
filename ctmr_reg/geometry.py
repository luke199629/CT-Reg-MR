"""Display grid and on-demand slice resampling.

The display grid is an axis-aligned box in the *reference* image's patient
coordinates (LPS mm), so axial/coronal/sagittal are true anatomical planes even
for oblique acquisitions. Every displayed slice of either volume is sampled from
this grid through a 4x4 transform (reference patient -> volume patient).

Radiological display convention:
  axial    rows = +y (anterior->posterior), cols = +x (patient right->left)
  coronal  rows = -z (superior->inferior),  cols = +x
  sagittal rows = -z,                       cols = +y (anterior->posterior)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .dicom_io import Volume

VIEWS = ("axial", "coronal", "sagittal")
SLICE_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}   # which of ix,iy,iz the slider moves


class DisplayGrid:
    def __init__(self, ref: Volume):
        i, j, k = ref.shape_ijk
        corners = np.array([[a, b, c, 1] for a in (0, i - 1) for b in (0, j - 1) for c in (0, k - 1)], float)
        pts = (ref.affine @ corners.T)[:3].T
        lo, hi = pts.min(0), pts.max(0)
        # Spacing along each patient axis = spacing of the voxel axis most aligned with it.
        cols = ref.affine[:3, :3]
        norms = np.linalg.norm(cols, axis=0)
        sp = np.empty(3)
        for a in range(3):
            v = np.argmax(np.abs(cols[a]) / norms)
            sp[a] = norms[v]
        self.spacing = sp
        self.origin = lo
        self.shape = (np.floor((hi - lo) / sp + 1e-6).astype(int) + 1)  # nx, ny, nz

    # ---- index <-> world
    def world(self, ijk) -> np.ndarray:
        return self.origin + np.asarray(ijk, float) * self.spacing

    def index(self, p) -> np.ndarray:
        return (np.asarray(p, float)[:3] - self.origin) / self.spacing

    def clamp(self, ijk) -> np.ndarray:
        return np.clip(np.round(ijk), 0, self.shape - 1).astype(int)

    # ---- per view geometry
    def n_slices(self, view) -> int:
        return int(self.shape[SLICE_AXIS[view]])

    def plane(self, view, s):
        """Return (origin, row_vec, col_vec, (h, w)) in world mm for slice index s."""
        nx, ny, nz = self.shape
        sx, sy, sz = self.spacing
        ox, oy, oz = self.origin
        ztop = oz + (nz - 1) * sz
        if view == "axial":
            return np.array([ox, oy, oz + s * sz]), np.array([0, sy, 0]), np.array([sx, 0, 0]), (ny, nx)
        if view == "coronal":
            return np.array([ox, oy + s * sy, ztop]), np.array([0, 0, -sz]), np.array([sx, 0, 0]), (nz, nx)
        if view == "sagittal":
            return np.array([ox + s * sx, oy, ztop]), np.array([0, 0, -sz]), np.array([0, sy, 0]), (nz, ny)
        raise ValueError(view)

    def pixel_mm(self, view):
        sx, sy, sz = self.spacing
        return {"axial": (sy, sx), "coronal": (sz, sx), "sagittal": (sz, sy)}[view]  # (row, col)

    def view_coords(self, view, ijk):
        """Grid index (ix,iy,iz) -> (slice, row, col) for this view."""
        ix, iy, iz = ijk
        nz = self.shape[2]
        if view == "axial":
            return iz, iy, ix
        if view == "coronal":
            return iy, nz - 1 - iz, ix
        return ix, nz - 1 - iz, iy

    def from_view(self, view, s, r, c):
        nz = self.shape[2]
        if view == "axial":
            return np.array([c, r, s], float)
        if view == "coronal":
            return np.array([c, s, nz - 1 - r], float)
        return np.array([s, c, nz - 1 - r], float)


def sample_plane(vol: Volume, T: np.ndarray, origin, row_vec, col_vec, hw, order=1) -> np.ndarray:
    """Sample *vol* on a plane given in reference world coords.

    T maps reference patient coords -> this volume's patient coords.
    Points outside the volume are NaN.
    """
    A = np.linalg.inv(vol.affine) @ T
    o = A[:3, :3] @ origin + A[:3, 3]
    dr = A[:3, :3] @ row_vec
    dc = A[:3, :3] @ col_vec
    h, w = hw
    r = np.arange(h, dtype=np.float32)[:, None]
    c = np.arange(w, dtype=np.float32)[None, :]
    i = o[0] + r * dr[0] + c * dc[0]
    j = o[1] + r * dr[1] + c * dc[1]
    k = o[2] + r * dr[2] + c * dc[2]
    arr = vol.array
    if arr.shape[0] == 1:  # single slice: allow half-voxel tolerance in k
        k = np.where(np.abs(k) <= 0.5, 0.0, k)
    out = map_coordinates(arr, [k, j, i], order=order, mode="constant", cval=np.nan, prefilter=False)
    return out.astype(np.float32, copy=False)


def sample_point(vol: Volume, T: np.ndarray, p) -> float:
    A = np.linalg.inv(vol.affine) @ T
    i, j, k = A[:3, :3] @ np.asarray(p, float) + A[:3, 3]
    v = map_coordinates(vol.array, [[k], [j], [i]], order=1, mode="constant", cval=np.nan, prefilter=False)
    return float(v[0])


def rigid_params(M: np.ndarray):
    """Rough decomposition for display: translation (mm), rotation angles (deg, xyz), det."""
    R = M[:3, :3]
    t = M[:3, 3]
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2]); ry = np.arctan2(-R[2, 0], sy); rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1]); ry = np.arctan2(-R[2, 0], sy); rz = 0.0
    return t, np.degrees([rx, ry, rz]), float(np.linalg.det(R))
