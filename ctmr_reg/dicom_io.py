"""DICOM reading: image series -> Volume, and Spatial Registration objects -> matrices.

Coordinate conventions
----------------------
* Volume.array is indexed [k, j, i] = [slice, row, column] (float32, rescaled values).
* Volume.affine is a 4x4 matrix mapping voxel (i, j, k, 1) to DICOM patient
  coordinates (LPS, millimetres).
* A Spatial Registration item's matrix maps points from the item's Frame of
  Reference (the "registered" space) into the registration object's own
  top-level Frame of Reference (the "reference" space), per DICOM PS3.3 C.20.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

REG_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.66.1",  # Spatial Registration Storage
    "1.2.840.10008.5.1.4.1.1.66.3",  # Deformable Spatial Registration Storage
}

ProgressCB = Optional[Callable[[int, int, str], bool]]  # (done, total, msg) -> keep going


# --------------------------------------------------------------------------- data classes
@dataclass
class SeriesInfo:
    series_uid: str
    modality: str
    description: str
    frame_of_reference_uid: str
    patient_name: str
    files: list[str] = field(default_factory=list)

    def label(self) -> str:
        return f"{self.modality}  {self.description or '(no description)'}  [{len(self.files)} files]"


@dataclass
class Volume:
    array: np.ndarray            # [k, j, i] float32
    affine: np.ndarray           # voxel (i,j,k,1) -> LPS mm
    modality: str
    series_uid: str
    frame_of_reference_uid: str
    description: str = ""
    patient_name: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def shape_ijk(self):
        k, j, i = self.array.shape
        return i, j, k

    @property
    def spacing(self):
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    def summary(self) -> str:
        i, j, k = self.shape_ijk
        sx, sy, sz = self.spacing
        return (f"{self.modality} {self.description}\n"
                f"{i}x{j}x{k} vox, {sx:.2f}x{sy:.2f}x{sz:.2f} mm\n"
                f"FoR …{self.frame_of_reference_uid[-12:]}")


@dataclass
class RegistrationItem:
    frame_of_reference_uid: str
    matrix: np.ndarray           # maps item FoR -> registration FoR
    matrix_type: str


@dataclass
class Registration:
    path: str
    frame_of_reference_uid: str  # top-level (reference) FoR
    items: list[RegistrationItem]
    warnings: list[str] = field(default_factory=list)

    def matrix_from_to(self, src_for: str, dst_for: str) -> Optional[np.ndarray]:
        """Matrix mapping patient coords in src_for to dst_for, if both FoRs are known."""
        def to_ref(uid):
            for it in self.items:
                if it.frame_of_reference_uid == uid:
                    return it.matrix
            if uid == self.frame_of_reference_uid:
                return np.eye(4)
            return None
        a, b = to_ref(src_for), to_ref(dst_for)
        if a is None or b is None:
            return None
        return np.linalg.inv(b) @ a

    def primary_matrix(self) -> np.ndarray:
        """The first non-identity matrix (fallback when FoR UIDs can't be matched)."""
        for it in self.items:
            if not np.allclose(it.matrix, np.eye(4)):
                return it.matrix.copy()
        return np.eye(4)


# --------------------------------------------------------------------------- scanning
def _iter_files(folder: str):
    for root, _dirs, files in os.walk(folder):
        for f in files:
            yield os.path.join(root, f)


def scan_folder(folder: str, progress: ProgressCB = None):
    """Read headers of every file under *folder*.

    Returns (series: dict[uid, SeriesInfo], reg_files: list[str]).
    """
    paths = list(_iter_files(folder))
    series: dict[str, SeriesInfo] = {}
    regs: list[str] = []
    for n, p in enumerate(paths):
        if progress and n % 20 == 0 and not progress(n, len(paths), "Scanning headers…"):
            break
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        except (InvalidDicomError, OSError, Exception):
            continue
        sop = str(ds.get("SOPClassUID", ""))
        modality = str(ds.get("Modality", ""))
        if sop in REG_SOP_CLASSES or modality == "REG":
            regs.append(p)
            continue
        if "ImagePositionPatient" not in ds or "ImageOrientationPatient" not in ds:
            continue
        uid = str(ds.get("SeriesInstanceUID", "unknown"))
        if uid not in series:
            series[uid] = SeriesInfo(
                series_uid=uid, modality=modality,
                description=str(ds.get("SeriesDescription", "")),
                frame_of_reference_uid=str(ds.get("FrameOfReferenceUID", "")),
                patient_name=str(ds.get("PatientName", "")))
        series[uid].files.append(p)
    if progress:
        progress(len(paths), len(paths), "Done")
    return series, regs


# --------------------------------------------------------------------------- volume loading
def _pixels(ds, path) -> np.ndarray:
    try:
        return ds.pixel_array
    except Exception:
        # Compressed transfer syntax without a pydicom decoder: let SimpleITK/GDCM do it.
        import SimpleITK as sitk
        return sitk.GetArrayFromImage(sitk.ReadImage(path))[0]


def load_volume(files: list[str], progress: ProgressCB = None) -> Volume:
    headers = []
    for n, p in enumerate(files):
        if progress and n % 20 == 0 and not progress(n, 2 * len(files), "Reading headers…"):
            raise RuntimeError("Cancelled")
        ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        headers.append((p, ds))
    if not headers:
        raise ValueError("No DICOM image files")

    warnings: list[str] = []
    # Keep the dominant orientation / matrix size (drops stray localisers).
    key = lambda ds: (tuple(np.round(np.array(ds.ImageOrientationPatient, float), 3)),
                      int(ds.Rows), int(ds.Columns))
    dominant, _ = Counter(key(ds) for _, ds in headers).most_common(1)[0]
    kept = [(p, ds) for p, ds in headers if key(ds) == dominant]
    if len(kept) < len(headers):
        warnings.append(f"Ignored {len(headers) - len(kept)} files with a different orientation/size.")

    first = kept[0][1]
    iop = np.array(first.ImageOrientationPatient, float)
    rcos, ccos = iop[:3], iop[3:]
    normal = np.cross(rcos, ccos)
    dr, dc = (float(v) for v in first.PixelSpacing)  # row spacing, column spacing

    # Sort along the slice normal; drop duplicate positions (e.g. multiple echoes).
    by_pos = defaultdict(list)
    for p, ds in kept:
        d = float(np.dot(np.array(ds.ImagePositionPatient, float), normal))
        by_pos[round(d, 3)].append((p, ds))
    positions = sorted(by_pos)
    if any(len(v) > 1 for v in by_pos.values()):
        warnings.append("Several images share a slice position (multi-echo/phase?); using the first of each.")
    slices = [by_pos[d][0] for d in positions]

    ipp0 = np.array(slices[0][1].ImagePositionPatient, float)
    if len(slices) > 1:
        ipp_last = np.array(slices[-1][1].ImagePositionPatient, float)
        step = (ipp_last - ipp0) / (len(slices) - 1)
        gaps = np.diff(positions)
        if gaps.size and (gaps.max() - gaps.min()) > 0.01 * max(abs(gaps.mean()), 1e-6) + 1e-3:
            warnings.append(f"Non-uniform slice spacing ({gaps.min():.2f}–{gaps.max():.2f} mm); "
                            "using the average.")
    else:
        st = float(first.get("SliceThickness", 1.0) or 1.0)
        step = normal * st

    affine = np.eye(4)
    affine[:3, 0] = rcos * dc
    affine[:3, 1] = ccos * dr
    affine[:3, 2] = step
    affine[:3, 3] = ipp0

    arr = np.empty((len(slices), int(first.Rows), int(first.Columns)), np.float32)
    for n, (p, _) in enumerate(slices):
        if progress and n % 10 == 0 and not progress(len(files) + n * len(files) // len(slices),
                                                     2 * len(files), "Reading pixels…"):
            raise RuntimeError("Cancelled")
        ds = pydicom.dcmread(p, force=True)
        px = _pixels(ds, p).astype(np.float32)
        slope = float(ds.get("RescaleSlope", 1) or 1)
        inter = float(ds.get("RescaleIntercept", 0) or 0)
        arr[n] = px * slope + inter

    return Volume(array=arr, affine=affine,
                  modality=str(first.get("Modality", "")),
                  series_uid=str(first.get("SeriesInstanceUID", "")),
                  frame_of_reference_uid=str(first.get("FrameOfReferenceUID", "")),
                  description=str(first.get("SeriesDescription", "")),
                  patient_name=str(first.get("PatientName", "")),
                  warnings=warnings)


# --------------------------------------------------------------------------- registration
def _compose(matrix_seq) -> tuple[np.ndarray, str]:
    """Matrices in a Matrix Sequence are applied in order: total = Mn @ … @ M1."""
    total, types = np.eye(4), []
    for ms in matrix_seq:
        m = np.array(ms.FrameOfReferenceTransformationMatrix, float).reshape(4, 4)
        total = m @ total
        types.append(str(ms.get("FrameOfReferenceTransformationMatrixType", "")))
    return total, "+".join(t for t in types if t)


def load_registration(path: str) -> Registration:
    ds = pydicom.dcmread(path, force=True)
    ref_for = str(ds.get("FrameOfReferenceUID", ""))
    items: list[RegistrationItem] = []
    warnings: list[str] = []

    for it in ds.get("RegistrationSequence", []):
        total, types = np.eye(4), []
        for mrs in it.get("MatrixRegistrationSequence", []):
            m, t = _compose(mrs.get("MatrixSequence", []))
            total = m @ total
            types.append(t)
        items.append(RegistrationItem(str(it.get("FrameOfReferenceUID", "")), total, "+".join(types)))

    for it in ds.get("DeformableRegistrationSequence", []):
        total = np.eye(4)
        for tag in ("PreDeformationMatrixRegistrationSequence", "PostDeformationMatrixRegistrationSequence"):
            for sub in it.get(tag, []):
                m = np.array(sub.FrameOfReferenceTransformationMatrix, float).reshape(4, 4)
                total = m @ total
        if "DeformableRegistrationGridSequence" in it:
            warnings.append("Deformable grid present — only the pre/post rigid matrices are used.")
        items.append(RegistrationItem(str(it.get("SourceFrameOfReferenceUID", "")), total, "DEFORMABLE(rigid part)"))

    if not items:
        raise ValueError("No Registration Sequence / matrix found in this file.")
    for it in items:
        if not np.allclose(it.matrix[3], [0, 0, 0, 1]):
            warnings.append("A matrix has a non-affine bottom row.")
    return Registration(path=path, frame_of_reference_uid=ref_for, items=items, warnings=warnings)
