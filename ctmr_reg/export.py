"""Export volumes (original or registered/resampled) as RAW (+ MetaImage header) or DICOM.

The registered export resamples the moving image onto the *fixed* image's native
voxel grid, so the result has exactly the fixed image's geometry and Frame of
Reference and lines up slice-for-slice with it in any other software.
"""
from __future__ import annotations

import datetime
import os
from typing import Callable, Optional

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from .dicom_io import Volume
from .geometry import sample_plane

ProgressCB = Optional[Callable[[int, int, str], bool]]

SOP_FOR_MODALITY = {"CT": "1.2.840.10008.5.1.4.1.1.2", "MR": "1.2.840.10008.5.1.4.1.1.4"}
PATIENT_STUDY_TAGS = (
    "PatientName", "PatientID", "PatientBirthDate", "PatientSex", "PatientAge",
    "StudyInstanceUID", "StudyDate", "StudyTime", "StudyID", "AccessionNumber",
    "ReferringPhysicianName", "StudyDescription", "PatientPosition",
)
ACQUISITION_TAGS = (  # copied from the image's own source series
    "Manufacturer", "ManufacturerModelName", "InstitutionName", "BodyPartExamined",
    "KVP", "ConvolutionKernel", "ScanningSequence", "SequenceVariant", "ScanOptions",
    "MRAcquisitionType", "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle",
    "MagneticFieldStrength", "ImagingFrequency", "ProtocolName", "SeriesDate", "SeriesTime",
)


# --------------------------------------------------------------------------- resampling
def resample_to(moving: Volume, fixed: Volume, T_fixed_to_moving: np.ndarray,
                fill: float | None = None, progress: ProgressCB = None) -> np.ndarray:
    """Resample *moving* onto *fixed*'s voxel grid. Returns array [k, j, i] float32."""
    A = fixed.affine
    ni, nj, nk = fixed.shape_ijk
    if fill is None:
        fill = float(np.nanmin(moving.array))
    out = np.empty((nk, nj, ni), np.float32)
    for k in range(nk):
        if progress and k % 5 == 0 and not progress(k, nk, "Resampling…"):
            raise RuntimeError("Cancelled")
        origin = A[:3, 3] + k * A[:3, 2]
        sl = sample_plane(moving, T_fixed_to_moving, origin, A[:3, 1], A[:3, 0], (nj, ni))
        out[k] = np.where(np.isnan(sl), fill, sl)
    return out


def derived_volume(array: np.ndarray, fixed: Volume, source: Volume, description: str) -> Volume:
    """Wrap a resampled array: geometry/FoR of *fixed*, modality/acquisition of *source*."""
    return Volume(array=array, affine=fixed.affine.copy(), modality=source.modality,
                  series_uid="", frame_of_reference_uid=fixed.frame_of_reference_uid,
                  description=description, patient_name=fixed.patient_name,
                  source_files=source.source_files)


# --------------------------------------------------------------------------- RAW
def write_raw(vol: Volume, path_mhd: str, dtype: str = "float32") -> list[str]:
    """Write <name>.raw (little-endian, x fastest then y then z) and a <name>.mhd header.

    The .mhd (MetaImage) header carries size, spacing, origin and direction in LPS, so the
    pair opens directly in 3D Slicer, ITK-SNAP, ImageJ/Fiji, MeVisLab, etc.
    """
    import SimpleITK as sitk

    base = os.path.splitext(path_mhd)[0]
    arr = vol.array
    if dtype == "int16":
        arr = np.clip(np.round(arr), -32768, 32767).astype(np.int16)
    elif dtype == "uint16":
        arr = np.clip(np.round(arr), 0, 65535).astype(np.uint16)
    else:
        arr = arr.astype(np.float32)
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr))  # [k,j,i] -> size (i,j,k)
    sp = vol.spacing
    img.SetSpacing([float(v) for v in sp])
    img.SetOrigin([float(v) for v in vol.affine[:3, 3]])
    img.SetDirection([float(v) for v in (vol.affine[:3, :3] / sp).ravel()])
    sitk.WriteImage(img, base + ".mhd", useCompression=False)
    return [base + ".mhd", base + ".raw"]


# --------------------------------------------------------------------------- DICOM
def _int_encoding(arr: np.ndarray):
    """Choose int16 storage + RescaleSlope/Intercept that represent *arr*."""
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if lo >= -32768 and hi <= 32767:
        return 1.0, 0.0
    slope = max(hi - lo, 1e-6) / 65000.0
    intercept = (hi + lo) / 2.0
    return slope, intercept


def write_dicom_series(vol: Volume, out_dir: str, fixed_template_file: str | None,
                       description: str, window: tuple[float, float] | None = None,
                       series_number: int = 9001, derivation: str = "",
                       progress: ProgressCB = None) -> list[str]:
    """Write *vol* as a classic one-file-per-slice DICOM series.

    Patient/study tags come from *fixed_template_file* (so the series lands in the same
    study and Frame of Reference as the fixed image); acquisition tags come from the
    volume's own source series.
    """
    os.makedirs(out_dir, exist_ok=True)
    fixed_t = pydicom.dcmread(fixed_template_file, stop_before_pixels=True, force=True) \
        if fixed_template_file else Dataset()
    src_t = pydicom.dcmread(vol.source_files[0], stop_before_pixels=True, force=True) \
        if vol.source_files else Dataset()

    modality = vol.modality if vol.modality in SOP_FOR_MODALITY else "MR"
    sop_class = SOP_FOR_MODALITY[modality]
    series_uid = generate_uid()
    now = datetime.datetime.now()
    slope, intercept = _int_encoding(vol.array)

    A = vol.affine
    sp = vol.spacing
    rcos, ccos = A[:3, 0] / sp[0], A[:3, 1] / sp[1]
    nk, nj, ni = vol.array.shape
    written = []
    for k in range(nk):
        if progress and k % 5 == 0 and not progress(k, nk, "Writing DICOM…"):
            raise RuntimeError("Cancelled")
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = sop_class
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        for tag in PATIENT_STUDY_TAGS:
            if tag in fixed_t:
                setattr(ds, tag, fixed_t.data_element(tag).value)
            elif tag in src_t:
                setattr(ds, tag, src_t.data_element(tag).value)
        for tag in ACQUISITION_TAGS:
            if tag in src_t:
                setattr(ds, tag, src_t.data_element(tag).value)
        if "StudyInstanceUID" not in ds:
            ds.StudyInstanceUID = generate_uid()
        ds.SOPClassUID = sop_class
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.Modality = modality
        ds.ImageType = ["DERIVED", "SECONDARY"]
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = series_number
        ds.SeriesDescription = description[:64]
        ds.FrameOfReferenceUID = vol.frame_of_reference_uid or generate_uid()
        ds.InstanceNumber = k + 1
        ds.ContentDate = now.strftime("%Y%m%d")
        ds.ContentTime = now.strftime("%H%M%S")
        if derivation:
            ds.DerivationDescription = derivation[:1024]

        ipp = A[:3, 3] + k * A[:3, 2]
        ds.ImagePositionPatient = [round(float(v), 6) for v in ipp]
        ds.ImageOrientationPatient = [round(float(v), 8) for v in np.r_[rcos, ccos]]
        ds.PixelSpacing = [round(float(sp[1]), 6), round(float(sp[0]), 6)]  # row, column
        ds.SliceThickness = round(float(sp[2]), 6)
        ds.SliceLocation = round(float(np.dot(ipp, np.cross(rcos, ccos))), 6)

        ds.Rows, ds.Columns = nj, ni
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope = f"{slope:.10g}"
        ds.RescaleIntercept = f"{intercept:.10g}"
        if modality == "CT":
            ds.RescaleType = "HU"
        if window:
            ds.WindowCenter = f"{window[0]:.6g}"
            ds.WindowWidth = f"{window[1]:.6g}"
        stored = np.clip(np.round((vol.array[k] - intercept) / slope), -32768, 32767).astype("<i2")
        ds.PixelData = stored.tobytes()

        path = os.path.join(out_dir, f"{modality}_{k + 1:04d}.dcm")
        ds.save_as(path, enforce_file_format=True)
        written.append(path)
    if progress:
        progress(nk, nk, "Done")
    return written
