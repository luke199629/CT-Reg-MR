"""Generate a synthetic CT + MR + DICOM Spatial Registration for testing.

The phantom is defined in CT patient coordinates. The MR series lives in its own
Frame of Reference and on a different (slightly oblique) grid; its content is the
phantom moved by a known rigid transform. The REG file's top-level Frame of
Reference is the CT's; it contains an identity item for the CT and an item for the
MR whose matrix maps MR patient coords -> CT patient coords (standard DICOM REG).

    python tools/make_sample_data.py [out_dir]
"""
import os
import sys

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

CT_SOP = "1.2.840.10008.5.1.4.1.1.2"
MR_SOP = "1.2.840.10008.5.1.4.1.1.4"
REG_SOP = "1.2.840.10008.5.1.4.1.1.66.1"


def rot(ax, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    R = np.eye(4)
    i, j = [(1, 2), (0, 2), (0, 1)][ax]
    R[i, i] = c; R[i, j] = -s; R[j, i] = s; R[j, j] = c
    return R


def phantom_ct(p):
    """HU values at points p (..., 3) in CT patient coords (mm)."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    out = np.full(x.shape, -1000.0)
    body = (x / 120) ** 2 + (y / 90) ** 2 + (z / 110) ** 2 <= 1
    out[body] = 30
    shell = body & ((x / 112) ** 2 + (y / 82) ** 2 + (z / 102) ** 2 > 1)
    out[shell] = 900                                                         # "skull"
    out[((x - 40) ** 2 + (y + 20) ** 2 + (z - 30) ** 2) < 20 ** 2] = 300     # sphere A
    out[((x + 45) ** 2 + (y - 25) ** 2 + (z + 20) ** 2) < 15 ** 2] = -300    # sphere B
    out[(np.abs(x) < 6) & (np.abs(y + 10) < 6) & body] = 600                # rod along z (asymmetric marker)
    out[(np.abs(z - 60) < 4) & (np.abs(y) < 50) & (x > 0) & (x < 80) & body] = 200  # plate on +x side
    return out


def ct_to_mr_intensity(hu):
    mr = np.zeros_like(hu)
    mr[hu > -900] = 600           # soft tissue
    mr[hu >= 800] = 80            # cortical bone dark on MR
    mr[(hu > 250) & (hu < 350)] = 1400
    mr[(hu > -350) & (hu < -250)] = 1000
    mr[(hu > 550) & (hu < 650)] = 1800
    mr[(hu > 150) & (hu < 250)] = 300
    return mr


def base(sop, modality, study_uid, series_uid, for_uid, desc):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = sop
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = sop
    ds.Modality = modality
    ds.PatientName = "PHANTOM^SYNTHETIC"
    ds.PatientID = "PHANTOM001"
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = for_uid
    ds.SeriesDescription = desc
    ds.StudyDate = "20261116"
    return ds


def write_series(out, sop, modality, study, for_uid, desc, affine, shape_ijk, values_fn, intercept):
    series = generate_uid()
    ni, nj, nk = shape_ijk
    os.makedirs(out, exist_ok=True)
    jj, ii = np.mgrid[0:nj, 0:ni]
    for k in range(nk):
        vox = np.stack([ii, jj, np.full_like(ii, k), np.ones_like(ii)], -1).astype(float)
        world = vox @ affine.T
        img = values_fn(world[..., :3])
        ds = base(sop, modality, study, series, for_uid, desc)
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = [float(v) for v in affine[:3, 3] + k * affine[:3, 2]]
        sx, sy = np.linalg.norm(affine[:3, 0]), np.linalg.norm(affine[:3, 1])
        ds.ImageOrientationPatient = [float(v) for v in np.r_[affine[:3, 0] / sx, affine[:3, 1] / sy]]
        ds.PixelSpacing = [float(sy), float(sx)]
        ds.SliceThickness = float(np.linalg.norm(affine[:3, 2]))
        ds.Rows, ds.Columns = nj, ni
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope = 1
        ds.RescaleIntercept = intercept
        ds.PixelData = np.round(img - intercept).astype(np.int16).tobytes()
        ds.save_as(os.path.join(out, f"{modality}_{k + 1:04d}.dcm"), enforce_file_format=True)
    return series


def main(out_dir="sample_data"):
    study = generate_uid()
    ct_for, mr_for = generate_uid(), generate_uid()

    # Known rigid transform: MR patient coords -> CT patient coords
    T = np.eye(4); T[:3, 3] = [12.0, -8.0, 15.0]
    M_mr2ct = T @ rot(2, 10) @ rot(0, -6)

    # CT: axial, 1.5 x 1.5 x 3 mm
    A_ct = np.diag([1.5, 1.5, 3.0, 1.0]); A_ct[:3, 3] = [-140, -110, -130]
    write_series(os.path.join(out_dir, "CT"), CT_SOP, "CT", study, ct_for, "Synthetic CT",
                 A_ct, (188, 148, 88), phantom_ct, intercept=-1024)

    # MR: different FoR, oblique grid (rotated 5° about x), 1.2 x 1.2 x 4 mm
    A_mr = rot(0, 5) @ np.diag([1.2, 1.2, 4.0, 1.0]); A_mr[:3, 3] = [-150, -125, -120]
    mr_fn = lambda q: ct_to_mr_intensity(phantom_ct(
        (np.concatenate([q, np.ones(q.shape[:-1] + (1,))], -1) @ M_mr2ct.T)[..., :3]))
    write_series(os.path.join(out_dir, "MR"), MR_SOP, "MR", study, mr_for, "Synthetic T1",
                 A_mr, (250, 210, 62), mr_fn, intercept=0)

    # REG
    reg = base(REG_SOP, "REG", study, generate_uid(), ct_for, "Synthetic MR->CT registration")
    reg.SOPInstanceUID = generate_uid()
    reg.file_meta.MediaStorageSOPInstanceUID = reg.SOPInstanceUID
    reg.ContentDate = "20261116"; reg.ContentTime = "120000"
    items = []
    for uid, M in ((ct_for, np.eye(4)), (mr_for, M_mr2ct)):
        m = Dataset()
        m.FrameOfReferenceTransformationMatrix = [float(v) for v in M.ravel()]
        m.FrameOfReferenceTransformationMatrixType = "RIGID"
        mrs = Dataset(); mrs.MatrixSequence = Sequence([m])
        it = Dataset(); it.FrameOfReferenceUID = uid
        it.MatrixRegistrationSequence = Sequence([mrs])
        items.append(it)
    reg.RegistrationSequence = Sequence(items)
    os.makedirs(os.path.join(out_dir, "REG"), exist_ok=True)
    reg.save_as(os.path.join(out_dir, "REG", "REG_MR_to_CT.dcm"), enforce_file_format=True)
    np.savetxt(os.path.join(out_dir, "ground_truth_MR_to_CT.txt"), M_mr2ct, fmt="%.10g",
               header="Ground-truth matrix: MR patient coords -> CT patient coords")
    print("Wrote", os.path.abspath(out_dir))


if __name__ == "__main__":
    main(*sys.argv[1:])
