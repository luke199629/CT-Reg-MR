"""Round-trip check of RAW/.mhd and DICOM export on the synthetic data.

    python tools/make_sample_data.py sample_data
    python tests/test_export.py
"""
import os
import sys
import tempfile

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ctmr_reg.dicom_io import load_registration, load_volume, scan_folder  # noqa: E402
from ctmr_reg.export import derived_volume, resample_to, write_dicom_series, write_raw  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "sample_data")


def dice(a, b):
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def main():
    series, regs = scan_folder(ROOT)
    by_mod = {s.modality: s for s in series.values()}
    ct, mr = load_volume(by_mod["CT"].files), load_volume(by_mod["MR"].files)
    reg = load_registration(regs[0])
    T_ct2mr = np.linalg.inv(reg.matrix_from_to(mr.frame_of_reference_uid, ct.frame_of_reference_uid))

    arr = resample_to(mr, ct, T_ct2mr)
    assert arr.shape == ct.array.shape
    vol = derived_volume(arr, ct, mr, "MR reg to CT")
    sphere_ct = (ct.array > 250) & (ct.array < 350)
    d = dice(sphere_ct, (arr > 1200) & (arr < 1600))
    print(f"resampled MR vs CT sphere Dice = {d:.3f}")
    assert d > 0.85

    with tempfile.TemporaryDirectory() as tmp:
        # RAW + MHD
        mhd, raw = write_raw(vol, os.path.join(tmp, "mr_reg.mhd"), "float32")
        assert os.path.getsize(raw) == arr.size * 4
        img = sitk.ReadImage(mhd)
        assert np.array_equal(sitk.GetArrayFromImage(img), arr)
        assert np.allclose(img.GetOrigin(), ct.affine[:3, 3])
        assert np.allclose(img.GetSpacing(), ct.spacing)
        assert np.allclose(np.array(img.GetDirection()).reshape(3, 3), ct.affine[:3, :3] / ct.spacing)
        # header-free read of the .raw itself
        flat = np.fromfile(raw, "<f4").reshape(arr.shape)
        assert np.array_equal(flat, arr)
        print("RAW/MHD round-trip OK:", open(mhd).read().replace("\n", " | ")[:200])

        # DICOM
        files = write_dicom_series(vol, os.path.join(tmp, "dcm"), ct.source_files[0], "MR reg to CT",
                                   window=(700, 1400), derivation="test")
        back = load_volume(files)
        assert back.modality == "MR"
        assert back.frame_of_reference_uid == ct.frame_of_reference_uid
        assert np.allclose(back.affine, ct.affine, atol=1e-4), (back.affine, ct.affine)
        err = np.abs(back.array - arr).max()
        print(f"DICOM round-trip: {len(files)} files, max abs error {err:.3f}, warnings={back.warnings}")
        assert err <= 0.5 + 1e-3  # int16 rounding
        import pydicom
        ds = pydicom.dcmread(files[0])
        assert ds.StudyInstanceUID == pydicom.dcmread(ct.source_files[0]).StudyInstanceUID
        print("Sample tags:", ds.PatientName, ds.Modality, ds.SeriesDescription, ds.ImageType)

        # CT (negative HU) export keeps values exactly
        files = write_dicom_series(ct, os.path.join(tmp, "ct"), ct.source_files[0], "CT copy")
        assert np.array_equal(load_volume(files).array, ct.array)
    print("OK")


if __name__ == "__main__":
    main()
