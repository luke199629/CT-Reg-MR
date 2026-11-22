"""Deformable CT <-> MR registration on top of the current rigid matrix.

Pipeline (same for every method)
  1. Working grid: same origin/orientation as the fixed image, isotropic spacing (default 2 mm).
  2. Fixed image and *rigidly* resampled moving image are put on that grid.
  3. The method returns a displacement field u(x) on the working grid, in mm (LPS):
         a fixed-space point x corresponds to the moving-space point  T_rigid(x + u(x)).
     This is the ITK/"resampling" convention (fixed -> moving).
  4. The moving image is warped onto the fixed image's native voxel grid for display/export.

Methods
  * elastix  – B-spline, Advanced Mattes mutual information, multi-resolution (itk-elastix).
  * ConvexAdam – MIND-SSC features, coupled convex optimisation + Adam (Siebert & Heinrich), GPU.
  * multiGradICON – pretrained multimodal foundation model + instance optimisation, GPU.
"""
from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import map_coordinates

from .dicom_io import Volume
from .geometry import sample_plane

Log = Optional[Callable[[str], None]]

METHODS = {
    "elastix": "elastix B-spline (mutual information)",
    "convexadam": "ConvexAdam (MIND, GPU)",
    "multigradicon": "multiGradICON (pretrained, GPU)",
}
MULTIGRADICON_URL = ("https://github.com/uncbiag/uniGradICON/releases/download/"
                     "multigradicon_weights/Step_2_final.trch")


def method_installed(key: str) -> bool:
    """Cheap check (no heavy imports) used to enable/disable GUI entries."""
    import importlib.util as u
    need = {"elastix": ["itk"], "convexadam": ["torch", "convexAdam"],
            "multigradicon": ["torch", "unigradicon", "icon_registration", "itk"]}[key]
    try:
        return all(u.find_spec(n) is not None for n in need)
    except Exception:
        return False


def method_available(key: str) -> tuple[bool, str]:
    try:
        if key == "elastix":
            import itk  # noqa: F401
            if not hasattr(itk, "elastix_registration_method"):
                return False, "itk-elastix not installed"
        elif key == "convexadam":
            import torch
            from convexAdam.convex_adam_MIND import convex_adam_pt  # noqa: F401
            if not torch.cuda.is_available():
                return True, "no CUDA GPU – will run on CPU (slow)"
        elif key == "multigradicon":
            import torch
            import unigradicon  # noqa: F401
            if not torch.cuda.is_available():
                return True, "no CUDA GPU – will run on CPU (slow)"
    except Exception as ex:  # ImportError and friends
        return False, f"not installed ({type(ex).__name__})"
    return True, ""


# --------------------------------------------------------------------------- grids
@dataclass
class Grid:
    affine: np.ndarray      # voxel (i,j,k,1) -> LPS mm
    shape: tuple            # (nk, nj, ni) array shape

    @property
    def spacing(self):
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    @property
    def directions(self):   # unit vectors of the i, j, k axes (columns)
        return self.affine[:3, :3] / self.spacing


def grid_of(vol: Volume) -> Grid:
    return Grid(vol.affine.copy(), vol.array.shape)


def working_grid(fixed: Volume, spacing_mm: float) -> Grid:
    """Isotropic grid with the fixed image's origin and orientation covering the same extent."""
    sp = fixed.spacing
    ni, nj, nk = fixed.shape_ijk
    extent = (np.array([ni, nj, nk]) - 1) * sp
    n = np.maximum(np.round(extent / spacing_mm).astype(int) + 1, 8)
    A = np.eye(4)
    A[:3, :3] = (fixed.affine[:3, :3] / sp) * spacing_mm
    A[:3, 3] = fixed.affine[:3, 3]
    return Grid(A, (int(n[2]), int(n[1]), int(n[0])))


def resample_on_grid(vol: Volume, T_grid_to_vol: np.ndarray, grid: Grid, fill=None, log: Log = None) -> np.ndarray:
    """Sample *vol* on *grid* through a 4x4 (grid patient coords -> vol patient coords)."""
    if fill is None:
        fill = float(np.nanmin(vol.array))
    A = grid.affine
    nk, nj, ni = grid.shape
    out = np.empty(grid.shape, np.float32)
    for k in range(nk):
        sl = sample_plane(vol, T_grid_to_vol, A[:3, 3] + k * A[:3, 2], A[:3, 1], A[:3, 0], (nj, ni))
        out[k] = np.where(np.isnan(sl), fill, sl)
    return out


def _normalise(arr: np.ndarray, modality: str) -> np.ndarray:
    if modality == "CT":
        lo, hi = -1000.0, 1500.0
    else:
        lo, hi = np.percentile(arr, [0.5, 99.5])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- result
@dataclass
class DeformResult:
    name: str
    method: str
    fixed_name: str
    moving_name: str
    params: dict
    T_rigid: np.ndarray             # fixed patient -> moving patient (rigid part used)
    grid: Grid                      # working grid
    dvf: np.ndarray                 # [k,j,i,3] mm, LPS (x,y,z components)
    jacobian: np.ndarray            # [k,j,i]
    warped: Volume                  # moving intensities on fixed native grid
    fixed_work: np.ndarray          # registration inputs on the working grid
    moving_work: np.ndarray
    seconds: float = 0.0
    log: list = field(default_factory=list)

    def u_at(self, x) -> np.ndarray:
        """Displacement (mm) at fixed-space point x (trilinear)."""
        g = np.linalg.inv(self.grid.affine) @ np.append(np.asarray(x, float)[:3], 1.0)
        return np.array([map_coordinates(self.dvf[..., c], [[g[2]], [g[1]], [g[0]]], order=1, mode="nearest")[0]
                         for c in range(3)])

    def map_point(self, x) -> np.ndarray:
        """Fixed-space point -> moving patient coords through rigid + deformation."""
        x = np.asarray(x, float)[:3]
        return (self.T_rigid @ np.append(x + self.u_at(x), 1.0))[:3]

    def map_volume(self, kind: str) -> Volume:
        """Jacobian ('jac') or displacement magnitude ('umag') as a Volume on the working grid
        (fixed-image patient coords), for display/sampling."""
        cache = self.__dict__.setdefault("_maps", {})
        if kind not in cache:
            arr = self.jacobian if kind == "jac" else np.linalg.norm(self.dvf, axis=-1)
            cache[kind] = Volume(array=np.ascontiguousarray(arr, dtype=np.float32),
                                 affine=self.grid.affine.copy(), modality="", series_uid="",
                                 frame_of_reference_uid=self.warped.frame_of_reference_uid)
        return cache[kind]

    def stats(self) -> dict:
        mag = np.linalg.norm(self.dvf, axis=-1)
        J = self.jacobian
        return dict(mean_mm=float(mag.mean()), p95_mm=float(np.percentile(mag, 95)), max_mm=float(mag.max()),
                    jac_min=float(J.min()), jac_max=float(J.max()),
                    folding_pct=float((J <= 0).mean() * 100))

    def summary(self) -> str:
        s = self.stats()
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return (f"{METHODS.get(self.method, self.method)}: {self.moving_name} → {self.fixed_name}, "
                f"{self.seconds:.1f} s\n{p}\n"
                f"|u| mean {s['mean_mm']:.2f} mm, 95% {s['p95_mm']:.2f} mm, max {s['max_mm']:.2f} mm\n"
                f"Jacobian {s['jac_min']:.3f} … {s['jac_max']:.3f}, folding {s['folding_pct']:.3f}% of voxels")


def jacobian_determinant(dvf: np.ndarray, grid: Grid) -> np.ndarray:
    """det(I + ∇u) with ∇ in patient mm. dvf [k,j,i,3]."""
    sp = grid.spacing          # i, j, k
    D = grid.directions        # columns: unit vectors of i, j, k
    # derivatives along array axes (k, j, i) of each component -> per-axis local derivative
    dk = np.gradient(dvf, sp[2], axis=0)
    dj = np.gradient(dvf, sp[1], axis=1)
    di = np.gradient(dvf, sp[0], axis=2)
    # du/dxi with columns ordered (i, j, k)
    G_local = np.stack([di, dj, dk], axis=-1)          # [..., comp, axis]
    G = G_local @ D.T                                   # ∂u/∂x (world)
    G = G + np.eye(3)
    return np.linalg.det(G).astype(np.float32)


def warp_to_native(moving: Volume, fixed: Volume, T_rigid: np.ndarray, grid: Grid, dvf: np.ndarray,
                   fill=None, log: Log = None) -> np.ndarray:
    """Moving intensities on the fixed native grid: m(T_rigid(x + u(x)))."""
    if fill is None:
        fill = float(np.nanmin(moving.array))
    Af = fixed.affine
    ni, nj, nk = fixed.shape_ijk
    G_inv = np.linalg.inv(grid.affine)
    M_inv = np.linalg.inv(moving.affine) @ T_rigid
    jj, ii = np.mgrid[0:nj, 0:ni].astype(np.float32)
    out = np.empty((nk, nj, ni), np.float32)
    comps = [np.ascontiguousarray(dvf[..., c]) for c in range(3)]
    for k in range(nk):
        if log and k % 20 == 0:
            log(f"Warping slice {k + 1}/{nk}")
        X = (Af[:3, 3][:, None, None] + Af[:3, 0][:, None, None] * ii + Af[:3, 1][:, None, None] * jj
             + Af[:3, 2][:, None, None] * k)                                  # [3, nj, ni]
        g = np.tensordot(G_inv[:3, :3], X, 1) + G_inv[:3, 3][:, None, None]   # grid i,j,k
        idx = [g[2], g[1], g[0]]
        U = np.stack([map_coordinates(c, idx, order=1, mode="nearest") for c in comps])
        P = X + U
        v = np.tensordot(M_inv[:3, :3], P, 1) + M_inv[:3, 3][:, None, None]
        sl = map_coordinates(moving.array, [v[2], v[1], v[0]], order=1, mode="constant", cval=np.nan,
                             prefilter=False)
        out[k] = np.where(np.isnan(sl), fill, sl)
    return out


# --------------------------------------------------------------------------- methods
def _itk_image(arr: np.ndarray, grid: Grid):
    import itk
    img = itk.image_from_array(np.ascontiguousarray(arr.astype(np.float32)))
    img.SetSpacing([float(v) for v in grid.spacing])
    img.SetOrigin([float(v) for v in grid.affine[:3, 3]])
    img.SetDirection(itk.matrix_from_array(np.ascontiguousarray(grid.directions.astype(np.float64))))
    return img


def run_elastix(fixed_w, moving_w, grid: Grid, p: dict, log: Log) -> np.ndarray:
    import itk
    f = _itk_image(_normalise(fixed_w, p["fixed_modality"]), grid)
    m = _itk_image(_normalise(moving_w, p["moving_modality"]), grid)
    po = itk.ParameterObject.New()
    pm = po.GetDefaultParameterMap("bspline", int(p["resolutions"]), float(p["grid_spacing_mm"]))
    bending = float(p.get("bending_weight", 0.0))
    if bending > 0:
        # MI + bending-energy penalty keeps the B-spline smooth where the images carry little signal.
        pm["Registration"] = ["MultiMetricMultiResolutionRegistration"]
        pm["Metric"] = ["AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty"]
        pm["Metric0Weight"] = ["1.0"]
        pm["Metric1Weight"] = [str(bending)]
    else:
        pm["Metric"] = ["AdvancedMattesMutualInformation"]
    pm["NumberOfHistogramBins"] = ["32"]
    pm["MaximumNumberOfIterations"] = [str(int(p["iterations"]))]
    pm["FinalGridSpacingInPhysicalUnits"] = [str(float(p["grid_spacing_mm"]))]
    pm["NumberOfSpatialSamples"] = [str(int(p.get("samples", 8192)))]
    pm["NewSamplesEveryIteration"] = ["true"]
    pm["ImageSampler"] = ["RandomCoordinate"]
    pm["WriteResultImage"] = ["false"]
    po.AddParameterMap(pm)
    import tempfile
    # elastix/transformix write side files (e.g. deformationField.nii); keep them out of the cwd.
    with tempfile.TemporaryDirectory(prefix="ctmr_elastix_") as tmp:
        log and log("Running elastix…")
        _, tpo = itk.elastix_registration_method(f, m, parameter_object=po, log_to_console=False,
                                                 output_directory=tmp)
        log and log("Computing deformation field…")
        dvf = itk.transformix_deformation_field(m, tpo, output_directory=tmp)
        return np.array(itk.array_from_image(dvf), dtype=np.float32)    # [k,j,i,3] mm (x,y,z)


def run_convexadam(fixed_w, moving_w, grid: Grid, p: dict, log: Log) -> np.ndarray:
    import torch
    from convexAdam.convex_adam_MIND import convex_adam_pt
    gs = int(p["grid_sp"])
    f = _normalise(fixed_w, p["fixed_modality"])
    m = _normalise(moving_w, p["moving_modality"])
    shape = f.shape
    pad = [(0, (-s) % (gs * 2)) for s in shape]              # divisible by grid_sp and grid_sp_adam
    f, m = np.pad(f, pad, mode="edge"), np.pad(m, pad, mode="edge")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log and log(f"Running ConvexAdam on {dev}…")
    disp = convex_adam_pt(f, m, mind_r=int(p["mind_r"]), mind_d=int(p["mind_d"]),
                          lambda_weight=float(p["lambda"]), grid_sp=gs, disp_hw=int(p["disp_hw"]),
                          selected_niter=int(p["iterations"]), grid_sp_adam=2, ic=True,
                          dtype=torch.float16 if dev.type == "cuda" else torch.float32, device=dev)
    disp = disp[:shape[0], :shape[1], :shape[2]].astype(np.float32)   # voxels along (k, j, i)
    sp, D = grid.spacing, grid.directions
    # voxel displacement along k,j,i -> mm in LPS
    return (disp[..., 0, None] * (D[:, 2] * sp[2]) + disp[..., 1, None] * (D[:, 1] * sp[1])
            + disp[..., 2, None] * (D[:, 0] * sp[0])).astype(np.float32)


def _weights_path() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    return os.path.join(base, "ctmr_reg", "weights", "multigradicon1.0", "Step_2_final.trch")


def run_multigradicon(fixed_w, moving_w, grid: Grid, p: dict, log: Log) -> np.ndarray:
    import itk
    import icon_registration.itk_wrapper as iw
    from unigradicon import get_multigradicon, make_sim, preprocess

    wpath = _weights_path()
    if not os.path.exists(wpath):
        os.makedirs(os.path.dirname(wpath), exist_ok=True)
        log and log("Downloading multiGradICON weights (first run only)…")
        urllib.request.urlretrieve(MULTIGRADICON_URL, wpath + ".part")
        os.replace(wpath + ".part", wpath)
    # Instance-optimisation similarity: MIND for CT<->MR (LNCC assumes matching contrast).
    sim = str(p.get("io_similarity", "mind"))
    net = get_multigradicon(loss_fn=make_sim(sim), weights_location=wpath)
    mod = {"CT": "ct", "MR": "mri"}
    f = preprocess(_itk_image(fixed_w, grid), mod[p["fixed_modality"]])
    m = preprocess(_itk_image(moving_w, grid), mod[p["moving_modality"]])
    io = int(p["io_iterations"])
    log and log(f"Running multiGradICON (instance optimisation {io} steps)…")
    phi_AB, _ = iw.register_pair(net, m, f, finetune_steps=io if io > 0 else None)

    # Displacement field on the working grid: resample coordinate maps through phi (fixed -> moving).
    log and log("Computing deformation field…")
    nk, nj, ni = grid.shape
    kk, jj, ii = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
    A = grid.affine
    X = [(A[c, 0] * ii + A[c, 1] * jj + A[c, 2] * kk + A[c, 3]).astype(np.float32) for c in range(3)]
    try:
        VT = itk.Image[itk.Vector[itk.F, 3], 3]
        tf = itk.TransformToDisplacementFieldFilter[VT, itk.D].New()
        tf.SetTransform(phi_AB)
        tf.SetReferenceImage(f)
        tf.SetUseReferenceImage(True)
        tf.Update()
        dvf = np.asarray(itk.array_from_image(tf.GetOutput()), dtype=np.float32)
    except Exception:
        dvf = np.empty(grid.shape + (3,), np.float32)
        for c in range(3):
            ref = _itk_image(X[c], grid)
            # pad far outside so linear extrapolation isn't needed for small displacements
            warped = itk.resample_image_filter(ref, transform=phi_AB, use_reference_image=True,
                                               reference_image=ref, default_pixel_value=np.nan)
            dvf[..., c] = itk.array_from_image(warped) - X[c]
        dvf = np.nan_to_num(dvf)
    del net
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return dvf


RUNNERS = {"elastix": run_elastix, "convexadam": run_convexadam, "multigradicon": run_multigradicon}

DEFAULT_PARAMS = {
    "elastix": {"grid_spacing_mm": 15.0, "iterations": 500, "resolutions": 3, "bending_weight": 300.0},
    "convexadam": {"lambda": 1.25, "iterations": 80, "grid_sp": 6, "disp_hw": 4, "mind_r": 1, "mind_d": 2},
    "multigradicon": {"io_iterations": 50, "io_similarity": "mind"},
}


def register(method: str, fixed: Volume, moving: Volume, fixed_name: str, moving_name: str,
             T_rigid: np.ndarray, params: dict, spacing_mm: float = 2.0, log: Log = None,
             name: str = "") -> DeformResult:
    t0 = time.time()
    lines: list[str] = []

    def _log(msg):
        lines.append(msg)
        if log:
            log(msg)

    grid = working_grid(fixed, spacing_mm)
    _log(f"Working grid {grid.shape[2]}×{grid.shape[1]}×{grid.shape[0]} @ {spacing_mm:g} mm")
    fill_m = -1000.0 if moving_name == "CT" else 0.0
    fixed_w = resample_on_grid(fixed, np.eye(4), grid, fill=-1000.0 if fixed_name == "CT" else 0.0)
    moving_w = resample_on_grid(moving, T_rigid, grid, fill=fill_m)
    p = dict(params, fixed_modality=fixed_name, moving_modality=moving_name)
    dvf = RUNNERS[method](fixed_w, moving_w, grid, p, _log)
    if dvf.shape != grid.shape + (3,):
        raise RuntimeError(f"Unexpected deformation field shape {dvf.shape}, expected {grid.shape + (3,)}")
    _log("Jacobian determinant…")
    jac = jacobian_determinant(dvf, grid)
    _log("Warping moving image onto fixed grid…")
    warped_arr = warp_to_native(moving, fixed, T_rigid, grid, dvf, fill=fill_m)
    warped = Volume(array=warped_arr, affine=fixed.affine.copy(), modality=moving.modality,
                    series_uid="", frame_of_reference_uid=fixed.frame_of_reference_uid,
                    description=f"{moving.description or moving_name} DIR {method}",
                    patient_name=fixed.patient_name, source_files=moving.source_files)
    res = DeformResult(name=name or METHODS[method], method=method, fixed_name=fixed_name,
                       moving_name=moving_name, params={k: v for k, v in params.items()},
                       T_rigid=T_rigid.copy(), grid=grid, dvf=dvf, jacobian=jac, warped=warped,
                       fixed_work=fixed_w, moving_work=moving_w, seconds=time.time() - t0, log=lines)
    _log("Done. " + res.summary().replace("\n", " | "))
    return res
