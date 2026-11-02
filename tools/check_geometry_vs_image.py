#!/usr/bin/env python3
"""
check_geometry_vs_image.py -- pre-flight: does the lumen surface used to build the mesh sit on the
patient's images, at the right size?

A surface that went through other software between the segmentation and the mesher (capping,
smoothing, unit or frame changes) can come out moved, mirrored or rescaled while keeping exactly
its shape -- nothing downstream notices, and every result inherits the error. This check compares
the surface with the image it was segmented on, and with the segmentation itself when available:

  1. field of view: fraction of surface points inside the image;
  2. wall contrast on the image: mean intensity 1.5 mm inside the wall over 3 mm outside, along
     the surface normals (reported; >1 means the surface encloses the brighter lumen);
  3. with --seg: the surface is aligned on the segmentation's own surface twice, by a RIGID
     (rotation + translation) and by a SIMILARITY (+ one isotropic scale) iterative-closest-point
     fit. A right-sized surface needs no scale (scale ~1, small rigid residual); a rescaled one is
     only explained by the similarity. Also the volume ratio (closed surface / segmentation).

FAIL (exit 1) when fewer than --min-fov of the points are in the image, or with --seg when the
fitted scale differs from 1 by more than --max-scale-dev, or when the rigid residual median exceeds
--max-rigid-mm. A report is printed and optionally written as JSON.

Images and segmentations: any single 3-D volume SimpleITK reads (NIfTI, NRRD, MHA, Slicer
.seg.nrrd). For a time-resolved series, split it first (chain_image/dicom_to_phases.py) and give
the phase that was segmented. Surfaces: STL/VTP/VTK in mm, in the image's physical frame (LPS, as
SimpleITK/ITK). A Slicer STL exported in RAS will fail the field-of-view or rigid test: re-export
in LPS, or pass --ras to flip x and y.

Usage:
  python3 tools/check_geometry_vs_image.py lumen_capped.stl --image phase04.nii.gz \
      --seg Segmentation_p4.seg.nrrd [--label 2] [--json report.json]
"""
import argparse
import json
import sys

import numpy as np


def load_surface(path, ras):
    import pyvista as pv
    s = pv.read(path)
    if not isinstance(s, pv.PolyData):
        s = s.extract_surface()
    s = s.triangulate().clean()
    if ras:
        s.points[:, :2] *= -1.0
    return s


def label_surface(seg_img, label):
    """Isosurface of one label of a segmentation (largest connected piece), in physical mm."""
    import SimpleITK as sitk
    import pyvista as pv
    from scipy.ndimage import label as cc
    if seg_img.GetNumberOfComponentsPerPixel() > 1:
        seg_img = sitk.VectorIndexSelectionCast(seg_img, 0)
    a = sitk.GetArrayFromImage(seg_img)
    m = (a == label) if label is not None else (a != 0)
    lab, n = cc(m)
    if n == 0:
        raise SystemExit("segmentation: no voxel with this label")
    m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    if not np.allclose(np.asarray(seg_img.GetDirection()), np.eye(3).ravel(), atol=1e-6):
        print("[check] WARNING: non-identity segmentation direction; surface built with origin + spacing only")
    g = pv.ImageData(dimensions=m.shape[::-1], spacing=seg_img.GetSpacing(), origin=seg_img.GetOrigin())
    g.point_data["m"] = m.astype(np.float32).ravel()           # x fastest, as the (z, y, x) array
    iso = g.contour([0.5], scalars="m").triangulate().clean()
    return iso, float(m.sum() * np.prod(seg_img.GetSpacing()))


def icp(P, Q, scale, iters=80, trim=90.0):
    """Align points P on the point cloud Q (rigid or similarity), centroid start, trimmed ICP.
    Returns (s, R, t, residual distances) with x -> s R x + t."""
    from scipy.spatial import cKDTree
    T = cKDTree(Q)
    cp, cq = P.mean(0), Q.mean(0)
    s = 1.0
    if scale:
        s = np.sqrt(((Q - cq) ** 2).sum(1).mean() / ((P - cp) ** 2).sum(1).mean())
    R, t = np.eye(3), cq - s * cp
    for _ in range(iters):
        X = s * P @ R.T + t
        d, i = T.query(X)
        w = d <= np.percentile(d, trim)
        A, B = P[w], Q[i[w]]
        ma, mb = A.mean(0), B.mean(0)
        U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma))
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        R = U @ D @ Vt
        if scale:
            s = float((S * np.diag(D)).sum() / ((A - ma) ** 2).sum())
        t = mb - s * ma @ R.T
    d, _ = T.query(s * P @ R.T + t)
    return s, R, t, d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("surface", help="lumen surface used for the mesh (STL/VTP, mm, LPS)")
    ap.add_argument("--image", required=True, help="the image volume the lumen was segmented on")
    ap.add_argument("--seg", help="the segmentation of that volume (e.g. Slicer .seg.nrrd)")
    ap.add_argument("--label", type=int, help="label value of the vessel in --seg (default: any non-zero)")
    ap.add_argument("--ras", action="store_true", help="the surface is in RAS: flip x and y")
    ap.add_argument("--min-fov", type=float, default=0.99)
    ap.add_argument("--max-scale-dev", type=float, default=0.05)
    ap.add_argument("--max-rigid-mm", type=float, default=1.5)
    ap.add_argument("--json")
    a = ap.parse_args()

    import SimpleITK as sitk
    from scipy.ndimage import gaussian_filter, map_coordinates

    surf = load_surface(a.surface, a.ras).compute_normals(auto_orient_normals=True, consistent_normals=True)
    P, N = np.asarray(surf.points, float), np.asarray(surf.point_normals, float)
    img = sitk.ReadImage(a.image, sitk.sitkFloat32)
    if img.GetDimension() != 3:
        raise SystemExit("--image must be a single 3-D volume (split time-resolved series first)")
    o, sp, n = np.array(img.GetOrigin()), np.array(img.GetSpacing()), np.array(img.GetSize())
    A = gaussian_filter(sitk.GetArrayFromImage(img), 0.7)
    rep, fail = dict(surface=a.surface, image=a.image), []

    idx = (P - o) / sp
    fov = float(np.mean(np.all((idx >= 0) & (idx <= n - 1), axis=1)))
    samp = lambda X: map_coordinates(A, ((X - o) / sp)[:, ::-1].T, order=1, mode="nearest")
    contrast = float(samp(P - 1.5 * N).mean() / max(samp(P + 3.0 * N).mean(), 1e-12))
    vol = float(surf.volume) if surf.n_open_edges == 0 else None
    rep.update(fov_fraction=fov, wall_contrast_in_over_out=contrast, surface_volume_ml=None if vol is None else vol / 1000)
    print(f"[check] field of view: {100 * fov:.1f} % of the surface points inside the image")
    print(f"[check] wall contrast (1.5 mm in / 3 mm out): {contrast:.2f}  (informative; >1 = encloses the brighter lumen)")
    if fov < a.min_fov:
        fail.append(f"only {100 * fov:.1f} % of the surface inside the image (frame? RAS vs LPS?)")

    if a.seg:
        iso, seg_vol = label_surface(sitk.ReadImage(a.seg), a.label)
        Q = np.asarray(iso.points, float)
        s_r, R_r, t_r, d_r = icp(P, Q, scale=False)
        s_s, R_s, t_s, d_s = icp(P, Q, scale=True)
        ang = lambda R: float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        rep.update(seg=a.seg, seg_volume_ml=seg_vol / 1000,
                   volume_ratio_surface_over_seg=None if vol is None else vol / seg_vol,
                   rigid=dict(residual_median_mm=float(np.median(d_r)), residual_p95_mm=float(np.percentile(d_r, 95)),
                              rotation_deg=ang(R_r)),
                   similarity=dict(scale=s_s, residual_median_mm=float(np.median(d_s)),
                                   residual_p95_mm=float(np.percentile(d_s, 95)), rotation_deg=ang(R_s)))
        print(f"[check] segmentation: {seg_vol / 1000:.1f} mL" + (f", surface {vol / 1000:.1f} mL (ratio {vol / seg_vol:.2f})" if vol else ""))
        print(f"[check] rigid fit      : residual median {np.median(d_r):.2f} mm, p95 {np.percentile(d_r, 95):.2f} mm, rotation {ang(R_r):.1f} deg")
        print(f"[check] similarity fit : scale {s_s:.3f}, residual median {np.median(d_s):.2f} mm, p95 {np.percentile(d_s, 95):.2f} mm")
        same_volume = vol is not None and abs(vol / seg_vol - 1.0) < 0.10
        if abs(s_s - 1.0) > a.max_scale_dev:
            fail.append(f"the surface needs a scale of {s_s:.3f} to match the segmentation"
                        + (" -- but the volumes agree: frame mismatch rather than size (mirrored? try --ras)" if same_volume
                           else " (rescaled geometry?)"))
        if np.median(d_r) > a.max_rigid_mm:
            fail.append(f"rigid residual median {np.median(d_r):.2f} mm > {a.max_rigid_mm} mm")

    rep.update(status="FAIL" if fail else "PASS", reasons=fail)
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)
    if fail:
        for r in fail:
            print(f"[check] FAIL: {r}")
        return 1
    print("[check] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
