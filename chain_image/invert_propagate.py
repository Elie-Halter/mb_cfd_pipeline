#!/usr/bin/env python3
"""
invert_propagate.py -- step 2 of the image-registration wall chain.

Carries the wall nodes of the reference mesh (built on the segmented phase) to every phase.
register_phases.py produced, for each phase k, a transform T_k(x) = x + d_k(x) from phase-k
coordinates to reference coordinates. A wall node P of the reference mesh sits at the point x
of phase k with T_k(x) = P, so the transform is INVERTED node by node: damped fixed-point
iteration x <- x - step (x + d_k(x) - P), with per-node step control, to a residual below
--tol (1e-3 mm). The displacement field d_k is evaluated once per phase by transformix on the
crop grid and interpolated trilinearly. Nodes where the iteration does not converge (locally
non-invertible field, typically a few nodes near the root in systole) are filled by
inverse-distance weighting of their converged neighbours and reported.

Because every phase is obtained from the same node set, the result is node-corresponding and
iso-topological by construction: one wall field with the mesh's own wall nodes.

Usage:
  python3 chain_image/invert_propagate.py --reg-dir reg/ --wall-vtp mesh-surfaces/wall.vtp \
      --out wall_positions.npz [--surface-stl lumen_ref.stl]

--wall-vtp is the wall face of the solver mesh (points in mm, GlobalNodeID), in the physical
frame of the images. With --surface-stl (closed lumen surface at the reference phase), the same
inversion is applied to its vertices and its enclosed volume is reported per phase, for
comparison with the warped-segmentation volumes of results.json.
Output: wall-field .npz (positions (n_phases, n_wall, 3) mm, gid, ref_phase).
Dependencies: itk-elastix, SimpleITK.
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wallfield as wf


def read_stl_triangles(path):
    import pyvista as pv
    s = pv.read(path).triangulate()
    return np.asarray(s.points, float)[s.faces.reshape(-1, 4)[:, 1:]]      # (m,3,3)


def tri_volume_ml(tri):
    return abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0 / 1000.0


def displacement_field(moving_itk, tp_file):
    """Interpolant of the transformix displacement field d(x) (x in mm, physical frame)."""
    import itk
    from scipy.interpolate import RegularGridInterpolator
    po = itk.ParameterObject.New()
    po.AddParameterFile(tp_file)
    field = itk.transformix_deformation_field(moving_itk, po)
    d = itk.GetArrayFromImage(field).astype(np.float64)                     # (z,y,x,3)
    o = np.array(field.GetOrigin()); s = np.array(field.GetSpacing())
    nz, ny, nx = d.shape[:3]
    axes = (o[2] + np.arange(nz) * s[2], o[1] + np.arange(ny) * s[1], o[0] + np.arange(nx) * s[0])
    f = RegularGridInterpolator(axes, d, bounds_error=False, fill_value=None)
    return lambda x: f(x[:, ::-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reg-dir", required=True, help="output directory of register_phases.py")
    ap.add_argument("--wall-vtp", required=True, help="wall face of the reference solver mesh")
    ap.add_argument("--out", required=True, help="wall-field .npz to write")
    ap.add_argument("--surface-stl", help="closed reference lumen surface, for the volume QC")
    ap.add_argument("--tol", type=float, default=1e-3, help="inversion residual target (mm)")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--idw-above", type=float, default=0.5,
                    help="residual (mm) above which a node is filled from its neighbours")
    ap.add_argument("--idw-k", type=int, default=12)
    a = ap.parse_args()

    import SimpleITK as sitk
    import itk
    import pyvista as pv

    crop = json.load(open(os.path.join(a.reg_dir, "crop.json")))
    ref, n = int(crop["ref_index"]), int(crop["n_phases"])
    ref_img = sitk.ReadImage(crop["ref_image"], sitk.sitkFloat32)
    roi = sitk.RegionOfInterest(ref_img, crop["size"], crop["lo"])
    moving = itk.GetImageFromArray(sitk.GetArrayFromImage(roi))
    moving.SetSpacing(roi.GetSpacing()); moving.SetOrigin(roi.GetOrigin())

    wall = pv.read(a.wall_vtp)
    W = np.asarray(wall.points, dtype=float)
    gid = np.asarray(wall.point_data["GlobalNodeID"]).astype(np.int64)
    tri = read_stl_triangles(a.surface_stl) if a.surface_stl else None
    tri_pts = tri.reshape(-1, 3) if tri is not None else None

    positions = np.zeros((n, len(W), 3))
    report = {}
    for k in range(n):
        t0 = time.time()
        if k == ref:
            positions[k] = W
            report[k] = dict(volume_ml=tri_volume_ml(tri) if tri is not None else None, max_disp_mm=0.0)
            continue
        disp = displacement_field(moving, os.path.join(a.reg_dir, f"tp_phase{k:02d}.txt"))
        x, rn = wf.invert_points(W, disp, tol=a.tol, max_iter=a.max_iter)
        bad = rn > a.idw_above
        if bad.any():
            x = wf.idw_fill(W, x, bad, k=a.idw_k)
        positions[k] = x
        rep = dict(max_residual_mm=float(rn.max()), n_above_0p1mm=int((rn > 0.1).sum()),
                   n_idw_filled=int(bad.sum()),
                   max_disp_mm=float(np.linalg.norm(x - W, axis=1).max()))
        if tri_pts is not None:
            xs, _ = wf.invert_points(tri_pts, disp, tol=a.tol, max_iter=a.max_iter)
            rep["volume_ml"] = tri_volume_ml(xs.reshape(-1, 3, 3))
        report[k] = rep
        print(f"[invert] phase {k:02d}: residual max {rep['max_residual_mm']:.2e} mm "
              f"(>0.1 mm: {rep['n_above_0p1mm']}, IDW-filled: {rep['n_idw_filled']}), "
              f"max |d| {rep['max_disp_mm']:.2f} mm"
              + (f", volume {rep['volume_ml']:.1f} mL" if "volume_ml" in rep else "")
              + f" ({time.time() - t0:.0f} s)", flush=True)
    wf.save_field(a.out, positions, gid, ref_phase=ref)
    json.dump(report, open(os.path.splitext(a.out)[0] + "_qc.json", "w"), indent=1)
    print(f"[invert] wrote {a.out} ({n} phases x {len(W)} wall nodes, reference phase {ref})")


if __name__ == "__main__":
    main()
