"""
Build a robust all-tetrahedral moving-boundary mesh for svMP, replacing SimVascular
(too fragile) with mmgs (surface clean/adapt) + TetGen (volume).

Two surface-sizing modes:
  - UNIFORM (default): constant target edge (surf_hmax). Robust, no boundary layer to invert.
  - RADIUS-BASED (RBM, --rbm N): local edge = local_caliber / N, clamped to [hmin, hmax].
    Refines the small supra-aortic branches (small caliber) while keeping the main aorta
    coarse — same idea as SimVascular's radius-based meshing, but with MMG (no SimVascular
    dependency). Caliber is measured per surface vertex by an inward normal ray to the
    opposite wall (no centerline / VMTK needed).

Pipeline:
  reference STL -> mmgs (clean; uniform or size-map adapt) -> TetGen (tet volume)
  -> tag caps (plane + radius + normal) -> mesh-complete.mesh.vtu (+ mesh-surfaces/*.vtp)

Usage:
  python3 tools/build_iso_mesh.py <stl_ref> <orig_surfaces_dir> <out_dir> [surf_hmax=0.5]
                                  [--rbm N_ACROSS] [--hmin H] [--hmax H]
  e.g. uniform : ... <stl> <surf> <out> 0.5
       RBM     : ... <stl> <surf> <out> --rbm 6 --hmin 0.2 --hmax 0.8
                 (~6 elements across the local diameter; branches finer, aorta ~hmax)
"""
import sys, os, subprocess, argparse
import numpy as np, pyvista as pv, vtk
from scipy.spatial import cKDTree

CAPS = ["asc", "desc", "btca", "lcca", "lsa"]


def _wb(d, p):
    w = vtk.vtkXMLUnstructuredGridWriter() if d.IsA("vtkUnstructuredGrid") else vtk.vtkXMLPolyDataWriter()
    w.SetFileName(p); w.SetInputData(d); w.SetDataModeToBinary(); w.SetCompressorTypeToNone(); w.Write()


def _caliber(surf):
    """Per-vertex local caliber (distance along the inward normal to the opposite wall)."""
    s = surf.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    N = np.asarray(s.point_data["Normals"]); P = np.asarray(s.points)
    diag = float(np.linalg.norm(P.max(0) - P.min(0)))
    obb = vtk.vtkOBBTree(); obb.SetDataSet(s); obb.BuildLocator()
    cal = np.full(len(P), np.nan)
    hit = vtk.vtkPoints()
    for i in range(len(P)):
        p0 = P[i] - 1e-3 * N[i]          # start just inside (avoid self-hit)
        p1 = P[i] - diag * N[i]          # shoot inward across the lumen
        hit.Reset()
        if obb.IntersectWithLine(p0, p1, hit, None) and hit.GetNumberOfPoints() > 0:
            cal[i] = np.linalg.norm(np.array(hit.GetPoint(0)) - P[i])
    # fill misses with the median, then 2 neighbour-averaging passes to de-noise
    med = np.nanmedian(cal); cal[np.isnan(cal)] = med
    faces = s.faces.reshape(-1, 4)[:, 1:]
    nbr = [[] for _ in range(len(P))]
    for a, b, c in faces:
        nbr[a] += [b, c]; nbr[b] += [a, c]; nbr[c] += [a, b]
    for _ in range(2):
        cal = np.array([np.mean([cal[i]] + [cal[j] for j in nbr[i]]) if nbr[i] else cal[i]
                        for i in range(len(P))])
    return cal


def _write_sol(path, sizes):
    """Write a medit isotropic size map (.sol) matching the .mesh vertex order."""
    with open(path, "w") as f:
        f.write("MeshVersionFormatted 2\nDimension 3\n\nSolAtVertices\n%d\n1 1\n" % len(sizes))
        f.write("\n".join("%.6f" % s for s in sizes))
        f.write("\nEnd\n")


def build(stl_ref, orig_surf_dir, out_dir, surf_hmax=0.5, rbm_n_across=None, hmin=0.2, hmax=None):
    import tetgen, meshio
    os.makedirs(os.path.join(out_dir, "mesh-surfaces"), exist_ok=True)
    if hmax is None:
        hmax = surf_hmax

    # 1. STL -> clean VTP
    s = pv.read(stl_ref).triangulate().clean(); s.clear_data()

    # 2. mmgs: clean + (optionally) adapt the surface to a radius-based size map
    if rbm_n_across is None:
        # medit .mesh I/O (NOT .vtp): mmgs only reads VTK formats when MMG is built with
        # VTK support, which is not guaranteed -> use medit, which always works.
        faces = s.faces.reshape(-1, 4)[:, 1:]
        meshio.write("/tmp/_iso_ref.mesh", meshio.Mesh(s.points, [("triangle", faces)]))
        subprocess.run(["mmgs_O3", "-in", "/tmp/_iso_ref.mesh", "-out", "/tmp/_iso_surf.mesh",
                        "-hmax", str(surf_hmax), "-hmin", str(surf_hmax * 0.7),
                        "-hausd", "0.08", "-nr"], check=True, capture_output=True)
        mm = meshio.read("/tmp/_iso_surf.mesh")
        tri = mm.cells_dict["triangle"]
        sm = pv.PolyData(mm.points, np.hstack([np.full((len(tri), 1), 3), tri]).ravel()).clean()
        sm.clear_data()
    else:
        # radius-based: target edge = caliber / N_across, clamped to [hmin, hmax]
        cal = _caliber(s)
        size = np.clip(cal / float(rbm_n_across), hmin, hmax)
        faces = s.faces.reshape(-1, 4)[:, 1:]
        meshio.write("/tmp/_iso_ref.mesh", meshio.Mesh(s.points, [("triangle", faces)]))
        _write_sol("/tmp/_iso_ref.sol", size)
        print(f"[build_iso] RBM size map: branches {size.min():.2f} mm -> aorta {size.max():.2f} mm "
              f"(caliber {cal.min():.1f}-{cal.max():.1f} mm, N_across={rbm_n_across})")
        # -hmin/-hmax = HARD floor/ceiling: mmgs must not create edges below hmin even at
        # sharp rims/ridges -> avoids tiny slivers (which degrade conditioning & mass conservation).
        subprocess.run(["mmgs_O3", "-in", "/tmp/_iso_ref.mesh", "-met", "/tmp/_iso_ref.sol",
                        "-hmin", str(hmin), "-hmax", str(hmax),
                        "-hgrad", "1.3", "-hausd", "0.08", "-nr", "-out", "/tmp/_iso_surf.mesh"],
                       check=True, capture_output=True)
        mm = meshio.read("/tmp/_iso_surf.mesh")
        tri = mm.cells_dict["triangle"]
        sm = pv.PolyData(mm.points, np.hstack([np.full((len(tri), 1), 3), tri]).ravel()).clean()
        sm.clear_data()

    # 3. TetGen: tet volume (the graded surface drives the volume grading)
    tet = tetgen.TetGen(sm)
    tet.tetrahedralize(order=1, mindihedral=18, minratio=1.4)
    g = tet.grid
    g.point_data.clear(); g.cell_data.clear()
    g.point_data["GlobalNodeID"] = np.arange(1, g.n_points+1, dtype=np.int32)
    g.cell_data["GlobalElementID"] = np.arange(1, g.n_cells+1, dtype=np.int32)
    g.cell_data["ModelRegionID"] = np.ones(g.n_cells, dtype=np.int32)

    # 4. boundary surface + tag caps (plane + radius + normal, from the original caps)
    b = g.extract_surface().triangulate()
    b.point_data["GlobalNodeID"] = (np.asarray(b.point_data["vtkOriginalPointIds"]) + 1).astype(np.int32)
    b.cell_data["GlobalElementID"] = (np.asarray(b.cell_data["vtkOriginalCellIds"]) + 1).astype(np.int32)
    cent = b.cell_centers().points
    b = b.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
    cn = np.asarray(b.cell_data["Normals"])
    fid = np.ones(b.n_cells, dtype=np.int32)
    for i, n in enumerate(CAPS):
        cap = pv.read(os.path.join(orig_surf_dir, f"{n}.vtp")).points
        cc = cap.mean(0); _, _, vt = np.linalg.svd(cap-cc); nrm = vt[2]/np.linalg.norm(vt[2])
        rad = np.linalg.norm(cap-cc, axis=1).max()
        d = cent-cc; dpl = np.abs(d@nrm); inpl = np.linalg.norm(d-np.outer(d@nrm, nrm), axis=1); al = np.abs(cn@nrm)
        m = (dpl < 1.2) & (inpl < rad*1.15) & (al > 0.5) & (fid == 1)
        fid[m] = i+2
    b.cell_data["ModelFaceID"] = fid

    # 5. write mesh-complete + surfaces
    _wb(g, os.path.join(out_dir, "mesh-complete.mesh.vtu"))
    for name, fv in [("wall", 1)] + [(n, i+2) for i, n in enumerate(CAPS)]:
        sub = b.extract_cells(np.where(fid == fv)[0]).extract_surface()
        keep = pv.PolyData(sub.points, sub.faces)
        keep.point_data["GlobalNodeID"] = np.asarray(sub.point_data["GlobalNodeID"]).astype(np.int32)
        keep.cell_data["GlobalElementID"] = np.asarray(sub.cell_data["GlobalElementID"]).astype(np.int32)
        keep.cell_data["ModelFaceID"] = np.full(keep.n_cells, fv, dtype=np.int32)
        _wb(keep, os.path.join(out_dir, "mesh-surfaces", f"{name}.vtp"))
    print(f"[build_iso] {g.n_cells} tets, {g.n_points} nodes -> {out_dir}")
    return g.n_cells


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stl_ref"); ap.add_argument("orig_surf_dir"); ap.add_argument("out_dir")
    ap.add_argument("surf_hmax", nargs="?", type=float, default=0.5)
    ap.add_argument("--rbm", type=float, default=None, metavar="N_ACROSS",
                    help="radius-based meshing: ~N elements across the local diameter (e.g. 6)")
    ap.add_argument("--hmin", type=float, default=0.2)
    ap.add_argument("--hmax", type=float, default=None)
    a = ap.parse_args()
    build(a.stl_ref, a.orig_surf_dir, a.out_dir, a.surf_hmax,
          rbm_n_across=a.rbm, hmin=a.hmin, hmax=a.hmax)
