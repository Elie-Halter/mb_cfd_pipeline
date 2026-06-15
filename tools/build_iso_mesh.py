"""
Build a robust ISOTROPIC-tetrahedral moving-boundary mesh (no fragile fine boundary
layer) for svMP. Replaces SimVascular (too fragile / segfaults) with mmgs (surface
cleanup) + TetGen (volume).

Rationale: a fine anisotropic boundary layer INVERTS under morphing (thousands of
tets collapse early -> remesh coarsens by ~10x -> WSS destroyed). The literature
(Fluent diffusion smoothing, COMSOL Yeoh, P1/P1 FEM studies) shows that for a P1/P1
solver (= svMP), a fine ISOTROPIC near-wall layer (~0.3-0.5 mm) is enough for WSS
AND does not invert.

Pipeline:
  reference STL -> mmgs (clean uniform surface) -> TetGen (tet volume) -> tag caps
  (plane + radius) -> mesh-complete.mesh.vtu (inline binary, svMP-readable)
  + mesh-surfaces/{wall,asc,desc,btca,lcca,lsa}.vtp

Usage: python3 tools/build_iso_mesh.py <stl_ref> <orig_surfaces_dir> <out_dir> [surf_hmax=0.5]
  surf_hmax 0.5 -> ~600k tets (near-wall ~0.3-0.5 mm); 0.8 -> ~240k (coarser/more robust).
"""
import sys, os, subprocess
import numpy as np, pyvista as pv, vtk
from scipy.spatial import cKDTree

CAPS = ["asc", "desc", "btca", "lcca", "lsa"]


def _wb(d, p):
    w = vtk.vtkXMLUnstructuredGridWriter() if d.IsA("vtkUnstructuredGrid") else vtk.vtkXMLPolyDataWriter()
    w.SetFileName(p); w.SetInputData(d); w.SetDataModeToBinary(); w.SetCompressorTypeToNone(); w.Write()


def build(stl_ref, orig_surf_dir, out_dir, surf_hmax=0.5):
    import tetgen
    os.makedirs(os.path.join(out_dir, "mesh-surfaces"), exist_ok=True)

    # 1. STL -> clean VTP
    s = pv.read(stl_ref).triangulate().clean(); s.clear_data()
    _wb(s, "/tmp/_iso_ref.vtp")

    # 2. mmgs: clean uniform surface (without -nr the ridges reopen holes; -nr = watertight + AR<5)
    subprocess.run(["mmgs_O3", "-in", "/tmp/_iso_ref.vtp", "-out", "/tmp/_iso_surf.vtp",
                    "-hmax", str(surf_hmax), "-hmin", str(surf_hmax*0.7), "-hausd", "0.08", "-nr"],
                   check=True, capture_output=True)

    # 3. TetGen: isotropic tet volume (robust, no boundary layer -> nothing to invert)
    sm = pv.read("/tmp/_iso_surf.vtp").triangulate().clean(); sm.clear_data()
    tet = tetgen.TetGen(sm)
    tet.tetrahedralize(order=1, mindihedral=18, minratio=1.4)
    g = tet.grid
    g.point_data.clear(); g.cell_data.clear()
    g.point_data["GlobalNodeID"] = np.arange(1, g.n_points+1, dtype=np.int32)
    g.cell_data["GlobalElementID"] = np.arange(1, g.n_cells+1, dtype=np.int32)
    g.cell_data["ModelRegionID"] = np.ones(g.n_cells, dtype=np.int32)

    # 4. boundary surface + tag caps (plane + radius + normal, from the original caps)
    b = g.extract_surface().triangulate()   # vtkOriginalPointIds (point) + vtkOriginalCellIds (cell)
    b.point_data["GlobalNodeID"] = (np.asarray(b.point_data["vtkOriginalPointIds"]) + 1).astype(np.int32)
    b.cell_data["GlobalElementID"] = (np.asarray(b.cell_data["vtkOriginalCellIds"]) + 1).astype(np.int32)  # parent tet (required by svMP)
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
    build(sys.argv[1], sys.argv[2], sys.argv[3],
          float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)
