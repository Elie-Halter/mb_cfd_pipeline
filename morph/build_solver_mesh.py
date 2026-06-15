"""
.o.mesh mesh (medit, mmg3d) -> svMP solver mesh:
  <out>/mesh-complete.mesh.vtu  (GlobalNodeID/GlobalElementID/ModelRegionID Int32)
  <out>/mesh-surfaces/{wall,asc,desc,btca,lcca,lsa}.vtp (GlobalNodeID, GlobalElementID=parent tet,
  ModelFaceID Int32) — caps tagged by plane+radius+normal from the original caps
  (same logic as the iso-mesh builder steps 4-5).
Node ordering is PRESERVED (= .o.mesh ordering) => the morph snapshots index 1:1 and
GlobalNodeID = index+1.
Usage: python3 morph/build_solver_mesh.py <in.o.mesh> <orig_surf_dir> <out_dir>
"""
import sys, os
import numpy as np, pyvista as pv, vtk, meshio

CAPS = ["asc", "desc", "btca", "lcca", "lsa"]


def _wb(d, p):
    w = vtk.vtkXMLUnstructuredGridWriter() if d.IsA("vtkUnstructuredGrid") else vtk.vtkXMLPolyDataWriter()
    w.SetFileName(p); w.SetInputData(d); w.SetDataModeToBinary(); w.SetCompressorTypeToNone(); w.Write()


def main(in_mesh, orig_surf_dir, out_dir):
    os.makedirs(os.path.join(out_dir, "mesh-surfaces"), exist_ok=True)
    m = meshio.read(in_mesh)
    pts = m.points.astype(float)
    tet = m.cells_dict["tetra"].astype(np.int64)
    a, b, c, d = pts[tet[:, 0]], pts[tet[:, 1]], pts[tet[:, 2]], pts[tet[:, 3]]
    v = np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a)
    if np.median(v) < 0:
        tet = tet[:, [0, 2, 1, 3]]
        v = -v
    assert (v > 0).all(), f"{(v<=0).sum()} non-positive tets in the reference mesh"

    cells = np.column_stack([np.full(len(tet), 4), tet]).ravel()
    g = pv.UnstructuredGrid(cells, np.full(len(tet), 10, np.uint8), pts)
    g.point_data["GlobalNodeID"] = np.arange(1, g.n_points + 1, dtype=np.int32)
    g.cell_data["GlobalElementID"] = np.arange(1, g.n_cells + 1, dtype=np.int32)
    g.cell_data["ModelRegionID"] = np.ones(g.n_cells, dtype=np.int32)

    bs = g.extract_surface().triangulate()
    bs.point_data["GlobalNodeID"] = (np.asarray(bs.point_data["vtkOriginalPointIds"]) + 1).astype(np.int32)
    bs.cell_data["GlobalElementID"] = (np.asarray(bs.cell_data["vtkOriginalCellIds"]) + 1).astype(np.int32)
    cent = bs.cell_centers().points
    bs = bs.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
    cn = np.asarray(bs.cell_data["Normals"])
    fid = np.ones(bs.n_cells, dtype=np.int32)
    for i, n in enumerate(CAPS):
        cap = pv.read(os.path.join(orig_surf_dir, f"{n}.vtp")).points
        cc = cap.mean(0); _, _, vt = np.linalg.svd(cap - cc); nrm = vt[2] / np.linalg.norm(vt[2])
        rad = np.linalg.norm(cap - cc, axis=1).max()
        dd = cent - cc
        dpl = np.abs(dd @ nrm)
        inpl = np.linalg.norm(dd - np.outer(dd @ nrm, nrm), axis=1)
        al = np.abs(cn @ nrm)
        msk = (dpl < 1.2) & (inpl < rad * 1.15) & (al > 0.5) & (fid == 1)
        fid[msk] = i + 2
    bs.cell_data["ModelFaceID"] = fid

    _wb(g, os.path.join(out_dir, "mesh-complete.mesh.vtu"))
    for name, fv in [("wall", 1)] + [(n, i + 2) for i, n in enumerate(CAPS)]:
        sub = bs.extract_cells(np.where(fid == fv)[0]).extract_surface()
        keep = pv.PolyData(sub.points, sub.faces)
        keep.point_data["GlobalNodeID"] = np.asarray(sub.point_data["GlobalNodeID"]).astype(np.int32)
        keep.cell_data["GlobalElementID"] = np.asarray(sub.cell_data["GlobalElementID"]).astype(np.int32)
        keep.cell_data["ModelFaceID"] = np.full(keep.n_cells, fv, dtype=np.int32)
        _wb(keep, os.path.join(out_dir, "mesh-surfaces", f"{name}.vtp"))
        print(f"  {name:5s}: {keep.n_cells:6d} tris, {keep.n_points:6d} pts")
    print(f"[solver_mesh] {g.n_cells} tets / {g.n_points} pts -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
