"""
Synthetic tiny test case (no patient data, no solver): a small all-tet cylinder with
inlet / outlet / wall surfaces and fake result VTUs (Velocity / Pressure / WSS /
Displacement). Used by the test suite to exercise the post-processing tools.
"""
import os
import numpy as np
import pyvista as pv
import vtk


def _wb(d, p):
    w = vtk.vtkXMLUnstructuredGridWriter() if d.IsA("vtkUnstructuredGrid") else vtk.vtkXMLPolyDataWriter()
    w.SetFileName(p); w.SetInputData(d); w.SetDataModeToBinary(); w.SetCompressorTypeToNone(); w.Write()


def make_mesh(out_dir, R=5.0, L=20.0, nz=12, nring=4, nth=16):
    """Build a tiny all-tet cylinder + tagged surfaces (wall=1, inlet=2, outlet=3)."""
    os.makedirs(os.path.join(out_dir, "mesh-surfaces"), exist_ok=True)
    pts = [[0, 0, z] for z in np.linspace(0, L, nz)]                      # axis
    for z in np.linspace(0, L, nz):
        for r in np.linspace(R / nring, R, nring):
            for t in np.linspace(0, 2 * np.pi, nth, endpoint=False):
                pts.append([r * np.cos(t), r * np.sin(t), z])
    cloud = pv.PolyData(np.array(pts, float))
    grid = cloud.delaunay_3d()                                            # tets (convex hull)
    grid = grid.extract_cells(np.arange(grid.n_cells))                    # -> UnstructuredGrid
    grid.point_data.clear(); grid.cell_data.clear()
    grid.point_data["GlobalNodeID"] = np.arange(1, grid.n_points + 1, dtype=np.int32)
    grid.cell_data["GlobalElementID"] = np.arange(1, grid.n_cells + 1, dtype=np.int32)
    _wb(grid, os.path.join(out_dir, "mesh-complete.mesh.vtu"))

    b = grid.extract_surface().triangulate()
    b.point_data["GlobalNodeID"] = (np.asarray(b.point_data["vtkOriginalPointIds"]) + 1).astype(np.int32)
    b.cell_data["GlobalElementID"] = (np.asarray(b.cell_data["vtkOriginalCellIds"]) + 1).astype(np.int32)
    cz = b.cell_centers().points[:, 2]
    fid = np.ones(b.n_cells, dtype=np.int32)                              # wall
    fid[cz < 0.05 * L] = 2                                                # inlet
    fid[cz > 0.95 * L] = 3                                                # outlet
    b.cell_data["ModelFaceID"] = fid
    for name, fv in [("wall", 1), ("inlet", 2), ("outlet", 3)]:
        sub = b.extract_cells(np.where(fid == fv)[0]).extract_surface()
        keep = pv.PolyData(sub.points, sub.faces)
        keep.point_data["GlobalNodeID"] = np.asarray(sub.point_data["GlobalNodeID"]).astype(np.int32)
        keep.cell_data["GlobalElementID"] = np.asarray(sub.cell_data["GlobalElementID"]).astype(np.int32)
        keep.cell_data["ModelFaceID"] = np.full(keep.n_cells, fv, dtype=np.int32)
        _wb(keep, os.path.join(out_dir, "mesh-surfaces", f"{name}.vtp"))
    return grid


def make_results(out_dir, grid, n_frames=8, start=10, every=10, moving=False, seed=0):
    """Write fake result_*.vtu frames with Velocity / Pressure / WSS / Displacement."""
    procs = os.path.join(out_dir, "4-procs"); os.makedirs(procs, exist_ok=True)
    rng = np.random.default_rng(seed)
    P0 = np.asarray(grid.points); n = len(P0)
    for k in range(n_frames):
        t = k / n_frames
        g = pv.UnstructuredGrid(grid)
        # plug-ish axial flow modulated in time, small noise
        u = np.zeros((n, 3)); u[:, 2] = 10.0 * (1 + 0.5 * np.sin(2 * np.pi * t)) + 0.1 * rng.standard_normal(n)
        g.point_data["Velocity"] = u
        g.point_data["Pressure"] = (100 * (1 - P0[:, 2] / P0[:, 2].max())).astype(float)
        wss = np.zeros((n, 3)); wss[:, 0] = 5.0 * (1 + 0.3 * np.sin(2 * np.pi * t)) + 0.05 * rng.standard_normal(n)
        g.point_data["WSS"] = wss
        if moving:                                                        # small radial pulsation
            d = np.zeros((n, 3)); d[:, 0] = 0.2 * np.sin(2 * np.pi * t) * P0[:, 0]
            d[:, 1] = 0.2 * np.sin(2 * np.pi * t) * P0[:, 1]
            g.point_data["Displacement"] = d
            g.points = P0 + d
        _wb(g, os.path.join(procs, f"results_{start + k * every:03d}.vtu"))
    return procs
