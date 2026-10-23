"""
benchtools.py -- shared helpers for the analytic benchmarks (B1 free-stream, B2 expanding
tube, B3 Womersley). No solver is called from here.

File conventions (identical to tools/build_iso_mesh.py and the patient pipeline):
  - all-tetrahedral mesh, coordinates in MM; the solver XML applies Mesh_scale_factor 0.1
    (mm -> cm; the solver works in CGS: g/cm3, poise, cm/s, dyn/cm2);
  - mesh-complete.mesh.vtu: GlobalNodeID (int32, 1..N), GlobalElementID (1..M), ModelRegionID;
  - mesh-surfaces/<face>.vtp: GlobalNodeID = id of the VOLUME node (1-based),
    GlobalElementID = id of the PARENT volume tetrahedron, ModelFaceID constant per face;
  - Prescribed_displacement file (STANDARD format, read_prescribed_disp_txt):
        nN nTP
        t_0
        gid x y z      (ABSOLUTE positions in CM = mm * 0.1; nN lines)
        t_1 ...
    The reader maps by GID and IGNORES nodes that are not on the face, so one global file
    (all boundary nodes) can serve several Add_BC; time interpolation is LINEAR and periodic
    with period = last t;
  - General (space-time) BC file (read_temp_spat_values):
        ndof nTP nNo         (nNo must be EXACTLY the face's node count -> one file PER face)
        t_0 ... t_{nTP-1}    (t_0 = 0, increasing)
        gid                  (1-based)
        v0x v0y v0z          (nTP lines of ndof values)
        ...
  - initial condition: Simulation_initialization_file_path reads a VTU carrying the arrays of
    the enabled spatial outputs (Velocity, Pressure, Displacement...); its point order must be
    the input mesh order (it is always rebuilt from mesh-complete.mesh.vtu).

Result files: svMultiPhysics reads the mesh in mm, scales it to cm internally, and writes the
result POINTS back in mm (already DEFORMED: do not add Displacement again) while Velocity stays
in cm/s. `result_points_cm` converts the points so that fluxes come out in cm3/s.
"""
import os
import glob
import re
import warnings
import numpy as np

# pyvista >= 0.47 announces a future default change of extract_surface; the current default
# ('dataset_surface') is the one the pipeline relies on (vtkOriginalPointIds/CellIds).
warnings.filterwarnings("ignore", message=r"(?s).*extract_surface")

MM2CM = 0.1     # Mesh_scale_factor used throughout the pipeline


# ---------------------------------------------------------------- VTK writers
def _wb(d, p):
    """Binary, uncompressed XML writer (the solver's reader rejects appended+zlib)."""
    import vtk
    w = vtk.vtkXMLUnstructuredGridWriter() if d.IsA("vtkUnstructuredGrid") \
        else vtk.vtkXMLPolyDataWriter()
    w.SetFileName(p); w.SetInputData(d); w.SetDataModeToBinary()
    w.SetCompressorTypeToNone(); w.Write()


# ---------------------------------------------------------------- gmsh -> pyvista
def _gmsh_to_grid():
    """Current gmsh tetrahedral mesh as a pv.UnstructuredGrid (mm)."""
    import gmsh
    import pyvista as pv
    import vtk
    tags, coords, _ = gmsh.model.mesh.getNodes()
    tags = np.asarray(tags, dtype=np.int64)
    pts = np.asarray(coords, dtype=float).reshape(-1, 3)
    order = np.argsort(tags)
    tags_s, pts_s = tags[order], pts[order]
    etypes, _, enodes = gmsh.model.mesh.getElements(3)
    if len(etypes) != 1 or etypes[0] != 4:
        raise RuntimeError(f"non-tetrahedral 3D elements: types={etypes}")
    conn_tags = np.asarray(enodes[0], dtype=np.int64).reshape(-1, 4)
    conn = np.searchsorted(tags_s, conn_tags)          # gmsh tag -> 0-based index
    assert (tags_s[conn] == conn_tags).all(), "inconsistent gmsh tag mapping"
    # drop nodes not used by any tetrahedron (e.g. OCC construction points)
    used = np.unique(conn)
    if len(used) != len(pts_s):
        remap = -np.ones(len(pts_s), np.int64); remap[used] = np.arange(len(used))
        conn, pts_s = remap[conn], pts_s[used]
    return pv.UnstructuredGrid({vtk.VTK_TETRA: conn}, pts_s)


def _gmsh_init(name, threads=1):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", threads)
    gmsh.model.add(name)


def gen_box_mesh(x0, y0, z0, dx, dy, dz, h):
    """Uniform tetrahedral box (mm)."""
    import gmsh
    _gmsh_init("box")
    gmsh.model.occ.addBox(x0, y0, z0, dx, dy, dz)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.8 * h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.mesh.generate(3)
    g = _gmsh_to_grid()
    gmsh.finalize()
    return g


def gen_cylinder_mesh(R, L, h=None, size_expr=None):
    """Cylinder along z on [0, L], radius R (mm). Uniform size h, OR a gmsh MathEval size
    field `size_expr` (string in x, y, z, mm) for radial grading."""
    import gmsh
    _gmsh_init("cyl")
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, R)
    gmsh.model.occ.synchronize()
    if size_expr is not None:
        f = gmsh.model.mesh.field
        f.add("MathEval", 1)
        f.setString(1, "F", size_expr)
        f.setAsBackgroundMesh(1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    else:
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.8 * h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.mesh.generate(3)
    g = _gmsh_to_grid()
    gmsh.finalize()
    return g


# ---------------------------------------------------------------- face classifiers
def classify_box(centers, normals, bounds, tol=1e-3):
    """Name each boundary triangle of a box x0,x1,y0,y1,z0,z1 (dominant normal)."""
    x0, x1, y0, y1, z0, z1 = bounds
    names = np.empty(len(centers), dtype=object)
    ax = np.argmax(np.abs(normals), axis=1)
    for i, (c, a) in enumerate(zip(centers, ax)):
        lo, hi = ((x0, x1), (y0, y1), (z0, z1))[a]
        names[i] = ("xyz"[a] + "0") if abs(c[a] - lo) < abs(c[a] - hi) else ("xyz"[a] + "1")
        plane = lo if names[i].endswith("0") else hi
        if abs(c[a] - plane) > tol * max(x1 - x0, y1 - y0, z1 - z0):
            raise RuntimeError(f"boundary triangle off its plane: center={c}, face={names[i]}")
    return names


def classify_cylinder(centers, normals, R, L, rtol=0.35):
    """Name each boundary triangle of a z-axis cylinder: wall, z0, z1."""
    names = np.empty(len(centers), dtype=object)
    for i, (c, nz) in enumerate(zip(centers, normals[:, 2])):
        if abs(nz) > 0.9:
            names[i] = "z0" if c[2] < L / 2 else "z1"
        else:
            names[i] = "wall"
            r = np.hypot(c[0], c[1])
            if abs(r - R) > rtol * R:
                raise RuntimeError(f"wall triangle at r={r:.3f} != R={R}")
    return names


# ---------------------------------------------------------------- solver mesh export
def write_svmp_mesh(g, out_dir, classify, face_ids):
    """Write mesh-complete.mesh.vtu + mesh-surfaces/<face>.vtp (pipeline convention).

    g          : all-tet pv.UnstructuredGrid (mm)
    classify   : f(centers, normals) -> array of face names (str), one per boundary triangle
    face_ids   : dict name -> ModelFaceID (int)
    Returns dict name -> 1-based volume GlobalNodeIDs of that face.
    """
    import pyvista as pv
    os.makedirs(os.path.join(out_dir, "mesh-surfaces"), exist_ok=True)
    g = g.copy()
    g.point_data.clear(); g.cell_data.clear()
    g.point_data["GlobalNodeID"] = np.arange(1, g.n_points + 1, dtype=np.int32)
    g.cell_data["GlobalElementID"] = np.arange(1, g.n_cells + 1, dtype=np.int32)
    g.cell_data["ModelRegionID"] = np.ones(g.n_cells, dtype=np.int32)

    b = g.extract_surface(pass_pointid=True, pass_cellid=True).triangulate()
    b.point_data["GlobalNodeID"] = (np.asarray(b.point_data["vtkOriginalPointIds"]) + 1).astype(np.int32)
    b.cell_data["GlobalElementID"] = (np.asarray(b.cell_data["vtkOriginalCellIds"]) + 1).astype(np.int32)
    cent = b.cell_centers().points
    b = b.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
    names = classify(cent, np.asarray(b.cell_data["Normals"]))

    fid = np.zeros(b.n_cells, dtype=np.int32)
    for nm, i in face_ids.items():
        fid[names == nm] = i
    if (fid == 0).any():
        raise RuntimeError(f"{(fid == 0).sum()} unclassified boundary triangles")
    b.cell_data["ModelFaceID"] = fid

    _wb(g, os.path.join(out_dir, "mesh-complete.mesh.vtu"))
    info = {}
    for nm, fv in face_ids.items():
        sub = b.extract_cells(np.where(fid == fv)[0]).extract_surface()
        keep = pv.PolyData(sub.points, sub.faces)
        keep.point_data["GlobalNodeID"] = np.asarray(sub.point_data["GlobalNodeID"]).astype(np.int32)
        keep.cell_data["GlobalElementID"] = np.asarray(sub.cell_data["GlobalElementID"]).astype(np.int32)
        keep.cell_data["ModelFaceID"] = np.full(keep.n_cells, fv, dtype=np.int32)
        _wb(keep, os.path.join(out_dir, "mesh-surfaces", f"{nm}.vtp"))
        info[nm] = np.asarray(sub.point_data["GlobalNodeID"], dtype=np.int64)
    print(f"[benchtools] {g.n_cells} tets, {g.n_points} nodes -> {out_dir}")
    for nm in face_ids:
        print(f"[benchtools]   face {nm}: {len(info[nm])} nodes")
    return info


# ---------------------------------------------------------------- solver input files
def write_disp_file(path, gids, pos_mm_fn, times, fmt="%.8f"):
    """STANDARD Prescribed_displacement file.
    gids       : (nN,) 1-based volume GlobalNodeIDs
    pos_mm_fn  : f(t) -> (nN,3) ABSOLUTE positions in MM (converted to CM here)
    times      : time samples, t[0]=0, period = t[-1]
    """
    gids = np.asarray(gids)
    with open(path, "w") as f:
        f.write(f"{len(gids)} {len(times)}\n")
        for t in times:
            pos = np.asarray(pos_mm_fn(t), dtype=float) * MM2CM
            f.write(fmt % t + "\n")
            np.savetxt(f, np.column_stack([gids, pos]), fmt=["%d", fmt, fmt, fmt])
    print(f"[benchtools] wrote {path}: {len(gids)} nodes x {len(times)} samples")


def write_temp_spat_file(path, gids, value_xyz, t_end=1000.0):
    """General BC file, CONSTANT in time (same vector at t=0 and t=t_end).
    gids must be EXACTLY the node set of the face (lFa.nNo)."""
    gids = np.asarray(gids)
    vx, vy, vz = value_xyz
    with open(path, "w") as f:
        f.write(f"3 2 {len(gids)}\n0.0\n{t_end}\n")
        for gid in gids:
            f.write(f"{gid}\n{vx} {vy} {vz}\n{vx} {vy} {vz}\n")
    print(f"[benchtools] wrote {path}: {len(gids)} nodes, constant value {value_xyz}")


def write_ic_vtu(mesh_vtu, path, velocity, pressure, displacement=None):
    """Initial-condition VTU (Simulation_initialization_file_path), same points/cells and
    point order as mesh-complete. velocity (N,3) cm/s; pressure (N,) dyn/cm2;
    displacement (N,3) cm or None (give zeros when a mesh equation is present)."""
    import pyvista as pv
    g = pv.read(mesh_vtu)
    ic = pv.UnstructuredGrid(g.cells, g.celltypes, g.points)
    ic.point_data["Velocity"] = np.asarray(velocity, dtype=float)
    ic.point_data["Pressure"] = np.asarray(pressure, dtype=float)
    if displacement is not None:
        ic.point_data["Displacement"] = np.asarray(displacement, dtype=float)
    _wb(ic, path)
    print(f"[benchtools] wrote {path} (IC: Velocity/Pressure"
          + ("/Displacement)" if displacement is not None else ")"))


def write_fourier_pressure_file(path, amplitude, period, npts=401, nmodes=64):
    """Temporal_values_file_path for an unsteady Neumann BC (solver Fourier fit).
    Header 'npts nmodes', then npts lines 't p(t)', p(t) = A sin(2 pi t / T)."""
    tt = np.linspace(0.0, period, npts)
    pp = amplitude * np.sin(2 * np.pi * tt / period)
    with open(path, "w") as f:
        f.write(f"{npts} {nmodes}\n")
        for t, p in zip(tt, pp):
            f.write(f"{t:.8f} {p:.10e}\n")
    print(f"[benchtools] wrote {path}: p(t)={amplitude}*sin(2 pi t/{period}), {npts} pts, {nmodes} modes")


# ---------------------------------------------------------------- post-processing
def result_points_cm(g):
    """Points of a results_*.vtu in CM (output points are in mm and already deformed)."""
    return np.asarray(g.points, dtype=float) * MM2CM


def find_results(run_dir):
    """Sorted results_###.vtu found in run_dir or run_dir/<n>-procs."""
    cands = sorted(glob.glob(os.path.join(run_dir, "results_*.vtu")), key=step_of)
    if not cands:
        for d in sorted(glob.glob(os.path.join(run_dir, "*-procs"))):
            cands = sorted(glob.glob(os.path.join(d, "results_*.vtu")), key=step_of)
            if cands:
                break
    return cands


def step_of(path):
    return int(re.search(r"results_(\d+)", os.path.basename(path)).group(1))


def tri_flux(points_cm, tris, u_nodal, orient_vec):
    """P1-exact flux of u through triangles (points cm, u cm/s -> cm3/s).
    orient_vec: approximate outward direction used to orient the triangle normals."""
    p0, p1, p2 = (points_cm[tris[:, k]] for k in range(3))
    an = 0.5 * np.cross(p1 - p0, p2 - p0)
    s = np.sign(an @ np.asarray(orient_vec, dtype=float))
    s[s == 0] = 1.0
    an = an * s[:, None]
    um = (u_nodal[tris[:, 0]] + u_nodal[tris[:, 1]] + u_nodal[tris[:, 2]]) / 3.0
    return float(np.einsum("ij,ij->", um, an))


def tet_volume_sum(points_cm, tets):
    a = points_cm[tets[:, 0]]
    return float(np.abs(np.einsum("ij,ij->i",
                                  np.cross(points_cm[tets[:, 1]] - a, points_cm[tets[:, 2]] - a),
                                  points_cm[tets[:, 3]] - a)).sum() / 6.0)


def cell_divergence(points_cm, tets, u_nodal):
    """P1 divergence per tetrahedron (constant); returns (div, vol)."""
    X0 = points_cm[tets[:, 0]]
    A = np.stack([points_cm[tets[:, k]] - X0 for k in (1, 2, 3)], axis=2)   # (m,3,3) columns
    vol = np.abs(np.linalg.det(A)) / 6.0
    G = np.linalg.inv(A)                                                    # rows = grad(xi_k)
    U0 = u_nodal[tets[:, 0]]
    dU = np.stack([u_nodal[tets[:, k]] - U0 for k in (1, 2, 3)], axis=1)    # (m,3,3) rows
    div = np.einsum("mki,mki->m", dU, G)     # sum_k dU_k . grad(xi_k)
    return div, vol


def fit_sincos(t, y, omega):
    """Least-squares fit y ~ a sin(wt) + b cos(wt); returns (amplitude, phase_rad) with
    y = amp sin(wt + phase)."""
    M = np.column_stack([np.sin(omega * t), np.cos(omega * t)])
    (a, b), *_ = np.linalg.lstsq(M, np.asarray(y, float), rcond=None)
    return float(np.hypot(a, b)), float(np.arctan2(b, a))
