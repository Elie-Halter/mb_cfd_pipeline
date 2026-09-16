"""
Build a robust all-tetrahedral moving-boundary mesh for svMP, replacing SimVascular
(too fragile) with mmgs (surface clean/adapt) + TetGen (volume).

WHY ISOTROPIC GRADED NEAR-WALL REFINEMENT (and NOT a prismatic boundary layer):
  svMP here is run all-tet. A prismatic (anisotropic) boundary layer is NOT an option:
  thin prisms invert under the prescribed wall morph (mesh motion drags the near-wall
  layer and the flat prisms flip negative). So the ONLY way to resolve the steep near-wall
  velocity gradient — i.e. to compute WSS accurately — is GRADED ISOTROPIC TET refinement:
  small tets at the wall, growing geometrically toward the interior.

  IMPORTANT distinction (do not conflate these two, they are independent):
    * Mesh MOTION (the morph) is KINEMATICS: it prescribes where nodes go each time step so
      the lumen follows the wall. It does NOT add spatial resolution. The morph does NOT
      "replace" a boundary layer.
    * Near-wall GRADIENT RESOLUTION is SPATIAL DISCRETIZATION: you need physically small
      elements normal to the wall to capture du/dn = WSS. That is what --hwall provides.
  Both are needed: the morph keeps the mesh valid as the wall moves; the graded near-wall
  sizing makes the WSS on that moving wall trustworthy.

Surface-sizing modes (which target edge length mmgs adapts the surface to):
  - UNIFORM (default): constant target edge (surf_hmax). Robust, no anisotropy to invert.
  - RADIUS-BASED (RBM, --rbm N): local edge = local_caliber / N, clamped to [hmin, hmax].
    Refines the small supra-aortic branches (small caliber) while keeping the main aorta
    coarse — same idea as SimVascular's radius-based meshing, but with MMG (no SimVascular
    dependency). Caliber is measured per surface vertex by an inward normal ray to the
    opposite wall (no centerline / VMTK needed).

Near-wall VOLUME grading (independent of the surface mode above):
  - --hwall H (default off): build an isotropic mmg3d SIZE MAP on the TET volume so that the
    element size is H at every node on the `wall` surface and grows geometrically (ratio
    <= --hgrad) toward the interior, capped at the local interior size (RBM/hmax, else
    surf_hmax). The size map is written as a medit `.sol` and fed to mmg3d via `-met`, with
    `-hgrad <hgrad>` enforcing the geometric growth. This replaces the plain `-optim` quality
    pass with a metric-driven adaptation pass (still `-nosurf`, surface frozen, slivers
    removed) so that the morph-registered surface is preserved while the interior is refined
    near the wall. When --hwall is OFF the behaviour is exactly the old `-optim` sliver pass.

  - --hwall-aniso H (default off): build an ANISOTROPIC (tensor) metric instead of the scalar
    one above. WHY: an isotropic size field cannot decouple the normal and tangential edge
    lengths, so with -nosurf (frozen registered surface, tangential edges pinned at ~0.6 mm)
    the isotropic --hwall plateaus at ~0.26 mm first-layer NORMAL spacing — it cannot make the
    normal spacing finer than the surface triangle it is forced to keep. The anisotropic metric
    breaks that coupling: at each volume node we request a SMALL edge length h_n NORMAL to the
    wall (~0.1-0.15 mm, for du/dn = WSS resolution) but keep the TANGENTIAL edge length h_t at
    the bulk/RBM size (~0.6 mm). The result is boundary-layer-like resolution (thin tets normal
    to the wall, coarse along it) in an ALL-TET mesh, without prismatic layers (which invert
    under the wall morph, see above) and WITHOUT exploding the element count: only the normal
    direction is refined, so the cost grows ~1/h_n in one direction rather than ~1/h^3.
    The metric tensor at a node at distance d from the wall, with wall-normal n, is
        M(x) = (1/h_n^2) (n (x) n) + (1/h_t^2) (I - n (x) n)
    where h_n(d) = min(H * hgrad**(d/H), h_t) grows geometrically away from the wall up to the
    bulk size h_t (so far from the wall M -> (1/h_t^2) I, i.e. isotropic bulk). It is written as
    a medit anisotropic `.sol` (type `1 3` = one tensor field, 6 components per vertex) and fed
    to mmg3d via `-met` together with `-nosurf` (freeze the registered surface) and
    `-hgrad <hgrad>` (gradation on the metric eigenvalues). mmg3d 5.8.0 natively accepts an
    anisotropic tensor metric this way (binary advertises ANISOTROPIC / MMG5_Tensor /
    SolAtVertices ... TENSOR; verified, see _write_sol_aniso for the component ordering).
    --hwall and --hwall-aniso are mutually exclusive (one metric drives the pass).

GCI (grid-convergence) family — how to build coarse/medium/fine for the WSS study:
  Scale the near-wall size AND the bulk size by a constant refinement ratio r ~ 1.3 so the
  whole mesh refines self-similarly (keeps cell-size ratio ~constant -> valid GCI):
    coarse : --hwall (H*r)   --hmax (Hmax*r)     [+ --rbm N if used]
    medium : --hwall  H      --hmax  Hmax
    fine   : --hwall (H/r)   --hmax (Hmax/r)
  e.g. with H=0.10 mm, Hmax=0.80 mm, r=1.3:
    coarse : --hwall 0.130 --hmax 1.040 --hgrad 1.2 [--rbm 6 --hmin 0.20]
    medium : --hwall 0.100 --hmax 0.800 --hgrad 1.2 [--rbm 6 --hmin 0.20]
    fine   : --hwall 0.077 --hmax 0.615 --hgrad 1.2 [--rbm 6 --hmin 0.20]
  Keep --hgrad and --rbm IDENTICAL across the three grids (only the absolute sizes scale),
  then feed the three WSS results to tools/gci.py for the Richardson/GCI estimate.

  ANISOTROPIC GCI family (--hwall-aniso): refine ONLY the wall-normal size H, keeping the
  tangential/bulk size (--hmax, and --rbm if used) CONSTANT across the three grids — this
  isolates the near-wall-normal (WSS) discretization error, which is the dominant one for WSS.
  Use a normal refinement ratio r_n ~ 1.33 (so h_n: coarse/medium/fine = 1.33:1:0.75):
    coarse : --hwall-aniso 0.20 --hmax 0.80 --hgrad 1.2 [--rbm 6 --hmin 0.20]
    medium : --hwall-aniso 0.15 --hmax 0.80 --hgrad 1.2 [--rbm 6 --hmin 0.20]
    fine   : --hwall-aniso 0.11 --hmax 0.80 --hgrad 1.2 [--rbm 6 --hmin 0.20]
  (0.20/0.15/0.11 mm ~ ratio 1.33; tangential/hmax held constant.) Keep --hmax, --hgrad and
  --rbm IDENTICAL across the three; only H scales. Feed the three WSS results to tools/gci.py.

Pipeline:
  reference STL -> mmgs (clean; uniform or size-map adapt) -> TetGen (tet volume)
  -> mmg3d (sliver removal; optionally near-wall metric-driven grading via --hwall)
  -> tag caps (plane + radius + normal) -> mesh-complete.mesh.vtu (+ mesh-surfaces/*.vtp)

Usage:
  python3 tools/build_iso_mesh.py <stl_ref> <orig_surfaces_dir> <out_dir> [surf_hmax=0.5]
                                  [--rbm N_ACROSS] [--hmin H] [--hmax H]
                                  [--hwall H | --hwall-aniso H] [--hgrad R]
  e.g. uniform : ... <stl> <surf> <out> 0.5
       RBM     : ... <stl> <surf> <out> --rbm 6 --hmin 0.2 --hmax 0.8
                 (~6 elements across the local diameter; branches finer, aorta ~hmax)
       WSS iso : ... <stl> <surf> <out> --rbm 6 --hmin 0.2 --hmax 0.8 --hwall 0.1 --hgrad 1.2
                 (0.1 mm ISOTROPIC tets at the wall, growing at ratio<=1.2 toward the interior)
       WSS BL  : ... <stl> <surf> <out> --rbm 6 --hmin 0.2 --hmax 0.8 --hwall-aniso 0.12 --hgrad 1.2
                 (ANISOTROPIC: ~0.12 mm NORMAL to the wall, ~bulk tangentially -> BL-like,
                  all-tet, count-controlled; see "Near-wall VOLUME grading" above)
"""
import sys, os, subprocess, argparse
import numpy as np, pyvista as pv, vtk
from scipy.spatial import cKDTree

CAPS = ["asc", "desc", "btca", "lcca", "lsa"]


def _rbm_volume_sizes(g, surf, n_across, hmin, cap):
    """Per-VOLUME-vertex radius-based size map for the mmg3d volume pass.

    The surface RBM grading (mmgs) only sizes the wall triangles; the legacy `-optim`
    volume pass then uses mmg's DEFAULT metric and homogenises the interior toward the
    quality-optimal size (~hmax), erasing the branch refinement. To keep branches fine
    THROUGH the volume, drive mmg3d with this metric instead: each volume node takes the
    radius-based size (caliber/N, clamped to [hmin, cap]) of its NEAREST boundary node, so
    a node inside a thin branch stays fine while the aortic lumen stays at `cap`."""
    bt = surf.triangulate()
    cal = _caliber(bt)
    ssize = np.clip(cal / float(n_across), hmin, cap)
    _, idx = cKDTree(np.asarray(bt.points)).query(g.points, k=1)
    return ssize[idx]


def _mmg_optimize(g, hwall=None, hgrad=1.2, hmax=None, hwall_aniso=None, rbm=None):
    """Remove slivers from the TetGen volume (mmg3d, surface frozen).

    TetGen's radius-edge quality bound (minratio) does NOT eliminate slivers (flat tets
    with near-zero volume). A sliver REFERENCE tet has a near-singular edge matrix W, so the
    rest-shape morph energy (M = W^-1) becomes ill-conditioned and the morph leaves ~100
    inverted tets under large wall motion. This pass (mmg3d -nosurf) cleans the interior while
    freezing the surface, bringing min shape quality from ~0.02 to >0.15 (sliver-free,
    matching the validated reference mesh). Returns a new UnstructuredGrid (a few % more
    interior nodes); falls back to the raw TetGen mesh if mmg3d is unavailable.

    Three modes (hwall and hwall_aniso are mutually exclusive):
      - hwall is None and hwall_aniso is None: pure quality optimization (mmg3d -optim),
        the legacy behaviour.
      - hwall set: build an ISOTROPIC near-wall size map (see _wall_size_map) and run mmg3d
        in METRIC-DRIVEN mode (-met file -hgrad), which refines the interior down to `hwall`
        at the wall and grows it geometrically (ratio<=hgrad) up to the local cap. NOTE:
        mmg3d's -optim and -met are mutually exclusive (optim ignores any input metric), so
        with --hwall we drop -optim and let the metric drive both refinement AND quality.
        -nosurf still freezes the morph-registered surface; slivers are removed by the
        metric-driven re-meshing (split/collapse/swap/move) the same way -optim cleaned them.
      - hwall_aniso set: build an ANISOTROPIC (tensor) near-wall metric (see
        _wall_metric_aniso) and run mmg3d in -met mode exactly as above, but the metric now
        requests a SMALL edge length NORMAL to the wall (hwall_aniso) and the BULK size
        tangentially. This yields boundary-layer-like normal resolution in an all-tet mesh
        (see module docstring). The wall NORMALS are taken from the volume's own extracted
        boundary surface (so the normal-direction nodes are exactly volume nodes).
    """
    import meshio
    surf = g.extract_surface()
    tri = surf.faces.reshape(-1, 4)[:, 1:]
    orig = np.asarray(surf.point_data["vtkOriginalPointIds"])
    meshio.write("/tmp/_mmgopt_in.mesh",
                 meshio.Mesh(g.points, [("triangle", orig[tri]), ("tetra", g.cells_dict[vtk.VTK_TETRA])]))
    cmd = ["mmg3d_O3", "-in", "/tmp/_mmgopt_in.mesh", "-out", "/tmp/_mmgopt_out.mesh",
           "-nosurf", "-hgrad", str(hgrad), "-v", "0"]
    cap = hmax if hmax is not None else float(np.nanmax(_edge_len(g)))
    wall_ids = np.unique(orig[tri].ravel())   # surface tri vertices index into g.points
    if hwall_aniso is not None:
        # anisotropic tensor metric: small NORMAL spacing, bulk tangential spacing.
        # `surf` is the volume's boundary (wall + caps), with normals taken at the nearest
        # boundary vertex -> wall.compute_normals(point_normals=True)['Normals'] pattern.
        M = _wall_metric_aniso(g.points, surf, hwall_aniso, hgrad, cap)
        _write_sol_aniso("/tmp/_mmgopt_in.sol", M)
        cmd += ["-met", "/tmp/_mmgopt_in.sol"]
        print(f"[build_iso] near-wall ANISOTROPIC metric: h_normal={hwall_aniso:.3f} mm at wall "
              f"-> bulk {cap:.3f} mm tangential/far (hgrad<={hgrad}), {len(wall_ids)} wall nodes")
    elif hwall is not None:
        # isotropic near-wall size field on the CURRENT (post-TetGen) volume node set
        sizes = _wall_size_map(g.points, wall_ids, hwall, hgrad, cap)
        _write_sol("/tmp/_mmgopt_in.sol", sizes)
        # HARD floor/ceiling: without -hmin the -nosurf grading pass can spawn sub-hwall
        # interior slivers (min edge ~0.01 mm -> CFL blow-up, MB divergence). Floor at
        # 0.5*hwall keeps the near-wall layer (~hwall) while collapsing the slivers.
        cmd += ["-met", "/tmp/_mmgopt_in.sol", "-hmin", str(round(hwall * 0.5, 4)), "-hmax", str(cap)]
        print(f"[build_iso] near-wall size map: hwall={hwall:.3f} mm -> cap "
              f"{sizes.max():.3f} mm (hgrad<={hgrad}), {len(wall_ids)} wall nodes")
    elif rbm is not None:
        # radius-based VOLUME metric: keep branches fine through the interior (instead of
        # the -optim default metric, which homogenises toward ~hmax and erases the grading)
        n_across, rbm_hmin = rbm
        sizes = _rbm_volume_sizes(g, surf, n_across, rbm_hmin, cap)
        _write_sol("/tmp/_mmgopt_in.sol", sizes)
        cmd += ["-met", "/tmp/_mmgopt_in.sol", "-hmin", str(rbm_hmin), "-hmax", str(cap)]
        print(f"[build_iso] RBM VOLUME metric: {sizes.min():.3f} -> {sizes.max():.3f} mm "
              f"(N_across={n_across}, hgrad<={hgrad}) -- keeps branches fine in the volume")
    else:
        cmd += ["-optim"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tag = ("-met (near-wall ANISOTROPIC grading)" if hwall_aniso is not None else
               "-met (near-wall grading)" if hwall is not None else
               "-met (RBM volume grading)" if rbm is not None else "-optim")
        print("[build_iso] ============================================================\n"
              f"[build_iso] ERROR: mmg3d {tag} FAILED -> falling back to RAW TetGen mesh\n"
              "[build_iso] (quasi-uniform, slivers NOT cleaned, branch grading NOT enforced).\n"
              "[build_iso] Common cause: disk full on /tmp. Check `df -h /tmp` and re-run.\n"
              "[build_iso] ============================================================\n"
              + r.stderr[-400:])
        return g
    m = meshio.read("/tmp/_mmgopt_out.mesh")
    return pv.UnstructuredGrid({vtk.VTK_TETRA: m.cells_dict["tetra"]}, m.points)


def _edge_len(g):
    """Rough characteristic edge length per tet (mean of its 6 edges), for an hmax fallback."""
    tet = g.cells_dict[vtk.VTK_TETRA]
    P = g.points
    e = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return np.mean([np.linalg.norm(P[tet[:, a]] - P[tet[:, b]], axis=1) for a, b in e], axis=0)


def _wall_size_map(points, wall_idx, hwall, hgrad, cap):
    """Per-vertex isotropic target size for a graded near-wall tet refinement.

    size(x) = min( hwall * hgrad ** (dist_to_wall(x) / hwall), cap )

    i.e. the requested edge length is `hwall` on the wall and grows geometrically with the
    distance to the wall at ratio `hgrad` per ~one wall-edge step, capped at `cap` (the bulk
    interior size = RBM/hmax). This is the same distance-to-wall idea used by Option-4a /
    the RBM caliber logic, reused here purely to drive the spatial size field. mmg3d's own
    -hgrad then enforces the gradation on the actual element graph; this analytic field just
    seeds the target so mmg3d knows WHERE to refine."""
    wall_pts = points[wall_idx]
    d = cKDTree(wall_pts).query(points, k=1)[0]      # Euclidean distance to nearest wall node
    size = hwall * np.power(hgrad, d / max(hwall, 1e-9))
    return np.minimum(size, cap)


def _wall_metric_aniso(points, wall_surf, hwall_n, hgrad, cap):
    """Per-vertex ANISOTROPIC metric tensor for a boundary-layer-like near-wall refinement.

    For every VOLUME vertex x:
      - find the nearest WALL surface vertex (cKDTree on the wall vertices, same logic as the
        isotropic path) -> distance d and the unit wall NORMAL n at that nearest wall vertex;
      - normal target size   h_n(d) = min( hwall_n * hgrad**(d/hwall_n), h_t )   (grows away
        from the wall up to the bulk size, so deep interior -> isotropic bulk);
      - tangential target size h_t = cap (the bulk/RBM/hmax size; the same `cap` used by the
        isotropic --hwall path. A per-vertex bulk field is not available here, so we use the
        single bulk value `cap`, i.e. --hmax/RBM cap);
      - metric  M = (1/h_n^2) n(x)n + (1/h_t^2) (I - n(x)n).
    Returns an (N,3,3) array of symmetric positive-definite tensors (one per volume vertex).

    n(x)n is the outer product; (I - n(x)n) is the projector onto the tangent plane, so M has
    eigenvalue 1/h_n^2 along n (short normal edges) and 1/h_t^2 in the tangent plane (coarse
    tangential edges). mmg interprets edge length under M as sqrt(e^T M e); requesting length 1
    means an edge along n is ~h_n long and an edge in the tangent plane is ~h_t long."""
    wp = np.asarray(wall_surf.points)
    if "Normals" in wall_surf.point_data:
        wn = np.asarray(wall_surf.point_data["Normals"])
    else:
        wn = np.asarray(wall_surf.compute_normals(point_normals=True, cell_normals=False,
                                                  auto_orient_normals=True).point_data["Normals"])
    wn = wn / np.maximum(np.linalg.norm(wn, axis=1, keepdims=True), 1e-12)
    d, j = cKDTree(wp).query(points, k=1)            # nearest wall vertex: distance + index
    n = wn[j]                                         # unit wall normal at the nearest wall vertex
    h_t = float(cap)
    h_n = hwall_n * np.power(hgrad, d / max(hwall_n, 1e-9))
    h_n = np.minimum(h_n, h_t)                        # never finer-normal than the bulk far away
    inv_n2 = 1.0 / np.maximum(h_n, 1e-9) ** 2         # eigenvalue along the normal
    inv_t2 = 1.0 / max(h_t, 1e-9) ** 2                # eigenvalue in the tangent plane
    nn = n[:, :, None] * n[:, None, :]               # (N,3,3) outer products n (x) n
    I = np.eye(3)[None]
    return inv_n2[:, None, None] * nn + inv_t2 * (I - nn)


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


def _write_sol_aniso(path, M):
    """Write a medit ANISOTROPIC (tensor) metric (.sol) matching the .mesh vertex order.

    M : (N,3,3) array of symmetric 3x3 metric tensors (one per vertex).

    medit/mmg COMPONENT ORDERING (VERIFIED against mmg 5.8.0 source, src/common/inout.c):
      The medit symmetric-matrix file order is the COLUMN-MAJOR upper triangle:
          m11  m12  m22  m13  m23  m33
      i.e.  Mxx  Mxy  Myy  Mxz  Myz  Mzz.
      How verified: mmg stores a tensor internally as [Mxx,Mxy,Mxz,Myy,Myz,Mzz] (see
      MMG5_build3DMetric, which assembles dbuf[0..5] = Mxx,Mxy,Mxz,Myy,Myz,Mzz from R*diag*R^T),
      and BOTH the reader (MMG5_readDoubleSol3D) and writer (MMG5_writeDoubleSol3D) swap file
      slots 2<->3 between file and internal storage. Applying that swap to the internal order
      gives the FILE order Mxx,Mxy,Myy,Mxz,Myz,Mzz used here. (This is the standard medit
      convention; note it is NOT the row-major m11 m12 m13 m22 m23 m33.)
      The header line "1 3" = one solution field of type 3 (tensor) at each vertex.
    """
    with open(path, "w") as f:
        f.write("MeshVersionFormatted 2\nDimension 3\n\nSolAtVertices\n%d\n1 3\n" % len(M))
        for m in M:
            # FILE order: m11 m12 m22 m13 m23 m33  ==  Mxx Mxy Myy Mxz Myz Mzz
            f.write("%.9g %.9g %.9g %.9g %.9g %.9g\n" %
                    (m[0, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2], m[2, 2]))
        f.write("End\n")


def build(stl_ref, orig_surf_dir, out_dir, surf_hmax=0.5, rbm_n_across=None, hmin=0.2, hmax=None,
          hwall=None, hgrad=1.2, hwall_aniso=None):
    import tetgen, meshio
    if hwall is not None and hwall_aniso is not None:
        raise SystemExit("[build_iso] --hwall and --hwall-aniso are mutually exclusive "
                         "(pick one near-wall metric).")
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
                        "-hgrad", "1.3", "-hausd", "0.20", "-nr", "-out", "/tmp/_iso_surf.mesh"],
                       check=True, capture_output=True)
        mm = meshio.read("/tmp/_iso_surf.mesh")
        tri = mm.cells_dict["triangle"]
        sm = pv.PolyData(mm.points, np.hstack([np.full((len(tri), 1), 3), tri]).ravel()).clean()
        sm.clear_data()

    # 3. TetGen: tet volume (the graded surface drives the volume grading)
    tet = tetgen.TetGen(sm)
    tet.tetrahedralize(order=1, mindihedral=18, minratio=1.4)
    g = tet.grid
    # remove slivers (poison the rest-shape morph energy -> inverted tets); when --hwall or
    # --hwall-aniso is set, the same pass ALSO grades the interior near the wall for WSS
    # resolution (isotropic size map, or anisotropic BL-like tensor metric respectively).
    # RBM (no near-wall metric): drive the volume pass with the radius-based metric so the
    # branch refinement survives the interior cleanup instead of -optim homogenising it.
    rbm_vol = (rbm_n_across, hmin) if (rbm_n_across is not None and hwall is None
                                       and hwall_aniso is None) else None
    g = _mmg_optimize(g, hwall=hwall, hgrad=hgrad, hmax=hmax, hwall_aniso=hwall_aniso, rbm=rbm_vol)
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
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--hwall", type=float, default=None, metavar="H",
                     help="near-wall ISOTROPIC target tet size in mm at the wall surface "
                          "(default off); grades to the bulk size for WSS resolution")
    grp.add_argument("--hwall-aniso", dest="hwall_aniso", type=float, default=None, metavar="H",
                     help="near-wall ANISOTROPIC target NORMAL tet size in mm at the wall "
                          "(default off); tangential size = bulk (--hmax/--rbm). BL-like "
                          "normal resolution in an all-tet mesh, count-controlled (no prisms). "
                          "Mutually exclusive with --hwall.")
    ap.add_argument("--hgrad", type=float, default=1.2, metavar="R",
                    help="geometric growth ratio of the near-wall size map/metric AND mmg3d "
                         "-hgrad (default 1.2)")
    a = ap.parse_args()
    build(a.stl_ref, a.orig_surf_dir, a.out_dir, a.surf_hmax,
          rbm_n_across=a.rbm, hmin=a.hmin, hmax=a.hmax, hwall=a.hwall, hgrad=a.hgrad,
          hwall_aniso=a.hwall_aniso)
