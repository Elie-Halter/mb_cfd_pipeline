"""
preflight_lib.py -- checks run BEFORE any solver launch, none of which needs the solver:
well-formed XML whose referenced files exist, a valid tetrahedral mesh (positive volumes,
contiguous IDs, boundary = exact union of the faces), displacement files (header, GIDs,
position at t=0 == reference coordinates x scale), General BC files (exact nNo, t0 = 0).
"""
import os
import xml.etree.ElementTree as ET
import numpy as np

MM2CM = 0.1
_FAILS = []


def check(ok, msg):
    print(f"  [{'OK  ' if ok else 'FAIL'}] {msg}")
    if not ok:
        _FAILS.append(msg)
    return ok


def summary():
    print()
    if _FAILS:
        print(f"[PREFLIGHT] {len(_FAILS)} FAILURE(S) -- do not run:")
        for m in _FAILS:
            print("   -", m)
        return 1
    print("[PREFLIGHT] all checks pass -- case ready.")
    return 0


def check_xml(xml_path, expected_faces):
    print(f"-- XML {os.path.basename(xml_path)}")
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        check(False, f"malformed XML: {e}")
        return None
    check(True, "well-formed XML")
    root = tree.getroot()
    mesh = root.find("Add_mesh")
    faces = [f.get("name") for f in mesh.findall("Add_face")]
    check(sorted(faces) == sorted(expected_faces),
          f"declared faces {sorted(faces)} == expected {sorted(expected_faces)}")
    for el in root.iter():
        if el.text and el.tag.endswith(("_file_path", "_path")) and "/" in el.text:
            p = el.text.strip()
            check(os.path.exists(p), f"exists: {p}")
    for eq in root.findall("Add_equation"):
        bcs = [b.get("name") for b in eq.findall("Add_BC")]
        print(f"    equation {eq.get('type')}: BCs on {bcs}")
        check(len(bcs) == len(set(bcs)), f"no duplicated BC ({eq.get('type')})")
        check(set(bcs) <= set(faces), f"BCs ({eq.get('type')}) are mesh faces")
    return tree


def check_mesh(mesh_dir, face_names):
    import pyvista as pv
    print(f"-- mesh {mesh_dir}")
    vtu = os.path.join(mesh_dir, "mesh-complete.mesh.vtu")
    if not check(os.path.exists(vtu), f"exists: {vtu}"):
        return None, None
    g = pv.read(vtu)
    check(set(g.cells_dict.keys()) == {10}, "all-tetrahedral (VTK_TETRA only)")
    tets = g.cells_dict[10]
    P = np.asarray(g.points)
    a = P[tets[:, 0]]
    v6 = np.einsum("ij,ij->i", np.cross(P[tets[:, 1]] - a, P[tets[:, 2]] - a), P[tets[:, 3]] - a)
    check((np.abs(v6) > 0).all(), f"no degenerate tet (min|6V| = {np.abs(v6).min():.3e} mm3)")
    gid = np.asarray(g.point_data["GlobalNodeID"])
    check((gid == np.arange(1, g.n_points + 1)).all(), "GlobalNodeID contiguous 1..N")
    eid = np.asarray(g.cell_data["GlobalElementID"])
    check((eid == np.arange(1, g.n_cells + 1)).all(), "GlobalElementID contiguous 1..M")

    n_btris = g.extract_surface().triangulate().n_cells
    tot, face_data = 0, {}
    for nm in face_names:
        fp = os.path.join(mesh_dir, "mesh-surfaces", f"{nm}.vtp")
        if not check(os.path.exists(fp), f"exists: {fp}"):
            continue
        s = pv.read(fp)
        fgid = np.asarray(s.point_data["GlobalNodeID"], dtype=np.int64)
        feid = np.asarray(s.cell_data["GlobalElementID"], dtype=np.int64)
        check(fgid.min() >= 1 and fgid.max() <= g.n_points, f"{nm}: node GIDs in [1,N]")
        check(feid.min() >= 1 and feid.max() <= g.n_cells, f"{nm}: parent GlobalElementID in [1,M]")
        d = np.linalg.norm(np.asarray(s.points) - P[fgid - 1], axis=1).max()
        check(d < 1e-9, f"{nm}: vtp points == vtu points (max gap {d:.2e} mm)")
        tris = s.faces.reshape(-1, 4)[:, 1:]
        tri_gids = np.sort(fgid[tris], axis=1)
        parents = np.sort(tets[feid - 1] + 1, axis=1)
        ok_par = all(np.isin(t, p).all() for t, p in zip(tri_gids, parents))
        check(ok_par, f"{nm}: every triangle belongs to its parent tet")
        tot += s.n_cells
        face_data[nm] = (fgid, s)
    check(tot == n_btris, f"faces cover the boundary: {tot} tris == {n_btris} boundary tris")
    return g, face_data


def check_disp_file(path, g, expect_gids, period, pos_fn_mm=None, tol=2e-7):
    """Header, GIDs, period, and position(t=0) == reference x scale (the reader subtracts
    the reference coordinates, so the initial displacement must be ~0). pos_fn_mm(t) -> (n,3)
    optionally checks an intermediate sample against the analytic motion."""
    print(f"-- displacement {os.path.basename(path)}")
    if not check(os.path.exists(path), f"exists: {path}"):
        return
    with open(path) as f:
        toks = f.read().split()
    nN, nT = int(toks[0]), int(toks[1])
    check(nN == len(expect_gids), f"header nN={nN} == expected nodes {len(expect_gids)}")
    need = 2 + nT * (1 + 4 * nN)
    check(len(toks) == need, f"file size: {len(toks)} tokens == {need} expected")
    blocks = np.asarray(toks[2:], dtype=float).reshape(nT, 1 + 4 * nN)
    times = blocks[:, 0]
    check(abs(times[0]) < 1e-12, "t0 = 0")
    check((np.diff(times) > 0).all(), "strictly increasing times")
    check(abs(times[-1] - period) < 1e-9, f"period = {times[-1]} == {period}")
    data0 = blocks[0, 1:].reshape(nN, 4)
    gids = data0[:, 0].astype(np.int64)
    check(set(gids) == set(np.asarray(expect_gids, dtype=np.int64)), "GID set == expected")
    P = np.asarray(g.points)
    ref_cm = P[gids - 1] * MM2CM
    d0 = np.linalg.norm(data0[:, 1:4] - ref_cm, axis=1).max()
    check(d0 < tol, f"position(t=0) == reference x 0.1 (max gap {d0:.2e} cm, file rounding)")
    if pos_fn_mm is not None:
        k = nT // 2
        datk = blocks[k, 1:].reshape(nN, 4)
        order = {int(gg): i for i, gg in enumerate(gids)}
        idx = np.array([order[int(gg)] for gg in datk[:, 0]])
        dk = np.linalg.norm(datk[:, 1:4] - pos_fn_mm(times[k])[idx] * MM2CM, axis=1).max()
        check(dk < tol, f"position(t={times[k]:.3f}) == analytic (max gap {dk:.2e} cm)")
    dmax = max(np.linalg.norm(blocks[k, 1:].reshape(nN, 4)[:, 1:4] - ref_cm, axis=1).max()
               for k in range(nT))
    print(f"    max |displacement| over the cycle = {dmax:.4f} cm")


def check_genbc_file(path, face_gids, ndof=3):
    print(f"-- General BC {os.path.basename(path)}")
    if not check(os.path.exists(path), f"exists: {path}"):
        return
    with open(path) as f:
        toks = f.read().split()
    nd, nT, nN = int(toks[0]), int(toks[1]), int(toks[2])
    check(nd == ndof, f"ndof = {nd}")
    check(nN == len(face_gids), f"nNo = {nN} == face nodes {len(face_gids)} (must be exact)")
    times = np.asarray(toks[3:3 + nT], dtype=float)
    check(abs(times[0]) < 1e-12, "t0 = 0")
    check((np.diff(times) > 0).all(), "increasing times")
    body = toks[3 + nT:]
    check(len(body) == nN * (1 + nT * nd), "consistent body size")
    gids = np.asarray(body[:: 1 + nT * nd], dtype=np.int64)
    check(set(gids) == set(np.asarray(face_gids, dtype=np.int64)), "GIDs == face nodes")
