#!/usr/bin/env python3
"""cohort_cap_label.py — cap + classify the openings of a raw (open) lumen surface.

Emits a CLOSED capped STL and labeled reference cap surfaces
(mesh-surfaces/{asc,desc,btca,lcca,lsa}.vtp) that build_iso_mesh.py consumes to
tag caps on the new tet mesh. This is the per-patient upstream step that
SimVascular did for 4826; here it is deterministic and auditable for the cohort.

Classification rule (validated vs 4826 ground truth, see --selftest):
  * the 3 SMALLEST openings (by disc diameter)      -> branches
  * of the 2 LARGEST openings, the larger           -> asc (inlet, near root)
                               the smaller           -> desc (main outlet)
  * branch naming btca<lcca<lsa = order of the branch centroids projected onto
    the (asc_centroid -> desc_centroid) direction (ascending). On 4826 this
    reproduces the anatomical proximal->distal order exactly.

A QC render (qc_caps.png) is written so the branch naming can be visually
confirmed before it is used for RCR assignment. Mesh generation itself is
name-agnostic (each cap is tagged by its own geometry).

Usage:
  cohort_cap_label.py RAW.(stl|vtp) OUT_DIR
  cohort_cap_label.py --selftest WALL_4826.vtp   # non-regression check
"""
import sys, os, argparse
import numpy as np
import pyvista as pv

CAPS = ["asc", "desc", "btca", "lcca", "lsa"]


def load_surface(path):
    s = pv.read(path)
    if not isinstance(s, pv.PolyData):
        s = s.extract_surface()
    # STL stores duplicated per-triangle vertices -> clean to share points, else
    # every edge looks like a boundary edge.
    s = s.clean(tolerance=1e-6).triangulate()
    return s


def boundary_loops(surf):
    """Return list of (loop_line_polydata) — one per open ring."""
    edges = surf.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                       manifold_edges=False, non_manifold_edges=False)
    if edges.n_cells == 0:
        return []
    conn = edges.connectivity()
    rid = np.asarray(conn.cell_data["RegionId"])
    loops = []
    for r in np.unique(rid):
        loops.append(conn.extract_cells(np.where(rid == r)[0]).extract_surface())
    return loops


def loop_geom(loop):
    pts = loop.points
    c = pts.mean(0)
    # plane normal = smallest singular vector; diameter from in-plane extent
    _, _, vt = np.linalg.svd(pts - c)
    nrm = vt[2] / np.linalg.norm(vt[2])
    rad = np.linalg.norm(pts - c, axis=1).max()
    return c, nrm, 2.0 * rad


def cap_loop(loop):
    """Fan-triangulate a boundary loop to its centroid -> a cap disc PolyData.
    Guaranteed watertight (reuses the exact rim points; TetGen requires closure).
    The thin fan triangles it creates on large openings are re-meshed by the
    downstream mmgs surface pass; rim slivers are controlled there via -hausd."""
    pts = loop.points
    c = pts.mean(0)
    lines = loop.lines.reshape(-1, 3)[:, 1:]  # pairs of local point ids
    ci = len(pts)
    allpts = np.vstack([pts, c])
    faces = np.hstack([[3, a, b, ci] for a, b in lines]).astype(np.int64)
    return pv.PolyData(allpts, faces)


def classify(loops):
    geoms = [loop_geom(l) for l in loops]
    cents = np.array([g[0] for g in geoms])
    diams = np.array([g[2] for g in geoms])
    order = np.argsort(diams)          # ascending diameter
    branch_idx = list(order[:-2])       # 3 smallest
    big_idx = list(order[-2:])          # 2 largest
    asc_idx = big_idx[np.argmax(diams[big_idx])]
    desc_idx = big_idx[np.argmin(diams[big_idx])]
    # branch order: project onto asc->desc direction
    ad = cents[desc_idx] - cents[asc_idx]
    ad = ad / np.linalg.norm(ad)
    proj = [(cents[b] - cents[asc_idx]) @ ad for b in branch_idx]
    branch_sorted = [branch_idx[k] for k in np.argsort(proj)]  # btca,lcca,lsa
    mapping = {"asc": asc_idx, "desc": desc_idx,
               "btca": branch_sorted[0], "lcca": branch_sorted[1], "lsa": branch_sorted[2]}
    return mapping, cents, diams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--selftest", action="store_true",
                    help="raw is 4826 wall.vtp; print classification, do not write")
    a = ap.parse_args()

    surf = load_surface(a.raw)
    loops = boundary_loops(surf)
    print(f"[cap] {surf.n_points} pts, {surf.n_cells} tris, {len(loops)} openings")
    if len(loops) != 5:
        print(f"[cap] WARNING expected 5 openings, got {len(loops)} — inspect before use")
    mapping, cents, diams = classify(loops)
    for name in CAPS:
        i = mapping[name]
        c = cents[i]
        print(f"  {name:5s} <- opening {i}  centre=({c[0]:7.2f},{c[1]:7.2f},{c[2]:7.2f})  ~diam={diams[i]:6.1f}")

    if a.selftest:
        # expected 4826 centroids (from gci_fine/mesh-surfaces)
        gt = {"asc": (19.46, -24.40, 46.81), "desc": (31.50, 5.81, 6.91),
              "btca": (13.29, -16.54, 97.39), "lcca": (27.49, -12.83, 95.07),
              "lsa": (31.40, -7.59, 96.17)}
        ok = True
        for name in CAPS:
            c = cents[mapping[name]]
            err = np.linalg.norm(np.array(gt[name]) - c)
            tag = "OK " if err < 3.0 else "FAIL"
            if err >= 3.0:
                ok = False
            print(f"  [selftest] {name:5s} err={err:5.2f} {tag}")
        print("[selftest] " + ("PASS — reproduces 4826 labels" if ok else "FAIL"))
        return

    os.makedirs(os.path.join(a.out_dir, "mesh-surfaces"), exist_ok=True)
    caps = {name: cap_loop(loops[mapping[name]]) for name in CAPS}
    for name, cap in caps.items():
        cap.save(os.path.join(a.out_dir, "mesh-surfaces", f"{name}.vtp"))
    # closed solid = wall + caps
    solid = surf.copy()
    for cap in caps.values():
        solid = solid.merge(cap)
    solid = solid.clean(tolerance=1e-6).triangulate()
    open_after = solid.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                             manifold_edges=False, non_manifold_edges=False).n_cells
    solid.save(os.path.join(a.out_dir, "capped.stl"))
    with open(os.path.join(a.out_dir, "cap_report.txt"), "w") as f:
        f.write(f"openings={len(loops)} open_edges_after_capping={open_after}\n")
        for name in CAPS:
            i = mapping[name]; c = cents[i]
            f.write(f"{name} centre={c.tolist()} diam={diams[i]:.2f}\n")
    print(f"[cap] wrote capped.stl (open edges after capping = {open_after}) + 5 labeled caps -> {a.out_dir}")

    # QC render
    try:
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(surf, color="lightgray", opacity=0.3)
        col = {"asc": "red", "desc": "blue", "btca": "green", "lcca": "orange", "lsa": "magenta"}
        for name, cap in caps.items():
            pl.add_mesh(cap, color=col[name])
            pl.add_point_labels([cents[mapping[name]]], [name], font_size=18, point_size=1)
        pl.screenshot(os.path.join(a.out_dir, "qc_caps.png"))
        print("[cap] wrote qc_caps.png")
    except Exception as e:
        print(f"[cap] QC render skipped: {e}")


if __name__ == "__main__":
    main()
