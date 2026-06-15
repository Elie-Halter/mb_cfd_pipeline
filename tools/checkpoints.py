#!/usr/bin/env python3
"""
Measurements at fixed CHECKPOINTS along the aorta (step 1/3).

Cut planes (origin + normal) -- the SAME ones for every patient/segmentation --
are defined in a JSON file. For each plane and each VTU snapshot, we compute:
  - cross-section area              [mm^2]
  - equivalent diameter 2*sqrt(A/pi) [mm]
  - flow rate Q = integral(v . n) dA [mL/s]  (triangle integration, /100: cm/s * mm^2 -> mL/s)
  - mean in-plane velocity          [cm/s]

Unit convention: VTU coords in mm, Velocity in cm/s.

checkpoints.json format:
  [{"name": "asc_prox", "origin": [x,y,z], "normal": [nx,ny,nz]}, ...]
(coords in mm, in the mesh frame)

Usage:
  python3 checkpoints.py <vtu|results_dir> checkpoints.json [--out measures.csv]
Help for creating the JSON:
  python3 checkpoints.py --make-template centerline.vtp 10 > checkpoints.json
    (samples 10 planes along a VMTK/VTP centerline, normals = tangents)
"""
import sys, os, glob, json, argparse
import numpy as np
import pyvista as pv


def slice_metrics(mesh, origin, normal):
    sl = mesh.slice(origin=origin, normal=normal)
    if sl.n_points == 0 or sl.n_cells == 0:
        return None
    sl = sl.triangulate()
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    pts = sl.points
    vel = np.asarray(sl.point_data["Velocity"]) if "Velocity" in sl.point_data else None
    faces = sl.faces.reshape(-1, 4)[:, 1:]
    area = 0.0
    Q = 0.0
    vsum = np.zeros(3)
    for tri in faces:
        i, j, k = tri
        avec = 0.5 * np.cross(pts[j] - pts[i], pts[k] - pts[i])  # mm^2
        a = np.linalg.norm(avec)
        area += a
        if vel is not None:
            vtri = (vel[i] + vel[j] + vel[k]) / 3.0   # cm/s
            Q += np.dot(vtri, n) * a                  # cm/s * mm^2
            vsum += vtri * a
    diam = 2.0 * np.sqrt(area / np.pi) if area > 0 else 0.0
    vmean = np.linalg.norm(vsum / area) if area > 0 and vel is not None else float("nan")
    return dict(area=area, diam=diam, Q=Q / 100.0, vmean=vmean)


def make_template(centerline_path, n):
    cl = pv.read(centerline_path)
    pts = cl.points
    idx = np.linspace(0, len(pts) - 1, n).astype(int)
    out = []
    for c, i in enumerate(idx):
        i0 = max(0, i - 1); i1 = min(len(pts) - 1, i + 1)
        tan = pts[i1] - pts[i0]; tan /= (np.linalg.norm(tan) + 1e-12)
        out.append({"name": f"cp{c:02d}", "origin": pts[i].tolist(),
                    "normal": tan.tolist()})
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--make-template", nargs=2, metavar=("CENTERLINE", "N"))
    args = ap.parse_args()

    if args.make_template:
        make_template(args.make_template[0], int(args.make_template[1]))
        return

    src, cp_json = args.inputs[0], args.inputs[1]
    checkpoints = json.load(open(cp_json))
    if os.path.isdir(src):
        vtus = sorted(glob.glob(os.path.join(src, "results_*.vtu")),
                      key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f)))))
    else:
        vtus = [src]
    if not vtus:
        raise SystemExit(f"no VTU in {src}")

    rows = []
    for f in vtus:
        step = int("".join(filter(str.isdigit, os.path.basename(f))) or 0)
        mesh = pv.read(f)
        for cp in checkpoints:
            m = slice_metrics(mesh, cp["origin"], cp["normal"])
            if m:
                rows.append((step, cp["name"], m["diam"], m["area"], m["Q"], m["vmean"]))

    hdr = f"{'step':>6} {'checkpoint':12} {'diam_mm':>9} {'area_mm2':>10} {'Q_mL/s':>9} {'vmean_cm/s':>11}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r[0]:6d} {r[1]:12} {r[2]:9.2f} {r[3]:10.1f} {r[4]:9.2f} {r[5]:11.2f}")
    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "checkpoint", "diam_mm", "area_mm2", "Q_mLs", "vmean_cms"])
            w.writerows(rows)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
