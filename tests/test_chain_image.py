"""
Image-registration wall chain (chain_image/) and tools/phase_subset_wall.py, on a synthetic
pulsating cylinder -- no images, no elastix, no solver:
  - transform inversion on an analytic smooth displacement field;
  - periodic PCHIP reproduces its keyframes and is periodic;
  - smooth -> taper -> write chain on a noisy synthetic wall field: rings pinned, reference
    phase unchanged, displacement file consistent with the mesh at t = 0, inlet flow negative;
  - phase subsets: keeping every phase is the identity, a subset reproduces its keyframes.
Run:  python3 tests/test_chain_image.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "chain_image"))
import synth
import wallfield as wf

checks = []


def check(name, ok, info=""):
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + info) if info and not ok else ''}")


def run(cmd):
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    # 1. inversion of x -> x + d(x) for a smooth, invertible analytic field
    rng = np.random.default_rng(1)
    P = rng.uniform(-10, 10, (500, 3))
    disp = lambda x: np.column_stack([1.5 * np.sin(x[:, 1] / 6), 1.0 * np.cos(x[:, 2] / 7),   # noqa: E731
                                      0.8 * np.sin((x[:, 0] + x[:, 1]) / 9)])
    x, rn = wf.invert_points(P, disp, tol=1e-6)
    check("transform inversion: x + d(x) = P", rn.max() < 1e-6 and np.abs(x + disp(x) - P).max() < 1e-6,
          f"residual {rn.max():.1e}")
    bad = np.zeros(len(P), bool); bad[:5] = True
    xf = wf.idw_fill(P, x, bad)
    check("IDW fill touches only the flagged nodes", np.array_equal(xf[~bad], x[~bad]) and np.isfinite(xf).all())

    # 2. periodic PCHIP
    tk = wf.phase_times(8, 0.8)
    vals = rng.random((8, 4, 3))
    for wrap in ("pad", "append"):
        f = wf.periodic_pchip(tk, vals, 0.8, wrap=wrap)
        ok = np.allclose(f(tk), vals) and np.allclose(f(np.array([0.13])), f(np.array([0.93])))
        check(f"periodic PCHIP ({wrap}): keyframes reproduced, periodic", ok)

    # 3. synthetic wall field on the test cylinder (wall / inlet / outlet)
    tmp = tempfile.mkdtemp(prefix="mbcfd_chain_")
    g = synth.make_mesh(tmp)
    import pyvista as pv
    mesh = os.path.join(tmp, "mesh-complete.mesh.vtu")
    surf = os.path.join(tmp, "mesh-surfaces")
    wall = pv.read(os.path.join(surf, "wall.vtp"))
    gid = np.asarray(wall.point_data["GlobalNodeID"]).astype(np.int64)
    X = np.asarray(g.points)[gid - 1]
    nph, ref, T = 10, 3, 0.8
    r = np.hypot(X[:, 0], X[:, 1])
    radial = np.column_stack([X[:, 0] / r, X[:, 1] / r, np.zeros(len(X))])
    amp = 0.4 * np.sin(2 * np.pi * (np.arange(nph) - ref) / nph)            # 0 at the reference
    pos = X[None] + amp[:, None, None] * radial[None]
    pos[np.arange(nph) != ref] += 0.02 * rng.standard_normal((nph - 1, len(X), 3))   # registration noise
    raw = os.path.join(tmp, "raw.npz")
    wf.save_field(raw, pos, gid, ref_phase=ref)

    sm = os.path.join(tmp, "smooth.npz")
    rc, out = run(["chain_image/smooth_wall_field.py", "--in", raw, "--wall-vtp", os.path.join(surf, "wall.vtp"),
                   "--period", str(T), "--out", sm, "--dt", "0.02"])
    ok = rc == 0 and os.path.exists(sm)
    if ok:
        ps, _, rr = wf.load_field(sm)
        ok = rr == ref and np.allclose(ps[ref], pos[ref])
    check("smooth_wall_field: runs, reference phase unchanged", ok, out[-500:])

    tp = os.path.join(tmp, "taper.npz")
    rc, out = run(["chain_image/taper_wall_field.py", "--in", sm, "--mesh", mesh, "--surfaces", surf,
                   "--caps", "inlet", "outlet", "--length", "4", "--sweep", "0", "4", "--out", tp])
    ok = rc == 0 and os.path.exists(tp)
    if ok:
        pt, _, _ = wf.load_field(tp)
        ring = np.isin(gid, np.concatenate([pv.read(os.path.join(surf, f"{c}.vtp")).point_data["GlobalNodeID"]
                                            for c in ("inlet", "outlet")]))
        ok = ring.any() and np.abs(pt[:, ring] - X[ring]).max() < 1e-12 and \
            np.linalg.norm(pt - X, axis=2).max() <= np.linalg.norm(ps - X, axis=2).max() + 1e-12
    check("taper_wall_field: cap rings pinned, motion never amplified", ok, out[-500:])

    csv = os.path.join(tmp, "inlet.csv")
    with open(csv, "w") as f:
        f.write("t_ms;Q_mlps\n")
        for t in np.linspace(0, 760, 20):
            f.write(f"{t:.1f};{50 + 200 * max(0.0, np.sin(np.pi * t / 300)):.3f}\n")
    disp_txt, flow_txt = os.path.join(tmp, "displacement_wall.txt"), os.path.join(tmp, "flow.txt")
    rc, out = run(["chain_image/write_wall_displacement.py", "--in", tp, "--mesh", mesh, "--period", str(T),
                   "--dt", "0.01", "--cycles", "2", "--preroll", "0.05", "--out", disp_txt,
                   "--flow-in", csv, "--flow-col", "1", "--flow-out", flow_txt])
    ok = rc == 0
    if ok:
        toks = open(disp_txt).read().split()
        n, nt = int(toks[0]), int(toks[1])
        blk = np.asarray(toks[2:], float).reshape(nt, 1 + 4 * n)
        first = blk[0, 1:].reshape(n, 4)
        t5 = blk[5, 1:].reshape(n, 4)                     # end of the pre-roll = reference phase
        fr = np.loadtxt(flow_txt, skiprows=1)
        ok = (n == len(gid) and nt == 5 + 5 + 160 and np.allclose(blk[:, 0], np.arange(nt) * 0.01)
              and np.abs(first[:, 1:] - X * 0.1).max() < 2e-6
              and np.abs(t5[:, 1:] - pt[ref] * 0.1).max() < 2e-6
              and abs(fr[0, 1]) < 1e-9 and (fr[1:, 1] < 0).all())     # ramp from 0, then inflow
    check("write_wall_displacement: header, clock, t=0 = mesh, negative inflow", ok, out[-500:])

    # 4. phase subsets
    sub_all = os.path.join(tmp, "all.npz")
    rc, out = run(["tools/phase_subset_wall.py", raw, "--keep"] + [str(k) for k in range(nph)] + ["--out", sub_all])
    ok = rc == 0 and np.allclose(wf.load_field(sub_all)[0], pos)
    check("phase_subset_wall: keeping every phase is the identity", ok, out[-400:])
    sub = os.path.join(tmp, "sub.npz")
    keep = [3, 5, 8]
    rc, out = run(["tools/phase_subset_wall.py", raw, "--keep"] + [str(k) for k in keep] + ["--out", sub])
    ok = rc == 0
    if ok:
        q, gq, rq = wf.load_field(sub)
        ok = np.allclose(q[keep], pos[keep]) and np.array_equal(gq, gid) and rq == ref and "error vs full field" in out
    check("phase_subset_wall: subset reproduces its keyframes, report printed", ok, out[-400:])
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n=== chain_image: {sum(checks)}/{len(checks)} PASS ===")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
