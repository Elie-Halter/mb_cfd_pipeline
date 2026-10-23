"""
Benchmark-suite tests that need neither the solver nor any data:
  - the analytic Womersley solution against its defining ODE (self-test);
  - the P1 divergence / flux / volume integrals of benchmarks/common on exact fields;
  - XML templates render to well-formed XML with every placeholder filled;
  - the B1 and B2 generators produce valid solver meshes and input files (pre-flight checks
    pass), gated on gmsh being installed.
Run:  python3 tests/test_benchmarks.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "benchmarks")
sys.path.insert(0, os.path.join(BENCH, "common"))

checks = []


def check(name, ok, info=""):
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + info) if info and not ok else ''}")


def run(cmd):
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    # 1. analytic Womersley self-test
    rc, out = run([os.path.join(BENCH, "B3_womersley", "womersley.py")])
    check("Womersley closed form (no-slip, ODE residual, Poiseuille limit, flow integral)",
          rc == 0 and "SELF-TEST OK" in out, out[-400:])

    # 2. P1 integrals on exact fields
    import benchtools as bt
    rng = np.random.default_rng(0)
    pts = rng.random((60, 3))
    import scipy.spatial
    tets = scipy.spatial.Delaunay(pts).simplices
    A = np.array([[0.3, 1.0, -0.2], [0.5, -0.7, 0.1], [0.0, 0.4, 0.9]])
    u = pts @ A.T + 1.0                                   # linear field: div = trace(A)
    div, vol = bt.cell_divergence(pts, tets, u)
    check("P1 divergence exact for a linear field", np.allclose(div, np.trace(A), atol=1e-9),
          f"max err {np.abs(div - np.trace(A)).max():.2e}")
    hull = scipy.spatial.ConvexHull(pts)
    check("tet volume sum == convex hull volume", abs(bt.tet_volume_sum(pts, tets) - hull.volume) < 1e-12)
    tri = np.array([[0, 1, 2]]); P = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0.]])
    q = bt.tri_flux(P, tri, np.tile([0, 0, 5.0], (3, 1)), (0, 0, 1))
    check("triangle flux (area 3 x u_n 5 = 15)", abs(q - 15.0) < 1e-12, str(q))
    amp, ph = bt.fit_sincos(np.linspace(0, 1, 41)[:-1], 2.0 * np.sin(2 * np.pi * np.linspace(0, 1, 41)[:-1] + 0.3),
                            2 * np.pi)
    check("sin/cos least-squares fit", abs(amp - 2.0) < 1e-12 and abs(ph - 0.3) < 1e-12)

    # 3. templates render
    import render_xml
    tmpl = {"B1_freestream": dict(CASE="/x"),
            "B2_expanding_tube": dict(CASE="/x", DT="0.005", NSTEPS="300", VTK_INC="5", RESTART_INC="100"),
            "B3_womersley": dict(CASE="/x", GRID="g1", NSTEPS="600")}
    for case, vals in tmpl.items():
        try:
            txt = render_xml.render(open(os.path.join(BENCH, case, "case.xml.in")).read(), vals)
            ok = "@" not in txt
        except SystemExit as e:
            ok, txt = False, str(e)
        check(f"{case}/case.xml.in renders to well-formed XML", ok, txt[-200:])

    # 4. generators + pre-flight (needs gmsh)
    try:
        import gmsh  # noqa: F401
    except ImportError:
        print("  [SKIP] case generators (gmsh not installed)")
    else:
        tmp = tempfile.mkdtemp(prefix="mbcfd_bench_")
        b1 = os.path.join(tmp, "B1")
        rc, out = run([os.path.join(BENCH, "B1_freestream", "gen_case.py"), "--out", b1])
        if rc == 0:
            with open(os.path.join(tmp, "b1.xml"), "w") as f:
                f.write(render_xml.render(open(os.path.join(BENCH, "B1_freestream", "case.xml.in")).read(),
                                          dict(CASE=b1)))
            rc, out = run([os.path.join(BENCH, "B1_freestream", "preflight.py"), "--case", b1,
                           "--xml", os.path.join(tmp, "b1.xml")])
        check("B1 generator -> pre-flight passes (mesh, motion, BC files, IC, XML)",
              rc == 0 and "all checks pass" in out, out[-600:])
        b2 = os.path.join(tmp, "B2")
        rc, out = run([os.path.join(BENCH, "B2_expanding_tube", "gen_case.py"), "--out", b2])
        if rc == 0:
            rc, out = run([os.path.join(BENCH, "B2_expanding_tube", "preflight.py"), "--case", b2])
        check("B2 generator -> pre-flight passes (mesh, analytic wall+cap motion, rings)",
              rc == 0 and "all checks pass" in out, out[-600:])
        shutil.rmtree(tmp, ignore_errors=True)

    n = len(checks)
    print(f"\n=== benchmarks: {sum(checks)}/{n} PASS ===")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
