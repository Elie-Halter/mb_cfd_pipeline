"""
Meshing tests: the radius-based size map (unit), and the MMG/TetGen toolchain on a
tiny surface (gated on the binaries / python packages being installed).
Run:  python3 tests/test_meshing.py
"""
import sys, os, shutil, subprocess, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "tools"))
import synth


def main():
    checks = []
    def check(name, ok, info=""):
        checks.append(ok); print(f"  [{'PASS' if ok else ('SKIP' if info == 'skip' else 'FAIL')}] {name}"
                                 f"{(' (' + info + ')') if info else ''}")

    tmp = tempfile.mkdtemp(prefix="mbcfd_mesh_")
    g = synth.make_mesh(tmp)
    surf = g.extract_surface().triangulate()

    # 1. radius-based caliber map (unit, no binary) -> positive, finite, varies
    import build_iso_mesh as B
    cal = B._caliber(surf)
    check("build_iso_mesh._caliber (radius map)",
          np.all(np.isfinite(cal)) and cal.min() > 0 and cal.max() > cal.min())

    # 2. MMG size-map round trip (needs mmgs_O3)
    if shutil.which("mmgs_O3"):
        import meshio
        faces = surf.faces.reshape(-1, 4)[:, 1:]
        meshio.write(tmp + "/s.mesh", meshio.Mesh(surf.points, [("triangle", faces)]))
        B._write_sol(tmp + "/s.sol", np.clip(cal / 6.0, 0.3, 2.0))
        rc = subprocess.run(["mmgs_O3", "-in", tmp + "/s.mesh", "-met", tmp + "/s.sol",
                             "-hmin", "0.3", "-hmax", "2.0", "-hgrad", "1.3", "-nr",
                             "-out", tmp + "/o.mesh"], capture_output=True).returncode
        check("mmgs_O3 size-map adaptation", rc == 0 and os.path.exists(tmp + "/o.mesh"))
    else:
        check("mmgs_O3 size-map adaptation", True, "skip")  # mmg not installed

    # 3. TetGen tetrahedralization (needs python 'tetgen')
    try:
        import tetgen
        t = tetgen.TetGen(surf); t.tetrahedralize(order=1, mindihedral=10, minratio=2.0)
        check("tetgen tetrahedralize", t.grid.n_cells > 0)
    except ImportError:
        check("tetgen tetrahedralize", True, "skip")  # tetgen not installed

    npass = sum(1 for c in checks if c)
    print(f"\n=== meshing: {npass}/{len(checks)} OK ===")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
