"""
tools/check_geometry_vs_image.py on a synthetic bright tube in a 3-D image -- no patient data:
  - the tube's own surface passes (scale ~1, small rigid residual);
  - the same surface shrunk to 2/3 about its centre fails (scale ~1.5 reported);
  - the surface mirrored in x and y (RAS instead of LPS) fails, and passes again with --ras.
Run:  python3 tests/test_geometry_check.py
"""
import json
import os
import subprocess
import sys
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
checks = []


def check(name, ok, info=""):
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + info) if info and not ok else ''}")


def run(args, js):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_geometry_vs_image.py")] + args + ["--json", js],
                       capture_output=True, text=True)
    return r.returncode, json.load(open(js)) if os.path.exists(js) else {}


def main():
    try:
        import SimpleITK as sitk
    except ImportError:
        print("  [SKIP] SimpleITK not installed (pip install SimpleITK)")
        return 0
    import pyvista as pv
    d = tempfile.mkdtemp(prefix="geomcheck_")
    # image: bent bright tube (radius 10 mm) in a 160 mm box, off-centre so that mirroring moves it
    sp = 1.5
    n = 96
    o = np.array([-20.0, -30.0, 0.0])
    z, y, x = np.meshgrid(*[o[::-1][i] + sp * np.arange(n) for i in range(3)], indexing="ij")
    cx = 40.0 + 15.0 * np.sin(z / 40.0)
    tube = ((x - cx) ** 2 + (y - 20.0) ** 2 < 10.0 ** 2) & (z > 15) & (z < 120)
    img = sitk.GetImageFromArray((100.0 + 300.0 * tube + 5.0 * np.random.default_rng(0).standard_normal(tube.shape)).astype(np.float32))
    seg = sitk.GetImageFromArray(tube.astype(np.uint8))
    for im in (img, seg):
        im.SetSpacing((sp,) * 3); im.SetOrigin(tuple(o))
    sitk.WriteImage(img, f"{d}/img.nii.gz"); sitk.WriteImage(seg, f"{d}/seg.nrrd")
    g = pv.ImageData(dimensions=(n, n, n), spacing=(sp,) * 3, origin=tuple(o))
    g.point_data["m"] = tube.astype(np.float32).ravel()
    s = g.contour([0.5], scalars="m").triangulate().clean().smooth_taubin(n_iter=30)
    s.save(f"{d}/ok.stl")
    c = s.points.mean(0); small = s.copy(); small.points = c + (s.points - c) / 1.5; small.save(f"{d}/small.stl")
    ras = s.copy(); ras.points[:, :2] *= -1; ras.save(f"{d}/ras.stl")
    base = ["--image", f"{d}/img.nii.gz", "--seg", f"{d}/seg.nrrd"]

    rc, r = run([f"{d}/ok.stl"] + base, f"{d}/ok.json")
    check("right surface passes", rc == 0 and abs(r["similarity"]["scale"] - 1) < 0.02, str(r.get("reasons")))
    rc, r = run([f"{d}/small.stl"] + base, f"{d}/small.json")
    check("surface shrunk to 2/3 fails with scale ~1.5", rc == 1 and abs(r["similarity"]["scale"] - 1.5) < 0.05,
          f"rc {rc}, scale {r.get('similarity', {}).get('scale')}")
    rc, r = run([f"{d}/ras.stl"] + base, f"{d}/ras.json")
    check("mirrored (RAS) surface fails", rc == 1, f"rc {rc}")
    rc, r = run([f"{d}/ras.stl", "--ras"] + base, f"{d}/ras2.json")
    check("mirrored surface passes with --ras", rc == 0, str(r.get("reasons")))
    print(f"{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
