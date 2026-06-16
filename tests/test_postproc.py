"""
Component tests: run every post-processing tool on a tiny synthetic case
(no patient data, no solver) and check it succeeds and produces its output.
Run:  python3 tests/test_postproc.py
"""
import sys, os, subprocess, tempfile, glob
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import synth


def run(cmd):
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)


def main():
    tmp = tempfile.mkdtemp(prefix="mbcfd_test_")
    g = synth.make_mesh(tmp)
    fb = synth.make_results(os.path.join(tmp, "FB"), g, n_frames=8, moving=False, seed=1)
    mb = synth.make_results(os.path.join(tmp, "MB"), g, n_frames=8, moving=True, seed=2)
    wall = os.path.join(tmp, "mesh-surfaces", "wall.vtp")
    surf = os.path.join(tmp, "mesh-surfaces")
    s0, s1 = 10, 80

    checks = []

    def check(name, ok, info=""):
        checks.append((name, ok, info))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + info) if info and not ok else ''}")

    # 1. hemo_indices -> TAWSS/OSI wall VTP
    rc, out = run(["tools/hemo_indices.py", fb, "--wall", wall,
                   "--cycle-start", str(s0), "--cycle-end", str(s1), "--out-prefix", tmp + "/h_"])
    check("hemo_indices.py (TAWSS/OSI)", rc == 0 and os.path.exists(tmp + "/h_wall_TAWSS_OSI.vtp"), out[-400:])

    # 2. extract_flowsplit_FB -> flow split
    rc, out = run(["tools/extract_flowsplit_FB.py", fb, surf, "--outlets", "outlet"])
    check("extract_flowsplit_FB.py (flow split)", rc == 0 and "split" in out.lower(), out[-400:])

    # 3. compare_FB_MB -> difference wall VTP
    rc, out = run(["tools/compare_FB_MB.py", "--fb", fb, wall, str(s0), str(s1),
                   "--mb", mb, wall, str(s0), str(s1), "--out-prefix", tmp + "/cmp_"])
    cmpvtp = tmp + "/cmp_FB_vs_MB_wall.vtp"
    check("compare_FB_MB.py (FB vs MB)", rc == 0 and os.path.exists(cmpvtp), out[-400:])

    # 4. make_figures -> PNGs
    if os.path.exists(cmpvtp):
        rc, out = run(["tools/make_figures.py", "--wall-vtp", cmpvtp, "--out-dir", tmp + "/figs"])
        npng = len(glob.glob(tmp + "/figs/*.png"))
        check("make_figures.py (figures)", rc == 0 and npng >= 3, f"{npng} png; " + out[-300:])
    else:
        check("make_figures.py (figures)", False, "skipped (no cmp VTP)")

    # 5. gci -> apparent order p == 2 on the analytic case
    rc, out = run(["tools/gci.py", "--h", "1", "2", "4", "--phi", "11", "14", "26"])
    check("gci.py (GCI, p=2 analytic)", rc == 0 and ("2.000" in out or "p          = 2" in out), out[-300:])

    # 6. make_patient_xml -> valid FB/MB XML from templates
    rc, out = run(["tools/make_patient_xml.py", "--fb-template", "FB_example.xml",
                   "--mb-template", "MB_example.xml", "--mesh", tmp + "/mesh-complete.mesh.vtu",
                   "--surf", surf, "--disp", "/dev/null", "--fb-out", tmp + "/FB.xml",
                   "--mb-out", tmp + "/MB.xml"])
    ok = rc == 0 and os.path.exists(tmp + "/MB.xml")
    if ok:
        import xml.etree.ElementTree as ET
        try:
            ET.parse(tmp + "/FB.xml"); ET.parse(tmp + "/MB.xml")
        except Exception as e:
            ok = False; out += str(e)
    check("make_patient_xml.py (derive XML)", ok, out[-300:])

    npass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n=== post-processing: {npass}/{len(checks)} PASS ===")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
