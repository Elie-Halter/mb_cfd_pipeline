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

    # 6b. the derived MB XML is the production configuration: caps Dir 0 in the mesh equation,
    #     wall Prescribed_displacement (fluid + mesh); an EXTENDED all-node file is refused
    ok = False
    if os.path.exists(tmp + "/MB.xml"):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import make_patient_xml as mpx
        import xml.etree.ElementTree as ET
        root = ET.parse(tmp + "/MB.xml").getroot()
        mesh_eq = [e for e in root.findall("Add_equation") if e.get("type") == "mesh"][0]
        dir0 = {b.get("name") for b in mesh_eq.findall("Add_BC") if (b.findtext("Type") or "").strip() == "Dir"}
        ok = (mpx.check_production_mb(root) == [] and dir0 == {"asc", "desc", "btca", "lcca", "lsa"})
        bad = tmp + "/MB_example_nocaps.xml"
        t = ET.parse(os.path.join(ROOT, "MB_example.xml"))
        me = [e for e in t.getroot().findall("Add_equation") if e.get("type") == "mesh"][0]
        for b in [b for b in me.findall("Add_BC") if b.get("name") != "wall"]:
            me.remove(b)
        t.write(bad)
        ext = tmp + "/disp_all_nodes.txt"
        with open(ext, "w") as f:
            f.write("1 1\nEXTENDED\n1\n0.0\n1 0 0 0\n")
        rc1, _ = run(["tools/make_patient_xml.py", "--fb-template", "FB_example.xml",
                      "--mb-template", bad, "--mesh", "m.vtu", "--surf", surf, "--disp", "/dev/null",
                      "--fb-out", tmp + "/FB2.xml", "--mb-out", tmp + "/MB2.xml"])
        rc2, _ = run(["tools/make_patient_xml.py", "--fb-template", "FB_example.xml",
                      "--mb-template", "MB_example.xml", "--mesh", "m.vtu", "--surf", surf,
                      "--disp", ext, "--fb-out", tmp + "/FB3.xml", "--mb-out", tmp + "/MB3.xml"])
        ok = ok and rc1 != 0 and rc2 != 0
    check("make_patient_xml.py (production MB: caps Dir 0, wall-only file)", ok)

    # 6c. run_patient.sh --dry-run: production chain printed, XMLs generated, nothing run
    env = tmp + "/case.env"
    with open(env, "w") as f:
        f.write(f"PATIENT=SYN\nWORK={tmp}/work\nRUNROOT={tmp}/runs\nREF_STL={tmp}/ref.stl\n"
                f"ORIG_SURF={surf}\nPHASES=\"0.40:{tmp}/a.stl 0.60:{tmp}/b.stl\"\n"
                f"RCR_ARGS=\"--flow {tmp}/inlet.txt --map 90\"\n")
    xd = tmp + "/dry_xml"
    r = subprocess.run(["bash", "run_patient.sh", "--dry-run", env], cwd=ROOT, capture_output=True,
                       text=True, env=dict(os.environ, DRYRUN_XML_DIR=xd, PYTHON=sys.executable))
    out = r.stdout + r.stderr
    ok = (r.returncode == 0 and "--register-only" in out and "run_pinned_morph.py" in out
          and "--L 12" in out and "extended_to_standard.py" in out and "--face" in out
          and not os.path.exists(tmp + "/work") and not os.path.exists(tmp + "/runs"))
    if ok:
        import xml.etree.ElementTree as ET
        root = ET.parse(xd + "/SYN_MB.xml").getroot()
        ET.parse(xd + "/SYN_FB.xml")
        ok = (mpx.check_production_mb(root) == [] and
              root.find(".//Prescribed_displacement_file_path").text.strip() == tmp + "/work/wall_displacement.txt")
    check("run_patient.sh --dry-run (production chain, XMLs parse)", ok, out[-600:])

    # 7. extended_to_standard -> STANDARD wall-only file from an EXTENDED morph file
    import numpy as np
    import pyvista as pv
    P = np.asarray(g.points); n = len(P); gidv = np.arange(1, n + 1)
    ext = tmp + "/disp_ext.txt"
    with open(ext, "w") as f:
        f.write(f"{n} 3\nEXTENDED\n")
        np.savetxt(f, gidv, fmt="%d")
        for k, t in enumerate([0.0, 0.5, 1.0]):
            f.write(f"{t:.6f}\n")
            np.savetxt(f, np.column_stack([gidv, (P * (1 + 0.01 * (k % 2))) * 0.1]), fmt="%d %.6f %.6f %.6f")
    rc, out = run(["tools/extended_to_standard.py", ext, tmp + "/disp_wall.txt", "--face", wall])
    ok = rc == 0
    if ok:
        toks = open(tmp + "/disp_wall.txt").read().split()
        wg = np.asarray(pv.read(wall).point_data["GlobalNodeID"]).astype(int)
        nn, nt = int(toks[0]), int(toks[1])
        blk = np.asarray(toks[2:], float).reshape(nt, 1 + 4 * nn)
        ok = (nn == len(wg) and nt == 3 and set(blk[1, 1:].reshape(nn, 4)[:, 0].astype(int)) == set(wg)
              and np.allclose(blk[:, 0], [0, 0.5, 1]))
    check("extended_to_standard.py (EXTENDED -> wall-only STANDARD)", ok, out[-300:])

    npass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n=== post-processing: {npass}/{len(checks)} PASS ===")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
