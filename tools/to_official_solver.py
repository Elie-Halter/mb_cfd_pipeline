#!/usr/bin/env python3
"""to_official_solver.py -- turn a moving-wall input written in the pipeline's authoring format into the input of
the OFFICIAL, unmodified svMultiPhysics (tested on commit 86729244).

The pipeline writes its moving-wall XML (MB_example.xml, tools/make_patient_xml.py, benchmarks/*/case.xml.in) with a
`Prescribed_displacement` boundary condition and one displacement file giving node POSITIONS at each time. The official
solver moves a wall through its FSI equation with a fluid domain only plus the mesh equation; this script rewrites:
  - <Add_equation type="fluid"> -> type="FSI" with <Domain id="0"> (fluid; density, viscosity, backflow), and
    <Domain> 0 </Domain> in Add_mesh;
  - every Prescribed_displacement BC (fluid and mesh) -> Dir + General + Temporal_and_spatial_values_file_path,
    one file per face, d = position - X * Mesh_scale_factor, same times; Impose_on_state_variable_integral is kept;
  - <Mesh_stiffness_exponent> (not read by the official solver) is removed.
Checked on the expanding-tube benchmark (same nodes, order and times, values within 5e-9 cm) and on the patient runs.
Usage: python3 to_official_solver.py IN.xml OUT.xml OUT_DIR_FOR_DISPLACEMENT_FILES
"""
import os, re, sys, numpy as np, pyvista as pv

src_xml, out_xml, ddir = sys.argv[1:4]; os.makedirs(ddir, exist_ok=True)
x = open(src_xml).read()
tag = lambda name, s=x: re.search(rf"<{name}>\s*(.*?)\s*</{name}>", s).group(1)
mesh_path = tag("Mesh_file_path"); scale = float(tag("Mesh_scale_factor"))
faces = dict(re.findall(r'<Add_face name="([^"]+)">\s*<Face_file_path>\s*(.*?)\s*</Face_file_path>', x))
disp_paths = set(re.findall(r"<Prescribed_displacement_file_path>\s*(.*?)\s*</Prescribed_displacement_file_path>", x))
assert len(disp_paths) == 1, disp_paths
fork_disp = disp_paths.pop()
mesh = pv.read(mesh_path); X = np.asarray(mesh.points, float) * scale
gidm = np.asarray(mesh.point_data["GlobalNodeID"]).astype(int)
assert np.array_equal(gidm, np.arange(1, len(gidm) + 1)), "GlobalNodeID of the mesh not contiguous"
with open(fork_disp, "rb") as f:
    hdr = f.readline().split(); assert len(hdr) == 2, hdr
    n, nt = map(int, hdr); raw = np.fromfile(f, dtype=float, sep=" ")
assert raw.size == nt * (1 + 4 * n), raw.size
blk = raw.reshape(nt, 1 + 4 * n); ts = blk[:, 0].copy(); D = blk[:, 1:].reshape(nt, n, 4)
gid = D[0, :, 0].astype(int)
assert all(np.array_equal(D[k, :, 0].astype(int), gid) for k in range(nt)), "node order changes between times"
assert abs(ts[0]) < 1e-12 and np.all(np.diff(ts) > 0)
order = {g: i for i, g in enumerate(gid)}

# prescribed faces (union of fluid and mesh equations)
pfaces = []
for m in re.finditer(r'<Add_BC name="([^"]+)">(.*?)</Add_BC>', x, re.S):
    if "Prescribed_displacement" in m.group(2) and m.group(1) not in pfaces: pfaces.append(m.group(1))
out = {}
for fn in pfaces:
    gw = np.asarray(pv.read(faces[fn]).point_data["GlobalNodeID"]).astype(int)
    idx = np.array([order[g] for g in gw])        # KeyError if a face node is missing from the displacement file
    d = D[:, idx, 1:] - X[gw - 1][None, :, :]
    p = os.path.join(os.path.abspath(ddir), f"{fn}_disp.txt")
    with open(p, "w") as fo:
        fo.write(f"3 {nt} {len(gw)}\n" + "\n".join(f"{t:.8f}" for t in ts) + "\n")
        for j, g in enumerate(gw):
            fo.write(f"{g}\n"); np.savetxt(fo, d[:, j, :], fmt="%.8e")
    out[fn] = p
    print(f"{fn}: {len(gw)} nodes x {nt} times ; |d(t=0)| max {np.abs(d[0]).max():.2e} ; |d| max {np.abs(d).max():.4f} cm -> {p}")

def conv(m):
    name, body = m.group(1), m.group(2)
    if "Prescribed_displacement" not in body: return m.group(0)
    impd = "Impose_on_state_variable_integral" in body
    return (f'<Add_BC name="{name}">\n    <Type> Dir </Type>\n    <Time_dependence> General </Time_dependence>\n'
            f'    <Temporal_and_spatial_values_file_path> {out[name]} </Temporal_and_spatial_values_file_path>\n'
            + ('    <Impose_on_state_variable_integral> true </Impose_on_state_variable_integral>\n' if impd else '') + '  </Add_BC>')
x = re.sub(r'<Add_BC name="([^"]+)">(.*?)</Add_BC>', conv, x, flags=re.S)
x = re.sub(r"[ \t]*<Mesh_stiffness_exponent>.*?</Mesh_stiffness_exponent>[^\n]*\n", "", x)
x = re.sub(r'(<Add_mesh name="[^"]+">)', r'\1\n  <Domain> 0 </Domain>', x, count=1)
f0 = x.index('<Add_equation type="fluid">'); fend = x.index('<LS type="NS">', f0)
lines = x[f0:fend].splitlines()
isprop = lambda l: any(k in l for k in ("<Density>", "<Viscosity", "<Value>", "</Viscosity>"))
props = [l.strip() for l in lines if isprop(l)]; bf = [l.strip() for l in lines if "Backflow" in l]
assert len(props) == 4 and len(bf) == 1, (props, bf)
keep = [l for l in lines if not isprop(l) and "Backflow" not in l]
head = "\n".join(keep).replace('<Add_equation type="fluid">', '<Add_equation type="FSI">').rstrip()
head += '\n\n  <Domain id="0">\n    <Equation> fluid </Equation>\n    ' + "\n    ".join(props) + "\n    " + bf[0] + "\n  </Domain>\n  "
x = x[:f0] + head + x[fend:]
x = x.replace('<svMultiPhysicsFile version="1.0">', '<svMultiPhysicsFile version="1.0">\n<!-- official svMultiPhysics input, '
              f'written by tools/to_official_solver.py from {os.path.basename(src_xml)} -->', 1)
xc = re.sub(r"<!--.*?-->", "", x, flags=re.S)
assert "Prescribed_displacement" not in xc and "Mesh_stiffness_exponent" not in xc and xc.count('type="FSI"') == 1, "authoring-format tag left"
open(out_xml, "w").write(x); print("wrote", out_xml)
