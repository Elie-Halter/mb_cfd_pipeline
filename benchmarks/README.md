# benchmarks/ — synthetic analytic verification suite

Three benchmarks with analytic solutions that exercise the patched solver on synthetic
geometries. They need **no patient data**: every mesh, motion, boundary condition and initial
condition is generated from the parameters in each `params.py`. They are the verification
benchmarks of Supplementary Section S1 (Table S3) of the companion paper.

| Case | Verifies | Analytic solution | Pass criterion | Reference result (patched) |
|---|---|---|---|---|
| **B1** free stream | free-stream preservation on a moving mesh | u = (U,0,0), uniform p, for any mesh motion | max\|u−U\|/U < 1e−4 | 1.69e−6 (also 1.69e−6 without patch 0014: not discriminating) |
| **B2** expanding tube | moving-wall no-slip, motion-driven flow, **discrete mass balance (DGCL)** | R(t) = R0 s(t), ΣQ_caps = −dV/dt = −2πRL dR/dt, u_r(R) = dR/dt | no-slip < 1e−4 cm/s; flow < 2 %; mass < 2 % | 2.68e−6 cm/s; 1.64 %; **3.3e−6** (6.66e−2 without 0014: 4.3 orders) |
| **B2t** (option `--full`) | temporal order of the discrete mass residual | — | observed p ≥ 2 | p = 2.88 (dt 5 → 2.5 ms), then 3.85 |
| **B3** Womersley | unsteady velocity at α ≈ 12.9 (aortic) | Womersley u_z(r,t), Q(t) | flow amplitude ±5 %, phase < 5° | 0.978, +2.6° (grid g1; 1.015, +4.3° on g2) |
| **B3r** (option `--full`) | profile error under refinement | — | core L2 error decreases monotonically | 28.2 % → 15.7 % → 11.3 % (g1 → g2 → g3) |

All reference values were obtained with 4 MPI ranks and the solver built by `install.sh`
(svMultiPhysics `97ef512` + the fifteen patches in `patches/svMP/`). Iterative-solver values such
as B1's 1.69e−6 can differ in the last digits with another rank count or platform; the pass
criteria leave orders of magnitude of margin. The analysis scripts in this directory were run
on the archived reference runs and reproduce every value quoted here; `gen_case.py` regenerates
the reference B1 and B2 meshes node for node (gmsh 4.15).

## Run

```bash
bash benchmarks/run_all.sh --no-solver     # generate all cases + every pre-flight check (minutes, no solver)
bash benchmarks/run_all.sh                 # B1, B2 (dt = 5 ms), B3 (grid g1): ~1.5-3 h on 4 cores
bash benchmarks/run_all.sh --full          # + B2 at dt/2, dt/4 and B3 on grids g2, g3 (many hours)
bash benchmarks/run_all.sh --cases "B2" --np 8
```

The solver is `$SVMP_BIN` (default `~/svmp_bin/svmultiphysics`, where `install.sh` puts it).
Cases and runs go to `benchmarks/_out/` (git-ignored; `--out DIR` to change). Runs are
sequential, and the script refuses to start the solver while another `svmultiphysics` process
is running. It ends with a PASS/FAIL summary and exits non-zero if any check fails.

### DGCL contrast (patch 0014 on/off)

```bash
bash benchmarks/build_nogcl.sh                     # same commit, all patches except 0014
SVMP_BIN_NOGCL=~/svmp_bin/svmultiphysics_nogcl bash benchmarks/run_all.sh --cases "B1 B2"
```

With `SVMP_BIN_NOGCL` set, B1 and B2 are also run with the contrast binary and a final table
reports the contrast. Expected: B1 identical for both binaries (1.69e−6); B2 mass balance
3.3e−6 (patched) versus 6.66e−2 (without 0014), and induced flow 1.64 % versus 7.36 %;
no-slip and wall-velocity amplitude unchanged.

### Per-case scripts

Each case directory holds the same set of scripts, which `run_all.sh` chains; they can be run
by hand:

| Script | Role |
|---|---|
| `params.py` | single source of all parameters (geometry, motion, time step, pass criteria) |
| `gen_case.py --out DIR` | gmsh all-tet mesh in the pipeline format (mm, GlobalNodeID/GlobalElementID/ModelFaceID), displacement / boundary / initial-condition files |
| `case.xml.in` | solver input template; `@CASE@` and the run parameters are filled in by `common/render_xml.py` |
| `preflight.py --case DIR [--xml case.xml]` | checks without the solver: XML paths, mesh validity, face coverage, displacement file (t=0 position = reference), BC files, IC |
| `analyze.py RUN_DIR ...` | verdicts from `results_*.vtu`, CSV of per-snapshot metrics in the run directory |

`common/benchtools.py` holds the mesh/file writers and the P1-exact flux, volume and
divergence integrals; `common/preflight_lib.py` the shared pre-flight checks.

## The benchmarks

### B1 — free-stream preservation (`B1_freestream/`)

Box [0,20]×[−5,5]×[−5,5] mm, uniform tets h ≈ 1.2 mm (6,320 tets). The mesh is dilated affinely about
(10,0,0) mm, s(t) = 1 + 0.04(1 − cos 2πt), T = 1 s (~26 % volume swing, zero initial mesh
velocity), prescribed on all six faces; the interior follows the mesh equation, the same code
path as a production run. u = (10,0,0) cm/s with uniform pressure is an exact solution of the
ALE Navier–Stokes equations for any mesh motion: Dirichlet (General, full vector,
`Zero_out_perimeter false`) on five faces, a resistance outlet on x1, exact IC. 200 steps of 5 ms.

The free stream is preserved to the floor of the iterative linear solver (~1e−7 per step,
1.69e−6 worst case), not to machine precision: tightening the tolerances makes the FSILS
Navier–Stokes solve singular on the pressure mode under mesh motion. The binary without patch
0014 gives the same value — for a uniform field every term of the nodal ALE residual vanishes
whatever the integration geometry — so B1 is a necessary test that does not discriminate the
DGCL fix.

### B2 — analytically expanding tube (`B2_expanding_tube/`)

Cylinder R0 = 10 mm, L = 50 mm, tets h ≈ 1.5 mm (22,251 tets). Wall and caps are prescribed:
(x,y) → s(t)(x,y), s = 1 + 0.05(1 − cos 2πt), so R goes from 10 to 11 mm (~21 % volume swing)
and the deformed domain is an exact cylinder. Moving no-slip on the wall
(`Prescribed_displacement` + `Impose_on_state_variable_integral`, as in production), p = 0 on
both caps, fluid at rest initially, 1.5 periods at dt = 5 ms. Verdicts on t ≥ 0.5 s:

- (a) no-slip: fluid velocity at the wall against the wall velocity actually imposed (slope of
  the piecewise-linear displacement file), second cycle — 2.68e−6 cm/s;
- (b) induced flow: max\|ΣQ_caps + dV/dt\| / max\|dV/dt\| — 1.64 %, dominated by the polygonal
  volume deficit of the faceted cylinder (spatial; invariant with dt);
- (c) discrete mass balance through the whole closed boundary,
  max\|ΣQ_caps + Q_wall\| / max\|dV/dt\| — **3.3e−6** with patch 0014, **6.66e−2** without:
  the discriminating DGCL result;
- (d) amplitude/phase of the mean radial wall velocity against R0 ds/dt — 1.000, +0.6°.

`temporal_order.py` (run by `--full`) repeats the run at dt/2 and dt/4 with the same motion:
the mass residual falls as 3.31e−6 → 4.49e−7 → 3.12e−8 (observed order 2.88, then 3.85), and
is unchanged (3.306e−6) when the fluid tolerances are tightened ×50 — a discretisation error,
not an iterative floor. The wall/cap rings must be prescribed from both sides, which requires
patch 0015.

### B3 — Womersley flow (`B3_womersley/`)

Rigid tube R = 10 mm, L = 60 mm, ρ = 1.06 g/cm³, μ = 0.04 P, T = 1 s, α = R√(ω/ν) ≈ 12.9.
p_in = 1333.2 sin(2πt) dyn/cm² (unsteady Neumann), p_out = 0, no-slip. The convective term
vanishes identically, so the Womersley solution is exact on the finite tube. The analytic IC
removes the ~14-cycle start-up transient; 3 cycles of 5 ms steps, analysis on the last one.
`womersley.py` holds the closed form and a self-test (no-slip, ODE residual by finite
differences, Poiseuille limit, flow rate = integral of the profile) that runs without the solver.

Radially graded grids, h(r) = h_core − (h_core − h_wall)(r/R)³:

| Grid | h_wall / h_core (mm) | ~tets | nodes | amplitude | phase | slab L2 | core L2 (cycle 2) |
|---|---|---|---|---|---|---|---|
| g1 | 0.90 / 2.2 | 38k | 9,322 | 0.978 | +2.6° | 27.5 % | 28.2 % |
| g2 | 0.55 / 1.6 | 122k | 27,634 | 1.015 | +4.3° | 13.9 % | 15.7 % |
| g3 | 0.35 / 1.0 | 431k | 88,289 | 1.025 | +4.7° | 8.9 % | 11.3 % |

(The g3 reference run covered two cycles only; its amplitude, phase and slab error are from
cycle 2.) The flow-rate amplitude and phase pass on every grid. The pointwise profile error decreases
monotonically under refinement but stays well above a few percent: the Stokes layer
(δ ≈ 1.1 mm) is thinner than the isotropic element on all three grids. The observed spatial
order is 1.61 then 0.86 (global 1.22): a formal order of accuracy is therefore **not** claimed
— it is a near-wall resolution limit of isotropic P1 grids, the same reason wall-shear
quantities are not reported as converged.

## What is not here

A manufactured-solution order study was not completed: imposing the velocity on every face
leaves the pressure mode unfixed (singular FSILS solve), and the solver's traction boundary
condition is spatial only, so a consistent manufactured pressure cannot be imposed without
modifying the solver. The meshes and results are not distributed: `gen_case.py` regenerates the
former in seconds to minutes, and `run_all.sh` the latter.
