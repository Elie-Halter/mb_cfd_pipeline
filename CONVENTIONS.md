# Conventions and checks

Four solver conventions that produce plausible-looking but wrong results if you get them wrong, and
the checks that tell you a run is sound. Read this once before your first run.

## 1. Inlet flow sign

svMultiPhysics imposes a Dirichlet flow rate as `v = Q(t) * profile * n_out`
(`set_bc.cpp::set_bc_dir_l`, outward nodal normal), so **a positive Q is an outflow**. An inlet
waveform must be **negative** — all the bundled `tests/cases/fluid/*/lumen_inlet.flow` are.

MRI waveforms are usually reported positive in the antegrade direction. Copied as-is, they run the
aorta backwards: blood enters through the descending aorta and the branches, while the flow split,
the |Q| waveforms and the correlation with MRI all still look right. Tell-tale signs are negative
outlet pressures, an apparent "pressure gauge offset", and a mass balance that only closes with a
flipped sign.

`run_patient.sh` builds the file with `tools/make_inlet_flow.py` and refuses to start if
`tools/check_inlet_sign.py` finds a positive mean inflow. Verify on results with
`tools/closed_flux.py`: the inlet flux must be negative at systole.

## 2. Period of temporal files

The solver's Fourier fit takes the period to be `t_last - t_first` of the file (`fft.cpp`). A
30-point file ending at 0.616 s therefore imposes T = 0.615 s, not the cycle you meant.
`make_inlet_flow.py` resamples the waveform at the solver time step over the whole run, so the period
is exact.

## 3. Fluid boundary condition on a moving wall

On a prescribed-motion wall the fluid `wall` BC must be `Prescribed_displacement` with
`Impose_on_state_variable_integral` (fluid velocity = wall velocity), **not** `Dirichlet 0`. With
`Dirichlet 0` the wall moves without displacing any fluid: you get `Q_in = sum Q_out` despite
`dV/dt != 0`, and any FB-versus-MB difference measured that way is worthless. `MB_example.xml` does
this correctly.

## 4. Cap pinning and taper

Caps are pinned (`Dir 0` in the mesh equation), and the prescribed wall displacement is tapered to
zero over a geodesic distance of 12 mm from the cap rings (`morph/pin_caps.py`). Without the taper,
the elements between a moving ring and a fixed cap invert about 20 ms into the cycle.

Cap `.vtp` files from gmsh or TetGen may be wound inconsistently. The flux tools orient the outward
normals geometrically (`tools/closed_flux.py`) or by cycle mean (`tools/mass_balance*.py`).

## 5. Checks that a run is sound

- **Valid mesh motion** — the morph advances the mesh through the full cycle with a positive Jacobian
  everywhere, no remeshing and no interpolation. Check the per-step minimum scaled Jacobian and the
  inverted-element count.
- **Mass conservation** — the instantaneous closed-surface flux sum (`tools/closed_flux.py`, P1-exact
  on the deformed geometry) is near zero at every saved step: below 0.01% of the peak inflow on our
  runs. The cycle-mean inflow equals the outflow. This check also selects the mesh.
- **Flow split** — within a few percent per outlet of the 4D-flow MRI target, with the outlet sum
  consistent with the prescribed inlet.
- **Wall hemodynamics** — `tools/hemo_indices.py` writes a wall VTP with TAWSS, OSI and helicity;
  `tools/compare_FB_MB.py` writes the FB, MB and difference maps plus summary statistics;
  `tools/gci.py` reports the grid-convergence index across three meshes.
