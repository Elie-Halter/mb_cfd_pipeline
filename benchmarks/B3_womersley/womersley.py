#!/usr/bin/env python3
"""
Analytic Womersley solution for p_in = A sin(wt), p_out = 0 on a straight rigid tube (R, L).

p(z,t) = A sin(wt) (1 - z/L)  ->  dp/dz = -(A/L) sin(wt) = Re{G e^{iwt}},  G = iA/L.
u_z(r,t) = Re{ (iG/(rho w)) [1 - J0(Lam r/R)/J0(Lam)] e^{iwt} },  Lam = i^{3/2} alpha,
alpha = R sqrt(w/nu).
Q(t)     = Re{ (iG/(rho w)) pi R^2 [1 - 2 J1(Lam)/(Lam J0(Lam))] e^{iwt} }.

This is an EXACT solution of the incompressible Navier-Stokes equations on the finite tube
with a uniform normal traction on each cap (the convective term vanishes identically).
Self-test (no solver): python3 womersley.py  -- no-slip, ODE residual, Poiseuille limit.
"""
import os
import sys
import numpy as np
from scipy.special import jv


class Womersley:
    def __init__(self, R, L, rho, nu, omega, A):
        self.R, self.L, self.rho, self.nu, self.omega, self.A = R, L, rho, nu, omega, A
        self.G = 1j * A / L
        self.alpha = R * np.sqrt(omega / nu)
        self.Lam = 1j ** 1.5 * self.alpha
        self.core = 1j * self.G / (rho * omega)

    def uhat(self, r):
        r = np.asarray(r, dtype=float)
        return self.core * (1.0 - jv(0, self.Lam * r / self.R) / jv(0, self.Lam))

    def u(self, r, t):
        """u_z(r,t) in cm/s (r in cm, t in s)."""
        return np.real(self.uhat(r) * np.exp(1j * self.omega * t))

    def dudt(self, r, t):
        return np.real(1j * self.omega * self.uhat(r) * np.exp(1j * self.omega * t))

    def Qhat(self):
        return self.core * np.pi * self.R ** 2 * \
            (1.0 - 2.0 * jv(1, self.Lam) / (self.Lam * jv(0, self.Lam)))

    def Q(self, t):
        """Flow rate (cm3/s) through a cross-section, positive towards +z."""
        return np.real(self.Qhat() * np.exp(1j * self.omega * np.asarray(t, dtype=float)))

    def p(self, z, t):
        return self.A * np.sin(self.omega * t) * (1.0 - z / self.L)


def selftest(verbose=True):
    """Checks the closed form against its defining ODE; returns a dict of the measured errors."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import params as pp
    w = Womersley(pp.R0_CM, pp.LT_CM, pp.RHO, pp.NU, pp.OMEGA, pp.P_AMP)
    out = {"alpha": w.alpha}
    assert 10.0 < w.alpha < 15.0, w.alpha
    # 1) exact no-slip
    out["noslip"] = abs(w.uhat(w.R)) / abs(w.core)
    assert out["noslip"] < 1e-12
    # 2) residual of  i rho w u + G - mu (u'' + u'/r) = 0  (finite differences)
    r = np.linspace(1e-4, w.R * 0.999, 20000)
    h = r[1] - r[0]
    u = w.uhat(r)
    upp = (u[2:] - 2 * u[1:-1] + u[:-2]) / h ** 2
    up = (u[2:] - u[:-2]) / (2 * h)
    res = 1j * pp.RHO * pp.OMEGA * u[1:-1] + w.G - pp.MU * (upp + up / r[1:-1])
    out["ode_residual"] = np.abs(res).max() / abs(w.G)
    assert out["ode_residual"] < 1e-4
    # 3) quasi-static limit (alpha -> 0): |Q| -> Poiseuille pi R^4 A / (8 mu L)
    w0 = Womersley(pp.R0_CM, pp.LT_CM, pp.RHO, pp.NU, pp.OMEGA / 1e4, pp.P_AMP)
    qp = np.pi * pp.R0_CM ** 4 * pp.P_AMP / (8 * pp.MU * pp.LT_CM)
    out["poiseuille"] = abs(abs(w0.Qhat()) - qp) / qp
    assert out["poiseuille"] < 1e-3
    # 4) flow rate = integral of the profile
    rr = np.linspace(0, w.R, 20001)
    qint = np.trapezoid(2 * np.pi * rr * w.uhat(rr), rr) if hasattr(np, "trapezoid") \
        else np.trapz(2 * np.pi * rr * w.uhat(rr), rr)
    out["flow_integral"] = abs(qint - w.Qhat()) / abs(w.Qhat())
    assert out["flow_integral"] < 1e-6
    if verbose:
        umax = np.abs(w.uhat(np.linspace(0, w.R, 400))).max()
        print(f"[womersley] alpha = {w.alpha:.3f}")
        print(f"[womersley] |u(R)|/|u_core| = {out['noslip']:.1e} (no-slip)")
        print(f"[womersley] max relative ODE residual = {out['ode_residual']:.2e}")
        print(f"[womersley] Poiseuille limit error = {out['poiseuille']:.2e}")
        print(f"[womersley] Q vs integral of u = {out['flow_integral']:.2e}")
        print(f"[womersley] max |u| = {umax:.2f} cm/s, max |Q| = {abs(w.Qhat()):.2f} cm3/s")
        print("[womersley] SELF-TEST OK")
    return out


if __name__ == "__main__":
    selftest()
