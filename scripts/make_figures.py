"""Reproduce every figure and table of the README.

    python scripts/make_figures.py            # all (about 10 min on a laptop)
    python scripts/make_figures.py forward mesh tb inverse photonic

Outputs go to figures/ (png) and results/ (json, md).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

from strainscape import Grid, Pillar, run_forward, summarize, raman_convolve      # noqa: E402
from strainscape.calibration import fit_rounding, mesh_sweep, print_rows           # noqa: E402
from strainscape.constants import A0, BETA, HBAR_OVER_E                            # noqa: E402
from strainscape.pmf import uniform_field_from_triaxial                            # noqa: E402
from strainscape import tightbinding as tb                                         # noqa: E402
from strainscape import inverse as inv                                             # noqa: E402
from strainscape import photonic as ph                                             # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "figure.dpi": 150})

# ---------------------------------------------------------------- reference geometry
# Lu et al., Nat. Commun. 14, 2580 (2023): 500 nm square pillars, 85 nm tall, ~1 um apart,
# maximum local atomic strain 2.4 %, Raman-derived strain 1.26 %, |B_ps| up to ~30 T.
L_CELL = 1500e-9
EPS_TARGET = 0.024
BASE = Pillar(half_side=250e-9, power=8.0, height=85e-9, tent_width=488e-9)


def pillar_to_dict(p: Pillar) -> dict:
    d = asdict(p)
    d["harmonics"] = {str(k): v for k, v in p.harmonics.items()}
    return d


def calibrated():
    """Edge rounding fitted so that the peak strain matches the reported 2.4 %."""
    cache = RES / "calibration.json"
    if cache.exists():
        return json.loads(cache.read_text())["rounding_m"]
    t = time.perf_counter()
    r = fit_rounding(BASE, L_CELL, EPS_TARGET)
    cache.write_text(json.dumps({"rounding_m": r, "rounding_nm": r * 1e9, "target_eps": EPS_TARGET,
                                 "pillar": pillar_to_dict(BASE), "seconds": time.perf_counter() - t}, indent=2))
    return r


def _map(ax, grid, field, title, cmap="viridis", sym=False, unit=""):
    ext = np.array([-grid.L / 2, grid.L / 2, -grid.L / 2, grid.L / 2]) * 1e9
    if sym:
        v = np.abs(field).max()
        im = ax.imshow(field, extent=ext, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    else:
        im = ax.imshow(field, extent=ext, origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if unit:
        cb.set_label(unit)
    return im


# ======================================================================== forward
def fig_forward(N: int = 750):
    r = calibrated()
    grid = Grid(L_CELL, N)
    res = run_forward(BASE, grid, rounding=r)
    s = summarize(res)
    s["rounding_nm"] = r * 1e9
    (RES / "forward_summary.json").write_text(json.dumps(s, indent=2))
    eps = res.principal_strain
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.6))
    _map(ax[0, 0], grid, res.h * 1e9, "(a) graphene height h", unit="nm")
    _map(ax[0, 1], grid, 100 * eps, "(b) principal strain (atomic scale)", cmap="magma", unit="%")
    _map(ax[0, 2], grid, 100 * raman_convolve(eps, grid.dx),
         f"(c) strain seen by a 361 nm Raman spot (max {100 * raman_convolve(eps, grid.dx).max():.2f} %)",
         cmap="magma", unit="%")
    _map(ax[1, 0], grid, res.B, f"(d) pseudo-magnetic field B_ps (max {np.abs(res.B).max():.1f} T)", sym=True, unit="T")
    _map(ax[1, 1], grid, res.P_mag / A0, "(e) |P|/a0, sublattice polarisation (chi(2) proxy)", cmap="inferno")
    # diagonal cut through a corner
    n = grid.N
    d = np.arange(n)
    diag_x = (d - n / 2) * grid.dx * np.sqrt(2) * 1e9
    ax[1, 2].plot(diag_x, 100 * eps[d, d], color="C3", label="strain (%)")
    ax2 = ax[1, 2].twinx()
    ax2.plot(diag_x, res.B[d, d], color="C0", lw=0.9, label="B_ps (T)")
    ax[1, 2].set_xlim(0, 0.75 * grid.L * 1e9 * np.sqrt(2) / 1.0)
    ax[1, 2].set_xlabel("distance along the diagonal (nm)")
    ax[1, 2].set_ylabel("strain (%)", color="C3"); ax2.set_ylabel("B_ps (T)", color="C0")
    ax[1, 2].set_title("(f) cut through the pillar corner")
    fig.suptitle(f"Forward model, 500 nm x 85 nm pillar, 1.5 um pitch, edge rounding {r * 1e9:.0f} nm, "
                 f"dx = {grid.dx * 1e9:.1f} nm", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_forward.png")
    plt.close(fig)
    print("forward:", json.dumps(s, indent=1))


# ======================================================================== mesh calibration
def fig_mesh(dx_list=(16e-9, 8e-9, 4e-9, 2e-9)):
    r_cal = calibrated()
    roundings = [r_cal, r_cal / 2, r_cal / 4, 0.0]
    all_rows = {}
    for r in roundings:
        dxs = list(dx_list) if r > 0 else list(dx_list)[:-1]
        rows = mesh_sweep(BASE, L_CELL, dxs, rounding=r)
        all_rows[f"{r * 1e9:.1f}"] = rows
        print(f"--- rounding {r * 1e9:.1f} nm"); print_rows(rows)
    (RES / "mesh_sweep.json").write_text(json.dumps(all_rows, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for key, rows in all_rows.items():
        dx = np.array([q["dx_nm"] for q in rows])
        ax[0].loglog(dx, [q["B_max_T"] for q in rows], "o-", label=f"rounding {key} nm")
        ax[1].semilogx(dx, np.array([q["eps_max_pct"] for q in rows]), "o-", label=f"rounding {key} nm")
        ax[2].semilogx(dx, [q["eps_raman_max_pct"] for q in rows], "s--", color="0.4", lw=0.8, ms=3)
        ax[2].semilogx(dx, [q["B_rms_tent_T"] for q in rows], "o-", label=f"rms B_ps, r={key} nm")
    ax[0].set_xlabel("grid spacing dx (nm)"); ax[0].set_ylabel("peak |B_ps| (T)")
    ax[0].set_title("(a) peak field: converges only with a rounding length")
    ax[0].invert_xaxis(); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("dx (nm)"); ax[1].set_ylabel("peak strain (%)"); ax[1].invert_xaxis()
    ax[1].set_title("(b) peak strain vs mesh")
    ax[2].set_xlabel("dx (nm)"); ax[2].set_ylabel("T  |  %"); ax[2].invert_xaxis()
    ax[2].set_title("(c) integrated quantities converge on coarse meshes\n(dashed grey: Raman-convolved strain, %)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_mesh_calibration.png")
    plt.close(fig)

    # markdown table
    lines = ["| rounding (nm) | dx (nm) | N | peak strain (%) | Raman strain (%) | peak B (T) | rms B in tent (T) | time (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for key, rows in all_rows.items():
        for q in rows:
            lines.append(f"| {key} | {q['dx_nm']:.1f} | {q['N']} | {q['eps_max_pct']:.2f} | {q['eps_raman_max_pct']:.2f} | "
                         f"{q['B_max_T']:.1f} | {q['B_rms_tent_T']:.2f} | {q['time_s']:.1f} |")
    (RES / "mesh_sweep.md").write_text("\n".join(lines) + "\n")


# ======================================================================== tight binding
def fig_tb(radius: float = 10e-9, B_target: float = 150.0, k: int = 160):
    fl = tb.honeycomb_flake(radius)
    c = -B_target / (8 * HBAR_OVER_E * BETA / (2 * A0))
    assert abs(uniform_field_from_triaxial(c) - B_target) < 1e-9
    H = tb.hamiltonian(fl, tb.triaxial_displacement(fl.pos, c))
    H0 = tb.hamiltonian(fl)
    vals, vecs = tb.spectrum_near(H, k=k)
    vals0, vecs0 = tb.spectrum_near(H0, k=k)
    rr = np.hypot(fl.pos[:, 0], fl.pos[:, 1])
    interior = rr < 0.65 * radius
    w = np.sum(np.abs(vecs[interior]) ** 2, axis=0)
    bulk = w > 0.6
    E_ll = tb.landau_level_energies(B_target, 3)
    e = np.linspace(-0.8, 0.8, 800)
    eta = 0.012
    wA, wB = tb.sublattice_weight(vecs, vals, fl.sub, 0.1, interior)
    out = {"atoms": fl.n, "B_T": B_target, "c_per_m": c, "eps_edge_pct": 200 * abs(c) * radius,
           "l_B_nm": tb.magnetic_length(B_target) * 1e9, "E_LL_pred_eV": E_ll.tolist(),
           "bulk_eigenvalues_eV": vals[bulk].tolist(),
           "n0_weight_A": wA, "n0_weight_B": wB}
    # match each predicted level to the nearest bulk eigenvalue cluster
    err = []
    for En in E_ll[E_ll != 0]:
        near = vals[bulk][np.abs(vals[bulk] - En) < 0.08]
        if len(near):
            err.append((En, float(near.mean()), float(abs(near.mean() - En) / abs(En))))
    out["level_match"] = err
    (RES / "tightbinding.json").write_text(json.dumps(out, indent=1))
    print("TB:", json.dumps(out["level_match"]), "n=0 weights A/B:", wA, wB)

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6))
    ax[0].plot(e, tb.dos(vals0, e, eta) / fl.n, color="0.5", label="pristine flake")
    ax[0].plot(e, tb.dos(vals[bulk], e, eta) / fl.n, color="C3", label="strained, bulk states")
    for En in E_ll:
        ax[0].axvline(En, color="C0", lw=0.7, ls="--")
    ax[0].set_xlabel("energy (eV)"); ax[0].set_ylabel("DOS (arb.)")
    ax[0].set_title(f"(a) pseudo-Landau levels, B_ps = {B_target:.0f} T\n dashed: E_n = sgn(n) v_F sqrt(2 e hbar B |n|)")
    ax[0].legend(fontsize=7)
    sel = np.abs(vals) < 0.1
    w0 = np.sum(np.abs(vecs[:, sel]) ** 2, axis=1)
    for s_, col, name in ((0, "C2", "A"), (1, "C1", "B")):
        m = fl.sub == s_
        ax[1].scatter(fl.pos[m, 0] * 1e9, fl.pos[m, 1] * 1e9, s=6, c=col, alpha=np.clip(w0[m] / w0.max(), 0.02, 1), label=name)
    ax[1].set_aspect("equal"); ax[1].set_xlabel("x (nm)"); ax[1].set_ylabel("y (nm)")
    ax[1].set_title(f"(b) n = 0 level lives on one sublattice\n weight A : B = {wA:.2f} : {wB:.2f} (interior)")
    ax[1].legend(fontsize=7, markerscale=2)
    # bond-strain map for reference
    i, j = fl.pairs[:, 0], fl.pairs[:, 1]
    r = fl.pos + tb.triaxial_displacement(fl.pos, c)
    d = np.linalg.norm(r[i] - r[j], axis=1) / A0 - 1
    mid = 0.5 * (fl.pos[i] + fl.pos[j]) * 1e9
    sc = ax[2].scatter(mid[:, 0], mid[:, 1], c=100 * d, s=4, cmap="coolwarm", vmin=-100 * abs(d).max(), vmax=100 * abs(d).max())
    plt.colorbar(sc, ax=ax[2], label="bond strain (%)")
    ax[2].set_aspect("equal"); ax[2].set_title("(c) triaxial bond strain giving uniform B_ps")
    ax[2].set_xlabel("x (nm)"); ax[2].set_ylabel("y (nm)")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_tightbinding.png")
    plt.close(fig)


# ======================================================================== inverse design
def fig_inverse(N_opt: int = 188, N_verify: int = 750, rounding: float = 10e-9, eps_cap: float = 0.03,
                maxiter: int = 8):
    space = inv.DesignSpace(L=L_CELL, height=85e-9)
    grid = Grid(L_CELL, N_opt)
    x0 = (BASE.half_side, BASE.power, BASE.tent_width, 0.0, 0.0)
    t = time.perf_counter()
    opt = inv.optimize(space, x0, grid, rounding, objective=inv.objective_mean_field,
                       eps_cap=eps_cap, maxiter=maxiter, surrogate=False, method="Powell")
    secs = time.perf_counter() - t
    print(f"inverse: {len(opt.history)} evaluations in {secs:.1f} s; f0 = {opt.history[0][1]:+.4f}, f* = {opt.fun:+.4f}")
    res_i = inv.evaluate(opt.initial, grid, rounding, surrogate=False)
    res_o = inv.evaluate(opt.pillar, grid, rounding, surrogate=False)
    ver_i, vres_i = inv.verify(opt.initial, L_CELL, N_verify, rounding)
    ver_o, vres_o = inv.verify(opt.pillar, L_CELL, N_verify, rounding)
    out = {"rounding_nm": rounding * 1e9, "eps_cap": eps_cap, "evaluations": len(opt.history), "seconds": secs,
           "initial": pillar_to_dict(opt.initial), "optimized": pillar_to_dict(opt.pillar),
           "surrogate": {"initial": summarize(res_i), "optimized": summarize(res_o)},
           "laplace_verification": {"initial": ver_i, "optimized": ver_o},
           "history": [float(v) for _, v in opt.history]}
    (RES / "inverse_design.json").write_text(json.dumps(out, indent=1))

    fig, ax = plt.subplots(2, 3, figsize=(11.5, 6.8))
    for row, (name, rs, rv) in enumerate((("initial (500 nm square)", res_i, vres_i), ("optimized footprint", res_o, vres_o))):
        _map(ax[row, 0], grid, rs.h * 1e9, f"{name}: h  (design mesh, dx = {grid.dx * 1e9:.0f} nm)", unit="nm")
        ax[row, 0].contour(*[a * 1e9 for a in grid.mesh()], rs.pillar.footprint(grid), levels=[0.5], colors="w", linewidths=0.6)
        _map(ax[row, 1], grid, np.abs(rs.B), f"|B_ps| design mesh, <|B|> = {np.abs(rs.B).mean():.2f} T, "
             f"eps_max = {100 * rs.principal_strain.max():.1f} %", cmap="inferno", unit="T")
        _map(ax[row, 2], Grid(L_CELL, N_verify), np.abs(rv.B), f"|B_ps| verification, dx = 2 nm, <|B|> = {np.abs(rv.B).mean():.2f} T, "
             f"eps_max = {100 * rv.principal_strain.max():.1f} %", cmap="inferno", unit="T")
    fig.suptitle(f"Inverse design: maximise cell-averaged |B_ps| subject to strain <= {100 * eps_cap:.0f} %, "
                 f"H = 85 nm, rounding {rounding * 1e9:.0f} nm", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_inverse_design.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3))
    hist = -np.array(out["history"]) * 10.0
    ax.plot(np.maximum.accumulate(hist), "-", color="C3", label="best so far")
    ax.plot(hist, ".", ms=2.5, color="0.5", alpha=0.6, label="evaluations")
    ax.set_ylim(0, 1.15 * hist.max())
    ax.set_xlabel("objective evaluation"); ax.set_ylabel("<|B_ps|> - penalty  (T)")
    ax.set_title("optimiser history (Powell on the Laplace drape)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig4b_inverse_history.png"); plt.close(fig)
    print("inverse verification:", json.dumps(out["laplace_verification"], indent=1))
    print("optimized pillar:", out["optimized"])


# ======================================================================== photonic twin
def fig_photonic(d0: float = 650e-9, width: float = 450e-9, radius_sites: float = 12.0):
    gaps = np.array([100, 150, 200, 250, 300, 400]) * 1e-9
    dxs = [20e-9, 10e-9, 5e-9, 2.5e-9, 1.25e-9]
    table, ell, kappa_at = ph.calibrate_coupling(gaps, dxs, width=width)
    kappa0 = float(kappa_at(d0 - width))
    fl, H, beta_ph = ph.photonic_lattice(radius_sites, d0, ell, kappa0, c=0.0)
    # choose c so that the photonic magnetic length is ~ 2 sites
    l_B = 2.0 * d0
    Bph = 1.0 / l_B ** 2
    c = -Bph / (8 * beta_ph / (2 * d0))
    fl_s, Hs, _ = ph.photonic_lattice(radius_sites, d0, ell, kappa0, c=c)
    rr = np.hypot(fl.pos[:, 0], fl.pos[:, 1])
    vals0, vecs0 = np.linalg.eigh(H)
    vals, vecs = np.linalg.eigh(Hs)
    interior = rr < 0.6 * radius_sites * d0
    w = np.sum(np.abs(vecs[interior]) ** 2, axis=0)
    bulk = w > 0.6
    E_ll = ph.photonic_landau_levels(Bph, kappa0, d0, 3)
    # propagation from the central site
    centre = int(np.argmin(rr))
    psi0 = np.zeros(fl.n); psi0[centre] = 1.0
    z = np.linspace(0, 12 / kappa0, 300)
    I0 = ph.propagate(H, psi0, z)
    Is = ph.propagate(Hs, psi0, z)
    R0 = ph.spreading_radius(I0, fl.pos, fl.pos[centre])
    Rs = ph.spreading_radius(Is, fl.pos, fl.pos[centre])
    out = {"ell_nm": ell * 1e9, "kappa_fit_per_um": {f"{g * 1e9:.0f}": float(kappa_at(g)) * 1e-6 for g in gaps},
           "kappa_table_per_um": (table * 1e-6).tolist(), "dx_nm": [d * 1e9 for d in dxs],
           "d0_nm": d0 * 1e9, "kappa0_per_um": kappa0 * 1e-6, "beta_ph": beta_ph, "sites": fl.n,
           "B_ph_per_um2": Bph * 1e-12, "l_B_over_d0": l_B / d0, "E_LL_pred_per_um": (E_ll * 1e-6).tolist(),
           "bulk_eigs_per_um": (vals[bulk] * 1e-6).tolist(),
           "spreading_radius_final_over_d0": {"pristine": float(R0[-1] / d0), "strained": float(Rs[-1] / d0)},
           "spreading_radius_max_over_d0": {"pristine": float(R0.max() / d0), "strained": float(Rs.max() / d0)}}
    (RES / "photonic.json").write_text(json.dumps(out, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for k_, dx in enumerate(dxs):
        ax[0].semilogy(gaps * 1e9, table[:, k_] * 1e-6, "o-", ms=3, label=f"dx = {dx * 1e9:g} nm")
    gg = np.linspace(gaps[0], gaps[-1], 100)
    ax[0].semilogy(gg * 1e9, kappa_at(gg) * 1e-6, "k--", lw=0.8, label=f"fit, decay length {ell * 1e9:.0f} nm")
    ax[0].set_xlabel("gap (nm)"); ax[0].set_ylabel("coupling kappa (1/um)")
    ax[0].set_title("(a) coupling from a 1D FD mode solver:\nmesh convergence of the hopping law")
    ax[0].legend(fontsize=6.5)
    e = np.linspace(-1.2 * 3 * kappa0, 1.2 * 3 * kappa0, 600)
    eta = 0.04 * kappa0
    ax[1].plot(e / kappa0, tb.dos(vals0, e, eta), color="0.5", label="pristine array")
    ax[1].plot(e / kappa0, tb.dos(vals[bulk], e, eta), color="C3", label="strained array, bulk")
    for En in E_ll:
        ax[1].axvline(En / kappa0, color="C0", lw=0.7, ls="--")
    ax[1].set_xlabel("propagation constant detuning / kappa0"); ax[1].set_ylabel("DOS")
    ax[1].set_title(f"(b) photonic pseudo-Landau levels, {fl.n} waveguides\n l_B = 2 d0, beta_ph = d0/ell = {beta_ph:.1f}")
    ax[1].legend(fontsize=7)
    ax[2].plot(z * kappa0, R0 / d0, color="0.5", label="pristine array")
    ax[2].plot(z * kappa0, Rs / d0, color="C3", label="strained array")
    ax[2].axhline(l_B / d0, color="C0", ls="--", lw=0.8, label="magnetic length l_B")
    ax[2].axhline(radius_sites, color="k", ls=":", lw=0.8, label="array edge")
    ax[2].set_xlabel("kappa0 z"); ax[2].set_ylabel("rms spreading radius / d0")
    ax[2].set_title("(c) walk exp(-iHz) launched at the centre:\nthe pseudo-field halts the ballistic spreading")
    ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig5_photonic_twin.png"); plt.close(fig)
    print("photonic:", json.dumps({k: v for k, v in out.items() if k not in ("kappa_table_per_um", "bulk_eigs_per_um")}, indent=1))


ALL = {"forward": fig_forward, "mesh": fig_mesh, "tb": fig_tb, "inverse": fig_inverse, "photonic": fig_photonic}

if __name__ == "__main__":
    which = sys.argv[1:] or list(ALL)
    for name in which:
        t = time.perf_counter()
        print(f"=== {name} ===")
        ALL[name]()
        print(f"=== {name} done in {time.perf_counter() - t:.1f} s")
