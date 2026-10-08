# -*- coding: utf-8 -*-
"""
PDE homework -- numerical experiments (numpy only; writes .dat tables for pgfplots).

A. 1-D lattice random walk -> diffusion equation (moments / distribution / convergence order)
   A+ genuine h,tau -> 0 continuum limit, plus the finite-step biased-walk variance correction
C. Fisher-KPP travelling wave: measured speed vs 2*sqrt(D*rho)
D. Tumour reaction-diffusion model (spherically symmetric): nutrient-limited growth,
   analytic quasi-steady nutrient profile, chemotaxis term, necrotic core,
   clipping statistics, long-time speed trend, dt- and grid-convergence
B. 2-D axisymmetric Fisher front (angular symmetry check)

All comments are ASCII to avoid any encoding trouble.
"""
import math
import numpy as np
import os

rng = np.random.default_rng(20240607)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figs", "data")
os.makedirs(OUT, exist_ok=True)

log = []


def note(s):
    print("  *", s)
    log.append(s)


def w(name, header, arr):
    arr = np.asarray(arr)
    np.savetxt(os.path.join(OUT, name), arr, header=header, comments="", fmt="%.8g")
    print(f"  [write] {name:28s} shape={arr.shape}")


# =====================================================================
# A. 1-D lattice random walk
# =====================================================================
def partA():
    print("[A] 1-D random walk -> diffusion equation")
    h, tau, M = 1.0, 1.0, 200000

    # A1: unbiased walk (p=q=1/2): Var(X_N) = N h^2 -> D = h^2/(2 tau)
    Ns = [10, 40, 90, 160, 250, 360, 490, 640]
    rows = []
    for N in Ns:
        X = (rng.integers(0, 2, size=(M, N)) * 2 - 1).sum(axis=1) * h
        t = N * tau
        rows.append([t, X.mean(), (X ** 2).mean(), X.var(), X.var() / (2 * t)])
    rows = np.array(rows)
    w("A1_rw_moments.dat", "t mean second_moment variance D_estimate", rows)
    note(f"A1: unbiased walk, N=640, M=200k: Var(X)={rows[-1,3]:.1f} (theory N*h^2=640), "
         f"D_est={rows[-1,4]:.4f} (theory h^2/(2tau)=0.5); mean={rows[-1,1]:.3f} ~ 0")

    # A2: distribution vs Gaussian fundamental solution
    N = 400
    X = (rng.integers(0, 2, size=(M, N)) * 2 - 1).sum(axis=1) * h
    t, D = N * tau, 0.5
    xs = np.arange(-6 * np.sqrt(2 * D * t), 6 * np.sqrt(2 * D * t) + 1, 2)
    cnt, _ = np.histogram(X, bins=np.append(xs - 1, xs[-1] + 1))
    P = cnt / (M * 2.0)
    Pg = np.exp(-xs ** 2 / (4 * D * t)) / np.sqrt(4 * np.pi * D * t)
    keep = np.zeros(len(xs), dtype=bool)
    keep[::3] = True                              # thin the table for plotting
    w("A2_rw_gauss.dat", "x P_mc P_gauss", np.column_stack([xs[keep], P[keep], Pg[keep]]))
    note(f"A2: N=400, M=200k: max|P_mc-P_gauss|={np.abs(P-Pg).max():.3e}, "
         f"relative L2 deviation={np.linalg.norm(P-Pg)/np.linalg.norm(Pg):.4f}")

    # A3: biased walk p=(1+beta)/2 -> drift v=(p-q)h/tau, same D
    beta = 0.10
    p = (1 + beta) / 2
    rows = []
    M2 = M // 4
    for N in [50, 200, 800]:
        X = np.where(rng.random((M2, N)) < p, 1.0, -1.0).sum(axis=1) * h
        t = N * tau
        rows.append([t, X.mean() / t, (2 * p - 1) * h / tau, X.var() / (2 * t), 0.5])
    rows = np.array(rows)
    w("A3_rw_drift.dat", "t v_measured v_theory D_measured D_theory", rows)
    note(f"A3: biased walk p={p:.3f}: v_num={rows[-1,1]:.5f} vs v_theory={(2*p-1)*h/tau:.5f}; "
         f"D_num={rows[-1,3]:.4f} vs 0.5 (drift does not change the 2nd moment)")

    # A4: master equation solved explicitly -> heat equation; relative L2 error vs order
    L, T, D, sig = 1.0, 0.005, 1.0, 0.02
    rows = []
    for n in [100, 200, 400, 800]:
        x = np.linspace(0, L, n)
        dx = L / (n - 1)
        u = np.exp(-((x - 0.5) ** 2) / (2 * sig ** 2))
        u[0] = u[-1] = 0.0
        dt = 0.4 * dx * dx / D
        nt = max(int(round(T / dt)), 1); dt = T / nt
        lam = D * dt / (dx * dx)
        for _ in range(nt):
            u[1:-1] = u[1:-1] + lam * (u[2:] - 2 * u[1:-1] + u[:-2])
        uex = np.zeros_like(x)
        for k in range(-6, 7):
            uex += np.exp(-((x - 0.5 - 2 * k * L) ** 2) / (4 * D * T + 2 * sig ** 2)) \
                   * sig / np.sqrt(2 * D * T + sig ** 2)
            uex -= np.exp(-((x + 0.5 - 2 * k * L) ** 2) / (4 * D * T + 2 * sig ** 2)) \
                   * sig / np.sqrt(2 * D * T + sig ** 2)
        err = np.linalg.norm(u - uex) / np.linalg.norm(uex)
        rows.append([dx, err])
    rows = np.array(rows)
    order = np.polyfit(np.log(rows[:, 0]), np.log(rows[:, 1]), 1)[0]
    w("A4_heat_order.dat", "dx rel_L2_error", rows)
    note(f"A4: master equation by explicit finite differences (Dirichlet), "
         f"relative L2 error order = {order:.2f} (theory 2)")


def partA_continuum():
    """Genuine h,tau -> 0 continuum limit with fixed macroscopic time T and diffusivity D.

    h_m = 2^-m,  tau_m = h_m^2/(2D),  N_m = T/tau_m.  We measure the Kolmogorov-Smirnov
    distance between the (rescaled) empirical distribution and the standard normal law.
    """
    print("[A+] random walk: genuine continuum limit h, tau -> 0")
    D, T, M = 0.5, 1.0, 200000
    rows = []
    dist_out = None
    erfv = np.vectorize(math.erf)
    for m in [1, 2, 3, 4, 5, 6]:
        h = 2.0 ** (-m)
        tau = h * h / (2 * D)
        N = int(round(T / tau))
        steps = np.where(rng.random((M, N)) < 0.5, h, -h).sum(axis=1)
        z = np.sort(steps / np.sqrt(2 * D * T))
        F = 0.5 * (1.0 + erfv(z / np.sqrt(2.0)))
        i = np.arange(1, M + 1)
        d_ks = max(np.abs(F - i / M).max(), np.abs(F - (i - 1) / M).max())
        rows.append([h, N, d_ks])
        if m == 6:
            xs = np.linspace(-4, 4, 161)
            cnt, edges = np.histogram(steps / np.sqrt(2 * D * T), bins=xs)
            xc = 0.5 * (edges[:-1] + edges[1:])
            dens = cnt / (M * (edges[1] - edges[0]))
            theo = np.exp(-xc ** 2 / 2) / np.sqrt(2 * np.pi)
            dist_out = (xc, dens, theo)
    rows = np.array(rows)
    w("A5_continuum_limit.dat", "h N_steps KS_distance", rows)
    p = np.polyfit(np.log(rows[:, 0]), np.log(rows[:, 2]), 1)[0]
    note("A5: fixed T=1, D=0.5, M=200k trajectories; h=2^-1..2^-6 with tau=h^2/(2D). "
         "KS distance between the rescaled walk distribution and N(0,1): "
         + ", ".join(f"{v:.4f}" for v in rows[:, 2])
         + f"; fitted slope vs log h = {p:.3f}")
    if dist_out is not None:
        xs, dens, theo = dist_out
        w("A6_walk_dist.dat", "z density_MC density_normal", np.column_stack([xs, dens, theo]))
        note(f"A6: finest case (h=2^-6, N={int(rows[-1,1])}): max|density_MC - normal| = "
             f"{np.abs(dens-theo).max():.3e}")

    # finite-step second moment of the biased walk:  Var = N h^2 (1 - beta^2)
    rows = []
    for beta in [0.05, 0.1, 0.2, 0.4]:
        p = (1 + beta) / 2
        X = np.where(rng.random((50000, 400)) < p, 1.0, -1.0).sum(axis=1)
        rows.append([beta, X.var(), 400 * (1 - beta ** 2), beta ** 2])
    rows = np.array(rows)
    w("A7_biased_variance.dat", "beta Var_measured Var_theory beta_squared", rows)
    note("A7: biased walk variance: Var_num = " + ", ".join(f"{v:.2f}" for v in rows[:, 1])
         + " vs N h^2 (1-beta^2) = " + ", ".join(f"{v:.2f}" for v in rows[:, 2])
         + " (the O(beta^2) correction disappears only in the weak-bias limit beta=O(h))")
    return rows


# =====================================================================
# C. Fisher-KPP travelling wave
# =====================================================================
def fisher_kpp(D=1.0, rho=1.0, L=400.0, N=8000, x0=30.0, T=120.0, cfl=0.4, tail0=None):
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    dt = cfl * dx * dx / D
    nt = max(int(T / dt), 1); dt = T / nt
    u = np.where(x < x0, 1.0, 0.0) if tail0 is None else np.where(x < x0, 1.0, np.exp(-tail0 * (x - x0)))
    for _ in range(nt):
        lap = np.zeros_like(u)
        lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
        un = u + dt * (D * lap + rho * u * (1 - u))
        un[0], un[-1] = 1.0, 0.0
        u = np.clip(un, 0, None)
    return x, u, T


def front(r, c, level=0.5):
    """outermost position where the profile crosses 'level' from above"""
    above = np.where(c >= level)[0]
    if len(above) == 0:
        return np.nan
    i = above[-1]
    if i >= len(c) - 1:
        return r[-1]
    return r[i] + (level - c[i]) / (c[i + 1] - c[i]) * (r[i + 1] - r[i])


def fisher_kpp_compact(D=1.0, rho=1.0, L=800.0, N=16000, x0=30.0, half_width=3.0, T=200.0, cfl=0.4):
    """Fisher-KPP with a genuinely compactly supported initial datum (u=1 on
    [x0-half_width, x0+half_width], u=0 outside), used to check the minimal-speed selection."""
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    dt = cfl * dx * dx / D
    nt = max(int(T / dt), 1); dt = T / nt
    u = ((x > x0 - half_width) & (x < x0 + half_width)).astype(float)
    for _ in range(nt):
        lap = np.zeros_like(u)
        lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
        un = u + dt * (D * lap + rho * u * (1 - u))
        un[0], un[-1] = 0.0, 0.0
        u = np.clip(un, 0, None)
    return x, u, T


def partC():
    print("[C] Fisher-KPP travelling wave")
    D = rho = 1.0
    xa, ua, Ta = fisher_kpp(D, rho, T=100.0)
    xb, ub, Tb = fisher_kpp(D, rho, T=140.0)
    c_inst = (front(xb, ub) - front(xa, ua)) / (Tb - Ta)
    c_th = 2 * np.sqrt(D * rho)
    note(f"C1: u_t=D u_xx+rho u(1-u), step data; x_f({Tb:.0f})={front(xb,ub):.2f}; "
         f"instantaneous speed c={c_inst:.4f} vs theory 2*sqrt(D rho)={c_th:.4f} "
         f"(rel. error {abs(c_inst-c_th)/c_th*100:.2f}%)")
    xc, uc, Tc = fisher_kpp(D, rho, T=120.0, tail0=1.0)
    xd, ud, Td = fisher_kpp(D, rho, T=150.0, tail0=1.0)
    c_tail = (front(xd, ud) - front(xc, uc)) / (Td - Tc)
    note(f"C1b: exponential-tail data exp(-x) (critical decay rate): c={c_tail:.4f} "
         f"(rel. error {abs(c_tail-c_th)/c_th*100:.2f}%)")
    # speed selection: a slowly decaying tail lambda0 < lambda_* gives c = D lambda0 + rho/lambda0
    xsl, usl, Tsl = fisher_kpp(D, rho, T=120.0, tail0=0.5)
    xsl2, usl2, Tsl2 = fisher_kpp(D, rho, T=150.0, tail0=0.5)
    c_slow = (front(xsl2, usl2) - front(xsl, usl)) / (Tsl2 - Tsl)
    note(f"C1c: slowly decaying tail exp(-0.5 x) (lambda0=0.5 < lambda_*=1): measured c={c_slow:.4f} "
         f"vs D*lambda0 + rho/lambda0 = {D*0.5 + rho/0.5:.4f} "
         f"(and vs the minimal speed {c_th:.4f})")
    # a genuinely compactly supported initial datum
    xk, uk, Tk = fisher_kpp_compact(D, rho, T=200.0, half_width=3.0, x0=30.0)
    xk2, uk2, Tk2 = fisher_kpp_compact(D, rho, T=250.0, half_width=3.0, x0=30.0)
    c_comp = (front(xk2, uk2) - front(xk, uk)) / (Tk2 - Tk)
    note(f"C1d: genuinely compactly supported data (u=1 on a window of half-width 3, 0 outside): "
         f"measured c={c_comp:.4f} vs minimal speed 2sqrt(D rho)={c_th:.4f} "
         f"(rel. error {abs(c_comp-c_th)/c_th*100:.2f}%)")
    xf = front(xb, ub)
    tail = (ub > 1e-6) & (ub < 1e-2) & (xb > xf)
    lam = -np.polyfit(xb[tail], np.log(ub[tail]), 1)[0]
    note(f"C2: spatial decay rate lambda_num={lam:.4f} vs theory c/(2D)=sqrt(rho/D)={np.sqrt(rho/D):.4f}")
    w("C1_wave_profiles.dat", "x u", np.column_stack([xb, ub]))

    # travelling-wave collapse.  Use a long domain and late times so the front is far from
    # both boundaries and the sampling windows overlap on a wide, front-centred range.
    # Strategy: fit the front position x_f(t) on several snapshots -> measured speed c_fit
    # (compare with the theory 2 sqrt(D rho)), then collapse the profiles onto z = x - c_fit*t.
    c_th = 2 * np.sqrt(D * rho)
    Lc, Nc, x0c = 1000.0, 5000, 50.0
    dxc = Lc / Nc
    dtc = 0.4 * dxc ** 2 / D
    t_end = 300.0
    ntc = int(t_end / dtc)
    dtc = t_end / ntc
    xc = np.linspace(0, Lc, Nc)
    uc = np.where(xc < x0c, 1.0, 0.0)
    snap_times = np.linspace(100.0, 300.0, 9)
    snap = {}
    todo = list(snap_times)
    for it in range(ntc + 1):
        t = it * dtc
        while todo and t >= todo[0] - 0.5 * dtc:
            snap[todo.pop(0)] = uc.copy()
        if it == ntc:
            break
        lap = np.zeros_like(uc)
        lap[1:-1] = (uc[2:] - 2 * uc[1:-1] + uc[:-2]) / dxc ** 2
        un = uc + dtc * (D * lap + rho * uc * (1 - uc))
        un[0], un[-1] = 1.0, 0.0
        uc = np.clip(un, 0, None)
    xf = np.array([front(xc, snap[T2]) for T2 in snap_times])
    c_fit = np.polyfit(snap_times, xf, 1)[0]
    note(f"C5: front positions x_f(t) at t={snap_times[0]:.0f}..{snap_times[-1]:.0f} give a fitted "
         f"speed c_fit={c_fit:.4f} vs theory 2sqrt(D rho)={c_th:.4f} "
         f"(relative error {abs(c_fit-c_th)/c_th*100:.2f}%)")
    prof = []
    for T2 in [200.0, 250.0, 300.0]:
        u2 = snap[T2]
        m = (u2 > 5e-4) & (u2 < 1.0 - 1e-9)
        prof.append((xc[m] - c_fit * T2, u2[m]))
    zlo = max(p[0].min() for p in prof)
    zhi = min(p[0].max() for p in prof)
    zg = np.linspace(zlo, zhi, 400)
    cols = [zg] + [np.interp(zg, z, u) for z, u in prof]
    w("C4_collapse.dat", "z u_t200 u_t250 u_t300", np.column_stack(cols))
    dev = max(np.abs(cols[1] - cols[2]).max(), np.abs(cols[1] - cols[3]).max())
    note(f"C4: collapse z=x-c_fit*t on the common window z in [{zlo:.1f},{zhi:.1f}]: the three "
         f"profiles (t=200,250,300) differ by at most {dev:.2e} "
         f"(a residual of this size at the steep front is expected from the {abs(c_fit-c_th)/c_th*100:.1f}% "
         f"speed error over a 50-time-unit separation)")

    rows = []
    for Dv in [0.5, 1.0, 2.0]:
        xv, uv, Tv = fisher_kpp(Dv, rho, T=100.0)
        cv = (front(xv, uv) - 30.0) / Tv
        rows.append([Dv, cv, 2 * np.sqrt(Dv * rho), cv / (2 * np.sqrt(Dv * rho))])
    rows = np.array(rows)
    w("C2_wave_speed.dat", "D c_measured c_theory ratio", rows)
    note("C3: D=0.5,1,2: measured speed " + ", ".join(f"{v:.3f}" for v in rows[:, 1])
         + " vs theory " + ", ".join(f"{v:.3f}" for v in rows[:, 2]))
    Ds = np.linspace(0.05, 3.0, 200)
    w("C3_speed_curve.dat", "D c_theory", np.column_stack([Ds, 2 * np.sqrt(Ds * rho)]))


# =====================================================================
# D. Tumour reaction-diffusion model, spherical finite volume
# =====================================================================
def thomas(a, b, c, d):
    """Thomas algorithm for a tridiagonal system (kept for reference/testing)"""
    n = len(b)
    cp = np.empty(n); dp = np.empty(n)
    cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
    for k in range(1, n):
        den = b[k] - a[k] * cp[k - 1]
        cp[k] = c[k] / den
        dp[k] = (d[k] - a[k] * dp[k - 1]) / den
    x = np.empty(n)
    x[-1] = dp[-1]
    for k in range(n - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return x


def tumor_sim(Dc=1.0, rho=1.0, Dn=1.0, kn=1.0, R=40.0, N=2000, T=20.0,
              seed_r=1.5, c_inf=1.0, chi=0.0, dt=None, snapshot_times=None, w0=1.0,
              sat=1.0, kappa=0.05, mu=1.0, n_nec=0.15):
    """Spherically symmetric tumour reaction-diffusion model (dimensionless)

       c_t = (1/r^2) ( r^2 [ Dc c_r - chi c n_r ] )_r
             + rho c (1 - sat*c) n/(kappa+n)  -  mu c H(n_nec - n)
       n_t = (1/r^2) ( r^2 n_r )_r - kn c n

    With the default sat=1 the proliferation term carries the logistic factor (1-c), so
    that for chi=0, n -> n_inf and kappa -> 0 the cell equation reduces EXACTLY to the
    Fisher-KPP equation  c_t = Dc lap c + rho c (1-c).

    BC (r=0 symmetry):
        c_r(0,t) = n_r(0,t) = 0
    BC (r=R_dom, outer boundary):
        zero TOTAL cell flux:  Dc c_r - chi c n_r = 0   (for chi=0 this is c_r = 0),
        n(R_dom,t) = c_inf.

    IC: c = 0.5[1 - tanh((r-R0)/w0)],  n = c_inf.

    Numerics: cell-centred finite-volume shells. The k-th shell is
    [r_{k-1/2}, r_{k+1/2}] with r_{k+/-1/2} = (k +/- 1/2) dr, i.e. the first shell covers
    [0, dr/2] as usual for cell-centred finite volumes in spherical symmetry; the
    symmetry condition at r=0 is imposed through the ghost value c_0 = c_2.
    Second-order centred differences in space, explicit Heun in time, followed by a
    positivity/saturation projection (clipping) whose effect is measured and reported.
    """
    dr = R / N
    rk = np.arange(1, N + 1) * dr               # cell centres (k-1/2)dr, k=1..N
    rL = (np.arange(1, N + 1) - 0.5) * dr        # inner faces r_{k-1/2}
    rR = (np.arange(1, N + 1) + 0.5) * dr        # outer faces r_{k+1/2}
    Vc = (rR ** 3 - rL ** 3) / 3.0
    # Finite-volume coefficients:  lap_k = cR_k (u_{k+1}-u_k) - cL_k (u_k-u_{k-1}) with
    # cL_k = A_L/(V_k dr), cR_k = A_R/(V_k dr);  V_k ~ r_k^2 dr gives cL+cR ~ 2/dr^2.
    cL = rL ** 2 / (Vc * dr)
    cR = rR ** 2 / (Vc * dr)
    cL[0] = 0.0                      # r=0 symmetry
    if dt is None:
        lam = np.max(cL + cR)
        dt = min(0.75 / (max(Dc, Dn) * lam), 1.0 / max(rho, 1e-12))
    nt = max(int(T / dt), 1); dt = T / nt
    c = 0.5 * (1.0 - np.tanh((rk - seed_r) / w0))
    n = np.full(N, c_inf)
    rec = {}
    snap = sorted(snapshot_times) if snapshot_times else []
    clip_stats = {'events': 0, 'max_c_corr': 0.0, 'max_n_corr': 0.0, 'sum_c_corr': 0.0}

    def rhs(cc, nn):
        gcL = np.zeros(N); gcL[1:] = cc[1:] - cc[:-1]
        gcR = np.zeros(N); gcR[:-1] = cc[1:] - cc[:-1]
        gnL = np.zeros(N); gnL[1:] = nn[1:] - nn[:-1]
        gnR = np.zeros(N); gnR[:-1] = nn[1:] - nn[:-1]
        gnR[-1] = c_inf - nn[-1]                 # Dirichlet ghost value for n
        if chi != 0.0:
            # zero TOTAL cell flux at r=R_dom:  Dc c_r - chi c n_r = 0
            # (in difference form over the last half interval, n_r is taken from gnR[-1])
            gcR[-1] = chi * cc[-1] * gnR[-1] / Dc
        div = Dc * (cR * gcR - cL * gcL)
        lap = Dn * (cR * gnR - cL * gnL)
        if chi != 0.0:
            # model flux: J_c = -Dc c_r + chi * c * n_r  (chi>0: cells climb the nutrient
            # gradient).  It enters the equation as  -div(chi c grad n), i.e. a MINUS sign
            # relative to the diffusive term.  Face values use the arithmetic mean of the
            # two neighbouring cell values.
            chiL = np.concatenate([[cc[0]], 0.5 * (cc[:-1] + cc[1:])])
            chiR = np.concatenate([0.5 * (cc[:-1] + cc[1:]), [cc[-1]]])
            div = div - chi * (cR * chiR * gnR - cL * chiL * gnL)
        prolif = rho * cc * (1.0 - sat * np.clip(cc, 0, 1)) * nn / (kappa + nn)
        death = mu * cc * (nn < n_nec)
        return div + prolif - death, lap - kn * cc * nn

    for it in range(nt + 1):
        t = it * dt
        if snap and abs(t - snap[0]) < 0.5 * dt:
            rec[snap.pop(0)] = (rk.copy(), c.copy(), n.copy())
        if it == nt:
            break
        k1c, k1n = rhs(c, n)                      # both derivatives at (c,n)
        c1 = c + dt * k1c
        n1 = n + dt * k1n
        k2c, k2n = rhs(c1, n1)                    # both derivatives at the predictor
        cn = c + 0.5 * dt * (k1c + k2c)
        nn_ = n + 0.5 * dt * (k1n + k2n)
        # positivity / saturation projection; record how much it actually changes things
        c_new = np.clip(cn, 0.0, 1.0)
        n_new = np.clip(nn_, 0.0, None)
        dc_ = np.abs(c_new - cn); dn_ = np.abs(n_new - nn_)
        if dc_.max() > 0.0 or dn_.max() > 0.0:
            clip_stats['events'] += 1
            clip_stats['max_c_corr'] = max(clip_stats['max_c_corr'], float(dc_.max()))
            clip_stats['max_n_corr'] = max(clip_stats['max_n_corr'], float(dn_.max()))
            clip_stats['sum_c_corr'] += float(dc_.sum())
        c, n = c_new, n_new
    return rk, c, n, dt, rec, clip_stats


def partD():
    print("[D] Tumour reaction-diffusion model (spherical)")
    R, N = 120.0, 2400

    # D0: quasi-steady nutrient profile: n'' + (2/r) n' = kn (c=1), n'(0)=0, n(Rr)=n_inf
    #     => n(r) = n_inf * Rr * sinh(r/L) / (r * sinh(Rr/L)),  L = sqrt(Dn/kn)
    Dn, kn, Rr, ninf = 1.0, 1.0, 5.0, 1.0
    L = np.sqrt(Dn / kn)
    rr = np.linspace(1e-3, Rr, 400)
    n_an = ninf * Rr * np.sinh(rr / L) / (rr * np.sinh(Rr / L))
    w("D0_nutrient_analytic.dat", "r n_analytic", np.column_stack([rr, n_an]))
    n_c = ninf * Rr / (L * np.sinh(Rr / L))
    note(f"D0: quasi-steady nutrient (c=1, R={Rr}, L=sqrt(Dn/kn)={L:.3f}): "
         f"n(r)=n_inf R sinh(r/L)/(r sinh(R/L)); n(0)={n_c:.4f} ({n_c/ninf*100:.1f}% of rim); "
         f"n(L)/n(0)={np.interp(L,rr,n_an)/n_c:.3f}; L/R={L/Rr:.2f}")
    w("D0b_nutrient_compare.dat", "r n_analytic", np.column_stack([rr, n_an]))
    # same analytic profile for a deeply hypoxic tumour (R/L = 30), used in Fig. 1
    prof30 = []
    for Rv in [1.0, 5.0, 30.0]:
        rv = np.linspace(1e-4, 1.0, 400)                # rho = r/R in (0,1]
        nv = Rv * np.sinh(Rv * rv) / (Rv * rv * np.sinh(Rv))
        prof30.append(nv)
    w("D0c_profiles_RoverL.dat", "rho n_RL1 n_RL5 n_RL30", np.column_stack([rv] + prof30))
    # verify the time-dependent solver against the same steady profile
    rs, cs, ns, dts, _, _ = tumor_sim(Dc=1.0, rho=0.0, Dn=1.0, kn=1.0, R=30.0, N=1500, T=6.0,
                                   seed_r=40.0, mu=0.0)
    ana_s = ana_s = 30.0 * np.sinh(rs / 1.0) / (rs * np.sinh(30.0))
    note(f"D0c: time-dependent solver vs analytic profile (R=30, rho=0): "
         f"max|num-ana|={np.abs(ns-ana_s).max():.3e}")

    # D1: baseline run: growth curve + profiles
    snap = [2.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    rr_, cc_, nn_, dt, rec, clip = tumor_sim(Dc=1.0, rho=1.0, Dn=1.0, kn=0.25, R=R, N=N, T=30.0,
                                       seed_r=2.0, w0=1.0, snapshot_times=snap)
    Rf = front(rr_, cc_)
    note(f"D1: baseline Dc=rho=1, Dn=1, kn=0.25 (L=2), R0=2: R(30)={Rf:.3f}, "
         f"mean speed {(Rf-2.0)/30:.4f}, dt={dt:.2e}; n(0,30)={nn_[0]:.4e}")
    note(f"D1b (clipping statistics over the whole run): events={clip['events']} of {int(30.0/dt)} steps, "
         f"max correction |dc|={clip['max_c_corr']:.2e}, max |dn|={clip['max_n_corr']:.2e}, "
         f"total sum|dc|={clip['sum_c_corr']:.2e}")
    rows = []
    for t in snap:
        r_, c_, n_ = rec[t]
        R_t = front(r_, c_)
        core = r_[(c_ < 0.5) & (r_ < R_t)]
        R_nec = core.max() if len(core) else 0.0
        rows.append([t, R_t, n_[0], R_nec])
    rows = np.array(rows)
    w("D2_growth_curve.dat", "t R_front n_center R_necrotic", rows)
    note("D2: R(t) = " + ", ".join(f"{v:.2f}" for v in rows[:, 1])
         + "; n(0,t) = " + ", ".join(f"{v:.4f}" for v in rows[:, 2])
         + "; R_nec(t) = " + ", ".join(f"{v:.2f}" for v in rows[:, 3]))
    inc = [(rows[i + 1, 1] - rows[i, 1]) / (rows[i + 1, 0] - rows[i, 0]) for i in range(len(rows) - 1)]
    note("D2b: incremental speed dR/dt = " + ", ".join(f"{v:.3f}" for v in inc)
         + " (nutrient-rich limit 2*sqrt(Dc*rho)=2 for kappa->0)")

    # D3: profiles at all snapshot times
    cols = []
    for t in snap:
        r_, c_, n_ = rec[t]
        m = r_ <= 55.0
        cols.append((r_[m], c_[m], n_[m]))
    step = max(len(cols[0][0]) // 350, 1)
    arr = [np.column_stack([r_[::step], c_[::step], n_[::step]]) for r_, c_, n_ in cols]
    nmin = min(len(a) for a in arr)
    w("D3_profiles.dat", " ".join(f"r{tg:.0f} c{tg:.0f} n{tg:.0f}" for tg in snap),
      np.hstack([a[:nmin] for a in arr]))

    # D4: nutrient penetration depth L = sqrt(Dn/kn)
    rows = []
    for (Dn_v, kn_v) in [(1.0, 1.0), (1.0, 0.25), (1.0, 0.0625)]:
        rr_, cc_, nn_, _, _, _ = tumor_sim(Dc=1.0, rho=1.0, Dn=Dn_v, kn=kn_v, R=R, N=N,
                                        T=25.0, seed_r=2.0, w0=1.0)
        R_t = front(rr_, cc_)
        core = rr_[(cc_ < 0.5) & (rr_ < R_t)]
        R_nec = core.max() if len(core) else 0.0
        rows.append([np.sqrt(Dn_v / kn_v), R_t, nn_[0], R_nec])
    rows = np.array(rows)
    w("D4_penetration.dat", "L R_front n_center R_necrotic", rows)
    note("D4: L=sqrt(Dn/kn)=" + ", ".join(f"{v:.2f}" for v in rows[:, 0])
         + " -> R(25)=" + ", ".join(f"{v:.2f}" for v in rows[:, 1])
         + ", n(0)=" + ", ".join(f"{v:.4f}" for v in rows[:, 2])
         + ", R_nec=" + ", ".join(f"{v:.2f}" for v in rows[:, 3]))

    # D5: necrotic core and viable rim at t=30
    r_, c_, n_ = rec[30.0]
    R_t = front(r_, c_)
    core = r_[(c_ < 0.5) & (r_ < R_t)]
    R_nec = core.max() if len(core) else 0.0
    nlow = r_[n_ < 0.15].max() if np.any(n_ < 0.15) else 0.0
    note(f"D5: t=30: R={R_t:.3f}, necrotic core R_nec={R_nec:.3f} (R_nec/R={R_nec/R_t:.3f}), "
         f"rim thickness ~{R_t-R_nec:.3f}; hypoxic radius (n<0.15)={nlow:.3f}")
    m = r_ <= 55.0
    step = max(int(m.sum()) // 450, 1)
    w("D7_necrotic_profile.dat", "r c n",
      np.column_stack([r_[m][::step], c_[m][::step], n_[m][::step]]))

    # D6: chemotaxis term chi (chi>0: cells climb the nutrient gradient, i.e. move outward)
    rows = []
    for chi in [0.0, 0.5, 1.0, -1.0]:
        rr_, cc_, nn_, _, _, _ = tumor_sim(Dc=1.0, rho=1.0, Dn=1.0, kn=0.25, R=R, N=N,
                                        T=20.0, seed_r=2.0, w0=1.0, chi=chi)
        rows.append([chi, front(rr_, cc_), cc_.max(), nn_[0]])
    rows = np.array(rows)
    w("D6_chemotaxis.dat", "chi R_front c_max n_center", rows)
    note("D6: chi = " + ", ".join(f"{v:+.1f}" for v in rows[:, 0])
         + " -> R(20) = " + ", ".join(f"{v:.2f}" for v in rows[:, 1]))

    # ------------------------------------------------------------------
    # D8: long-time trend of the front speed (does R(t)/t approach 2 sqrt(rho/(1+kappa))?)
    # ------------------------------------------------------------------
    times = [30.0, 50.0, 100.0, 200.0]
    rows = []
    for Tt in times:
        rr2, cc2, nn2, dt2, _, _ = tumor_sim(Dc=1.0, rho=1.0, Dn=1.0, kn=0.25, R=400.0, N=8000,
                                             T=Tt, seed_r=2.0, w0=1.0)
        Rf2 = front(rr2, cc2)
        rows.append([Tt, Rf2, (Rf2 - 2.0) / Tt, nn2[0]])
    rows = np.array(rows)
    w("D8_speed_trend.dat", "t R_front R_over_t n_center", rows)
    asym = 2.0 * np.sqrt(1.0 / (1.0 + 0.05))
    note("D8: front position and mean speed R(t)/t at t = "
         + ", ".join(f"{v:.0f}" for v in rows[:, 0]) + ": R/t = "
         + ", ".join(f"{v:.4f}" for v in rows[:, 2])
         + f"; reference 2*sqrt(rho/(1+kappa)) = {asym:.4f}")

    # ------------------------------------------------------------------
    # D9: time-step convergence of the full tumour solver (Heun + clipping)
    # ------------------------------------------------------------------
    rows = []
    dt_auto = None
    for fac in [1.0, 0.5, 0.25]:
        if dt_auto is None:
            r_, c_, n_, dt_auto, _, _ = tumor_sim(Dc=1.0, rho=1.0, Dn=1.0, kn=0.25, R=120.0,
                                                  N=1200, T=10.0, seed_r=2.0, w0=1.0)
            rows.append([dt_auto, front(r_, c_), n_[0]])
        else:
            r_, c_, n_, dtv, _, _ = tumor_sim(Dc=1.0, rho=1.0, Dn=1.0, kn=0.25, R=120.0,
                                              N=1200, T=10.0, seed_r=2.0, w0=1.0,
                                              dt=dt_auto * fac)
            rows.append([dtv, front(r_, c_), n_[0]])
    rows = np.array(rows)
    w("D9_dt_convergence.dat", "dt R_front n_center", rows)
    dR = [abs(rows[i, 1] - rows[-1, 1]) for i in range(len(rows))]
    note("D9: time-step refinement (dt, R(10)): "
         + ", ".join(f"({rows[i,0]:.2e}, {rows[i,1]:.6f})" for i in range(len(rows)))
         + f"; max |R(dt)-R(dt/4)| = {max(dR[:2]):.2e}")

    # ------------------------------------------------------------------
    # D10: grid convergence of the full tumour solver
    # ------------------------------------------------------------------
    rows = []
    for NN in [600, 1200, 2400]:
        rr4, cc4, nn4, dt4, _, _ = tumor_sim(Dc=1.0, rho=1.0, Dn=1.0, kn=0.25, R=120.0, N=NN,
                                             T=10.0, seed_r=2.0, w0=1.0)
        rows.append([120.0 / NN, front(rr4, cc4), nn4[0]])
    rows = np.array(rows)
    w("D10_grid_convergence.dat", "dr R_front n_center", rows)
    note("D10: grid refinement (dr, R(10)): "
         + ", ".join(f"({rows[i,0]:.4f}, {rows[i,1]:.6f})" for i in range(len(rows)))
         + f"; max |R - R(finest)| = {max(abs(rows[i,1]-rows[-1,1]) for i in range(len(rows))):.2e}")
    return rec


def partB():
    print("[B] 2-D axisymmetric Fisher front")
    Nr, Nth, Rmax = 400, 360, 70.0
    r = np.linspace(0, Rmax, Nr)
    th = np.linspace(0, 2 * np.pi, Nth, endpoint=False)
    dr, dth = r[1] - r[0], th[1] - th[0]
    D = rho = 1.0
    dt = 0.25 * min(dr, 0.5 * Rmax * dth) ** 2 / D
    Tsnap = [10.0, 20.0, 30.0]
    T = Tsnap[-1]
    nt = max(int(T / dt), 1); dt = T / nt
    u = np.zeros((Nr, Nth)); u[r < 3.0, :] = 1.0
    hist = {}
    todo = list(Tsnap)
    for it in range(nt + 1):
        t = it * dt
        if todo and abs(t - todo[0]) < 0.5 * dt:
            todo.pop(0)
            hist[round(t, 6)] = np.array([front(r, u[:, j]) for j in range(Nth)])
        if it == nt:
            break
        urr = np.zeros_like(u); urr[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dr ** 2
        ur = np.zeros_like(u); ur[1:-1] = (u[2:] - u[:-2]) / (2 * dr)
        utt = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / (r[:, None] ** 2 * dth ** 2 + 1e-12)
        lap = urr + ur / (r[:, None] + 1e-12) + utt
        un = u + dt * (D * lap + rho * u * (1 - u))
        un[0] = un[1]; un[-1] = 0.0
        u = np.clip(un, 0, 1)
    keys = sorted(hist)
    rows = [[np.degrees(th[j])] + [hist[k][j] for k in keys] for j in range(0, Nth, 6)]
    rows = np.array(rows)
    w("B1_axisym_front.dat", "theta_deg " + " ".join(f"R_t{int(k)}" for k in keys), rows)
    spread = [(np.nanmax(hist[k]) - np.nanmin(hist[k])) / np.nanmean(hist[k]) * 100 for k in keys]
    note("B1: angular spread of R(theta): "
         + ", ".join(f"t={int(k)}: {s:.2e}%" for k, s in zip(keys, spread))
         + " (machine precision -> radially symmetric solution is stable)")
    Rs = [np.nanmean(hist[k]) for k in keys]
    note("B1b: Rbar(t)=" + ", ".join(f"{v:.3f}" for v in Rs)
         + f"; speed (t={int(keys[-2])}->{int(keys[-1])}) = {(Rs[-1]-Rs[-2])/(keys[-1]-keys[-2]):.4f} "
         + f"vs 2sqrt(D rho)={2*np.sqrt(D*rho):.3f}")


if __name__ == "__main__":
    partA(); partA_continuum(); partC(); partD(); partB()
    with open(os.path.join(HERE, "..", "figs", "numerics_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\nDONE")
