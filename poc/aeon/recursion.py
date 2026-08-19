"""
aeon/recursion.py
====================

THE CANONICAL DEFINITION OF RECURSION (Gen 6). Single source of truth.
Any other code in this repo that references "the recursion cell" must
agree with this module.

WHAT RECURSION IS
-----------------
A gate-free, contractive, recurrent cell with a bounded persistent
hidden state. The cell evolves a state h_t in R^H in response to an
input x_t in R^{in_dim}, with the guarantee that the recurrent map's
largest singular value is strictly less than 1. This bound is not
enforced after the fact; it is built into the parameterization.

No setting of the parameters violates the per-matrix bound.
(The joint map carries a separate, weaker guarantee -- see SCOPE below.)

THE CERTIFICATE
---------------
    sigma_max(W_h) < MARGIN_H,   sigma_max(W_c) < MARGIN_C,
    with MARGIN_H, MARGIN_C < 1.

SCOPE OF THAT CERTIFICATE -- read this before relying on it.

The bound above is on the two matrices INDIVIDUALLY. It is not a
bound on the joint state map. The state is (h, c), and h reaches
h_next by two paths -- directly through W_h, and through c_next,
which enters the same tanh. At the origin (sech^2 = 1) the joint
Jacobian is

    J = [[W_h + lam*W_c, (1-lam)I],
         [    lam*W_c,   (1-lam)I]]

and sigma_max(J) can exceed 1 while both matrix bounds hold.
Measured over 200 seeds (H=64, s=1.5, margins 0.98/0.95), with
sigma(W_h) and sigma(W_c) inside their margins on every seed:

    lam    mean sigma(J)   exceeds 1     mean rho(J)
    0.50      1.0287        142/200         0.7607
    0.70      0.9904         84/200         0.6901

lam = 0.50 was the old default. lam = 0.70 is the current one, and the
move is why it changed: raising lambda cut the Euclidean violation rate
from 71% of seeds to 42% and pulled the mean just below 1. It did NOT
eliminate the violation. The cell is still not a Euclidean contraction
at default init, and the Banach argument does not apply in that metric.

What does hold:

  * rho(J) < 1 on every seed measured -- the joint map is
    asymptotically stable and the state cannot grow without bound.

  * Because rho(J) < 1, there exists P > 0 with J^T P J - P < 0, and
    the map is a strict contraction IN THE P-METRIC. Measured
    ||J||_P < 1 on 200/200 seeds at both lambda values (mean 0.825 at
    lam = 0.70, max 0.925).

SCOPE OF THE P-METRIC RESULT -- this is a LOCAL statement.

P is solved for J at the origin, i.e. for the LINEARIZATION at one
point. It is not a metric uniform over the reachable set, so it is not
a nonlinear contraction certificate in the sense of Lohmiller & Slotine
(1998), "On Contraction Analysis for Non-linear Systems", Automatica
34(6). That would require a single metric M(x) > 0 with

    dF/dx^T M + M dF/dx + dM/dt  <=  -beta M  < 0

holding uniformly over reachable states. Aeon does not currently have
one. What is published here is: strict contraction in the P-metric of
the linearization at the origin, plus asymptotic stability.

The Euclidean bound on W_h and W_c remains useful and cheap; it just
is not the whole guarantee, and must not be quoted as if it were.

THE ATLAS (two equivalent implementations of the same cell)
-----------------------------------------------------------
CHART B (parameterization-as-certificate, default for training):
    W = sigmoid(s) * MARGIN * Cayley(A) @ diag(tanh(d))
    sigma_max(W) <= sigmoid(s) * MARGIN < MARGIN, by construction.

CHART A (direct storage + projection, for audit and reference):
    W is an nn.Linear weight. After each forward, project to
    sigma_max <= MARGIN by rescaling.

Both produce the same forward pass on matched parameters, to ~1e-7
precision. The equivalence_check() function at the bottom verifies
this.

A NOTE ON lambda
----------------
lam = sigmoid(log_lambda) and log_lambda is a free parameter. At
lam = 0 the carry becomes a pure integrator with no decay and
rho(J) = 1.0000 exactly -- stability is forfeited. Nothing in the
parameterization prevents log_lambda from reaching that value, so
the guarantee is conditional on lam, and audit_certificates()
reports lam on every call. Default init is sigmoid^-1(0.7).

BANNED
------
* No phi = 4/pi anywhere. Refuted by Delta-1 ablation.
* No "purge" / circuit breaker doing load-bearing stability work.
  Stability is the parameterization, not a safety net.
* MARGIN values are config, not hardcoded inside the cell logic.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spectral utilities
# ---------------------------------------------------------------------------

def sigma_max(W: torch.Tensor) -> float:
    """Largest singular value of W. Pure read-off, never modifies.

    torch.linalg.svdvals does not support bf16/fp16 on CUDA (cusolver
    gesvdj only supports fp32/fp64). The audit doesn't care about
    precision, so cast to fp32 for the SVD.
    """
    Wd = W.detach()
    if Wd.dtype in (torch.bfloat16, torch.float16):
        Wd = Wd.float()
    return torch.linalg.svdvals(Wd).max().item()


def project_sigma_(W: torch.Tensor, target: float):
    """In-place rescale W so largest singular value equals `target`.
    Used by Chart A only."""
    s = sigma_max(W)
    if s > 1e-8:
        with torch.no_grad():
            W.mul_(target / s)


def cayley(A: torch.Tensor) -> torch.Tensor:
    """
    Cayley transform A -> orthogonal Q via
        S = A - A^T            (skew-symmetric)
        Q = (I + S)^{-1} (I - S)
    Q satisfies Q^T Q = I. Smooth in A so gradients flow cleanly.

    torch.linalg.solve does not support bf16 on CUDA (cusolver lu_factor
    only supports fp32/fp64), so we compute the solve in fp32 and cast back.
    """
    orig_dtype = A.dtype
    A_f = A.float() if A.dtype in (torch.bfloat16, torch.float16) else A
    S = A_f - A_f.transpose(-2, -1)
    I = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device)
    Q = torch.linalg.solve(I + S, I - S)
    return Q.to(orig_dtype)


# ---------------------------------------------------------------------------
# CHART B: parameterization-as-certificate (used in AeonBlock)
# ---------------------------------------------------------------------------
#
# step:
#   c_{t+1} = (1 - lam) * c_t + lam * tanh(W_c @ h_t)           [delta-decay carry]
#   h_{t+1} = tanh( W_x @ x_t  +  W_h @ h_t  +  c_{t+1} )       [contractive update]
# with
#   W_h = sigmoid(s_h) * MARGIN_H * Cayley(A_h) @ diag(tanh(d_h))
#   W_c = sigmoid(s_c) * MARGIN_C * Cayley(A_c) @ diag(tanh(d_c))
#   lam = sigmoid(log_lambda)
#

class RecursionChartB(nn.Module):
    def __init__(self, in_dim: int, H: int,
                 margin_H: float = 0.98,
                 margin_C: float = 0.95):
        super().__init__()
        assert 0.0 < margin_H < 1.0
        assert 0.0 < margin_C < 1.0
        self.H = H
        self.in_dim = in_dim
        self.MARGIN_H = margin_H
        self.MARGIN_C = margin_C

        # Cayley-D parameters
        self.A_h = nn.Parameter(0.05 * torch.randn(H, H))
        self.d_h = nn.Parameter(0.5  * torch.randn(H))
        self.s_h = nn.Parameter(torch.tensor(1.5))
        self.A_c = nn.Parameter(0.05 * torch.randn(H, H))
        self.d_c = nn.Parameter(0.5  * torch.randn(H))
        self.s_c = nn.Parameter(torch.tensor(1.5))

        # sigmoid^-1(0.7): lam=0.5 sits outside the contractive window; lam=0
        # makes the carry a pure integrator (rho(J) = 1). See module docstring.
        self.log_lambda = nn.Parameter(torch.tensor(0.8473))  # sigmoid -> 0.70

        # Input projection (in_dim -> H)
        self.W_x = nn.Linear(in_dim, H, bias=True)

    def _build(self, A, d, s, margin):
        Q = cayley(A)
        D = torch.diag(torch.tanh(d))
        scale = torch.sigmoid(s) * margin
        return scale * (Q @ D)

    def W_h_mat(self) -> torch.Tensor:
        return self._build(self.A_h, self.d_h, self.s_h, self.MARGIN_H)

    def W_c_mat(self) -> torch.Tensor:
        return self._build(self.A_c, self.d_c, self.s_c, self.MARGIN_C)

    def step(self, x_t: torch.Tensor, h_t: torch.Tensor, c_t: torch.Tensor):
        """One time-step. x_t (B, in_dim), h_t and c_t (B, H). Returns (h, c)."""
        Wh = self.W_h_mat()
        Wc = self.W_c_mat()
        lam = torch.sigmoid(self.log_lambda)
        c_next = (1.0 - lam) * c_t + lam * torch.tanh(h_t @ Wc.T)
        h_next = torch.tanh(self.W_x(x_t) + h_t @ Wh.T + c_next)
        return h_next, c_next

    def forward(self, x: torch.Tensor):
        """For standalone use. (B, T, in_dim) -> (B, T, H)."""
        B, T, _ = x.shape
        h = x.new_zeros(B, self.H)
        c = x.new_zeros(B, self.H)
        out = []
        for t in range(T):
            h, c = self.step(x[:, t], h, c)
            out.append(h)
        return torch.stack(out, dim=1)


# ---------------------------------------------------------------------------
# CHART A: direct storage + projection (for audit and parity check)
# ---------------------------------------------------------------------------

class RecursionChartA(nn.Module):
    def __init__(self, in_dim: int, H: int,
                 margin_H: float = 0.98,
                 margin_C: float = 0.95):
        super().__init__()
        self.H = H
        self.in_dim = in_dim
        self.MARGIN_H = margin_H
        self.MARGIN_C = margin_C

        self.W_x = nn.Linear(in_dim, H, bias=True)
        self.W_h = nn.Linear(H, H, bias=False)
        self.W_c = nn.Linear(H, H, bias=False)
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

        with torch.no_grad():
            sh = sigma_max(self.W_h.weight)
            sc = sigma_max(self.W_c.weight)
            if sh > 0: self.W_h.weight.mul_(0.6 * margin_H / sh)
            if sc > 0: self.W_c.weight.mul_(0.6 * margin_C / sc)

    def project_now(self):
        with torch.no_grad():
            project_sigma_(self.W_h.weight, self.MARGIN_H * 0.999)
            project_sigma_(self.W_c.weight, self.MARGIN_C * 0.999)

    def step(self, x_t, h_t, c_t):
        self.project_now()
        lam = torch.sigmoid(self.log_lambda)
        c_next = (1.0 - lam) * c_t + lam * torch.tanh(self.W_c(h_t))
        h_next = torch.tanh(self.W_x(x_t) + self.W_h(h_t) + c_next)
        return h_next, c_next

    def forward(self, x):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.H)
        c = x.new_zeros(B, self.H)
        out = []
        for t in range(T):
            h, c = self.step(x[:, t], h, c)
            out.append(h)
        return torch.stack(out, dim=1)


# ---------------------------------------------------------------------------
# Three-chart audit
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Joint-map certificate  (the (h, c) system, not the individual matrices)
# ---------------------------------------------------------------------------

def joint_jacobian(Wh: torch.Tensor, Wc: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Jacobian of the joint (h, c) update, evaluated AT THE ORIGIN.

        c_next = (1-lam)c + lam*tanh(W_c h)
        h_next = tanh(W_x x + W_h h + c_next)

    WHY THE ORIGIN, AND WHAT THAT DOES AND DOES NOT BUY.

    The general Jacobian at an arbitrary state is

        J_gen = [[D1(W_h + lam*D2*W_c), D1(1-lam)I],
                 [     lam*D2*W_c,         (1-lam)I]]

    with D1 = diag(sech^2(pre-activation of the outer tanh)) and
    D2 = diag(sech^2(h @ W_c^T)), both with entries in (0, 1].

    SOUND: D1 factors out on the left --

        J_gen = blkdiag(D1, I) @ J_mid,   ||blkdiag(D1, I)||_2 <= 1

    so sigma(J_gen) <= sigma(J_mid). The OUTER tanh can only reduce the
    Jacobian, never enlarge it. (Checked over 400 random (D1, D2)
    draws: sigma with D1 present never exceeded sigma at D1 = I.)

    NOT ESTABLISHED: D2 is not a left factor. It sits INSIDE the block
    W_h + lam*D2*W_c, where shrinking it can ENLARGE that block through
    cancellation. With W_h and W_c adversarially aligned
    (W_h = -1.2*lam*W_c, H=32, lam=0.7), sigma is non-monotonic in d2
    and the origin is not the maximum:

        d2 = 1.00 (origin)  sigma = 0.5406
        d2 = 0.50           sigma = 0.4389   <- minimum
        d2 = 0.01           sigma = 0.6187   <- above the origin

    So the origin is NOT proven to be the worst case over D2, and this
    docstring does not claim it is. See tests/test_recursion.py::
    test_origin_is_not_worst_case_over_d2, which pins the
    counterexample so the claim cannot be quietly restored.

    The origin is used because it is the natural reference point and
    because the D1 half of the argument is sound there. Callers who
    need a bound over reachable states should use
    chart_D_sampled_max_sigma from audit_certificates(), and read its
    caveat: random D2 sampling does not find the adversarial case.
    """
    H = Wh.shape[0]
    I = torch.eye(H, dtype=Wh.dtype, device=Wh.device)
    top = torch.cat([Wh + lam * Wc, (1.0 - lam) * I], dim=1)
    bot = torch.cat([lam * Wc,      (1.0 - lam) * I], dim=1)
    return torch.cat([top, bot], dim=0)


def joint_jacobian_general(Wh: torch.Tensor, Wc: torch.Tensor, lam: float,
                           d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    """
    Joint Jacobian at an arbitrary state, given the two sech^2 diagonals.
    d1, d2 are vectors of length H with entries in (0, 1].
    joint_jacobian(...) is the special case d1 = d2 = 1.
    """
    H = Wh.shape[0]
    I = torch.eye(H, dtype=Wh.dtype, device=Wh.device)
    D1 = torch.diag(d1.to(Wh.dtype)); D2 = torch.diag(d2.to(Wh.dtype))
    top = torch.cat([D1 @ (Wh + lam * D2 @ Wc), D1 @ ((1.0 - lam) * I)], dim=1)
    bot = torch.cat([lam * D2 @ Wc,             (1.0 - lam) * I],        dim=1)
    return torch.cat([top, bot], dim=0)


def solve_dlyap(J: torch.Tensor, iters: int = 64) -> torch.Tensor:
    """
    Solve J^T P J - P = -I for P > 0 by squaring:  P = sum_k (J^T)^k J^k.
    Converges iff rho(J) < 1. Each pass squares J, so ~64 passes is ample.
    """
    P = torch.eye(J.shape[0], dtype=J.dtype, device=J.device)
    A = J.clone()
    for _ in range(iters):
        P = P + A.transpose(-2, -1) @ P @ A
        A = A @ A
        if torch.linalg.matrix_norm(A, 2) < 1e-14:
            break
    return P


def p_metric_norm(J: torch.Tensor, P: torch.Tensor) -> float:
    """||J||_P = sqrt(lam_max(P^-1/2 J^T P J P^-1/2)). Contraction iff < 1."""
    L = torch.linalg.cholesky(P)
    M = torch.linalg.solve_triangular(
        L, (L.transpose(-2, -1) @ J).transpose(-2, -1), upper=False
    ).transpose(-2, -1)
    return torch.linalg.svdvals(M)[0].item()


def audit_certificates(cell, perturb_seed: int = 0):
    """Run the three-chart audit on a Recursion cell. Returns a dict."""
    with torch.no_grad():
        if isinstance(cell, RecursionChartB):
            Wh = cell.W_h_mat()
            Wc = cell.W_c_mat()
        else:
            cell.project_now()
            Wh = cell.W_h.weight
            Wc = cell.W_c.weight

        s_h = sigma_max(Wh)
        s_c = sigma_max(Wc)

        H = cell.H
        M = Wh.T @ Wh - torch.eye(H, device=Wh.device)
        max_eig = torch.linalg.eigvalsh(M).max().item()

        perturbed = None
        if isinstance(cell, RecursionChartB):
            saved = {k: getattr(cell, k).detach().clone()
                     for k in ["A_h", "d_h", "s_h"]}
            g = torch.Generator(device=cell.A_h.device).manual_seed(perturb_seed)
            with torch.no_grad():
                cell.A_h.add_(torch.randn(cell.A_h.shape, generator=g,
                                          device=cell.A_h.device))
                cell.d_h.add_(torch.randn(cell.d_h.shape, generator=g,
                                          device=cell.d_h.device))
                cell.s_h.add_(torch.randn(cell.s_h.shape, generator=g,
                                          device=cell.s_h.device))
            perturbed = sigma_max(cell.W_h_mat())
            with torch.no_grad():
                for k, v in saved.items():
                    getattr(cell, k).copy_(v)

    # ---- Chart D: the joint (h, c) map -----------------------------------
    with torch.no_grad():
        lam = float(torch.sigmoid(cell.log_lambda)) if hasattr(cell, "log_lambda") else 0.5
        J = joint_jacobian(Wh.double(), Wc.double(), lam)
        sigma_J = float(torch.linalg.svdvals(J)[0])
        rho_J = float(abs(torch.linalg.eigvals(J)).max())
        if rho_J < 1.0:
            P = solve_dlyap(J)
            P_min_eig = float(torch.linalg.eigvalsh(P).min())
            lmi_resid = float(torch.linalg.eigvalsh(J.T @ P @ J - P).max())
            norm_P = p_metric_norm(J, P)
            # Independent check. solve_dlyap targets J^T P J - P = -I exactly,
            # so lmi_resid < 0 is guaranteed whenever the series converged and
            # proves nothing on its own. This verifies the solver actually hit
            # its target, so a failed/truncated solve is caught rather than
            # silently certified.
            resid_I = float(torch.linalg.matrix_norm(
                J.T @ P @ J - P + torch.eye(J.shape[0], dtype=J.dtype,
                                            device=J.device), 2))
            solver_ok = bool(resid_I < 1e-6)
        else:
            P_min_eig = lmi_resid = resid_I = float("nan")
            norm_P = float("inf")
            solver_ok = False

        # Sampled reachable states. CAVEAT: random D2 does not find the
        # adversarial alignment in joint_jacobian()'s docstring, so this is
        # evidence, not a bound.
        g = torch.Generator(device="cpu").manual_seed(perturb_seed)
        Hc = Wh.shape[0]
        sampled = [sigma_J]
        for _ in range(32):
            d1 = torch.rand(Hc, generator=g, dtype=torch.float64)
            d2 = torch.rand(Hc, generator=g, dtype=torch.float64)
            Js = joint_jacobian_general(Wh.double().cpu(), Wc.double().cpu(),
                                        lam, d1, d2)
            sampled.append(float(torch.linalg.svdvals(Js)[0]))
        sampled_max = max(sampled)

    out = {
        "lambda":                        lam,
        "chart_A_sigma_Wh":              s_h,
        "chart_A_sigma_Wc":              s_c,
        "chart_A_bound_Wh":              cell.MARGIN_H,
        "chart_A_bound_Wc":              cell.MARGIN_C,
        "chart_A_holds":                 bool(s_h < cell.MARGIN_H + 1e-6 and
                                              s_c < cell.MARGIN_C + 1e-6),
        "chart_C_max_eig_WhTWh_minus_I": max_eig,
        "chart_C_lyapunov_holds":        bool(max_eig < 0),
    }
    out.update({
        # Chart D -- joint map. sigma_J is the Euclidean norm and may exceed 1
        # even when charts A/B/C all pass; that is expected, not a bug.
        "chart_D_sigma_J":               sigma_J,
        "chart_D_euclidean_contraction": bool(sigma_J < 1.0),
        "chart_D_rho_J":                 rho_J,
        "chart_D_asymptotically_stable": bool(rho_J < 1.0),
        "chart_D_P_min_eig":             P_min_eig,
        "chart_D_lmi_residual":          lmi_resid,
        # measured CONTRACTION RATE in the P-metric of the linearization at
        # the origin. This is a rate, not a passed test -- see below.
        "chart_D_norm_P":                norm_P,
        "chart_D_sampled_max_sigma":     sampled_max,
        # solver self-check: did the doubling series actually reach its target?
        "chart_D_dlyap_residual":        resid_I,
        "chart_D_dlyap_solver_ok":       solver_ok,
        # NOTE: joint_holds is EQUIVALENT to rho(J) < 1 and is not three
        # independent tests. solve_dlyap() targets J^T P J - P = -I, so once
        # rho(J) < 1 the series converges, lmi_resid < 0 follows by
        # construction, and that forces ||J||_P < 1. Reporting all three as
        # conjuncts would be one test wearing three hats -- the same defect
        # charts A/B/C had. The independent content is chart_D_dlyap_solver_ok
        # (the solver hit its target) and the RATE chart_D_norm_P.
        "chart_D_joint_holds":           bool(rho_J < 1.0 and solver_ok),
    })
    if perturbed is not None:
        out["chart_B_perturbed_sigma"] = perturbed
        out["chart_B_holds_under_N01"] = bool(perturbed < cell.MARGIN_H)
    return out


# ---------------------------------------------------------------------------
# Atlas equivalence: Chart A and Chart B describe the same cell
# ---------------------------------------------------------------------------

def equivalence_check(in_dim: int = 1, H: int = 32,
                      margin_H: float = 0.98, margin_C: float = 0.95,
                      batch: int = 4, seq: int = 20, tol: float = 1e-4,
                      seed: int = 0):
    torch.manual_seed(seed)
    a = RecursionChartA(in_dim, H, margin_H, margin_C)
    a.project_now()
    b = RecursionChartB(in_dim, H, margin_H, margin_C)

    Wh_a = a.W_h.weight.detach().clone()
    Wc_a = a.W_c.weight.detach().clone()
    with torch.no_grad():
        b.W_x.weight.copy_(a.W_x.weight)
        b.W_x.bias.copy_(a.W_x.bias)
        b.log_lambda.copy_(a.log_lambda)
    b.W_h_mat = lambda: Wh_a
    b.W_c_mat = lambda: Wc_a

    x = torch.randn(batch, seq, in_dim)
    with torch.no_grad():
        ya = a(x)
        yb = b(x)
    max_abs_diff = (ya - yb).abs().max().item()

    return {"max_abs_diff": max_abs_diff,
            "passes": bool(max_abs_diff < tol),
            "tol": tol, "shape": tuple(ya.shape)}


# ---------------------------------------------------------------------------
# Self-test (also lives as tests/test_recursion.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Recursion canonical self-test")
    print("=" * 70)

    eq = equivalence_check()
    print(f"\nequivalence Chart A vs Chart B:")
    print(f"  max_abs_diff = {eq['max_abs_diff']:.2e}   "
          f"passes < {eq['tol']}: {eq['passes']}")
    assert eq["passes"], "ATLAS BROKEN: Chart A and B disagree"

    cell = RecursionChartB(in_dim=3, H=64)
    aud = audit_certificates(cell)
    print(f"\nChart B at init:")
    for k, v in aud.items():
        print(f"  {k:<35} {v}")
    assert aud["chart_A_holds"], "Chart B init violates sigma bound"
    assert aud["chart_C_lyapunov_holds"], "Chart B init violates Lyapunov LMI"
    assert aud["chart_B_holds_under_N01"], "Chart B fails under N(0,1) perturbation"
    assert aud["chart_D_joint_holds"], "Chart B init violates the joint-map certificate"
    assert aud["chart_D_dlyap_solver_ok"], "dlyap solver did not reach its target at init"

    cell = RecursionChartB(in_dim=3, H=64)
    opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
    for k in range(50):
        x = torch.randn(8, 30, 3)
        h = cell(x)
        loss = (h ** 2).mean() + h.std()
        opt.zero_grad(); loss.backward(); opt.step()
    aud = audit_certificates(cell)
    print(f"\nChart B after 50 training steps:")
    for k, v in aud.items():
        print(f"  {k:<35} {v}")
    assert aud["chart_A_holds"], "Training broke sigma bound"
    assert aud["chart_C_lyapunov_holds"], "Training broke Lyapunov LMI"
    assert aud["chart_D_joint_holds"], "Training broke the joint-map certificate"
    assert aud["chart_D_dlyap_solver_ok"], "dlyap solver did not reach its target"

    # Charts A/B/C bound W_h and W_c. They pass while the JOINT map is not a
    # Euclidean contraction, so asserting only on them confirms the wrong
    # operator green. Chart D is the one that can actually fail here.
    if not aud["chart_D_euclidean_contraction"]:
        print(f"\n  note: sigma(J) = {aud['chart_D_sigma_J']:.4f} >= 1 -- the joint map is")
        print(f"        not a Euclidean contraction at this init. It contracts in the")
        print(f"        P-metric instead: ||J||_P = {aud['chart_D_norm_P']:.4f} < 1.")
        print(f"        This is expected. See SCOPE in the module docstring.")

    print("\nall recursion reference tests pass")
