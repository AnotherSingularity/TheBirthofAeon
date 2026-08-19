"""tests/test_recursion.py — the recursion self-test from the spec, as pytest.

Mirrors aeon/recursion.py's __main__ self-test: atlas equivalence, the
certificate at init, and the certificate surviving training.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.recursion import (
    RecursionChartB,
    audit_certificates,
    equivalence_check,
)


def test_atlas_equivalence():
    eq = equivalence_check()
    assert eq["passes"], f"ATLAS BROKEN: max_abs_diff={eq['max_abs_diff']:.2e}"


def test_certificate_at_init():
    cell = RecursionChartB(in_dim=3, H=64)
    aud = audit_certificates(cell)
    assert aud["chart_A_holds"], "Chart B init violates sigma bound"
    assert aud["chart_C_lyapunov_holds"], "Chart B init violates Lyapunov LMI"
    assert aud["chart_B_holds_under_N01"], "Chart B fails under N(0,1) perturbation"


def test_certificate_survives_training():
    torch.manual_seed(0)
    cell = RecursionChartB(in_dim=3, H=64)
    opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
    for _ in range(50):
        x = torch.randn(8, 30, 3)
        h = cell(x)
        loss = (h ** 2).mean() + h.std()
        opt.zero_grad(); loss.backward(); opt.step()
    aud = audit_certificates(cell)
    assert aud["chart_A_holds"], "Training broke sigma bound"
    assert aud["chart_C_lyapunov_holds"], "Training broke Lyapunov LMI"


def test_origin_is_not_worst_case_over_d2():
    """
    Regression pin for the claim that the origin is the worst-case state.

    It is not. D1 (the outer tanh derivative) factors out on the left and can
    only reduce sigma, but D2 sits inside the block (W_h + lam*D2*W_c), where
    shrinking it can ENLARGE that block through cancellation.

    With W_h and W_c adversarially aligned, sigma is non-monotonic in d2 and
    the value at d2 -> 0 exceeds the value at the origin. If this test ever
    fails, the counterexample has been invalidated -- do not "fix" it by
    relaxing the assert; re-derive it.
    """
    import torch
    from aeon.recursion import RecursionChartB, joint_jacobian_general

    torch.manual_seed(0)
    H, lam = 32, 0.7
    cell = RecursionChartB(in_dim=4, H=H)
    Wc = cell.W_c_mat().detach().double()
    Wh = -1.2 * lam * Wc                       # adversarial alignment
    ones = torch.ones(H, dtype=torch.float64)

    def sigma(d2_scalar):
        d2 = d2_scalar * ones
        J = joint_jacobian_general(Wh, Wc, lam, ones, d2)
        return float(torch.linalg.svdvals(J)[0])

    s_origin = sigma(1.0)
    s_small = sigma(0.01)

    assert s_small > s_origin, (
        f"origin ({s_origin:.4f}) should NOT be the max; "
        f"d2=0.01 gives {s_small:.4f}"
    )
    assert sigma(0.5) < s_origin, "sigma should dip before rising (non-monotonic in d2)"
