"""
Regression test for the Item 3 fix ("## Item 3 -- JD preconditioner
auto-factor" in SPEC.md).

Root cause (VALIDATION Claim 5, SPEC.md lines 503-533): the non-mg branch of
``TACSFrequencyAnalysis::solve()`` (``TACSBuckling.cpp``, ``if (jd) {...}``
non-mg ``else``) assembles ``mmat``/``kmat`` but never calls
``jd_op->setEigenvalueEstimate()`` -- the *only* code path that builds
``pc_mat`` and factors the JD preconditioner for the non-mg case. A caller
who constructs the raw JD branch of ``tacs.TACS.FrequencyAnalysis`` and never
calls ``setSigma()`` first (the JD constructor itself does not call it
either) runs FGMRES against an unfactored/garbage preconditioner from
iteration 1, producing ``nan`` Ritz values -- reproduced directly by
VALIDATION's ``exp_c5_jd_precond_gap.py``.

Expected RED state (written before Task 3.1 lands)
---------------------------------------------------
Before the fix: ``solve()`` returns ``0`` (not ``1``) -- Item 2's
finiteness gate (``JacobiDavidson.cpp``'s post-harvest-loop check) already
catches the resulting NaN Ritz values and reports non-convergence rather
than a silent bogus ``1`` -- and ``extractEigenvalue(0)`` returns ``nan``,
which fails the Lanczos-reference comparison.

Scope note -- why num_eigs=1, not a multi-mode sweep
-----------------------------------------------------
SPEC's own verification plan for Item 3 only requires exercising *that the
preconditioner gets auto-factored*, not general JD convergence robustness
for multiple/near-degenerate modes -- that broader robustness question is
explicitly Item 6's concern (SPEC's Item 6 GO criterion specifically calls
out "correctly resolving the degenerate pair at modes 1/2", the exact
VALIDATION Claim 9 failure mode this item does not attempt to fix).
``plate.bdf``'s modes 1/2 are a near-degenerate pair
(``396969.8881662692``/``396969.8881667623``, ~5e-10 relative split, per
``test_shell_plate_quad.py``'s ``FUNC_REFS``); confirmed during
implementation that requesting 2+ modes makes raw JD's convergence flaky
(observed ~20% non-convergence rate across repeated runs at generous
iteration budgets) *independent of this fix* -- reproduced identically via
the pre-existing "call setSigma() explicitly first" control path, so this
is pre-existing raw-JD sensitivity to its unseeded random starting vector
(SPEC/PLAN's own precedent: ``JacobiDavidson.cpp``'s ``V[0]->setRand(-1.0,
1.0)``, same class of flakiness already documented and worked around for
SEP in ``test_modal_solve_flag_warning.py``'s num_eigs=99 fix), not a
regression introduced here. Restricting this test to num_eigs=1 (the
well-separated fundamental mode) isolates "does solve() auto-factor the
preconditioner" from "is raw JD convergence itself robust for clustered
spectra," keeping this test deterministic across both real and complex
builds.

Also confirmed during implementation: the originally-planned
max_lanczos=20 (i.e. max_jd_size, per the Cython positional-parameter
reuse), eig_tol=eig_rtol=1e-9 budget from VALIDATION's own
exp_c5_jd_precond_gap.py recipe was itself flaky for this same reason
(non-deterministic across the random starting vector, worse under the
complex-mode build's different floating-point rounding trajectory) --
widened to max_lanczos=60, fgmres_size=20, eig_tol=eig_rtol=1e-8, verified
robust across 12/12 repeated runs in complex mode and 10/10 in real mode
before landing this test.

Reference eigenvalue is ``test_shell_plate_quad.py``'s ``FUNC_REFS``
``modal_eigsm.0`` (sigma=2e5, same ``plate.bdf`` mesh), compared at
rtol=1e-5 per SPEC's verification plan (VALIDATION's own control-run
tolerance).
"""

import os
import unittest

import numpy as np
from mpi4py import MPI

from tacs import pytacs, elements, constitutive, TACS

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_file = os.path.join(base_dir, "input_files/plate.bdf")

# Reference fundamental-mode eigenvalue from test_shell_plate_quad.py's
# FUNC_REFS (same mesh, sigma=2e5).
LANCZOS_REF_MODE0 = 87437.50645925231


def elem_call_back(
    dv_num, comp_id, comp_descript, elem_descripts, global_dvs, **kwargs
):
    rho = 2500.0
    E = 70e9
    nu = 0.3
    ys = 464.0e6
    tplate = 0.005
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    con = constitutive.IsoShellConstitutive(prop, t=tplate, tNum=dv_num)
    elem = elements.Quad4Shell(None, con)
    scale = [100.0]
    return elem, scale


class JDPrecondAutoFactorTest(unittest.TestCase):
    # NOTE: named to include "modal" is not necessary for Item 3's own gate
    # (SPEC notes no existing pyTACS test reaches the JD branch, so there is
    # no risk of double-counting in the Task 6.4 "-k modal or buckling"
    # sweep) -- left un-suffixed deliberately per SPEC.md lines 624-627.
    N_PROCS = 1

    def _make_modal_problem(self, sigma, num_eigs):
        comm = MPI.COMM_WORLD
        fea_assembler = pytacs.pyTACS(bdf_file, comm)
        fea_assembler.initialize(elem_call_back)
        prob = fea_assembler.createModalProblem("modal", sigma, num_eigs)
        prob.assembler.assembleMatType(TACS.STIFFNESS_MATRIX, prob.K)
        prob.assembler.assembleMatType(TACS.MASS_MATRIX, prob.M)
        return prob

    def test_jd_solve_converges_without_explicit_setSigma(self):
        """
        Raw tacs.TACS.FrequencyAnalysis, JD branch (solver=None), sigma=2e5,
        num_eigs=1 (see module docstring for why not more) -- deliberately
        does NOT call setSigma()/setEigenvalueEstimate() before solve(),
        mirroring VALIDATION's exp_c5_jd_precond_gap.py primary (non-control)
        case. Pre-fix this reproduces the nan Ritz-value gap; post-fix,
        solve() auto-factors the preconditioner every call and converges
        cleanly.
        """
        sigma = 2e5
        num_eigs = 1

        prob = self._make_modal_problem(sigma, num_eigs)
        pcmat = prob.assembler.createSchurMat()
        # A plausible starting preconditioner matrix (unshifted K) -- the
        # point of this test is that solve() itself must factor this,
        # not that the caller pre-factors it.
        prob.assembler.assembleMatType(TACS.STIFFNESS_MATRIX, pcmat)
        pc = TACS.Pc(pcmat)

        jd_freq = TACS.FrequencyAnalysis(
            prob.assembler,
            sigma,
            prob.M,
            prob.K,
            None,  # solver=None -> JD branch
            PC=pcmat,
            pc=pc,
            max_lanczos=60,  # positionally max_jd_size, per the Cython gotcha
            fgmres_size=20,
            num_eigs=num_eigs,
            eig_tol=1e-8,
            eig_rtol=1e-8,
            eig_atol=1e-30,
        )

        # Deliberately NOT calling jd_freq.setSigma(sigma) here -- that is
        # exactly the gap this fix closes: solve() itself must now trigger
        # the preconditioner factorization.
        solve_flag = jd_freq.solve(print_flag=False)

        self.assertEqual(solve_flag, 1)
        eigval, _err = jd_freq.extractEigenvalue(0)
        self.assertTrue(
            np.isfinite(eigval), msg=f"mode 0 eigenvalue not finite: {eigval}"
        )
        self.assertAlmostEqual(
            eigval / LANCZOS_REF_MODE0,
            1.0,
            delta=1e-5,
            msg=f"mode 0: {eigval} vs ref {LANCZOS_REF_MODE0}",
        )


if __name__ == "__main__":
    unittest.main()
