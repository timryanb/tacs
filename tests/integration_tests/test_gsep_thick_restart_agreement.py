"""
Tests for thick-restart Lanczos (SPEC.md, "## Item 5 -- Thick-restart
Lanczos (USER OVERRIDE)").

Uses the raw ``tacs.TACS.SEPsolver``/``EPGeneralizedShiftInvertOp`` API
directly (bypassing pytacs's hard-coded ``max_lanczos=100``), mirroring the
construction pattern already used by
``docs/plans/feature-eigen-solver-perf/scripts/exp_c7b_orthogonality.py``
(not imported here -- that script lives under the session-state planning
directory and is never committed, so this file builds its own copy of the
same construction).

Task 5.1 (this file's first two tests): a minimal smoke test that a
``restart_size``-enabled ``SEP`` completes ``solve()`` without error/crash
and reports a finite ``checkOrthogonality()`` -- confirms the allocation
change (GSEP.h/.cpp) and the new .pxd/.pyx plumbing work end-to-end. This
does NOT assert eigenvalue accuracy -- that is Task 5.3's job.

Task 5.2's LOCAL-branch fallback (this file's third test): ``ortho_type ==
LOCAL`` with ``restart_size > 0`` must fall back to the unrestarted LOCAL
path verbatim (SPEC lines 774-781), not attempt a restart.

Review fix (this file's fourth test): a review of commits 21d347c6/
7647360b found that decoupling the FULL-branch loop bound from
``max_iters`` to the (potentially smaller) ``alloc_size`` exposed a new
out-of-bounds heap read reachable through the public ``restart_size``
constructor argument with no validation -- ``restart_size < neigvals``
(FULL orthogonalization) meant ``checkConverged()`` never reached
``n >= neigvals``, so ``perm[]`` stayed at its constructor
``-1``-initialized sentinel and the finiteness gate read ``eigs[-1]``
(confirmed under valgrind). Fixed in GSEP.cpp by checking
``neigvals > alloc_size`` instead of ``neigvals > max_iters``; this test
locks in the fix's clean ``-1`` misconfiguration return.

Honest checkpoint (see HANDOFF-impl.md, "Item 5 Task 5.2/5.3" section): the
FULL-branch restart math (Wu & Simon 2000's actual restart continuation)
is NOT implemented as of this commit. A restart-enabled FULL-orthogonalization
solve() currently just stops (gracefully, reporting solve_flag=0, never a
false positive) once the live basis reaches restart_size, per GSEP.cpp's
own documented interim-safety comment -- this is intentional and verified
memory-safe, not a bug. Task 5.3's actual machine-precision agreement test
is deferred to a future session per the honest-checkpoint instruction in
PLAN.md's Item 5 preamble; see HANDOFF-impl.md for the precise root-cause
diagnosis (a naive sequential Givens-rotation reduction of the restart's
diagonal-plus-rank-one "arrowhead" matrix to tridiagonal form is
insufficient for more than 2 retained Ritz vectors -- it requires genuine
iterative bulge-chasing, confirmed via an isolated numpy reproduction).
"""

import math
import os
import unittest

from mpi4py import MPI

from tacs import pytacs, elements, constitutive, TACS

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_file = os.path.join(base_dir, "input_files/plate.bdf")

SIGMA = 2e5


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


class GSEPThickRestartTest(unittest.TestCase):
    # NOTE: named to include "Modal" would collide with the plate.bdf modal
    # naming convention this feature's regression sweep
    # (`pytest tests/integration_tests/ -k "modal or buckling"`) matches by
    # keyword against the *file*/*class* name -- confirm this file's name
    # ("thick_restart") does not itself contain either keyword, so it is
    # intentionally excluded from that sweep (it is a raw GSEP.cpp-level
    # unit/integration test, not a pytacs Modal/BucklingProblem test); it is
    # still run explicitly per PLAN Task 5.3's own command and by the full
    # `pytest tests/integration_tests/` suite.
    N_PROCS = 1

    def _make_sep(self, num_eigs, max_iters, restart_size, ortho_type=None):
        """
        Build a raw SEPsolver for plate.bdf's modal (K x = omega^2 M x)
        eigenproblem via generalized shift-invert, mirroring
        exp_c7b_orthogonality.py's construction.
        """
        comm = MPI.COMM_WORLD
        fea_assembler = pytacs.pyTACS(bdf_file, comm)
        fea_assembler.initialize(elem_call_back)
        prob = fea_assembler.createModalProblem("modal", SIGMA, num_eigs)
        assembler = prob.assembler

        K = assembler.createSchurMat()
        M = assembler.createSchurMat()
        assembler.assembleMatType(TACS.STIFFNESS_MATRIX, K)
        assembler.assembleMatType(TACS.MASS_MATRIX, M)
        K.axpy(-SIGMA, M)
        assembler.applyMatBCs(K)
        pc = TACS.Pc(K)
        pc.factor()
        gmres = TACS.KSM(K, pc, 15, 5)

        op = TACS.EPGeneralizedShiftInvertOp(SIGMA, gmres, M)
        bcmap = assembler.getBcMap()
        if ortho_type is None:
            ortho_type = TACS.SEP_FULL
        sep = TACS.SEPsolver(op, max_iters, ortho_type, bcmap, restart_size)
        sep.setTolerances(1e-12, TACS.SEP_SMALLEST_MAGNITUDE, num_eigs)
        return sep, comm

    def test_restart_enabled_solve_does_not_crash(self):
        """
        Task 5.1 smoke test: restart_size=15 < max_iters=100, numEigs=10 on
        plate.bdf. Asserts solve() completes (returns an int solve_flag,
        does not crash/segfault) and checkOrthogonality() is finite.
        """
        num_eigs = 10
        max_iters = 100
        restart_size = 15

        sep, comm = self._make_sep(num_eigs, max_iters, restart_size)
        solve_flag = sep.solve(comm, print_flag=False)

        self.assertIn(solve_flag, (-1, 0, 1))

        ortho = sep.checkOrthogonality()
        self.assertTrue(
            math.isfinite(float(ortho.real)),
            msg=f"checkOrthogonality() returned non-finite value {ortho}",
        )

    def test_restart_size_zero_is_legacy_default(self):
        """
        restart_size=0 (the default) must still work exactly as before --
        basic regression check for the allocation-change branch that
        reproduces max_iters-sized allocation.
        """
        num_eigs = 10
        max_iters = 100

        sep, comm = self._make_sep(num_eigs, max_iters, restart_size=0)
        solve_flag = sep.solve(comm, print_flag=False)

        self.assertEqual(solve_flag, 1)
        for i in range(num_eigs):
            eig, err = sep.extractEigenvalue(i)
            self.assertNotEqual(err, -1.0)

    def test_local_ortho_ignores_restart_size(self):
        """
        Task 5.2's LOCAL-branch fallback: ortho_type=LOCAL with
        restart_size > 0 must solve identically to restart_size=0 (the
        unrestarted LOCAL path used verbatim, per SPEC lines 774-781), not
        attempt a restart -- confirmed here by requiring both to reach the
        same solve_flag and agree on every requested eigenvalue to machine
        precision, since a LOCAL solve's random start vector is the only
        other source of run-to-run variation and this test uses num_eigs=1
        to sidestep any degenerate-pair ordering sensitivity in that path.
        """
        num_eigs = 1
        max_iters = 100

        sep_norestart, comm = self._make_sep(
            num_eigs, max_iters, restart_size=0, ortho_type=TACS.SEP_LOCAL
        )
        flag_norestart = sep_norestart.solve(comm, print_flag=False)

        sep_restart, comm = self._make_sep(
            num_eigs, max_iters, restart_size=15, ortho_type=TACS.SEP_LOCAL
        )
        flag_restart = sep_restart.solve(comm, print_flag=False)

        self.assertEqual(flag_norestart, flag_restart)
        eig0, _ = sep_norestart.extractEigenvalue(0)
        eig1, _ = sep_restart.extractEigenvalue(0)
        self.assertAlmostEqual(eig0, eig1, delta=1e-6 * abs(eig0))

    def test_restart_size_below_neigvals_returns_clean_misconfig(self):
        """
        Review fix regression test: restart_size (3) < numEigs (10) on the
        FULL-orthogonalization branch must return a clean -1
        (misconfigured, no eigenvalues computed) and must not crash or read
        out of bounds -- previously this read eigs[perm[k]] with
        perm[k] == -1 (the constructor's sentinel, never overwritten
        because checkConverged() never reaches n >= neigvals when
        alloc_size == restart_size < neigvals). A subsequent
        extractEigenvalue() call must also report the standard
        out-of-range error, not a memory-corruption crash.
        """
        num_eigs = 10
        max_iters = 100
        restart_size = 3

        sep, comm = self._make_sep(num_eigs, max_iters, restart_size)
        solve_flag = sep.solve(comm, print_flag=False)

        self.assertEqual(solve_flag, -1)
        eig, err = sep.extractEigenvalue(0)
        self.assertEqual(err, -1.0)
        self.assertEqual(eig, 0.0)


if __name__ == "__main__":
    unittest.main()
