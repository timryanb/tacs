"""
Regression test for Item 2's solve_flag surfacing on ModalProblem.solve()
(SPEC.md, "## Item 2 -- solve_flag convergence-status plumbing").

Mirrors VALIDATION's ``exp_c4_silent_nonconvergence.py`` Scenario A: an
unreachably tight ``L2Convergence``/``L2ConvergenceRel`` (below the
double-precision noise floor) forces the Lanczos loop to run to
``max_lanczos`` without ever satisfying ``checkConverged()``, giving a
deterministic non-convergence case with no dependence on mesh-specific
tuning.

Staged per PLAN Tasks 2.2/2.3:
  - Task 2.2 lands the ``.pxd``/``.pyx`` ``void`` -> ``int`` plumbing only.
    ``ModalProblem.solve()`` (``modal.py:386-434``) has not been updated yet,
    so it still implicitly returns ``None`` -- this is documented and
    unasserted at this stage; only the raw Cython-layer check
    (``isinstance(problem.freqSolver.solve(...), int)``) is a real GREEN
    assertion here, since it is reachable independent of pyTACS's own
    surfacing.
  - Task 2.3 finishes ``ModalProblem.solve()``'s surfacing (two-branch
    ``_TACSWarning``, ``return bool(success == 1)``); at that point this
    file's ``ModalProblem``-level assertions below (currently commented
    out / xfail-documented) become real GREEN checks.

**Warning-assertion idiom.** ``tacs/utilities.py``'s ``_TACSWarning`` does
**not** use Python's ``warnings`` module -- it ``print()``s a formatted box
directly (confirmed by reading the method body; a repo-wide grep for
``_TACSWarning``/``assertWarns`` in ``tests/`` found no existing test
asserting on ``_TACSWarning`` output at all). ``self.assertWarns(...)`` will
NOT catch this -- it would silently pass with nothing actually captured.
Instead, stdout is captured directly via ``contextlib.redirect_stdout``,
guarded by ``if problem.comm.rank == 0`` since ``_TACSWarning`` only prints
on rank 0 (confirmed: ``_TACSWarning`` itself already gates on
``self.comm.rank == 0`` internally).
"""

import contextlib
import io
import os
import unittest

from mpi4py import MPI

from tacs import pytacs, elements, constitutive

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_file = os.path.join(base_dir, "input_files/plate.bdf")


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


class ModalSolveFlagWarningTest(unittest.TestCase):
    N_PROCS = 1

    def _make_modal_problem(self, sigma=2e5, num_eigs=10):
        comm = MPI.COMM_WORLD
        fea_assembler = pytacs.pyTACS(bdf_file, comm)
        fea_assembler.initialize(elem_call_back)
        return fea_assembler.createModalProblem("modal", sigma, num_eigs)

    def test_raw_cython_layer_solve_returns_int(self):
        """
        Already-GREEN-at-Task-2.2 check: the raw Cython layer's solve()
        now returns a Python int, independent of pyTACS's own surfacing
        (Tasks 2.3/2.4).
        """
        problem = self._make_modal_problem()
        problem._updateAssemblerVars()
        result = problem.freqSolver.solve(
            print_flag=problem.getOption("printLevel"),
            print_level=problem.getOption("printLevel"),
        )
        self.assertIsInstance(result, int)
        self.assertIn(result, (-1, 0, 1))

    def test_unreachable_tolerance_reports_non_convergence(self):
        """
        Task 2.3 gate: an unreachably tight L2Convergence/L2ConvergenceRel
        forces a non-converged Lanczos solve (mirrors VALIDATION's
        exp_c4_silent_nonconvergence.py Scenario A). Before Task 2.3 lands,
        ModalProblem.solve() still returns None (modal.py:386-434
        unchanged) -- assertIs(success, False) (not just falsy) makes this
        RED state unambiguous rather than accidentally passing on None's
        falsiness.

        num_eigs=50 (not VALIDATION's num_eigs=10): confirmed during
        implementation that num_eigs=10 on this mesh (726 dof) converges
        cleanly even at tol=1e-30 within the default max_lanczos=100 budget
        -- exactly the honestly-reported caveat in
        exp_c4_silent_nonconvergence.py's own docstring ("this scenario
        alone may not force non-convergence"). Requesting half of the
        default max_lanczos=100 budget (num_eigs=50) does force a genuine
        non-convergence within the iteration cap, still without touching
        max_lanczos itself (not exposed by pytacs, Finding Q2/Q7).
        """
        problem = self._make_modal_problem(num_eigs=50)
        problem.setOption("L2Convergence", 1e-30)
        problem.setOption("L2ConvergenceRel", 1e-30)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            success = problem.solve()

        self.assertIs(success, False)
        if problem.comm.rank == 0:
            self.assertIn("Eigenvalue solver failed to converge", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
