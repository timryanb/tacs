"""
Regression test for Item 2's solve_flag surfacing on BucklingProblem.solve()
(SPEC.md, "## Item 2 -- solve_flag convergence-status plumbing"), repeating
test_modal_solve_flag_warning.py's pattern for BucklingProblem (PLAN Task
2.4: "either is acceptable, SPEC only requires 'repeat for
BucklingProblem'").

An unreachably tight ``L2Convergence``/``L2ConvergenceRel`` forces the
Lanczos loop to run to ``max_lanczos`` without ever satisfying
``checkConverged()``. ``plate_shear_buckle.bdf`` (this repo's buckling-test
convention mesh, e.g. ``test_shell_plate_buckling_shear.py``) needs
``numEigs=99`` (not 10) to force genuine, seed-independent non-convergence
within the default ``max_lanczos=100`` budget -- confirmed during
implementation that smaller ``numEigs`` values (5, 10, 50, 70, and even 90)
either converge cleanly at ``tol=1e-30`` on this mesh or are order-dependent
on prior ``rand()`` state from earlier tests in the same process (see
``test_unreachable_tolerance_reports_non_convergence``'s docstring below),
mirroring the same honestly-reported caveat found for the modal case
(test_modal_solve_flag_warning.py).

**Warning-assertion idiom.** Same as test_modal_solve_flag_warning.py:
``_TACSWarning`` prints directly (no ``warnings`` module), so stdout is
captured via ``contextlib.redirect_stdout`` rather than
``self.assertWarns(...)``, guarded by ``if problem.comm.rank == 0``.
"""

import contextlib
import io
import os
import unittest

from mpi4py import MPI

from tacs import pytacs, elements, constitutive

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_file = os.path.join(base_dir, "input_files/plate_shear_buckle.bdf")


def elem_call_back(
    dv_num, comp_id, comp_descript, elem_descripts, global_dvs, **kwargs
):
    rho = 2500.0
    E = 70e9
    nu = 0.33
    ys = 464.0e6
    tplate = 0.07
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    con = constitutive.IsoShellConstitutive(prop, t=tplate, tNum=dv_num)
    elem = elements.Quad4Shell(None, con)
    scale = [100.0]
    return elem, scale


class BucklingSolveFlagWarningTest(unittest.TestCase):
    N_PROCS = 2

    def _make_buckling_problem(self, sigma=10.0, num_eigs=99):
        comm = MPI.COMM_WORLD
        fea_assembler = pytacs.pyTACS(bdf_file, comm)
        fea_assembler.initialize(elem_call_back)
        return fea_assembler.createBucklingProblem("buckling", sigma, num_eigs)

    def test_raw_cython_layer_solve_returns_int(self):
        """
        Raw Cython layer: BucklingAnalysis.solve() now returns a Python int,
        independent of pyTACS's own surfacing.
        """
        problem = self._make_buckling_problem()
        problem._updateAssemblerVars()
        result = problem.buckleSolver.solve(print_flag=problem.getOption("printLevel"))
        self.assertIsInstance(result, int)
        self.assertIn(result, (-1, 0, 1))

    def test_get_variables_and_get_modal_error_out_of_range_index_raises(self):
        """
        SPEC.md Item 1 "Error handling" section: buckling.py's
        getVariables()/getModalError() (extractEigenvalue call sites) must
        raise ValueError, not silently return a bogus value, when the C++
        layer's out-of-range sentinel (error == -1.0) fires. Uses a
        normally-converged problem with a deliberately huge out-of-range
        index (150, safely beyond BucklingAnalysis's default
        max_lanczos=100 -- niters can never exceed max_iters regardless of
        mesh/config, so this is unconditionally out of range), not a
        request within numEigs (which the tridiagonal solve may already
        have incidentally computed at a converged iteration count larger
        than numEigs -- confirmed during implementation that on this mesh
        even index 50 is NOT out of range at num_eigs=5, niters converges
        past 50 -- and would not reliably reproduce error==-1.0).
        """
        problem = self._make_buckling_problem(num_eigs=5)
        problem.solve()
        with self.assertRaises(ValueError):
            problem.getVariables(150)
        with self.assertRaises(ValueError):
            problem.getModalError(150)

    def test_unreachable_tolerance_reports_non_convergence(self):
        """
        numEigs=99 (not VALIDATION's num_eigs=10): confirmed during
        implementation that num_eigs in {5, 10, 50, 70} on this mesh
        converge cleanly even at tol=1e-30 within the default
        max_lanczos=100 budget. numEigs=90 was tried first and is *not*
        robust: SEP::solve()'s starting vector is unseeded libc rand()
        state (SEP::SEP()/Q[0]->setRand()), so its outcome depends on how
        many prior rand() calls happened earlier in the same process --
        num_eigs=90 converged when this file was run in isolation but
        failed to converge (this test's intended RED/non-convergence case)
        when run after other eigensolver tests in the same pytest session,
        and vice versa in a complex-scalar build. Requesting num_eigs=99 --
        one short of the default max_lanczos=100 budget, leaving essentially
        no slack for the Krylov process to build a margin regardless of
        its random starting vector -- forces a genuine, seed-independent
        non-convergence within the iteration cap, without touching
        max_lanczos itself (not exposed by pytacs).
        """
        problem = self._make_buckling_problem(num_eigs=99)
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
