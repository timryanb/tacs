"""
Regression test for Item 6, Task 6.2b -- gated Jacobi-Davidson exposure on
``ModalProblem`` (SPEC.md, "## Item 6 -- JD pyTACS exposure").

Gate outcome this test exercises: **NO-GO**. Task 6.1's re-benchmark
(``docs/plans/feature-eigen-solver-perf/scripts/exp_c9_rerun_hardened.py``,
never committed -- see its header comment for the full recorded numbers)
found JD, even through the now-hardened path (Items 2/3), 6-9x slower than
Lanczos on every repeated run and converging only 20-80% of the time
(process-random-seed dependent) at the SPEC-recommended re-benchmark
budgets -- both criteria of SPEC's Go/No-Go gate (SPEC.md lines 916-925)
fail. Per SPEC lines 926-936 / PLAN.md Task 6.2b, the pyTACS-level option
therefore ships **double-gated**: ``eigenSolver="jacobi-davidson"`` alone
raises (hard stop); ``allowExperimentalEigenSolver=True`` must also be set,
which activates JD and fires an experimental-solver ``_TACSWarning``
regardless. Lanczos remains the default in both outcomes (non-negotiable,
SPEC line 938-940).

Expected RED state (written before Task 6.2b lands)
----------------------------------------------------
Today, ``ModalProblem.defaultOptions`` has no ``eigenSolver`` entry at all.
``setOption("eigenSolver", "jacobi-davidson")`` falls into ``BaseUI.
setOption``'s "unknown option" branch, which only ``_TACSWarning``s and
returns (it does not raise) -- so ``getOption("eigenSolver")`` then raises
``AttributeError`` (unknown option name), not the ``Error`` this test
expects from the *hard-stop* validation Task 6.2b adds. Similarly,
``allowExperimentalEigenSolver`` does not exist yet, so setting it raises
the same unknown-option warning-then-``AttributeError`` path. Both RED
failures are therefore ``AttributeError``, not the assertions this test
makes once Task 6.2b lands.

Test-recipe deviation, documented per this feature's established precedent
----------------------------------------------------------------------------
SPEC's own defaults for a JD-enabled ``ModalProblem`` (``maxJacobiDavidson
Size=20``, ``maxJDGMRESSize=30``, plus ``ModalProblem``'s own default
``L2Convergence``/``L2ConvergenceRel=1e-12``) were confirmed during
implementation to **never converge** for even the single fundamental mode
of ``plate.bdf`` (0/10 fresh-process trials converged at those exact
defaults) -- an even more severe version of the same pre-existing raw-JD
robustness gap ``test_jd_precond_autofactor.py``'s module docstring
documents ("this configuration is flaky, independent of the Item 3 fix").
This is exactly why the gate resolved NO-GO and is not a bug in this test's
setup. To keep the "does the opt-in mechanism itself work" assertion
deterministic (this test's actual purpose -- it is not a JD-robustness
regression test, that question is Item 6's own gate, already answered
above), the GREEN case below overrides ``maxJacobiDavidsonSize=60``,
``maxJDGMRESSize=20``, and loosens ``L2Convergence``/``L2ConvergenceRel``
to ``1e-8``, mirroring ``test_jd_precond_autofactor.py``'s own
independently-verified-robust recipe (12/12 complex, 10/10 real repeated
trials at that exact budget for ``num_eigs=1``, the same count used here).
"""

import contextlib
import io
import os
import unittest

from mpi4py import MPI

from tacs import pytacs, elements, constitutive
from tacs.utilities import Error as TACSError

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_file = os.path.join(base_dir, "input_files/plate.bdf")

# Reference fundamental-mode eigenvalue, test_shell_plate_quad.py's FUNC_REFS
# (same mesh, sigma=2e5).
LANCZOS_REF_MODE0 = 87437.50645925231
SIGMA = 2e5
NUM_EIGS = 1


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


class ModalJDSolverOptionTest(unittest.TestCase):
    N_PROCS = 1

    def _make_modal_problem(self, options=None):
        comm = MPI.COMM_WORLD
        fea_assembler = pytacs.pyTACS(bdf_file, comm)
        fea_assembler.initialize(elem_call_back)
        return fea_assembler.createModalProblem(
            "modal", SIGMA, NUM_EIGS, options=options
        )

    def test_lanczos_default_unaffected(self):
        """
        EigenSolver defaults to 'lanczos'; this item's changes must not
        perturb the existing, already-verified Lanczos code path at all.
        """
        problem = self._make_modal_problem()
        self.assertEqual(problem.getOption("eigenSolver"), "lanczos")
        self.assertFalse(problem.getOption("allowExperimentalEigenSolver"))

        success = problem.solve()
        self.assertIs(success, True)
        funcs = {}
        problem.evalFunctions(funcs)
        self.assertAlmostEqual(
            funcs["modal_eigsm.0"] / LANCZOS_REF_MODE0, 1.0, delta=1e-5
        )

    def test_jacobi_davidson_without_experimental_flag_raises(self):
        """
        eigenSolver='jacobi-davidson' alone (allowExperimentalEigenSolver
        left at its False default) is a hard construction-time stop, per
        the NO-GO gate outcome -- not merely a warning.
        """
        with self.assertRaises(TACSError):
            self._make_modal_problem(options={"eigenSolver": "jacobi-davidson"})

    def test_jacobi_davidson_with_experimental_flag_works_and_warns(self):
        """
        Setting both options together activates the (unendorsed, NO-GO)
        JD path -- it must still produce a working solve, matching the
        Lanczos reference eigenvalue -- and fire the experimental-solver
        warning documented in SPEC.md lines 931-932.
        """
        options = {
            "eigenSolver": "jacobi-davidson",
            "allowExperimentalEigenSolver": True,
            # See module docstring's "Test-recipe deviation" section for
            # why these override the (confirmed non-convergent at this
            # mesh/mode-count) library-native defaults.
            "maxJacobiDavidsonSize": 60,
            "maxJDGMRESSize": 20,
            "L2Convergence": 1e-8,
            "L2ConvergenceRel": 1e-8,
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            problem = self._make_modal_problem(options=options)

        if problem.comm.rank == 0:
            # NOTE: assert on the specific SPEC-quoted phrase, not just the
            # substring "experimental" -- confirmed during implementation
            # that a bare "experimental" substring check produces a
            # false-positive PASS even pre-implementation, because
            # BaseUI.setOption's unrelated "'allowexperimentaleigensolver'
            # is not a valid option" warning (fired for every option this
            # test sets, before Task 6.2b's defaultOptions entries exist)
            # itself contains "experimental" as a substring of the option
            # NAME -- not evidence of the actual activation warning firing.
            self.assertIn("unvalidated-for-general-use", buf.getvalue().lower())

        success = problem.solve()
        self.assertIs(success, True)
        funcs = {}
        problem.evalFunctions(funcs)
        self.assertAlmostEqual(
            funcs["modal_eigsm.0"] / LANCZOS_REF_MODE0, 1.0, delta=1e-5
        )


if __name__ == "__main__":
    unittest.main()
