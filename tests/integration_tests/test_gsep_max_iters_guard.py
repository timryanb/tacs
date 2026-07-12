"""
Regression test for the Item 1 segfault guard (SPEC.md, "## Item 1 -- Segfault
guard in SEP::extractEigenvalue/extractEigenvector").

Root cause (VALIDATION Claim 4, SPEC.md lines 31-50): ``SEP::SEP()`` allocates
``eigs``/``eigvecs``/``perm`` off ``max_iters`` with no initialization; if the
caller's ``max_iters < neigvals`` (SEP) / ``max_jd_size < max_eigen_vectors``
(JD), ``SEP::checkConverged`` never runs, ``niters``/``neigs_computed`` end up
inconsistent, and a direct ``extractEigenvalue`` call reads an uninitialized
``perm[n]`` used as an index -- a deterministic out-of-bounds heap read
(confirmed to segfault in VALIDATION's crash-boundary sweep,
``exp_c4b_crash_boundary.py``, this file's Lanczos test mirrors that script).

Expected RED state (written before Items 1.1-1.3 land)
-------------------------------------------------------
Before the fix, calling ``.solve()`` with ``max_lanczos < num_eigs`` and then
``.extractEigenvalue(0)`` segfaults the interpreter -- a hard process crash,
not a clean pytest assertion failure. A crashed test process cannot itself
report a normal FAIL, so the RED state for this file is documented here
rather than asserted: running
``mpirun -n 1 python3 -m pytest tests/integration_tests/test_gsep_max_iters_guard.py -v``
against pre-1.2/1.3 source terminates with a non-zero/segfault exit rather
than a pytest report at all. This is consistent with SPEC's crash-boundary
table (VALIDATION ``exp_c4b_crash_boundary.py``, ``max_lanczos`` in
``{3, 5, 9}`` against ``num_eigs=10``).

Task 1.1 alone (the ``neigs_computed`` bookkeeping member) does not remove
the crash -- the entry guard (Task 1.2) and tightened bounds check (Task 1.3)
are what stop it; this file's tests are only expected to pass once 1.1+1.2+1.3
have all landed (implemented together with Item 2's ``void`` to ``int``
signature change per PLAN's merge note, since the ``-1`` sentinel only means
something once ``solve()`` returns ``int``).
"""

import os
import unittest

from mpi4py import MPI

from tacs import pytacs, elements, constitutive, TACS

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


class GSEPModalMaxItersGuardTest(unittest.TestCase):
    # NOTE: named to include "Modal" (this file's tests are built on
    # tacs.TACS.FrequencyAnalysis, the modal-analysis raw layer) so that
    # `pytest tests/integration_tests/ -k "modal or buckling"` (this
    # feature's PLAN Task 6.4 regression sweep) selects these
    # segfault-guard tests too -- confirmed during implementation via
    # `--collect-only` that the un-renamed class name matched neither
    # keyword and was silently skipped by that sweep.
    N_PROCS = 1

    def _make_modal_problem(self, sigma, num_eigs):
        comm = MPI.COMM_WORLD
        fea_assembler = pytacs.pyTACS(bdf_file, comm)
        fea_assembler.initialize(elem_call_back)
        prob = fea_assembler.createModalProblem("modal", sigma, num_eigs)
        prob.assembler.assembleMatType(TACS.STIFFNESS_MATRIX, prob.K)
        prob.assembler.assembleMatType(TACS.MASS_MATRIX, prob.M)
        return prob

    def test_lanczos_max_iters_below_num_eigs_does_not_crash(self):
        """
        Raw tacs.TACS.FrequencyAnalysis, Lanczos branch (a real KSM solver
        object is passed), with max_lanczos (3) < num_eigs (10) -- mirrors
        VALIDATION's exp_c4b_crash_boundary.py crash-boundary sweep.
        """
        sigma = 2e5
        num_eigs = 10
        max_lanczos = 3

        prob = self._make_modal_problem(sigma, num_eigs)
        freq = TACS.FrequencyAnalysis(
            prob.assembler,
            sigma,
            prob.M,
            prob.K,
            prob.gmres,
            max_lanczos=max_lanczos,
            num_eigs=num_eigs,
            eig_tol=1e-12,
        )

        # NOTE: solve()'s return value is intentionally not asserted before
        # the extractEigenvalue() call below -- pre-fix, this exact call
        # ordering (solve() returns cleanly; the crash is inside
        # extractEigenvalue's uninitialized-perm[n] read, not inside solve()
        # itself) is what reproduces VALIDATION's documented segfault. Both
        # assertions are checked together only after the crash-prone call
        # returns successfully.
        solve_flag = freq.solve(print_flag=False)
        eigval, error = freq.extractEigenvalue(0)

        self.assertEqual(solve_flag, -1)
        self.assertEqual(error, -1.0)

    def test_jd_max_jd_size_below_num_eigenvalues_does_not_crash(self):
        """
        Raw tacs.TACS.FrequencyAnalysis, JD branch (solver=None), with
        max_jd_size (passed positionally as max_lanczos=3, per the Cython
        parameter-name reuse documented in SPEC/PLAN Task 6.2a) < num_eigs
        (10) -- mirrors VALIDATION Claim 5's raw JD construction pattern.
        JD's extractEigenvalue does not have SEP's OOB-read bug (its bounds
        check is already `n < nconverged`, itself bounded by array sizes
        allocated off max_jd_size/max_eigen_vectors), so this is a
        fail-fast/consistency addition, not an independent crash fix. Per
        PLAN Task 1.2, only `.solve() == -1` and "no crash" are asserted
        here -- unlike SEP, a post-guard extractEigenvalue(0) call on JD
        does NOT reliably return error == -1.0: n=0 is still < the
        (guard-independent) max_eigen_vectors budget, so JD's second
        ("not yet converged, in-progress Ritz estimate") branch is taken
        instead of its out-of-range branch, returning a meaningless-but-
        not-crashing value from never-populated Ritz arrays. This is
        confirmed-safe (no OOB memory access, per SPEC lines 126-146) but
        not the same clean -1.0 contract SEP's tightened bounds check (Task
        1.3) guarantees.
        """
        sigma = 2e5
        num_eigs = 10
        max_jd_size = 3

        prob = self._make_modal_problem(sigma, num_eigs)
        pcmat = prob.assembler.createSchurMat()
        prob.assembler.assembleMatType(TACS.STIFFNESS_MATRIX, pcmat)
        pc = TACS.Pc(pcmat)

        jd_freq = TACS.FrequencyAnalysis(
            prob.assembler,
            sigma,
            prob.M,
            prob.K,
            None,
            PC=pcmat,
            pc=pc,
            max_lanczos=max_jd_size,
            fgmres_size=15,
            num_eigs=num_eigs,
            eig_tol=1e-9,
            eig_rtol=1e-9,
            eig_atol=1e-30,
        )

        solve_flag = jd_freq.solve(print_flag=False)
        # Does not crash -- the return value itself is not asserted, see
        # the docstring above.
        jd_freq.extractEigenvalue(0)

        self.assertEqual(solve_flag, -1)

    def test_extract_eigenvalue_beyond_neigs_computed_does_not_crash(self):
        """
        Task 1.3 defense-in-depth: a normally-configured solve
        (max_lanczos=100 > num_eigs=10) followed by a direct
        extractEigenvalue(50) call -- an index that ran but was never
        computed/converged to. Per SPEC.md lines 116-124, this specific case
        might already have passed against the pre-1.3 `n >= niters` check
        (defense-in-depth, not guaranteed to independently RED); the tightened
        `n >= neigs_computed` check makes it robust regardless of niters'
        value.
        """
        sigma = 2e5
        num_eigs = 10
        max_lanczos = 100

        prob = self._make_modal_problem(sigma, num_eigs)
        freq = TACS.FrequencyAnalysis(
            prob.assembler,
            sigma,
            prob.M,
            prob.K,
            prob.gmres,
            max_lanczos=max_lanczos,
            num_eigs=num_eigs,
            eig_tol=1e-12,
        )

        solve_flag = freq.solve(print_flag=False)
        eigval, error = freq.extractEigenvalue(50)

        self.assertIn(solve_flag, (0, 1))
        self.assertEqual(error, -1.0)
        self.assertEqual(eigval, 0.0)


if __name__ == "__main__":
    unittest.main()
