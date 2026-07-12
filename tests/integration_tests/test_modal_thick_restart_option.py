"""
Tests for the pyTACS options layer of thick-restart Lanczos (SPEC.md,
"## Item 5 -- Thick-restart Lanczos (USER OVERRIDE)", PLAN Task 5.4).

Unlike test_gsep_thick_restart_agreement.py (which drives the raw
``tacs.TACS.SEPsolver``/``FrequencyAnalysis`` API directly), this file
exercises the ``useThickRestartLanczos``/``lanczosRestartSize`` options on
``ModalProblem`` and ``BucklingProblem`` themselves -- "thick-restart
benefits any Lanczos consumer, not just modal" (SPEC line 754) -- confirming
the pyTACS-level wiring (``_resolveLanczosRestartSize`` in
``tacs/problems/modal.py``, shared by both problem classes; threaded to the
raw ``FrequencyAnalysis``/``BucklingAnalysis`` constructors via their new
``restart_size`` keyword, `tacs/TACS.pyx`) end to end.

``lanczosRestartSize`` is set explicitly (not left at its 0/"pick a safe
default" setting) in the agreement tests below, sized with real headroom
above ``2*numEigs`` per this feature's "restart_size headroom" finding
(see test_gsep_thick_restart_agreement.py's module docstring) -- this
keeps these tests using the same well-verified regime as that file's own
committed tests, rather than separately re-verifying the default-resolution
formula's own headroom choice at length here.
"""

import os
import unittest

from mpi4py import MPI

from tacs import pytacs, elements, constitutive

base_dir = os.path.dirname(os.path.abspath(__file__))
modal_bdf_file = os.path.join(base_dir, "input_files/plate.bdf")
buckling_bdf_file = os.path.join(base_dir, "input_files/ss_plate.bdf")


def modal_elem_call_back(
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


def buckling_elem_call_back(
    dv_num, comp_id, comp_descript, elem_descripts, global_dvs, **kwargs
):
    rho = 2500.0
    E = 205e9
    nu = 0.3
    ys = 464.0e6
    tplate = 0.020
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    con = constitutive.IsoShellConstitutive(prop, t=tplate, tNum=dv_num)
    elem = elements.Quad4Shell(None, con)
    scale = [100.0]
    return elem, scale


class ModalThickRestartOptionTest(unittest.TestCase):
    N_PROCS = 1

    def test_modal_problem_restart_agrees_with_default(self):
        """
        ModalProblem with useThickRestartLanczos=True (lanczosRestartSize=25,
        giving keep=min(24,20)=20=2*numEigs with headroom=5 for numEigs=10 --
        this file's module docstring explains why headroom matters) must
        match the default (useThickRestartLanczos=False) run to machine
        precision.
        """
        comm = MPI.COMM_WORLD
        numEigs = 10
        sigma = 2e5

        fea_assembler = pytacs.pyTACS(modal_bdf_file, comm)
        fea_assembler.initialize(modal_elem_call_back)

        prob_default = fea_assembler.createModalProblem("modal_default", sigma, numEigs)
        prob_restart = fea_assembler.createModalProblem(
            "modal_restart",
            sigma,
            numEigs,
            options={"useThickRestartLanczos": True, "lanczosRestartSize": 25},
        )

        flag_default = prob_default.solve()
        flag_restart = prob_restart.solve()
        self.assertTrue(flag_default)
        self.assertTrue(flag_restart)

        eval_funcs = [f"eigsm.{i}" for i in range(numEigs)]
        funcs_default = {}
        funcs_restart = {}
        prob_default.evalFunctions(funcs_default, evalFuncs=eval_funcs)
        prob_restart.evalFunctions(funcs_restart, evalFuncs=eval_funcs)

        for i in range(numEigs):
            v_default = funcs_default[f"modal_default_eigsm.{i}"]
            v_restart = funcs_restart[f"modal_restart_eigsm.{i}"]
            rel = abs(v_restart - v_default) / abs(v_default)
            self.assertLess(
                rel,
                1e-9,
                msg=f"idx{i}: default={v_default!r} restart={v_restart!r} rel={rel:.3e}",
            )

    def test_modal_problem_proactive_guard(self):
        """
        Item 1's guard, applied proactively at the pyTACS option layer
        (SPEC): setting lanczosRestartSize <= numEigs with
        useThickRestartLanczos=True must raise before construction, not
        silently defer to the C++-level -1 return.
        """
        comm = MPI.COMM_WORLD
        numEigs = 10
        sigma = 2e5

        fea_assembler = pytacs.pyTACS(modal_bdf_file, comm)
        fea_assembler.initialize(modal_elem_call_back)

        with self.assertRaises(ValueError):
            fea_assembler.createModalProblem(
                "modal_bad_restart",
                sigma,
                numEigs,
                options={
                    "useThickRestartLanczos": True,
                    "lanczosRestartSize": numEigs,
                },
            )

    def test_buckling_problem_restart_agrees_with_default(self):
        """
        BucklingProblem with useThickRestartLanczos=True (same
        lanczosRestartSize=25/numEigs=10 configuration as the modal test
        above) must match the default run to machine precision -- confirms
        "thick-restart benefits any Lanczos consumer, not just modal"
        (SPEC line 754) is actually true of the pyTACS wiring, not just the
        raw SEP/GSEP.cpp layer.
        """
        comm = MPI.COMM_WORLD
        numEigs = 10
        sigma = 10.0

        fea_assembler = pytacs.pyTACS(buckling_bdf_file, comm)
        fea_assembler.initialize(buckling_elem_call_back)

        prob_default = fea_assembler.createBucklingProblem(
            "buckling_default", sigma, numEigs
        )
        prob_default.addLoadFromBDF(loadID=1)
        prob_default.setOption("L2Convergence", 1e-20)
        prob_default.setOption("L2ConvergenceRel", 1e-20)

        prob_restart = fea_assembler.createBucklingProblem(
            "buckling_restart",
            sigma,
            numEigs,
            options={"useThickRestartLanczos": True, "lanczosRestartSize": 25},
        )
        prob_restart.addLoadFromBDF(loadID=1)
        prob_restart.setOption("L2Convergence", 1e-20)
        prob_restart.setOption("L2ConvergenceRel", 1e-20)

        flag_default = prob_default.solve()
        flag_restart = prob_restart.solve()
        self.assertTrue(flag_default)
        self.assertTrue(flag_restart)

        eval_funcs = [f"eigsb.{i}" for i in range(numEigs)]
        funcs_default = {}
        funcs_restart = {}
        prob_default.evalFunctions(funcs_default, evalFuncs=eval_funcs)
        prob_restart.evalFunctions(funcs_restart, evalFuncs=eval_funcs)

        for i in range(numEigs):
            v_default = funcs_default[f"buckling_default_eigsb.{i}"]
            v_restart = funcs_restart[f"buckling_restart_eigsb.{i}"]
            rel = abs(v_restart - v_default) / abs(v_default)
            self.assertLess(
                rel,
                1e-9,
                msg=f"idx{i}: default={v_default!r} restart={v_restart!r} rel={rel:.3e}",
            )

    def test_buckling_problem_proactive_guard(self):
        """
        Same proactive guard as test_modal_problem_proactive_guard, for
        BucklingProblem.
        """
        comm = MPI.COMM_WORLD
        numEigs = 10
        sigma = 10.0

        fea_assembler = pytacs.pyTACS(buckling_bdf_file, comm)
        fea_assembler.initialize(buckling_elem_call_back)

        with self.assertRaises(ValueError):
            fea_assembler.createBucklingProblem(
                "buckling_bad_restart",
                sigma,
                numEigs,
                options={
                    "useThickRestartLanczos": True,
                    "lanczosRestartSize": numEigs,
                },
            )


if __name__ == "__main__":
    unittest.main()
