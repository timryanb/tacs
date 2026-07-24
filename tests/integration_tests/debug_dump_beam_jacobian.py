"""
Temporary diagnostic for docs/plans/bugfix-trans-beam-macos-blowup.

Part 1: dumps the assembled global tangent matrix (alpha*K + beta*C +
gamma*M) for the transient_beam.bdf / Beam2 model, bit-exactly (as hex
floats), for the two distinct (alpha, beta, gamma) coefficient triples the
real 20-step, dt=0.1 BDF-2 transient run actually uses at Newton iteration 0
of each step (order-1 ramp at step 1, steady-state order-2 BDF for steps
2-20 on this uniform grid) -- hand-derived from
TACSBDFIntegrator::get2ndBDFCoeff. TACSBeamLinearModel's addJacobian output
does not depend on the current state (vars/dvars/ddvars) at all -- only on
Xpts, material properties, and (alpha, beta, gamma) -- so this reproduces
exactly the numeric computation the real transient performs at each step's
first Newton iteration, without needing to run the transient loop itself.
Ran once already (Experiment 5): bit-identical across Real-MacOS /
Real-MacOS-NoFPContract / Real-Ubuntu -- rules out Jacobian *assembly* as
H5's locus.

Part 2: dumps the assembled residual vector (assembleRes, no alpha/beta/
gamma needed -- the residual is evaluated at the current state, not
linearized) at a fixed, deterministic, nonzero state
(vars[i]=0.001*(i+1), dvars[i]=0.002*(i+1), ddvars[i]=0.003*(i+1)) -- unlike
the Jacobian, addResidual *does* depend on state, so this is the next
narrowing step per Experiment 5's conclusion (residual assembly vs. the
linear solve/factorization are the two remaining state-dependent stages).

Run in serial (1 rank). The dense matrix is extracted column-by-column via
Mat.mult() against unit probe vectors rather than Mat.getDenseMatrix()
(avoids the Schur local-map reorder path -- a less-exercised binding -- in
case that's a confound).

Every stage prints a marker with flush=True and the whole body runs under a
top-level try/except that prints a full traceback to stdout, so a first
attempt that dies silently on one platform (no Python-level output at all)
localizes to a stage on retry. Intended to be run identically on
Real-Ubuntu / Real-MacOS / Real-MacOS-NoFPContract CI jobs and diffed.
Remove once the investigation closes.
"""

import os
import sys
import traceback

print("DIAG: script start", flush=True)

import numpy as np
from mpi4py import MPI

print("DIAG: imported numpy, mpi4py", flush=True)

from tacs import constitutive, elements, pytacs

print("DIAG: imported tacs", flush=True)


def main():
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("run this diagnostic with a single MPI rank (serial)")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    bdf_file = os.path.join(base_dir, "input_files", "transient_beam.bdf")

    rho = 27.0  # density kg/m^3
    E = 70.0e2  # Young's modulus (Pa)
    nu = 0.3  # Poisson's ratio
    ys = 2.7e-2  # yield stress
    t = 0.2  # m
    w = 0.5  # m

    def elem_call_back(dv_num, comp_id, comp_descript, elem_descripts, global_dvs, **kwargs):
        prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
        con = constitutive.IsoRectangleBeamConstitutive(prop, t=t, tNum=dv_num, w=w, wNum=dv_num + 1)
        refAxis = np.array([0.0, 1.0, 0.0])
        transform = elements.BeamRefAxisTransform(refAxis)
        return elements.Beam2(transform, con)

    fea_assembler = pytacs.pyTACS(bdf_file, comm, options={"writeCoordinateFrame": True})
    print("DIAG: built pyTACS", flush=True)
    fea_assembler.initialize(elem_call_back)
    print("DIAG: initialized fea_assembler", flush=True)

    tacs_probs = fea_assembler.createTACSProbsFromBDF()
    print("DIAG: created tacs_probs", flush=True)
    assembler = list(tacs_probs.values())[0].assembler

    res = assembler.createVec()
    mat = assembler.createSchurMat()
    probe = assembler.createVec()
    outp = assembler.createVec()
    n = res.getSize()
    print(f"DIAG: created assembler objects, n={n}", flush=True)

    # (label, alpha, beta, gamma) -- see module docstring
    coeff_sets = [
        ("step1_order1", 1.0, 10.0, 100.0),
        ("step2plus_order2", 1.0, 15.0, 225.0),
    ]

    for label, alpha, beta, gamma in coeff_sets:
        assembler.zeroVariables()
        res.zeroEntries()
        mat.zeroEntries()
        assembler.assembleJacobian(alpha, beta, gamma, res, mat)
        print(f"### DUMP {label} alpha={alpha} beta={beta} gamma={gamma} assembled", flush=True)

        dense_cols = []
        for j in range(n):
            probe.zeroEntries()
            probe_arr = probe.getArray()
            probe_arr[j] = 1.0
            outp.zeroEntries()
            mat.mult(probe, outp)
            dense_cols.append(np.array(outp.getArray(), copy=True))
        dense = np.column_stack(dense_cols)
        print(f"### DUMP {label} shape={dense.shape}", flush=True)

        flat = dense.ravel(order="C")
        for i, v in enumerate(flat):
            fv = float(v.real if np.iscomplexobj(v) else v)
            print(f"DUMP {label}[{i}] = {fv.hex()}", flush=True)
        print(f"### DUMP {label} END", flush=True)

    # Part 2: residual assembly at a fixed nonzero state (see module docstring)
    label = "residual_state1"
    vars_arr_vals = [0.001 * (i + 1) for i in range(n)]
    dvars_arr_vals = [0.002 * (i + 1) for i in range(n)]
    ddvars_arr_vals = [0.003 * (i + 1) for i in range(n)]

    vars_vec = assembler.createVec()
    dvars_vec = assembler.createVec()
    ddvars_vec = assembler.createVec()
    vars_vec.getArray()[:] = vars_arr_vals
    dvars_vec.getArray()[:] = dvars_arr_vals
    ddvars_vec.getArray()[:] = ddvars_arr_vals
    assembler.setVariables(vars_vec, dvars_vec, ddvars_vec)
    print(f"DIAG: set nonzero state for {label}", flush=True)

    res.zeroEntries()
    assembler.assembleRes(res)
    print(f"### DUMP {label} assembled", flush=True)

    res_arr = np.array(res.getArray(), copy=True)
    print(f"### DUMP {label} shape={res_arr.shape}", flush=True)
    for i, v in enumerate(res_arr):
        fv = float(v.real if np.iscomplexobj(v) else v)
        print(f"DUMP {label}[{i}] = {fv.hex()}", flush=True)
    print(f"### DUMP {label} END", flush=True)


try:
    main()
    print("DIAG: script done OK", flush=True)
except BaseException:
    print("DIAG: SCRIPT EXCEPTION", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    raise
