"""
Temporary diagnostic for docs/plans/bugfix-trans-beam-macos-blowup.

Dumps the assembled global tangent matrix (alpha*K + beta*C + gamma*M) for
the transient_beam.bdf / Beam2 model, bit-exactly (as hex floats), for the
two distinct (alpha, beta, gamma) coefficient triples the real 20-step,
dt=0.1 BDF-2 transient run actually uses at Newton iteration 0 of each step
(order-1 ramp at step 1, steady-state order-2 BDF for steps 2-20 on this
uniform grid) -- hand-derived from TACSBDFIntegrator::get2ndBDFCoeff.

TACSBeamLinearModel's addJacobian output does not depend on the current
state (vars/dvars/ddvars) at all -- only on Xpts, material properties, and
(alpha, beta, gamma) -- so this reproduces exactly the numeric computation
the real transient performs at each step's first Newton iteration, without
needing to run the transient loop itself. Run in serial (1 rank) so
Mat.getDenseMatrix() (dense, un-permuted) is available.

Intended to be run identically on Real-Ubuntu / Real-MacOS /
Real-MacOS-NoFPContract CI jobs and diffed. Remove once the investigation
closes.
"""

import os

import numpy as np
from mpi4py import MPI

from tacs import constitutive, elements, pytacs

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
fea_assembler.initialize(elem_call_back)

tacs_probs = fea_assembler.createTACSProbsFromBDF()
assembler = list(tacs_probs.values())[0].assembler

res = assembler.createVec()
mat = assembler.createSchurMat()

# (label, alpha, beta, gamma) -- see module docstring
COEFF_SETS = [
    ("step1_order1", 1.0, 10.0, 100.0),
    ("step2plus_order2", 1.0, 15.0, 225.0),
]

for label, alpha, beta, gamma in COEFF_SETS:
    assembler.zeroVariables()
    res.zeroEntries()
    mat.zeroEntries()
    assembler.assembleJacobian(alpha, beta, gamma, res, mat)
    dense = np.asarray(mat.getDenseMatrix())

    print(f"### DUMP {label} alpha={alpha} beta={beta} gamma={gamma}", flush=True)
    print(f"### DUMP {label} shape={dense.shape}", flush=True)
    flat = dense.ravel(order="C")
    for i, v in enumerate(flat):
        fv = float(v.real if np.iscomplexobj(v) else v)
        print(f"DUMP {label}[{i}] = {fv.hex()}", flush=True)
    print(f"### DUMP {label} END", flush=True)
