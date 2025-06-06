"""
==============================================================================
Optimization of a stiffened plate
==============================================================================
@File    :   optimize_stiffened_plate.py
@Date    :   2024/07/09
@Author  :   Alasdair Christison Gray
@Description : A simple example of optimising a stiffened panel using the
blade-stiffened shell constitutive model. We will optimise the sizing of the
plate and its stiffeners to minimise mass subject to the material and buckling
failure criteria included in the TACSBladeStiffenedShell class.

This example is based on the composite panel design example example in section
13.2 of "Design and Analysis of Composite Structures with Applications to
Aerospace Structures" by Christos Kassapoglou. The task is to size a 0.75 x
1.5m stiffened panel based on 2 loadcases:
(a) Uniform pressure of 62,500 Pa
(b) Applied in-plane loads Nx = −350 N/mm, Nxy = 175 N/mm

The panel should not fail in either case and the out-of-plane deflection
should be less than 10mm in case (a).

We should not expect to achieve exactly the same result as in the book as
there are a number of modelling differences between the two cases, including
but not limited to the following:
- The textbook case does not present an optimum sizing, just a reasonable
sizing that has a sufficient margin of safety.
- The textbook case uses J stiffeners, whereas the TACSBladeStiffenedShell
class uses T stiffeners.
- I have increased the pressure load and allowable deflection in case (a) by a
factor of 5 so that the plate is both stiffness and failure critical in this
case.
- The textbook case computes the laminate stiffness properties using classical
lamination theory and a discrete stacking sequence, whereas the
TACSBladeStiffenedShell class uses smeared stiffness approach that ignores the
stacking sequence.
- The buckling analysis of the skin and stiffener in the textbook accounts for
load redistribution from the skin to the stiffeners when the skin buckles, the
TACSBladeStiffenedShell class does not account for this effect.
- The textbook case uses two different types of ply within the skin, whereas
the TACSBladeStiffenedShell class uses a single ply type for the skin and
stiffener laminates
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import argparse

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import openmdao.api as om
from mphys.core import Multipoint, MPhysVariables
from mphys.scenarios import ScenarioStructural

# ==============================================================================
# Extension modules
# ==============================================================================
from tacs import elements, constitutive, functions
from tacs.mphys import TacsBuilder

# ==============================================================================
# Process some command line arguments
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--output", default="output", help="Output directory")
parser.add_argument(
    "--useGP",
    default=False,
    action="store_true",
    help="whether to use the Gaussian process buckling predictions or not",
)
parser.add_argument(
    "--useStiffPitchDV",
    action="store_true",
    help="Use stiffener pitch as a design variable",
)
parser.add_argument(
    "--usePlyFracDV",
    action="store_true",
    help="Use ply fractions as design variables",
)
parser.add_argument(
    "--includeStiffenerBuckling",
    action="store_true",
    help="Include stiffener buckling in the failure criteria",
)
parser.add_argument(
    "--opt",
    default="slsqp",
    choices=["slsqp", "snopt", "ipopt", "nlpqlp", "psqp"],
    help="Optimizer to use",
)
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output, exist_ok=True)

# Overall plate dimensions
width = 0.75
length = 1.5

# Material properties (UD tape properties from textbook case)
rho = 1609.0
E1 = 137.9e9
E2 = 11.7e9
nu12 = 0.29
G12 = 4.82e9
G13 = 4.82e9
Xt = 2068e6
Xc = 1723e6
Yt = 96.5e6
Yc = 338e6
S12 = 124e6

# Baseline panel sizing
panelLength = length
panelWidth = width

stiffenerPitch = 0.125
stiffenerPitchMin = 0.1
stiffenerPitchMax = width / 2

panelThickness = 2.1717e-3
panelThicknessMin = 0.6e-3
panelThicknessMax = 0.1

stiffenerHeight = 57e-3
stiffenerHeightMin = 25e-3
stiffenerHeightMax = 0.15

stiffenerThickness = stiffenerHeight / 8.8
stiffenerThicknessMin = 0.6e-3
stiffenerThicknessMax = 0.1

# Ply angles/initial ply fractions
ply_angles = np.deg2rad([0.0, -45.0, 45.0, 90.0])
skin_ply_fractions = np.array([0.13953, 0.36047, 0.36047, 0.13953])
stiffener_ply_fractions = np.array([0.59402, 0.16239, 0.16239, 0.0812])

# Shear and compressive traction loads
Ny = 350e3  # N/m
Nxy = 175e3  # N/m

# Pressure load and max deflection
MAX_DISP = 10e-3  # m
PRESSURE_LOAD = 62.5e3  # Pa

flangeFraction = 1.0
minFlangeGap = 50e-3


def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    # Create ply object
    ortho_prop = constitutive.MaterialProperties(
        rho=rho,
        E1=E1,
        E2=E2,
        nu12=nu12,
        G12=G12,
        G13=G13,
        G23=G13,
        Xt=Xt,
        Xc=Xc,
        Yt=Yt,
        Yc=Yc,
        S12=S12,
    )
    ply = constitutive.OrthotropicPly(0.1, ortho_prop)

    # --- Define skin/stiffener design variables ---
    currentDVNum = dvNum

    panelLengthNum = -1

    if args.useStiffPitchDV:
        stiffenerPitchNum = currentDVNum
        currentDVNum = currentDVNum + 1
    else:
        stiffenerPitchNum = -1

    panelThicknessNum = currentDVNum
    currentDVNum = currentDVNum + 1

    # Assign each ply fraction a unique DV
    if args.usePlyFracDV:
        skin_ply_fraction_dv_nums = np.array(
            [
                currentDVNum,
                currentDVNum + 1,
                currentDVNum + 2,
                currentDVNum + 3,
            ],
            dtype=np.intc,
        )
        currentDVNum = currentDVNum + 4
    else:
        skin_ply_fraction_dv_nums = -np.ones(len(ply_angles), dtype=np.intc)

    stiffenerHeightNum = currentDVNum
    currentDVNum = currentDVNum + 1

    stiffenerThicknessNum = currentDVNum
    currentDVNum = currentDVNum + 1

    # Don't assign a DV to the stiffener ply fractions, optimiser will just make as many plies as possible 0 deg
    stiffener_ply_fraction_dv_nums = -np.ones(len(ply_angles), dtype=np.intc)

    if not (args.useGP):
        con = constitutive.BladeStiffenedShellConstitutive(
            panelPly=ply,
            stiffenerPly=ply,
            panelLength=panelLength,
            stiffenerPitch=stiffenerPitch,
            panelThick=panelThickness,
            panelPlyAngles=ply_angles,
            panelPlyFracs=skin_ply_fractions,
            stiffenerHeight=stiffenerHeight,
            stiffenerThick=stiffenerThickness,
            stiffenerPlyAngles=ply_angles,
            stiffenerPlyFracs=stiffener_ply_fractions,
            panelLengthNum=panelLengthNum,
            stiffenerPitchNum=stiffenerPitchNum,
            panelThickNum=panelThicknessNum,
            panelPlyFracNums=skin_ply_fraction_dv_nums,
            stiffenerHeightNum=stiffenerHeightNum,
            stiffenerThickNum=stiffenerThicknessNum,
            stiffenerPlyFracNums=stiffener_ply_fraction_dv_nums,
            flangeFraction=flangeFraction,
        )
    else:  # args.useGP is True
        panelWidthNum = -1

        con = constitutive.GPBladeStiffenedShellConstitutive(
            panelPly=ply,
            stiffenerPly=ply,
            panelLength=panelLength,
            stiffenerPitch=stiffenerPitch,
            panelThick=panelThickness,
            panelPlyAngles=ply_angles,
            panelPlyFracs=skin_ply_fractions,
            stiffenerHeight=stiffenerHeight,
            stiffenerThick=stiffenerThickness,
            stiffenerPlyAngles=ply_angles,
            stiffenerPlyFracs=stiffener_ply_fractions,
            panelLengthNum=panelLengthNum,
            stiffenerPitchNum=stiffenerPitchNum,
            panelThickNum=panelThicknessNum,
            panelPlyFracNums=skin_ply_fraction_dv_nums,
            stiffenerHeightNum=stiffenerHeightNum,
            stiffenerThickNum=stiffenerThicknessNum,
            stiffenerPlyFracNums=stiffener_ply_fraction_dv_nums,
            panelWidth=panelWidth,
            panelWidthNum=panelWidthNum,
            flangeFraction=flangeFraction,
        )

    con.setStiffenerPitchBounds(stiffenerPitchMin, stiffenerPitchMax)
    con.setPanelThicknessBounds(panelThicknessMin, panelThicknessMax)
    con.setStiffenerThicknessBounds(stiffenerThicknessMin, stiffenerThicknessMax)
    con.setPanelPlyFractionBounds(
        np.array([0.1, 0.1, 0.1, 0.1]), np.array([1.0, 1.0, 1.0, 1.0])
    )

    # We need to enforce that stiffenerHeight <= stiffenerPitch - minFlangeGap, if we are not
    # using a stiffener pitch DV we can simply enforce this as an upper bound
    # on the stiffener height, otherwise we need to enforce this using a
    # linear constraint in the constraint_setup function
    if args.useStiffPitchDV:
        con.setStiffenerHeightBounds(stiffenerHeightMin, stiffenerHeightMax)
    else:
        con.setStiffenerHeightBounds(stiffenerHeightMin, stiffenerPitch - minFlangeGap)

    con.setFailureModes(
        includePanelMaterialFailure=True,
        includeStiffenerMaterialFailure=True,
        includeLocalBuckling=True,
        includeGlobalBuckling=True,
        includeStiffenerColumnBuckling=args.includeStiffenerBuckling,
        includeStiffenerCrippling=args.includeStiffenerBuckling,
    )

    # stiffeners and 0 degree plies are oriented with the y direction
    refAxis = np.array([0.0, 1.0, 0.0])
    transform = elements.ShellRefAxisTransform(refAxis)

    # Pass back the appropriate tacs element object
    elem = elements.Quad4Shell(transform, con)

    # Design variable scaling factors
    stiffPitchScale = 10.0
    panelThicknessScale = 1e2
    stiffenerHeightScale = 1e1
    stiffenerThicknessScale = 1e2
    DVScales = []
    if args.useStiffPitchDV:
        DVScales.append(stiffPitchScale)
    DVScales.append(panelThicknessScale)
    if args.usePlyFracDV:
        DVScales += [1.0] * 4
    DVScales += [stiffenerHeightScale, stiffenerThicknessScale]

    return elem, DVScales


def addForceToEdge(problem, edgeNodeIDs, cornerNodes, totalForce):
    numNodes = len(edgeNodeIDs)
    F = np.zeros((numNodes, 6))
    for ii in range(numNodes):
        if edgeNodeIDs[ii] in cornerNodes:
            F[ii, :] = 0.5
        else:
            F[ii, :] = 1.0
    for jj in range(6):
        F[:, jj] *= totalForce[jj] / np.sum(F[:, jj])

    problem.addLoadToNodes(edgeNodeIDs, F, nastranOrdering=True)


def problem_setup(scenario_name, fea_assembler, problem):
    """
    Helper function to add fixed forces and eval functions
    to structural problems used in tacs builder
    """
    # Use the output directory specified by the command line argument
    output_dir = os.path.join(args.output, scenario_name)
    problem.setOption("outputDir", output_dir)

    # Add TACS Functions
    problem.addFunction(
        "KSFailure",
        functions.KSFailure,
        safetyFactor=1.5,
        ksWeight=100.0,
    )
    problem.addFunction("Mass", functions.StructuralMass)

    # ==============================================================================
    # Add forces
    # ==============================================================================
    if scenario_name == "CompAndShear":
        # We have only applied the minimum necessary boundary conditions to
        # restrict rigid body motion of the plate we will therefore apply the
        # compressive forces on both the min and max x edges, and shear forces on
        # all 4 edges

        # First we need to get the node coordinates
        bdfInfo = fea_assembler.getBDFInfo()
        # cross-reference bdf object to use some of pynastrans advanced features
        bdfInfo.cross_reference()
        nastranNodeNums = list(bdfInfo.node_ids)
        nodeCoords = bdfInfo.get_xyz_in_coord()
        xMinNodes = np.nonzero(
            np.abs(np.min(nodeCoords[:, 0]) - nodeCoords[:, 0]) <= 1e-6
        )[0]
        xMinNodeIDs = [nastranNodeNums[ii] for ii in xMinNodes]

        xMaxNodes = np.nonzero(
            np.abs(np.max(nodeCoords[:, 0]) - nodeCoords[:, 0]) <= 1e-6
        )[0]
        xMaxNodeIDs = [nastranNodeNums[ii] for ii in xMaxNodes]

        yMinNodes = np.nonzero(
            np.abs(np.min(nodeCoords[:, 1]) - nodeCoords[:, 1]) <= 1e-6
        )[0]
        yMinNodeIDs = [nastranNodeNums[ii] for ii in yMinNodes]

        yMaxNodes = np.nonzero(
            np.abs(np.max(nodeCoords[:, 1]) - nodeCoords[:, 1]) <= 1e-6
        )[0]
        yMaxNodeIDs = [nastranNodeNums[ii] for ii in yMaxNodes]

        # Find the corner nodes because we need to apply smaller point forces to them
        cornerNodes = list(set(yMinNodeIDs) & set(xMinNodeIDs))
        cornerNodes.extend(list(set(yMinNodeIDs) & set(xMaxNodeIDs)))
        cornerNodes.extend(list(set(yMaxNodeIDs) & set(xMinNodeIDs)))
        cornerNodes.extend(list(set(yMaxNodeIDs) & set(xMaxNodeIDs)))

        # yMin face, compressive load in +y direction, shear in -x direction
        addForceToEdge(
            problem,
            yMinNodeIDs,
            cornerNodes,
            [-Nxy * width, Ny * width, 0.0, 0.0, 0.0, 0.0],
        )
        # yMax face, compressive load in -y direction, shear in +x direction
        addForceToEdge(
            problem,
            yMaxNodeIDs,
            cornerNodes,
            [Nxy * width, -Ny * width, 0.0, 0.0, 0.0, 0.0],
        )
        # xMin face shear in -y direction
        addForceToEdge(
            problem,
            xMinNodeIDs,
            cornerNodes,
            [0.0, -Nxy * length, 0.0, 0.0, 0.0, 0.0],
        )
        # xMax face shear in +y direction
        addForceToEdge(
            problem,
            xMaxNodeIDs,
            cornerNodes,
            [0.0, Nxy * length, 0.0, 0.0, 0.0, 0.0],
        )
    elif scenario_name == "Pressure":
        allComponents = fea_assembler.selectCompIDs()
        problem.addPressureToComponents(allComponents, PRESSURE_LOAD)

        # We need to limit the max Z displacement to < 10mm
        problem.addFunction(
            "MaxDispZ",
            functions.KSDisplacement,
            ksWeight=100.0,
            direction=[0.0, 0.0, -1.0],
        )


def constraint_setup(scenario_name, fea_assembler, constraint_list):
    """
    Helper function to setup tacs constraint classes
    """
    if scenario_name == "CompAndShear":
        allComponents = fea_assembler.selectCompIDs()
        if args.usePlyFracDV or args.useStiffPitchDV:
            constr = fea_assembler.createDVConstraint("DVCon")
        else:
            constr = None

        if args.usePlyFracDV:
            # Skin ply fractions must add up to 1
            firstSkinPlyFracNum = 2 if args.useStiffPitchDV else 1
            constr.addConstraint(
                "SkinPlyFracSum",
                allComponents,
                dvIndices=list(range(firstSkinPlyFracNum, firstSkinPlyFracNum + 4)),
                dvWeights=[1.0, 1.0, 1.0, 1.0],
                lower=1.0,
                upper=1.0,
            )
            # Fractions of + and -45 degree plies should be equal
            constr.addConstraint(
                "SkinLaminateBalance",
                allComponents,
                dvIndices=[firstSkinPlyFracNum + 1, firstSkinPlyFracNum + 2],
                dvWeights=[1.0, -1.0],
                lower=0.0,
                upper=0.0,
            )

        # Enforce the minimum gap between adjacent stiffner flanges, the flange
        # width is equal to the stiffener height multiplied by the flange fraction
        # so the constraint is:
        # stiffPitch - stiffHeight*flangeFraction >= minFlangeGap
        if args.useStiffPitchDV:
            stiffenerHeightInd = 2 if not args.usePlyFracDV else 6
            constr.addConstraint(
                "StiffenerOverlap",
                allComponents,
                dvIndices=[0, stiffenerHeightInd],
                dvWeights=[1.0, -flangeFraction],
                lower=minFlangeGap,
            )

        if constr is not None:
            constraint_list.append(constr)


scenarioNames = ["CompAndShear", "Pressure"]
meshNames = {"CompAndShear": "plate.bdf", "Pressure": "plate_pinned_edges.bdf"}

# BDF file containing mesh
bdfFile = os.path.join(os.path.dirname(__file__), "plate.bdf")


class PlateModel(Multipoint):
    def setup(self):
        for ii, scenarioName in enumerate(scenarioNames):
            struct_builder = TacsBuilder(
                mesh_file=os.path.join(
                    os.path.dirname(__file__), meshNames[scenarioName]
                ),
                element_callback=element_callback,
                problem_setup=problem_setup,
                constraint_setup=constraint_setup,
                check_partials=True,
            )
            struct_builder.initialize(self.comm)

            # We only need to setup the design variable and mesh components once as both scenarios will use the same design variables and mesh coordinates.
            if ii == 0:
                init_dvs = struct_builder.get_initial_dvs()
                dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
                dvs.add_output("dv_struct", init_dvs)
                lb, ub = struct_builder.get_dv_bounds()
                structDVScaling = np.array(struct_builder.fea_assembler.scaleList)
                self.add_design_var(
                    "dv_struct", lower=lb, upper=ub, scaler=structDVScaling
                )

                self.add_subsystem(
                    "mesh", struct_builder.get_mesh_coordinate_subsystem()
                )

            self.mphys_add_scenario(
                scenarioName, ScenarioStructural(struct_builder=struct_builder)
            )
            self.connect(
                f"mesh.{MPhysVariables.Structures.Mesh.COORDINATES}",
                f"{scenarioName}.{MPhysVariables.Structures.COORDINATES}",
            )

            self.connect("dv_struct", f"{scenarioName}.dv_struct")

            # Use the output directory specified by the command line argument
            scenario_output_dir = os.path.join(args.output, scenarioName)
            os.makedirs(scenario_output_dir, exist_ok=True)

    def configure(self):
        # Add TACS constraints
        firstScenario = self.__getattribute__(scenarioNames[0])
        if hasattr(firstScenario.struct_post, "constraints"):
            for system in firstScenario.struct_post.constraints.system_iter():
                constraint = system.constr
                constraintFuncNames = constraint.getConstraintKeys()
                bounds = {}
                constraint.getConstraintBounds(bounds)
                for conName in constraintFuncNames:
                    if self.comm.rank == 0:
                        print("Adding constraint: ", conName)
                    name = f"{scenarioNames[0]}.{system.name}.{conName}"
                    lb = bounds[f"{system.name}_{conName}"][0]
                    ub = bounds[f"{system.name}_{conName}"][1]
                    if all(lb == ub):
                        self.add_constraint(name, equals=lb, linear=True)
                    else:
                        self.add_constraint(name, lower=lb, upper=ub, linear=True)


# ==============================================================================
# OpenMDAO setup
# ==============================================================================

prob = om.Problem()
prob.model = PlateModel()
model = prob.model

# Declare design variables, objective, and constraint
model.add_objective("CompAndShear.Mass")
for scenarioName in scenarioNames:
    model.add_constraint(f"{scenarioName}.KSFailure", upper=1.0, linear=False)

model.add_constraint("Pressure.MaxDispZ", upper=MAX_DISP, scaler=1e3)

# Configure optimizer
debug_print = ["objs", "nl_cons", "ln_cons", "desvars"]
prob.driver = om.pyOptSparseDriver(
    optimizer=args.opt.upper(), print_opt_prob=True, debug_print=debug_print
)
# Use the output directory specified by the command line argument
prob.driver.options["hist_file"] = os.path.join(args.output, "structOpt.hst")


# Setup OpenMDAO problem
prob.setup(mode="rev")

prob.run_model()

# Output N2 representation of OpenMDAO model
# Use the output directory specified by the command line argument
om.n2(prob, show_browser=False, outfile=os.path.join(args.output, "tacs_struct.html"))

# Run optimization
prob.run_driver()

# --- Print out optimal values ---
dv_struct = prob.get_val("dv_struct")

currentDVNum = 0
if args.useStiffPitchDV:
    optStiffPitch = dv_struct[currentDVNum]
    currentDVNum += 1
else:
    optStiffPitch = stiffenerPitch
optSkinThickness = dv_struct[currentDVNum]
currentDVNum += 1
if args.usePlyFracDV:
    optSkinPlyFrac = dv_struct[currentDVNum : currentDVNum + 4]
    currentDVNum += 4
else:
    optSkinPlyFrac = skin_ply_fractions
optStiffHeight = dv_struct[currentDVNum]
currentDVNum += 1
optStiffThickness = dv_struct[currentDVNum]
currentDVNum += 1
optStiffPlyFrac = stiffener_ply_fractions

print("Optimal sizing:")
print("================================")
print(f"Stiffener pitch: {optStiffPitch*1e3} mm")
print(f"Panel thickness: {optSkinThickness*1e3} mm")
print("Skin ply fractions:")
print(optSkinPlyFrac)
print(f"Stiffener height: {optStiffHeight*1e3} mm")
print(f"Stiffener thickness: {optStiffThickness*1e3} mm")
print("Stiffener ply fractions:")
print(optStiffPlyFrac)
