"""
pyModal_problem
"""

# =============================================================================
# Imports
# =============================================================================
import warnings
import os
import numpy as np
from collections import OrderedDict
import time
from .base import TACSProblem
import tacs.TACS

class ModalProblem(TACSProblem):

    def __init__(self, name, fGuess, numFreqs,
                 assembler, comm, outputViewer=None, meshLoader=None,
                 options={}):
        """
        The main purpose of this class is to represent all relevant
        information for a modal analysis.

        Parameters
        ----------
        name : str
            Name of this tacs problem

        fGuess : float
            Guess for the lowest natural frequency (rad/s)

        numFreqs : int
            Number of eigenfrequencies to solve for

        assembler : assembler
            Cython object responsible for creating and setting tacs objects used to solve problem

        comm : MPI Intracomm
            The comm object on which to create the pyTACS object.

        outputViewer : TACSToFH5 object
            Cython object used to write out f5 files that can be converted and used for postprocessing.

        meshLoader : pyMeshLoader object
            pyMeshLoader object used to create the assembler.

        options : dict
            Dictionary holding problem-specific option parameters.
        """
        # python object name
        self.objectName = 'ModalProblem'

        # Problem name
        self.name = name

        # Defualt setup for common problem class objects
        super().__init__(assembler, comm, outputViewer, meshLoader)

        # Set time interval parmeters
        self.sigma = fGuess**2
        self.numEigs = numFreqs

        # Default Option List
        defOpts = {
            'outputdir': [str, './'],

            # Solution Options
            'L2Convergence': [float, 1e-12],
            'L2ConvergenceRel': [float, 1e-12],
            'subSpaceSize': [int, 10],
            'nRestarts': [int, 15],

            # Output Options
            'outputFrequency': [int, 0],
            'writeSolution': [bool, True],
            'numberSolutions': [bool, True],
            'printTiming': [bool, False],
            'printLevel': [int, 0],

        }

        # Process the default options which are added to self.options
        # under the 'defaults' key. Make sure the key are lower case
        self.options = {}
        def_keys = defOpts.keys()
        self.options['defaults'] = {}
        for key in def_keys:
            self.options['defaults'][key.lower()] = defOpts[key]
            self.options[key.lower()] = defOpts[key]

        # Set user-defined options
        for key in options:
            self.setOption(key, options[key])

        # Create problem-specific variables
        self._createVariables()

    def _createVariables(self):
        """
        Internal to create the objects required by TACS Integrator
        """

        self.callCounter = -1

        # Solve the eigenvalue problem
        self.M = self.assembler.createSchurMat()
        self.K = self.assembler.createSchurMat()

        self.pc = tacs.TACS.Pc(self.K)

        # Assemble and factor the stiffness/Jacobian matrix. Factor the
        # Jacobian and solve the linear system for the displacements
        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        self.assembler.assembleJacobian(alpha, beta, gamma, None, self.K);
        self.pc.factor()  # LU factorization of stiffness matrix

        subspace = self.getOption('subSpaceSize')
        restarts = self.getOption('nRestarts')
        self.gmres = tacs.TACS.KSM(self.K, self.pc, subspace, restarts)

        eigTol = self.getOption('L2Convergence')

        # Create the frequency analysis object
        self.freqSolver = tacs.TACS.FrequencyAnalysis(self.assembler, self.sigma, self.M, self.K, self.gmres,
                                                      num_eigs=self.numEigs, eig_tol=eigTol)

    def getNumFrequencies(self):
        """
        Get the number of eigenvalues requested from solver for this problem.

        Returns
        ----------
        numEigs : int
            Number of eigenvalues.
        """
        return self.numEigs

    def addFunction(self, funcName, funcHandle, compIDs=None, **kwargs):
        """
        NOT SUPPORTED FOR THIS PROBLEM
        """
        self.TACSWarning("addFunction method not supported for this class.")

    def evalFunctions(self, funcs, evalFuncs=None, ignoreMissing=False):
        """
        NOT SUPPORTED FOR THIS PROBLEM
        """
        self.TACSWarning("evalFunctions method not supported for this class.")

    def evalFunctionsSens(self, funcsSens, evalFuncs=None):
        """
        NOT SUPPORTED FOR THIS PROBLEM
        """
        self.TACSWarning("evalFunctionsSens method not supported for this class.")

    ####### Transient solver methods ########

    def _updateAssemblerVars(self):
        """
        Make sure that the assembler is using
        the input variables associated with this problem
        """

        self.assembler.setDesignVars(self.x)
        self.assembler.setNodes(self.Xpts)

    def solve(self):
        """
        Solve the time integrated transient problem. The
        forces must already be set.
        """
        startTime = time.time()

        self.callCounter += 1

        setupProblemTime = time.time()

        # Set problem vars to assembler
        self._updateAssemblerVars()

        initSolveTime = time.time()

        # Solve the frequency analysis problem
        self.freqSolver.solve(print_level=self.getOption('printLevel'))

        solveTime = time.time()

        # If timing was was requested print it, if the solution is nonlinear
        # print this information automatically if prinititerations was requested.
        if self.getOption('printTiming'):
            self.pp('+--------------------------------------------------+')
            self.pp('|')
            self.pp('| TACS Solve Times:')
            self.pp('|')
            self.pp('| %-30s: %10.3f sec' % ('TACS Setup Time', setupProblemTime - startTime))
            self.pp('| %-30s: %10.3f sec' % ('TACS Solve Init Time', initSolveTime - setupProblemTime))
            self.pp('| %-30s: %10.3f sec' % ('TACS Solve Time', solveTime - initSolveTime))
            self.pp('|')
            self.pp('| %-30s: %10.3f sec' % ('TACS Total Solution Time', solveTime - startTime))
            self.pp('+--------------------------------------------------+')

        return

    def getVariables(self, index, states=None):
        """
         Return the current state values for one mode of the current problem

         Parameters
         ----------
         index : int
             Mode index to return solution for.

         states : BVec or ndarray or None
             Place eigenvector for mode into this array (optional).

         Returns
         --------
         freq: float
             Eigenfrequency (rad/s) for mode

         states : ndarray
             Eigenvector for mode
         """
        eig, err = self.freqSolver.extractEigenvalue(index)
        freq = np.sqrt(eig)
        eigVector = self.assembler.createVec()
        self.freqSolver.extractEigenvector(index, eigVector)
        # Inplace assignment if vectors were provided
        if isinstance(states, tacs.TACS.Vec):
            states.copyValues(eigVector)
        elif isinstance(states, np.ndarray):
            states[:] = eigVector.getArray()
        return freq, eigVector.getArray()

    def writeSolution(self, indices=None):
        """
        This is a generic shell function that writes the output
        file(s).  The intent is that the user or calling program can
        call this function and pyTACS writes all the files that the
        user has defined. It is recommended that this function is used
        along with the associated logical flags in the options to
        determine the desired writing procedure

        Parameters
        ----------

        Indices : int or list[int] or None
            Mode index or indices to get state variables for.
            If None, returns a solution for all modes.
            Defaults to None.
        """
        # Unless the writeSolution option is off write actual file:
        if self.getOption('writeSolution'):
            # Check input
            outputDir = self.getOption('outputDir')
            baseName = self.name
            if self.getOption('numberSolutions'):
                baseName = baseName + '_%3.3d' % self.callCounter

            # If indices is None, output all modes
            if indices is None:
                indices = np.arange(self.numEigs)

            # Write out each specified mode
            indices = np.atleast_1d(indices)
            vec = self.assembler.createVec()
            for index in indices:
                # Extract eigenvector
                freq, _ = self.getVariables(index, states=vec)
                # Set eigen mode in assembler
                self.assembler.setVariables(vec)
                # Write out mode shape as f5 file
                modeName = baseName + '_%3.3d' % index
                fileName = os.path.join(outputDir, modeName) + '.f5'
                self.outputViewer.writeToFile(fileName)