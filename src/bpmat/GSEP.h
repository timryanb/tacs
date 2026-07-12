/*
  This file is part of TACS: The Toolkit for the Analysis of Composite
  Structures, a parallel finite-element code for structural and
  multidisciplinary design optimization.

  Copyright (C) 2010 University of Toronto
  Copyright (C) 2012 University of Michigan
  Copyright (C) 2014 Georgia Tech Research Corporation
  Additional copyright (C) 2010 Graeme J. Kennedy and Joaquim
  R.R.A. Martins All rights reserved.

  TACS is licensed under the Apache License, Version 2.0 (the
  "License"); you may not use this software except in compliance with
  the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef TACS_GSEP_H
#define TACS_GSEP_H

#include "KSM.h"

/*!
  The following contains classes required for computation of
  general, symmetric eigenvalue problems (GSEP).

  The Lanczos method is used to construct an orthonormal basis of a
  Krylov subspace. This method produces a tri-diagonal Hessenberg
  matrix. Approximate eigenvalues of the full matrix may be found by
  discarding the last row of the rectangular - equivalent to
  projecting the matrix onto the orthonormal basis - and computing the
  eigenvalues of the much smaller system.

  The method is implemented in a block-form such that operations
  are performed with a block-size of P.

  Solve the eigenvalue problem:

  A x = lambda B x
*/
class EPOperator : public TACSObject {
 public:
  virtual ~EPOperator() {}

  // Create a vector associated with this operator
  // ---------------------------------------------
  virtual TACSVec *createVec() = 0;

  // Compute y = A *x
  // -----------------
  virtual void mult(TACSVec *x, TACSVec *y) = 0;

  // Compute the inner product <x,y>
  // -------------------------------
  virtual TacsScalar dot(TACSVec *x, TACSVec *y) { return x->dot(y); }

  // Compute || B *x || - this is used to compute the eigenvalue error
  // ------------------------------------------------------------------
  virtual TacsScalar errorNorm(TACSVec *x) { return 1.0; }

  // Convert the shifted spectrum back to the regular spectrum
  // ---------------------------------------------------------
  virtual TacsScalar convertEigenvalue(TacsScalar value) { return value; }
};

/*!
  Solve the regular eigenvalue problem without any shift/invert
  operations
*/
class EPRegular : public EPOperator {
 public:
  EPRegular(TACSMat *_mat);
  ~EPRegular();

  TACSVec *createVec();
  void mult(TACSVec *x, TACSVec *y);

 private:
  TACSMat *mat;
};

/*!
  Shift and invert the spectrum
*/
class EPShiftInvert : public EPOperator {
 public:
  EPShiftInvert(TacsScalar _sigma, TACSKsm *_ksm);
  ~EPShiftInvert();

  TACSVec *createVec();
  void mult(TACSVec *x, TACSVec *y);

  // The eigenvalues are computed as mu = 1.0/( eig - sigma )
  // eig = 1.0/mu + sigma
  TacsScalar convertEigenvalue(TacsScalar value);

 private:
  TACSKsm *ksm;
  TacsScalar sigma;
};

/*!
  Solve the generalized eigenvalue problem,

  A x = lambda B x

  Here B must be positive definite. The following shifted
  problem is solved,

  ( A - sigma B )^{-1} B x = 1.0/( lambda - sigma ) x

  In this case the B-inner products must be used so inner = B.
*/
class EPGeneralizedShiftInvert : public EPOperator {
 public:
  EPGeneralizedShiftInvert(TacsScalar _sigma, TACSKsm *_ksm, TACSMat *_inner);
  ~EPGeneralizedShiftInvert();

  void setSigma(TacsScalar _sigma);
  TACSVec *createVec();
  void mult(TACSVec *x, TACSVec *y);  // Compute y = (A - sigma B)^{-1}*inner*x
  TacsScalar dot(TACSVec *x, TACSVec *y);  // Compute <x,y> = x^{T}*inner*y
  TacsScalar errorNorm(TACSVec *x);
  TacsScalar convertEigenvalue(TacsScalar value);

 private:
  TacsScalar sigma;
  TACSKsm *ksm;
  TACSMat *inner;
  TACSVec *temp;
};

/*
  Solve a generalized eigenvalue problem for a buckling problem,

  A x = - lambda B x

  (Note the sign!) Here A = K, must be positive definite, but B
  does not need to be. The shifted problem is,

  (A - sigma B)^{-1} A x = lambda/(lambda + sigma) x

  Here A-inner products are required, so inner = A.
*/
class EPBucklingShiftInvert : public EPOperator {
 public:
  EPBucklingShiftInvert(TacsScalar _sigma, TACSKsm *_ksm, TACSMat *_inner);
  ~EPBucklingShiftInvert();

  void setSigma(TacsScalar _sigma);
  TACSVec *createVec();
  void mult(TACSVec *x, TACSVec *y);  // Compute y = (A - sigma B)^{-1}*inner*x
  TacsScalar dot(TACSVec *x, TACSVec *y);  // Compute <x,y> = x^{T}*inner*y
  TacsScalar errorNorm(TACSVec *x);
  TacsScalar convertEigenvalue(TacsScalar value);

 private:
  TacsScalar sigma;
  TACSKsm *ksm;
  TACSMat *inner;
  TACSVec *temp;
};

/*
  The symmetric eigenvalue solver.

  This eigensolver can be used to solve either simple or generalized
  eigenvalue problems depending on the definition of the operator.
  The operator may also use a shift and invert strategy to accelerate
  convergence of the Lanczos method.

  Note that the full orthogonalization is suggested (and is the
  default) since this has better numerical properties. The Lanczos
  vectors lose orthogonality as the eigenvalues converge.
*/
class SEP : public TACSObject {
 public:
  // Set the type of orthogonalization to use
  enum OrthoType { FULL, LOCAL };
  enum EigenSpectrum {
    SMALLEST,
    LARGEST,
    SMALLEST_MAGNITUDE,
    LARGEST_MAGNITUDE
  };

  SEP(EPOperator *_Op, int _max_iters, OrthoType _ortho_type = FULL,
      TACSBcMap *_bcs = NULL, int _restart_size = 0);
  ~SEP();

  // Set the orthogonalization strategy
  void setOrthoType(OrthoType _ortho_type);

  // Set the solution tolerances, type of spectrum and number of eigenvalues
  void setTolerances(double _tol, EigenSpectrum _spectrum, int _neigvals);

  // Reset the eigenproblem operator
  void setOperator(EPOperator *_Op);

  // Set the thick-restart basis size (SPEC Item 5). 0 (the default)
  // disables thick restart entirely and reproduces today's monolithic
  // single-Krylov-space path byte-for-byte; a value in (0, max_iters)
  // bounds the live Lanczos basis at _restart_size vectors. NOTE: the Wu &
  // Simon (2000) restart procedure itself is not yet implemented for the
  // FULL-orthogonalization branch (see GSEP.cpp's solve()) -- reaching
  // the bound currently just stops the solve gracefully (solve_flag=0),
  // and calling this setter after construction has no effect on that
  // branch's behavior (see GSEP.cpp's SEP::setRestartSize docstring).
  void setRestartSize(int _restart_size);

  // Solve the eigenproblem. Returns a solve_flag: 1 if converged, 0 if it
  // ran without full convergence (or produced a non-finite eigenvalue),
  // -1 if misconfigured (neigvals > max_iters -- no eigenvalues computed).
  int solve(KSMPrint *ksm_print = NULL, KSMPrint *ksm_file = NULL);

  // Extract the eigenvalues and eigenvectors from the solver
  TacsScalar extractEigenvalue(int n, TacsScalar *error);
  TacsScalar extractEigenvector(int n, TACSVec *ans, TacsScalar *error);

  // Check the orthogonality of the Lanczos subspace (expensive)
  TacsScalar checkOrthogonality();
  void printOrthogonality();

 private:
  // Sort the eigenvalues by the desired spectrum
  void sortEigenvalues(TacsScalar *values, int neigs, int *permutation);

  // Check whether the right eigenvalues have converged
  int checkConverged(TacsScalar *A, TacsScalar *B, int n);

  // Data used to determine which spectrum to use and when
  // enough eigenvalues are converged
  double tol;
  enum EigenSpectrum spectrum;
  int neigvals;

  // The eigenproblem operator object
  EPOperator *Op;

  // The number of iterations completed thus far
  int niters;

  // The number of eigenvalues actually computed (i.e. the largest n for
  // which ComputeEigsTriDiag/sortEigenvalues has populated
  // eigs/eigvecs/perm[0..n-1]) -- distinct from niters, which may be
  // nonzero even when the misconfiguration guard in solve() causes no
  // eigenvalues to be computed at all.
  int neigs_computed;

  // Thick-restart basis size (SPEC Item 5): 0 disables thick restart
  // (legacy path); a positive value <= max_iters bounds the live Lanczos
  // basis, triggering a restart once it is reached.
  int restart_size;

  // The number of vectors actually allocated in Q/Alpha/Beta/eigs/eigvecs/
  // perm below (restart_size when thick restart is active, max_iters
  // otherwise) -- distinct from max_iters, which continues to mean "give
  // up after this many total Lanczos steps" (SPEC lines 812-816) even when
  // thick restart bounds the *live* basis to something smaller.
  int alloc_size;

  // The coefficients of the symmetric tridiagonal matrix
  TacsScalar *Alpha, *Beta;

  // Eigenvalues/eigenvectors of the tridiagonal matrix
  TacsScalar *eigs, *eigvecs;

  // Permutation yielding an ascending ordering of the eigenvalues after
  // the shift has been applied
  int *perm;

  // The type of orthogonalization to use
  OrthoType ortho_type;

  int max_iters;
  TACSVec **Q;  // The Vectors for the eigenvalue problem...

  // Boundary conditions that are applied
  TACSBcMap *bcs;
};

#endif  // TACS_GSEP_H
