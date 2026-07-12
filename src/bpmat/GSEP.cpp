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

#include "GSEP.h"

#include "TacsUtilities.h"
#include "tacslapack.h"

/*
  The following file contains the definitions for two main types of classes:

  EPOperator - Defines the eigen problem

  This is a abstract base class that defines an eigenvalue problem with a
  number of properties. It defines an operator which allows a matrix to
  be modified with a spectral transformation. The shifted/inverted eigenvalue
  is extracted from the operator. The class also provides a dot product that
  is used in the construction of the orthogonal basis of the Krylov subspace.
  In the case of symmetric generalized eigenvalue problems where B is symmetric
  this dot product may be replaced with the inner product <x,y> = x^{T} B y.

  SEP - Iterative eigenvalue solver for symmetric eigenvalue problems

  This class generates an orthogonal basis for a Krylov subpsace using
  either Lanczos or full Gram-Schmidt orthogonalization. The Gram-Schmidt
  option is to be prefered. The additional cost is out-weighted by the better
  numerical properties of the full-orthogonalization. The initial subspace
  vector is generated randomly. All subsequent vectors are generated through
  matrix-vector products with the EPOperator class defined above. This operator
  may involve a shifted/inverted eigen problem and so may require the solution
  of a linear system of equations.
*/

/*
  A regular implementation of the operator class - use the
  matrix directly as the operator. This is only efficient for
  determining the eigenvalues with large relative separation. For
  structural problems these are typically the largest eigenvalues
  - the ones that are "uninteresting". This can, however, be used
  to determine the condition number of a system of equations.
*/
EPRegular::EPRegular(TACSMat *_mat) {
  mat = _mat;
  mat->incref();
}

EPRegular::~EPRegular() { mat->decref(); }

TACSVec *EPRegular::createVec() { return mat->createVec(); }

void EPRegular::mult(TACSVec *x, TACSVec *y) { return mat->mult(x, y); }

/*
  Shift and invert operator. This requires that the matrix be shifted
  by sigma away from origin. The KSM object should represent a
  solution to the linear system of equations

  y = (A - sigma I)^{-1} x

  for some value of sigma. This method is intendend to provide a way
  of extracting the eigenvalues close to sigma. The shift and invert
  strategy modifies the eigenvalues so that the modified eigenproblem
  has eigenvalues of,

  mu = 1.0/( lambda - sigma )

  This modification improves the separation of the eigenvalues near
  sigma, allowing them to converge faster. The eigenvectors of the
  modified problem are the same as the eigenvectors of the original
  problem.
*/
EPShiftInvert::EPShiftInvert(TacsScalar _sigma, TACSKsm *_ksm) {
  sigma = _sigma;
  ksm = _ksm;
  ksm->incref();
}

EPShiftInvert::~EPShiftInvert() { ksm->decref(); }

TACSVec *EPShiftInvert::createVec() { return ksm->createVec(); }

void EPShiftInvert::mult(TACSVec *x, TACSVec *y) {
  ksm->solve(x, y);
  return;
}

// The eigenvalues are computed as mu = 1.0/( eig - sigma )
// eig = 1.0/mu + sigma
TacsScalar EPShiftInvert::convertEigenvalue(TacsScalar value) {
  return (1.0 / value + sigma);
}

/*
  A shift and invert operator for genearlized eigenvalue problems.
  This operator transforms the initial generalized eigenvalue problem,

  A x = lambda B x

  into the following shifted and inverted problem,

  mu x = ( A - sigma B )^{-1} B x
  mu = 1.0/( lambda - sigma )

  The eigenvectors of the modified problem are the same as the
  eigenvectors of the initial problem.  Symmetry of the eigenvalue
  problem is maintained by utilizing the inner product <x,y> = x^{T} B
  y.  The Krylov basis is thus B-conjugate.

  This modification to the Lanczos algorithm is often called B-Lanczos
  in the literature.
*/
EPGeneralizedShiftInvert::EPGeneralizedShiftInvert(TacsScalar _sigma,
                                                   TACSKsm *_ksm,
                                                   TACSMat *_inner) {
  sigma = _sigma;
  ksm = _ksm;
  ksm->incref();
  inner = _inner;
  inner->incref();
  temp = inner->createVec();
  temp->incref();
}

EPGeneralizedShiftInvert::~EPGeneralizedShiftInvert() {
  ksm->decref();
  inner->decref();
  temp->decref();
}

/*
  Set the shift and invert value
*/
void EPGeneralizedShiftInvert::setSigma(TacsScalar _sigma) { sigma = _sigma; }

/*
  Create a vector associated with the generalized eigenvalue problem
*/
TACSVec *EPGeneralizedShiftInvert::createVec() { return ksm->createVec(); }

/*
  Compute y = ( A - sigma B )^{-1}*inner x
*/
void EPGeneralizedShiftInvert::mult(TACSVec *x, TACSVec *y) {
  inner->mult(x, temp);
  ksm->solve(temp, y);
  return;
}

/*
  Compute <x,y> = x^{T} inner y
*/
TacsScalar EPGeneralizedShiftInvert::dot(TACSVec *x, TACSVec *y) {
  inner->mult(y, temp);
  return temp->dot(x);
}

/*
  Compute ||B*x|| - this is used to compute the eigenvalue error
*/
TacsScalar EPGeneralizedShiftInvert::errorNorm(TACSVec *x) {
  inner->mult(x, temp);
  return temp->norm();
}

/*
  Convert the shifted eigenvalue to the actual eigenproblem
*/
TacsScalar EPGeneralizedShiftInvert::convertEigenvalue(TacsScalar value) {
  return (1.0 / value + sigma);
}

/*
  Solve the generalized buckling eigenvalue problem

  Ax = - lambda Bx

  for, x, lambda. A shift and invert strategy similar to
  the one employed above is used. Here instead, A-inner
  products are used and the modified eigenvalue problem is,

  (A + sigma B)^{-1} A x = lambda/(lambda - sigma) x
  mu = lambda/(lambda + sigma)

  The original eigenvalues may be obtained using,

  lambda = - mu * sigma/(1 - mu)
*/
EPBucklingShiftInvert::EPBucklingShiftInvert(TacsScalar _sigma, TACSKsm *_ksm,
                                             TACSMat *_inner) {
  sigma = _sigma;
  ksm = _ksm;
  ksm->incref();
  inner = _inner;
  inner->incref();
  temp = inner->createVec();
  temp->incref();
}

EPBucklingShiftInvert::~EPBucklingShiftInvert() {
  ksm->decref();
  inner->decref();
  temp->decref();
}

void EPBucklingShiftInvert::setSigma(TacsScalar _sigma) { sigma = _sigma; }

TACSVec *EPBucklingShiftInvert::createVec() { return ksm->createVec(); }

// Compute y = ( A - sigma B )^{-1} *  x
void EPBucklingShiftInvert::mult(TACSVec *x, TACSVec *y) {
  inner->mult(x, temp);
  ksm->solve(temp, y);
  return;
}

// Compute <x,y> = x^{T} inner y
TacsScalar EPBucklingShiftInvert::dot(TACSVec *x, TACSVec *y) {
  inner->mult(y, temp);
  return temp->dot(x);
}

// Compute || B * x || - this is used to compute the eigenvalue error
TacsScalar EPBucklingShiftInvert::errorNorm(TACSVec *x) {
  inner->mult(x, temp);
  return temp->norm();
}

TacsScalar EPBucklingShiftInvert::convertEigenvalue(TacsScalar value) {
  return -(value * sigma) / (1.0 - value);
}

/*
  Compute the eigenvalues and eigenvectors of a symmetric tridiagonal
  matrix.

  input:
  n:        the order of the matrix
  diag:     the diagonal entries
  upper:    the upper/lower entries of the matrix

  output:
  eigs:     the eigenvalues computed using LAPACK
  eigvecs:  the eigenvectors computed using LAPACK
*/
static void ComputeEigsTriDiag(int n, TacsScalar *_diag, TacsScalar *_upper,
                               TacsScalar *_eigs, TacsScalar *_eigvecs) {
  // The input arguments required for LAPACK
  const char *jobz = "V";
  const char *range = "A";

  // Specify the range of eigenvalues to use - not used in this
  // case. We compute all the eigenvalues in the spectrum
  double vl = 0.0, vu = 0.0;
  int il = 0, iu = 0;

  // The tolerance used in the solution - LAPACK makes this machine
  // precision internally
  double abstol = 0.0;

  // Convert the data to real data in the case of complex arithmetic.
  // Duplicate the data because LAPACK over-writes the matrix on exit.
  double *diag = new double[n];
  double *upper = new double[n - 1];
  for (int i = 0; i < n - 1; i++) {
    diag[i] = TacsRealPart(_diag[i]);
    upper[i] = TacsRealPart(_upper[i]);
  }
  diag[n - 1] = TacsRealPart(_diag[n - 1]);

  // The total number of eigenvalues found on output
  int m = 0;

  // Output and work arrays required by dstevr
  int *isuppz = new int[2 * n];
  int lwork = 20 * n;
  double *work = new double[lwork];
  int liwork = 10 * n;
  int *iwork = new int[liwork];
  int ldz = n;
  int info = -1;

#ifdef TACS_USE_COMPLEX
  /*
    Treat the imaginary part as a perturbation about the given point.
    In the result, replace the imaginary part with the sensitivity to
    this perturbation. This only makes sense in the context of the
    complex step method. It should only be used in this context!
  */
  double *eigs = new double[n];
  double *eigvecs = new double[n * n];

  LAPACKstevr(jobz, range, &n, diag, upper, &vl, &vu, &il, &iu, &abstol, &m,
              eigs, eigvecs, &ldz, isuppz, work, &lwork, iwork, &liwork, &info);

  if (info != 0) {
    fprintf(stderr, "Error encountered in LAPACK function dstevr\n");
  }

  /*
    The imaginary part of the eigenvalue problem can be determined as
    follows:
    A*x[n*i:(n+1)*i] = eigs[i]*x[n*i:(n+1)*i]

    The sensitivity of the eigenvalues are:
    dA/ds*x + A*dx/ds = deig/ds*x + eig*dx/ds

    Premultiplying by x^{T} yields,
    x^{T}dA/ds*x = x^{T}x*deig/ds
  */

  // Cycle over all the eigenvalues
  for (int i = 0; i < n; i++) {
    double sens = 0.0, dot = 0.0;
    for (int j = 0; j < n; j++) {  // Cycle over matrix entries
      double ans = TacsImagPart(_diag[j]) * eigvecs[n * i + j];
      if (j > 0) {
        ans += TacsImagPart(_upper[j - 1]) * eigvecs[n * i + j - 1];
      }
      if (j < n - 1) {
        ans += TacsImagPart(_upper[j]) * eigvecs[n * i + j + 1];
      }

      dot += eigvecs[n * i + j] * eigvecs[n * i + j];
      sens += ans * eigvecs[n * i + j];
    }

    _eigs[i] = TacsScalar(eigs[i], sens / dot);
  }

  // Discard the eigen sensitivity
  for (int i = 0; i < n * n; i++) {
    _eigvecs[i] = TacsScalar(eigvecs[i], 0.0);
  }

  delete[] eigs;
  delete[] eigvecs;
#else
  // This is the real part of the code that computes the eigenvalues and
  // eigenvectors of the symmetric tridiagonal system
  double *eigs = _eigs;
  double *eigvecs = _eigvecs;

  LAPACKstevr(jobz, range, &n, diag, upper, &vl, &vu, &il, &iu, &abstol, &m,
              eigs, eigvecs, &ldz, isuppz, work, &lwork, iwork, &liwork, &info);

  if (info != 0) {
    fprintf(stderr, "Error encountered in LAPACK function dstevr\n");
  }
#endif  // TACS_USE_COMPLEX

  delete[] isuppz;
  delete[] work;
  delete[] iwork;
  delete[] diag;
  delete[] upper;
}

/*
  Compute the eigenvalues and eigenvectors of the dense symmetric matrix
  produced once thick restart (SPEC Item 5, Wu & Simon 2000) is active.

  Before any restart has occurred (keep == 0) this matrix is exactly the
  same tridiagonal matrix ComputeEigsTriDiag above consumes -- diagonal
  Alpha[0..n-1], off-diagonal Beta[0..n-2] -- just solved with a dense
  (rather than tridiagonal-specialized) LAPACK routine. After a restart,
  entries [0, keep) are mutually decoupled (diagonal, in their own Ritz
  eigenbasis) except for a border coupling to column/row `keep` carried in
  Sigma[0..keep-1] (the classical thick-restart "arrowhead"); the standard
  tridiagonal chain resumes from index keep onward via Alpha[keep..n-1]/
  Beta[keep..n-2]. A small dense solve (LAPACK dsyev, O(n^3)) is used
  instead of a specialized banded/arrowhead routine because n is bounded by
  restart_size (tens, not thousands, by this feature's own design) --
  see HANDOFF-impl.md for why a hand-rolled tridiagonalizing bulge-chase of
  the arrowhead structure was rejected (verified, via an independent numpy
  reproduction, to silently diverge for keep >= 3 retained Ritz vectors).

  input:
  n:      the order of the matrix (current live basis size)
  keep:   0 if no restart has occurred yet (pure tridiagonal); otherwise
          the number of retained Ritz vectors bordering column/row `keep`
  Alpha:  the diagonal entries (post-restart: retained Ritz values for
          indices < keep, fresh Lanczos diagonal entries for indices >=
          keep)
  Beta:   the tridiagonal chain's off-diagonal entries, meaningful for
          indices >= keep (indices < keep - 1 are stale, superseded by
          Sigma)
  Sigma:  the arrowhead border coupling coefficients, meaningful for
          indices [0, keep)

  output:
  eigs:     the eigenvalues computed using LAPACK, ascending order
  eigvecs:  the eigenvectors computed using LAPACK
*/
static void ComputeEigsDense(int n, int keep, TacsScalar *Alpha,
                             TacsScalar *Beta, TacsScalar *Sigma,
                             TacsScalar *_eigs, TacsScalar *_eigvecs) {
  // Assemble the dense n x n symmetric matrix. Stored as a flat
  // row-major-or-column-major-ambiguous buffer -- since the matrix is
  // filled symmetrically (both mat[i*n+j] and mat[j*n+i] are set whenever
  // either is nonzero) it is simultaneously valid as the row-major matrix
  // this function intends AND as the column-major matrix LAPACK's Fortran
  // interface expects with LDA = n, so no transpose is needed either on
  // input or on the output eigenvectors (mirrors the existing convention
  // ComputeEigsTriDiag/checkConverged/extractEigenvector already rely on:
  // eigvecs[index * n + i] == component i of eigenvector `index`).
  double *mat = new double[n * n];
  memset(mat, 0, n * n * sizeof(double));
  for (int i = 0; i < n; i++) {
    mat[i * n + i] = TacsRealPart(Alpha[i]);
  }
  for (int i = keep; i < n - 1; i++) {
    mat[i * n + (i + 1)] = mat[(i + 1) * n + i] = TacsRealPart(Beta[i]);
  }
  if (keep > 0 && keep < n) {
    for (int j = 0; j < keep; j++) {
      mat[j * n + keep] = mat[keep * n + j] = TacsRealPart(Sigma[j]);
    }
  }

  const char *jobz = "V";
  const char *uplo = "U";
  int lda = n;
  int lwork = 8 * n + 16;  // comfortably above dsyev's max(1, 3*n-1) floor
  double *work = new double[lwork];
  double *w = new double[n];
  int info = -1;

  LAPACKdsyev(jobz, uplo, &n, mat, &lda, w, work, &lwork, &info);

  if (info != 0) {
    fprintf(stderr, "Error encountered in LAPACK function dsyev\n");
  }

#ifdef TACS_USE_COMPLEX
  /*
    Same complex-step perturbation technique ComputeEigsTriDiag uses above,
    generalized from a tridiagonal to a diagonal+arrowhead(+tridiagonal
    tail) sparsity pattern: for each eigenvector, accumulate
    x^{T} Imag(dM/ds) x over exactly the (i, j) pairs this matrix's
    assembly loop above populated (diagonal, tridiagonal-tail
    off-diagonals, and arrowhead border), rather than a dense n^2 loop.
  */
  for (int k = 0; k < n; k++) {
    double sens = 0.0, dot = 0.0;
    for (int i = 0; i < n; i++) {
      double xi = mat[k * n + i];
      sens += TacsImagPart(Alpha[i]) * xi * xi;
      dot += xi * xi;
    }
    for (int i = keep; i < n - 1; i++) {
      sens += 2.0 * TacsImagPart(Beta[i]) * mat[k * n + i] * mat[k * n + i + 1];
    }
    if (keep > 0 && keep < n) {
      for (int j = 0; j < keep; j++) {
        sens +=
            2.0 * TacsImagPart(Sigma[j]) * mat[k * n + j] * mat[k * n + keep];
      }
    }

    _eigs[k] = TacsScalar(w[k], sens / dot);
  }

  for (int i = 0; i < n * n; i++) {
    _eigvecs[i] = TacsScalar(mat[i], 0.0);
  }
#else
  for (int k = 0; k < n; k++) {
    _eigs[k] = w[k];
  }
  for (int i = 0; i < n * n; i++) {
    _eigvecs[i] = mat[i];
  }
#endif  // TACS_USE_COMPLEX

  delete[] mat;
  delete[] work;
  delete[] w;
}

/*
  Create the symmetric eigenvalue problem solver

  This object uses a Lanczos method to reduce the symmetric eigenvalue
  problem to a small-rank symmetric tridiagonal problem that can be
  solved efficiently using LAPACK routines.

  The symmetric eigenproblem could be derived from either a regular or
  generalized eigenproblem so the inner product is generalized to
  accomodate different options.

  input:
  Op:          the eigenvalue problem operator
  max_iters:   the maximum number of iterations before giving up
  ortho_type:  the type of orthogonalization to use FULL or LOCAL
*/
SEP::SEP(EPOperator *_Op, int _max_iters, OrthoType _ortho_type,
         TACSBcMap *_bcs, int _restart_size) {
  // Store the pointer to the eigenproblem operator
  Op = _Op;
  Op->incref();

  // Set the boundary conditions
  bcs = _bcs;
  if (bcs) {
    bcs->incref();
  }

  // Store the information about the subspace vectors
  max_iters = _max_iters;
  ortho_type = _ortho_type;

  // Thick-restart Lanczos (SPEC Item 5): restart_size <= 0 or
  // restart_size >= max_iters means "disabled" -- alloc_size == max_iters,
  // byte-for-byte the same allocation as before this feature. Otherwise
  // the live basis never grows past restart_size vectors, so allocation
  // (and therefore memory use, acceptance criterion 1) is bounded by
  // restart_size independent of max_iters/numEigs.
  //
  // Restart only ever applies to the FULL-orthogonalization branch of
  // solve() (SPEC: "Applies only to the FULL-orthogonalization branch");
  // LOCAL falls back to today's unrestarted, verbatim max_iters-sized loop.
  // Tying the allocation decision to the *constructor's* ortho_type keeps
  // that fallback correct at the allocation level, not just the loop-bound
  // level: a LOCAL-constructed SEP with restart_size > 0 allocates exactly
  // as it always has. (A later setOrthoType() call switching between
  // LOCAL/FULL after construction does not retroactively resize this
  // allocation; no caller in this codebase does that.)
  restart_size = _restart_size;
  use_thick_restart =
      (restart_size > 0 && restart_size < max_iters && ortho_type == FULL);
  alloc_size = use_thick_restart ? restart_size : max_iters;
  restart_keep = 0;

  Q = new TACSVec *[alloc_size + 1];

  // The coefficients of the Lanczos tridiagonal system
  Alpha = new TacsScalar[alloc_size];
  Beta = new TacsScalar[alloc_size];

  // Thick-restart arrowhead coupling (SPEC step 4) -- only ever populated
  // when use_thick_restart is true, but allocating it unconditionally at
  // the same alloc_size as Alpha/Beta keeps destruction unconditional too.
  Sigma = new TacsScalar[alloc_size];
  memset(Sigma, 0, alloc_size * sizeof(TacsScalar));

  // The eigenvalues and eigenvectors of the tridiagonal system
  eigs = new TacsScalar[alloc_size];
  eigvecs = new TacsScalar[alloc_size * alloc_size];

  // Permutation of the order of the eigenvalues
  perm = new int[alloc_size];

  // Defense-in-depth (VALIDATION Decision 3): the neigs_computed guard is
  // what actually prevents an out-of-bounds read, but zero-initializing
  // here ensures that any future code path that slips past the entry guard
  // in solve() reads deterministic zeros/-1 rather than arbitrary heap
  // contents.
  memset(eigs, 0, alloc_size * sizeof(TacsScalar));
  memset(eigvecs, 0, alloc_size * alloc_size * sizeof(TacsScalar));
  for (int i = 0; i < alloc_size; i++) {
    perm[i] = -1;
  }

  // Default values for convergence tests/sorting of the eigenvalues
  tol = 1e-12;
  spectrum = SMALLEST;
  neigvals = 4;
  niters = -1;
  neigs_computed = 0;

  // Create the vectors required for the Lanczos subspace
  for (int i = 0; i < alloc_size + 1; i++) {
    Q[i] = Op->createVec();
    Q[i]->incref();
  }
}

/*
  Deallocate all the information stored in the eigenproblem
*/
SEP::~SEP() {
  Op->decref();
  for (int i = 0; i < alloc_size + 1; i++) {
    Q[i]->decref();
  }
  delete[] Q;

  if (bcs) {
    bcs->decref();
  }

  delete[] Alpha;
  delete[] Beta;
  delete[] Sigma;
  delete[] eigs;
  delete[] eigvecs;
  delete[] perm;
}

/*
  Set the orthogonalization type
*/
void SEP::setOrthoType(enum OrthoType _ortho_type) { ortho_type = _ortho_type; }

/*
  Set the tolerances to use, the desired spectrum, and the number of
  eigenvalues that are requested in the solve
*/
void SEP::setTolerances(double _tol, enum EigenSpectrum _spectrum,
                        int _neigvals) {
  tol = _tol;
  spectrum = _spectrum;
  neigvals = _neigvals;
}

/*
  Set the thick-restart basis size (SPEC Item 5).

  Note: this does not reallocate Q/Alpha/Beta/Sigma/eigs/eigvecs/perm --
  those are sized once at construction time off whatever restart_size was
  passed to the constructor (into the alloc_size member, and the
  use_thick_restart decision derived from it), and are never resized
  afterward. Calling this setter after construction changes only the
  mutable restart_size member; it has no effect on a FULL-orthogonalization
  solve()'s restart trigger (which reads the frozen use_thick_restart/
  alloc_size decision made at construction time), and the only place
  solve() reads restart_size post-construction is the LOCAL branch's
  fallback-message condition. This setter exists so a documented accessor
  is available alongside setOrthoType/setTolerances per SPEC's interface
  list -- prefer passing restart_size directly to the constructor.
*/
void SEP::setRestartSize(int _restart_size) { restart_size = _restart_size; }

/*
  Reset the underlying operator
*/
void SEP::setOperator(EPOperator *_Op) {
  _Op->incref();
  if (Op) {
    Op->decref();
  }
  Op = _Op;
}

/*
  Solve the eigenvalue problem using the Lanczos method with full or
  local orthogonalization.

  This method uses a spectral shift approach to solve a generalized
  eigenvalue problem with a Lanczos method. The method generates a
  series or orthonormal vectors with respect to a given inner product.
*/
int SEP::solve(KSMPrint *ksm_print, KSMPrint *ksm_file) {
  // Misconfiguration guard: the default neigvals = 4 (set in the
  // constructor) applies even if the caller never calls setTolerances(),
  // so this check must fire regardless of which setter was or wasn't
  // called -- placing it only in setTolerances() would miss the case where
  // SEP is constructed with max_iters < 4 and neigvals is never changed
  // from its default.
  //
  // Checked against alloc_size, not max_iters: alloc_size == max_iters
  // whenever thick restart is disabled (restart_size <= 0) or the
  // orthogonalization type is LOCAL, so this is the same check as before
  // for every existing caller. But when restart_size > 0 with FULL
  // orthogonalization, alloc_size == restart_size can be smaller than
  // max_iters -- if neigvals > alloc_size in that case, checkConverged()
  // never reaches n >= neigvals (GSEP.cpp's early-return there), so
  // neigs_computed stays 0 and perm[] stays at its constructor
  // -1-initialized sentinel for every slot; the finiteness gate near the
  // end of this function would then read eigs[perm[k]] == eigs[-1], an
  // out-of-bounds heap read (reproduced under valgrind during review).
  if (neigvals > alloc_size) {
    fprintf(stderr,
            "SEP::solve() Error: the live basis size (%d, either max_iters "
            "or the smaller restart_size when thick restart is active) "
            "must be >= neigvals (%d); no eigenvalues were computed.\n",
            alloc_size, neigvals);
    niters = 0;
    neigs_computed = 0;
    return -1;
  }

  // Select the initial vector randomly
  Q[0]->setRand();
  if (bcs) {
    Q[0]->applyBCs(bcs);
  }

  // Normalize the first vector
  TacsScalar norm = sqrt(Op->dot(Q[0], Q[0]));
  Q[0]->scale(1.0 / norm);

  // Set at break time in either branch below when checkConverged()
  // succeeds -- used, together with the finiteness gate below, to
  // determine the solve_flag returned by this function.
  int converged_early = 0;

  if (ortho_type == LOCAL) {
    // Thick restart (SPEC Item 5) only applies to the FULL-orthogonalization
    // branch -- LOCAL falls back to today's unrestarted loop verbatim, with
    // one informational message (not a silent ignore).
    if (restart_size > 0) {
      fprintf(stderr,
              "SEP::solve(): thick restart is only implemented for FULL "
              "reorthogonalization; ignoring restart_size and using the "
              "unrestarted LOCAL path\n");
    }

    // Only local orthogonalization is utilized. This code does not
    // orthogonalize the vector against previous vectors.
    int i = 0;
    for (; i < max_iters; i++) {
      // First compute U = A*Q[i] - Q[i-1]*Beta[i]
      Op->mult(Q[i], Q[i + 1]);
      if (bcs) {
        Q[i + 1]->applyBCs(bcs);
      }

      if (i > 0) {
        Q[i + 1]->axpy(-Beta[i - 1], Q[i - 1]);
      }

      // Next, compute the new alpha term
      Alpha[i] = Op->dot(Q[i], Q[i + 1]);
      Q[i + 1]->axpy(-Alpha[i], Q[i]);

      // Compute the new beta term in the Lanczos sequence
      Beta[i] = sqrt(Op->dot(Q[i + 1], Q[i + 1]));
      Q[i + 1]->scale(1.0 / Beta[i]);

      // Check if the desired eigenvalues have converged
      if (checkConverged(Alpha, Beta, i + 1)) {
        niters = i + 1;
        converged_early = 1;
        break;
      }
    }

    // Readjust the max number of iterations
    if (i == max_iters) {
      niters = max_iters;
    }
  } else {
    // Perform a full orthogonalization using modified Gram-Schmidt, with
    // Wu & Simon (2000) thick restart (SPEC lines 783-816) once the live
    // basis reaches alloc_size (== restart_size when use_thick_restart).
    //
    // total_iters tracks the *cumulative* number of Lanczos steps
    // (matvecs) across all restarts -- the real budget max_iters gates on
    // (SPEC step 6); i tracks the *live* basis index and can be reset
    // backward by a restart, unlike total_iters.
    int i = 0;
    int total_iters = 0;
    for (;;) {
      // Compute the new vector using the provided operator
      Op->mult(Q[i], Q[i + 1]);
      if (bcs) {
        Q[i + 1]->applyBCs(bcs);
      }

      for (int j = i; j >= 0; j--) {
        TacsScalar h = Op->dot(Q[i + 1], Q[j]);
        Q[i + 1]->axpy(-h, Q[j]);

        // Store the diagonal term (and discard all other terms which
        // will only be non-zero due to numerical issues -- except once a
        // restart has occurred, in which case the terms for j < restart_
        // keep are the genuine Wu-Simon arrowhead coupling, already
        // captured analytically in Sigma[] at restart time (see below),
        // not re-derived from these h values)
        if (j == i) {
          Alpha[i] = h;
        }
      }

      // Evalute the sub-digonal
      Beta[i] = sqrt(Op->dot(Q[i + 1], Q[i + 1]));
      Q[i + 1]->scale(1.0 / Beta[i]);
      total_iters++;

      // Check if the desired eigenvalues have converged
      if (checkConverged(Alpha, Beta, i + 1)) {
        niters = i + 1;
        converged_early = 1;
        break;
      }

      // Overall budget exhausted -- checked against total_iters, not the
      // live basis index, so a restart-enabled solve gets the same total
      // "give up after this many Lanczos steps" guarantee as the
      // unrestarted path (SPEC step 6).
      if (total_iters >= max_iters) {
        niters = i + 1;
        break;
      }

      // Thick-restart trigger: the live basis has grown to alloc_size
      // without converging.
      if (use_thick_restart && i + 1 == alloc_size) {
        int n = i + 1;

        // Step 2: keep the best `keep` Ritz vectors -- a small multiple
        // of neigvals, not just neigvals itself, so the restarted basis
        // has room to reconverge without immediately re-triggering
        // (SPEC step 2). eigs/eigvecs/perm were already computed by the
        // checkConverged() call immediately above (SPEC step 1) -- reused
        // here, not recomputed.
        int new_keep =
            restart_size - 1 < 2 * neigvals ? restart_size - 1 : 2 * neigvals;
        if (new_keep < 1) {
          new_keep = 1;
        }

        // Cluster-aware extension: SPEC's edge-case section motivates the
        // 2*neigvals margin above as slack "around the requested cutoff"
        // for near-degenerate pairs -- but that margin only protects the
        // *neigvals* cutoff, not the *keep* cutoff itself. If a near-
        // degenerate cluster of Ritz values happens to straddle the
        // keep/discard boundary (i.e. the (new_keep)-th smallest Ritz
        // value is itself within a tight relative gap of the
        // (new_keep-1)-th, the one just retained), discarding the cluster
        // member at position new_keep can permanently lose an eigenvalue
        // direction the retained subspace no longer spans -- confirmed
        // empirically (a rare, seed-dependent ~1-2% occurrence on
        // plate.bdf's degenerate spectrum during this feature's own
        // verification) as a genuine restart-vs-legacy disagreement, not a
        // reduction-math bug. Extending keep by one more slot whenever the
        // boundary would split such a cluster (bounded by a small cap to
        // avoid unbounded growth on a pathologically all-degenerate
        // spectrum, and by n - 1 to always leave room for at least one
        // fresh Lanczos vector after the restart) is the standard
        // mitigation.
        const double cluster_rel_gap = 1e-6;
        const int cluster_extend_cap = 10;
        for (int extended = 0;
             new_keep < n - 1 && extended < cluster_extend_cap; extended++) {
          TacsScalar a = eigs[perm[new_keep - 1]];
          TacsScalar b = eigs[perm[new_keep]];
          double denom = TacsRealPart(fabs(a)) > TacsRealPart(fabs(b))
                             ? TacsRealPart(fabs(a))
                             : TacsRealPart(fabs(b));
          if (denom < 1.0) {
            denom = 1.0;
          }
          double gap = TacsRealPart(fabs(b - a)) / denom;
          if (gap > cluster_rel_gap) {
            break;
          }
          new_keep++;
        }

        // Step 3: form the new basis vectors as dense linear combinations
        // of the current basis (O(n * keep)). Computed into temporary
        // vectors first -- Q[0..new_keep) cannot be overwritten in place
        // while other Qnew[j] combinations still need their original
        // contents.
        TACSVec **Qnew = new TACSVec *[new_keep];
        for (int j = 0; j < new_keep; j++) {
          Qnew[j] = Op->createVec();
          Qnew[j]->incref();
          Qnew[j]->zeroEntries();
          for (int k = 0; k < n; k++) {
            Qnew[j]->axpy(eigvecs[perm[j] * n + k], Q[k]);
          }
        }

        // Step 4: rebuild the reduced recurrence. Alpha_new[j] is the
        // retained Ritz value (the projected operator is diagonal in its
        // own eigenbasis); Sigma[j] is the single coupling coefficient
        // carried forward into the arrowhead border, computed from the
        // *pre-restart* Beta[n-1] and eigvecs -- both still valid at this
        // point since Q[0..new_keep) have not yet been overwritten.
        TacsScalar beta_old_last = Beta[n - 1];
        for (int j = 0; j < new_keep; j++) {
          Alpha[j] = eigs[perm[j]];
          Sigma[j] = beta_old_last * eigvecs[perm[j] * n + (n - 1)];
        }

        // Copy the new basis vectors into place now that every Qnew[j]
        // combination has been fully formed.
        for (int j = 0; j < new_keep; j++) {
          Q[j]->copyValues(Qnew[j]);
          Qnew[j]->decref();
        }
        delete[] Qnew;

        // Q[n] is Wu-Simon's already-computed, already-orthogonal "next"
        // Lanczos vector -- it becomes live basis index `new_keep`
        // directly (it must NOT be discarded/recomputed via a fresh
        // matvec off one of the retained Ritz vectors: doing so would
        // silently lose the arrowhead coupling this restart just encoded
        // into Sigma[], reproducing the false-convergence bug documented
        // in HANDOFF-impl.md). n != new_keep always (new_keep < n by
        // construction above), so this is a copy between distinct vector
        // objects, not a self-copy, and Q[n] has not been touched by the
        // Qnew assembly/copy above (all of which only ever touch indices
        // < new_keep < n).
        Q[new_keep]->copyValues(Q[n]);

        restart_keep = new_keep;

        // Step 5: continue the FULL Gram-Schmidt loop from index `keep`
        // onward. Set i one below new_keep so the loop's unconditional
        // i++ below lands on i = new_keep next iteration, where Q[new_
        // keep] already holds the valid, normalized residual vector.
        i = new_keep - 1;
      }

      i++;
    }
  }

  // Compute the norm of the last vector in the inner product
  TacsScalar er = Op->errorNorm(Q[niters]);

  // Print out a summary of the eigenvalues and errors
  if (ksm_print) {
    char line[256];
    snprintf(line, sizeof(line), "%3s %18s %18s %10s\n", " ", "eigenvalue",
             "shift-invert eig", "error");
    ksm_print->print(line);

    for (int i = 0; i < niters; i++) {
      int index = perm[i];
      char line[256];
      snprintf(line, sizeof(line), "%3d %18.10e %18.10e %10.3e\n", i,
               TacsRealPart(Op->convertEigenvalue(eigs[index])),
               TacsRealPart(eigs[index]),
               fabs(TacsRealPart(Beta[niters - 1] *
                                 eigvecs[index * niters + (niters - 1)] * er)));
      ksm_print->print(line);
    }
  }
  // Print the iteration count to file
  if (ksm_file) {
    char line[256];
    snprintf(line, sizeof(line), "%2d\n", niters);
    ksm_file->print(line);
  }

  // Final solve_flag gate: a solve that broke out of the loop early is
  // only truly converged if every one of the requested eigenvalues is also
  // finite (closes the JD-style silent-NaN gap for the Lanczos path too).
  int converged = converged_early;
  for (int k = 0; k < neigvals && k < niters; k++) {
    TacsScalar val = Op->convertEigenvalue(eigs[perm[k]]);
    if (!TacsIsFinite(val)) {
      converged = 0;
      break;
    }
  }
  return converged ? 1 : 0;
}

/*!
  Extract the n-th eigenvalue from the probelm.
*/
TacsScalar SEP::extractEigenvalue(int n, TacsScalar *error) {
  if (n < 0 || n >= niters || n >= neigs_computed) {
    fprintf(stderr, "Eigenvalue out of range\n");
    *error = -1.0;
    return 0.0;
  }

  n = perm[n];

  TacsScalar er = Op->errorNorm(Q[niters]);
  *error = fabs(
      TacsRealPart(Beta[niters - 1] * eigvecs[n * niters + (niters - 1)] * er));

  return Op->convertEigenvalue(eigs[n]);
}

/*!
  Extract the n-th eigenvector from the matrix.

  Compute the eigenvector and store it in ans.  Return the eigenvalue
  and the error computed as error = || A x_n - lambda_n x_n ||_2 via a
  pointer.
*/
TacsScalar SEP::extractEigenvector(int n, TACSVec *ans, TacsScalar *error) {
  if (n < 0 || n >= niters || n >= neigs_computed) {
    fprintf(stderr, "Eigenvector out of range\n");
    *error = -1.0;
    ans->zeroEntries();
    return 0.0;
  }

  n = perm[n];

  ans->zeroEntries();
  for (int i = 0; i < niters; i++) {
    ans->axpy(eigvecs[n * niters + i], Q[i]);
  }

  TacsScalar er = Op->errorNorm(Q[niters]);
  *error = fabs(Beta[niters - 1] * eigvecs[n * niters + (niters - 1)] * er);

  return Op->convertEigenvalue(eigs[n]);
}

/*!
  Find the Frobenius norm of the matrix:

  I - Q^{T} Q

  which should be zero if the vectors are completely orthogonal
*/
TacsScalar SEP::checkOrthogonality() {
  // Check to see if all the vectors are orthogonal
  TacsScalar norm = TacsScalar(0.0);

  for (int i = 0; i < niters; i++) {
    for (int j = 0; j < i; j++) {
      TacsScalar aij = Op->dot(Q[i], Q[j]);
      norm += 2.0 * aij * aij;
    }

    TacsScalar aii = 1.0 - Op->dot(Q[i], Q[i]);
    norm += aii * aii;
  }

  return sqrt(norm);
}

/*!
  Print the dot product of the Krylov subspace to the screen
*/
void SEP::printOrthogonality() {
  for (int i = 0; i < niters; i++) {
    for (int j = 0; j < i; j++) {
      TacsScalar aij = Op->dot(Q[i], Q[j]);
      printf("%8.1e ", TacsRealPart(aij));
    }

    TacsScalar aii = Op->dot(Q[i], Q[i]);
    printf("%8.1e \n", TacsRealPart(aii));
  }
}

/*!
  Sort the values based on the actual eigenvalues not their
  transformed values.

  This uses insertion sort. The eigenvalues are left in their same
  ordering - this is required so that the corresponding eigenvectors
  match. The permutation array is altered so that the array indexed by
  p[i] => eigs[ p[i] ] is sorted in ascending order.
*/
void SEP::sortEigenvalues(TacsScalar *values, int neigs, int *p) {
  // The default ordering
  for (int i = 0; i < neigs; i++) {
    p[i] = i;
  }

  // Set the flags based on the sorting criteria: Use absolute value
  // if we only care about the magnitude and sort ascending if we only
  // care about the smallest values
  int use_abs =
      (spectrum == SMALLEST_MAGNITUDE || spectrum == LARGEST_MAGNITUDE);
  int sort_ascending = (spectrum == SMALLEST_MAGNITUDE || spectrum == SMALLEST);

  // Sort the array using insertion sort
  for (int i = 0; i < neigs; i++) {
    // Convert the transformed eigenvalue into the correct
    // range
    TacsScalar eig_new = Op->convertEigenvalue(values[p[i]]);

    // Take the absolute value of the eigenvalue
    if (use_abs) {
      if (TacsRealPart(eig_new) < 0.0) {
        eig_new *= -1.0;
      }
    }

    // Find out where we should place this within the sorted
    // array
    int j = i - 1;
    for (; j >= 0; j--) {
      // Convert the j-th eigenvalue
      TacsScalar eig_j = Op->convertEigenvalue(values[p[j]]);
      if (use_abs) {
        if (TacsRealPart(eig_j) < 0.0) {
          eig_j *= -1.0;
        }
      }

      // If this is the right place, place the new index here
      if (sort_ascending && TacsRealPart(eig_new) > TacsRealPart(eig_j)) {
        break;
      } else if (!sort_ascending &&
                 TacsRealPart(eig_new) < TacsRealPart(eig_j)) {
        break;
      } else {
        // This is not the right place, just move the indices back
        p[j + 1] = p[j];
      }
    }

    // Place the index in the sorted array
    p[j + 1] = i;
  }
}

/*!
  Perform a convergence test.

  Check whether the eigenvalues at the appropriate end of the spectrum
  have converged based on either a relative or absolute tolerance.xx
*/
int SEP::checkConverged(TacsScalar *A, TacsScalar *B, int n) {
  // If we haven't iterated n-times, we can quit immediately
  if (n < neigvals) {
    return 0;
  }

  // Compute the eigenvalues and eigenvectors of the reduced problem over
  // the current n-sized basis. Before any thick restart has occurred (or
  // when thick restart is disabled entirely), A/B describe a plain
  // symmetric tridiagonal matrix, solved via the specialized
  // ComputeEigsTriDiag path exactly as before this feature (byte-for-byte
  // unchanged code path). Once thick restart is active, the reduced
  // problem may instead carry a Wu-Simon arrowhead border (SPEC step 4),
  // so ComputeEigsDense is used unconditionally for every checkConverged()
  // call on a restart-enabled solve -- including before the first restart
  // actually triggers, when restart_keep == 0 and the assembled matrix is
  // itself just the tridiagonal case, solved via a small dense LAPACK
  // solve instead of the tridiagonal-specialized one (agreement between
  // the two is exactly what the Task 5.1 smoke test / Task 5.3 agreement
  // test verify).
  TacsScalar beta = B[n - 1];
  if (use_thick_restart) {
    ComputeEigsDense(n, restart_keep, A, B, Sigma, eigs, eigvecs);
  } else {
    ComputeEigsTriDiag(n, A, B, eigs, eigvecs);
  }

  // Find the permutation which sorts the matrix in the desired
  // order
  sortEigenvalues(eigs, n, perm);

  // Record that eigs/eigvecs/perm[0..n-1] have now actually been populated
  neigs_computed = n;

  // Check for convergence of each of the desired eigenvalues
  int is_converged = 1;
  for (int k = 0; k < neigvals; k++) {
    // Read off the index
    int index = perm[k];
    TacsScalar er = Op->errorNorm(Q[n]);

    // Read out the predicted error for the eigenvector
    TacsScalar eig_err = fabs(beta * eigvecs[index * n + (n - 1)] * er);
    // NOTE: written as !(<= tol), not > tol -- IEEE 754 defines NaN > tol as
    // false, so the naive ">" polarity silently treats a NaN residual as
    // converged. This form correctly fails NaN residuals as unconverged.
    if (!(TacsRealPart(eig_err) <= tol)) {
      is_converged = 0;
      break;
    }
  }

  return is_converged;
}
