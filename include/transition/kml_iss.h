
/*
 *   Copyright 2024 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file
 *
 * This exposes the same interface of KML_SOLVER, as documented in
 *
 * https://www.hikunpeng.com/document/detail/en/kunpengaccel/math-lib/devg-kml/kunpengaccel_kml_16_0287.html
 *
 * @author Alberto Scolari
 * @date 15/01/2024
 */

#ifndef _H_ALP_KML_ISS
#define _H_ALP_KML_ISS

#include <stddef.h> // for size_t

#include "solver.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void KmlSolverTask;

// KML solver library error values
#define KMLSS_NO_ERROR 0
#define KMLSS_NONZERO_INDEXING 1
#define KMLSS_MISSING_DIAGONAL_ELEMENT 2
#define KMLSS_ZERO_DIAGONAL_ELEMENT 3
#define KMLSS_NO_MEMORY 4
#define KMLSS_NULL_ARGUMENT 5
#define KMLSS_BAD_DATA_SIZE 6
#define KMLSS_BAD_DATA 7
#define KMLSS_BAD_SELECTOR 8
#define KMLSS_BAD_N 9
#define KMLSS_BAD_NB 10
#define KMLSS_BAD_LDX 11
#define KMLSS_BAD_LDB 12
#define KMLSS_BAD_HANDLE 13
#define KMLSS_BAD_PRECONDITIONER 14
#define KMLSS_INVALID_CALL_ORDER 15
#define KMLSS_BAD_MATRIX_FORMAT 16
#define KMLSS_REORDERING_PROBLEM 1001
#define KMLSS_ZERO_PARTIAL_PIVOT 1002
#define KMLSS_INTERNAL_ERROR 1000001
#define KMLSS_NOT_IMPLEMENTED 1000002

// error value for errors non prescribed by KML
#define KMLSS_OTHER_ERROR 2000002

// KML solvers parameters value
#define KMLSS_FILL_IN 0
#define KMLSS_PERM 1
#define KMLSS_REFINEMENT_MAX_STEPS 2
#define KMLSS_THRESHOLD 3
#define KMLSS_MAX_ITERATION_COUNT 4
#define KMLSS_RESTART_PARAM 5
#define KMLSS_ITERATION_COUNT 6
#define KMLSS_TOLERANCE 7
#define KMLSS_INCREASE_ACCURACY 8
#define KMLSS_PRECONDITIONER_TYPE 9
#define KMLSS_ORTHOGONALIZATION_TYPE 10
#define KMLSS_BOOST_THRESHOLD 11
#define KMLSS_SCALING_TYPE 12
#define KMLSS_MATRIX_FORMAT 13
#define KMLSS_REFINEMENT_STEPS 14
#define KMLSS_REFINEMENT_TOLERANCE_LEVEL 15
#define KMLSS_REFINEMENT_RESIDUAL 16
#define KMLSS_PIVOTING_THRESHOLD 17
#define KMLSS_MATCHING_TYPE 18


#define KML_CG_PREFIXED( name ) KmlIssCg ## name

// Initialization routines to create a solver task

int KML_CG_PREFIXED( InitSI )( KmlSolverTask **, int, const float *, const int *, const int * );
int KML_CG_PREFIXED( InitDI )( KmlSolverTask **, int, const double *, const int *, const int * );

// Setters for preconditioner

int KML_CG_PREFIXED( SetUserPreconditionerSI )( KmlSolverTask **, void *, int (*)( void *, float * ) );
int KML_CG_PREFIXED( SetUserPreconditionerDI )( KmlSolverTask **, void *, int (*)( void *, double * ) );

// Setters for solver parameters

typedef int KML_SOLVER_PARAM;
int KML_CG_PREFIXED( SetSII )( KmlSolverTask **, KML_SOLVER_PARAM, const int *, int );
int KML_CG_PREFIXED( SetDII )( KmlSolverTask **, KML_SOLVER_PARAM, const int *, int );
int KML_CG_PREFIXED( SetSIS )( KmlSolverTask **, KML_SOLVER_PARAM, const float *, int );
int KML_CG_PREFIXED( SetDID )( KmlSolverTask **, KML_SOLVER_PARAM, const double *, int );

// Analyze problem before solving

int KML_CG_PREFIXED( AnalyzeSI )( KmlSolverTask ** );
int KML_CG_PREFIXED( AnalyzeDI )( KmlSolverTask ** );

// Analyze a sparse matrix and change it storage mode

int KML_CG_PREFIXED( FactorizeSI )( KmlSolverTask ** );
int KML_CG_PREFIXED( FactorizeDI )( KmlSolverTask ** );

// Run the solver

int KML_CG_PREFIXED( SolveSI )( KmlSolverTask **, int, float *, int, const float *, int );
int KML_CG_PREFIXED( SolveDI )( KmlSolverTask **, int, double *, int, const double *, int );

// Get parameters after solving

int KML_CG_PREFIXED( GetSII )( KmlSolverTask **, KML_SOLVER_PARAM, int *, int );
int KML_CG_PREFIXED( GetDII )( KmlSolverTask **, KML_SOLVER_PARAM, int *, int );
int KML_CG_PREFIXED( GetSIS )( KmlSolverTask **, KML_SOLVER_PARAM, float *, int );
int KML_CG_PREFIXED( GetDID )( KmlSolverTask **, KML_SOLVER_PARAM, double *, int );

// De-allocate data and destroy the solver task

int KML_CG_PREFIXED( CleanSI )( KmlSolverTask ** );
int KML_CG_PREFIXED( CleanDI )( KmlSolverTask ** );

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end ifdef _H_ALP_KML_ISS
