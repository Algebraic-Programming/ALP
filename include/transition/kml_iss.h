

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

int KmlIssCgInitSI( KmlSolverTask **, int, const float *, const int *, const int * );
int KmlIssCgInitDI( KmlSolverTask **, int, const double *, const int *, const int * );


int KmlIssCgSetUserPreconditionerSI(KmlSolverTask **, void *, int (*)(void *, float *) );
int KmlIssCgSetUserPreconditionerDI(KmlSolverTask **, void *, int (*)(void *, double *) );

typedef int KML_SOLVER_PARAM;


int KmlIssCgSetSII(KmlSolverTask **, KML_SOLVER_PARAM, const int *, int );
int KmlIssCgSetDII(KmlSolverTask **, KML_SOLVER_PARAM, const int *, int );
int KmlIssCgSetSIS(KmlSolverTask **, KML_SOLVER_PARAM, const float *, int );
int KmlIssCgSetDID(KmlSolverTask **, KML_SOLVER_PARAM, const double *, int );


int KmlIssCgAnalyzeSI(KmlSolverTask ** );
int KmlIssCgAnalyzeDI(KmlSolverTask ** );


int KmlIssCgFactorizeSI(KmlSolverTask ** );
int KmlIssCgFactorizeDI(KmlSolverTask ** );


int KmlIssCgSolveSI(KmlSolverTask **, int, float *, int, const float *, int );
int KmlIssCgSolveDI(KmlSolverTask **, int, double *, int, const double *, int );


int KmlIssCgGetSII(KmlSolverTask **, KML_SOLVER_PARAM, int *, int );
int KmlIssCgGetDII(KmlSolverTask **, KML_SOLVER_PARAM, int *, int );
int KmlIssCgGetSIS(KmlSolverTask **, KML_SOLVER_PARAM, float *, int );
int KmlIssCgGetDID(KmlSolverTask **, KML_SOLVER_PARAM, double *, int );


int KmlIssCgCleanSI(KmlSolverTask **);
int KmlIssCgCleanDI(KmlSolverTask **);


#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end ifdef _H_ALP_KML_ISS
