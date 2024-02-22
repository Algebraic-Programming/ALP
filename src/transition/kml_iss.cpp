
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
 * This implements the routines of KML_SOLVER, as in
 *
 * https://www.hikunpeng.com/document/detail/en/kunpengaccel/math-lib/devg-kml/kunpengaccel_kml_16_0287.html
 *
 * @author Alberto Scolari
 * @date 26/01/2024
 */

#include <graphblas.hpp>

#include <kml_iss.h>
#include <solver.h>

#include <assert.h>


#define KMLSS_SET_ARG( nd, param, case_val, sparse_err_t_val )                      \
	if( nd != 1 ) { return KMLSS_BAD_DATA_SIZE; }                               \
	int err = KMLSS_NO_ERROR;                                                   \
	switch ( param ) {                                                          \
		case case_val:                                                      \
		{                                                                   \
			const sparse_err_t r = sparse_err_t_val;                    \
			if( r != NO_ERROR ) {                                       \
				err = ( r == NULL_ARGUMENT ) ?                      \
					KMLSS_NULL_ARGUMENT : KMLSS_INTERNAL_ERROR; \
			}                                                           \
			break;                                                      \
		}                                                                   \
	default:                                                                    \
		err = KMLSS_BAD_SELECTOR;                                           \
	}                                                                           \
	return err;

// do not propagate FAILED in case of failed convergence (KML convention)
#define KMLSS_SOLVE( nb, sparse_err_t_val )          \
	if( nb != 1 ) { return KMLSS_BAD_NB; }       \
	const sparse_err_t r = sparse_err_t_val;     \
	if( r == FAILED ) { return KMLSS_NO_ERROR; } \
	int err = KMLSS_NO_ERROR;                    \
	if( r != NO_ERROR ) {                        \
		err = ( r == NULL_ARGUMENT )         \
			? KMLSS_NULL_ARGUMENT        \
			: KMLSS_INTERNAL_ERROR;      \
	}                                            \
	return err;

#define KMLSS_GET_ARG( nd, param, case_val, sparse_err_t_val, err ) \
	if( nd != 1 ) { return KMLSS_BAD_DATA_SIZE; }               \
	int err = KMLSS_NO_ERROR;                                   \
	switch ( param ) {                                          \
		case case_val:                                      \
		{                                                   \
			const sparse_err_t r = sparse_err_t_val;    \
			if( r != NO_ERROR ) {                       \
				err = ( r == NULL_ARGUMENT )        \
					? KMLSS_NULL_ARGUMENT       \
					: KMLSS_INTERNAL_ERROR;     \
			}                                           \
			break;                                      \
		}                                                   \
		default:                                            \
			err = KMLSS_BAD_SELECTOR;                   \
	}

/**
 * Converts a sparse_err_t into an int.
 */
static int sparse_err_t_2_int( const sparse_err_t err ) {
	int result = 0;
	switch ( err ) {
		case NO_ERROR:
		{
			result = KMLSS_NO_ERROR;
			break;
		}
		case NULL_ARGUMENT:
		{
			result = KMLSS_NULL_ARGUMENT;
			break;
		}
		case ILLEGAL_ARGUMENT:
		{
			result = KMLSS_BAD_N;
			break;
		}
		case OUT_OF_MEMORY:
		{
			result = KMLSS_NO_MEMORY;
			break;
		}
		case FAILED:
		{
			result = KMLSS_OTHER_ERROR;
			break;
		}
		case UNKNOWN:
		{
			result = KMLSS_INTERNAL_ERROR;
			break;
		}
	}
	return result;
}


int KML_CG_PREFIXED( InitSI )(
	KmlSolverTask **handle, int n, const float *a,
	const int *ja, const int *ia
) {
	sparse_err_t err = sparse_cg_init_sii( handle, n, a, ja, ia );
	return sparse_err_t_2_int( err );
}

int KML_CG_PREFIXED( InitDI )(
	KmlSolverTask **handle, int n, const double *a,
	const int *ja, const int *ia
) {
	sparse_err_t err = sparse_cg_init_dii( handle, n, a, ja, ia );
	return sparse_err_t_2_int( err );
}

/**
 * The data object for preconditioners.
 *
 * Enables translation between sparse_cg_t preconditioners and KML
 * preconditioners.
 */
template< typename T >
struct sparse_t_precond_data {

	/** The system size. */
	size_t n;

	/** The KML user preconditioner. */
	int (*kml_preconditioner)( void *, T * );

	/** The KML user data. */
	void * kml_data;

};

/**
 * The sparse_cg_t preconditioner function that calls the KML user-defined
 * preconditioner.
 */
template< typename T >
static int sparse_t_preconditioner(
	T * const out,
	const T * const in,
	void * const data_p
) {
	// get data
	assert( data_p );
	if( data_p == NULL ) { return 10; }
	const struct sparse_t_precond_data< T > &data =
		*reinterpret_cast< struct sparse_t_precond_data< T > * >(
			data_p);

	// we'll be a little bit smart about converting this signature to an in-place
	// one by relying on ALP primitives that will auto-parallelise the necessary
	// copy
	grb::Vector< T > alp_out =
		grb::internal::template wrapRawVector< T >( data.n, out );
	const grb::Vector< T > alp_in =
		grb::internal::template wrapRawVector< T >( data.n, in );
	grb::RC rc = grb::set< grb::descriptors::dense >( alp_out, alp_in );
	rc = rc ? rc : grb::wait();
	if( rc != grb::RC::SUCCESS ) { return 20; }

	// now out equals in and we can call the KML preconditioner signature
	return (data.kml_preconditioner)( data.kml_data, out );
}

int KML_CG_PREFIXED( SetUserPreconditionerSI )(
	KmlSolverTask ** handle_p,
	void * data, int (*preconditioner)( void *, float * )
) {
	int (*sparse_t_function_p) (
		float * const,
		const float * const,
		void * const
	) = nullptr;
	void * sparse_t_data_p = nullptr;
	sparse_cg_handle_t handle = *handle_p;
	size_t size;

	sparse_err_t rc = sparse_cg_get_preconditioner_sii(
		handle, &sparse_t_function_p, &sparse_t_data_p );
	rc = rc ? rc : sparse_cg_get_size_sii( handle, &size );
	if( rc != sparse_err_t::NO_ERROR ) { return sparse_err_t_2_int( rc ); }

	if( preconditioner == nullptr ) {
		// in this case, no preconditioning is requested
		// if we had data stored, delete it
		if( sparse_t_data_p != nullptr && preconditioner == nullptr ) {
			{
				struct sparse_t_precond_data< float > * const sparse_t_data =
					reinterpret_cast< sparse_t_precond_data< float > * >( sparse_t_data_p );
				delete sparse_t_data;
			}
			sparse_t_data_p = nullptr;
		}
	} else {
		// in this case, preconditioning is requested
		sparse_t_function_p = &(sparse_t_preconditioner< float >);
		// if we had no data already stored, create it
		if( sparse_t_data_p == nullptr ) {
			try {
				struct sparse_t_precond_data< float > * const sparse_t_data =
					new struct sparse_t_precond_data< float >();
				sparse_t_data_p = sparse_t_data;
			} catch( ... ) {
				return KMLSS_NO_MEMORY;
			}
		}
		// (re-)initialise sparse_t_data
		{
			struct sparse_t_precond_data< float > * const sparse_t_data =
				reinterpret_cast< sparse_t_precond_data< float > * >( sparse_t_data_p );
			sparse_t_data->n = size;
			sparse_t_data->kml_preconditioner = preconditioner;
			sparse_t_data->kml_data = data;
		}
	}

	// activate the selected preconditioner
	rc = sparse_cg_set_preconditioner_sii(
		handle, sparse_t_function_p, sparse_t_data_p );

	// done
	return sparse_err_t_2_int( rc );
}

int KML_CG_PREFIXED( SetUserPreconditionerDI )(
	KmlSolverTask ** handle_p,
	void * data, int (*preconditioner)( void *, double * )
) {
	int (*sparse_t_function_p) (
		double * const,
		const double * const,
		void * const
	) = nullptr;
	void * sparse_t_data_p = nullptr;
	sparse_cg_handle_t handle = *handle_p;
	size_t size;

	sparse_err_t rc = sparse_cg_get_preconditioner_dii(
		handle, &sparse_t_function_p, &sparse_t_data_p );
	rc = rc ? rc : sparse_cg_get_size_dii( handle, &size );
	if( rc != sparse_err_t::NO_ERROR ) { return sparse_err_t_2_int( rc ); }

	if( preconditioner == nullptr ) {
		// in this case, no preconditioning is requested
		// if we had data stored, delete it
		if( sparse_t_data_p != nullptr && preconditioner == nullptr ) {
			{
				struct sparse_t_precond_data< double > * const sparse_t_data =
					reinterpret_cast< sparse_t_precond_data< double > * >( sparse_t_data_p );
				delete sparse_t_data;
			}
			sparse_t_data_p = nullptr;
		}
	} else {
		// in this case, preconditioning is requested
		sparse_t_function_p = &(sparse_t_preconditioner< double >);
		// if we had no data already stored, create it
		if( sparse_t_data_p == nullptr ) {
			try {
				struct sparse_t_precond_data< double > * const sparse_t_data =
					new struct sparse_t_precond_data< double >();
				sparse_t_data_p = sparse_t_data;
			} catch( ... ) {
				return KMLSS_NO_MEMORY;
			}
		}
		// (re-)initialise sparse_t_data
		{
			struct sparse_t_precond_data< double > * const sparse_t_data =
				reinterpret_cast< sparse_t_precond_data< double > * >( sparse_t_data_p );
			sparse_t_data->n = size;
			sparse_t_data->kml_preconditioner = preconditioner;
			sparse_t_data->kml_data = data;
		}
	}

	// activate the selected preconditioner
	rc = sparse_cg_set_preconditioner_dii(
		handle, sparse_t_function_p, sparse_t_data_p );

	// done
	return sparse_err_t_2_int( rc );
}

int KML_CG_PREFIXED( SetSII )(
	KmlSolverTask **handle, KML_SOLVER_PARAM param, const int *data, int nd
) {
	KMLSS_SET_ARG( nd, param, KMLSS_THRESHOLD,
		sparse_cg_set_max_iter_count_sii( handle, *data ) );
}

int KML_CG_PREFIXED( SetSIS )(
	KmlSolverTask **handle, KML_SOLVER_PARAM param,
	const float *data, int nd
) {
	KMLSS_SET_ARG( nd, param, KMLSS_MAX_ITERATION_COUNT,
		sparse_cg_set_tolerance_sii( handle, *data ) );
}

int KML_CG_PREFIXED( SetDII )(
	KmlSolverTask **handle, KML_SOLVER_PARAM param,
	const int *data, int nd
) {
	KMLSS_SET_ARG( nd, param, KMLSS_THRESHOLD,
		sparse_cg_set_max_iter_count_dii( handle, *data ) );
}

int KML_CG_PREFIXED( SetDID )(
	KmlSolverTask **handle, KML_SOLVER_PARAM param,
	const double *data, int nd
) {
	KMLSS_SET_ARG( nd, param, KMLSS_MAX_ITERATION_COUNT,
		sparse_cg_set_tolerance_dii( handle, *data ) );
}


int KML_CG_PREFIXED( AnalyzeSI )( KmlSolverTask ** ) {
	return KMLSS_NOT_IMPLEMENTED;
}

int KML_CG_PREFIXED( AnalyzeDI )( KmlSolverTask ** ) {
	return KMLSS_NOT_IMPLEMENTED;
}


int KML_CG_PREFIXED( FactorizeSI )( KmlSolverTask ** ) {
	return KMLSS_NOT_IMPLEMENTED;
}

int KML_CG_PREFIXED( FactorizeDI )( KmlSolverTask ** ) {
	return KMLSS_NOT_IMPLEMENTED;
}


int KML_CG_PREFIXED( SolveSI )(
	KmlSolverTask **handle, int nb, float *x, int,
	const float *b, int
) {
	KMLSS_SOLVE( nb, sparse_cg_solve_sii( *handle, x, b ) );
}

int KML_CG_PREFIXED( SolveDI )(
	KmlSolverTask **handle, int nb, double *x, int,
	const double *b, int
) {
	KMLSS_SOLVE( nb, sparse_cg_solve_dii( *handle, x, b ) );
}


int KML_CG_PREFIXED( GetSII )(
	KmlSolverTask **handle, KML_SOLVER_PARAM param,
	int *data, int nd
) {
	size_t s = 0;
	KMLSS_GET_ARG( nd, param, KMLSS_ITERATION_COUNT,
		sparse_cg_get_iter_count_sii( *handle, &s ), ret );
	if( ret == NO_ERROR ) { *data = static_cast< int >( s ); }
	return ret;
}

int KML_CG_PREFIXED( GetDII )(
	KmlSolverTask **handle, KML_SOLVER_PARAM param,
	int *data, int nd
) {
	size_t s = 0;
	KMLSS_GET_ARG( nd, param, KMLSS_ITERATION_COUNT,
		sparse_cg_get_iter_count_dii( *handle, &s ), ret );
	if( ret == NO_ERROR ) { *data = static_cast< int >( s ); }
	return ret;
}

int KML_CG_PREFIXED( GetSIS )(
	KmlSolverTask **handle, KML_SOLVER_PARAM param,
	float *data, int nd
) {
	KMLSS_GET_ARG( nd, param, KMLSS_TOLERANCE,
		sparse_cg_get_residual_sii(*handle, data), ret );
	return ret;
}

int KML_CG_PREFIXED( GetDID )(
	KmlSolverTask **handle, KML_SOLVER_PARAM param,
	double *data, int nd
) {
	KMLSS_GET_ARG( nd, param, KMLSS_TOLERANCE,
		sparse_cg_get_residual_dii(*handle, data), ret );
	return ret;
}

int KML_CG_PREFIXED( CleanSI )( KmlSolverTask ** handle_p ) {
	void * data = nullptr;
	sparse_cg_handle_t handle = *handle_p;
	sparse_cg_preconditioner_sxx_t preconditioner = nullptr;

	sparse_err_t rc = sparse_cg_get_preconditioner_sii(
		handle, &preconditioner, &data );
	rc = rc ? rc : sparse_cg_destroy_sii( handle );
	if( rc != sparse_err_t::NO_ERROR ) { return sparse_err_t_2_int( rc ); }

	if( data != nullptr ) {
		auto * const sparse_t_data =
			reinterpret_cast< sparse_t_precond_data< float > * >( data );
		try {
			delete sparse_t_data;
		} catch( ... ) {
			return KMLSS_INTERNAL_ERROR;
		}
	}

	return NO_ERROR;
}

int KML_CG_PREFIXED( CleanDI )( KmlSolverTask ** handle_p ) {
	void * data = nullptr;
	sparse_cg_handle_t handle = *handle_p;
	sparse_cg_preconditioner_dxx_t preconditioner = nullptr;

	sparse_err_t rc = sparse_cg_get_preconditioner_dii(
		handle, &preconditioner, &data );
	rc = rc ? rc : sparse_cg_destroy_dii( handle );
	if( rc != sparse_err_t::NO_ERROR ) { return sparse_err_t_2_int( rc ); }

	if( data != nullptr ) {
		auto * const sparse_t_data =
			reinterpret_cast< sparse_t_precond_data< double > * >( data );
		try {
			delete sparse_t_data;
		} catch( ... ) {
			return KMLSS_INTERNAL_ERROR;
		}
	}

	return NO_ERROR;
}

