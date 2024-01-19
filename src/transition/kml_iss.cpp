
#include <kml_iss.h>

#include <solver.h>


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
		case UNKNOWN:
		{
			result = KMLSS_INTERNAL_ERROR;
			break;
		}
	}
	return result;
}

#define KML_CG_PREFIXED( name ) KmlIssCg ## name


int KML_CG_PREFIXED(InitSI)( KmlSolverTask **handle, int n, const float *a, const int *ja, const int *ia ) {
	sparse_err_t err = sparse_cg_init_sii( handle, n, a, ja, ia );
	return sparse_err_t_2_int( err );
}

int KML_CG_PREFIXED(InitDI)( KmlSolverTask **handle, int n, const double *a, const int *ja, const int *ia ) {
	sparse_err_t err = sparse_cg_init_dii( handle, n, a, ja, ia );
	return sparse_err_t_2_int( err );
}



int KML_CG_PREFIXED(SetUserPreconditionerSI)(KmlSolverTask **, void *, int (*)(void *, float *)) {
	return KMLSS_NOT_IMPLEMENTED;
}

int KML_CG_PREFIXED(SetUserPreconditionerDI)(KmlSolverTask **, void *, int (*)(void *, double *)) {
	return KMLSS_NOT_IMPLEMENTED;
}



#define KMLSS_SET_ARG( nd, param, case_val, sparse_err_t_val )	\
	if( nd != 1 ) { return KMLSS_BAD_DATA_SIZE; }				\
	int err = KMLSS_NO_ERROR;									\
	switch ( param )											\
	{															\
	case case_val:												\
	{															\
		const sparse_err_t r = sparse_err_t_val;				\
		if( r != NO_ERROR ) {									\
			err = ( r == NULL_ARGUMENT ) ?						\
				KMLSS_NULL_ARGUMENT : KMLSS_INTERNAL_ERROR;		\
		}														\
		break;													\
	}															\
	default:													\
		err = KMLSS_BAD_SELECTOR;								\
	}															\
	return err;

int KML_CG_PREFIXED(SetSII)(KmlSolverTask **handle, KML_SOLVER_PARAM param, const int *data, int nd) {
	KMLSS_SET_ARG( nd, param, KMLSS_THRESHOLD, sparse_cg_set_max_iter_count_sii( handle, *data ) );
}

int KML_CG_PREFIXED(SetSIS)(KmlSolverTask **handle, KML_SOLVER_PARAM param, const float *data, int nd) {
	KMLSS_SET_ARG( nd, param, KMLSS_MAX_ITERATION_COUNT, sparse_cg_set_tolerance_sii( handle, *data ) );
}

int KML_CG_PREFIXED(SetDII)(KmlSolverTask **handle, KML_SOLVER_PARAM param, const int *data, int nd) {
	KMLSS_SET_ARG( nd, param, KMLSS_THRESHOLD, sparse_cg_set_max_iter_count_dii( handle, *data ) );
}

int KML_CG_PREFIXED(SetDID)(KmlSolverTask **handle, KML_SOLVER_PARAM param, const double *data, int nd) {
	KMLSS_SET_ARG( nd, param, KMLSS_MAX_ITERATION_COUNT, sparse_cg_set_tolerance_dii( handle, *data ) );
}



int KML_CG_PREFIXED(AnalyzeSI)(KmlSolverTask **) { return KMLSS_NOT_IMPLEMENTED; }

int KML_CG_PREFIXED(AnalyzeDI)(KmlSolverTask **) { return KMLSS_NOT_IMPLEMENTED; }



int KML_CG_PREFIXED(FactorizeSI)(KmlSolverTask **) { return KMLSS_NOT_IMPLEMENTED; }

int KML_CG_PREFIXED(FactorizeDI)(KmlSolverTask **) { return KMLSS_NOT_IMPLEMENTED; }



#define KMLSS_SOLVE( nb, sparse_err_t_val )				\
	if( nb != 1 ) { return KMLSS_BAD_NB; }				\
	int err = KMLSS_NO_ERROR;							\
	const sparse_err_t r = sparse_err_t_val;			\
	if( r != NO_ERROR ) {								\
		err = ( r == NULL_ARGUMENT ) ?					\
			KMLSS_NULL_ARGUMENT : KMLSS_INTERNAL_ERROR;	\
	}													\
	return err;


int KML_CG_PREFIXED(SolveSI)( KmlSolverTask **handle, int nb, float *x, int, const float *b, int ) {
	KMLSS_SOLVE( nb, sparse_cg_solve_sii( *handle, x, b ) );
}

int KML_CG_PREFIXED(SolveDI)( KmlSolverTask **handle, int nb, double *x, int, const double *b, int ) {
	KMLSS_SOLVE( nb, sparse_cg_solve_dii( *handle, x, b ) );
}



#define KMLSS_GET_ARG( nd, param, case_val, sparse_err_t_val, err )	\
	if( nd != 1 ) { return KMLSS_BAD_DATA_SIZE; }					\
	int err = KMLSS_NO_ERROR;										\
	switch ( param )												\
	{																\
	case case_val:													\
	{																\
		const sparse_err_t r = sparse_err_t_val;					\
		if( r != NO_ERROR ) {										\
			err = ( r == NULL_ARGUMENT ) ?							\
				KMLSS_NULL_ARGUMENT : KMLSS_INTERNAL_ERROR;			\
		}															\
		break;														\
	}																\
	default:														\
		err = KMLSS_BAD_SELECTOR;									\
	}
	// return err;

int KML_CG_PREFIXED(GetSII)( KmlSolverTask **handle, KML_SOLVER_PARAM param, int *data, int nd ) {
	size_t s = 0;
	KMLSS_GET_ARG( nd, param, KMLSS_ITERATION_COUNT, sparse_cg_get_iter_count_sii( *handle, &s ), ret );
	if( ret == NO_ERROR ) { *data = static_cast< int >( s ); }
	return ret;
}

int KML_CG_PREFIXED(GetDII)( KmlSolverTask **handle, KML_SOLVER_PARAM param, int *data, int nd ) {
	size_t s = 0;
	KMLSS_GET_ARG( nd, param, KMLSS_ITERATION_COUNT, sparse_cg_get_iter_count_dii( *handle, &s ), ret );
	if( ret == NO_ERROR ) { *data = static_cast< int >( s ); }
	return ret;
}

int KML_CG_PREFIXED(GetSIS)( KmlSolverTask **handle, KML_SOLVER_PARAM param, float *data, int nd ) {
	KMLSS_GET_ARG( nd, param, KMLSS_TOLERANCE, sparse_cg_get_residual_sii(*handle, data), ret );
	return ret;
}

int KML_CG_PREFIXED(GetDID)( KmlSolverTask **handle, KML_SOLVER_PARAM param, double *data, int nd ) {
	KMLSS_GET_ARG( nd, param, KMLSS_TOLERANCE, sparse_cg_get_residual_dii(*handle, data), ret );
	return ret;
}
