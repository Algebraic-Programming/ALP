
/*
 *   Copyright 2021 Huawei Technologies Co., Ltd.
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

/*
 * @authors Anders Hansson, Aristeidis Mastoras, A. N. Yzelman
 * @date May, 2022
 */

#include <exception>
#include <iostream>
#include <vector>

#include <inttypes.h>

#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>

#include <graphblas.hpp>

using namespace grb;

struct input {
	char filenameL[ 1024 ];
	char filenameR[ 1024 ];
	bool direct;
	size_t rep;
};

struct output {
	int error_code;
	size_t rep;
	grb::utils::TimerResults times;
	PinnedVector< double > pinnedVector;
	size_t result_nnz;
};

void grbProgram( const struct input & data_in, struct output & out ) {
	// get user process ID
	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );

	// get input n
	grb::utils::Timer timer;
	timer.reset();

	// sanity checks on input
	if( data_in.filenameL[ 0 ] == '\0' ) {
		std::cerr << s << ": no file name given as input for left matrix." << std::endl;
		out.error_code = ILLEGAL;
		return;
	} else if( data_in.filenameR[ 0 ] == '\n' ) {
		std::cerr << s << ": no file name given as input for right matrix." << std::endl;
		out.error_code = ILLEGAL;
		return;
	}

	// assume successful run
	out.error_code = 0;

	// create local parser
	grb::utils::MatrixFileReader< double,
		std::conditional< ( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ), grb::config::RowIndexType, grb::config::ColIndexType >::type >
		parserL( data_in.filenameL, data_in.direct );

	grb::utils::MatrixFileReader< double,
		std::conditional< ( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ), grb::config::RowIndexType, grb::config::ColIndexType >::type >
		parserR( data_in.filenameR, data_in.direct );

	assert( parserL.n() == parserR.m() );

	const size_t l = parserL.m();
	const size_t m = parserL.n();
	const size_t n = parserR.n();

	out.times.io = timer.time();
	timer.reset();

	// load into GraphBLAS
	Matrix< double > A( l, m ), B( m, n );
	{
		RC rc = buildMatrixUnique( A, parserL.begin( SEQUENTIAL ), parserL.end( SEQUENTIAL ), SEQUENTIAL );
		/* Once internal issue #342 is resolved this can be re-enabled
		const RC rc = buildMatrixUnique( A,
		    parser.begin( PARALLEL ), parser.end( PARALLEL),
		    PARALLEL
		);*/
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to buildMatrixUnique did not succeed for the "
					  << "left-hand matrix "
					  << "(" << toString( rc ) << ")." << std::endl;
			out.error_code = 10;
			return;
		}

		rc = buildMatrixUnique( B, parserR.begin( SEQUENTIAL ), parserR.end( SEQUENTIAL ), SEQUENTIAL );

		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to buildMatrixUnique did not succeed for the "
					  << "right-hand matrix "
					  << "(" << toString( rc ) << ")." << std::endl;
			out.error_code = 20;
			return;
		}
	}

	// std::cout << "nnz(D) = " << internal::getCurrentNonzeroes( D ) << std::endl;

	// check number of nonzeroes
	try {
		const size_t global_nnzL = nnz( A );
		const size_t global_nnzR = nnz( B );
		const size_t parser_nnzL = parserL.nz();
		const size_t parser_nnzR = parserR.nz();
		if( global_nnzL != parser_nnzL ) {
			std::cerr << "Left matrix Failure: global nnz (" << global_nnzL << ") "
					  << "does not equal parser nnz (" << parser_nnzL << ")." << std::endl;
			return;
		} else if( global_nnzR != parser_nnzR ) {
			std::cerr << "Right matrix Failure: global nnz (" << global_nnzR << ") "
					  << "does not equal parser nnz (" << parser_nnzR << ")." << std::endl;
			return;
		}

	} catch( const std::runtime_error & ) {
		std::cout << "Info: nonzero check skipped as the number of nonzeroes "
				  << "cannot be derived from the matrix file header. The "
				  << "grb::Matrix reports " << nnz( A ) << " nonzeroes in left "
				  << "and " << nnz( B ) << " n right \n";
	}

	RC rc = SUCCESS;

	// test default SpMSpM run
	const Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > ring;

	const Monoid< grb::operators::add< double >, grb::identities::zero > monoid;

	// by default, copy input requested repetitions to output repititions performed
	out.rep = data_in.rep;

	// time a single call
	{
		Matrix< double > C( l, n );
		Matrix< double > D( l, n );
		Matrix< double > E( l, n );

		grb::utils::Timer subtimer;
		subtimer.reset();
		// rc = rc ? rc : grb::mxm( C, A, B, ring, RESIZE );

		// rc = rc ? rc : grb::mxm( C, A, B, ring, RESIZE );
		// rc = rc ? rc : grb::eWiseApply( D, C, A, monoid, RESIZE );

		assert( rc == SUCCESS );

		// rc = rc ? rc : grb::mxm( C, A, B, ring );

		// rc = rc ? rc : grb::mxm( C, A, B, ring );
		// rc = rc ? rc : grb::eWiseApply( D, C, A, monoid );
		assert( rc == SUCCESS );

		double single_time = subtimer.time();

		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to mxm did not succeed (" << toString( rc ) << ")." << std::endl;
			out.error_code = 70;
			return;
		}
		if( rc == SUCCESS ) {
			rc = collectives<>::reduce( single_time, 0, operators::max< double >() );
		}
		if( rc != SUCCESS ) {
			out.error_code = 80;
			return;
		}
		out.times.useful = single_time;
		const size_t deduced_inner_reps = static_cast< size_t >( 100.0 / single_time ) + 1;
		if( rc == SUCCESS && out.rep == 0 ) {
			if( s == 0 ) {
				std::cout << "Info: cold mxm completed"
						  << ". Time taken was " << single_time << " ms. "
						  << "Deduced inner repetitions parameter of " << out.rep << " "
						  << "to take 1 second or more per inner benchmark.\n";
				out.rep = deduced_inner_reps;
			}
			return;
		}
	}

	if( out.rep > 1 ) {
		std::cerr << "Error: more than 1 inner repetitions are not supported due to "
				  << "having to time the symbolic phase while not timing the initial matrix "
				  << "allocation cost\n";
		out.error_code = 90;
		return;
	}

	// allocate output for benchmark
	Matrix< double > C( n, n);
	Matrix< double > D( n, n );
	//Matrix< double > E( n, n );	

	// that was the preamble
	out.times.preamble = timer.time();

	grb::wait();
	// do benchmark
	double time_taken;
	timer.reset();
	

#ifndef NDEBUG
	// rc = rc ? rc : grb::mxm( C, A, B, ring, RESIZE );

	// rc = rc ? rc : grb::mxm( C, A, B, ring, RESIZE );
	// rc = rc ? rc : grb::eWiseApply( D, C, A, monoid, RESIZE );

	assert( rc == SUCCESS );
	// rc = rc ? rc : grb::mxm( C, A, B, ring );
	// rc = rc ? rc : grb::mxm( C, A, B, ring );
	// rc = rc ? rc : grb::eWiseApply( D, C, A, monoid );
	assert( rc == SUCCESS );
#else
	std::cout << "matrix IDs" << std::endl;
	std::cout << "getID(A) = " << grb::getID(A) <<", value of nnz( A ) = " << nnz( A ) << std::endl;
	std::cout << "getID(B) = " << grb::getID(B) << std::endl;
	std::cout << "getID(C) = " << grb::getID(C) << std::endl;
	std::cout << "getID(D) = " << grb::getID(D) << std::endl;
	//std::cout << "getID(E) = " << grb::getID(E) << std::endl;

	/*
	// example to test mxm only in nonblocking 
	(void)grb::mxm( C, A, B, ring, RESIZE );
	(void)grb::mxm( D, C, A, ring, RESIZE );
	(void)grb::mxm( E, C, D, ring, RESIZE );

	(void)grb::mxm( C, A, B, ring );
	(void)grb::mxm( D, C, A, ring );
	(void)grb::mxm( E, C, D, ring );
	*/

	/*
	// example to test mxm only in reference 
	(void)grb::mxm( C, A, B, ring, RESIZE );
	(void)grb::mxm( C, A, B, ring );
	(void)grb::mxm( D, C, A, ring, RESIZE );
	(void)grb::mxm( D, C, A, ring );
	(void)grb::mxm( E, C, D, ring, RESIZE );
	(void)grb::mxm( E, C, D, ring );
	*/
	
	/*
	// example to test eWiseApply in nonblocking
	(void)grb::eWiseApply( C, A, B, monoid, RESIZE );
	(void)grb::eWiseApply( C, A, B, monoid );
	
	(void)grb::eWiseApply( D, B, C, monoid, RESIZE );
	(void)grb::eWiseApply( D, B, C, monoid );		

	double value = 0;
	(void)grb::internal::foldl( value, D, monoid );
	std::cout << "sum of matrix D (using foldl)= " << value << std::endl;

	(void)grb::eWiseApply( E, A, D, monoid, RESIZE );
	(void)grb::eWiseApply( E, A, D, monoid, EXECUTE );	
	*/

	
	// example to test eWiseApply and mxm in nonblocking

	// C = AB
	//(void)grb::mxm( C, A, B, ring, RESIZE );
	//(void)grb::mxm( C, A, B, ring );
	
	/*
	auto &C_raw = internal::getCRS( C );
	
	std::cout << "row pointers of C " << std::endl;
	for( size_t i = 0; i < grb::nrows( C ) + 1; i++ ) {
		std::cout << C_raw.col_start[ i ] << ", ";
	}
	std::cout << std::endl;	
	
	std::cout << "col indices C " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity( C ); i++ ) {
		std::cout << C_raw.row_index[ i ] << std::endl;
	}
	std::cout << std::endl;
	
	std::cout << "values C " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity( C ); i++ ) {
		std::cout << C_raw.values[ i ] << std::endl;
	}
	std::cout << std::endl;
	*/
	
	/*
	// D = B*C 	
	(void)grb::eWiseApply( D, B, C, monoid, RESIZE );
	(void)grb::eWiseApply( D, B, C, monoid );	
	*/
	
	/*
	auto &D_raw = internal::getCRS( D );
	std::cout << "internal::getNonzeroCapacity( D ) = " << internal::getNonzeroCapacity( D ) << std::endl;
	std::cout << "row pointers of D " << std::endl;
	for( size_t i = 0; i < grb::nrows( D ) + 1; i++ ) {
		std::cout << D_raw.col_start[ i ] << ",";
	}
	std::cout << std::endl;
	
	std::cout << "internal::getNonzeroCapacity( D ) = " << internal::getNonzeroCapacity( D ) << std::endl;
	std::cout << "internal::getCurrentNonzeroes( D ) = " << internal::getCurrentNonzeroes( D ) << std::endl;
	std::cout << "col indices D " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity( D ); i++ ) {
		std::cout << D_raw.row_index[ i ] << ",";
	}
	std::cout << std::endl;
		
	std::cout << "values D " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity( D ); i++ ) {
		std::cout << D_raw.values[ i ] << ",";
	}
	std::cout << std::endl;
	*/

	/*
	// this primitive enforces executing the pipeline of D. 
	// after this primitive, matrices C and D must be fully computed	
	(void)grb::mxm( E, A, D, ring, RESIZE );
	(void)grb::mxm( E, A, D, ring );	
	*/		

	// this primitive force pipeline of E to execute
	/*
	double value = 0;
	(void)grb::internal::foldl( value, E, monoid );
	std::cout << "sum of matrix E (using foldl)= " << value << std::endl;	
	*/
	//grb::wait( E );

	/*
	auto &E_raw = internal::getCCS( E );
	std::cout << "internal::getNonzeroCapacity( E ) = " << internal::getNonzeroCapacity( E ) << std::endl;
	std::cout << "col pointers of E " << std::endl;
	for( size_t i = 0; i < grb::ncols( E ) + 1; i++ ) {
		std::cout << E_raw.col_start[ i ] << ",";
	}
	std::cout << std::endl;
	
	std::cout << "row indices E " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity( E ); i++ ) {
		std::cout << E_raw.row_index[ i ] << ",";
	}
	std::cout << std::endl;
		
	std::cout << "values E " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity( E ); i++ ) {
		std::cout << E_raw.values[ i ] << ",";
	}
	std::cout << std::endl;
	*/

	// EXAMPLE TO TEST MASKED VERSION OF MXM
	// WE USE THE SAME INPUT MATRIX A AS MASK
	// B is mask and A is matrix
	// computation of C = A^2
	std::cout << "in mxm RESIZE" << std::endl;
	grb::mxm(C, A, A,ring, RESIZE);
	std::cout << "in mxm EXECUTE" << std::endl;
	grb::mxm(C, A, A,ring);
	//grb::wait(C);

	//std::cout << "in mxm masked RESIZE" << std::endl;
	// computation of D = A^3
	grb::mxm_masked( D, B, C, A, ring, RESIZE );
	std::cout << "in mxm masked EXECUTE" << std::endl;
	grb::mxm_masked( D, B, C, A, ring);
	
	//std::cout << "in foldl " << std::endl;
	//trace of D
	double count = 0;
	grb::foldl( count, D, monoid );

	printf( "value of D = %lf \n", count / 6 );
	//std::cout << "value of D = " << count / 6 << std::endl;

	/*
	auto &C_raw = internal::getCRS( C );
	std::cout << "row pointers of C " << std::endl;
	for( size_t i = 0; i < grb::nrows( C ) + 1; i++ ) {
	    std::cout << C_raw.col_start[ i ] << ", ";
	}
	std::cout << std::endl;

	std::cout << "col indices C " << std::endl;
	for( size_t i = 0; i < internal::getCurrentNonzeroes( C ); i++ ) {
	    std::cout << C_raw.row_index[ i ] << ", ";
	}
	std::cout << std::endl;

	std::cout << "values C " << std::endl;
	for( size_t i = 0; i < internal::getCurrentNonzeroes( C ); i++ ) {
	    std::cout << C_raw.values[ i ] << ", ";
	}
	std::cout << std::endl;
	*/

	/*
	auto &D_raw = internal::getCRS( D );

	std::cout << "row pointers of D " << std::endl;
	for( size_t i = 0; i < grb::nrows( D ) + 1; i++ ) {
	    std::cout << D_raw.col_start[ i ] << ", ";
	}
	std::cout << std::endl;

	std::cout << "col indices D " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity( D ); i++ ) {
	    std::cout << D_raw.row_index[ i ] << ", ";
	}
	std::cout << std::endl;

	std::cout << "values D " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity( D ); i++ ) {
	    std::cout << D_raw.values[ i ] << ", ";
	}
	std::cout << std::endl;
	*/
	

#endif

	//grb::wait();
	time_taken = timer.time();
	if( rc == SUCCESS ) {
		out.times.useful = time_taken / static_cast< double >( out.rep );
	}
	// print timing at root process
	if( grb::spmd<>::pid() == 0 ) {
		std::cout << "Time taken for a " << out.rep << " "
				  << "mxm calls (hot start): " << out.times.useful << ". "
				  << "Error code is " << out.error_code << std::endl;
	}

	
	// start postamble
	timer.reset();

	// set error code
	if( rc == FAILED ) {
		out.error_code = 30;
		// no convergence, but will print output
	} else if( rc != SUCCESS ) {
		std::cerr << "Benchmark run returned error: " << toString( rc ) << "\n";
		out.error_code = 35;
		return;
	}

	// finish timing
	time_taken = timer.time();
	out.times.postamble = time_taken;

	/*
	auto &C_raw = internal::getCRS( C );
	std::cout << "row pointers of C " << std::endl;
	for( size_t i = 0; i < grb::nrows( C ) + 1; i++ ) {
		std::cout << C_raw.col_start[ i ] << ",";
	}
	std::cout << std::endl;

	std::cout << "col indices C " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity( C ); i++ ) {
		std::cout << C_raw.row_index[ i ] << ",";
	}
	std::cout << std::endl;

	std::cout << "values C " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity( C ); i++ ) {
		std::cout << C_raw.values[ i ] << ",";
	}
	std::cout << std::endl;

	auto &D_raw = internal::getCRS( D );
	std::cout << "internal::getNonzeroCapacity( D ) = " << internal::getNonzeroCapacity( D ) << std::endl;
	std::cout << "row pointers of D " << std::endl;
	for( size_t i = 0; i < grb::nrows( D ) + 1; i++ ) {
		std::cout << D_raw.col_start[ i ] << ",";
	}
	std::cout << std::endl;

	std::cout << "col indices D " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity( D ); i++ ) {
		std::cout << D_raw.row_index[ i ] << ",";
	}
	std::cout << std::endl;

	std::cout << "values D " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity( D ); i++ ) {
		std::cout << D_raw.values[ i ] << ",";
	}
	std::cout << std::endl;

	auto &E_raw = internal::getCRS( E );
	std::cout << "internal::getNonzeroCapacity( E ) = " << internal::getNonzeroCapacity( E ) << std::endl;
	std::cout << "row pointers of E " << std::endl;
	for( size_t i = 0; i < grb::nrows( E ) + 1; i++ ) {
		std::cout << E_raw.col_start[ i ] << ",";
	}
	std::cout << std::endl;

	std::cout << "col indices E " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity( E ); i++ ) {
		std::cout << E_raw.row_index[ i ] << ",";
	}
	std::cout << std::endl;

	std::cout << "values E " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity( E ); i++ ) {
		std::cout << E_raw.values[ i ] << ",";
	}
	std::cout << std::endl;
	*/

	/*
	// this enforces the pipeline execution
	double sum_value = 0;
	(void)grb::internal::foldl( sum_value, E, monoid );
	std::cout << "sum of matrix E (using foldl)= " << sum_value << std::endl;
	*/
	/*
	std::cout << "Matrix C, vector that stores nnz in each tile " << std::endl;
	auto & nnz_tiles_C = internal::getNonzerosTiles( C );
	for (size_t i = 0; i < nnz_tiles_C.size(); i++)
	{
		std::cout << "nnz_tiles_C[" << i << "] = " << nnz_tiles_C[ i ] << std::endl;
	}
	

	//auto &D_raw = internal::getCRS( D );
	std::cout << "getID(D) = " << grb::getID( D );
	std::cout << ", vector that stores nnz in each tile " << std::endl;
	auto & nnz_tiles_D = internal::getNonzerosTiles( D );
	for (size_t i = 0; i < nnz_tiles_D.size(); i++)
	{
		std::cout << "nnz_tiles_C[" << i << "] = " << nnz_tiles_D[ i ] << std::endl;
	}
	*/
	
	/*
	auto &C_raw = internal::getCRS( C );
	std::cout << "row pointers of C " << std::endl;
	for( size_t i = 0; i < grb::nrows(C)+1; i++ ) {
	    std::cout << C_raw.col_start[ i ] << ",";
	}
	std::cout << std::endl;

	std::cout << "col indices C " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity(C); i++ ) {
	    std::cout << C_raw.row_index[ i ] << ",";
	}
	std::cout << std::endl;

	std::cout << "values of C " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity(C); i++ ) {
	    std::cout << C_raw.values[ i ] << ",";
	}
	std::cout << std::endl;
	
	
	//For matrix D
	//auto &D_raw = internal::getCRS( D );
	std::cout << "----ooo00000----" << std::endl;
	std::cout << "row pointers of D " << std::endl;
	for( size_t i = 0; i < grb::nrows(D)+1; i++ ) {
	    std::cout << D_raw.col_start[ i ] << ",";
	}
	std::cout << std::endl;

	std::cout << "col indices D " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity(D); i++ ) {
	    std::cout << D_raw.row_index[ i ] << ",";
	}
	std::cout << std::endl;

	std::cout << "values of D " << std::endl;
	for( size_t i = 0; i < internal::getNonzeroCapacity(D); i++ ) {
	    std::cout << D_raw.values[ i ] << ",";
	}
	std::cout << std::endl;
	*/
	
	int nnz = 0;
	
	auto it = D.begin();
	while( it != D.end() ) {
		if( ( *it ).second != 0.0 ) {
			nnz++;
		}
		it.operator++();
	}
	
	out.result_nnz = nnz;

	// done
	return;
}

int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 3 || argc > 7 ) {
		std::cout << "Usage: " << argv[ 0 ] << " <datasetL> <datasetR> <direct/indirect> "
				  << "(inner iterations) (outer iterations) (verification <truth-file>)\n";
		std::cout << "<datasetL>, <datasetR>, and <direct/indirect> are mandatory arguments.\n";
		std::cout << "<datasetL> is the left matrix of the multiplication and "
				  << "<datasetR> is the right matrix \n";
		std::cout << "(inner iterations) is optional, the default is " << grb::config::BENCHMARKING::inner() << ". "
				  << "If set to zero, the program will select a number of iterations "
				  << "approximately required to take at least one second to complete.\n";
		std::cout << "(outer iterations) is optional, the default is " << grb::config::BENCHMARKING::outer() << ". "
				  << "This value must be strictly larger than 0." << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;
#ifndef NDEBUG
	std::cerr << "Warning: benchmark driver compiled without the NDEBUG macro "
			  << "defined(!)\n";
#endif

	// the input struct
	struct input in;

	// get file name Left
	(void)strncpy( in.filenameL, argv[ 1 ], 1023 );
	in.filenameL[ 1023 ] = '\0';

	// get file name Right
	(void)strncpy( in.filenameR, argv[ 2 ], 1023 );
	in.filenameL[ 1023 ] = '\0';

	// get direct or indirect addressing
	if( strncmp( argv[ 3 ], "direct", 6 ) == 0 ) {
		in.direct = true;
	} else {
		in.direct = false;
	}

	// get inner number of iterations
	in.rep = grb::config::BENCHMARKING::inner();
	char * end = nullptr;
	if( argc >= 5 ) {
		in.rep = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 3 ] << " "
					  << "for number of inner experiment repititions." << std::endl;
			return 2;
		}
	}

	// get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();
	if( argc >= 6 ) {
		outer = strtoumax( argv[ 5 ], &end, 10 );
		if( argv[ 5 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 4 ] << " "
					  << "for number of outer experiment repititions." << std::endl;
			return 4;
		}
	}

	std::cout << "Executable called with parameters:  Left matrix A = " << in.filenameL << ", right matrix B = " << in.filenameR << ", "
			  << "inner repititions = " << in.rep << ", and outer reptitions = " << outer << std::endl;

	// the output struct
	struct output out;

	// set standard exit code
	grb::RC rc = SUCCESS;

	// launch estimator (if requested)
	if( in.rep == 0 ) {
		grb::Launcher< AUTOMATIC > launcher;
		rc = launcher.exec( &grbProgram, in, out, true );
		if( rc == SUCCESS ) {
			in.rep = out.rep;
		}
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec returns with non-SUCCESS error code " << (int)rc << std::endl;
			return 6;
		}
	}

	// launch benchmark
	if( rc == SUCCESS ) {
		grb::Benchmarker< AUTOMATIC > benchmarker;
		rc = benchmarker.exec( &grbProgram, in, out, 1, outer, true );
	}
	if( rc != SUCCESS ) {
		std::cerr << "benchmarker.exec returns with non-SUCCESS error code " << grb::toString( rc ) << std::endl;
		return 8;
	}

	std::cout << "Error code is " << out.error_code << ".\n";

	std::cout << "Number of non-zeroes in output matrix: " << out.result_nnz << "\n";

	if( out.error_code == 0 && out.pinnedVector.size() > 0 ) {
		std::cerr << std::fixed;
		std::cerr << "Output matrix: (";
		for( size_t k = 0; k < out.pinnedVector.nonzeroes(); k++ ) {
			const auto & nonZeroValue = out.pinnedVector.getNonzeroValue( k );
			std::cerr << nonZeroValue << ", ";
		}
		std::cerr << ")" << std::endl;
		std::cerr << std::defaultfloat;
	}

	if( out.error_code != 0 ) {
		std::cerr << std::flush;
		std::cerr << "Test FAILED\n";
	} else {
		std::cout << "Test OK\n";
	}
	std::cout << std::endl;

	// done
	return out.error_code;
}
