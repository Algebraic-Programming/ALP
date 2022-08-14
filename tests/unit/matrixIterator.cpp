
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

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <cstdlib>

#include <graphblas.hpp>
#include <graphblas/utils.hpp> // grb::equals
#include <graphblas/SynchronizedNonzeroIterator.hpp>


#include <utils/matrix_values_check.hpp>
#include <graphblas/utils/iterators/NonzeroIterator.hpp>


using namespace grb;

// nonzero values
static const int data[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };
static const double data_double[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };

// diagonal matrix
static size_t I1[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
static size_t J1[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };

// matrix with empty rows and columns
static size_t I2[ 15 ] = { 1, 1, 3, 3, 6, 6, 6, 7, 7, 12, 12, 12, 13, 13, 13 };
static size_t J2[ 15 ] = { 0, 1, 4, 5, 8, 10, 11, 11, 12, 9, 11, 14, 2, 10, 14 };
// empty rows: 0, 2, 4, 5, 8, 9, 10, 11, 14
// empty cols: 2, 3, 6, 7, 9, 13

static bool test_vector_of_zeroes(
	std::vector< size_t > &v, const char * const name
) {
	std::vector< size_t >::const_iterator max_it =
		std::max_element( v.cbegin(), v.cend() );
	bool result = true;
	if( *max_it != 0 ) {
		std::cerr << "a " << name << " entry is wrong" << std::endl;
		for( size_t i = 0; i < v.size(); i++ ) {
			std::cerr << name << " " << i << ", count " << v[ i ] << std::endl;
		}
		result = false;
	}
	return result;
}

template< typename IteratorType >
RC checkCoordinates(
		const IteratorType &it,
		const IteratorType &copy,
		const bool silent = false
) {
	if( it.i() != copy.i() || it.j() != copy.j() ) {
		if( !silent ) {
			std::cerr << "Iterator copy yields coordinates different from original:\n"
				<< "\t" << it.i() << " != " << copy.i() << " AND/OR\n"
				<< "\t" << it.j() << " != " << copy.j() << ".\n";
		}
		return FAILED;
	}
	return SUCCESS;
}

template< typename IteratorType >
RC checkCopy(
	const IteratorType &it,
	const typename std::enable_if<
		!std::is_same< typename IteratorType::ValueType, void >::value,
	void >::type * = nullptr
) {
	IteratorType copy;
	copy = it;
	grb::RC ret = checkCoordinates( copy, it );
	if( it.v() != copy.v() ) {
		std::cerr << "Iterator copy yields values different from original:\n"
			<< "\t" << it.v() << " != " << copy.v() << ".\n";
		ret = FAILED;
	}
	if( ret != SUCCESS ) { return ret; }

	// if copy-assignment was OK, let us try copy construction
	IteratorType copied( it );
	ret = checkCoordinates( copied, it );
	if( it.v() != copied.v() ) {
		std::cerr << "Iterator copy yields values different from original:\n"
			<< "\t" << it.v() << " != " << copied.v() << ".\n";
		ret = FAILED;
	}
	return ret;
}

template< typename IteratorType >
RC checkCopy(
	const IteratorType &it,
	const typename std::enable_if<
		std::is_same< typename IteratorType::ValueType, void >::value,
	void >::type * = nullptr
) {
	IteratorType copy;
	copy = it;
	grb::RC ret = checkCoordinates( copy, it );
	if( ret != SUCCESS ) { return ret; }

	// if copy-assignment was OK, let us try copy construction
	IteratorType copied( it );
	ret = checkCoordinates( copied, it );
	return ret;
}

template< typename IteratorType >
RC checkMove(
	const IteratorType &it,
	const typename std::enable_if<
		std::is_same< typename IteratorType::ValueType, void >::value,
	void >::type * = nullptr
) {
	grb::Matrix< typename IteratorType::ValueType > empty( 0, 0 );
	IteratorType dummy = empty.cbegin();
	IteratorType copy = it;
	dummy = std::move( copy );
	grb::RC ret = checkCoordinates( dummy, it, true );
	if( ret != SUCCESS ) {
		std::cerr << "Moved iterator yields coordinates different from original:\n"
			<< "\t" << it.i() << " != " << dummy.i() << " AND/OR\n"
			<< "\t" << it.j() << " != " << dummy.j() << ".\n";
	}
	if( ret != SUCCESS ) { return ret; }

	// if move-assignment was OK, let us now try the move constructor
	IteratorType moved( std::move( dummy ) );
	ret = checkCoordinates( moved, it, true );
	if( ret != SUCCESS ) {
		std::cerr << "Moved iterator yields coordinates different from original:\n"
			<< "\t" << it.i() << " != " << moved.i() << " AND/OR\n"
			<< "\t" << it.j() << " != " << moved.j() << ".\n";
	}
	return ret;
}

template< typename IteratorType >
RC checkMove(
	const IteratorType &it,
	const typename std::enable_if<
		!std::is_same< typename IteratorType::ValueType, void >::value,
	void >::type * = nullptr
) {
	grb::Matrix< typename IteratorType::ValueType > empty( 0, 0 );
	IteratorType dummy = empty.cbegin();
	IteratorType copy = it;
	dummy = std::move( copy );
	grb::RC ret = checkCoordinates( dummy, it, true );
	if( ret != SUCCESS ) {
		std::cerr << "Moved iterator yields coordinates different from original:\n"
			<< "\t" << it.i() << " != " << dummy.i() << " AND/OR\n"
			<< "\t" << it.j() << " != " << dummy.j() << ".\n";
	}
	if( it.v() != dummy.v() ) {
		std::cerr << "Moved iterator yields values different from original:\n"
			<< "\t" << it.v() << " != " << dummy.v() << ".\n";
		ret = ret ? ret : FAILED;
	}
	if( ret != SUCCESS ) { return ret; }

	// if move-assignment was OK, let us now try the move constructor
	IteratorType moved( std::move( dummy ) );
	ret = checkCoordinates( moved, it, true );
	if( ret != SUCCESS ) {
		std::cerr << "Moved iterator yields coordinates different from original:\n"
			<< "\t" << it.i() << " != " << moved.i() << " AND/OR\n"
			<< "\t" << it.j() << " != " << moved.j() << ".\n";
	}
	if( it.v() != moved.v() ) {
		std::cerr << "Moved iterator yields values different from original:\n"
			<< "\t" << it.v() << " != " << moved.v() << ".\n";
		ret = ret ? ret : FAILED;
	}
	return ret;
}

template< typename ValT >
RC checkMoveAndCopy(
	const Matrix< ValT > &mat
) {
	grb::RC ret = SUCCESS;
	for( auto it = mat.cbegin(); it != mat.cend(); ++it ) {
		ret = ret ? ret : checkMove( it );
		ret = ret ? ret : checkCopy( it );
	}
	return ret;
}

template< typename ValT, typename OrigIterT >
RC test_matrix_iter(
	OrigIterT orig_begin, OrigIterT orig_end,
	size_t row_col_offset, const Matrix< ValT > &mat
) {
	if( checkMoveAndCopy( mat ) != SUCCESS ) {
		return FAILED;
	}

	using NZC = internal::NonzeroStorage< size_t, size_t, ValT >;
	std::vector< NZC > mat_values;
	utils::get_matrix_nnz( mat, mat_values );
	utils::row_col_nz_sort< size_t, size_t, ValT >( mat_values.begin(),
		mat_values.end() );

	size_t num_local_matrix_nzs;
	bool locally_equal = utils::compare_non_zeroes< ValT >(
		nrows(mat),
		utils::makeNonzeroIterator< size_t, size_t, ValT >( orig_begin ),
		utils::makeNonzeroIterator< size_t, size_t, ValT >( orig_end ),
		utils::makeNonzeroIterator< size_t, size_t, ValT >( mat_values.cbegin() ),
		utils::makeNonzeroIterator< size_t, size_t, ValT >( mat_values.cend() ),
		num_local_matrix_nzs, std::cerr, true
	);

	static_assert( std::is_unsigned< size_t >::value, "use unsigned count" );
	std::vector< size_t > row_count( 15, 0 ), col_count( 15, 0 );
	for( auto it = orig_begin; it != orig_end; ++it ) {
		if( grb::internal::Distribution<>::global_index_to_process_id(
				it.i(), 15, grb::spmd<>::nprocs()
			) != spmd<>::pid()
		) {
			continue;
		}
		(void) row_count[ it.i() - row_col_offset ]++;
		(void) col_count[ it.j() - row_col_offset ]++;
	}
	for( const NZC &nz : mat_values ) {
		(void) row_count[ nz.i() - row_col_offset ]--;
		(void) col_count[ nz.j() - row_col_offset ]--;
	}

	// in case of negative count, use arithmetic wrap-around of size_t (unsigned)
	bool rows_match = test_vector_of_zeroes( row_count, "row" );
	bool cols_match = test_vector_of_zeroes( col_count, "column" );

	size_t count = num_local_matrix_nzs;
	RC rc = collectives<>::allreduce( count, grb::operators::add< size_t >() );
	if( rc != SUCCESS ) {
		std::cerr << "Cannot reduce nonzero count\n";
		return PANIC;
	}
	if( count != 15 ) {
		std::cerr << "\tunexpected number of entries ( " << count << " ), "
			<< "expected 15.\n";
		return FAILED;
	}

	return locally_equal && count == nnz( mat ) && rows_match && cols_match
		? SUCCESS
		: FAILED;
}

template< typename ValT >
RC test_matrix(
	size_t num_nnz, const size_t * rows, const size_t * cols,
	const ValT * values,
	size_t row_col_offset, const Matrix< ValT > &mat
) {
	auto orig_begin = internal::makeSynchronized( rows, cols, values, num_nnz );
	auto orig_end = internal::makeSynchronized( rows + num_nnz, cols + num_nnz,
		values + num_nnz, 0 );
	grb::RC ret = test_matrix_iter( orig_begin, orig_end, row_col_offset, mat );
	if(
		collectives<>::allreduce( ret, grb::operators::any_or< RC >() ) != SUCCESS
	) {
		std::cerr << "Cannot reduce error code\n";
		ret = PANIC;
	}
	return ret;
}

template< typename ValT >
RC test_matrix(
	size_t num_nnz, const size_t * rows, const size_t * cols,
	size_t row_col_offset, const Matrix< ValT > &mat
) {
	auto orig_begin = internal::makeSynchronized( rows, cols, num_nnz );
	auto orig_end = internal::makeSynchronized( rows + num_nnz, cols + num_nnz, 0 );
	grb::RC ret = test_matrix_iter( orig_begin, orig_end, row_col_offset, mat );
	if(
		collectives<>::allreduce( ret, grb::operators::any_or< RC >() ) != SUCCESS
	) {
		std::cerr << "Cannot reduce error code\n";
		ret = PANIC;
	}
	return ret;
}

void grb_program( const size_t &n, grb::RC &rc ) {
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;

	// initialize test
	grb::Matrix< double > A( 15, 15 );
	grb::Matrix< double > B( 15, 15 );
	grb::Matrix< double > C( n, n );
	grb::Matrix< void > D( n, n );

	rc = grb::resize( A, 15 );
	rc = rc ? rc : grb::resize( B, 15 );
	rc = rc ? rc : grb::resize( C, 15 );
	rc = rc ? rc : grb::resize( D, 15 );
	rc = rc ? rc : grb::buildMatrixUnique( A, I1, J1, data, 15, SEQUENTIAL );
	rc = rc ? rc : grb::buildMatrixUnique( B, I2, J2, data, 15, SEQUENTIAL );
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation FAILED\n";
		std::cerr << std::flush;
		return;
	}

	// test output iteration for A
	rc = test_matrix( 15, I1, J1, data_double, 0, A );
	if( rc != SUCCESS ) {
		std::cerr << "\tsubtest 1 (diagonal 15 x 15 matrix) FAILED" << std::endl;
		return;
	}

	// test output iteration for B
	rc = test_matrix( 15, I2, J2, data_double, 0, B );
	if( rc != SUCCESS ) {
		std::cerr << "\tsubtest 2 (general 15 x 15 matrix) FAILED" << std::endl;
		return;
	}

	const size_t offset = n - 15;
	for( size_t i = 0; i < 15; ++i ) {
		I1[ i ] += offset;
		I2[ i ] += offset;
		J1[ i ] += offset;
		J2[ i ] += offset;
	}
	rc = rc ? rc : grb::buildMatrixUnique( C, I2, J2, data, 15, SEQUENTIAL );
	rc = rc ? rc : grb::buildMatrixUnique( D, I1, J1, 15, SEQUENTIAL );
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation 2 FAILED" << std::endl;
		return;
	}

	// test output iteration for C
	rc = test_matrix( 15, I2, J2, data_double, offset, C );
	if( rc != SUCCESS ) {
		std::cerr << "\tsubtest 3 (general " << n << " x " << n << " matrix) FAILED"
			<< std::endl;
		return;
	}

	rc = test_matrix( 15, I1, J1, offset, D );
	if( rc != SUCCESS ) {
		std::cerr << "\tsubtest 4 (diagonal pattern " << n << " x " << n << " "
			<< "matrix) FAILED" << std::endl;
		return;
	}
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 100;

	// error checking
	if( argc > 2 ) {
		printUsage = true;
	}
	if( argc == 2 ) {
		size_t read;
		std::istringstream ss( argv[ 1 ] );
		if( !(ss >> read)) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( !ss.eof() ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( read % 2 != 0 ) {
			std::cerr << "Given value for n is odd\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an even integer (test size)"
			<< std::endl;
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, in, out, true ) != SUCCESS ) {
		std::cout << std::flush;
		std::cerr << "Launching test FAILED" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << std::flush;
		std::cerr << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
		return 255;
	} else {
		std::cout << "Test OK" << std::endl;
		return 0;
	}
}

