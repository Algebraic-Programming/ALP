
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
 * @author Alberto Scolari
 * @date 20/06/2022
 * @brief test facilities for the buildMatrixUnique() API call
 *
 * They test whether the generated matrix stores all the elements, but NOT
 * whether they are stored in a specific order or format, since the
 * specification does not prescribe any of these details for the matrix produced
 * via buildMatrixUnique() (nor for matrices in gineral).
 */

#include <iostream>
#include <cstddef>
#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <array>
#include <vector>
#include <iterator>
#include <utility>
#include <unordered_map>
#include <cassert>
#include <sstream>
#include <type_traits>
#include <cstdlib>

#include <graphblas.hpp>
#include <graphblas/utils/NonzeroStorage.hpp>
#include <graphblas/utils/NonzeroIterator.hpp>

#include <utils/assertions.hpp>
#include <utils/matrix_generators.hpp>
#include <utils/matrix_values_check.hpp>


using namespace grb;

#define LOG() std::cout
#define MAIN_LOG( text ) if ( spmd<>::pid() == 0 ) { LOG() << text; }

template< typename T > void test_matrix_sizes_match(
	const Matrix< T >& mat1,
	const Matrix< T >& mat2
) {
	ASSERT_EQ( grb::nrows( mat1 ), grb::nrows( mat2 ) );
	ASSERT_EQ( grb::ncols( mat1 ), grb::ncols( mat2 ) );
}

using DefRowT = std::size_t;
using DefColT = std::size_t;
template< typename T > using NZ = utils::NonzeroStorage< DefRowT, DefColT, T >;

/**
 * @brief gets the nonzeroes of \p mat, stores them into \p values and sorts them.
 */
template<
	typename T,
	enum Backend implementation
>
static void get_nnz_and_sort(
	const Matrix< T, implementation >& mat,
	std::vector< NZ< T > >& values
) {
	utils::get_matrix_nnz( mat, values );
	utils::row_col_nz_sort< DefRowT, DefColT, T >( values.begin(), values.end() );
}

/**
 * @brief compares the nonzeroes of \p mat1 and \p mat2 and returns true iff they
 * are equal in number and value.
 */
template<
	typename T,
	enum Backend implementation
>
bool matrices_values_are_equal(
	const Matrix< T, implementation >& mat1,
	const Matrix< T, implementation >& mat2,
	std::size_t& num_mat1_nnz,
	std::size_t& num_mat2_nnz,
	bool log_all_differences = false
) {

	std::vector< NZ< T > > serial_values;
	get_nnz_and_sort( mat1, serial_values );

	std::vector< NZ< T > > parallel_values;
	get_nnz_and_sort( mat2, parallel_values );

	const std::size_t mat_size{ grb::nnz( mat1) };

	if( serial_values.size() != parallel_values.size() ) {
		LOG() << "the numbers of entries differ" << std::endl;
		return false;
	}

	if( serial_values.size() != mat_size && implementation != Backend::BSP1D ) {
		LOG() << "different number of non-zeroes: actual: " << serial_values.size()
			<< ", expected: " << mat_size << std::endl;
		return false;
	}

	std::size_t checked_values;
	bool match { grb::utils::compare_non_zeroes< T >( grb::nrows( mat1 ),
			utils::makeNonzeroIterator< DefRowT, DefColT, T >( serial_values.cbegin() ),
			utils::makeNonzeroIterator< DefRowT, DefColT, T >( serial_values.cend() ),
			utils::makeNonzeroIterator< DefRowT, DefColT, T >( parallel_values.cbegin() ),
			utils::makeNonzeroIterator< DefRowT, DefColT, T >( parallel_values.cend() ),
			checked_values, LOG(), log_all_differences ) };

	if( checked_values != parallel_values.size() ) {
		LOG() << "cannot check all non-zeroes" << std::endl;
		return false;
	}
	enum RC rc{ collectives<>::allreduce( checked_values, grb::operators::add< std::size_t >() ) };
	ASSERT_RC_SUCCESS( rc );
	if( checked_values != mat_size ) {
		LOG() << "total number of non-zeroes different from matrix size" << std::endl;
		return false;
	}
	num_mat1_nnz = serial_values.size();
	rc = collectives<>::allreduce( num_mat1_nnz, grb::operators::add< std::size_t >() );
	ASSERT_RC_SUCCESS( rc );
	num_mat2_nnz = parallel_values.size();
	rc = collectives<>::allreduce( num_mat2_nnz, grb::operators::add< std::size_t >() );
	ASSERT_RC_SUCCESS( rc );

	return match;
}

/**
 * @brief build a matrix storing the nonzeroes in the range [ \p begin, \p end )
 * 	and checks whether the call to #buildMatrixUnique is successful and whether
 * 	the produced matrix contains the correct number of nonzeroes.
 *
 * @tparam T matrix value type
 * @tparam IterT type of the input iterator, which MUST have a \p - operator
 * @tparam implementation ALP backend
 * @param m matrix to build
 * @param begin beginning of nonzeroes
 * @param end end of nonzeroes
 * @param expected_num_global_nnz expected number of nonzeroes stored in the entire matrix
 * @param expected_num_local_nnz expected number of nonzeroes stored on the local node
 * 									(possibly != \p expected_num_global_nnz) for distributed backends
 * @param mode whether the input iterator is sequential or parallel
 */
template<
	typename T,
	typename IterT,
	enum Backend implementation = config::default_backend
>
void build_matrix_and_check(
	Matrix< T, implementation >& m,
	IterT begin,
	IterT end,
	std::size_t expected_num_global_nnz,
	std::size_t expected_num_local_nnz,
	IOMode mode
) {
	ASSERT_EQ( end - begin, static_cast< typename IterT::difference_type >( expected_num_local_nnz ) );

	RC ret { buildMatrixUnique( m, begin, end, mode ) };
	ASSERT_RC_SUCCESS( ret );
	ASSERT_EQ( nnz( m ), expected_num_global_nnz );
}

/**
 * @brief tests matrix generation for both the sequential and the parallel mode,
 * 	checking that the number of nonzeros and the values themselves are equal.
 *
 * @tparam T matrix value type
 * @tparam IterT type of the input iterator, which MUST have a \p - operator and
 * 	static methods \p IterT::make_begin and \p IterT::make_end
 * @tparam implementation ALP backend
 * @param sequential_matrix matrix to be populated from sequential input
 * @param parallel_matrix matrix to be populated from parallel input
 * @param iter_sizes sizes to be passed to the iterator generator
 */
template<
	typename T,
	typename IterT,
	enum Backend implementation
>
void test_matrix_generation(
	Matrix< T, implementation >& sequential_matrix,
	Matrix< T, implementation >& parallel_matrix,
	const typename IterT::input_sizes_t& iter_sizes
) {
	constexpr bool iterator_is_random{ std::is_same<
		typename std::iterator_traits<IterT>::iterator_category,
		std::random_access_iterator_tag >::value };

	MAIN_LOG( ">> " << ( iterator_is_random ? "RANDOM" : "FORWARD" ) << " ITERATOR "
		<< "-- size " << nrows( sequential_matrix ) << " x "
		<< ncols( sequential_matrix ) << std::endl
	);

	const std::size_t num_nnz{ IterT::compute_num_nonzeroes( iter_sizes) };
	build_matrix_and_check( sequential_matrix, IterT::make_begin( iter_sizes ),
		IterT::make_end( iter_sizes ), num_nnz, num_nnz, IOMode::SEQUENTIAL );

	const std::size_t par_num_nnz{ utils::compute_parallel_num_nonzeroes( num_nnz ) };

	build_matrix_and_check( parallel_matrix, IterT::make_parallel_begin( iter_sizes),
		IterT::make_parallel_end( iter_sizes), num_nnz, par_num_nnz, IOMode::PARALLEL );

	test_matrix_sizes_match( sequential_matrix, parallel_matrix );
	std::size_t serial_nz, par_nz;
	ASSERT_TRUE( matrices_values_are_equal( sequential_matrix, parallel_matrix,
		serial_nz, par_nz, false ) );
	ASSERT_EQ( par_nz, serial_nz );

	MAIN_LOG( "<< OK" << std::endl );
}

/**
 * @brief generates a matrix of \p nrows x \p ncols from the values stored in
 * 	\p mat_nzs, sorting them if \p sort_nzs is true
 */
template<
	typename ValT,
	enum Backend implementation = config::default_backend
>
void test_matrix_from_vectors(
	std::size_t nrows, std::size_t ncols,
	std::vector< NZ< ValT > >& mat_nzs,
	bool sort_nzs = false
) {
	Matrix< ValT, implementation > mat( nrows, ncols );
	const std::size_t num_original_nzs{ mat_nzs.size() };
	const std::size_t per_proc{ ( num_original_nzs + spmd<>::nprocs() - 1 ) / spmd<>::nprocs() };
	const std::size_t first{ std::min( per_proc * spmd<>::pid(), num_original_nzs) };
	const std::size_t last{ std::min( first + per_proc, num_original_nzs ) };

#ifdef _DEBUG
	for( unsigned i{ 0 }; i < spmd<>::nprocs(); i++) {
		if( spmd<>::pid() == i ) {
			std::cout << "process " << i << " from " << first << " last " << last << std::endl;
		}
		spmd<>::barrier();
	}
#endif

	RC ret{ buildMatrixUnique( mat,
		utils::makeNonzeroIterator< DefRowT, DefColT, ValT >( std::next( mat_nzs.begin(), first ) ),
		utils::makeNonzeroIterator< DefRowT, DefColT, ValT >( std::next( mat_nzs.begin(), last ) ),
		IOMode::PARALLEL ) // PARALLEL is needed here because each process advances iterators
							// differently via std::next()
	};
	ASSERT_RC_SUCCESS( ret );
	ASSERT_EQ( nnz( mat ), mat_nzs.size() );

	std::vector< NZ< ValT > > sorted_mat_values;
	get_nnz_and_sort( mat, sorted_mat_values );
	std::size_t num_sorted_mat_values{ sorted_mat_values.size() };
	// reduce for sparse matrices: only the total number should be equal
	RC rc{ collectives<>::allreduce( num_sorted_mat_values, grb::operators::add< std::size_t >() ) };
	ASSERT_RC_SUCCESS( rc );
	ASSERT_EQ( num_sorted_mat_values, mat_nzs.size() );

	if ( sort_nzs ) {
		utils::row_col_nz_sort< DefRowT, DefColT, ValT >( mat_nzs.begin(), mat_nzs.end() );
	}

	std::size_t checked_nzs;
	ASSERT_TRUE(
		grb::utils::compare_non_zeroes< ValT >( nrows,
			utils::makeNonzeroIterator< DefRowT, DefColT, ValT >( mat_nzs.cbegin() ),
			utils::makeNonzeroIterator< DefRowT, DefColT, ValT >( mat_nzs.cend() ),
			utils::makeNonzeroIterator< DefRowT, DefColT, ValT >( sorted_mat_values.cbegin() ),
			utils::makeNonzeroIterator< DefRowT, DefColT, ValT >( sorted_mat_values.cend() ),
			checked_nzs, LOG(), true )
	);
	rc = collectives<>::allreduce( checked_nzs, grb::operators::add< std::size_t >() );
	ASSERT_RC_SUCCESS( rc );

	ASSERT_EQ( checked_nzs, mat_nzs.size() );

	MAIN_LOG( "<< OK" << std::endl );
}

/**
 * @brief shuffles the elements of the vector \p mat_nzs to a random order.
 *
 * A pseudo-randomizer is used with constant seed in order to always get the same
 * numbers sequence, thus ensuring reproducibility.
 */
template< typename ValT > void randomize_vector(
	std::vector< NZ< ValT > >& mat_nzs
) {
	std::srand( 13 );
	struct randomizer {
		typename std::iterator_traits< typename std::vector< NZ< ValT > >::iterator >::difference_type
			operator()( std::size_t n ) {
				return std::rand() % n;
			}
	} r;
	std::random_shuffle( mat_nzs.begin(), mat_nzs.end(), r );
}

/**
 * @brief generates a vector of nonzeroes from the iterator of type \p ParIterT,
 *  permutes the vector and generates a matrix of values of type \p ValT and sizes
 * 	\p nrows x \p ncols from it, finally testing it via test_matrix_from_vectors()
 */
template<
	typename ValT,
	typename ParIterT,
	enum Backend implementation = config::default_backend
>
void test_matrix_from_permuted_iterators(
	std::size_t nrows, std::size_t ncols,
	const typename ParIterT::input_sizes_t& iter_sizes
) {
	using NZC = NZ< ValT >;
	std::vector< NZC > mat_nz;

	for( ParIterT it{ ParIterT::make_begin( iter_sizes ) }; it != ParIterT::make_end( iter_sizes ); ++it ) {
		mat_nz.push_back( NZC( it.i(), it.j(), it.v() ) );
	}
	randomize_vector( mat_nz );

	test_matrix_from_vectors( nrows, ncols, mat_nz, true );
}

/**
 * @brief generates matrices of sizes \p nrows x \p ncols from the iterators of type
 * 	\p ParIterT (random access iterator) and of type \p SeqIterT (forward iterator),
 * 	testing that their values are the same
 *
 * @tparam T matrix value type
 * @tparam ParIterT random access iterator type for parallel generation
 * @tparam SeqIterT forward iterator for sequential generation
 * @tparam implementation ALp backend
 * @param nrows number of matrix rows
 * @param ncols number of matrix columns
 * @param iter_sizes size for the iterator creation via the static methods
 * 	\p IteratorType::make_begin and \p IteratorType::make_end
 */
template<
	typename T,
	typename ParIterT,
	typename SeqIterT,
	enum Backend implementation = config::default_backend
>
void test_sequential_and_parallel_matrix_generation(
		std::size_t nrows, std::size_t ncols,
		const typename ParIterT::input_sizes_t& iter_sizes
) {
	Matrix< T, implementation > par_sequential_matrix( nrows, ncols );
	Matrix< T, implementation > par_parallel_matrix( nrows, ncols );
	test_matrix_generation< T, ParIterT, implementation >( par_sequential_matrix, par_parallel_matrix, iter_sizes );

	Matrix< T, implementation > seq_sequential_matrix( nrows, ncols );
	Matrix< T, implementation > seq_parallel_matrix( nrows, ncols );
	test_matrix_generation< T, SeqIterT, implementation >( seq_sequential_matrix, seq_parallel_matrix, iter_sizes );

	// cross-check parallel vs sequential
	std::size_t serial_nz, par_nz;
	ASSERT_TRUE( matrices_values_are_equal( par_parallel_matrix, seq_parallel_matrix,
		serial_nz, par_nz, true ) );
	const std::size_t serial_num_nnz{ SeqIterT::compute_num_nonzeroes( iter_sizes) },
		parallel_num_nnz{ ParIterT::compute_num_nonzeroes( iter_sizes) };

	ASSERT_EQ( serial_num_nnz, parallel_num_nnz ); // check iterators work properly
	// now check the number of non-zeroes returned via iterators globlly match
	ASSERT_EQ( serial_nz, parallel_num_nnz );
	ASSERT_EQ( par_nz, parallel_num_nnz );

	MAIN_LOG( ">> RANDOMLY PERMUTED" << std::endl );
	test_matrix_from_permuted_iterators< T, ParIterT >( nrows, ncols, iter_sizes );
}

/**
 * @brief tests the matrix generation from custom vectors.
 *
 * The generation is inherently parallel, because the underlying nonzeroes container
 * (std::vector) produces random access iterators.
 */
template< enum Backend implementation = config::default_backend >
	void test_matrix_from_custom_vectors()
{
	constexpr std::size_t num_matrices{ 2 };
	using NZC = NZ< int >;
	using SP = std::pair< std::size_t, std::size_t >;

	std::array< SP, num_matrices > sizes{ SP( 7, 7 ), SP( 3456, 8912 ) };
	std::array< std::vector< NZC >, num_matrices > coordinates{
		std::vector< NZC >{ NZC(0,1,0), NZC(0,3,1), NZC(0,4,-1), NZC(0,5,-2), NZC(0,6,-3),
			NZC(1,3,2), NZC(1,4,-4), NZC(1,5,-5), NZC(1,6,-6),
			NZC(2,2,3),
			NZC(3,4,4),
			NZC(4,0,5), NZC(4,2,6),
			NZC(5,0,7), NZC(5,1,8), NZC(5,2,9), NZC(5,3,10), NZC(5,4,11), NZC(5,5,12)
		},
		std::vector< NZC >{ NZC( 1, 2, 0 ), NZC( 1, 4, 1 ), NZC( 1, 5, 2 ), NZC( 1, 7, 3 ),
			NZC( 2, 0, 4 ), NZC( 2, 1, 5 ), NZC( 2, 2, 6 ),
			NZC( 3, 1, 7 ), NZC( 3, 2, 8 ), NZC( 3, 4, 9 ), NZC( 3, 8909, 10 ), NZC( 3, 8910, 11 ), NZC( 3, 8911, 12 ),
			NZC( 3452, 2000, 13 ), NZC( 3452, 2002, 14 ), NZC( 3452, 8910, 15 ), NZC( 3452, 8911, 16 )
		}
	};

	for( std::size_t i{ 0 }; i < num_matrices; i++ ) {
		std::vector< NZC >& mat_nz{ coordinates[ i ] };
		SP& size{ sizes[ i ] };

		MAIN_LOG( ">>>> CUSTOM " << size.first << " x " << size.second << std::endl
			<< ">> SORTED NON-ZEROES" << std::endl );

		test_matrix_from_vectors( size.first, size.second, mat_nz );

		randomize_vector( mat_nz );
		MAIN_LOG( ">> RANDOMLY PERMUTED NON-ZEROES" << std::endl );
		test_matrix_from_vectors( size.first, size.second, mat_nz, true );
	}
}

static const char* const std_caption{ "got exception: " };

static void print_exception_text(
	const char * text,
	const char * caption = std_caption
) {
	std::stringstream stream;
	if( spmd<>::nprocs() > 1UL ) {
		stream << "Machine " << spmd<>::pid() << " - ";
	}
	stream << caption << std::endl
		<< ">>>>>>>>" << std::endl
		<< text << std::endl
		<< "<<<<<<<<" << std::endl;
	LOG() << stream.str();
}

/**
 * @brief tests building the matrix from invalid inputs, which should cause the
 * 	generation to fail.
 */
void test_invalid_inputs()
{

	using NZC = NZ< int >;
	constexpr std::size_t rows{ 6 }, cols{ 7 };

	std::array< std::vector< NZC >, 2 > coordinates{
		std::vector< NZC >{ NZC(0,1,0), NZC(0,3,1), NZC(0,4,-1), NZC(0,5,-2), NZC(0,6,-3),
			NZC(1,3,2), NZC(1,4,-4), NZC(1,5,-5), NZC(1,6,-6),
			NZC(2,2,3),
			NZC(rows,4,4), // wrong row
			NZC(4,0,5), NZC(4,2,6),
			NZC(5,0,7), NZC(5,1,8), NZC(5,2,9), NZC(5,3,10), NZC(5,4,11), NZC(5,5,12)
		},
		std::vector< NZC >{ NZC(0,1,0), NZC(0,3,1), NZC(0,4,-1), NZC(0,5,-2), NZC(0,6,-3),
			NZC(1,3,2), NZC(1,4,-4), NZC(1,5,-5), NZC(1,6,-6),
			NZC(2,2,3),
			NZC(3,cols + 1,4), // wrong column
			NZC(4,0,5), NZC(4,2,6),
			NZC(5,0,7), NZC(5,1,8), NZC(5,2,9), NZC(5,3,10), NZC(5,4,11), NZC(5,5,12)
		}
	};

	for( std::vector< NZC > &c : coordinates ) {
		Matrix< int > m( rows, cols );
		RC ret = buildMatrixUnique( m,
			utils::makeNonzeroIterator< DefRowT, DefColT, int >( c.cbegin() ),
			utils::makeNonzeroIterator< DefRowT, DefColT, int >( c.cend() ),
			PARALLEL
		);
		ASSERT_NE( ret, SUCCESS );
	}
}

void grbProgram(
	const void *,
	const size_t, int &error
) {

	try {

		MAIN_LOG( "==== Testing building from invalid inputs" << std::endl );
		test_invalid_inputs();
		MAIN_LOG( "<< OK" << std::endl );

		// test generation of diagonal matrices of multiple sizes
		std::initializer_list< std::size_t > diag_sizes{ spmd<>::nprocs(), spmd<>::nprocs() + 9,
			spmd<>::nprocs() + 16, 100003 };

		MAIN_LOG( "==== Testing diagonal matrices" << std::endl );
		for( const std::size_t& mat_size : diag_sizes ) {
			test_sequential_and_parallel_matrix_generation< int, utils::diag_iterator< true >, utils::diag_iterator< false > >(
				mat_size, mat_size, mat_size );
		}

		// test the generation of badn matrices, of multiple sizes and bands
		std::initializer_list< std::size_t > band_sizes{ 17, 77, 107, 11467, 41673 };
		for( const std::size_t& mat_size : band_sizes ) {
			MAIN_LOG( "==== Testing matrix with band 1" << std::endl );
			test_sequential_and_parallel_matrix_generation< int,
				utils::band_iterator< 1, true >, utils::band_iterator< 1, false > >(
				mat_size, mat_size, mat_size );

			MAIN_LOG( "==== Testing matrix with band 2" << std::endl );
			test_sequential_and_parallel_matrix_generation< int,
				utils::band_iterator< 2, true >, utils::band_iterator< 2, false > >(
				mat_size, mat_size, mat_size );

			MAIN_LOG( "==== Testing matrix with band 7" << std::endl );
			test_sequential_and_parallel_matrix_generation< int,
				utils::band_iterator< 7, true >, utils::band_iterator< 7, false > >(
				mat_size, mat_size, mat_size );

			MAIN_LOG( "==== Testing matrix with band 8" << std::endl );
			test_sequential_and_parallel_matrix_generation< int,
				utils::band_iterator< 8, true >, utils::band_iterator< 8, false > >(
				mat_size, mat_size, mat_size );
		}

		// test dense matrices
		std::initializer_list< std::array< std::size_t, 2 > > matr_sizes{
			{ spmd<>::nprocs(), spmd<>::nprocs() }, { 77, 70 }, { 130, 139 }, { 1463, 5376 }
		};
		MAIN_LOG( "==== Testing dense matrices" << std::endl );
		for( const std::array< std::size_t, 2 >& mat_size : matr_sizes ) {
			test_sequential_and_parallel_matrix_generation< int,
				utils::dense_mat_iterator< int, true >,
				utils::dense_mat_iterator< int, false > >(
				mat_size[0], mat_size[1], mat_size );
		}

		// test sparse matrices from custom vectors
		MAIN_LOG( "==== Testing sparse matrix from custom vectors" << std::endl );
		test_matrix_from_custom_vectors();

	} catch ( const std::exception& e ) {
		print_exception_text( e.what() );
		error = 1;
	} catch( ... ) {
		LOG() << "unknown exception" <<std::endl;
		error = 1;
	}
	// assumes SUCCESS is the smallest value in enum RC to perform reduction
	static_assert( SUCCESS < FAILED, "RC::SUCCESS should be the minimum value of RC" );
	RC rc_red = collectives<>::allreduce( error, grb::operators::max< int >() );
	if ( rc_red != SUCCESS ) {
		std::cerr << "Cannot reduce error code, communication issue!" << std::endl;
		std::abort();
	}
	if( error != 0 ) {
		LOG() << "Some process caught an exception" << std::endl;
	}
}

int main(
	int argc,
	char ** argv
) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << std::endl;

	int error{ 0 };

	Launcher< AUTOMATIC > launcher;
	if( launcher.exec( &grbProgram, nullptr, 0, error ) != SUCCESS ) {
		std::cout << "Could not launch test" << std::endl;
		error = 255;
	}
	if( error == 0 ) {
		std::cout << "Test OK" << std::endl;
	} else {
		std::cout << "Test FAILED" << std::endl;
	}

	return error;
}

