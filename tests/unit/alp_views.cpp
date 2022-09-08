
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

#include <utility>
#include <iostream>

#include <alp.hpp>
#include "../utils/print_alp_containers.hpp"

template< typename T >
void init_matrix( std::vector< T > &A, const size_t rows, const size_t cols ) {

	for( size_t row = 0; row < rows; ++row ) {
		for( size_t col = 0; col < cols; ++col ) {
			A[ row * cols + col ] = row + col;
		}
	}
}

template< typename T >
void print_stdvec_as_matrix( std::string name, const std::vector< T > &vA, const size_t m, const size_t n, const size_t lda ) {

	std::cout << "Vec " << name << ":" << std::endl;
	for( size_t row = 0; row < m; ++row ) {
		std::cout << "[\t";
		for( size_t col = 0; col < n; ++col ) {
			std::cout << vA[ row * lda + col ] << "\t";
		}
		std::cout << "]" << std::endl;
	}
}

template< typename Structure, typename T >
void stdvec_build_matrix( std::vector< T > &vA, const size_t m, const size_t n, const size_t lda, const T zero, const T one ) {

	if( std::is_same< Structure, alp::structures::General >::value ) {
		std::fill( vA.begin(), vA.end(), one );
	} else if( std::is_same< Structure, alp::structures::Symmetric >::value ) {
		std::fill( vA.begin(), vA.end(), one );
	}
}

template< typename MatType, typename T >
void diff_stdvec_matrix(
	const std::vector< T > &vA, const size_t m, const size_t n, const size_t lda,
	const MatType &mA,
	double threshold=1e-7
) {

	if( std::is_same< typename MatType::structure, alp::structures::General >::value ) {
		for( size_t row = 0; row < m; ++row ) {
			for( size_t col = 0; col < n; ++col ) {
				double va = ( double )( vA[ row * lda + col ] );
				double vm = ( double )( alp::internal::access( mA, alp::internal::getStorageIndex( mA, row, col ) ) );
				double re = std::abs( ( va - vm ) / va );
				if( re > threshold ) {
					std::cout << "Error ( " << row << ", " << col << " ): " << va << " v " << vm << std::endl;
				}
			}
		}
	} else if( std::is_same< typename MatType::structure, alp::structures::Symmetric >::value ) {
		for( size_t row = 0; row < m; ++row ) {
			for( size_t col = row; col < n; ++col ) {
				double va = ( double )( vA[ row * lda + col ] );
				double vm = ( double )( alp::internal::access( mA, alp::internal::getStorageIndex( mA, row, col ) ) );
				double re = std::abs( ( va - vm ) / va );
				if( re > threshold ) {
					std::cout << "Error ( " << row << ", " << col << " ): " << va << " v " << vm << std::endl;
				}
			}
		}
	}
}



// alp program
void alpProgram( const size_t &n, alp::RC &rc ) {

	typedef double T;

	alp::Semiring< alp::operators::add< T >, alp::operators::mul< T >, alp::identities::zero, alp::identities::one > ring;

	T zero = ring.getZero< T >();

	// allocate
	const size_t m = 2 * n;
	std::vector< T > M_data( m * n, zero );
	init_matrix( M_data, m, n );

	alp::Matrix< T, alp::structures::General > M( m, n );
	alp::buildMatrix( M, M_data.begin(), M_data.end() );
	print_matrix( "M", M );
	std::cout << "------------" << std::endl;

	// gather view
	auto Mv = alp::get_view( M, alp::utils::range( 1, 3 ), alp::utils::range( 1, 3 ) );
	print_matrix( "Mv", Mv );
	std::cout << "------------" << std::endl;

	// transposed view
	auto MT = alp::get_view< alp::view::Views::transpose >( M );
	print_matrix( "M^T", MT );
	std::cout << "------------" << std::endl;

	// row-view
	auto Mrow = alp::get_view( M, m - 2, alp::utils::range( 1, n - 1 ) );
	print_vector( "Mrow", Mrow );
	std::cout << "------------" << std::endl;

	// row-view on a symmetric matrix
	alp::Matrix< T, alp::structures::Symmetric > A( n, n );
	alp::set( A, alp::get_view< alp::structures::Symmetric >( M, alp::utils::range( 0, n ), alp::utils::range( 0, n ) ) );
	auto Arow = alp::get_view( A, 2, alp::utils::range( 2, n ) );
	print_vector( "Arow", Arow );
	(void)Arow;

	// column-view
	auto Mcol = alp::get_view( M, alp::utils::range( 1, m - 1 ), n - 2 );
	print_vector( "Mcol", Mcol );
	std::cout << "------------" << std::endl;

	// diagonal view on a general (non-square) matrix
	auto Mdiag = alp::get_view< alp::view::Views::diagonal >( M );
	print_vector( "Mdiag", Mdiag );

	// diagonal view on a square matrix
	auto Msquare = alp::get_view< alp::structures::Square >( M, alp::utils::range( 0, 5 ), alp::utils::range( 0, 5 ) );
	auto Mdiagsquare = alp::get_view< alp::view::Views::diagonal >( Msquare );
	print_vector( "Mdiagsquare", Mdiagsquare );

	// view over a vector
	auto Mdiagpart = alp::get_view( Mdiag, alp::utils::range( 1, 3 ) );
	print_vector( "Mdiagpart", Mdiagpart );

	// Vector views
	// allocate vector
	std::vector< T > v_data( m, zero );
	init_matrix( v_data, m, 1 );
	alp::Vector< T, alp::structures::General > v( m );
	alp::buildMatrix( static_cast< decltype( v )::base_type & >( v ), v_data.begin(), v_data.end() );
	print_vector( "v", v );

	// gather view over a vector
	auto v_view = alp::get_view( v, alp::utils::range( 1, 3 ) );
	print_vector( "v_view", v_view );

	// matrix view over an original vector
	auto v_mat_view = alp::get_view< alp::view::matrix >( v );
	print_matrix( "v_mat_view", v_mat_view );

	// matrix view over a vector, which is a vector view over a matrix
	auto Mrow_mat_view = alp::get_view< alp::view::matrix >( Mrow );
	print_matrix( "Mrow_mat_view", Mrow_mat_view );

	rc = alp::SUCCESS;

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
		if( ! ( ss >> read ) ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( ! ss.eof() ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( read == 0 ) {
			std::cerr << "n must be a positive number\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an integer, the "
					 "test size.\n";
		return 1;
	}
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	alp::Launcher< alp::AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alpProgram, in, out, true ) != alp::SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != alp::SUCCESS ) {
		std::cerr << "Test FAILED (" << alp::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}

