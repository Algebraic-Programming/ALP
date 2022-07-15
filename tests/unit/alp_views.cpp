
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

#include "alp.hpp"

template< typename MatrixType >
void print_matrix( std::string name, const MatrixType &A ) {

	if( ! alp::internal::getInitialized( A ) ) {
		std::cout << "Matrix " << name << " uninitialized.\n";
		return;
	}

	std::cout << "Matrix " << name << " of size " << alp::dims( A ).first << " x " << alp::dims( A ).second << " contains the following elements:\n";

	for( size_t row = 0; row < alp::nrows( A ); ++row ) {
		std::cout << "[\t";
		for( size_t col = 0; col < alp::ncols( A ); ++col ) {
			auto pos  = alp::internal::getStorageIndex( A, row, col );
			// std::cout << "(" << pos << "): ";
			std::cout << alp::internal::access( A, pos ) << "\t";
		}
		std::cout << "]\n";
	}
}

template< typename VectorType >
void print_vector( std::string name, const VectorType &v ) {

	if( ! alp::internal::getInitialized( v ) ) {
		std::cout << "Vector " << name << " uninitialized.\n";
		return;
	}

	std::cout << "Vector " << name << " of size " << alp::dims( v ).first << " contains the following elements:\n";

	std::cout << "[";
	for( size_t row = 0; row < alp::nrows( v ); ++row ) {
		std::cout << v[ row ] << "\t";
	}
	std::cout << "]\n";
}

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

	auto MT = alp::get_view< alp::view::Views::transpose >( M );
	print_matrix( "M^T", MT );

	auto Mdiag = alp::get_view< alp::view::Views::diagonal >( M );
	print_vector( "Mdiag", Mdiag );

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

