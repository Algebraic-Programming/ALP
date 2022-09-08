
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

#include <alp.hpp>
#include "../utils/print_alp_containers.hpp"

using namespace alp;

template< typename T >
void print_stdvec_as_matrix( std::string name, const std::vector< T > & vA, const size_t m, const size_t n, const size_t lda ) {

	std::cout << "Vec " << name << ":" << std::endl;
	for( size_t row = 0; row < m; ++row ) {
		std::cout << "[\t";
		for( size_t col = 0; col < n; ++col ) {
			std::cout << vA[ row * lda + col ] << "\t";
		}
		std::cout << "]" << std::endl;
	}
}

template< typename T, typename Operator, typename Monoid >
void mxm_stdvec_as_matrix(	std::vector< T > & vC, const size_t ldc,
							const std::vector< T > & vA, const size_t lda,
							const std::vector< T > & vB, const size_t ldb,
							const size_t m, const size_t k, const size_t n,
							const Operator oper,
							const Monoid monoid ) {
    
	T temp;

	print_stdvec_as_matrix("vA", vA, n, n, n);
	print_stdvec_as_matrix("vB", vB, n, n, n);
	print_stdvec_as_matrix("vC - PRE", vC, n, n, n);

	for( size_t i = 0; i < m; ++i ) {
		for( size_t j = 0; j < n; ++j ) {
			T & c_val { vC[ i * ldc + j ] };
			for( size_t l = 0; l < k; ++l ) {
					const T & a_val { vA[ i * lda + l ] };
					const T & b_val { vB[ l * ldb + j ] };
					// std::cout << c_val << " += " << a_val << " * " << b_val << std::endl;
					(void)internal::apply( temp, a_val, b_val, oper );
					// std::cout << "temp = " << temp << std::endl;
					(void)internal::foldl( c_val, temp, monoid.getOperator() );
			}
		}
	}

	print_stdvec_as_matrix("vC - POST", vC, n, n, n);

}

template< typename Structure, typename T >
void stdvec_build_matrix( std::vector< T > & vA, const size_t m, const size_t n, const size_t lda, const T zero, const T one ) {

	if( std::is_same< Structure, structures::General >::value ) {
		std::fill( vA.begin(), vA.end(), one );
	} else if( std::is_same< Structure, structures::Symmetric >::value ) {
		std::fill( vA.begin(), vA.end(), one );
	} else if( std::is_same< Structure, structures::UpperTriangular >::value ) {
		for( size_t row = 0; row < m; ++row ) {
			for( size_t col = 0; col < row; ++col ) {
				vA[ row * lda + col ] = zero;
			}
			for( size_t col = row; col < n; ++col ) {
				vA[ row * lda + col ] = one;
			}
		}
	}

}

template< typename MatType, typename T >
void diff_stdvec_matrix( const std::vector< T > & vA, const size_t m, const size_t n, const size_t lda,
						 const MatType & mA, double threshold=1e-7 ) {

	if( std::is_same< typename MatType::structure, structures::General >::value ) {
		for( size_t row = 0; row < m; ++row ) {
			for( size_t col = 0; col < n; ++col ) {
				double va = ( double )( vA[ row * lda + col ] );
				double vm = ( double )( internal::access( mA, internal::getStorageIndex( mA, row, col ) ) );
				double re = std::abs( ( va - vm ) / va );
				if( re > threshold ) {
					std::cout << "Error ( " << row << ", " << col << " ): " << va << " v " << vm << std::endl; 
				}
			}
		}
	} else if( std::is_same< typename MatType::structure, structures::Symmetric >::value ) {
		for( size_t row = 0; row < m; ++row ) {
			for( size_t col = row; col < n; ++col ) {
				double va = ( double )( vA[ row * lda + col ] );
				double vm = ( double )( internal::access( mA, internal::getStorageIndex( mA, row, col ) ) );
				double re = std::abs( ( va - vm ) / va );
				if( re > threshold ) {
					std::cout << "Error ( " << row << ", " << col << " ): " << va << " v " << vm << std::endl; 
				}
			}
		}
	} else if( std::is_same< typename MatType::structure, structures::UpperTriangular >::value ) {
		for( size_t row = 0; row < m; ++row ) {
			for( size_t col = row; col < n; ++col ) {
				double va = ( double )( vA[ row * lda + col ] );
				double vm = ( double )( internal::access( mA, internal::getStorageIndex( mA, row, col ) ) );
				double re = std::abs( ( va - vm ) / va );
				if( re > threshold ) {
					std::cout << "Error ( " << row << ", " << col << " ): " << va << " v " << vm << std::endl; 
				}
			}
		}
	}

}



void alp_program( const size_t & n, alp::RC & rc ) {

	typedef double T;

	alp::Semiring< alp::operators::add< T >, alp::operators::mul< T >, alp::identities::zero, alp::identities::one > ring;

	T one  = ring.getOne< T >();
	T zero = ring.getZero< T >();

	std::vector< T > A_data( n * n, one );
	std::vector< T > B_data( n * n, one );
	std::vector< T > C_data( n * n, zero );

	std::cout << "\tTesting dense General mxm " << n << std::endl;
	// initialize test
	alp::Matrix< T, structures::General > A( n, n );
	alp::Matrix< T, structures::General > B( n, n );
	alp::Matrix< T, structures::General > C( n, n );

	// Initialize input matrices
	rc = alp::buildMatrix( A, A_data.begin(), A_data.end() );
	rc = alp::buildMatrix( B, B_data.begin(), B_data.end() );
	rc = alp::buildMatrix( C, C_data.begin(), C_data.end() );

	print_matrix("A", A);
	print_matrix("B", B);
	print_matrix("C - PRE", C);

	rc = alp::mxm( C, A, B, ring );

	print_matrix("C - POST", C);

	std::vector< T > A_vec( n * n, one );
	std::vector< T > B_vec( n * n, one );
	std::vector< T > C_vec( n * n, zero );

	mxm_stdvec_as_matrix( C_vec, n, A_vec, n, B_vec, n, n, n, n, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid() );

	diff_stdvec_matrix( C_vec, n, n, n, C );

	std::cout << "\n\n=========== Testing Uppertriangular ============\n\n";

	alp::Matrix< T, structures::UpperTriangular > UA( n );
	alp::Matrix< T, structures::UpperTriangular > UB( n );
	alp::Matrix< T, structures::UpperTriangular > UC( n );

	rc = alp::buildMatrix( UA, A_data.begin(), A_data.end() );
	rc = alp::buildMatrix( UB, B_data.begin(), B_data.end() );
	stdvec_build_matrix< structures::General >( C_data, n, n, n, zero, zero );
	rc = alp::buildMatrix( UC, C_data.begin(), C_data.end() );

	print_matrix("UC - PRE", UC);
	rc = alp::mxm( UC, UA, UB, ring );
	print_matrix("UC - POST", UC);

	stdvec_build_matrix< structures::UpperTriangular >( A_vec, n, n, n, zero, one );
	stdvec_build_matrix< structures::UpperTriangular >( B_vec, n, n, n, zero, one );
	stdvec_build_matrix< structures::General >( C_vec, n, n, n, zero, zero );

	mxm_stdvec_as_matrix( C_vec, n, A_vec, n, B_vec, n, n, n, n, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid() );

	diff_stdvec_matrix( C_vec, n, n, n, UC );

	std::cout << "\n\n=========== Testing Symmetric Output ============\n\n";

	alp::Matrix< T, structures::Symmetric > SC( n );

	stdvec_build_matrix< structures::General >( C_data, n, n, n, zero, zero );
	rc = alp::buildMatrix( SC, C_data.begin(), C_data.end() );

	print_matrix("SC - PRE", SC);
	rc = alp::mxm( SC, A, alp::get_view< alp::view::transpose >( A ), ring );
	print_matrix("SC - POST", SC);

	stdvec_build_matrix< structures::General >( A_vec, n, n, n, zero, one );
	stdvec_build_matrix< structures::Symmetric >( C_vec, n, n, n, zero, zero );

	mxm_stdvec_as_matrix( C_vec, n, A_vec, n, A_vec, n, n, n, n, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid() );

	diff_stdvec_matrix( C_vec, n, n, n, SC );

}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 6;

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
		std::cerr << "  -n (optional, default is 100): an even integer, the "
					 "test size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << " " << in << "\n";
	alp::Launcher< AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alp_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cerr << "Test FAILED (" << alp::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}

