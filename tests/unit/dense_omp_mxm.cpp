
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
#ifdef NDEBUG
 #include "../utils/print_alp_containers.hpp"
#endif

using namespace alp;


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

template< typename T, typename Operator, typename Monoid >
void mxm_stdvec_as_matrix(	
	std::vector< T > &vC, const size_t ldc,
	const std::vector< T > &vA, const size_t lda,
	const std::vector< T > &vB, const size_t ldb,
	const size_t m, const size_t k, const size_t n,
	const Operator oper,
	const Monoid monoid 
) {
    
	T temp;

#ifndef NDEBUG
	print_stdvec_as_matrix( "vA", vA, n, n, n );
	print_stdvec_as_matrix( "vB", vB, n, n, n );
	print_stdvec_as_matrix( "vC - PRE", vC, n, n, n );
#endif

	for( size_t i = 0; i < m; ++i ) {
		for( size_t j = 0; j < n; ++j ) {
			T &c_val { vC[ i * ldc + j ] };
			for( size_t l = 0; l < k; ++l ) {
					const T &a_val { vA[ i * lda + l ] };
					const T &b_val { vB[ l * ldb + j ] };
					// std::cout << c_val << " += " << a_val << " * " << b_val << std::endl;
					(void)internal::apply( temp, a_val, b_val, oper );
					// std::cout << "temp = " << temp << std::endl;
					(void)internal::foldl( c_val, temp, monoid.getOperator() );
			}
		}
	}

#ifndef NDEBUG
	print_stdvec_as_matrix( "vC - POST", vC, n, n, n );
#endif
}

template< typename Structure, typename T >
void stdvec_build_matrix( std::vector< T > &vA, const size_t m, const size_t n, const size_t lda, const T zero, const T one ) {

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

template< typename Structure, typename T >
void stdvec_build_matrix( std::vector< T > &vA, const size_t m, const size_t n, const size_t lda, const T zero, const T one, const T inc ) {

		T val = one;
		if( std::is_same< Structure, structures::General >::value ) {
			for( size_t row = 0; row < m; ++row ) {
				for( size_t col = 0; col < n; ++col ) {
					vA[ row * lda + col ] = val;
					val += inc;
				}
			}
		} else if( std::is_same< Structure, structures::Symmetric >::value ) {
			for( size_t row = 0; row < m; ++row ) {
				for( size_t col = row; col < n; ++col ) {
					vA[ row * lda + col ] = vA[ col * lda + row ] = val;
					val += inc;
				}
			}
		} else if( std::is_same< Structure, structures::UpperTriangular >::value ) {
			for( size_t row = 0; row < m; ++row ) {
				for( size_t col = 0; col < row; ++col ) {
					vA[ row * lda + col ] = zero;
				}
				for( size_t col = row; col < n; ++col ) {
					vA[ row * lda + col ] = val;
					val += inc;
				}
			}
		}

}

template< typename Structure, typename T >
void stdvec_build_matrix_packed( std::vector< T > &vA, const T one ) {

	std::fill( vA.begin(), vA.end(), one );

}

template< typename Structure, typename T >
void stdvec_build_matrix_packed( std::vector< T > &vA, const T one, const T inc ) {

		T val = one;
		if( std::is_same< Structure, structures::Symmetric >::value ) { // Assumes Packed Row - Upper
			for( auto &elem: vA ) {
				elem = val;
				val += inc;
			}
		} else if( std::is_same< Structure, structures::UpperTriangular >::value ) { // Assumes Packed Row - Upper
			for( auto &elem: vA ) {
				elem = val;
				val += inc;
			}
		}

}

template< typename MatType, typename T >
void diff_stdvec_matrix( const std::vector< T > &vA, const size_t m, const size_t n, const size_t lda,
						 const MatType & mA, double threshold = 1e-7 ) {

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

template < typename T, typename SemiringT >
void run_mxm( const size_t m, const size_t k, const size_t n, alp::RC &rc ) {

	const SemiringT ring;
	const T one  = ring.template getOne< T >();
	const T zero = ring.template getZero< T >();

	std::vector< T > A_data( m * k );
	std::vector< T > B_data( k * n );
	std::vector< T > C_data( m * n, zero );

	std::vector< T > A_vec( m * k );
	std::vector< T > B_vec( k * n );
	std::vector< T > C_vec( m * n, zero );

	std::cout << "\tTesting dense General mxm " << m << " " << k << " " << n << std::endl;

	stdvec_build_matrix< structures::General >( A_data, m, k, k, zero, one, one );
	stdvec_build_matrix< structures::General >( B_data, k, n, n, zero, one, one );

	// initialize test
	alp::Matrix< T, structures::General > A( m, k );
	alp::Matrix< T, structures::General > B( k, n );
	alp::Matrix< T, structures::General > C( m, n );

	// Initialize input matrices
	rc = rc ? rc : alp::buildMatrix( A, A_data.begin(), A_data.end() );
	if ( rc != alp::SUCCESS ) {
		std::cerr << "\tIssues building A" << std::endl;
		return;
	}
	rc = rc ? rc : alp::buildMatrix( B, B_data.begin(), B_data.end() );
	rc = rc ? rc : alp::buildMatrix( C, C_data.begin(), C_data.end() );

	if ( rc != alp::SUCCESS ) {
		std::cerr << "\tIssues building matrices" << std::endl;
		return;
	}

#ifndef NDEBUG
	print_matrix( "A", A );
	print_matrix( "B", B );
	print_matrix( "C - PRE", C );
#endif

	rc = rc ? rc : alp::mxm( C, A, B, ring );

#ifndef NDEBUG
	print_matrix( "C - POST", C );
#endif

	if ( rc != alp::SUCCESS )
		return;

	stdvec_build_matrix< structures::General >( A_vec, m, k, k, zero, one, one );
	stdvec_build_matrix< structures::General >( B_vec, k, n, n, zero, one, one );

	mxm_stdvec_as_matrix( C_vec, n, A_vec, k, B_vec, n, m, k, n, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid() );

	diff_stdvec_matrix( C_vec, m, n, n, C );


	std::cout << "\tDone." << std::endl;

}

#define M ( alp::config::BLOCK_ROW_DIM * n )
#define K ( alp::config::BLOCK_COL_DIM * 2 * n )
#define N ( alp::config::BLOCK_COL_DIM * 3 * n )

void alp_program( const size_t &n, alp::RC &rc ) {

	using T = double;

	using SemiringT = alp::Semiring< 
		alp::operators::add< T >, alp::operators::mul< T >, 
		alp::identities::zero, alp::identities::one 
	>;

	rc = alp::SUCCESS;

	/** 
	 * Testing cubic mxm.
	 */
	run_mxm< T, SemiringT >( M, M, M, rc );

	/**
	 * Testing rectangular mxm
	 */
	run_mxm< T, SemiringT >( M, K, N, rc );

	/**
	 * Testing outer-prod of blocks mxm
	 */
	run_mxm< T, SemiringT >( M, alp::config::BLOCK_COL_DIM, N, rc );

	/**
	 * Testing dot-prod of blocks mxm
	 */
	run_mxm< T, SemiringT >( alp::config::BLOCK_ROW_DIM, M, alp::config::BLOCK_COL_DIM, rc );

}

int main( int argc, char **argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 4;

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

