
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
#include <type_traits>

#include <alp.hpp>

using namespace alp;

template< typename T, typename Structure >
void print_matrix( std::string name, const alp::Matrix< T, Structure > & A) {

	if( ! alp::internal::getInitialized( A ) ) {
		std::cout << "Matrix " << name << " uninitialized.\n";
		return;
	}
	
	std::cout << name << ":" << std::endl;
	for( size_t row = 0; row < alp::nrows( A ); ++row ) {
		std::cout << "[\t";
		for( size_t col = 0; col < alp::ncols( A ); ++col ) {
			auto pos  = internal::getStorageIndex( A, row, col );
			// std::cout << "(" << pos << "): ";
			std::cout << internal::access(A, pos ) << "\t";
		}
		std::cout << "]" << std::endl;
	}
}

template< typename T, typename Structure >
void print_vector( std::string name, const alp::Vector< T, Structure > & v) {

	print_matrix( name, static_cast< const typename alp::Vector< T, Structure >::base_type & >( v ) );
}

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

	// print_stdvec_as_matrix("vA", vA, m, k, lda);
	// print_stdvec_as_matrix("vB", vB, k, n, ldb);
	// print_stdvec_as_matrix("vC - PRE", vC, m, n, ldc);

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

	// print_stdvec_as_matrix("vC - POST", vC, m, n, ldc);

}

template< 
	typename Structure, 
	typename T, size_t band_pos,
	std::enable_if_t<
		band_pos == std::tuple_size< typename Structure::band_intervals >::value
	> * = nullptr
> void stdvec_build_matrix_band( std::vector< T > & vA, const size_t m, const size_t n, const size_t lda, const T zero, const T one ) {
	(void)vA;
	(void)m;
	(void)n;
	(void)lda;
	(void)zero;
	(void)one;
}

template< 
	typename Structure, 
	typename T, 
	size_t band_pos,
	std::enable_if_t<
		band_pos < std::tuple_size< typename Structure::band_intervals >::value
	> * = nullptr
> void stdvec_build_matrix_band( std::vector< T > & vA, const std::ptrdiff_t m, const std::ptrdiff_t n, const size_t lda, const T zero, const T one ) {

	constexpr std::ptrdiff_t cl_a = std::tuple_element< band_pos, typename Structure::band_intervals >::type::left;
	constexpr std::ptrdiff_t cu_a = std::tuple_element< band_pos, typename Structure::band_intervals >::type::right;

	const std::ptrdiff_t l_a = ( cl_a < -m + 1 ) ? -m + 1 : cl_a ;
	const std::ptrdiff_t u_a = ( cu_a > n ) ? n : cu_a ;

	for( std::ptrdiff_t b_idx = l_a; b_idx < u_a; ++b_idx  ) {
		std::ptrdiff_t row = ( b_idx < 0 ) ? -b_idx : 0;
		std::ptrdiff_t col = ( b_idx < 0 ) ? 0 : b_idx;
		for(  ; row < m && col < n; ++row, ++col ) {
			vA[ row * lda + col ] = one;
		}
	}

	stdvec_build_matrix_band< Structure, T, band_pos + 1 >( vA, m, n, lda, zero, one);

}

template< typename Structure = structures::General, typename T >
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
	} else { // Treat as Band Matrix
		std::fill( vA.begin(), vA.end(), zero );
		stdvec_build_matrix_band< Structure, T, 0 >( vA, m, n, lda, zero, one);
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

template< typename VecType, typename T >
void diff_stdvec_vector( const std::vector< T > & vA, const size_t m, const VecType & v, double threshold=1e-7 ) {
	diff_stdvec_matrix( vA, m, 1, 1, static_cast< const typename VecType::base_type & >( v ), threshold );
}

void alp_program( const size_t & n, alp::RC & rc ) {

	typedef double T;

	alp::Semiring< alp::operators::add< T >, alp::operators::mul< T >, alp::identities::zero, alp::identities::one > ring;

	T one  = ring.getOne< T >();
	T zero = ring.getZero< T >();

	std::vector< T > A_data( n * n, one );
	std::vector< T > v_data( n, one );
	std::vector< T > u_data( n, zero );

	std::cout << "\tTesting dense General mxm " << n << std::endl;
	// initialize test
	alp::Matrix< T, structures::General > A( n, n );
	alp::Vector< T > v( n );
	alp::Vector< T > u( n );

	// Initialize input matrices
	rc = alp::buildMatrix( A, A_data.begin(), A_data.end() );
	rc = alp::buildVector( v, v_data.begin(), v_data.end() );
	rc = alp::buildVector( u, u_data.begin(), u_data.end() );

	print_matrix("A", A);
	print_vector("v", v);
	print_vector("u - PRE", u);

	rc = alp::mxv( u, A, v, ring );

	print_vector("u - POST", u);

	std::vector< T > A_vec( n * n, one );
	std::vector< T > v_vec( n, one );
	std::vector< T > u_vec( n, zero );

	mxm_stdvec_as_matrix( u_vec, 1, A_vec, n, v_vec, 1, n, n, 1, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid() );

	diff_stdvec_vector( u_vec, n, u );

	std::cout << "\n\n=========== Testing Uppertriangular ============\n\n";

	alp::Matrix< T, structures::UpperTriangular > UA( n );

	rc = alp::buildMatrix( UA, A_data.begin(), A_data.end() );
	rc = alp::buildVector( u, u_data.begin(), u_data.end() );

	print_vector("u - PRE", u);
	rc = alp::mxv( u, UA, v, ring );
	print_vector("u - POST", u);

	stdvec_build_matrix< structures::UpperTriangular >( A_vec, n, n, n, zero, one );
	stdvec_build_matrix( u_vec, n, 1, 1, zero, zero );

	mxm_stdvec_as_matrix( u_vec, 1, A_vec, n, v_vec, 1, n, n, 1, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid() );

	diff_stdvec_vector( u_vec, n, u );

	std::cout << "\n\n=========== Testing Symmetric ============\n\n";

	alp::Matrix< T, structures::Symmetric > SA( n );

	rc = alp::buildMatrix( SA, A_data.begin(), A_data.end() );
	rc = alp::buildVector( u, u_data.begin(), u_data.end() );

	print_vector("u - PRE", u);
	rc = alp::mxv( u, SA, v, ring );
	print_vector("u - POST", u);

	stdvec_build_matrix< structures::Symmetric >( A_vec, n, n, n, zero, one );
	stdvec_build_matrix( u_vec, n, 1, 1, zero, zero );

	mxm_stdvec_as_matrix( u_vec, 1, A_vec, n, v_vec, 1, n, n, 1, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid() );

	diff_stdvec_vector( u_vec, n, u );

	std::cout << "\n\n=========== Testing Band ============\n\n";

	typedef alp::structures::Band< alp::Interval<-2>, alp::Interval<1>, alp::Interval<3> > BandT;
	alp::Matrix< T, BandT > BA( n, n );

	rc = alp::buildMatrix( BA, A_data.begin(), A_data.end() );
	rc = alp::buildVector( u, u_data.begin(), u_data.end() );

	print_vector("u - PRE", u);
	rc = alp::mxv( u, BA, v, ring );
	print_vector("u - POST", u);

	stdvec_build_matrix< BandT >( A_vec, n, n, n, zero, one );
	stdvec_build_matrix( u_vec, n, 1, 1, zero, zero );

	mxm_stdvec_as_matrix( u_vec, 1, A_vec, n, v_vec, 1, n, n, 1, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid() );

	diff_stdvec_vector( u_vec, n, u );

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

