
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

#ifdef DEBUG
#include <cblas.h>
#endif

enum structures { GE, UPSY };

#define PACKED_SIZE( n ) ( n * ( n + 1 ) / 2 )


template< typename T >
void stdvec_print_matrix( std::string name, const std::vector< T > & vA, const size_t m, const size_t n, const size_t lda ) {

	std::cout << "Vec " << name << ":" << std::endl;
	for( size_t row = 0; row < m; ++row ) {
		std::cout << "[\t";
		for( size_t col = 0; col < n; ++col ) {
			std::cout << vA[ row * lda + col ] << "\t";
		}
		std::cout << "]" << std::endl;
	}
}

template< typename T, structures Structure=GE >
void stdvec_build_matrix( std::vector< T > & vA, const size_t m, const size_t n, const size_t lda, const T zero, const T one, const T inc ) {

		T val = one;
		switch( Structure ) {
			case GE:
				for( size_t row = 0; row < m; ++row ) {
					for( size_t col = 0; col < n; ++col ) {
						vA[ row * lda + col ] = val;
						val += inc;
					}
				}
				break;
			case UPSY:
				for( size_t row = 0; row < m; ++row ) {
					for( size_t col = row; col < n; ++col ) {
						vA[ row * lda + col ] = vA[ col * lda + row ] = val;
						val += inc;
					}
				}
				break;
			default:
				std::cout << "stdvec_build_matrix: Unrecognized matrix structure." << std::endl;
		}

}

template< typename T >
void stdvec_build( std::vector< T > & vA, const T one, const T inc ) {

	T val = one;
	for( auto & elem: vA ) {
		elem = val;
		val += inc;
	}

}

template< typename T, char Structure=GE >
bool stdvec_diff_matrix( const std::vector< T > & vA, const size_t m, const size_t n, const size_t lda,
						 const std::vector< T > & vB, const size_t ldb, double threshold=1e-7 ) {

	bool ok = true;

	switch( Structure ) {
		case GE:
			for( size_t row = 0; row < m; ++row ) {
				for( size_t col = 0; col < n; ++col ) {
					double va = ( double )( vA[ row * lda + col ] );
					double vb = ( double )( vB[ row * ldb + col ] );
					double re = std::abs( ( va - vb ) / va );
					if( re > threshold ) {
						std::cout << "Error ( " << row << ", " << col << " ): " << va << " v " << vb << std::endl;
						ok = false;
					}
				}
			}
			break;
		case UPSY:
			for( size_t row = 0; row < m; ++row ) {
				for( size_t col = row; col < n; ++col ) {
					double va = ( double )( vA[ row * lda + col ] );
					double vb = ( double )( vB[ row * ldb + col ] );
					double re = std::abs( ( va - vb ) / va );
					if( re > threshold ) {
						std::cout << "Error ( " << row << ", " << col << " ): " << va << " v " << vb << std::endl; 
						ok = false;
					}
				}
			}
			break;
		default:
			std::cout << "stdvec_diff_matrix: Unrecognized matrix structure." << std::endl;
			ok = false;
		}

	return ok;
}

/**
 * BLAS 2 spr: Computes rank-1 update of a symmetric matrix \a A stored in upper 
 *             packed format: \f$ A = A + \alpha x x^T \f$.
 * 
 * @tparam T          Matrix and vector's entry type.
 * @param[in] n       Size parameter. 
 * @param[in] alpha   Scalar factor.
 * @param[in] x       Pointer to input vector of length \a n.
 * @param[in,out] ap  Pointer to input/output packed array of length 
 *                    <em> n * (n + 1)/2 </em>.
 */
template < typename T >
void spr_up(const size_t n, const T alpha, const T *x, T *ap) {

	T * tmp = new T[ PACKED_SIZE( n ) ]();

	// equiv. of alp::eWiseMul( tmp, alpha, outer( x ) ) 
	for( size_t i = 0; i < n; ++i ) {
		for( size_t j = i; j < n; ++j ) {
			tmp[ (2 * n - i - 1) * i / 2 + j ] += alpha * ( x[ i ] * x[ j ] );
		}
	}

	// equiv. of alp::foldl( A, tmp ) 
	for( size_t i = 0; i < n; ++i ) {
		for( size_t j = i; j < n; ++j ) {
			ap[ (2 * n - i - 1) * i / 2 + j ] += tmp[ (2 * n - i - 1) * i / 2 + j ];
		}
	}

}

/**
 * BLAS 2 spr2: Computes rank-1 update of a symmetric matrix \a A stored in upper 
 *             packed format: \f$ A = A + \alpha x y^T + \alpha y x^T  \f$.
 * 
 * @tparam T          Matrix and vector entry type.
 * @param[in] n       Size parameter. 
 * @param[in] alpha   Scalar factor.
 * @param[in] x       First input vector of length \a n.
 * @param[in] y       Second input vector of length \a n.
 * @param[in,out] ap  Input/output matrix in packed array of length 
 *                    <em> n * (n + 1)/2 </em>.
 */
template < typename T >
void spr2_up(const size_t n, const T alpha, const T *x, const T *y, T *ap) {

	T * tmp = new T[ PACKED_SIZE( n ) ]();

	// equiv. of alp::eWiseAdd( tmp, outer( x, y ), outer( y, x ) ) 
	for( size_t i = 0; i < n; ++i ) {
		for( size_t j = i; j < n; ++j ) {
			tmp[ (2 * n - i - 1) * i / 2 + j ] +=  ( x[ i ] * y[ j ] ) + ( y[ i ] * x[ j ] );
		}
	}

	// equiv. of alp::eWiseMul( A, alpha, tmp ) 
	for( size_t i = 0; i < n; ++i ) {
		for( size_t j = i; j < n; ++j ) {
			ap[ (2 * n - i - 1) * i / 2 + j ] += alpha * tmp[ (2 * n - i - 1) * i / 2 + j ];
		}
	}

}

/**
 * BLAS 3 syrk: Computes rank-k update (not transposed) of a symmetric matrix 
 *              \a A stored in upper full format with a fixed negative \a alpha: 
 *              \f$ C = C - A A^T \f$.
 * 
 * @tparam T          Matrix and vector entry type.
 * @param[in] n       Size parameter. 
 * @param[in] k       Size parameter. Indicates the rank of the matrix update. 
 * @param[in] A       Input matrix of size <em> n * k </em>.
 * @param[in,out] C   Input/output matrix of size <em> n * n </em>. Only upper
 *                    part can be accessed.
 */
template < typename T >
void syrk_up_ntrans_negscal(const size_t n, const size_t k, const T *A, T *C) {

	T * tmp = new T[ n * n ]();

	// equiv. of alp::mxm( tmp, A, A^T ) 
	for( size_t i = 0; i < n; ++i ) {
		for( size_t j = i; j < n; ++j ) {
			for( size_t h = 0; h < k; ++h ) {
				tmp[ i * n + j ] +=  A[ i * k + h ] * A[ j * k + h ];
			}
		}
	}

	// equiv. of alp::foldl( C, tmp, Minus ) 
	for( size_t i = 0; i < n; ++i ) {
		for( size_t j = i; j < n; ++j ) {
			C[ i * n + j ] -= tmp[ i * n + j ];
		}
	}

}

/**
 * BLAS 3 syrk: Computes rank-k update (not transposed) of a symmetric matrix 
 *              \a A stored in upper full format: 
 *              \f$ C = \beta C + \alpha A A^T \f$.
 * 
 * @tparam T          Matrix and vector entry type.
 * @param[in] n       Size parameter. 
 * @param[in] k       Size parameter. Indicates the rank of the matrix update. 
 * @param[in] alpha   Scalar factor.
 * @param[in] A       Input matrix of size <em> n * k </em>.
 * @param[in] lda     Leading dimension of matrix \a A. 
 * @param[in] beta    Scalar factor.
 * @param[in,out] C   Input/output matrix of size <em> n * n </em>. Only upper
 *                    part can be accessed.
 * @param[in] ldc     Leading dimension of matrix \a C. 
 */
template < typename T >
void syrk_up_ntrans(const size_t n, const size_t k, const T alpha, const T *A, const size_t lda, const T beta, T *C, const size_t ldc) {

	T * tmp = new T[ n * n ]();

	// equiv. of alp::foldr( beta, C, Times ) 
	for( size_t i = 0; i < n; ++i ) {
		for( size_t j = i; j < n; ++j ) {
			C[ i * ldc + j ] *= beta;
		}
	}

	// equiv. of alp::mxm( tmp, A, A^T ) 
	for( size_t i = 0; i < n; ++i ) {
		for( size_t j = i; j < n; ++j ) {
			for( size_t h = 0; h < k; ++h ) {
				tmp[ i * n + j ] +=  A[ i * lda + h ] * A[ j * lda + h ];
			}
		}
	}

	// equiv. of alp::eWiseMul( C, alpha, tmp ) 
	for( size_t i = 0; i < n; ++i ) {
		for( size_t j = i; j < n; ++j ) {
			C[ i * ldc + j ] += alpha * tmp[ i * n + j ];
		}
	}

}

#define FLOAT double

int main( int argc, char ** argv ) {

	const size_t n = 6;
	const size_t k = 3;
	const size_t ld = 2 * n;

	const FLOAT alpha = 2.;
	const FLOAT beta = 2.;

	bool test = true;

	std::vector< FLOAT > 
		x( n ), y( n ), ap( n * ( n + 1 ) / 2 ), 
		A( n * k ), A_wld( n * ld ), C( n * n ), C_wld( n * ld );

#ifdef DEBUG
	std::vector< FLOAT > 
		ap_test( n * ( n + 1 ) / 2 ), C_test( n * n ), C_wld_test( n * ld );
#endif

	std::cout << "\nTest SPR...";

	stdvec_build( x, 1., 1. );
	stdvec_build( ap, 1., 1. );

#ifdef PRINT_VECS
	std::cout << "\nalpha: " << alpha << std::endl;
	stdvec_print_matrix( "x", x, 1, n, n );
	stdvec_print_matrix( "PRE ap", ap, 1, PACKED_SIZE( n ), PACKED_SIZE( n ) );
#endif

	spr_up( n, alpha, &( x[0] ), &( ap[0] ) );

#ifdef PRINT_VECS
	stdvec_print_matrix( "POST ap", ap, 1, PACKED_SIZE( n ), PACKED_SIZE( n ) );
#endif 

#ifdef DEBUG
	stdvec_build( ap_test, 1., 1. );

#ifdef PRINT_VECS
	stdvec_print_matrix( "PRE ap_test", ap_test, 1, PACKED_SIZE( n ), PACKED_SIZE( n ) );
#endif 

	cblas_dspr( CblasRowMajor, CblasUpper, n, alpha, &( x[0] ), 1, &( ap_test[0] ) );

#ifdef PRINT_VECS
	stdvec_print_matrix( "POST ap_test", ap_test, 1, PACKED_SIZE( n ), PACKED_SIZE( n ) );
#endif 

	test = stdvec_diff_matrix( ap, 1, ap.size(), ap.size(), ap_test, ap_test.size() );
#endif
	std::cout << ( test ? "OK." : "KO." ) << std::endl;


	std::cout << "\nTest SPR2...";

	stdvec_build( y, 1., 1. );
	stdvec_build( ap, 1., 1. );

#ifdef PRINT_VECS
	std::cout << "\nalpha: " << alpha << std::endl;
	stdvec_print_matrix( "x", x, 1, n, n );
	stdvec_print_matrix( "y", y, 1, n, n );
	stdvec_print_matrix( "PRE ap", ap, 1, PACKED_SIZE( n ), PACKED_SIZE( n ) );
#endif

	spr2_up( n, alpha, &( x[0] ), &( y[0] ), &( ap[0] ));

#ifdef PRINT_VECS
	stdvec_print_matrix( "POST ap", ap, 1, PACKED_SIZE( n ), PACKED_SIZE( n ) );
#endif 

#ifdef DEBUG
	stdvec_build( ap_test, 1., 1. );

#ifdef PRINT_VECS
	stdvec_print_matrix( "PRE ap_test", ap_test, 1, PACKED_SIZE( n ), PACKED_SIZE( n ) );
#endif 

	cblas_dspr2( CblasRowMajor, CblasUpper, n, alpha, &( x[0] ), 1, &( y[0] ), 1, &( ap_test[0] ) );

#ifdef PRINT_VECS
	stdvec_print_matrix( "POST ap_test", ap_test, 1, PACKED_SIZE( n ), PACKED_SIZE( n ) );
#endif 

	test = stdvec_diff_matrix( ap, 1, ap.size(), ap.size(), ap_test, ap_test.size() );
#endif
	std::cout << ( test ? "OK." : "KO." ) << std::endl;

	std::cout << "\nTest SYRK (downdate)...";

	stdvec_build_matrix( A, n, k, k, 0., 1., 1. );
	stdvec_build_matrix< FLOAT, UPSY >( C, n, n, n, 0., 1., 1. );

#ifdef PRINT_VECS
	std::cout << "\n";
	stdvec_print_matrix( "A", A, n, k, k );
	stdvec_print_matrix( "PRE C", C, n, n, n );
#endif

	syrk_up_ntrans_negscal( n, k, &( A[0] ), &( C[0] ) );

#ifdef PRINT_VECS
	stdvec_print_matrix( "POST C", C, n, n, n );
#endif

#ifdef DEBUG
	stdvec_build_matrix< FLOAT, UPSY >( C_test, n, n, n, 0., 1., 1. );

#ifdef PRINT_VECS
	stdvec_print_matrix( "PRE C_test", C_test, n, n, n );
#endif

	cblas_dsyrk( CblasRowMajor, CblasUpper, CblasNoTrans, n, k, -1., &( A[0] ), k, 1, &( C_test[0] ), n );

#ifdef PRINT_VECS
	stdvec_print_matrix( "POST C_test", C_test, n, n, n );
#endif

	test = stdvec_diff_matrix< FLOAT, UPSY >( C, n, n, n, C_test, n );
#endif
	std::cout << ( test ? "OK." : "KO." ) << std::endl;

	std::cout << "\nTest SYRK...";

	stdvec_build_matrix( A_wld, n, k, ld, 0., 1., 1. );
	stdvec_build_matrix< FLOAT, UPSY >( C_wld, n, n, ld, 0., 1., 1. );

#ifdef PRINT_VECS
	std::cout << "\nalpha: " << alpha << std::endl;
	std::cout << "beta: " << beta << std::endl;
	stdvec_print_matrix( "A_wld", A_wld, n, k, ld );
	stdvec_print_matrix( "PRE C_wld", C_wld, n, n, ld );
#endif

	syrk_up_ntrans( n, k, alpha, &( A_wld[0] ), ld, beta, &( C_wld[0] ), ld );

#ifdef PRINT_VECS
	stdvec_print_matrix( "POST C_wld", C_wld, n, n, ld );
#endif

#ifdef DEBUG
	stdvec_build_matrix< FLOAT, UPSY >( C_wld_test, n, n, ld, 0., 1., 1. );

#ifdef PRINT_VECS
	stdvec_print_matrix( "PRE C_wld_test", C_wld_test, n, n, ld );
#endif

	cblas_dsyrk( CblasRowMajor, CblasUpper, CblasNoTrans, n, k, alpha, &( A_wld[0] ), ld, beta, &( C_wld_test[0] ), ld );

#ifdef PRINT_VECS
	stdvec_print_matrix( "POST C_wld_test", C_wld_test, n, n, ld );
#endif

	test = stdvec_diff_matrix< FLOAT, UPSY >( C_wld, n, n, ld, C_wld_test, ld );
#endif
	std::cout << ( test ? "OK." : "KO." ) << std::endl;

	return 0;
}

