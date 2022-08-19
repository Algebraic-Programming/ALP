
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

#ifndef _H_ALP_TEST_UTILS_ALP_MATVEC_UTILS_
#define _H_ALP_TEST_UTILS_ALP_MATVEC_UTILS_

#include <iostream>
#include <sstream>
#include <vector>
#include <type_traits>

#include <alp.hpp>

/**
 * @brief Print container in matrix format.
 * 
 * @tparam T 
 * @tparam Structure 
 * @param name 
 * @param A 
 */
template< typename T, typename Structure >
void print_alp_container( std::string name, const alp::Matrix< T, Structure, alp::Density::Dense > & A) {
    
    typedef typename std::remove_const< typename std::remove_reference< decltype( A ) >::type >::type A_type;

    static_assert( std::is_same< typename A_type::mapping_polynomial_type, alp::storage::polynomials::Full_type >::value, "print_alp_matrix_dense_full cannot print from non-full storage." );

    if( ! alp::internal::getInitialized( A ) ) {
        std::cout << "Matrix " << name << " uninitialized.\n";
        return;
    }
    
    std::cout << name << ":" << std::endl;
    for( size_t row = 0; row < alp::nrows( A ); ++row ) {
        std::cout << "[\t";
        for( size_t col = 0; col < alp::ncols( A ); ++col ) {
            auto pos  = alp::internal::getStorageIndex( A, row, col );
            std::cout << alp::internal::access(A, pos ) << "\t";
        }
        std::cout << "]" << std::endl;
    }
}

template< typename T, typename Structure >
void print_alp_container( std::string name, const alp::Vector< T, Structure, alp::Density::Dense > & v) {

    print_alp_container( name, static_cast< const typename alp::Vector< T, Structure >::base_type & >( v ) );
}

/**
 * @brief Check the infinity norm relative error.
 * 
 * @tparam MatType 
 * @tparam T 
 * @param vA 
 * @param m 
 * @param n 
 * @param lda 
 * @param mA 
 * @param tol 
 */
template< typename MatType, typename T >
void check_inf_norm_relerr( const std::vector< T > & vA, const size_t m, const size_t n, const size_t lda,
                        const MatType & mA, double tol=1e-7 ) {

    static_assert( std::is_convertible< T, double >::value, "check_inf_norm_relerr only applicable to double-castable T." );

    if( tol <= 0 ) {
        std::cout << "Inf Norm: Please provide a positive tolerance." << std::endl;
        return;
    } 

    constexpr const double eps = std::numeric_limits< double >::epsilon();

    double infn_va = 0;
    double infn_diff = 0;

    auto acc_row = [&vA, &lda, &mA, &infn_va, &infn_diff](size_t row, size_t lb, size_t ub) {

            double sum_abs_row_va = 0;
            double sum_abs_row_diff = 0;

            for( size_t col = lb; col < ub; ++col ) {
                double vaij = ( double )( vA[ row * lda + col ] );
                double maij = ( double )( alp::internal::access( mA, alp::internal::getStorageIndex( mA, row, col ) ) );
                sum_abs_row_va += std::abs( vaij );
                sum_abs_row_diff += std::abs( vaij - maij );
            }

            infn_va   = ( sum_abs_row_va > infn_va ) ? sum_abs_row_va : infn_va;
            infn_diff = ( sum_abs_row_diff > infn_diff ) ? sum_abs_row_diff : infn_diff;

    };

    if( std::is_same< typename MatType::structure, alp::structures::General >::value ) {
        for( size_t row = 0; row < m; ++row ) {
            acc_row(row, 0, n);
        }
    } else if( std::is_same< typename MatType::structure, alp::structures::Symmetric >::value 
                || std::is_same< typename MatType::structure, alp::structures::UpperTriangular >::value ) {

        for( size_t row = 0; row < m; ++row ) {
            acc_row(row, row, n);
        }
    } else {
        std::cout << "MatType not supported." << std::endl;
    }

    if( infn_diff >= tol * infn_va + eps ) {
        std::cout << "Inf Norm Error: " << infn_diff << " >= " << tol << " * " << infn_va << " + " << eps << " ( = " << tol * infn_va + eps << " )" << std::endl;
    }
}

template< typename VecType, typename T >
void check_inf_norm_relerr( const std::vector< T > & vA, const size_t m, const VecType & v, double threshold=1e-7 ) {
    check_inf_norm_relerr( vA, m, 1, 1, static_cast< const typename VecType::base_type & >( v ), threshold );
}

/**
 * @brief Print \a std::vector \a vA in matrix format.
 * 
 * @tparam T 
 * @param name 
 * @param vA 
 * @param m 
 * @param n 
 * @param lda 
 */
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

/**
 * @brief Build dense container within vector \a vA taking structure \a Structure
 *        and \a band_pos into account.
 * 
 * @tparam Structure 
 * @tparam T 
 * @tparam band_pos 
 * @param vA 
 * @param m 
 * @param n 
 * @param lda 
 * @param zero 
 * @param one 
 */
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

/**
 * @brief Build dense container within vector \a vA taking structure \a Structure
 *        into account.
 */
template< typename Structure = alp::structures::General, typename T >
void stdvec_build_matrix( std::vector< T > & vA, const size_t m, const size_t n, const size_t lda, const T zero, const T one ) {

	if( std::is_same< Structure, alp::structures::General >::value ) {
		std::fill( vA.begin(), vA.end(), one );
	} else if( std::is_same< Structure, alp::structures::Symmetric >::value ) {
		std::fill( vA.begin(), vA.end(), one );
	} else if( std::is_same< Structure, alp::structures::UpperTriangular >::value ) {
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


#endif // _H_ALP_TEST_UTILS_ALP_MATVEC_UTILS_
