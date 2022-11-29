
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

#ifndef _H_TEST_UTILS_PRINT_ALP_CONTAINERS
#define _H_TEST_UTILS_PRINT_ALP_CONTAINERS

#include <iomanip>
#include <cstdlib>
#include <climits>

#include <alp.hpp>

template<
	typename MatrixType,
	std::enable_if_t< alp::is_matrix< MatrixType >::value > * = nullptr
>
void print_matrix( std::string name, const MatrixType &A ) {

	if( ! alp::internal::getInitialized( A ) ) {
		std::cout << "Matrix " << name << " uninitialized. Nothing to print.\n";
		return;
	}

	constexpr bool is_sym { alp::structures::is_a< typename alp::inspect_structure< MatrixType >::type, alp::structures::Symmetric >::value };
	// Temporary until adding multiple symmetry directions
	constexpr bool sym_up { is_sym };

	std::cout << name << "= [\n";
	for( size_t row = 0; row < alp::nrows( A ); ++row ) {
		std::cout << " [";
		for( size_t col = 0; col < alp::ncols( A ); ++col ) {
			if( alp::is_non_zero< typename alp::inspect_structure< MatrixType >::type >( row, col ) ) {
				const auto k = ( !is_sym || ( is_sym && ( sym_up == ( row < col ) ) ) ) ?
					alp::internal::getStorageIndex( A, row, col ) :
					alp::internal::getStorageIndex( A, col, row );
				auto val = alp::internal::access( A, k );
				val = std::abs(val) < 1.e-10 ? 0 : val;
				std::cout << std::setprecision( 10 ) << "\t" << val  ;
				if( col + 1 != alp::ncols( A ) ) {
					std::cout <<  ",";
				}
			} else {
				std::cout << std::setprecision( 0 ) << "\t" << "0, ";
			}
		}
		if( row + 1 != alp::nrows( A ) ) {
			std::cout << "\t" << "]," << "\n";
		} else {
			std::cout << "\t" << "]" << "\n";
		}

	}
	std::cout << "]\n";
}

template<
	typename VectorType,
	std::enable_if_t< alp::is_vector< VectorType >::value > * = nullptr
>
void print_vector( std::string name, const VectorType &v ) {

	if( ! alp::internal::getInitialized( v ) ) {
		std::cout << "Vector " << name << " uninitialized. Nothing to print.\n";
		return;
	}

	std::cout << name << ":" << std::endl;
	std::cout << "[";
	for( size_t i = 0; i < alp::nrows( v ); ++i ) {
		std::cout << std::setprecision( 3 ) << "\t" << v[ i ];
	}
	std::cout << "\t" << "]" << "\n";
}

#endif // _H_TEST_UTILS_PRINT_ALP_CONTAINERS
