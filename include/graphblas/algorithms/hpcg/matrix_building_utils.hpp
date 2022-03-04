
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

/**
 * @file hpcg_matrix_building_utils.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @brief Utilities to build the matrices for HPCG simulations in an arbitrary number of dimensions.
 * @date 2021-04-30
 */

#ifndef _H_GRB_ALGORITHMS_MATRIX_BUILDING_UTILS
#define _H_GRB_ALGORITHMS_MATRIX_BUILDING_UTILS

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <graphblas.hpp>

#include "ndim_matrix_builders.hpp"


namespace grb {
	namespace algorithms {

		/**
		 * @brief Builds a \p DIMS -dimensional system matrix for HPCG simulation.
		 *
		 * This routine initializes \p M to a matrix representing a \p DIMS -dimensions system of sizes
		 * \p sys_sizes, with an iteration halo of size \p halo_size . The matrix diagonal values are initialized
		 * to \p diag_value while the other non-zero values are initialized to \p non_diag_value .
		 *
		 * @tparam DIMS system dimensions
		 * @tparam T type of matrix values
		 * @tparam B matrix GraphBLAS backend
		 * @param M the matrix to be initialized; it must be already constructed
		 * @param sys_sizes the sizes of the physical system
		 * @param halo_size the size of the halo of point to iterate in
		 * @param diag_value diagonal value
		 * @param non_diag_value value outside of the diagonal
		 * @return grb::RC the success value returned when trying to build the matrix
		 */
		template< std::size_t DIMS, typename T, enum grb::Backend B >
		grb::RC build_ndims_system_matrix( grb::Matrix< T, B > & M, const std::array< std::size_t, DIMS > & sys_sizes, std::size_t halo_size, T diag_value, T non_diag_value ) {
			static_assert( DIMS > 0, "DIMS must be > 0" );
			std::size_t n { std::accumulate( sys_sizes.cbegin(), sys_sizes.cend(), 1UL, std::multiplies< std::size_t >() ) };
			if( grb::nrows( M ) != n || grb::nrows( M ) != grb::ncols( M ) ) {
				throw std::invalid_argument( "wrong matrix dimensions: matrix should "
											"be square"
											" and in accordance with given system "
											"sizes" );
			}
			grb::algorithms::matrix_generator_iterator< DIMS, T > begin( sys_sizes, 0UL, halo_size, diag_value, non_diag_value );
			grb::algorithms::matrix_generator_iterator< DIMS, T > end( sys_sizes, n, halo_size, diag_value, non_diag_value );
			return buildMatrixUnique( M, begin, end, grb::IOMode::SEQUENTIAL );
		}

		/**
		 * @brief Builds a coarsener matrix for an HPCG simulation.
		 *
		 * It initializes \p M as a rectangular matrix, with rows corresponding to the coarser system
		 * (of dimensions \p coarser_sizes - output) and columns corresponding to the finer system
		 * (of dimensions \p finer_sizes - input). The resulting coarsening matrix takes in input the finer system
		 * and coarsens it by keeping one element every \a S , where \a S is the ratio between the finer and
		 * the coarser dimension (computed for each dimension). In this way each \p DIMS -dimensional finer element
		 * corresponds to its bounding coarser element.
		 *
		 * For the coarsening to be feasible, the sizes of the finer system \b must be a multiple of those of the
		 * coarser system. If this condition is not met, an exception is thrown.
		 *
		 * @tparam DIMS system dimensions
		 * @tparam T type of matrix values
		 * @tparam B matrix GraphBLAS backend
		 * @param M the matrix to be initialized; it must be already constructed with proper dimensions
		 * @param coarser_sizes sizes of the coarser system
		 * @param finer_sizes sizes of the finer system; each one \b must be a multiple of the corresponding value
		 * 			in \p coarser_size , otherwise an exception is thrown
		 * @return grb::RC the success value returned when trying to build the matrix
		 */
		template< std::size_t DIMS, typename T, enum grb::Backend B >
		grb::RC build_ndims_coarsener_matrix( grb::Matrix< T, B > & M, const std::array< std::size_t, DIMS > & coarser_sizes, const std::array< std::size_t, DIMS > & finer_sizes ) {
			static_assert( DIMS > 0, "DIMS must be > 0" );
			std::size_t const rows { std::accumulate( coarser_sizes.cbegin(), coarser_sizes.cend(), 1UL, std::multiplies< std::size_t >() ) };
			for( std::size_t i { 0 }; i < coarser_sizes.size(); i++ ) {
				std::size_t step = finer_sizes[ i ] / coarser_sizes[ i ];
				if( step * coarser_sizes[ i ] != finer_sizes[ i ] ) {
					throw std::invalid_argument( "finer sizes should be a multiple of "
												"coarser sizes" );
				}
			}
			std::size_t const cols { std::accumulate( finer_sizes.cbegin(), finer_sizes.cend(), 1UL, std::multiplies< std::size_t >() ) };
			if( grb::nrows( M ) != rows || grb::ncols( M ) != cols ) {
				throw std::invalid_argument( "wrong matrix dimensions: matrix should "
											"be rectangular"
											" with rows == <product of coarser sizes> "
											"and cols == <product of finer sizes>" );
			}

			grb::algorithms::coarsener_generator_iterator< DIMS, T > begin( coarser_sizes, finer_sizes, 0 );
			grb::algorithms::coarsener_generator_iterator< DIMS, T > end( coarser_sizes, finer_sizes, rows );
			return buildMatrixUnique( M, begin, end, grb::IOMode::SEQUENTIAL );
		}

		/**
		 * @brief Populates \p masks with static color mask generated for a squared matrix of size \p matrix_size .
		 *
		 * Colors are built in the range [0, \p colors ), with the mask for color 0 being the array
		 * of values true in the positions \f$ [0, colors, 2*colors, ..., floor((system_size - 1)/colors) * color] \f$,
		 * for color 1 in the positions \f$ [1, 1+colors, 1+2*colors, ..., floor((system_size - 2)/colors) * color] \f$,
		 * etc.; the mask for color 0 is in \c masks[0], for color 1 in \c masks[1] and so on.
		 *
		 * The vectors stored in \p masks (assumed empty at the beginning) are built inside the function and populated
		 * only with the \c true values, leading to sparse vectors. This saves on storage space and allows
		 * GraphBLAS routines (like \c eWiseLambda() ) to iterate only on true values.
		 *
		 * @tparam B GraphBLAS backend for the vector
		 * @param masks output vector of color masks
		 * @param matrix_size size of the system matrix
		 * @param colors numbers of colors masks to build; it must be < \p matrix_size
		 * @return grb::RC the success value returned when trying to build the vector
		 */
		template< enum grb::Backend B >
		grb::RC build_static_color_masks( std::vector< grb::Vector< bool, B > > & masks, std::size_t matrix_size, std::size_t colors ) {
			if( ! masks.empty() ) {
				throw std::invalid_argument( "vector of masks is expected to be "
											"empty" );
			}
			if( matrix_size < colors ) {
				throw std::invalid_argument( "syztem size is < number of colors: too "
											"small" );
			}
			grb::RC rc { grb::SUCCESS };
			masks.reserve( colors );
			for( std::size_t i { 0U }; i < colors; i++ ) {
				// build in-place, assuming the compiler deduces the right constructor according to B
				masks.emplace_back( matrix_size );
				grb::Vector< bool > & mask = masks.back();
				// grb::set(mask, false); // DO NOT initialize false's explicitly, otherwise
				// RBGS will touch them too and the runtime will increase!
				for( std::size_t j = i; j < matrix_size; j += colors ) {
					rc = grb::setElement( mask, true, j );
					assert( rc == grb::SUCCESS );
					if( rc != grb::SUCCESS )
						return rc;
				}
			}
			return rc;
		}

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MATRIX_BUILDING_UTILS
