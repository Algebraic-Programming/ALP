
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
#include <limits.h>
#include <iterator>
#include <type_traits>

#include <graphblas.hpp>

#include "ndim_matrix_builders.hpp"


namespace grb {
	namespace algorithms {

		template< typename T > void partition_nonzeroes(
				T num_nonzeroes,
				T& first_offset,
				T& last_offset
		) {
			const size_t num_procs{ spmd<>::nprocs() };
			const T per_process{ ( num_nonzeroes + num_procs - 1 ) / num_procs }; // round up
			first_offset = std::min( per_process * static_cast< T >( spmd<>::pid() ), num_nonzeroes );
			last_offset = std::min( first_offset + per_process, num_nonzeroes );
		}

		template< typename IterT > void partition_iteration_range(
			size_t num_nonzeroes,
			IterT &begin,
			IterT &end
		) {
			static_assert( std::is_base_of< std::random_access_iterator_tag,
				typename std::iterator_traits< IterT >::iterator_category >::value,
				"the given iterator is not a random access one" );
			assert( num_nonzeroes == static_cast< size_t >( end - begin ) );
			size_t first, last;
			partition_nonzeroes( num_nonzeroes, first, last );
			if( last < num_nonzeroes ) {
				end = begin;
				end += last;
			}
			begin += first;
		}

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
		template<
			std::size_t DIMS,
			typename coord_t,
			typename T,
			enum grb::Backend B
		> grb::RC build_ndims_system_matrix(
			grb::Matrix< T, B > & M,
			const grb::algorithms::hpcg_builder< DIMS, coord_t, T > & hpcg_system,
			T diag_value,
			T non_diag_value
		) {
			if( hpcg_system.system_size() > std::numeric_limits< coord_t >::max() ) {
				throw std::domain_error( "CoordT cannot store the matrix coordinates" );
			}
			/*
			std::array< coord_t, DIMS > _sys_sizes;
			for( size_t i = 0; i < DIMS; i++ ) _sys_sizes[i] = sys_sizes[i];
			grb::algorithms::hpcg_builder< DIMS, coord_t, T > hpcg_system( _sys_sizes, halo_size );
			*/
			grb::algorithms::matrix_generator_iterator< DIMS, coord_t, T > begin(
				hpcg_system.make_begin_iterator( diag_value, non_diag_value ) );
			grb::algorithms::matrix_generator_iterator< DIMS, coord_t, T > end(
				hpcg_system.make_end_iterator( diag_value, non_diag_value )
			);
			partition_iteration_range( hpcg_system.num_neighbors(), begin, end );

			return buildMatrixUnique( M, begin, end, grb::IOMode::PARALLEL );
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
		 *                    in \p coarser_size , otherwise an exception is thrown
		 * @return grb::RC the success value returned when trying to build the matrix
		 */
		template<
			std::size_t DIMS,
			typename T,
			enum grb::Backend B
		> grb::RC build_ndims_coarsener_matrix(
			grb::Matrix< T, B > & M,
			const std::array< std::size_t, DIMS > & coarser_sizes,
			const std::array< std::size_t, DIMS > & finer_sizes
		) {
			static_assert( DIMS > 0, "DIMS must be > 0" );
			size_t const rows { std::accumulate( coarser_sizes.cbegin(), coarser_sizes.cend(), 1UL, std::multiplies< size_t >() ) };
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
			using coord_t = unsigned;
			if( rows > std::numeric_limits< coord_t >::max() ) {
				throw std::domain_error( "CoordT cannot store the row coordinates" );
			}
			if( cols > std::numeric_limits< coord_t >::max() ) {
				throw std::domain_error( "CoordT cannot store the column coordinates" );
			}
			std::array< coord_t, DIMS > _coarser_sizes, _finer_sizes;
			for( size_t i = 0; i < DIMS; i++ ) {
				_coarser_sizes[i] = coarser_sizes[i];
				_finer_sizes[i] = finer_sizes[i];
			}
			grb::algorithms::hpcg_coarsener_builder< DIMS, coord_t, T > coarsener( _coarser_sizes, _finer_sizes );
			grb::algorithms::coarsener_generator_iterator< DIMS, coord_t, T > begin( coarsener.make_begin_iterator() );
			grb::algorithms::coarsener_generator_iterator< DIMS, coord_t, T > end(
				coarsener.make_end_iterator()
			);
			partition_iteration_range( coarsener.system_size(), begin, end );
			return buildMatrixUnique( M, begin, end, grb::IOMode::PARALLEL );
		}

		template< typename CoordT >
		struct true_iter {

			static const bool __TRUE = true;

			using self_t = true_iter< CoordT >;
			using iterator_category = std::random_access_iterator_tag;
			using value_type = bool;
			using pointer = const bool *;
			using reference = const bool&;
			using difference_type = long;

			true_iter() = delete;

			true_iter( CoordT first ): index( first ) {}

			true_iter( const self_t & ) = default;

			self_t & operator=( const self_t & ) = default;

			bool operator!=( const self_t & other ) const {
				return this->index != other.index;
			}

			self_t & operator++() noexcept {
				(void) index++;
				return *this;
			}

			self_t & operator+=( size_t increment ) noexcept {
				index += increment;
				return *this;
			}

			difference_type operator-( const self_t & other ) noexcept {
				return static_cast< difference_type >( this->index - other.index );
			}

			pointer operator->() const {
				return &__TRUE;
			}

			reference operator*() const {
				return *(this->operator->());
			}

		private:
			CoordT index;
		};

		template< typename CoordT > const bool true_iter< CoordT >::__TRUE;

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
		grb::RC build_static_color_masks(
			std::size_t matrix_size,
			const std::vector< std::vector< size_t > > &per_color_rows,
			std::vector< grb::Vector< bool, B > > & masks
		) {
			if( ! masks.empty() ) {
				throw std::invalid_argument( "vector of masks is expected to be empty" );
			}
			for( size_t i = 0; i < per_color_rows.size(); i++ ) {
				const std::vector< size_t > & rows = per_color_rows[ i ];
				/*
				{
					std::cout << "\ncolor " << i << std::endl;
					for( size_t row : rows ) {
						std::cout << row << " ";
					}
					std::cout << std::endl;
				}
				*/
				masks.emplace_back( matrix_size );
				grb::Vector< bool > & output_mask = masks.back();
				std::vector< size_t >::const_iterator begin = rows.cbegin();
				std::vector< size_t >::const_iterator end = rows.cend();
				// partition_iteration_range( rows.size(), begin, end );
				grb::RC rc = grb::buildVectorUnique( output_mask, begin , end, true_iter< size_t >( 0 ),
					true_iter< size_t >( std::distance( begin, end ) ), IOMode::SEQUENTIAL );
				if( rc != SUCCESS ) {
					std::cerr << "error while creating output mask for color " << i << ": "
						<< toString( rc ) << std::endl;
					return rc;
				}
				/*
				{
					std::cout << "mask color " << i << std::endl;
					size_t count = 0;
					for( const auto & v : output_mask ) {
						std::cout << v.first << " ";
						count++;
						if( count > 20 ) break;
					}
					std::cout << std::endl;
				}
				*/
			}
			return grb::SUCCESS;
		}

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MATRIX_BUILDING_UTILS
