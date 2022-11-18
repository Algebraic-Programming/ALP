
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
 * @file hpcg_system_building_utils.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @brief Utilities to build an antire system for HPCG simulations in an arbitrary number of dimensions.
 * @date 2021-04-30
 */

#ifndef _H_GRB_ALGORITHMS_HPCG_SYSTEM_BUILDING_UTILS
#define _H_GRB_ALGORITHMS_HPCG_SYSTEM_BUILDING_UTILS

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include <graphblas.hpp>
#include <graphblas/utils/iterators/partition_range.hpp>

#include "ndim_matrix_builders.hpp"

#include "coloring.hpp"

namespace grb {
	namespace algorithms {

		template< typename CoordType > void split_rows_by_color(
			const std::vector< CoordType > & row_colors,
			size_t num_colors,
			std::vector< std::vector< CoordType > > & per_color_rows
		) {
			per_color_rows.resize( num_colors );
			for( CoordType i = 0; i < row_colors.size(); i++ ) {
				per_color_rows[ row_colors[ i ] ].push_back( i );
			}
		}

		template <
			size_t DIMS,
			typename coord_t,
			typename NonzeroType,
			enum grb::Backend B
		> grb::RC populate_system_matrix(
			const grb::algorithms::HPCGBuilder< DIMS, coord_t, NonzeroType > &system_generator,
			grb::Matrix< NonzeroType, B > &M
		) {
			const size_t pid { spmd<>::pid() };

			if( pid == 0) {
				std::cout << "- generating system matrix...";
			}
			typename grb::algorithms::HPCGBuilder< DIMS, coord_t, NonzeroType >::Iterator begin(
				system_generator.make_begin_iterator() );
			typename grb::algorithms::HPCGBuilder< DIMS, coord_t, NonzeroType >::Iterator end(
				system_generator.make_end_iterator()
			);
			grb::utils::partition_iteration_range_on_procs( system_generator.num_neighbors(), begin, end );
			return buildMatrixUnique( M, begin, end, grb::IOMode::PARALLEL );
		}


		template<
			typename coord_t,
			size_t DIMS,
			typename IOType,
			typename NonzeroType
		> grb::RC populate_coarsener(
			const grb::algorithms::HPCGBuilder< DIMS, coord_t, NonzeroType > &finer_system_generator,
			const grb::algorithms::HPCGBuilder< DIMS, coord_t, NonzeroType > &coarser_system_generator,
			coarsening_data< IOType, NonzeroType > &coarsener
		) {
			static_assert( DIMS > 0, "DIMS must be > 0" );

			const std::array< coord_t, DIMS > &finer_sizes = finer_system_generator.get_generator().get_sizes();
			const std::array< coord_t, DIMS > &coarser_sizes = coarser_system_generator.get_generator().get_sizes();
			const size_t finer_size = finer_system_generator.system_size();
			const size_t coarser_size = coarser_system_generator.system_size();

			if( coarser_size >= finer_size ) {
				throw std::invalid_argument( "wrong sizes");
			}

			size_t const rows { coarser_size };
			size_t const cols { finer_size };

			assert( finer_sizes.size() == coarser_sizes.size() );

			for( size_t i { 0 }; i < coarser_sizes.size(); i++ ) {
				std::ldiv_t ratio = std::ldiv( finer_sizes[ i ], coarser_sizes[ i ] );
				if( ratio.quot < 2 || ratio.rem != 0 ) {
					throw std::invalid_argument( "finer sizes should be a multiple of coarser sizes" );
				}
			}
			grb::Matrix< NonzeroType > &M = coarsener.coarsening_matrix;
			if( grb::nrows( M ) != rows || grb::ncols( M ) != cols ) {
				throw std::invalid_argument( "wrong matrix dimensions: matrix should be rectangular"
											" with rows == <coarser size> and cols == <finer size>" );
			}

			grb::algorithms::hpcg_coarsener_builder< DIMS, coord_t, NonzeroType > coarsener_builder( coarser_sizes, finer_sizes );
			grb::algorithms::coarsener_generator_iterator< DIMS, coord_t, NonzeroType > begin( coarsener_builder.make_begin_iterator() );
			grb::algorithms::coarsener_generator_iterator< DIMS, coord_t, NonzeroType > end( coarsener_builder.make_end_iterator() );
			grb::utils::partition_iteration_range_on_procs( coarsener_builder.system_size(), begin, end );
			return buildMatrixUnique( M, begin, end, grb::IOMode::PARALLEL );
		}

		namespace internal {

			template< typename CoordType > struct true_iter {

				static const bool __TRUE = true;

				using self_t = true_iter< CoordType >;
				using iterator_category = std::random_access_iterator_tag;
				using value_type = bool;
				using pointer = const bool *;
				using reference = const bool&;
				using difference_type = long;

				true_iter() = delete;

				true_iter( CoordType first ): index( first ) {}

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
				CoordType index;
			};

			template< typename CoordType > const bool true_iter< CoordType >::__TRUE;

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
				size_t matrix_size,
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

		} // namespace internal

		template<
			typename coord_t,
			size_t DIMS,
			typename T
		> grb::RC populate_smoothing_data(
			const grb::algorithms::HPCGBuilder< DIMS, coord_t, T > &system_generator,
			smoother_data< T > &smoothing_info
		) {
			const size_t pid { spmd<>::pid() };

			grb::RC rc = set( smoothing_info.A_diagonal, system_generator.get_diag_value() );
			if( rc != grb::SUCCESS ) {
				if( pid == 0 ) {
					std::cout << "error: " << __LINE__ << std::endl;
				}
				return rc;
			}

			if( pid == 0 ) {
				std::cout << "- running coloring heuristics...";
			}
			std::vector< coord_t > colors, color_counters;
			color_matrix_greedy( system_generator.get_generator(), colors, color_counters );
			std::vector< std::vector< coord_t > > per_color_rows;
			split_rows_by_color( colors, color_counters.size(), per_color_rows );
			if( rc != grb::SUCCESS ) {
				if( pid == 0 ) {
					std::cout << "error: " << __LINE__ << std::endl;
				}
				return rc;
			}
			if( pid == 0 ) {
				std::cout <<"- found " << color_counters.size() << " colors,"
					<< " generating color masks...";
			}
			return internal::build_static_color_masks( system_generator.system_size(),
				per_color_rows, smoothing_info.color_masks );
		}

		/**
		 * @brief Container of the parameter for HPCG simulation generation: physical system characteristics and
		 * coarsening information.
		 *
		 * @tparam DIMS dimensions of the physical system
		 * @tparam T type of matrix values
		 */
		template< size_t DIMS, typename T >
		struct hpcg_system_params {
			std::array< size_t, DIMS > physical_sys_sizes;
			size_t halo_size;
			T diag_value;
			T non_diag_value;
			size_t min_phys_size;
			size_t max_levels;
			size_t coarsening_step;
		};

		template<
			size_t DIMS,
			typename coord_t,
			typename T
		> void build_hpcg_multigrid_generators(
			const hpcg_system_params< DIMS, T > &params,
			std::vector< grb::algorithms::HPCGBuilder< DIMS, coord_t, T > > &mg_generators
		) {
			static_assert( DIMS > 0, "DIMS must be > 0" );

			size_t const current_size{ std::accumulate( params.physical_sys_sizes.cbegin(), params.physical_sys_sizes.cend(), 1UL,
				std::multiplies< size_t >() ) };
			if( current_size > std::numeric_limits< coord_t >::max() ) {
				throw std::domain_error( "CoordT cannot store the matrix coordinates" );
			}
			size_t min_physical_size { *std::min_element( params.physical_sys_sizes.cbegin(), params.physical_sys_sizes.cend() ) };
			if( min_physical_size < params.min_phys_size ) {
				throw std::domain_error( "the initial system is too small" );
			}

			std::array< coord_t, DIMS > coord_sizes;
			// type-translate coordinates
			std::copy( params.physical_sys_sizes.cbegin(), params.physical_sys_sizes.cend(), coord_sizes.begin() );

			// generate linked list of hierarchical coarseners
			for( size_t coarsening_level = 0UL;
				min_physical_size >= params.min_phys_size && coarsening_level <= params.max_levels;
				coarsening_level++ ) {

				// build generator
				mg_generators.emplace_back( coord_sizes, params.halo_size, params.diag_value, params.non_diag_value );

				// prepare for new iteration
				min_physical_size /= params.coarsening_step;
				std::for_each( coord_sizes.begin(), coord_sizes.end(),
					[ &params ]( coord_t &v ){ v /= params.coarsening_step; });
			}
		}

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_HPCG_SYSTEM_BUILDING_UTILS
