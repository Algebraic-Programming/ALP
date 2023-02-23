
/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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
 * @file system_building_utils.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Utilities to build an antire system for HPCG simulations in an arbitrary number of dimensions.
 */

#ifndef _H_GRB_ALGORITHMS_HPCG_SYSTEM_BUILDING_UTILS
#define _H_GRB_ALGORITHMS_HPCG_SYSTEM_BUILDING_UTILS

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include <string>

#include <graphblas.hpp>
#include <graphblas/utils/iterators/partition_range.hpp>

#include "system_builder.hpp"
#include "single_point_coarsener.hpp"
#include "average_coarsener.hpp"
#include "greedy_coloring.hpp"

namespace grb {
	namespace algorithms {

		/**
		 * Container of the parameter for HPCG simulation generation: physical system characteristics and
		 * coarsening information.
		 *
		 * @tparam DIMS dimensions of the physical system
		 * @tparam T type of matrix values
		 */
		template<
			size_t DIMS,
			typename NonzeroType
		> struct HPCGSystemParams {
			std::array< size_t, DIMS > physical_sys_sizes;
			size_t halo_size;
			NonzeroType diag_value;
			NonzeroType non_diag_value;
			size_t min_phys_size;
			size_t max_levels;
			size_t coarsening_step;
		};

		/**
		 * Builds all required system generators for an entire multi-grid simulation; each generator
		 * corresponds to a level of the HPCG system multi-grid, with increasingly coarser sizes, and can
		 * generate the system matrix of that level. All required pieces of information required to build
		 * the levels is stored in \p params.
		 *
		 * @tparam DIMS number of dimensions
		 * @tparam CoordType type storing the coordinates and the sizes
		 * @tparam NonzeroType type of the nonzero
		 * @param[in] params structure with the parameters to build an entire HPCG simulation
		 * @param[out] mg_generators std::vector of HPCGSystemBuilder, one per layer of the multi-grid
		 */
		template<
			size_t DIMS,
			typename CoordType,
			typename NonzeroType
		> void hpcg_build_multigrid_generators(
			const HPCGSystemParams< DIMS, NonzeroType > &params,
			std::vector< grb::algorithms::HPCGSystemBuilder< DIMS, CoordType, NonzeroType > > &mg_generators
		) {
			static_assert( DIMS > 0, "DIMS must be > 0" );

			size_t const current_size = std::accumulate( params.physical_sys_sizes.cbegin(),
				params.physical_sys_sizes.cend(), 1UL, std::multiplies< size_t >() );
			if( current_size > std::numeric_limits< CoordType >::max() ) {
				throw std::domain_error( "CoordType cannot store the matrix coordinates" );
			}
			size_t min_physical_size = *std::min_element( params.physical_sys_sizes.cbegin(),
				params.physical_sys_sizes.cend() );
			if( min_physical_size < params.min_phys_size ) {
				throw std::domain_error( "the initial system is too small" );
			}

			std::array< CoordType, DIMS > coord_sizes;
			// type-translate coordinates
			std::copy( params.physical_sys_sizes.cbegin(), params.physical_sys_sizes.cend(),
				coord_sizes.begin() );

			// generate hierarchical coarseners
			for( size_t coarsening_level = 0UL;
				min_physical_size >= params.min_phys_size && coarsening_level <= params.max_levels;
				coarsening_level++ ) {

				// build generator
				mg_generators.emplace_back( coord_sizes, params.halo_size,
					params.diag_value, params.non_diag_value );

				// prepare for new iteration
				min_physical_size /= params.coarsening_step;
				std::for_each( coord_sizes.begin(), coord_sizes.end(),
					[ &params ]( CoordType &v ) {
						std::ldiv_t ratio = std::ldiv( v, params.coarsening_step );
						if( ratio.rem != 0 ) {
							throw std::invalid_argument(
								std::string( "system size " ) + std::to_string( v ) +
								std::string( " is not divisible by " ) +
								std::to_string( params.coarsening_step )
							);
						}
						v = ratio.quot;
					});
			}
		}

		/**
		 * Populates the system matrix \p M out of the builder \p system_generator.
		 *
		 * The matrix \p M must have been previously allocated and initialized with the proper sizes,
		 * as this procedure only populates it with the nozeroes generated by \p system_generator.
		 *
		 * This function takes care of the parallelism by employing random-access iterators and by
		 * \b parallelizing the generation across multiple processes in case of distributed execution.
		 */
		template <
			size_t DIMS,
			typename CoordType,
			typename NonzeroType,
			typename Logger
		> grb::RC hpcg_populate_system_matrix(
			const grb::algorithms::HPCGSystemBuilder< DIMS, CoordType, NonzeroType > &system_generator,
			grb::Matrix< NonzeroType > &M,
			Logger & logger
		) {

			logger << "- generating system matrix...";
			typename grb::algorithms::HPCGSystemBuilder< DIMS, CoordType, NonzeroType >::Iterator begin(
				system_generator.make_begin_iterator() );
			typename grb::algorithms::HPCGSystemBuilder< DIMS, CoordType, NonzeroType >::Iterator end(
				system_generator.make_end_iterator()
			);
			grb::utils::partition_iteration_range_on_procs( spmd<>::nprocs(), spmd<>::pid(),
				system_generator.num_neighbors(), begin, end );
			return buildMatrixUnique( M, begin, end, grb::IOMode::PARALLEL );
		}

		/**
		 * Populates the coarsening data \p coarsener (in particular the coarsening matrix) from the
		 * builder of the finer system \p finer_system_generator and that of the coarser system
		 * \p coarser_system_generator.
		 *
		 * This function takes care of parallelizing the generation by using a random-access iterator
		 * to generate the coarsening matrix and by distributing the generation across nodes
		 * of a distributed system (if any).
		 * @tparam IterBuilderType type of the matrix builder, either SinglePointCoarsenerBuilder
		 *  or AverageCoarsenerBuilder
		 * @tparam DIMS number of dimensions
		 * @tparam CoordType type storing the coordinates and the sizes
		 * @tparam NonzeroType type of the nonzero
		 * @param finer_system_generator object generating the finer system
		 * @param coarser_system_generator object generating the finer system
		 * @param coarsener structure with the matrix to populate
		 */
		template<
			typename IterBuilderType,
			size_t DIMS,
			typename CoordType,
			typename IOType,
			typename NonzeroType
		> grb::RC hpcg_populate_coarsener_any_builder(
			const grb::algorithms::HPCGSystemBuilder< DIMS, CoordType, NonzeroType > &finer_system_generator,
			const grb::algorithms::HPCGSystemBuilder< DIMS, CoordType, NonzeroType > &coarser_system_generator,
			CoarseningData< IOType, NonzeroType > &coarsener
		) {
			static_assert( DIMS > 0, "DIMS must be > 0" );

			const std::array< CoordType, DIMS > &finer_sizes = finer_system_generator.get_generator().get_sizes();
			const std::array< CoordType, DIMS > &coarser_sizes = coarser_system_generator.get_generator().get_sizes();
			const size_t finer_size = finer_system_generator.system_size();
			const size_t coarser_size = coarser_system_generator.system_size();

			if( coarser_size >= finer_size ) {
				throw std::invalid_argument( "wrong sizes");
			}

			size_t const rows = coarser_size;
			size_t const cols = finer_size;

			assert( finer_sizes.size() == coarser_sizes.size() );

			grb::Matrix< NonzeroType > &M = coarsener.coarsening_matrix;
			if( grb::nrows( M ) != rows || grb::ncols( M ) != cols ) {
				throw std::invalid_argument( "wrong matrix dimensions: matrix should be rectangular"
											" with rows == <coarser size> and cols == <finer size>" );
			}

			IterBuilderType coarsener_builder( finer_sizes, coarser_sizes );
			typename IterBuilderType::Iterator begin( coarsener_builder.make_begin_iterator() ),
				end( coarsener_builder.make_end_iterator() );
			grb::utils::partition_iteration_range_on_procs( spmd<>::nprocs(), spmd<>::pid(),
				coarsener_builder.system_size(), begin, end );
			return buildMatrixUnique( M, begin, end, grb::IOMode::PARALLEL );
		}

		/**
		 * Populates a coarsener that samples one element every \a 2^DIMS .
		 */
		template<
			size_t DIMS,
			typename CoordType,
			typename IOType,
			typename NonzeroType
		> grb::RC hpcg_populate_coarsener(
			const grb::algorithms::HPCGSystemBuilder< DIMS, CoordType, NonzeroType > &finer_system_generator,
			const grb::algorithms::HPCGSystemBuilder< DIMS, CoordType, NonzeroType > &coarser_system_generator,
			CoarseningData< IOType, NonzeroType > &coarsener
		) {
			return hpcg_populate_coarsener_any_builder<
				grb::algorithms::SinglePointCoarsenerBuilder< DIMS, CoordType, NonzeroType > >
				( finer_system_generator, coarser_system_generator, coarsener );
		}

		/**
		 * Populates a coarsener that averages over \a 2^DIMS elements.
		 */
		template<
			size_t DIMS,
			typename CoordType,
			typename IOType,
			typename NonzeroType
		> grb::RC hpcg_populate_coarsener_avg(
			const grb::algorithms::HPCGSystemBuilder< DIMS, CoordType, NonzeroType > &finer_system_generator,
			const grb::algorithms::HPCGSystemBuilder< DIMS, CoordType, NonzeroType > &coarser_system_generator,
			CoarseningData< IOType, NonzeroType > &coarsener
		) {
			return hpcg_populate_coarsener_any_builder<
				grb::algorithms::AverageCoarsenerBuilder< DIMS, CoordType, NonzeroType > >
				( finer_system_generator, coarser_system_generator, coarsener );
		}

		namespace internal {

			/**
			 * Store row values based on their color into separate vectors.
			 *
			 * @param[in] row_colors for each row (corresponding to a vector position) its color
			 * @param[in] num_colors number of colors, i.e. max across all values in \p row_colors + 1
			 * @param[out] per_color_rows for each position \a i it stores an std::vector with all rows
			 *  of color \a i inside \p row_colors
			 */
			template< typename CoordType > void hpcg_split_rows_by_color(
				const std::vector< CoordType > & row_colors,
				size_t num_colors,
				std::vector< std::vector< CoordType > > & per_color_rows
			) {
				per_color_rows.resize( num_colors );
				for( CoordType i = 0; i < row_colors.size(); i++ ) {
					per_color_rows[ row_colors[ i ] ].push_back( i );
				}
			}

			/**
			 * Utility class implementing a random-access iterator that always returns a
			 * \c true value.
			 *
			 * It is used in the following to build mask vectors via buildVectorUnique(), where
			 * all the non-zero positions are \c true.
			 *
			 * @tparam CoordType type of the internal coordinate
			 */
			template< typename CoordType > struct true_iter {

				// static const bool __TRUE;

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
				const bool __TRUE = true; // for its address to be passed outside
			};

			/**
			 * Populates \p masks with static color mask generated for a squared matrix of size \p matrix_size .
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
			 * @param masks output vector of color masks
			 * @param matrix_size size of the system matrix
			 * @param colors numbers of colors masks to build; it must be < \p matrix_size
			 * @return grb::RC the success value returned when trying to build the vector
			 */
			grb::RC hpcg_build_static_color_masks(
				size_t matrix_size,
				const std::vector< std::vector< size_t > > &per_color_rows,
				std::vector< grb::Vector< bool> > &masks
			) {
				if( ! masks.empty() ) {
					throw std::invalid_argument( "vector of masks is expected to be empty" );
				}
				for( size_t i = 0; i < per_color_rows.size(); i++ ) {
					const std::vector< size_t > & rows = per_color_rows[ i ];
#ifdef _DEBUG
					{
						std::cout << "\ncolor " << i << std::endl;
						for( size_t row : rows ) {
							std::cout << row << " ";
						}
						std::cout << std::endl;
					}
#endif
					masks.emplace_back( matrix_size );
					grb::Vector< bool > & output_mask = masks.back();
					std::vector< size_t >::const_iterator begin = rows.cbegin();
					std::vector< size_t >::const_iterator end = rows.cend();
					// partition_iteration_range( rows.size(), begin, end );
					grb::RC rc = grb::buildVectorUnique( output_mask, begin , end, true_iter< size_t >( 0 ),
						true_iter< size_t >( rows.size() ), IOMode::SEQUENTIAL );
					if( rc != SUCCESS ) {
						std::cerr << "error while creating output mask for color " << i << ": "
							<< toString( rc ) << std::endl;
						return rc;
					}
#ifdef _DEBUG
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
#endif
				}
				return grb::SUCCESS;
			}

		} // namespace internal

		/**
		 * Populates the smoothing information \p smoothing_info for a Red-Black Gauss-Seidel smoother
		 * to be used for an HPCG simulation. The information about the mesh to smooth are passed
		 * via \p system_generator.
		 *
		 * Steps for the smoother generation:
		 *
		 * 1. the mesh elements (the system matrix rows) are colored via a greedy algorithm, so that
		 *  no two neighboring elements have the same color; this phase colors the \b entire system
		 *  and cannot be parallelized, even in a distributed system, since the current coloring algorithm
		 *  is \b not distributed
		 * 2. rows are split according to their color
		 * 3. for each color \a c the color mask with the corresponding rows is generated:
		 *  a dedicated sparse grb::Vector<bool> signals the rows of color \a c (by marking them as \c true
		 *  ); such a vector allows updating all rows of color \a c in \b parallel when used as a mask
		 *  to an mxv() operation (as done during smoothing)
		 */
		template<
			size_t DIMS,
			typename CoordType,
			typename NonzeroType,
			typename Logger
		> grb::RC hpcg_populate_smoothing_data(
			const grb::algorithms::HPCGSystemBuilder< DIMS, CoordType, NonzeroType > &system_generator,
			SmootherData< NonzeroType > &smoothing_info,
			Logger & logger
		) {
			grb::RC rc = set( smoothing_info.A_diagonal, system_generator.get_diag_value() );
			if( rc != grb::SUCCESS ) {
				logger << "error: " << __LINE__ << std::endl;
				return rc;
			}

			logger << "- running coloring heuristics...";
			std::vector< CoordType > colors, color_counters;
			hpcg_greedy_color_ndim_system( system_generator.get_generator(), colors, color_counters );
			std::vector< std::vector< CoordType > > per_color_rows;
			internal::hpcg_split_rows_by_color( colors, color_counters.size(), per_color_rows );
			colors.clear();
			colors.shrink_to_fit();
			if( rc != grb::SUCCESS ) {
				logger << "error: " << __LINE__ << std::endl;
				return rc;
			}
			logger <<"- found " << color_counters.size() << " colors,"
				<< " generating color masks...";
			return internal::hpcg_build_static_color_masks( system_generator.system_size(),
				per_color_rows, smoothing_info.color_masks );
		}

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_HPCG_SYSTEM_BUILDING_UTILS
