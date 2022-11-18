
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

#ifndef _H_GRB_ALGORITHMS_SYSTEM_BUILDING_UTILS
#define _H_GRB_ALGORITHMS_SYSTEM_BUILDING_UTILS

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>

#include <graphblas.hpp>
#include <graphblas/utils/Timer.hpp>

#include "hpcg_data.hpp"
#include "matrix_building_utils.hpp"

#include "coloring.hpp"

#ifndef MASTER_PRINT
#define INTERNAL_MASTER_PRINT
#define MASTER_PRINT( pid, txt ) if( pid == 0 ) { std::cout << txt; }
#endif


namespace grb {
	namespace algorithms {

		/**
		 * @brief Divide each value of \p source by \p step and store the result into \p destination.
		 *
		 * @tparam DIMS size of passed arrays
		 */
		template< std::size_t DIMS >
		void divide_array( std::array< std::size_t, DIMS > & destination, const std::array< std::size_t, DIMS > & source, std::size_t step ) {
			for( std::size_t i { 0 }; i < destination.size(); i++ ) {
				destination[ i ] = source[ i ] / step;
			}
		}

		/**
		 * @brief Container of the parameter for HPCG simulation generation: physical system characteristics and
		 * coarsening information.
		 *
		 * @tparam DIMS dimensions of the physical system
		 * @tparam T type of matrix values
		 */
		template< std::size_t DIMS, typename T >
		struct hpcg_system_params {
			std::array< std::size_t, DIMS > physical_sys_sizes;
			std::size_t halo_size;
			T diag_value;
			T non_diag_value;
			std::size_t min_phys_size;
			std::size_t max_levels;
			std::size_t coarsening_step;
		};

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

		// SystemData must have a zero_temp_vectors()
		template< std::size_t DIMS, typename IOType, typename NonzeroType, typename SystemData >
		grb::RC build_base_system(
			typename std::enable_if<
				std::is_base_of< system_data< IOType, NonzeroType >, SystemData >::value,
			SystemData& >::type system,
			size_t system_size,
			const std::array< std::size_t, DIMS > & physical_sys_sizes,
			size_t halo_size,
			NonzeroType diag_value,
			NonzeroType non_diag_value,
			std::array< double, 4 > & times
		) {

			grb::RC rc { grb::SUCCESS };
			const size_t pid { spmd<>::pid() };
			grb::utils::Timer timer;
			static const char * const log_prefix = "  -- ";

			using coord_t = size_t;
			static_assert( DIMS > 0, "DIMS must be > 0" );
			size_t n { std::accumulate( physical_sys_sizes.cbegin(), physical_sys_sizes.cend(),
				1UL, std::multiplies< size_t >() ) };
			if( n > std::numeric_limits< coord_t >::max() ) {
				throw std::domain_error( "CoordT cannot store the matrix coordinates" );
			}
			std::array< coord_t, DIMS > sys_sizes;
			for( size_t i = 0; i < DIMS; i++ ) sys_sizes[i] = physical_sys_sizes[i];
			grb::algorithms::hpcg_builder< DIMS, coord_t, NonzeroType > system_generator( sys_sizes, halo_size );

			MASTER_PRINT( pid, log_prefix << "generating system matrix..." );
			timer.reset();
			rc = build_ndims_system_matrix< DIMS, coord_t, NonzeroType >(
				system.A,
				system_generator,
				diag_value, non_diag_value
			);
			if( rc != grb::SUCCESS ) {
				return rc;
			}
			times[ 0 ] = timer.time();
			MASTER_PRINT( pid, " time (ms) " << times[ 0 ] << std::endl );

			// set values of vectors
			MASTER_PRINT( pid, log_prefix << "populating vectors..." );
			timer.reset();
			rc = set( system.A_diagonal, diag_value );
			if( rc != grb::SUCCESS ) {
				return rc;
			}
			rc = system.zero_temp_vectors();
			if( rc != grb::SUCCESS ) {
				return rc;
			}
			times[ 1 ] = timer.time();
			MASTER_PRINT( pid, " time (ms) " << times[ 1 ] << std::endl );

			MASTER_PRINT( pid, log_prefix << "running coloring heuristics..." );
			timer.reset();
			std::vector< coord_t > colors, color_counters;
			color_matrix_greedy( system_generator.get_generator(), colors, color_counters );
			std::vector< std::vector< coord_t > > per_color_rows;
			split_rows_by_color( colors, color_counters.size(), per_color_rows );
			if( rc != grb::SUCCESS ) {
				return rc;
			}
			times[ 2 ] = timer.time();
			MASTER_PRINT( pid, " found " << color_counters.size() << " colors, time (ms) "
				<< times[ 2 ] << std::endl );


			MASTER_PRINT( pid, log_prefix << "generating color masks..." );
			timer.reset();
			rc = build_static_color_masks( system_size, per_color_rows, system.color_masks );
			if( rc != grb::SUCCESS ) {
				return rc;
			}
			times[ 3 ] = timer.time();
			MASTER_PRINT( pid, " time (ms) " << times[ 3 ] << std::endl );

			return rc;
		}

		/**
		 * @brief Generates an entire HPCG problem according to the parameters in \p params , storing it in \p holder .
		 *
		 * @tparam DIMS dimensions of the system
		 * @tparam T type of matrix values
		 * @param holder std::unique_ptr to store the HPCG problem into
		 * @param params parameters container to build the HPCG problem
		 * @return grb::SUCCESS if every GraphBLAS operation (to generate vectors and matrices) succeeded,
		 * otherwise the first unsuccessful return value
		 */
		template< std::size_t DIMS, typename T = double >
		grb::RC build_hpcg_system(
			std::unique_ptr< grb::algorithms::hpcg_data< T, T, T > > & holder,
			const hpcg_system_params< DIMS, T > & params
		) {
			// n is the system matrix size
			const std::size_t n { std::accumulate( params.physical_sys_sizes.cbegin(),
				params.physical_sys_sizes.cend(), 1UL, std::multiplies< std::size_t >() ) };

			grb::algorithms::hpcg_data< T, T, T > * data { new grb::algorithms::hpcg_data< T, T, T >( n ) };

			assert( ! holder ); // should be empty
			holder = std::unique_ptr< grb::algorithms::hpcg_data< T, T, T > >( data );

			// initialize the main (=uncoarsened) system matrix
			grb::RC rc { grb::SUCCESS };
			const size_t pid { spmd<>::pid() };
			grb::utils::Timer timer;

			std::array< double, 4 > times;
			MASTER_PRINT( pid, "\n-- main system" << std::endl );
			rc = build_base_system< DIMS, T, T, grb::algorithms::hpcg_data< T, T, T > >( *data, n, params.physical_sys_sizes, params.halo_size,
				params.diag_value, params.non_diag_value, times );
			if( rc != grb::SUCCESS ) {
				MASTER_PRINT( pid, " error: " << toString( rc ) );
				return rc;
			}
			MASTER_PRINT( pid, "-- main system generation time (ms) "
				"(system matrix,vectors,coloring,color masks):" << times[ 0 ] << "," << times[ 1 ]
				<< "," << times[ 2 ] << "," << times[ 3 ] << std::endl;
			);

			// initialize coarsening with additional pointers and dimensions copies to iterate and divide
			grb::algorithms::multi_grid_data< T, T > ** coarser = &data->coarser_level;
			assert( *coarser == nullptr );
			std::array< std::size_t, DIMS > coarser_sizes;
			std::array< std::size_t, DIMS > previous_sizes( params.physical_sys_sizes );
			std::size_t min_physical_coarsened_size { *std::min_element( previous_sizes.cbegin(), previous_sizes.cend() ) / params.coarsening_step };
			// coarsen system sizes into coarser_sizes
			divide_array( coarser_sizes, previous_sizes, params.coarsening_step );
			std::size_t coarsening_level = 0UL;

			// generate linked list of hierarchical coarseners
			while( min_physical_coarsened_size >= params.min_phys_size && coarsening_level < params.max_levels ) {
				assert( *coarser == nullptr );
				// compute size of finer and coarser matrices
				std::size_t coarser_size { std::accumulate( coarser_sizes.cbegin(), coarser_sizes.cend(), 1UL, std::multiplies< std::size_t >() ) };
				std::size_t previous_size { std::accumulate( previous_sizes.cbegin(), previous_sizes.cend(), 1UL, std::multiplies< std::size_t >() ) };
				// build data structures for new level
				grb::algorithms::multi_grid_data< T, T > * new_coarser { new grb::algorithms::multi_grid_data< double, double >( coarser_size, previous_size ) };
				// install coarser level immediately to cleanup in case of build error
				*coarser = new_coarser;

				MASTER_PRINT( pid, "-- level " << coarsening_level << "\n  -- generating coarsening matrix...\n" );
				timer.reset();
				// initialize coarsener matrix, system matrix and diagonal vector for the coarser level
				rc = build_ndims_coarsener_matrix< DIMS >( new_coarser->coarsening_matrix, coarser_sizes, previous_sizes );
				if( rc != grb::SUCCESS ) {
					MASTER_PRINT( pid, " error: " << toString( rc ) );
					return rc;
				}
				double coarsener_gen_time{ timer.time() };

				rc = build_base_system< DIMS, T, T, grb::algorithms::multi_grid_data< T, T > >( *new_coarser, coarser_size, coarser_sizes, params.halo_size,
					params.diag_value, params.non_diag_value, times );
				if( rc != grb::SUCCESS ) {
					MASTER_PRINT( pid, " error: " << toString( rc ) );
					return rc;
				}
				MASTER_PRINT( pid, "-- level generation time (ms) "
					"(level,coarsening matrix,system matrix,vectors,coloring,color masks):"
					<< coarsening_level << "," << coarsener_gen_time << "," << times[ 0 ] << "," << times[ 1 ]
					<< "," << times[ 2 ] << "," << times[ 3 ] << std::endl;
				);

				// prepare for new iteration
				coarser = &new_coarser->coarser_level;
				min_physical_coarsened_size /= params.coarsening_step;
				previous_sizes = coarser_sizes;
				divide_array( coarser_sizes, coarser_sizes, params.coarsening_step );
				coarsening_level++;
			}
			return rc;
		}

	} // namespace algorithms
} // namespace grb

#ifdef INTERNAL_MASTER_PRINT
#undef INTERNAL_MASTER_PRINT
#undef MASTER_PRINT
#endif

#endif // _H_GRB_ALGORITHMS_SYSTEM_BUILDING_UTILS
