
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

#ifndef _H_HPCG_SYSTEM_BUILDING_UTILS
#define _H_HPCG_SYSTEM_BUILDING_UTILS

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>

#include <graphblas/algorithms/hpcg_data.hpp>
#include <graphblas/rc.hpp>

#include "hpcg_matrix_building_utils.hpp"

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
	const std::array< std::size_t, DIMS > & physical_sys_sizes;
	const std::size_t halo_size;
	const std::size_t num_colors;
	const T diag_value;
	const T non_diag_value;
	const std::size_t min_phys_size;
	const std::size_t max_levels;
	const std::size_t coarsening_step;
};

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
grb::RC build_hpcg_system( std::unique_ptr< grb::algorithms::hpcg_data< T, T, T > > & holder, hpcg_system_params< DIMS, T > & params ) {
	// n is the system matrix size
	const std::size_t n { std::accumulate( params.physical_sys_sizes.cbegin(), params.physical_sys_sizes.cend(), 1UL, std::multiplies< std::size_t >() ) };

	grb::algorithms::hpcg_data< T, T, T > * data { new grb::algorithms::hpcg_data< T, T, T >( n ) };

	assert( ! holder ); // should be empty
	holder = std::unique_ptr< grb::algorithms::hpcg_data< T, T, T > >( data );

	// initialize the main (=uncoarsened) system matrix
	grb::RC rc { grb::SUCCESS };
	rc = build_ndims_system_matrix< DIMS, T >( data->A, params.physical_sys_sizes, params.halo_size, params.diag_value, params.non_diag_value );

	if( rc != grb::SUCCESS ) {
		std::cerr << "Failure to generate the initial system (" << toString( rc ) << ") of size " << n << std::endl;
		return rc;
	}

	// set values of diagonal vector
	set( data->A_diagonal, params.diag_value );

	build_static_color_masks( data->color_masks, n, params.num_colors );

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
		grb::algorithms::multi_grid_data< double, double > * new_coarser { new grb::algorithms::multi_grid_data< double, double >( coarser_size, previous_size ) };
		// install coarser level immediately to cleanup in case of build error
		*coarser = new_coarser;
		// initialize coarsener matrix, system matrix and diagonal vector for the coarser level
		rc = build_ndims_coarsener_matrix< DIMS >( new_coarser->coarsening_matrix, coarser_sizes, previous_sizes );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Failure to generate coarsening matrix (" << toString( rc ) << ")." << std::endl;
			return rc;
		}
		rc = build_ndims_system_matrix< DIMS, T >( new_coarser->A, coarser_sizes, params.halo_size, params.diag_value, params.non_diag_value );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Failure to generate system matrix (" << toString( rc ) << ")for size " << coarser_size << std::endl;
			return rc;
		}
		set( new_coarser->A_diagonal, params.diag_value );
		// build color masks for coarser level (same masks, but with coarser system size)
		rc = build_static_color_masks( new_coarser->color_masks, coarser_size, params.num_colors );

		// prepare for new iteration
		coarser = &new_coarser->coarser_level;
		min_physical_coarsened_size /= params.coarsening_step;
		previous_sizes = coarser_sizes;
		divide_array( coarser_sizes, coarser_sizes, params.coarsening_step );
		coarsening_level++;
	}
	return rc;
}

#endif // _H_HPCG_SYSTEM_BUILDING_UTILS
