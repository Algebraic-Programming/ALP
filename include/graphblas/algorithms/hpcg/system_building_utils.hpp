
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

#include <graphblas.hpp>
#include <graphblas/utils/parser.hpp>

#include "hpcg_data.hpp"
#include "matrix_building_utils.hpp"


namespace grb {
	namespace algorithms {

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
		template< typename T = double, typename SYSINP >
		grb::RC build_hpcg_system( std::unique_ptr< grb::algorithms::hpcg_data< T, T, T > > & holder, const std::size_t max_levels, SYSINP &in ) {
			grb::RC rc { grb::SUCCESS };
			std::size_t coarsening_level = 0UL;
			const size_t n_A = in.matAbuffer[ coarsening_level ].get_n();
			grb::algorithms::hpcg_data< T, T, T > * data { new grb::algorithms::hpcg_data< T, T, T >( n_A ) };
			rc = buildMatrixUnique( data->A,
									in.matAbuffer[ coarsening_level ].i_data,
									in.matAbuffer[ coarsening_level ].j_data,
									in.matAbuffer[ coarsening_level ].v_data,
									in.matAbuffer[ coarsening_level ].size(),
									SEQUENTIAL
									);

			/* Once internal issue #342 is resolved this can be re-enabled
			   const RC rc = buildMatrixUnique( L,
			   parser.begin( PARALLEL ), parser.end( PARALLEL),
			   PARALLEL
			   );*/
			if( rc != SUCCESS ) {
				std::cerr << "Failure: call to buildMatrixUnique did not succeed "
						  << "(" << toString( rc ) << ")." << std::endl;
				return rc;
			}
#ifdef HPCG_PRINT_STEPS
			std::cout << " buildMatrixUnique: constructed data->A " << nrows(data->A) << " x " << ncols(data->A) << " matrix \n";
#endif
			assert( ! holder ); // should be empty
			holder = std::unique_ptr< grb::algorithms::hpcg_data< T, T, T > >( data );

			{
				RC rc = grb::buildVector( data->A_diagonal,
										  in.matMbuffer[coarsening_level].v_data,
										  in.matMbuffer[coarsening_level].v_data + in.matMbuffer[coarsening_level].size(),
										  SEQUENTIAL );
				if ( rc != SUCCESS ) {
					std::cerr << " buildVector failed!\n ";
					return rc;
				}
#ifdef HPCG_PRINT_STEPS
				std::cout << " buildVector: data->A_diagonal "
						  << size(data->A_diagonal) << " vector \n";
#endif
			}

			std::size_t coarser_size;
			std::size_t previous_size=n_A;
			
			// initialize coarsening with additional pointers and dimensions copies to iterate and divide
			grb::algorithms::multi_grid_data< T, T > ** coarser = &data->coarser_level;
			assert( *coarser == nullptr );
			
			// generate linked list of hierarchical coarseners
			while( coarsening_level  < max_levels ) {
				assert( *coarser == nullptr );
				
				coarser_size = in.matAbuffer[ coarsening_level + 1 ].get_n();

				// build data structures for new level
				grb::algorithms::multi_grid_data< double, double > * new_coarser { new grb::algorithms::multi_grid_data< double, double >( coarser_size, previous_size ) };
				
				// install coarser level immediately to cleanup in case of build error
				*coarser = new_coarser;

				// initialize coarsener matrix, system matrix and diagonal vector for the coarser level
				{
					rc = buildMatrixUnique( new_coarser->coarsening_matrix,
											in.matRbuffer[ coarsening_level ].i_data,
											in.matRbuffer[ coarsening_level ].j_data,
											in.matRbuffer[ coarsening_level ].v_data,
											in.matRbuffer[ coarsening_level ].size(),
											SEQUENTIAL
											);

					if( rc != SUCCESS ) {
						std::cerr << "Failure: call to buildMatrixUnique did not succeed "
								  << "(" << toString( rc ) << ")." << std::endl;
						return rc;
					}
#ifdef HPCG_PRINT_STEPS
					std::cout << " buildMatrixUnique: constructed new_coarser->coarsening_matrix "
							  << nrows(new_coarser->coarsening_matrix) << " x " << ncols(new_coarser->coarsening_matrix) << " matrix \n";
#endif
				}
				{
					rc = buildMatrixUnique( new_coarser->A,
											in.matAbuffer[ coarsening_level + 1 ].i_data,
											in.matAbuffer[ coarsening_level + 1 ].j_data,
											in.matAbuffer[ coarsening_level + 1 ].v_data,
											in.matAbuffer[ coarsening_level + 1 ].size(),
											SEQUENTIAL
											);

					if( rc != SUCCESS ) {
						std::cerr << "Failure: call to buildMatrixUnique did not succeed "
								  << "(" << toString( rc ) << ")." << std::endl;
						return rc;
					}
#ifdef HPCG_PRINT_STEPS
					std::cout << " buildMatrixUnique: constructed new_coarser->A "
							  << nrows(new_coarser->A) << " x " << ncols(new_coarser->A) << " matrix \n";
#endif
				}
				
				if( coarsening_level + 1 < max_levels ) {
					RC rc = grb::buildVector( new_coarser->A_diagonal,
											  in.matMbuffer[ coarsening_level + 1 ].v_data,
											  in.matMbuffer[ coarsening_level + 1 ].v_data + in.matMbuffer[ coarsening_level + 1 ].size(),
											  SEQUENTIAL );
					if ( rc != SUCCESS ) {
						std::cerr << " buildVector failed!\n ";
						return rc;
					}
#ifdef HPCG_PRINT_STEPS
					std::cout << " buildVector: new_coarser->A_diagonal "
							  << size(new_coarser->A_diagonal) << " vector \n";
#endif
				}
				
				// prepare for new iteration
				coarser = &new_coarser->coarser_level;
				previous_size = coarser_size;
				coarsening_level++;
			}
			return rc;
		}

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_SYSTEM_BUILDING_UTILS
