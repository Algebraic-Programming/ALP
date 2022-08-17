
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
 * Utilities to build an antire system for AMG simulations.
 * @file amg_system_building_utils.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @author Denis Jelovina (denis.jelovina@huawei.com)
 * @date 2022-08-10
 */

#ifndef _H_GRB_ALGORITHMS_SYSTEM_BUILDING_UTILS
#define _H_GRB_ALGORITHMS_SYSTEM_BUILDING_UTILS

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <graphblas.hpp>
#include <graphblas/utils/parser.hpp>
#include "amg_data.hpp"

namespace grb {

	namespace algorithms {

		/**
		 * Generates an entire AMG problem, storing it in \p holder.
		 *
		 * @tparam DIMS dimensions of the system
		 * @tparam T type of matrix values
		 * @param holder std::unique_ptr to store the AMG problem into
		 * @param params parameters container to build the AMG problem
		 * @return grb::SUCCESS if every GraphBLAS operation (to generate vectors
		 *          and matrices) succeeded, otherwise the first unsuccessful return value
		 */
		template< typename T = double, typename SYSINP >
		grb::RC build_amg_system(
				std::unique_ptr< grb::algorithms::amg_data< T, T, T > > &holder,
				SYSINP &in
			) {
			const size_t max_levels = in.Amat_data.size() - 1;
#ifdef AMG_PRINT_STEPS
			std::cout << " in.Amat_data.size() = " << in.Amat_data.size() << "\n";
			std::cout << " in.Rmat_data.size() = " << in.Rmat_data.size() << "\n";
			std::cout << " in.Dvec_data.size() = " << in.Dvec_data.size() << "\n";
#endif
			grb::RC rc = grb::SUCCESS;
			std::size_t coarsening_level = 0UL;
			const size_t n_A = in.Amat_data[ coarsening_level ].n;
			grb::algorithms::amg_data< T, T, T > *data = new grb::algorithms::amg_data< T, T, T >( n_A );
			rc = buildMatrixUnique(
				data->A,
				in.Amat_data[ coarsening_level ].i_data.begin(),
				in.Amat_data[ coarsening_level ].j_data.begin(),
				in.Amat_data[ coarsening_level ].v_data.begin(),
				in.Amat_data[ coarsening_level ].nz,
				PARALLEL
			);

			if( rc != SUCCESS ) {
				std::cerr << "Failure: call to buildMatrixUnique did not succeed "
						  << "(" << toString( rc ) << ")." << std::endl;
				return rc;
			}
#ifdef AMG_PRINT_STEPS
			std::cout << " buildMatrixUnique: constructed data->A " << nrows( data->A )
			          << " x " << ncols( data->A ) << " matrix \n";
#endif
			assert( ! holder ); // should be empty
			holder = std::unique_ptr< grb::algorithms::amg_data< T, T, T > >( data );

			{
				// ** TODO:  build vector from AMGCL data **
				RC rc = grb::buildVector(
					data->A_diagonal,
					in.Dvec_data[ coarsening_level ].begin(),
					in.Dvec_data[ coarsening_level ].end(),
					PARALLEL
				);
				if ( rc != SUCCESS ) {
					std::cerr << " buildVector failed!\n ";
					return rc;
				}
#ifdef AMG_PRINT_STEPS
				std::cout << " buildVector: data->A_diagonal "
				          << size( data->A_diagonal ) << " vector \n";
#endif
			}

			std::size_t coarser_size;
			std::size_t previous_size = n_A;

			// initialize coarsening with additional pointers and
			// dimensions copies to iterate and divide
			grb::algorithms::multi_grid_data< T, T > **coarser = &data->coarser_level;
			assert( *coarser == nullptr );

			// generate linked list of hierarchical coarseners
			while( coarsening_level  < max_levels ) {
				assert( *coarser == nullptr );

				coarser_size = in.Amat_data[ coarsening_level + 1 ].n;

				// build data structures for new level
				grb::algorithms::multi_grid_data< double, double > *new_coarser =
					new grb::algorithms::multi_grid_data< double, double >( coarser_size, previous_size );

				// install coarser level immediately to cleanup in case of build error
				*coarser = new_coarser;

				// initialize coarsener matrix, system matrix and
				// diagonal vector for the coarser level
				{
					rc = buildMatrixUnique(
						new_coarser->coarsening_matrix,
						in.Rmat_data[ coarsening_level ].i_data.begin(),
						in.Rmat_data[ coarsening_level ].j_data.begin(),
						in.Rmat_data[ coarsening_level ].v_data.begin(),
						in.Rmat_data[ coarsening_level ].nz,
						PARALLEL
					);

					if( rc != SUCCESS ) {
						std::cerr << "Failure: call to buildMatrixUnique did not succeed "
								  << "(" << toString( rc ) << ")." << std::endl;
						return rc;
					}
#ifdef AMG_PRINT_STEPS
					std::cout << " buildMatrixUnique: constructed new_coarser->coarsening_matrix "
					          << nrows( new_coarser->coarsening_matrix ) << " x "
					          << ncols( new_coarser->coarsening_matrix ) << " matrix \n";
#endif
				}
				{
					rc = buildMatrixUnique(
						new_coarser->A,
						in.Amat_data[ coarsening_level + 1 ].i_data.begin(),
						in.Amat_data[ coarsening_level + 1 ].j_data.begin(),
						in.Amat_data[ coarsening_level + 1 ].v_data.begin(),
						in.Amat_data[ coarsening_level + 1 ].nz,
						PARALLEL
					);

					if( rc != SUCCESS ) {
						std::cerr << "Failure: call to buildMatrixUnique did not succeed "
						          << "(" << toString( rc ) << ")." << std::endl;
						return rc;
					}
#ifdef AMG_PRINT_STEPS
					std::cout << " buildMatrixUnique: constructed new_coarser->A "
							  << nrows( new_coarser->A ) << " x "
							  << ncols( new_coarser->A ) << " matrix \n";
#endif
				}

				RC rc = grb::buildVector(
					new_coarser->A_diagonal,
					in.Dvec_data[ coarsening_level + 1 ].begin(),
					in.Dvec_data[ coarsening_level + 1 ].end(),
					PARALLEL
				);

				if ( rc != SUCCESS ) {
					std::cerr << " buildVector failed!\n ";
					return rc;
				}
#ifdef AMG_PRINT_STEPS
				std::cout << " buildVector: new_coarser->A_diagonal "
				          << size(new_coarser->A_diagonal) << " vector \n";
#endif

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
