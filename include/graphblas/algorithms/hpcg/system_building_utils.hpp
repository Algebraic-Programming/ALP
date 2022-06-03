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
		template< std::size_t DIMS, typename T = double, typename SYSINP >
		grb::RC build_hpcg_system( std::unique_ptr< grb::algorithms::hpcg_data< T, T, T > > & holder, hpcg_system_params< DIMS, T > & params, SYSINP &in ) {
			grb::RC rc { grb::SUCCESS };
			std::size_t coarsening_level = 0UL;
			grb::utils::MatrixFileReader< T, std::conditional<
				( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ),
												 grb::config::RowIndexType,
												 grb::config::ColIndexType >::type
										  > parser_A( in.matAfiles[coarsening_level].c_str(), true );
			const size_t n_A = parser_A.n();
			grb::algorithms::hpcg_data< T, T, T > * data { new grb::algorithms::hpcg_data< T, T, T >( n_A ) };
			rc = buildMatrixUnique( data->A,
									parser_A.begin( SEQUENTIAL ), parser_A.end( SEQUENTIAL),
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
			
			assert( ! holder ); // should be empty
			holder = std::unique_ptr< grb::algorithms::hpcg_data< T, T, T > >( data );

			{
				T *buffer;
				std::ifstream inFile;
				inFile.open(in.matMfiles[ coarsening_level ]);
				if (inFile.is_open())  	{
					size_t n=n_A;
					buffer = new T [ n ];
					for (size_t i = 0; i < n; i++) {
						inFile >> buffer[i];
					}
						
					inFile.close(); // CLose input file
				
					RC rc = grb::buildVector( data->A_diagonal, buffer, buffer + n, SEQUENTIAL );
					if ( rc != SUCCESS ) {
						std::cerr << " buildVector failed!\n ";
						return rc;
					}
					delete [] buffer;
				}
				/* Once internal issue #342 is resolved this can be re-enabled
				   const RC rc = buildMatrixUnique( L,
				   parser.begin( PARALLEL ), parser.end( PARALLEL),
				   PARALLEL
				   );*/
			}

			std::size_t coarser_size;
			std::size_t previous_size=n_A;
			
			// initialize coarsening with additional pointers and dimensions copies to iterate and divide
			grb::algorithms::multi_grid_data< T, T > ** coarser = &data->coarser_level;
			assert( *coarser == nullptr );
			
			// generate linked list of hierarchical coarseners
			while( coarsening_level  < params.max_levels ) {
				assert( *coarser == nullptr );
				
				grb::utils::MatrixFileReader< T, std::conditional<
					( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ),
													 grb::config::RowIndexType,
													 grb::config::ColIndexType >::type
											  > parser_A_next( in.matAfiles[ coarsening_level + 1 ].c_str(), true );
				
				coarser_size = parser_A_next.n();

				// build data structures for new level
				grb::algorithms::multi_grid_data< double, double > * new_coarser { new grb::algorithms::multi_grid_data< double, double >( coarser_size, previous_size ) };
				
				// install coarser level immediately to cleanup in case of build error
				*coarser = new_coarser;
				// initialize coarsener matrix, system matrix and diagonal vector for the coarser level

				
				{
					grb::utils::MatrixFileReader< T, std::conditional<
						( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ),
														 grb::config::RowIndexType,
														 grb::config::ColIndexType >::type
												  > parser_P( in.matRfiles[coarsening_level].c_str(), true );
					
					rc = buildMatrixUnique( new_coarser->coarsening_matrix,
											parser_P.begin( SEQUENTIAL ), parser_P.end( SEQUENTIAL),
											SEQUENTIAL
											);

					if( rc != SUCCESS ) {
						std::cerr << "Failure: call to buildMatrixUnique did not succeed "
								  << "(" << toString( rc ) << ")." << std::endl;
						return rc;
					}
					
				}
				{
					rc = buildMatrixUnique( new_coarser->A,
											parser_A_next.begin( SEQUENTIAL ), parser_A_next.end( SEQUENTIAL),
											SEQUENTIAL
											);

					if( rc != SUCCESS ) {
						std::cerr << "Failure: call to buildMatrixUnique did not succeed "
								  << "(" << toString( rc ) << ")." << std::endl;
						return rc;
					}					
					
				}				

				//set( new_coarser->A_diagonal, params.diag_value );
				if( coarsening_level + 1 < params.max_levels ) {
					T *buffer;
					std::ifstream inFile;
					inFile.open(in.matMfiles[ coarsening_level + 1 ]);
					if (inFile.is_open())  	{
						size_t n=grb::nrows(new_coarser->A);
						buffer = new T [ n ];
						for (size_t i = 0; i < n; i++) {
							inFile >> buffer[i];
						}
						
						inFile.close(); // CLose input file
				
						RC rc = grb::buildVector( new_coarser->A_diagonal, buffer, buffer + n, SEQUENTIAL );
						if ( rc != SUCCESS ) {
							std::cerr << " buildVector failed!\n ";
							return rc;
						}
						delete [] buffer;
					}

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
