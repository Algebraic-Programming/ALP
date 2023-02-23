
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
 * @file multigrid_building_utils.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Utilities to allocate data for an entire multi-grid simulation.
 */

#ifndef _H_GRB_ALGORITHMS_MULTIGRID_BUILDING_UTILS
#define _H_GRB_ALGORITHMS_MULTIGRID_BUILDING_UTILS

#include <cstddef>
#include <memory>
#include <vector>

namespace grb {
	namespace algorithms {

		/**
		 * Allocates all the levels for an entire multi-grid simulation for the multi-grid v-cycle,
		 * the coarsener and the smoother. This routine just allocates and initializes the data structures,
		 * but does \b not populate them, which depends on the specific algorithms.
		 *
		 * Thanks to the templating, this routine is meant to be independent from the specific algorithm
		 * choosen for the simulation, but simply implements the logic to move from one level (finer)
		 * to the next one (coarser). To be used with any data structure, the constructor of each
		 * structure must meet a certain interface, as explained in the following.
		 *
		 * Note: structures are allocated on the heap and manged via an std::unique_ptr for efficiency
		 * and convenience: since they may store large data amounts, moving them via their move (copy)
		 * constructor (as required for the growth of an std::vector) may be costly, and forces the user
		 * to implement the move constructor for each type (which may be annoying).
		 * Furthermore, avoiding movement (copy) entirely protects against possible bugs
		 * in move (copy)-constructor logic (not uncommon in prototypes).
		 *
		 * @tparam MGInfoType type holding the information to run the chosen multi-grid algorithm:
		 * 	its constructor must take in input the coarsening level (0 to \p mg_sizes.size() )
		 *  and the size of the system matrix for that level
		 * @tparam CoarsenerInfoType type holding the information for the coarsener;
		 *  its constructor must take in input the size of the finer system matrix and that of
		 *  the coarser system matrix (in this order)
		 * @tparam SmootherInfoType type holding the information for the smoother;
		 *  its constructor must take in input the size of the system matrix for that level
		 * @tparam TelControllerType telemetry controller type, to (de)activate time measurement at compile-time
		 *
		 * @param mg_sizes sizes of the system matrix for each level of the multi-grid
		 * @param system_levels system data (system matrix, residual, solution, ...) for each level
		 * @param coarsener_levels at position \a i of this vector, data to coarsen from level \a i
		 *  (system size \p mg_sizes [i] ) to level \a i+1 (system size \p mg_sizes [i+1] )
		 * @param smoother_levels smoother data for each level
		 * @param tt telemetry controller to control time tracing
		 */
		template<
			typename MGInfoType,
			typename CoarsenerInfoType,
			typename SmootherInfoType,
			typename TelControllerType
		> void multigrid_allocate_data(
			std::vector< std::unique_ptr< MGInfoType > > & system_levels,
			std::vector< std::unique_ptr< CoarsenerInfoType > > & coarsener_levels,
			std::vector< std::unique_ptr< SmootherInfoType > > & smoother_levels,
			const std::vector< size_t > & mg_sizes,
			const TelControllerType & tt
		) {
			if( mg_sizes.size() == 0 ) {
				throw std::invalid_argument( "at least one size should be available" );
			}
			size_t finer_size = mg_sizes[ 0 ];
			system_levels.emplace_back( new MGInfoType( tt, 0, finer_size ) );  // create main system
			smoother_levels.emplace_back( new SmootherInfoType( finer_size ) ); // create smoother for main
			for( size_t i = 1; i < mg_sizes.size(); i++ ) {
				size_t coarser_size = mg_sizes[ i ];
				if( coarser_size >= finer_size ) {
					throw std::invalid_argument( "system sizes not monotonically decreasing" );
				}
				coarsener_levels.emplace_back( new CoarsenerInfoType( finer_size, coarser_size ) );
				system_levels.emplace_back( new MGInfoType( tt, i, coarser_size ) );
				smoother_levels.emplace_back( new SmootherInfoType( coarser_size ) );
				finer_size = coarser_size;
			}
		}

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_BUILDING_UTILS
