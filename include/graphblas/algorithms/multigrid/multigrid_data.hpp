
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
 * @file multigrid_data.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Data structure definition to store the information of a single multi-grid level.
 */

#ifndef _H_GRB_ALGORITHMS_HPCG_DATA
#define _H_GRB_ALGORITHMS_HPCG_DATA

#include <vector>
#include <cstddef>

#include <graphblas.hpp>


namespace grb {

	namespace algorithms {

		/**
		 * This data structure stores information for a \b single multi-grid level. This information
		 * dependes exclusively on the size of the underlying physical system.

		 *
		 * Internal ALP/GraphBLAS containers are initialized to the proper size,
		 * but their values are \b not initialized as this depends on the specific algorithm chosen
		 * for the multi-grid solver. Populating them is user's task.
		 *
		 * @tparam IOType Type of values of the vectors for intermediate results
		 * @tparam NonzeroType Type of the values stored inside the system matrix \p A
		 *                     and the coarsening matrix #Ax_finer
		 */
		template<
			typename IOType,
			typename NonzeroType
		> struct MultiGridData {

			const size_t level; ///< level of the grid (0 for the finest physical system)
			const size_t system_size; ///< size of the system, i.e. side of the #A system matrix
			grb::Matrix< NonzeroType > A; ///< system matrix
			grb::Vector< IOType > z; ///< multi-grid solution
			grb::Vector< IOType > r; ///< residual

			/**
			 * Construct a new multigrid data object from level information and system size.
			 */
			MultiGridData(
				size_t _level,
				size_t sys_size
			) :
				level( _level ),
				system_size( sys_size ),
				A( sys_size, sys_size ),
				z( sys_size ),
				r( sys_size ) {}

			// for safety, disable copy semantics
			MultiGridData( const MultiGridData< IOType, NonzeroType > & o ) = delete;

			MultiGridData<IOType, NonzeroType > & operator=( const MultiGridData< IOType, NonzeroType > & ) = delete;

			grb::RC init_vectors( IOType zero ) {
				grb::RC rc = grb::set( z, zero );
				rc = rc ? rc : grb::set( r, zero );
				return rc;
			}
		};

	} // namespace algorithms

} // namespace grb

#endif // _H_GRB_ALGORITHMS_HPCG_DATA

