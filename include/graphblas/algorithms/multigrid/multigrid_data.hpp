
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
 * @file hpcg_data.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @brief Data structures to store HPCG input/output data.
 * @date 2021-04-30
 */

#ifndef _H_GRB_ALGORITHMS_HPCG_DATA
#define _H_GRB_ALGORITHMS_HPCG_DATA

#include <vector>
#include <cstddef>

#include <graphblas.hpp>


namespace grb {

	namespace algorithms {

		/**
		 * @brief Data container for all multi-grid inputs and outputs.
		 *
		 * @tparam IOType Type of values of the vectors for intermediate results
		 * @tparam NonzeroType Type of the values stored inside the system matrix \p A
		 *                     and the coarsening matrix #Ax_finer
		 *
		 * This data structure stores information for a full multi-grid V cycle, i.e.
		 * - input and output vectors for solution, residual and temporary vectors
		 * - coarsening information, in particular the #coarsening_matrix that
		 *   coarsens a larger system of size #finer_size to the current system
		 *   of size #system_size
		 * - the next level of coarsening, pointed to by #coarser_level, possibly being \c nullptr
		 *   if no further coarsening is desired; note that this information is automatically
		 *   destructed on object destruction (if any)
		 *
		 * Vectors stored here refer to the \b coarsened system (with the exception of #Ax_finer),
		 * thus having size #system_size; this also holds for the system matrix #A,
		 * while #coarsening_matrix has size #system_size \f$ \times \f$ #finer_size.
		 * Hence, the typical usage of this data structure is to coarsen \b external vectors, e.g. vectors
		 * coming from another \code multigrid_data<IOType, NonzeroType> \endcode object whose #system_size equals
		 * \code this-> \endcode #fines_size, via \code this-> \endcode #coarsening_matrix and store the coarsened
		 * vectors internally. Mimicing the recursive behavior of standard multi-grid simulations,
		 * the information for a further coarsening is stored inside #coarser_level, so that the
		 * hierarchy of coarsened levels is reflected inside this data structure.
		 *
		 * As for \ref system_data, internal vectors and matrices are initialized to the proper size,
		 * but their values are \b not initialized.
		 */
		template<
			typename IOType,
			typename NonzeroType
		> struct multigrid_data {

			const size_t level;
			const size_t system_size; ///< size of the system, i.e. side of the #A
			grb::Matrix< NonzeroType > A;                   ///< system matrix
			grb::Vector< IOType > z;                        ///< multi-grid solution
			grb::Vector< IOType > r;                        ///< residual

			multigrid_data(
				size_t _level,
				size_t sys_size
			) :
				level( _level ),
				system_size( sys_size ),
				A( sys_size, sys_size ),
				z( sys_size ),
				r( sys_size ) {}

			// for safety, disable copy semantics
			multigrid_data( const multigrid_data< IOType, NonzeroType > & o ) = delete;

			multigrid_data<IOType, NonzeroType > & operator=( const multigrid_data< IOType, NonzeroType > & ) = delete;

			grb::RC zero_temp_vectors() {
				grb::RC rc = grb::set( z, 0 );
				rc = rc ? rc : grb::set( r, 0 );
				return rc;
			}
		};

	} // namespace algorithms

} // namespace grb

#endif // _H_GRB_ALGORITHMS_HPCG_DATA

