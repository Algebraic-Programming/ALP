
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
 * Data structures to store AMG input/output data.
 * @file amg_data.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @author Denis Jelovina (denis.jelovina@huawei.com)
 * @date 2022-10-08
 */

#ifndef _H_GRB_ALGORITHMS_AMG_DATA
#define _H_GRB_ALGORITHMS_AMG_DATA

#include <vector>
#include <cstddef>
#include <graphblas.hpp>

namespace grb {

	namespace algorithms {

		/**
		 * basic data container for the AMG algorithm, storing \b only the
		 * data in common between the full CG run and the V-cycle multi-grid solver.
		 * Additional data are stored in inheriting data structures.
		 *
		 * @tparam IOType type of values of the vectors for intermediate results
		 * @tparam NonzeroType type of the values stored inside the system matrix #A
		 */
		template< typename IOType, typename NonzeroType >
		struct system_data {
			const std::size_t system_size; ///< size of the system, i.e. side of the #A
			grb::Matrix< NonzeroType > A;                   ///< system matrix
			grb::Vector< IOType > A_diagonal;               ///< vector with the diagonal of #A
			grb::Vector< IOType > z;                        ///< multi-grid solution
			grb::Vector< IOType > r;                        ///< residual
			grb::Vector< IOType > smoother_temp;            ///< for smoother's intermediate results
			std::vector< grb::Vector< bool > > color_masks; ///< for color masks

			/**
			 * Constructor building all the stored vectors and matrices.
			 *
			 * Stored vectors and matrices are constructed according to
			 * \p sys_size but \b not initialized
			 * to any value internally, as initialization is up to users's code.
			 *
			 * @param[in] sys_size the size of the underlying physical system,
			 *                      i.e. the size of vectors and the number
			 *                      of rows and columns of the #A matrix.
			 */
			system_data( std::size_t sys_size ) :
				system_size( sys_size ), A( sys_size, sys_size ), A_diagonal( sys_size ), z( sys_size ), r( sys_size ),
				smoother_temp( sys_size ) {}
			// for safety, disable copy semantics
			system_data( const system_data & o ) = delete;
			system_data & operator=( const system_data & ) = delete;
		};

		/**
		 * Data container for all multi-grid inputs and outputs.
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
		 * - the next level of coarsening, pointed to by #coarser_level,
		 *        possibly being \c nullptr if no further coarsening is desired;
		 *        note that this information is automatically
		 *        destructed on object destruction (if any)
		 *
		 * Vectors stored here refer to the \b coarsened system
		 * (with the exception of #Ax_finer),
		 * thus having size #system_size; this also holds for the system matrix #A,
		 * while #coarsening_matrix has size #system_size \f$ \times \f$ #finer_size.
		 * Hence, the typical usage of this data structure is to coarsen \b external
		 * vectors, e.g. vectors coming from another
		 * \code multi_grid_data<IOType, NonZeroType>
		 * \endcode object whose #system_size equals
		 * \code this-> \endcode #fines_size, via
		 * \code this-> \endcode #coarsening_matrix and store the coarsened
		 * vectors internally. Mimicing the recursive behavior of standard multi-grid
		 * simulations, the information for a further coarsening is stored inside
		 * #coarser_level, so that the  hierarchy of coarsened levels is reflected
		 * inside this data structure.
		 *
		 * As for \ref system_data, internal vectors and matrices are initialized to the
		 * proper size, but their values are \b not initialized.
		 */
		template< typename IOType, typename NonzeroType >
		struct multi_grid_data : public system_data< IOType, NonzeroType > {

			const std::size_t finer_size; ///< ssize of the finer system to coarse from;
			grb::Vector< IOType > Ax_finer; ///< finer vector for intermediate computations, of size #finer_size
			grb::Matrix< NonzeroType > coarsening_matrix; ///< matrix of size #system_size \f$ \times \f$ #finer_size
			                                              ///< to coarsen an input vector of size #finer_size
			                                              ///< into a vector of size #system_size
			struct multi_grid_data< IOType, NonzeroType > *coarser_level; ///< pointer to next coarsening level, for recursive
			                                                               ///< multi-grid V cycle implementations

			/**
			 * Construct a new \c multi_grid_data_object by initializing internal
			 *     data structures and setting #coarser_level to \c nullptr.
			 * @param[in] coarser_size size of the current system,
			 *            i.e. size \b after coarsening
			 * @param[in] _finer_size  size of the finer system,
			 *            i.e. size of external objects \b before coarsening
			 */
			multi_grid_data( std::size_t coarser_size, std::size_t _finer_size ) :
				system_data< IOType, NonzeroType >( coarser_size ), finer_size( _finer_size ),
				Ax_finer( finer_size ), coarsening_matrix( coarser_size, finer_size ) {
				coarser_level = nullptr;
			}

			/**
			 * @brief Destroys the \c multi_grid_data_object object
			 * by destroying #coarser_level.
			 */
			virtual ~multi_grid_data() {
				if( coarser_level != nullptr ) {
					delete coarser_level;
				}
			}
		};

		/**
		 * @brief Data stucture to store the data for a full AMG run:
		 * system vectors and matrix, coarsening information and temporary vectors.
		 *
		 * This data structures contains all the needed vectors and matrices to
		 * solve a linear system \f$ A x = b \f$. As for \ref system_data,
		 * internal elements are built and their sizes properly initialized
		 * to #system_size, but internal values are \b not initialized, as
		 * they are left to user's logic. Similarly, the coarsening information
		 * in #coarser_level is to be initialized by users by properly
		 * building a \code multi_grid_data<IOType, NonzeroType> \endcode object
		 * and storing its pointer into #coarser_level; on destruction, #coarser_level
		 * will also be properly destroyed without user's intervention.
		 *
		 * @tparam IOType type of values of the vectors for intermediate results
		 * @tparam NonzeroType type of the values stored inside the system matrix #A
		 * @tparam InputType type of the values of the right-hand side vector #b
		 */
		template< typename IOType, typename NonzeroType, typename InputType >
		struct amg_data : public system_data< IOType, NonzeroType > {

			grb::Vector< InputType > b; ///< right-side vector of known values
			grb::Vector< IOType > u;    ///< temporary vectors (typically for CG exploration directions)
			grb::Vector< IOType > p;    ///< temporary vector (typically for x refinements coming from the multi-grid run)
			grb::Vector< IOType > x;    // system solution being refined over the iterations: it us up to the user
			///< to set the initial solution value

			struct multi_grid_data< IOType, NonzeroType > *coarser_level; ///< information about the coarser system, for
			                                                               ///< the multi-grid run

			/**
			 * Construct a new \c amg_data object by building vectors and matrices
			 * and by setting #coarser_level to \c nullptr (i.e. no coarser level is assumed).
			 *
			 * @param[in] sys_size the size of the simulated system,
			 *            i.e. of all the internal vectors and matrices
			 */
			amg_data( std::size_t sys_size ) : system_data< IOType, NonzeroType >( sys_size ), b( sys_size ), u( sys_size ), p( sys_size ), x( sys_size ) {
				coarser_level = nullptr;
			}

			/**
			 * Destroy the \c amg_data object by destroying the #coarser_level informartion,
			 *  if any.
			 */
			virtual ~amg_data() {
				if( coarser_level != nullptr ) {
					delete coarser_level;
				}
			}
		};

	} // namespace algorithms

} // namespace grb

#endif // _H_GRB_ALGORITHMS_AMG_DATA
