
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
 * @file multigrid_v_cycle.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @brief This file contains the routines for multi-grid solution refinement, including the main routine
 *        and those for coarsening and refinement of the tentative solution.
 * @date 2021-04-30
 */

#ifndef _H_GRB_ALGORITHMS_MULTIGRID_V_CYCLE
#define _H_GRB_ALGORITHMS_MULTIGRID_V_CYCLE

#include <cassert>
#include <vector>

#include <graphblas.hpp>

#include "hpcg_data.hpp"
#include "red_black_gauss_seidel.hpp"


namespace grb {
	namespace algorithms {
		/**
		 * @brief Namespace for interfaces that should not be used outside of the algorithm namespace.
		 */
		namespace internal {

			/**
			 * @brief computes the coarser residual vector \p coarsening_data.r by coarsening
			 *        \p coarsening_data.Ax_finer - \p r_fine via \p coarsening_data.coarsening_matrix.
			 *
			 * The coarsening information are stored inside \p coarsening_data.
			 *
			 * @tparam IOType type of result and intermediate vectors used during computation
			 * @tparam NonzeroType type of matrix values
			 * @tparam Ring the ring of algebraic operators zero-values
			 * @tparam Minus the minus operator for subtractions
			 *
			 * @param[in] r_fine fine residual vector
			 * @param[in,out] coarsening_data \ref multi_grid_data data structure storing the information for coarsening
			 * @param[in] ring the ring to perform the operations on
			 * @param[in] minus the \f$ - \f$ operator for vector subtractions
			 * @return grb::RC::SUCCESS if the algorithm could correctly terminate, the error code of the first
			 *                          unsuccessful operation otherwise
			 */
			template< typename IOType,
				typename NonzeroType,
				class Ring,
				class Minus >
			grb::RC compute_coarsening( const grb::Vector< IOType > & r_fine, // fine residual
				struct multi_grid_data< IOType, NonzeroType > & coarsening_data,
				const Ring & ring,
				const Minus & minus ) {
				RC ret { SUCCESS };
				ret = ret ? ret : grb::eWiseApply( coarsening_data.Ax_finer, r_fine, coarsening_data.Ax_finer,
									  minus ); // Ax_finer = r_fine - Ax_finer
				assert( ret == SUCCESS );

				// actual coarsening, from  ncols(*coarsening_data->A) == *coarsening_data->system_size * 8
				// to *coarsening_data->system_size
				ret = ret ? ret : grb::set( coarsening_data.r, 0 );
				ret = ret ? ret : grb::mxv( coarsening_data.r, coarsening_data.coarsening_matrix, coarsening_data.Ax_finer,
									  ring ); // r = coarsening_matrix * Ax_finer
				return ret;
			}

			/**
			 * @brief computes the prolongation of the coarser solution \p coarsening_data.z and stores it into
			 * \p x_fine.
			 *
			 * For prolongation, this function uses the matrix \p coarsening_data.coarsening_matrix by transposing it.
			 *
			 * @tparam IOType type of result and intermediate vectors used during computation
			 * @tparam NonzeroType type of matrix values
			 * @tparam Ring the ring of algebraic operators zero-values
			 *
			 * @param[out] x_fine the solution vector to store the prolonged solution into
			 * @param[in,out] coarsening_data information for coarsening
			 * @param[in] ring the ring to perform the operations on
			 * @return grb::RC::SUCCESS if the algorithm could correctly terminate, the error code of the first
			 * unsuccessful operation otherwise
			 */
			template< typename IOType,
				typename NonzeroType,
				class Ring >
			grb::RC compute_prolongation( grb::Vector< IOType > & x_fine, // fine residual
				struct multi_grid_data< IOType, NonzeroType > & coarsening_data,
				const Ring & ring ) {
				RC ret { SUCCESS };
				// actual refining, from  *coarsening_data->syztem_size == nrows(*coarsening_data->A) / 8
				// to nrows(x_fine)
				ret = ret ? ret : set( coarsening_data.Ax_finer, 0 );

				ret = ret ? ret : grb::mxv< grb::descriptors::transpose_matrix >( coarsening_data.Ax_finer, coarsening_data.coarsening_matrix, coarsening_data.z, ring );
				assert( ret == SUCCESS );

				ret = ret ? ret : grb::foldl( x_fine, coarsening_data.Ax_finer, ring.getAdditiveMonoid() ); // x_fine += Ax_finer;
				assert( ret == SUCCESS );
				return ret;
			}

			/**
			 * @brief Runs \p smoother_steps iteration of the Red-Black Gauss-Seidel smoother, with inputs and outputs stored
			 * inside \p data.
			 *
			 * @tparam IOType type of result and intermediate vectors used during computation
			 * @tparam NonzeroType type of matrix values
			 * @tparam Ring the ring of algebraic operators zero-values
			 *
			 * @param[in,out] data \ref system_data data structure with relevant inpus and outputs: system matrix, initial solution,
			 *                     residual, system matrix colors, temporary vectors
			 * @param[in] smoother_steps how many smoothing steps to run
			 * @param[in] ring the ring to perform the operations on
			 * @return grb::RC::SUCCESS if the algorithm could correctly terminate, the error code of the first
			 *                          unsuccessful operation otherwise
			 */
			template< typename IOType, typename NonzeroType, class Ring >
			grb::RC run_smoother( system_data< IOType, NonzeroType > & data, const std::size_t smoother_steps, const Ring & ring ) {
				RC ret { SUCCESS };

				for( std::size_t i { 0 }; i < smoother_steps && ret == SUCCESS; i++ ) {
					ret = ret ? ret : red_black_gauss_seidel( data, ring );
					assert( ret == SUCCESS );
				}
				return ret;
			}

			/**
			 * @brief Multi-grid V cycle implementation to refine a given solution.
			 *
			 * A full multi-grid run goes through the following steps:
			 * -# if \p presmoother_steps \f$ > 0 \f$, \p presmoother_steps of the Red-Black Gauss-Seidel smoother are run
			 *    to improve on the initial solution stored into \p data.z
			 * -# the coarsening of \f$ r - A*z \f$ is computed to find the coarser residual vector
			 * -# a multi-grid run is recursively performed on the coarser system
			 * -# the tentative solution from the coarser multi-grid run is prolonged and added to the current tentative solution
			 *    into \p data.z
			 * -# this solution is further smoothed for \p postsmoother_steps steps
			 *
			 * If coarsening information is not available, the multi-grid run consists in a single smmothing run.
			 *
			 * Failuers of GraphBLAS operations are handled by immediately stopping the execution and by returning
			 * the failure code.
			 *
			 * @tparam IOType type of result and intermediate vectors used during computation
			 * @tparam NonzeroType type of matrix values
			 * @tparam Ring the ring of algebraic operators zero-values
			 * @tparam Minus the minus operator for subtractions
			 *
			 * @param[in,out] data \ref multi_grid_data object storing the relevant data for the multi-grid run of the current
			 *                     clevel
			 * @param[in,out] coarsening_data pointer to information for the coarsening/refinement operations and for the
			 *                recursive multi-grid run on the coarsened system; if \c nullptr, no coarsening/refinement occurs
			 *                and only smoothing occurs on the current solution
			 * @param[in] presmoother_steps number of pre-smoother steps
			 * @param[in] postsmoother_steps number of post-smoother steps
			 * @param[in] ring the ring to perform the operations on
			 * @param[in] minus the \f$ - \f$ operator for vector subtractions
			 * @return grb::RC::SUCCESS if the algorithm could correctly terminate, the error code of the first
			 *                          unsuccessful operation otherwise
			 */
			template< typename IOType, typename NonzeroType, class Ring, class Minus >
			grb::RC multi_grid( system_data< IOType, NonzeroType > & data,
				struct multi_grid_data< IOType, NonzeroType > * const coarsening_data,
				const size_t presmoother_steps,
				const size_t postsmoother_steps,
				const Ring & ring,
				const Minus & minus ) {
				RC ret { SUCCESS };
#ifdef HPCG_PRINT_STEPS
				DBG_println( "mg BEGINNING {" );
#endif

				// clean destination vector
				ret = ret ? ret : grb::set( data.z, 0 );
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( data.r, "initial r" );
#endif
				if( coarsening_data == nullptr ) {
					// compute one round of Gauss Seidel and return
					ret = ret ? ret : run_smoother( data, 1, ring );
					assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
					DBG_print_norm( data.z, "smoothed z" );
					DBG_println( "} mg END" );
#endif
					return ret;
				}

				struct multi_grid_data< IOType, NonzeroType > & cd {
					*coarsening_data
				};

				// pre-smoother
				ret = ret ? ret : run_smoother( data, presmoother_steps, ring );
				assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( data.z, "pre-smoothed z" );
#endif

				ret = ret ? ret : grb::set( cd.Ax_finer, 0 );
				ret = ret ? ret : grb::mxv( cd.Ax_finer, data.A, data.z, ring );
				assert( ret == SUCCESS );

				ret = ret ? ret : compute_coarsening( data.r, cd, ring, minus );
				assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( cd.r, "coarse r" );
#endif

				ret = ret ? ret : multi_grid( cd, cd.coarser_level, presmoother_steps, postsmoother_steps, ring, minus );
				assert( ret == SUCCESS );

				ret = ret ? ret : compute_prolongation( data.z, cd, ring );
				assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( data.z, "prolonged z" );
#endif

				// post-smoother
				ret = ret ? ret : run_smoother( data, postsmoother_steps, ring );
				assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( data.z, "post-smoothed z" );
				DBG_println( "} mg END" );
#endif

				return ret;
			}

		} // namespace internal
	}     // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_V_CYCLE
