
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
#include <type_traits>
#include <memory>
#include <utility>

#include <graphblas.hpp>

#include <graphblas/utils/iterators/IteratorValueAdaptor.hpp>

#include "multigrid_data.hpp"

namespace grb {
	namespace algorithms {
		/**
		 * @brief Namespace for interfaces that should not be used outside of the algorithm namespace.
		 */
		namespace internal {



		} // namespace internal

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
		 * @param[in,out] data \ref multigrid_data object storing the relevant data for the multi-grid run of the current
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
		template<
			typename IOType,
			typename NonzeroType,
			typename MGSysIterType,
			typename MGSmootherType,
			typename CoarsenerType,
			class Ring,
			class Minus
		> grb::RC multi_grid(
			MGSysIterType mgiter_begin,
			const MGSysIterType mgiter_end,
			MGSmootherType &smoother,
			CoarsenerType &coarsener,
			const Ring &ring,
			const Minus &minus
		) {
			static_assert( std::is_base_of< multigrid_data< IOType, NonzeroType >,
				typename std::decay< decltype( *mgiter_begin ) >::type >::value, "the iterator type MGSysIterType"
				" must reference an object of type multigrid_data< IOType, NonzeroType >" );

			RC ret { SUCCESS };
			assert( mgiter_begin != mgiter_end );
			multigrid_data< IOType, NonzeroType > &finer_system = *mgiter_begin;
			++mgiter_begin;

#ifdef HPCG_PRINT_STEPS
			DBG_println( "mg BEGINNING {" );
#endif


			// clean destination vector
			ret = ret ? ret : grb::set( finer_system.z, 0 );
#ifdef HPCG_PRINT_STEPS
			DBG_print_norm( finer_system.r, "initial r" );
#endif
			if( !( mgiter_begin != mgiter_end ) ) {
				// compute one round of Gauss Seidel and return
				ret = ret ? ret : smoother.nonrecursive_smooth( finer_system );
				assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( finer_system.z, "smoothed z" );
				DBG_println( "} mg END" );
#endif
				return ret;
			}
			multigrid_data< IOType, NonzeroType > &coarser_system = *mgiter_begin;

			// pre-smoother
			ret = ret ? ret : smoother.pre_smooth( finer_system );
			assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
			DBG_print_norm( finer_system.z, "pre-smoothed z" );
#endif

			ret = ret ? ret : coarsener.coarsen_residual( finer_system, coarser_system );
			assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
			DBG_print_norm( coarser_system.r, "coarse r" );
#endif

			ret = ret ? ret : multi_grid< IOType, NonzeroType, MGSysIterType,
				MGSmootherType, CoarsenerType, Ring, Minus >( mgiter_begin, mgiter_end,
				smoother, coarsener, ring, minus );
			assert( ret == SUCCESS );

			ret = ret ? ret : coarsener.prolong_solution( coarser_system, finer_system );
			assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
			DBG_print_norm( finer_system.z, "prolonged z" );
#endif

			// post-smoother
			ret = ret ? ret : smoother.post_smooth( finer_system );
			assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
			DBG_print_norm( finer_system.z, "post-smoothed z" );
			DBG_println( "} mg END" );
#endif

			return ret;
		}

		template<
			typename IOType,
			typename NonzeroType,
			typename InputType,
			typename MGSmootherType,
			typename CoarsenerType,
			class Ring,
			class Minus
		> struct multigrid_runner {

			static_assert( std::is_default_constructible< Ring >::value,
				"cannot construct the Ring with default values" );
			static_assert( std::is_default_constructible< Minus >::value,
				"cannot construct the Minus operator with default values" );
			static_assert( std::is_move_constructible< MGSmootherType >::value,
				"MGSmootherType must be move-constructible");
			static_assert( std::is_move_constructible< CoarsenerType >::value,
				"CoarsenerType must be move-constructible");

			using MultiGridInputType = multigrid_data< IOType, NonzeroType >;

			// check the interface between HPCG and MG match
			static_assert( std::is_base_of< typename MGSmootherType::SmootherInputType,
				MultiGridInputType >::value, "input type of the Smoother kernel must match the input from Multi-Grid" );

			MGSmootherType smoother_runner;
			CoarsenerType coarsener_runner;
			std::vector< std::unique_ptr< MultiGridInputType > > system_levels;
			Ring ring;
			Minus minus;

			struct Extractor {
				MultiGridInputType & operator()(
					typename std::vector< std::unique_ptr< MultiGridInputType > >::reference &ref
				) {
					return *ref.get();
				}

				const MultiGridInputType & operator()(
					typename std::vector< std::unique_ptr< MultiGridInputType > >::const_reference &ref
				) const {
					return *ref.get();
				}
			};

			using UniquePtrExtractor = grb::utils::IteratorValueAdaptor<
				typename std::vector< std::unique_ptr< MultiGridInputType > >::iterator,
				Extractor
			>;


			multigrid_runner(
				MGSmootherType &&_smoother_runner,
				CoarsenerType &&_coarsener_runner
			) : smoother_runner( std::move( _smoother_runner ) ),
				coarsener_runner( std::move(  _coarsener_runner ) ) {}

			inline grb::RC operator()(
				MultiGridInputType &system
			) {
				return multi_grid< IOType, NonzeroType, UniquePtrExtractor, MGSmootherType, CoarsenerType, Ring, Minus >(
					UniquePtrExtractor( system_levels.begin() += system.level ), UniquePtrExtractor( system_levels.end() ),
					smoother_runner, coarsener_runner, ring, minus );
			}
		};

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_V_CYCLE
