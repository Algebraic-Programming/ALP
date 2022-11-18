
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
 * @file multigrid_v_cycle.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * This file contains the routines for multi-grid solution refinement, including the main routine
 *        and those for coarsening and refinement of the tentative solution.
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
		 * Multi-grid V cycle implementation to refine a given solution.
		 *
		 * A full multi-grid run goes through the following steps:
		 *
		 * 1. calls the pre-smoother to improve on the initial solution stored into \p mgiter_begin->z
		 * 2. coarsens the residual vector
		 * 3. recursively solves the coarser system
		 * 4. prolongs the coarser solution into the \p mgiter_begin->z
		 * 5. further smooths the solution wih a post-smoother call
		 *
		 * The algorithm moves across grid levels via the STL-like iterators \p mgiter_begin
		 * and \p mgiter_end and accesses the grid data via the former (using the operator \c * ): when
		 * \p mgiter_begin \c == \p mgiter_end , a smoothing round is invoked and the recursion halted.
		 *
		 * Failuers of GraphBLAS operations are handled by immediately stopping the execution
		 * and returning the failure code.
		 *
		 * @tparam IOType type of result and intermediate vectors used during computation
		 * @tparam NonzeroType type of matrix values
		 * @tparam MGSysIterType type of the iterator across grid levels
		 * @tparam MGSmootherType type of the smoother runner, with prescribed methods for the various
		 *  smoothing steps
		 * @tparam CoarsenerType type of the coarsener runner, with prescribed methods for coarsening
		 *  and prolongation
		 * @tparam Ring the ring of algebraic operators zero-values
		 * @tparam Minus the minus operator for subtractions
		 *
		 * @param mgiter_begin iterator pointing to the current level of the multi-grid
		 * @param mgiter_end end iterator, indicating the end of the recursion
		 * @param smoother callable object to invoke the smoothing steps
		 * @param coarsener callable object to coarsen and prolong (between current and coarser grid levels)
		 * @param ring the ring to perform the operations on
		 * @param minus the \f$ - \f$ operator for vector subtractions
		 * @return grb::RC if the algorithm could correctly terminate, the error code of the first
		 *  unsuccessful operation otherwise
		 */
		template <
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
			static_assert( std::is_base_of< MultiGridData< IOType, NonzeroType >,
				typename std::decay< decltype( *mgiter_begin ) >::type >::value, "the iterator type MGSysIterType"
				" must reference an object of type MultiGridData< IOType, NonzeroType >" );

			RC ret = SUCCESS;
			assert( mgiter_begin != mgiter_end );
			MultiGridData< IOType, NonzeroType > &finer_system = *mgiter_begin;
			++mgiter_begin;

#ifdef HPCG_PRINT_STEPS
			DBG_println( "mg BEGINNING {" );
#endif

			// clean destination vector
			ret = ret ? ret : grb::set( finer_system.z, ring. template getZero< IOType >() );
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
			MultiGridData< IOType, NonzeroType > &coarser_system = *mgiter_begin;

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

		/**
		 * Callable object to invoke the V-cycle multi-grid algorithm, which also requires
		 * a smoother and a coarsener object.
		 *
		 * It is built by transferring into it the state of both the smoother and the coarsener,
		 * in order to avoid use-after-free issues.
		 *
		 * @tparam IOType type of result and intermediate vectors used during computation
		 * @tparam NonzeroType type of matrix values
		 * @tparam MGSysIterType type of the iterator across grid levels
		 * @tparam MGSmootherType type of the smoother runner, with prescribed methods for the various
		 *  smoothing steps
		 * @tparam CoarsenerType type of the coarsener runner, with prescribed methods for coarsening
		 *  and prolongation
		 * @tparam Ring the ring of algebraic operators and zero values
		 * @tparam Minus the minus operator for subtractions
		 */
		template<
			typename IOType,
			typename NonzeroType,
			typename MGSmootherType,
			typename CoarsenerType,
			class Ring,
			class Minus
		> struct MultiGridRunner {

			static_assert( std::is_default_constructible< Ring >::value,
				"cannot construct the Ring with default values" );
			static_assert( std::is_default_constructible< Minus >::value,
				"cannot construct the Minus operator with default values" );
			static_assert( std::is_move_constructible< MGSmootherType >::value,
				"MGSmootherType must be move-constructible");
			static_assert( std::is_move_constructible< CoarsenerType >::value,
				"CoarsenerType must be move-constructible");

			using MultiGridInputType = MultiGridData< IOType, NonzeroType >;

			// check the interface between HPCG and MG match
			static_assert( std::is_base_of< typename MGSmootherType::SmootherInputType,
				MultiGridInputType >::value, "input type of the Smoother kernel must match the input from Multi-Grid" );

			MGSmootherType smoother_runner; ///< object to run the smoother
			CoarsenerType coarsener_runner; ///< object to run the coarsener
			std::vector< std::unique_ptr< MultiGridInputType > > system_levels; ///< levels of the grid (finest first)
			Ring ring; ///< algebraic ring
			Minus minus; ///< minus operator

			// operator to extract the reference out of an std::unique_ptr object
			struct __extractor {
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

			using __unique_ptr_extractor = grb::utils::IteratorValueAdaptor<
				typename std::vector< std::unique_ptr< MultiGridInputType > >::iterator,
				__extractor
			>;

			/**
			 * Construct a new MultiGridRunner object by moving in the state of the pre-built
			 * smoother and coarsener.
			 */
			MultiGridRunner(
				MGSmootherType &&_smoother_runner,
				CoarsenerType &&_coarsener_runner
			) : smoother_runner( std::move( _smoother_runner ) ),
				coarsener_runner( std::move(  _coarsener_runner ) ) {}

			/**
			 * Operator to invoke a full multi-grid run starting from the given level.
			 */
			inline grb::RC operator()( MultiGridInputType &system ) {
				return multi_grid< IOType, NonzeroType, __unique_ptr_extractor,
					MGSmootherType, CoarsenerType, Ring, Minus >(
					__unique_ptr_extractor( system_levels.begin() += system.level ),
					__unique_ptr_extractor( system_levels.end() ),
					smoother_runner, coarsener_runner, ring, minus );
			}
		};

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_V_CYCLE
