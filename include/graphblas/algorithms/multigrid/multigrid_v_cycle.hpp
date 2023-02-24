
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
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <graphblas.hpp>
#include <graphblas/utils/iterators/IteratorValueAdaptor.hpp>
#include <graphblas/utils/telemetry/OutputStream.hpp>

#include "multigrid_data.hpp"


namespace grb {
	namespace algorithms {

		/**
		 * Callable object to invoke the V-cycle multi-grid algorithm, which also requires
		 * a smoother and a coarsener object.
		 *
		 * It is built by transferring into it the state of both the smoother and the coarsener,
		 * in order to avoid use-after-free issues.
		 *
		 * @tparam MGTypes types container for algebraic information (IOType, NonzeroType, Ring, Minus)
		 * @tparam MGSmootherType type of the smoother runner, with prescribed methods for the various
		 *  smoothing steps
		 * @tparam CoarsenerType type of the coarsener runner, with prescribed methods for coarsening
		 * @tparam descr descriptors with statically-known data for computation and containers
		 * @tparam DbgOutputStreamType type for the debugging stream, i.e. the stream to trace simulation
		 * 	results alongside execution; the default type #grb::utils::telemetry::OutputStreamOff disables
		 * 	all output at compile time
		 */
		template<
			typename MGTypes,
			typename MGSmootherType,
			typename CoarsenerType,
			typename TelControllerType,
			Descriptor descr = descriptors::no_operation,
			typename DbgOutputStreamType = grb::utils::telemetry::OutputStreamOff
		> struct MultiGridRunner {

			using self_t = MultiGridRunner< MGTypes, MGSmootherType, CoarsenerType, TelControllerType, descr >;
			// algebraic types
			using IOType = typename MGTypes::IOType;
			using NonzeroType = typename MGTypes::NonzeroType;
			using Ring = typename MGTypes::Ring;
			using Minus = typename MGTypes::Minus;
			using MultiGridInputType = MultiGridData< IOType, NonzeroType, TelControllerType >;
			// runners
			using SmootherRunnerType = MGSmootherType;
			using CoarsenerRunnerType = CoarsenerType;

			static_assert( std::is_default_constructible< Ring >::value,
				"cannot construct the Ring with default values" );
			static_assert( std::is_default_constructible< Minus >::value,
				"cannot construct the Minus operator with default values" );

			// check the interface between HPCG and MG match
			static_assert( std::is_base_of< typename MGSmootherType::SmootherInputType, MultiGridInputType >::value,
				"input type of the Smoother kernel must match the input from Multi-Grid" );

			MGSmootherType & smoother_runner; ///< object to run the smoother
			CoarsenerType & coarsener_runner; ///< object to run the coarsener
			DbgOutputStreamType dbg_logger;   ///< logger to trace execution

			std::vector< std::unique_ptr< MultiGridInputType > > system_levels; ///< levels of the grid (finest first)
			Ring ring;                                                          ///< algebraic ring
			Minus minus;                                                        ///< minus operator

			// operator to extract the reference out of an std::unique_ptr object
			struct __extractor {
				MultiGridInputType * operator()(
					typename std::vector< std::unique_ptr< MultiGridInputType > >::reference & ref ) {
					return ref.get();
				}

				const MultiGridInputType * operator()(
					typename std::vector< std::unique_ptr< MultiGridInputType > >::const_reference & ref ) const {
					return ref.get();
				}
			};

			using __unique_ptr_extractor = grb::utils::IteratorValueAdaptor<
				typename std::vector< std::unique_ptr< MultiGridInputType > >::iterator, __extractor >;

			/**
			 * Construct a new MultiGridRunner object by moving in the state of the pre-built
			 * smoother and coarsener.
			 *
			 * The debug logger is deactivated.
			 */
			MultiGridRunner(
				MGSmootherType & _smoother_runner,
				CoarsenerType & _coarsener_runner
			) :
				smoother_runner( _smoother_runner ),
				coarsener_runner( _coarsener_runner )
			{
				static_assert( std::is_default_constructible< DbgOutputStreamType >::value );
			}

			/**
			 * Construct a new MultiGridRunner object by moving in the state of the pre-built
			 * smoother and coarsener and with a user-given debug logger.
			 */
			MultiGridRunner(
				MGSmootherType & _smoother_runner,
				CoarsenerType & _coarsener_runner,
				DbgOutputStreamType & _dbg_logger
			) :
				smoother_runner( _smoother_runner ),
				coarsener_runner( _coarsener_runner ),
				dbg_logger( _dbg_logger ) {}

			/**
			 * Operator to invoke a full multi-grid run starting from the given level.
			 */
			inline grb::RC operator()( MultiGridInputType & system ) {
				return this->operator()( __unique_ptr_extractor( system_levels.begin() += system.level ),
					__unique_ptr_extractor( system_levels.end() ) );
			}

			/**
			 * Operator to invoke a multi-grid run among given levels.
			 */
			inline grb::RC operator()(
				__unique_ptr_extractor begin,
				const __unique_ptr_extractor end
			) {
				begin->mg_stopwatch.start();
				grb::RC ret = multi_grid( begin, end );
				begin->mg_stopwatch.stop();
				return ret;
			}

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
			 * @param mgiter_begin iterator pointing to the current level of the multi-grid
			 * @param mgiter_end end iterator, indicating the end of the recursion
			 * @return grb::RC if the algorithm could correctly terminate, the error code of the first
			 *  unsuccessful operation otherwise
			 */
			grb::RC multi_grid(
				__unique_ptr_extractor mgiter_begin,
				const __unique_ptr_extractor mgiter_end
			) {
				RC ret = SUCCESS;
				assert( mgiter_begin != mgiter_end );
				MultiGridInputType & finer_system = *mgiter_begin;
				++mgiter_begin;

				dbg_logger << "mg BEGINNING {" << std::endl;

				// clean destination vector
				ret = ret ? ret : grb::set< descr >( finer_system.z, ring.template getZero< IOType >() );
				dbg_logger << ">>> initial r: " << finer_system.r << std::endl;

				if( ! ( mgiter_begin != mgiter_end ) ) {
					// compute one round of Gauss Seidel and return
					ret = ret ? ret : smoother_runner.nonrecursive_smooth( finer_system );
					assert( ret == SUCCESS );
					dbg_logger << ">>> smoothed z: " << finer_system.z << std::endl;
					dbg_logger << "} mg END" << std::endl;
					return ret;
				}
				MultiGridInputType & coarser_system = *mgiter_begin;

				// pre-smoother
				ret = ret ? ret : smoother_runner.pre_smooth( finer_system );
				assert( ret == SUCCESS );
				dbg_logger << ">>> pre-smoothed z: " << finer_system.z << std::endl;

				ret = ret ? ret : coarsener_runner.coarsen_residual( finer_system, coarser_system );
				assert( ret == SUCCESS );
				dbg_logger << ">>> coarse r: " << coarser_system.r << std::endl;

				ret = ret ? ret : this->operator()( mgiter_begin, mgiter_end );
				assert( ret == SUCCESS );

				ret = ret ? ret : coarsener_runner.prolong_solution( coarser_system, finer_system );
				assert( ret == SUCCESS );
				dbg_logger << ">>> prolonged z: " << finer_system.z << std::endl;

				// post-smoother
				ret = ret ? ret : smoother_runner.post_smooth( finer_system );
				assert( ret == SUCCESS );
				dbg_logger << ">>> post-smoothed z: " << finer_system.z << std::endl;
				dbg_logger << "} mg END" << std::endl;

				return ret;
			}
		};

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_V_CYCLE
