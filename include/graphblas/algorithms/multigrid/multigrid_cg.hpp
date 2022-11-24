
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
 * @dir include/graphblas/algorithms/mutligrid
 * This folder contains the implementation of the algorithms for a basic multi-grid V-cycle solver:
 * Conjugate Gradient with multi-grid, a basic V-cycle multi-grid implementation, a single-matrix coarsener/
 * prolonger, an implementation of a Red-Black Gauss-Seidel smoother. These algorithms can be composed
 * via their specific runners, as in the example HPCG benchmark.
 */

/**
 * @file multigrid_cg.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Algorithm and runner for a Conjugate Gradient solver augmented with a multi-grid solver.
 */

#ifndef _H_GRB_ALGORITHMS_MULTIGRID_CG
#define _H_GRB_ALGORITHMS_MULTIGRID_CG

#include <type_traits>
#include <utility>

#include <graphblas.hpp>
#include <graphblas/utils/Timer.hpp>

#include "multigrid_data.hpp"


namespace grb {
	namespace algorithms {

		/**
		 * Data stucture to store the vectors specific to the Conjugate Gradient algorithm,
		 * including inputs, outputs and temporary vectors.
		 *
		 * Input and output vectors use the same naming scheme as for the corresponding mathematics,
		 * where the equation to solve is conventionally written as \f$ A x = b \f$.
		 *
		 * @tparam IOType type of values of the vectors for intermediate results
		 * @tparam NonzeroType type of the values stored inside the system matrix #A
		 * @tparam InputType type of the values of the right-hand side vector #b
		 */
		template<
			typename IOType,
			typename NonzeroType,
			typename InputType
		> struct MultiGridCGData {

			grb::Vector< InputType > b; ///< Right-side vector of known values.
			grb::Vector< IOType > u;    ///< temporary vectors (typically for CG exploration directions)
			grb::Vector< IOType > p;    ///< temporary vector (typically for x refinements coming from the multi-grid run)
			grb::Vector< IOType > x;    ///< system solution being refined over the iterations: it us up to the user
			///< to set the initial solution value to something meaningful


			/**
			 * Construct a new \c MultiGridCGData object by building its vectors with size \p sys_size.
			 */
			MultiGridCGData( size_t sys_size ) :
				b( sys_size ),
				u( sys_size ),
				p( sys_size ),
				x( sys_size ) {}

			grb::RC init_vectors( IOType zero ) {
				grb::RC rc = grb::set( u, zero );
				rc = rc ? rc : grb::set( p, zero );
				return rc;
			}
		};

		/**
		 * Container for various options and algebraic abstractions to be passed to a CG simulation with multi-grid.
		 */
		template <
			typename IOType,
			typename ResidualType,
			class Ring,
			class Minus
		> struct CGOptions {
			bool with_preconditioning; ///<  whether preconditioning is enabled
			size_t max_iterations; ///< max number of allowed iterations for CG: after that, the solver is halted
									///< and the result achieved so far returned
			ResidualType tolerance; ///< ratio between initial residual and current residual that halts the solver
										///< if reached, for the solution is to be considered "good enough"
			bool print_iter_stats; ///< whether to print information on the multi-grid and the residual on each iteration
			Ring ring; ///< algebraic ring to be used
			Minus minus; ///< minus operator to be used
		};

		/**
		 * Structure for the output information of a CG run.
		 */
		template < typename ResidualType > struct CGOutInfo {
			size_t iterations; ///< number of iterations performed
			ResidualType norm_residual; ///< norm of the final residual
		};

		/**
		 * Conjugate Gradient algorithm implementation augmented by a Multi-Grid solver,
		 * inspired to the High Performance Conjugate Gradient benchmark.
		 *
		 * This CG solver calls the MG solver at the beginning of each iteration to improve
		 * the initial solution via the residual (thanks to the smoother) and then proceeds with
		 * the standard CG iteration.
		 *
		 * Failures of GraphBLAS operations are handled by immediately stopping the execution and by returning
		 * the failure code.
		 *
		 * @tparam IOType type of result and intermediate vectors used during computation
		 * @tparam ResidualType type of the residual norm
		 * @tparam NonzeroType type of matrix values
		 * @tparam InputType type of values of the right-hand side vector b
		 * @tparam MultiGridrunnerType type for the multi-grid runner object
		 * @tparam Ring algebraic ring type
		 * @tparam Minus minus operator
		 *
		 * @param cg_data data for the CG solver only
		 * @param cg_opts options for the CG solver
		 * @param grid_base base (i.e., finer) level of the multi-grid, with the information of the physical system
		 * @param MultiGridRunner runner object (functor) to call the multi-grid solver
		 * @param out_info solver output information
		 * @return grb::RC SUCCESS in case of succesful run
		 */
		template<
			typename IOType,
			typename ResidualType,
			typename NonzeroType,
			typename InputType,
			typename MultiGridrunnerType,
			class Ring = Semiring< grb::operators::add< IOType >, grb::operators::mul< IOType >, grb::identities::zero, grb::identities::one >,
			class Minus = operators::subtract< IOType >
		> grb::RC multigrid_conjugate_gradient(
			MultiGridCGData< IOType, NonzeroType, InputType > &cg_data,
			const CGOptions< IOType, ResidualType, Ring, Minus > &cg_opts,
			MultiGridData< IOType, NonzeroType > &grid_base,
			MultiGridrunnerType &multigrid_runner,
			CGOutInfo< ResidualType > &out_info
		) {
			const grb::Matrix< NonzeroType > &A = grid_base.A; // system matrix
			grb::Vector< IOType > &r = grid_base.r;  // residual vector
			grb::Vector< IOType > &z = grid_base.z;  // pre-conditioned residual vector
			grb::Vector< IOType > &x = cg_data.x; // initial (and final) solution
			const grb::Vector< InputType > &b = cg_data.b; // right-side value
			grb::Vector< IOType > &p = cg_data.p;  // direction vector
			grb::Vector< IOType > &Ap = cg_data.u; // temp vector
			grb::RC ret = SUCCESS;

			const IOType io_zero = cg_opts.ring.template getZero< IOType >();
			ret = ret ? ret : grb::set( Ap, io_zero );
			ret = ret ? ret : grb::set( r, io_zero );
			ret = ret ? ret : grb::set( p, io_zero );

			ret = ret ? ret : grb::set( p, x );
			// Ap = A * x
			ret = ret ? ret : grb::mxv< grb::descriptors::dense >( Ap, A, x, cg_opts.ring );
			assert( ret == SUCCESS );
			// r = b - Ap
			ret = ret ? ret : grb::eWiseApply( r, b, Ap, cg_opts.minus );
			assert( ret == SUCCESS );

			const ResidualType residual_zero = cg_opts.ring.template getZero< ResidualType >();
			ResidualType norm_residual = residual_zero;
			// norm_residual = r' * r
			ret = ret ? ret : grb::dot( norm_residual, r, r, cg_opts.ring );
			assert( ret == SUCCESS );

			// compute sqrt to avoid underflow
			norm_residual = std::sqrt( norm_residual );

			// initial norm of residual
			out_info.norm_residual = norm_residual;
			const ResidualType norm_residual_initial = norm_residual;
			ResidualType old_r_dot_z = residual_zero, r_dot_z = residual_zero, beta = residual_zero;
			size_t iter = 0;

			grb::utils::Timer timer;

#ifdef HPCG_PRINT_STEPS
			DBG_print_norm( p, "start p" );
			DBG_print_norm( Ap, "start Ap" );
			DBG_print_norm( r, "start r" );
#endif
			do {
#ifdef HPCG_PRINT_STEPS
				DBG_println( "========= iteration " << iter << " =========" );
#endif
				if( cg_opts.with_preconditioning ) {
					if( cg_opts.print_iter_stats ) {
						timer.reset();
					}
					ret = ret ? ret : multigrid_runner( grid_base );
					assert( ret == SUCCESS );
					if( cg_opts.print_iter_stats ) {
						double duration = timer.time();
						std::cout << "iteration, pre-conditioner: " << iter << ","
							<< duration << std::endl;
					}
				} else {
					ret = ret ? ret : grb::set( z, r ); // z = r;
					assert( ret == SUCCESS );
				}
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( z, "initial z" );
#endif
				if( iter == 0 ) {
					ret = ret ? ret : grb::set( p, z ); //  p = z;
					assert( ret == SUCCESS );
					ret = ret ? ret : grb::dot( r_dot_z, r, z, cg_opts.ring ); // r_dot_z = r' * z;
					assert( ret == SUCCESS );
				} else {
					old_r_dot_z = r_dot_z;
					// r_dot_z = r' * z
					r_dot_z = cg_opts.ring.template getZero< ResidualType >();
					ret = ret ? ret : grb::dot( r_dot_z, r, z, cg_opts.ring );
					assert( ret == SUCCESS );

					beta = r_dot_z / old_r_dot_z;
					// Ap  = 0
					ret = ret ? ret : grb::set( Ap, io_zero );
					assert( ret == SUCCESS );
					// Ap += beta * p
					ret = ret ? ret : grb::eWiseMul( Ap, beta, p, cg_opts.ring );
					assert( ret == SUCCESS );
					// Ap = Ap + z
					ret = ret ? ret : grb::eWiseApply( Ap, Ap, z, cg_opts.ring.getAdditiveOperator() );
					assert( ret == SUCCESS );
					// p = Ap
					std::swap( Ap, p );
					assert( ret == SUCCESS );
				}
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( p, "middle p" );
#endif
				// Ap = A * p
				ret = ret ? ret : grb::set( Ap, io_zero );
				ret = ret ? ret : grb::mxv< grb::descriptors::dense >( Ap, A, p, cg_opts.ring );
				assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( Ap, "middle Ap" );
#endif
				// pAp = p' * Ap
				ResidualType pAp = cg_opts.ring.template getZero< ResidualType >();
				ret = ret ? ret : grb::dot( pAp, Ap, p, cg_opts.ring );
				assert( ret == SUCCESS );

				ResidualType alpha = r_dot_z / pAp;
				// x += alpha * p
				ret = ret ? ret : grb::eWiseMul( x, alpha, p, cg_opts.ring );
				assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( x, "end x" );
#endif
				// r += - alpha * Ap
				ret = ret ? ret : grb::eWiseMul( r, -alpha, Ap, cg_opts.ring );
				assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( r, "end r" );
#endif
				// residual = r' * r
				norm_residual = cg_opts.ring.template getZero< ResidualType >();
				ret = ret ? ret : grb::dot( norm_residual, r, r, cg_opts.ring );
				assert( ret == SUCCESS );

				norm_residual = std::sqrt( norm_residual );

				if( cg_opts.print_iter_stats ) {
					std::cout << "iteration, residual: " << iter << "," << norm_residual << std::endl;
				}

				++iter;
				out_info.iterations = iter;
				out_info.norm_residual = norm_residual;
			} while( iter < cg_opts.max_iterations &&
				norm_residual / norm_residual_initial > cg_opts.tolerance && ret == SUCCESS );

			return ret;
		}

		/**
		 * Runner object incapsulating all information to run a Conjugate Gradient solver
		 * with multi-grid.
		 *
		 * The multi-grid runner must be constructed separately (depending on the chosen algorithm)
		 * and move-transfered during construction of this runner.
		 * The \p MultiGridrunnerType must implement a functional interface whose input (from CG)
		 * is the structure with the system information for one level of the grid.
		 *
		 * @tparam IOType type of result and intermediate vectors used during computation
		 * @tparam NonzeroType type of matrix values
		 * @tparam InputType type of values of the right-hand side vector b
		 * @tparam ResidualType type of the residual norm
		 * @tparam MultiGridrunnerType type for the multi-grid runner object
		 * @tparam Ring algebraic ring type
		 * @tparam Minus minus operator
		 */
		template<
			typename IOType,
			typename NonzeroType,
			typename InputType,
			typename ResidualType,
			typename MultiGridRunnerType,
			class Ring,
			class Minus
		> struct MultiGridCGRunner {

			using HPCGInputType = MultiGridCGData< IOType, NonzeroType, InputType >;

			static_assert( std::is_default_constructible< Ring >::value,
				"cannot construct the Ring with default values" );
			static_assert( std::is_default_constructible< Minus >::value,
				"cannot construct the Minus operator with default values" );
			static_assert( std::is_move_constructible< MultiGridRunnerType >::value,
				"cannot construct the Multi-Grid runner by move" );

			// default value: override with your own
			CGOptions< IOType, ResidualType, Ring, Minus > cg_opts = { true, 10,
				Ring(). template getZero< ResidualType >(), false, Ring(), Minus() };

			MultiGridRunnerType mg_runner;

			/**
			 * Construct a new MultiGridCGRunner object by moving the required MG runner.
			 *
			 * Moving the state of the MG is safer in that it avoids use-after-free issues,
			 * as the state of the MG runner is managed automatically with this object.
			 */
			MultiGridCGRunner(
				MultiGridRunnerType &&_mg_runner
			) : mg_runner( std::move( _mg_runner ) ) {}

			/**
			 * Functional operator to invoke a full CG-MG computation.
			 *
			 * @param grid_base base level of the grid
			 * @param cg_data data for CG
			 * @param out_info output information from CG
			 * @return grb::RC indicating the success or the error occurred
			 */
			inline grb::RC operator()(
				typename MultiGridRunnerType::MultiGridInputType &grid_base,
				MultiGridCGData< IOType, NonzeroType, InputType > &cg_data,
				CGOutInfo< ResidualType > &out_info
			) {
				return multigrid_conjugate_gradient( cg_data, cg_opts, grid_base, mg_runner, out_info );
			}

		};

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_CG
