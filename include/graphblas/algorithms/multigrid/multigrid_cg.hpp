
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
#include <graphblas/utils/telemetry/Timeable.hpp>
#include <graphblas/utils/telemetry/OutputStream.hpp>

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
		 * Structure for the output information of a CG run.
		 */
		template < typename ResidualType > struct CGOutInfo {
			size_t iterations; ///< number of iterations performed
			ResidualType norm_residual; ///< norm of the final residual
		};

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
		 * @tparam descr descriptors with statically-known data for computation and containers
		 */
		template<
			typename MGCGTypes,
			typename MultiGridRunnerType,
			typename TelTokenType,
			Descriptor descr = descriptors::no_operation,
			typename DbgOutputStreamType = grb::utils::telemetry::OutputStreamOff
		> struct MultiGridCGRunner : public grb::utils::telemetry::Timeable< TelTokenType > {

			using IOType = typename MGCGTypes::IOType;
			using NonzeroType = typename MGCGTypes::NonzeroType;
			using InputType = typename MGCGTypes::InputType;
			using ResidualType = typename MGCGTypes::ResidualType;
			using Ring = typename MGCGTypes::Ring;
			using Minus = typename MGCGTypes::Minus;
			using HPCGInputType = MultiGridCGData< IOType, NonzeroType, InputType >;
			using MGRunnerType = MultiGridRunnerType;

			static_assert( std::is_default_constructible< Ring >::value,
				"cannot construct the Ring with default values" );
			static_assert( std::is_default_constructible< Minus >::value,
				"cannot construct the Minus operator with default values" );
			static_assert( std::is_move_constructible< MultiGridRunnerType >::value,
				"cannot construct the Multi-Grid runner by move" );

			Ring ring; ///< algebraic ring to be used
			Minus minus; ///< minus operator to be used
			bool with_preconditioning = true; ///<  whether preconditioning is enabled
			size_t max_iterations = 10; ///< max number of allowed iterations for CG: after that, the solver is halted
									///< and the result achieved so far returned
			ResidualType tolerance = ring. template getZero< ResidualType >(); ///< ratio between initial residual and current residual that halts the solver
										///< if reached, for the solution is to be considered "good enough"

			MultiGridRunnerType &mg_runner;
			DbgOutputStreamType dbg_logger;

			/**
			 * Construct a new MultiGridCGRunner object by moving the required MG runner.
			 *
			 * Moving the state of the MG is safer in that it avoids use-after-free issues,
			 * as the state of the MG runner is managed automatically with this object.
			 */
			MultiGridCGRunner(
				const TelTokenType & tt,
				MultiGridRunnerType &_mg_runner
			) :
				grb::utils::telemetry::Timeable< TelTokenType >( tt ),
				mg_runner( _mg_runner ),
				dbg_logger()
			{
				static_assert( std::is_default_constructible< DbgOutputStreamType >::value );
			}

			MultiGridCGRunner(
				const TelTokenType & tt,
				MultiGridRunnerType & _mg_runner,
				DbgOutputStreamType & _dbg_logger
			) :
				grb::utils::telemetry::Timeable< TelTokenType >( tt ),
				mg_runner( _mg_runner ),
				dbg_logger( _dbg_logger )
			{}

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
				this->start();
				grb::RC ret = multigrid_conjugate_gradient( cg_data, grid_base, out_info );
				this->stop();
				return ret;
			}

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
			 *
			 * @param cg_data data for the CG solver only
			 * @param grid_base base (i.e., finer) level of the multi-grid, with the information of the physical system
			 * @param out_info solver output information
			 * @return grb::RC SUCCESS in case of succesful run
			 */
			grb::RC multigrid_conjugate_gradient(
				HPCGInputType &cg_data,
				typename MultiGridRunnerType::MultiGridInputType &grid_base,
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

				const IOType io_zero = ring.template getZero< IOType >();
				ret = ret ? ret : grb::set( Ap, io_zero );
				ret = ret ? ret : grb::set( r, io_zero );
				ret = ret ? ret : grb::set( p, io_zero );

				ret = ret ? ret : grb::set( p, x );
				// Ap = A * x
				ret = ret ? ret : grb::mxv< descr >( Ap, A, x, ring );
				assert( ret == SUCCESS );
				// r = b - Ap
				ret = ret ? ret : grb::eWiseApply< descr >( r, b, Ap, minus );
				assert( ret == SUCCESS );

				const ResidualType residual_zero = ring.template getZero< ResidualType >();
				ResidualType norm_residual = residual_zero;
				// norm_residual = r' * r
				ret = ret ? ret : grb::dot< descr >( norm_residual, r, r, ring );
				assert( ret == SUCCESS );

				// compute sqrt to avoid underflow
				norm_residual = std::sqrt( norm_residual );

				// initial norm of residual
				out_info.norm_residual = norm_residual;
				const ResidualType norm_residual_initial = norm_residual;
				ResidualType old_r_dot_z = residual_zero, r_dot_z = residual_zero, beta = residual_zero;
				size_t iter = 0;

				dbg_logger << ">>> start p: " << p << std::endl;
				dbg_logger << ">>> start Ap: " << Ap << std::endl;
				dbg_logger << ">>> start r: " << r << std::endl;

				do {
					dbg_logger << "========= iteration " << iter << " =========" << std::endl;

					if( with_preconditioning ) {
						ret = ret ? ret : mg_runner( grid_base );
						assert( ret == SUCCESS );
					} else {
						// z = r
						ret = ret ? ret : grb::set( z, r );
						assert( ret == SUCCESS );
					}
					dbg_logger << ">>> initial z: " << z << std::endl;

					if( iter == 0 ) {
						//  p = z
						ret = ret ? ret : grb::set< descr >( p, z );
						assert( ret == SUCCESS );
						// r_dot_z = r' * z
						ret = ret ? ret : grb::dot< descr >( r_dot_z, r, z, ring );
						assert( ret == SUCCESS );
					} else {
						old_r_dot_z = r_dot_z;
						// r_dot_z = r' * z
						r_dot_z = ring.template getZero< ResidualType >();
						ret = ret ? ret : grb::dot< descr >( r_dot_z, r, z, ring );
						assert( ret == SUCCESS );

						beta = r_dot_z / old_r_dot_z;
						// Ap  = 0
						ret = ret ? ret : grb::set< descr >( Ap, io_zero );
						assert( ret == SUCCESS );
						// Ap += beta * p
						ret = ret ? ret : grb::eWiseMul< descr >( Ap, beta, p, ring );
						assert( ret == SUCCESS );
						// Ap = Ap + z
						ret = ret ? ret : grb::eWiseApply< descr >( Ap, Ap, z, ring.getAdditiveOperator() );
						assert( ret == SUCCESS );
						// p = Ap
						std::swap( Ap, p );
						assert( ret == SUCCESS );
					}
					dbg_logger << ">>> middle p: " << p << std::endl;

					// Ap = A * p
					ret = ret ? ret : grb::set< descr >( Ap, io_zero );
					ret = ret ? ret : grb::mxv< descr >( Ap, A, p, ring );
					assert( ret == SUCCESS );
					dbg_logger << ">>> middle Ap: " << Ap << std::endl;

					// pAp = p' * Ap
					ResidualType pAp = ring.template getZero< ResidualType >();
					ret = ret ? ret : grb::dot< descr >( pAp, Ap, p, ring );
					assert( ret == SUCCESS );

					ResidualType alpha = r_dot_z / pAp;
					// x += alpha * p
					ret = ret ? ret : grb::eWiseMul< descr >( x, alpha, p, ring );
					assert( ret == SUCCESS );
					dbg_logger << ">>> end x: " << x << std::endl;

					// r += - alpha * Ap
					ret = ret ? ret : grb::eWiseMul< descr >( r, -alpha, Ap, ring );
					assert( ret == SUCCESS );
					dbg_logger << ">>> end r: " << r << std::endl;

					// residual = r' * r
					norm_residual = ring.template getZero< ResidualType >();
					ret = ret ? ret : grb::dot< descr >( norm_residual, r, r, ring );
					assert( ret == SUCCESS );

					norm_residual = std::sqrt( norm_residual );

					++iter;
					out_info.iterations = iter;
					out_info.norm_residual = norm_residual;
				} while( iter < max_iterations &&
					norm_residual / norm_residual_initial > tolerance && ret == SUCCESS );

				return ret;
			}

		};

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_CG
