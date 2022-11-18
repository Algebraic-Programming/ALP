
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
 * @file hpcg.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @brief File with the main routine to run a full HPCG simulation, comprising multi-grid runs
 *        with Red-Black Gauss-Seidel smoothing.
 * @date 2021-04-30
 */

#ifndef _H_GRB_ALGORITHMS_MULTIGRID_CG
#define _H_GRB_ALGORITHMS_MULTIGRID_CG

#include <type_traits>
#include <utility>

#include <graphblas.hpp>

#include "multigrid_data.hpp"

#include <graphblas/utils/Timer.hpp>


namespace grb {
	namespace algorithms {

		/**
		 * @brief Data stucture to store the data for a full HPCG run: system vectors and matrix,
		 * coarsening information and temporary vectors.
		 *
		 * This data structures contains all the needed vectors and matrices to solve a linear system
		 * \f$ A x = b \f$. As for \ref system_data, internal elements are built and their sizes properly initialized
		 * to #system_size, but internal values are \b not initialized, as they are left to user's logic.
		 * Similarly, the coarsening information in #coarser_level is to be initialized by users by properly
		 * building a \code multigrid_data<IOType, NonzeroType> \endcode object and storing its pointer into
		 * #coarser_level; on destruction, #coarser_level will also be properly destroyed without
		 * user's intervention.
		 *
		 * @tparam IOType type of values of the vectors for intermediate results
		 * @tparam NonzeroType type of the values stored inside the system matrix #A
		 * @tparam InputType type of the values of the right-hand side vector #b
		 */
		template<
			typename IOType,
			typename NonzeroType,
			typename InputType
		> struct mg_cg_data {

			grb::Vector< InputType > b; ///< right-side vector of known values
			grb::Vector< IOType > u;    ///< temporary vectors (typically for CG exploration directions)
			grb::Vector< IOType > p;    ///< temporary vector (typically for x refinements coming from the multi-grid run)
			grb::Vector< IOType > x;    // system solution being refined over the iterations: it us up to the user
			///< to set the initial solution value


			/**
			 * @brief Construct a new \c hpcg_data object by building vectors and matrices and by setting
			 * #coarser_level to \c nullptr (i.e. no coarser level is assumed).
			 *
			 * @param[in] sys_size the size of the simulated system, i.e. of all the internal vectors and matrices
			 */
			mg_cg_data( size_t sys_size ) :
				b( sys_size ),
				u( sys_size ),
				p( sys_size ),
				x( sys_size ) {}

			grb::RC zero_temp_vectors() {
				grb::RC rc = grb::set( u, 0 );
				rc = rc ? rc : grb::set( p, 0 );
				return rc;
			}
		};

		template <
			typename IOType,
			typename ResidualType,
			class Ring = Semiring< grb::operators::add< IOType >, grb::operators::mul< IOType >, grb::identities::zero, grb::identities::one >,
			class Minus = operators::subtract< IOType >
		>
		struct cg_options {
			bool with_preconditioning;
			size_t max_iterations;
			ResidualType tolerance;
			bool print_iter_stats;
			Ring ring;
			Minus minus;
		};


		template < typename ResidualType > struct cg_out_data {
			size_t iterations;
			ResidualType norm_residual;
		};

		/**
		 * @brief High-Performance Conjugate Gradient algorithm implementation running entirely on GraphBLAS.
		 *
		 * Finds the solution x of an \f$ A x = b \f$ algebraic system by running the HPCG algorithm.
		 * The implementation here closely follows the reference HPCG benchmark used for the HPCG500 rank,
		 * visible at https://github.com/hpcg-benchmark/hpcg.
		 * The only difference is the usage of a Red-Black Gauss-Seidel smoother instead of the standard one
		 * for performance reasons, as the standard Gauss-Seidel algorithm is inherently sequential and not
		 * expressible in terms of standard linear algebra operations.
		 * In particular, this implementation (as the standard one) couples a standard CG algorithm with a V-cycle
		 * multi-grid solver to initially refine the tentative solution. This refinement step depends on the
		 * availability of coarsening information, which should be stored inside \p data; otherwise,
		 * the refinement is not performed and only the CG algorithm is run. For more information on inputs
		 * and on coarsening information, you may consult the \ref hpcg_data class documentation.
		 *
		 * This implementation assumes that the vectors and matrices inside \p data are all correctly initialized
		 * and populated with the proper values; in particular
		 * - hpcg_data#x with the initial tentative solution (iterative solutions are also stored here)
		 * - hpcg_data#A with the system matrix
		 * - hpcg_data#b with the right-hand side vector \f$ b \f$
		 * - hpcg_data#A_diagonal with the diagonal values of the matrix
		 * - hpcg_data#color_masks with the color masks for this level
		 * - hpcg_data#coarser_level with the information for the coarser multi-grid run (if any)
		 * The other vectors are assumed to be inizialized (via the usual grb::Vector#Vector(size_t) constructor)
		 * but not necessarily populated with values, as they are internally populated when needed; hence,
		 * any previous values are overwritten.
		 *
		 * Failuers of GraphBLAS operations are handled by immediately stopping the execution and by returning
		 * the failure code.
		 *
		 * @tparam IOType type of result and intermediate vectors used during computation
		 * @tparam ResidualType type of the residual norm
		 * @tparam NonzeroType type of matrix values
		 * @tparam InputType type of values of the right-hand side vector b
		 * @tparam Ring the ring of algebraic operators zero-values
		 * @tparam Minus the minus operator for subtractions
		 *
		 * @param[in,out] data \ref hpcg_data object storing inputs, outputs and temporary vectors used for the computation,
		 *                     as long as the information for the recursive multi-grid runs
		 * @param[in] with_preconditioning whether to use pre-conditioning, i.e. to perform multi-grid runs
		 * @param[in] presmoother_steps number of pre-smoother steps, for multi-grid runs
		 * @param[in] postsmoother_steps nomber of post-smoother steps, for multi-grid runs
		 * @param[in] max_iterations maximum number if iterations the simulation may run for; once reached,
		 *                           the simulation stops even if the residual norm is above \p tolerance
		 * @param[in] tolerance the tolerance over the residual norm, i.e. the value of the residual norm to stop
		 *                      the simulation at
		 * @param[out] iterations numbers of iterations performed
		 * @param[out] norm_residual norm of the final residual
		 * @param[in] ring the ring to perform the operations on
		 * @param[in] minus the \f$ - \f$ operator for vector subtractions
		 * @return grb::RC::SUCCESS if the algorithm could correctly terminate, the error code of the first
		 *                          unsuccessful operation otherwise
		 */
		template<
			typename IOType,
			typename ResidualType,
			typename NonzeroType,
			typename InputType,
			typename MultiGridrunnerType,
			class Ring = Semiring< grb::operators::add< IOType >, grb::operators::mul< IOType >, grb::identities::zero, grb::identities::one >,
			class Minus = operators::subtract< IOType >
		> grb::RC mg_cg(
			multigrid_data< IOType, NonzeroType > &grid_base,
			mg_cg_data< IOType, NonzeroType, InputType > &data,
			const cg_options< IOType, ResidualType > &cg_opts,
			MultiGridrunnerType &multigrid_runner,
			cg_out_data< ResidualType > &out_data
		) {
			ResidualType alpha;

			const grb::Matrix< NonzeroType > &A { grid_base.A };
			grb::Vector< IOType > &r { grid_base.r };  // residual vector
			grb::Vector< IOType > &z { grid_base.z };  // pre-conditioned residual vector
			grb::Vector< IOType > &x { data.x };
			const grb::Vector< InputType > &b { data.b };
			grb::Vector< IOType > &p { data.p };  // direction vector
			grb::Vector< IOType > &Ap { data.u }; // temp vector
			grb::RC ret { SUCCESS };

			ret = ret ? ret : grb::set( Ap, 0 );
			ret = ret ? ret : grb::set( r, 0 );
			ret = ret ? ret : grb::set( p, 0 );

			ret = ret ? ret : grb::set( p, x );
			ret = ret ? ret : grb::mxv< grb::descriptors::dense >( Ap, A, x, cg_opts.ring ); // Ap = A * x
			assert( ret == SUCCESS );

			ret = ret ? ret : grb::eWiseApply( r, b, Ap, cg_opts.minus ); // r = b - Ap;
			assert( ret == SUCCESS );

			ResidualType norm_residual = cg_opts.ring.template getZero< ResidualType >();
			ret = ret ? ret : grb::dot( norm_residual, r, r, cg_opts.ring ); // norm_residual = r' * r;
			assert( ret == SUCCESS );

			// compute sqrt to avoid underflow
			norm_residual = std::sqrt( norm_residual );

			// initial norm of residual
			out_data.norm_residual = norm_residual;
			const ResidualType norm_residual_initial { norm_residual };
			ResidualType old_r_dot_z { 0.0 }, r_dot_z { 0.0 }, beta { 0.0 };
			size_t iter { 0 };

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

				ResidualType pAp;

				if( iter == 0 ) {
					ret = ret ? ret : grb::set( p, z ); //  p = z;
					assert( ret == SUCCESS );

					ret = ret ? ret : grb::dot( r_dot_z, r, z, cg_opts.ring ); // r_dot_z = r' * z;
					assert( ret == SUCCESS );
				} else {
					old_r_dot_z = r_dot_z;

					r_dot_z = cg_opts.ring.template getZero< ResidualType >();
					ret = ret ? ret : grb::dot( r_dot_z, r, z, cg_opts.ring ); // r_dot_z = r' * z;
					assert( ret == SUCCESS );

					beta = r_dot_z / old_r_dot_z;
					ret = ret ? ret : grb::clear( Ap );                         // Ap  = 0;
					ret = ret ? ret : grb::eWiseMulAdd( Ap, beta, p, z, cg_opts.ring ); // Ap += beta * p + z;
					std::swap( Ap, p );                                         // p = Ap;
					assert( ret == SUCCESS );
				}
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( p, "middle p" );
#endif

				ret = ret ? ret : grb::set( Ap, 0 );
				ret = ret ? ret : grb::mxv< grb::descriptors::dense >( Ap, A, p, cg_opts.ring ); // Ap = A * p;
				assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( Ap, "middle Ap" );
#endif
				pAp = cg_opts.ring.template getZero< ResidualType >();
				ret = ret ? ret : grb::dot( pAp, Ap, p, cg_opts.ring ); // pAp = p' * Ap
				assert( ret == SUCCESS );

				alpha = r_dot_z / pAp;

				ret = ret ? ret : grb::eWiseMul( x, alpha, p, cg_opts.ring ); // x += alpha * p;
				assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( x, "end x" );
#endif

				ret = ret ? ret : grb::eWiseMul( r, -alpha, Ap, cg_opts.ring ); // r += - alpha * Ap;
				assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( r, "end r" );
#endif

				norm_residual = cg_opts.ring.template getZero< ResidualType >();
				ret = ret ? ret : grb::dot( norm_residual, r, r, cg_opts.ring ); // residual = r' * r;
				assert( ret == SUCCESS );

				norm_residual = std::sqrt( norm_residual );

				if( cg_opts.print_iter_stats ) {
					std::cout << "iteration, residual: " << iter << "," << norm_residual << std::endl;
				}

				++iter;
				out_data.iterations = iter;
				out_data.norm_residual = norm_residual;
			} while( iter < cg_opts.max_iterations &&
				norm_residual / norm_residual_initial > cg_opts.tolerance && ret == SUCCESS );

			return ret;
		}




		template<
			typename IOType,
			typename NonzeroType,
			typename InputType,
			typename ResidualType,
			typename MultiGridRunnerType,
			class Ring,
			class Minus

		> struct mg_cg_runner {

			using HPCGInputType = mg_cg_data< IOType, NonzeroType, InputType >;

			static_assert( std::is_default_constructible< Ring >::value,
				"cannot construct the Ring with default values" );
			static_assert( std::is_default_constructible< Minus >::value,
				"cannot construct the Minus operator with default values" );
			// static_assert( std::is_copy_constructible< MultiGridRunnerType >::value,
			// 	"cannot construct the Multi-Grid runner by copy" );
			static_assert( std::is_move_constructible< MultiGridRunnerType >::value,
				"cannot construct the Multi-Grid runner by move" );

			// default value: override with your own
			cg_options< IOType, ResidualType, Ring, Minus > cg_opts{ true, 10, 0.0, false, Ring(), Minus() };

			MultiGridRunnerType mg_runner;

			mg_cg_runner(
				MultiGridRunnerType &&_mg_runner
			) : mg_runner( std::move( _mg_runner ) ) {}

			inline grb::RC operator()(
				typename MultiGridRunnerType::MultiGridInputType &grid_base,
				mg_cg_data< IOType, NonzeroType, InputType > &data,
				cg_out_data< ResidualType > &out_data
			) {
				return mg_cg( grid_base, data, cg_opts, mg_runner, out_data );
			}

		};

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_CG
