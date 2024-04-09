
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
 * Algebraic Multi-grid algorithm relying on level_ matrices from AMGCL.
 * @file amg.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @author Denis Jelovina (denis.jelovina@huawei.com)
 * @date 2022-08-10
 */

#ifndef _H_GRB_ALGORITHMS_AMG
#define _H_GRB_ALGORITHMS_AMG
#ifdef _DEBUG
 #define AMG_PRINT_STEPS
#endif

#include <graphblas.hpp>
#include "amg_data.hpp"
#include "multigrid_v_cycle.hpp"
#include <utils/print_vec_mat.hpp>

namespace grb {

	namespace algorithms {

		/**
		 * Algebraic Multi-grid algorithm relying on level_ matrices from AMGCL.
		 *
		 * Finds the solution x of an \f$ A x = b \f$ algebraic system by running
		 * the AMG algorithm.  AMG implementation (as the standard one) couples a
		 * standard CG algorithm with a V-cycle  multi-grid solver to initially
		 * refine the tentative solution. This refinement step depends on the
		 * availability of coarsening information, which should be stored inside
		 * \p data; otherwise, the refinement is not performed and only the CG
		 * algorithm is run.
		 *
		 * This implementation assumes that the vectors and matrices inside \p data
		 * are all correctly initialized and populated with the proper values;
		 * in particular
		 * - amg_data#x with the initial tentative solution
		 *              (iterative solutions are also stored here)
		 * - amg_data#A with the system matrix
		 * - amg_data#b with the right-hand side vector \f$ b \f$
		 * - amg_data#A_diagonal with the diagonal values of the matrix
		 * - amg_data#coarser_level with the information for
		 *                the coarser multi-grid run (if any)
		 * The other vectors are assumed to be inizialized
		 * (via the usual grb::Vector#Vector(size_t) constructor)
		 * but not necessarily populated with values, as they are internally populated
		 * when needed; hence, any previous values are overwritten.
		 *
		 * Failuers of GraphBLAS operations are handled by immediately stopping the
		 * execution and by returning the failure code.
		 *
		 * @tparam IOType type of result and intermediate vectors used during computation
		 * @tparam ResidualType type of the residual norm
		 * @tparam NonzeroType type of matrix values
		 * @tparam InputType type of values of the right-hand side vector b
		 * @tparam Ring the ring of algebraic operators zero-values
		 * @tparam Minus the minus operator for subtractions
		 *
		 * @param[in,out] data \ref amg_data object storing inputs, outputs and temporary
		 *                   vectors used for the computation, as long as the information
		 *                   for the recursive multi-grid runs
		 * @param[in] with_preconditioning whether to use pre-conditioning,
		 *                     i.e. to perform multi-grid runs
		 * @param[in] presmoother_steps number of pre-smoother steps, for multi-grid runs
		 * @param[in] postsmoother_steps nomber of post-smoother steps,
		 *                          for multi-grid runs
		 * @param[in] max_iterations maximum number if iterations the simulation may run
		 *                       for; once reached, the simulation stops even if the
		 *                       residual norm is above \p tolerance
		 * @param[in] tolerance the tolerance over the residual norm, i.e. the value of
		 *                      the residual norm to stop  the simulation at
		 * @param[out] iterations numbers of iterations performed
		 * @param[out] norm_residual norm of the final residual
		 * @param[in] ring the ring to perform the operations on
		 * @param[in] minus the \f$ - \f$ operator for vector subtractions
		 * @return grb::RC::SUCCESS if the algorithm could correctly terminate, the error
		 *                          code of the first unsuccessful operation otherwise
		 */
		template< typename IOType,
			typename ResidualType,
			typename NonzeroType,
			typename InputType,
			class Ring = Semiring<
				grb::operators::add< IOType >,
				grb::operators::mul< IOType >,
				grb::identities::zero,
				grb::identities::one
			>,
			class Minus = operators::subtract< IOType > >
		grb::RC amg(
			amg_data< IOType, NonzeroType, InputType > &data,
			bool with_preconditioning,
			const size_t presmoother_steps,
			const size_t postsmoother_steps,
			const size_t max_iterations,
			const ResidualType tolerance,
			size_t &iterations,
			ResidualType &norm_residual,
			const Ring &ring = Ring(),
			const Minus &minus = Minus()
		) {
			ResidualType alpha;
			const grb::Matrix< NonzeroType > &A = data.A;
			grb::Vector< IOType > &x = data.x;
			const grb::Vector< InputType > &b = data.b;
			grb::Vector< IOType > &r = data.r;  // residual vector
			grb::Vector< IOType > &p = data.p;  // direction vector
			grb::Vector< IOType > &Ap = data.u; // temp vector
			grb::Vector< IOType > &z = data.z;  // pre-conditioned residual vector
			grb::RC ret = SUCCESS;

			ret = ret ? ret : grb::set( Ap, 0 );
			ret = ret ? ret : grb::set( r, 0 );
			ret = ret ? ret : grb::set( p, 0 );

			ret = ret ? ret : grb::set( p, x );
			ret = ret ? ret : grb::mxv( Ap, A, x, ring ); // Ap = A * x
			assert( ret == SUCCESS );

			ret = ret ? ret : grb::eWiseApply( r, b, Ap, minus ); // r = b - Ap;
			assert( ret == SUCCESS );

			norm_residual = ring.template getZero< ResidualType >();
			ret = ret ? ret : grb::dot( norm_residual, r, r, ring ); // norm_residual = r' * r;
			assert( ret == SUCCESS );

			// compute sqrt to avoid underflow
			norm_residual = std::sqrt( norm_residual );

			// initial norm of residual
			const ResidualType norm_residual_initial = norm_residual;
			ResidualType old_r_dot_z = 0.0;
			ResidualType r_dot_z = 0.0;
			ResidualType beta = 0.0;
			size_t iter = 0;

#ifdef AMG_PRINT_STEPS
			print_norm( p, "start p" );
			print_norm( Ap, "start Ap" );
			print_norm( r, "start r" );
#endif

			do {
#ifdef AMG_PRINT_STEPS
				DBG_println( "========= iteration " << iter << " =========" );
#endif
				if( with_preconditioning ) {
					//ret = ret ? ret : grb::set( z, r ); // z = r;
					ret = ret ? ret : internal::multi_grid(
						data, data.coarser_level, presmoother_steps, postsmoother_steps, ring, minus
					);
					//ret = ret ? ret : grb::set( x, z );
					assert( ret == SUCCESS );
				} else {
					ret = ret ? ret : grb::set( z, r ); // z = r;
					assert( ret == SUCCESS );
				}
#ifdef AMG_PRINT_STEPS
				print_norm( z, "initial z" );
#endif

				ResidualType pAp;

				if( iter == 0 ) {
					ret = ret ? ret : grb::set( p, z ); //  p = z;
					assert( ret == SUCCESS );

					ret = ret ? ret : grb::dot( r_dot_z, r, z, ring ); // r_dot_z = r' * z;
					assert( ret == SUCCESS );
				} else {
					old_r_dot_z = r_dot_z;

					r_dot_z = ring.template getZero< ResidualType >();
					ret = ret ? ret : grb::dot( r_dot_z, r, z, ring ); // r_dot_z = r' * z;
					assert( ret == SUCCESS );

					beta = r_dot_z / old_r_dot_z;
					ret = ret ? ret : grb::clear( Ap );                         // Ap  = 0;
					ret = ret ? ret : grb::eWiseMulAdd( Ap, beta, p, z, ring ); // Ap += beta * p + z;
					std::swap( Ap, p );                                         // p = Ap;
					assert( ret == SUCCESS );
				}
#ifdef AMG_PRINT_STEPS
				print_norm( p, "middle p" );
#endif
				ret = ret ? ret : grb::set( Ap, 0 );
				ret = ret ? ret : grb::mxv( Ap, A, p, ring ); // Ap = A * p;

				assert( ret == SUCCESS );
#ifdef AMG_PRINT_STEPS
				print_norm( Ap, "middle Ap" );
#endif
				pAp = static_cast< ResidualType >( 0.0 );
				ret = ret ? ret : grb::dot( pAp, Ap, p, ring ); // pAp = p' * Ap
				assert( ret == SUCCESS );

				alpha = r_dot_z / pAp;

				ret = ret ? ret : grb::eWiseMul( x, alpha, p, ring ); // x += alpha * p;
				assert( ret == SUCCESS );
#ifdef AMG_PRINT_STEPS
				print_norm( x, "end x" );
#endif
				ret = ret ? ret : grb::eWiseMul( r, -alpha, Ap, ring ); // r += - alpha * Ap;
				assert( ret == SUCCESS );
#ifdef AMG_PRINT_STEPS
				print_norm( r, "end r" );
#endif

				norm_residual = static_cast< ResidualType >( 0.0 );
				ret = ret ? ret : grb::dot( norm_residual, r, r, ring ); // residual = r' * r;
				assert( ret == SUCCESS );

				norm_residual = std::sqrt( norm_residual );

#ifdef AMG_PRINT_STEPS
				std::cout << " ---> norm_residual=" << norm_residual << "\n";
#endif
				++iter;
			} while(
				iter < max_iterations &&
				norm_residual / norm_residual_initial > tolerance &&
				ret == SUCCESS
				);
			iterations = iter;
			return ret;
		}

	} // namespace algorithms

} // namespace grb

#endif // _H_GRB_ALGORITHMS_AMG
