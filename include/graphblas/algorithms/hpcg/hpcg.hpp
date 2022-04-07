
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

#ifndef _H_GRB_ALGORITHMS_HPCG
#define _H_GRB_ALGORITHMS_HPCG

#include <graphblas.hpp>

#include "hpcg_data.hpp"
#include "multigrid_v_cycle.hpp"


namespace grb {
	namespace algorithms {

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
		template< typename IOType,
			typename ResidualType,
			typename NonzeroType,
			typename InputType,
			class Ring = Semiring< grb::operators::add< IOType >, grb::operators::mul< IOType >, grb::identities::zero, grb::identities::one >,
			class Minus = operators::subtract< IOType > >
		grb::RC hpcg( hpcg_data< IOType, NonzeroType, InputType > &data,
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
			IOType alpha;
			IOType dotproduct;

			const grb::Matrix< NonzeroType > &A { data.A };
			grb::Vector< IOType > &x { data.x };
			const grb::Vector< InputType > &b { data.b };
			grb::Vector< IOType > &r { data.r };  // residual vector
			grb::Vector< IOType > &p { data.p };  // direction vector
			grb::Vector< IOType > &Ap { data.u }; // temp vector
			grb::Vector< IOType > &z { data.z };  // pre-conditioned residual vector
			grb::RC ret { SUCCESS };

			ret = ret ? ret : grb::set( Ap, 0 );
			ret = ret ? ret : grb::set( r, 0 );
			ret = ret ? ret : grb::set( p, 0 );

			ret = ret ? ret : grb::set( p, x );
			ret = ret ? ret : grb::mxv( Ap, A, x, ring ); // Ap = A * x
			assert( ret == SUCCESS );

			ret = ret ? ret : grb::eWiseApply( r, b, Ap, minus ); // r = b - Ap;
			assert( ret == SUCCESS );

			dotproduct =  static_cast< IOType >( 0.0 );
			ret = ret ? ret : grb::dot( dotproduct, r, r, ring ); // norm_residual = r' * r;
			norm_residual = std::abs( dotproduct );
			assert( ret == SUCCESS );

			// compute sqrt to avoid underflow
			norm_residual = std::sqrt( norm_residual );

			// initial norm of residual
			const ResidualType norm_residual_initial { norm_residual };
			IOType old_r_dot_z { 0.0 }, r_dot_z { 0.0 }, beta { 0.0 };
			size_t iter { 0 };

#ifdef HPCG_PRINT_STEPS
			DBG_print_norm( p, "start p" );
			DBG_print_norm( Ap, "start Ap" );
			DBG_print_norm( r, "start r" );
#endif

			do {
#ifdef HPCG_PRINT_STEPS
				DBG_println( "========= iteration " << iter << " =========" );
#endif
				if( with_preconditioning ) {
					ret = ret ? ret : internal::multi_grid( data, data.coarser_level, presmoother_steps, postsmoother_steps, ring, minus );
					assert( ret == SUCCESS );
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

					ret = ret ? ret : grb::dot( r_dot_z, r, z, ring ); // r_dot_z = r' * z;
					assert( ret == SUCCESS );
				} else {
					old_r_dot_z = r_dot_z;

					r_dot_z = static_cast< IOType >( 0.0 );;
					ret = ret ? ret : grb::dot( r_dot_z, r, z, ring ); // r_dot_z = r' * z;
					assert( ret == SUCCESS );

					beta = r_dot_z / old_r_dot_z;
					ret = ret ? ret : grb::clear( Ap );                         // Ap  = 0;
					ret = ret ? ret : grb::eWiseMulAdd( Ap, beta, p, z, ring ); // Ap += beta * p + z;
					std::swap( Ap, p );                                         // p = Ap;
					assert( ret == SUCCESS );
				}
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( p, "middle p" );
#endif

				ret = ret ? ret : grb::set( Ap, 0 );
				ret = ret ? ret : grb::mxv( Ap, A, p, ring ); // Ap = A * p;
				assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( Ap, "middle Ap" );
#endif
				dotproduct =  static_cast< IOType >( 0.0 );
				ret = ret ? ret : grb::dot( dotproduct, Ap, p, ring ); // dotproduct  = p' * Ap
				assert( ret == SUCCESS );

				alpha = r_dot_z / dotproduct;

				ret = ret ? ret : grb::eWiseMul( x, alpha, p, ring ); // x += alpha * p;
				assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( x, "end x" );
#endif

				ret = ret ? ret : grb::eWiseMul( r, -alpha, Ap, ring ); // r += - alpha * Ap;
				assert( ret == SUCCESS );
#ifdef HPCG_PRINT_STEPS
				DBG_print_norm( r, "end r" );
#endif

				dotproduct =  static_cast< IOType >( 0.0 );
				ret = ret ? ret : grb::dot( dotproduct, r, r, ring ); // residual = r' * r;
				norm_residual = std::abs( dotproduct );
				assert( ret == SUCCESS );

				norm_residual = std::sqrt( norm_residual );

				++iter;
			} while( iter < max_iterations && norm_residual / norm_residual_initial > tolerance && ret == SUCCESS );

			iterations = iter;
			return ret;
		}

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_HPCG
