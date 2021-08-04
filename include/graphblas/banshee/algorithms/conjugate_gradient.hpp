
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
 * @author Dan Iorga
 * @brief Modification of the standard CG algorithm for Banshee
 */

#ifndef _H_GRB_ALGORITHMS_CONJUGATE_GRADIENT
#define _H_GRB_ALGORITHMS_CONJUGATE_GRADIENT

#include <cstdio>

#include <graphblas.hpp>

namespace grb {

	namespace algorithms {

		/**
		 * Implements a standard Conjugate Gradients (CG) method on arbitrary fields.
		 *
		 * @tparam descr        The user descriptor
		 * @tparam IOType       The input/output vector nonzero type
		 * @tparam ResidualType The type of the residual
		 * @tparam NonzeroType  The matrix nonzero type
		 * @tparam InputType    The right-hand side vector nonzero type
		 * @tparam Ring         The semiring under which to perform CG
		 * @tparam Minus        The minus operator corresponding to the inverse of the
		 *                      additive operator of the given \a Ring.
		 * @tparam Divide       The division operator corresponding to the inverse of
		 *                      the multiplicative operator of the given \a Ring.
		 *
		 * By default, i.e., if none of \a ring, \a minus, or \a divide (nor their
		 * types) are explicitly provided by the user, the natural field on double
		 * data types will be assumed.
		 *
		 * \note An abstraction of a field that encapsulates \a Ring, \a Minus, and
		 *       \a Divide may be more appropriate. This will also naturally ensure
		 *       that demands on domain types are met.
		 *
		 * \warning The current implementation does not do static domain checking yet.
		 *
		 * \note The banshee version differs from the regular version only in that no
		 *       support for sqrt was available at the time of writing. Therefore the
		 *       convergence check is done using the square of the given tolerance.
		 */
		template< Descriptor descr = descriptors::no_operation,
			typename IOType,
			typename ResidualType,
			typename NonzeroType,
			typename InputType,
			class Ring = Semiring< grb::operators::add< IOType >, grb::operators::mul< IOType >, grb::identities::zero, grb::identities::one >,
			class Minus = operators::subtract< double >,
			class Divide = operators::divide< double > >
		grb::RC conjugate_gradient( grb::Vector< IOType > & x,
			const grb::Matrix< NonzeroType > & A,
			const grb::Vector< InputType > & b,
			const size_t max_iterations,
			const ResidualType tol,
			size_t & iterations,
			ResidualType & residual,
			grb::Vector< IOType > & r,
			grb::Vector< IOType > & u,
			grb::Vector< IOType > & temp,
			const Ring & ring = Ring(),
			const Minus & minus = Minus(),
			const Divide & divide = Divide() ) {
			static_assert( std::is_floating_point< ResidualType >::value,
				"Can only use the CG algorithm with floating-point residual "
				"types." ); // unless some different norm were used: issue #89

			ResidualType alpha, sigma;
			grb::set( temp, 0 );
			grb::set( r, 0 );

			grb::RC ret = grb::mxv( temp, A, x, ring ); // temp = A * x
			assert( ret == SUCCESS );

			if( ret == SUCCESS ) {
				ret = grb::eWiseApply( r, b, temp, minus ); // r = b - temp;
				assert( ret == SUCCESS );
			}
			if( ret == SUCCESS ) {
				ret = grb::set( u, r ); // u = r;
				assert( ret == SUCCESS );
			}
			if( ret == SUCCESS ) {
				ret = grb::dot( sigma, r, r, ring ); // sigma = r' * r;
				assert( ret == SUCCESS );
			}

			// std::cout << "sigma = " << sigma << std::endl;

			size_t iter = 0;

			do {
				// std::cout << "Iteration: " << iter+1 << "\n\n" << std::endl;

				if( ret == SUCCESS ) {
					ret = grb::mxv( temp, A, u, ring ); // temp = A * u;
					assert( ret == SUCCESS );
				}
				if( ret == SUCCESS ) {
					ret = grb::dot( residual, temp, u, ring ); // residual = u' * temp
					assert( ret == SUCCESS );
				}
				if( ret == SUCCESS ) {
					ret = grb::apply( alpha, sigma, residual,
						divide ); // alpha = sigma / residual;
					// std::cout << "alpha-1 = " << alpha << std::endl;
					assert( ret == SUCCESS );
				}
				if( ret == SUCCESS ) {
					ret = grb::eWiseMulAdd( x, alpha, u, x, ring ); // x = x + alplha * u;
					assert( ret == SUCCESS );
				}
				if( ret == SUCCESS ) {
					ret = grb::eWiseMul( temp, alpha, temp, ring ); // temp = alplha * temp;
					assert( ret == SUCCESS );
				}
				if( ret == SUCCESS ) {
					ret = grb::eWiseApply( r, r, temp, minus ); // r = r - temp;
					assert( ret == SUCCESS );
				}
				if( ret == SUCCESS ) {
					ret = grb::dot( residual, r, r, ring ); // residual = r' * r;
					assert( ret == SUCCESS );
					// std::cout << "residual-2 = " << residual << std::endl;
				}
				if( ret == SUCCESS ) {
					if( residual < tol * tol ) {
						break;
					}

					ret = grb::apply( alpha, residual, sigma,
						divide ); // alpha = residual / sigma;
					assert( ret == SUCCESS );
				}
				if( ret == SUCCESS ) {
					ret = grb::eWiseMulAdd( u, alpha, u, r, ring ); // u = r + alpha * u;
					assert( ret == SUCCESS );
				}
				sigma = residual; // sigma = residual;
			} while( iter++ < max_iterations && ret == SUCCESS );

			// output
			iterations = iter;

			if( ret != SUCCESS ) {
				return FAILED;
			}
			return SUCCESS;
		}

	} // namespace algorithms

} // end namespace grb

#endif // end _H_GRB_ALGORITHMS_CONJUGATE_GRADIENT
