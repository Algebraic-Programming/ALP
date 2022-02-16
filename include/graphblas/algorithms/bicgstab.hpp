
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

/*
 * @author A. N. Yzelman
 * @date 15th of February, 2022
 *
 * Implementation time, to be taken with a pinch of salt:
 *  - 50 minutes, excluding error handling, documentation, and testing.
 *  - 10 minutes to get it to compile, once the smoke test was generated.
 *  - 15 minutes to incorporate proper error handling plus printing of warnings
 *    and errors.
 */

#ifndef _H_GRB_ALGORITHMS_BICGSTAB
#define _H_GRB_ALGORITHMS_BICGSTAB

#include <graphblas.hpp>

#include <iostream>
#include <type_traits>

#ifdef _DEBUG
 #include <cmath> // for sqrt, making the silent assumption that ResidualType
                  // is a supported type for it
#endif


namespace grb {

	namespace algorithms {

		/**
		 * Solves a linear system \f$ b = Ax \f$ with \f$ x \f$ unknown by using the
		 * bi-conjugate gradient (bi-CG) stabilised method; i.e., BiCGstab.
		 *
		 * @tparam descr        Any descriptor to use for the computation (optional).
		 * @tparam IOType       The solution vector element type.
		 * @tparam NonzeroType  The system matrix entry type.
		 * @tparam InputType    The element type of the right-hand side vector.
		 * @tparam ResidualType The type of the residuals used during computation.
		 *
		 * @tparam Semiring The semiring under which to perform the BiCGstab
		 * @tparam Minus    The inverse operator of the additive operator of
		 *                  \a Semiring.
		 * @tparam Divide   The inverse of the multiplicative operator of
		 *                  \a Semiring.
		 *
		 * By default, these will be the regular add, mul, subtract, and divide over
		 * the types \a IOType, \a NonzeroType, \a InputType, and/or \a ResidualType,
		 * as appropriate.
		 *
		 * Does not perform any preconditioning.
		 *
		 * @param[in,out] x On input: an initial guess to the solution \f$ Ax=b \f$.
		 *                  On output: if #grb::SUCCESS is returned, the solution to
		 *                  \f$ Ax=b \f$ within the given tolerance \a tol.
		 * @param[in]     A The square non-singular system matrix \f$ A \f$.
		 * @param[in]     b The dense right-hand side vector \f$ b \f$.
		 *
		 * If the size of \f$ A \f$ is \f$ n \times n \f$, then the sizes of \a x and
		 * \a b must be \f$ n \f$ also.
		 *
		 * A guess on input captured by \a x is optional. However, if \a x is dense
		 * on input, this algorithm will assume \a x is a valid initial guess.
		 *
		 * \warning grb::set( x, zero ); is not a valid initial guess. This algorithm
		 *          will return with an error if given this input.
		 *
		 * \note If unsure what constitutes a valid guess, it is recommended to call
		 *       #grb::clear(x) prior to this function. An initial guess will then be
		 *       provided.
		 *
		 * If there are zeroes in \a b, then this algorithm assumes they are explicit
		 * zeroes; \a b must be structurally dense.
		 *
		 * @param[in]     max_iterations The maximum number of iterations this
		 *                               algorithm may perform.
		 * @param[in]     tol            The relative tolerance which determines when
		 *                               an approximated solution \f$ x \f$ becomes
		 *                               acceptable. Must be positive and non-zero.
		 *
		 * @param[out]    iterations When #grb::SUCCESS is returned, the number of
		 *                           iterations that were required to obtain an
		 *                           acceptable approximate solution.
		 * @param[out]    residual   When #grb::SUCCESS is returned, the square of the
		 *                           2-norm of the residual; i.e., \f$ (r,r) \f$,
		 *                           where \f$ r = b - Ax \f$.
		 *
		 * @param[in] r, rhat, p, v, s, t Work space required for BiCGstab. These
		 *                                must all be vectors of length \f$ n \f$.
		 *
		 * Valid descriptors to this algorithm are:
		 *   -# descriptors::no_casting
		 *   -# descriptors::transpose
		 *
		 * @returns #grb::SUCCESS  If an acceptable solution is returned.
		 * @returns #grb::FAILED   If the algorithm failed to find an acceptable
		 *                         approximate solution.
		 * @returns #grb::MISMATCH If two or more of the input arguments have
		 *                         incompatible sizes.
		 * @returns #grb::ILLEGAL  If \a b is structurally sparse.
		 * @returns #grb::ILLEGAL  If \a tol is zero or negative.
		 *
		 * \par Performance semantics
		 *
		 *   -# This function does not allocate nor free dynamic memory, nor shall it
		 *      make any system calls.
		 *
		 * For performance semantics regarding work, inter-process data movement,
		 * intra-process data movement, synchronisations, and memory use, please see
		 * the specification of the ALP primitives this function relies on.
		 */
		template< Descriptor descr = descriptors::no_operation,
			typename IOType, typename NonzeroType, typename InputType,
			typename ResidualType,
			class Semiring = Semiring<
				operators::add< InputType, InputType, InputType >,
				operators::mul< IOType, NonzeroType, InputType >,
				identities::zero, identities::one
			>,
			class Minus = operators::subtract< ResidualType >,
			class Divide = operators::divide< ResidualType >
		>
		RC bicgstab( grb::Vector< IOType > &x,
			const grb::Matrix< NonzeroType > &A,
			const grb::Vector< InputType > &b,
			const size_t max_iterations,
			ResidualType tol,
			size_t &iterations,
			ResidualType &residual,
			Vector< InputType > &r,
			Vector< InputType > &rhat,
			Vector< InputType > &p,
			Vector< InputType > &v,
			Vector< InputType > &s,
			Vector< InputType > &t,
			const Semiring &semiring = Semiring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			// static checks
			static_assert( !( descr & descriptors::no_casting ) || (
					std::is_same< IOType, NonzeroType >::value &&
					std::is_same< IOType, InputType >::value &&
					std::is_same< IOType, ResidualType >::value
				), "no_casting descriptor was set but containers with differing domains "
				"were given."
			);
			static_assert( !( descr & descriptors::no_casting ) || (
					std::is_same< NonzeroType, typename Semiring::D1 >::value &&
					std::is_same< IOType, typename Semiring::D2 >::value &&
					std::is_same< InputType, typename Semiring::D3 >::value &&
					std::is_same< InputType, typename Semiring::D4 >::value
				), "no_casting descriptor was set, but semiring has incompatible domains "
				"with the given containers."
			);
			static_assert( !( descr & descriptors::no_casting ) || (
					std::is_same< InputType, typename Minus::D1 >::value &&
					std::is_same< InputType, typename Minus::D2 >::value &&
					std::is_same< InputType, typename Minus::D3 >::value
				), "no_casting descriptor was set, but given minus operator has "
				"incompatible domains with the given containers."
			);
			static_assert( !( descr & descriptors::no_casting ) || (
					std::is_same< ResidualType, typename Divide::D1 >::value &&
					std::is_same< ResidualType, typename Divide::D2 >::value &&
					std::is_same< ResidualType, typename Divide::D3 >::value
				), "no_casting descriptor was set, but given divide operator has "
				"incompatible domains with the given tolerance type."
			);
			static_assert( std::is_floating_point< ResidualType >::value,
				"Require floating-point residual type."
			);

#ifdef _DEBUG
			std::cout << "Entering bicgstab; "
				<< "tol = " << tol << ", "
				<< "max_iterations = " << max_iterations << "\n";
#endif

			// descriptor for indiciating dense computations
			constexpr Descriptor dense_descr = descr | descriptors::dense;

			// get an alias to zero and one in case 1 and 0 can't cast properly
			const ResidualType zero = semiring.template getZero< ResidualType >();
			const ResidualType one  = semiring.template getOne< ResidualType >();

			// dynamic checks
			const size_t n = nrows( A );
			if( n != ncols( A ) ) {
				return MISMATCH;
			}
			if( n != size( x ) ) {
				return MISMATCH;
			}
			if( n != size( b ) ) {
				return MISMATCH;
			}
			if( n != nnz( b ) ) {
				return ILLEGAL;
			}
			if( n != size( r ) || n != size( rhat ) || n != size( p ) ||
				n != size( p ) || n != size( s ) || n != size( t )
			) {
				return MISMATCH;
			}
			if( tol <= zero ) {
				return ILLEGAL;
			}

#ifdef _DEBUG
			std::cout << "\t dynamic run-time error checking passed\n";
#endif

			// prelude
			ResidualType b_norm_squared = zero;
			RC ret = dot< dense_descr >( b_norm_squared, b, b, semiring );
			if( ret ) {
				std::cerr << "Error: BiCGstab encountered \"" << toString(ret)
					<< "\" during computation of the norm of b\n";
				return ret;
			}

			// make it so that we do not need to take square roots when detecting
			// convergence
			tol *= tol;
			tol *= b_norm_squared;
#ifdef _DEBUG
			std::cout << "Effective squared relative tolerance is " << tol << "\n";
#endif

			// if no proper guess was given, make one
			// warn: the present code assumes that any dense input is a valid guess
			if( nnz( x ) != n ) {
				IOType guess = semiring.template getOne< IOType >();
				ret = foldl( guess, n, divide );
				ret = ret ? ret : set( x, guess );
			}

			// compute residual
			residual = zero;
			ret = ret ? ret : set( r, zero );
			ret = ret ? ret : mxv< dense_descr >( r, A, x, semiring ); // r = Ax
			ret = ret ? ret : foldr< dense_descr >( b, r, minus ); // r = b - Ax
			ret = ret ? ret : dot< dense_descr >( residual, r, r, semiring );

			// check for prelude error
			if( ret ) {
				std::cerr << "Error: BiCGstab encountered \"" << toString(ret)
					<< "\" during prelude\n";
				return ret;
			}

			// check if the guess was good enough
			if( residual < tol ) {
				return SUCCESS;
			}

#ifdef _DEBUG
			std::cout << "\t prelude completed\n";
#endif

			// start iterations
			ret = ret ? ret : set( rhat, r );
			ret = ret ? ret : set( p, zero );
			ret = ret ? ret : set( v, zero );
			ResidualType rho, rho_old, alpha, beta, omega, temp;
			rho_old = alpha = omega = one;
			iterations = 0;

			for( ; ret == SUCCESS && iterations < max_iterations; ++iterations ) {

#ifdef _DEBUG
				std::cout << "\t iteration " << iterations << " starts\n";
#endif

				// rho = ( rhat, r )
				rho = zero;
				ret = ret ? ret : dot< dense_descr >( rho, rhat, r, semiring );
#ifdef _DEBUG
				std::cout << "\t\t rho = " << rho << "\n";
#endif
				if( ret == SUCCESS && utils::equals( rho, zero, 2*n-1 ) ) {
					std::cerr << "Error: BiCGstab detects r at iteration " << iterations <<
						" is orthogonal to r-hat\n";
					return FAILED;
				}

				// beta = (rho / rho_old) * (alpha / omega)
				ret = ret ? ret : apply( beta, rho, rho_old, divide );
				ret = ret ? ret : apply( temp, alpha, omega, divide );
				ret = ret ? ret : foldl( beta, temp, semiring.getMultiplicativeOperator() );
#ifdef _DEBUG
				std::cout << "\t\t beta = " << beta << "\n";
#endif

				// p = r + beta ( p - omega * v )
				ret = ret ? ret : eWiseLambda( [&r,&beta,&p,&v,&omega,&semiring,&minus] (const size_t i) {
						InputType tmp;
						apply( tmp, omega, v[i], semiring.getMultiplicativeOperator() );
						foldl( p[ i ], tmp, minus );
						foldr( beta, p[ i ], semiring.getMultiplicativeOperator() );
						foldr( r[ i ], p[ i ], semiring.getAdditiveOperator() );
					}, v, b
				);

				// v = Ap
				ret = ret ? ret : set( v, zero );
				ret = ret ? ret : mxv< dense_descr >( v, A, p, semiring );

				// alpha = rho / (rhat, v)
				alpha = zero;
				ret = ret ? ret : dot< dense_descr >( alpha, rhat, v, semiring );
				if( utils::equals( alpha, zero, 2*n-1 ) ) {
					std::cerr << "Error: BiCGstab detects rhat is orthogonal to v=Ap "
						<< "at iteration " << iterations << ".\n";
					return FAILED;
				}
				ret = ret ? ret : foldr( rho, alpha, divide );
#ifdef _DEBUG
				std::cout << "\t\t alpha = " << alpha << "\n";
#endif

				// x += alpha * p is post-poned to either the pre-stabilisation exit, or
				// after the stabilisation step
				//ret = ret ? ret : eWiseMul( x, alpha, p, semiring );

				// s = r - alpha * v
				{
					ResidualType minus_alpha = zero;
					ret = ret ? ret : foldl( minus_alpha, alpha, minus );
					ret = ret ? ret : set( s, r );
					ret = ret ? ret : eWiseMul< dense_descr >( s, minus_alpha, v, semiring );
				}

				// check residual
				residual = zero;
				ret = ret ? ret : dot< dense_descr >( residual, s, s, semiring );
				assert( residual > zero ); // we just assert this one rather than checking for it
#ifdef _DEBUG
				std::cout << "\t\t running residual, pre-stabilisation: " << sqrt(residual) << "\n";
#endif
				if( ret == SUCCESS && residual < tol ) {
					// update result (x += alpha * p) and exit
					ret = eWiseMul< dense_descr >( x, alpha, p, semiring );
					return ret;
				}

				// t = As
				ret = ret ? ret : set( t, zero );
				ret = ret ? ret : mxv< dense_descr >( t, A, s, semiring );

				// omega = (t, s) / (t, t);
				omega = temp = zero;
				ret = ret ? ret : dot< dense_descr >( temp, t, s, semiring );
#ifdef _DEBUG
				std::cout << "\t\t (t, s) = " << temp << "\n";
#endif
				if( ret == SUCCESS && utils::equals( rho, zero, 2*n-1 ) ) {
					std::cerr << "Error: BiCGstab detects As at iteration " << iterations <<
						" is orthogonal to s\n";
					return FAILED;
				}
				ret = ret ? ret : dot< dense_descr >( omega, t, t, semiring );
#ifdef _DEBUG
				std::cout << "\t\t (t, t) = " << omega << "\n";
#endif
				assert( omega > zero ); // we just assert this one rather than checking for it
				ret = ret ? ret : foldr( temp, omega, divide );
#ifdef _DEBUG
				std::cout << "\t\t omega = " << omega << "\n";
#endif

				// x += alpha * p + omega * s
				ret = ret ? ret : eWiseMul< dense_descr >( x, alpha, p, semiring );
				ret = ret ? ret : eWiseMul< dense_descr >( x, omega, s, semiring );

				// r = s - omega * t
				{
					ResidualType minus_omega = zero;
					ret = ret ? ret : foldl( minus_omega, omega, minus );
					ret = ret ? ret : set( r, s );
					ret = ret ? ret : eWiseMul< dense_descr >( r, minus_omega, t, semiring );
				}

				// check residual
				residual = zero;
				ret = ret ? ret : dot< dense_descr >( residual, r, r, semiring );
				assert( residual > zero ); // we just assert this one rather than checking for it
#ifdef _DEBUG
				std::cout << "\t\t running residual, post-stabilisation: " << sqrt(residual) << ". "
					<< "Residual squared: " << residual << ".\n";
#endif
				if( ret == SUCCESS ) {
				       if( residual < tol ) { return SUCCESS; }
				
					// go to next iteration
					rho_old = rho;
				}
			}

			if( ret == SUCCESS ) {
				// if we are here, then we did not detect convergence
				std::cerr << "Warning: call to BiCGstab did not converge within "
					<< max_iterations << " iterations. Squared two-norm of the running "
					<< "residual is " << residual << ". "
					<< "Target residual squared: " << tol << ".\n";
				return FAILED;
			} else {
				// if we are here, we exited due to an ALP error code
				std::cerr << "Error: BiCGstab encountered error \"" << toString(ret)
					<< "\" while iterating to " << iterations << ", ";
				if( iterations == max_iterations ) {
					std::cerr << "which also is the maximum number of iterations.\n";
				} else {
					std::cerr << "which is below the maximum number of iterations of "
						<< max_iterations << "\n";
				}
				return ret;
			}
		}

	}
}

#endif // end _H_GRB_ALGORITHMS_BICGSTAB

