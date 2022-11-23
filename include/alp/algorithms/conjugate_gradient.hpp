
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
 * @author Aristeidis Mastoras
 */

#ifndef _H_ALP_ALGORITHMS_CONJUGATE_GRADIENT_ALP
#define _H_ALP_ALGORITHMS_CONJUGATE_GRADIENT_ALP

#include <cstdio>

#include <alp.hpp>

namespace alp {

	namespace algorithms {

		/**
		 * Solves a linear system \f$ b = Ax \f$ with \f$ x \f$ unknown by the Conjugate
		 * Gradients (CG) method on general fields.
		 *
		 * Does not perform any preconditioning.
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
		 * Valid descriptors to this algorithm are:
		 *   -# descriptors::no_casting
		 *
		 * By default, i.e., if none of \a ring, \a minus, or \a divide (nor their
		 * types) are explicitly provided by the user, the natural field on double
		 * data types will be assumed.
		 *
		 * \note An abstraction of a field that encapsulates \a Ring, \a Minus, and
		 *       \a Divide may be more appropriate. This will also naturally ensure
		 *       that demands on domain types are met.
		 *
		 * @param[in,out] x              On input: the initial guess to the solution.
		 *                               On output: the last computed approximation.
		 * @param[in]     A              The (square) positive semi-definite system
		 *                               matrix.
		 * @param[in]     b              The known right-hand side in \f$ Ax = b \f$.
		 * @param[in]     max_iterations The maximum number of CG iterations.
		 * @param[in]     tol            The requested relative tolerance.
		 * @param[out]    residual       The residual corresponding to output \a x.
		 * @param[in,out] r              A temporary vector of the same size as \a x.
		 * @param[in,out] u              A temporary vector of the same size as \a x.
		 * @param[in,out] temp           A temporary vector of the same size as \a x.
		 * @param[in]     ring           The semiring under which to perform the CG.
		 * @param[in]     minus          The inverse of the additive operator of
		 *                               \a ring.
		 * @param[in]     divide         The inverse of the multiplicative operator
		 *                               of \a ring.
		 *
		 * \todo There is a sqrt(...) operator that lives outside of the current
		 *       algebraic abstractions. Would be great if that could be eliminated.
		 *       See internal issue #89.
		 */
		template< Descriptor descr = descriptors::no_operation,
			typename IOType, typename IOStructure,
			typename ResidualType, typename ResidualStructure,
			typename NonzeroType, typename NonzeroStructure, typename NonzeroStorage, typename NonzeroView,
			typename InputType, typename InputStructure, typename InputStorage, typename InputView,
			class Ring = Semiring< operators::add< IOType >, operators::mul< IOType >, identities::zero, identities::one >,
			class Minus = operators::subtract< IOType >,
			class Divide = operators::divide< IOType > >
		RC conjugate_gradient( Vector< IOType, structures::General, Sparse > & x,
			const Matrix< NonzeroType, structures::SymmetricPositiveDefinite, Sparse > & A,
			const Vector< InputType, structures::General, Sparse > & b,
			const size_t max_iterations,
			Scalar< ResidualType > tol,
			size_t & iterations,
			Scalar< ResidualType, ResidualStructure > & residual,
			Vector< IOType, structures::General, Sparse > & r,
			Vector< IOType, structures::General, Sparse > & u,
			Vector< IOType, structures::General, Sparse > & temp,
			const Ring & ring = Ring(),
			const Minus & minus = Minus(),
			const Divide & divide = Divide() ) {

			// static checks
			static_assert( std::is_floating_point< ResidualType >::value,
				"Can only use the CG algorithm with floating-point residual "
				"types." ); // unless some different norm were used: issue #89
			static_assert( !( descr & descriptors::no_casting ) || (
					std::is_same< IOType, ResidualType >::value &&
					std::is_same< IOType, NonzeroType >::value &&
					std::is_same< IOType, InputType >::value
				), "One or more of the provided containers have differing element types "
				"while the no-casting descriptor has been supplied"
			);
			static_assert( !( descr & descriptors::no_casting ) || (
					std::is_same< NonzeroType, typename Ring::D1 >::value &&
					std::is_same< IOType, typename Ring::D2 >::value &&
					std::is_same< InputType, typename Ring::D3 >::value &&
					std::is_same< InputType, typename Ring::D4 >::value
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

			constexpr const Descriptor descr_dense = descr | descriptors::dense;
			Scalar< ResidualType > zero( ring.template getZero< ResidualType >() );
			const size_t n = ncols( A );

			// dynamic checks
			{
				const size_t m = nrows( A );
				if( size( x ) != n ) {
					return MISMATCH;
				}
				if( size( b ) != m ) {
					return MISMATCH;
				}
				if( size( r ) != n || size( u ) != n || size( temp ) != n ) {
					std::cerr << "Error: provided workspace vectors are not of the correct "
						<< "length.\n";
					return MISMATCH;
				}
				if( m != n ) {
					std::cerr << "Warning: alp::algorithms::conjugate_gradient requires "
						<< "square input matrices, but a non-square input matrix was "
						<< "given instead.\n";
					return ILLEGAL;
				}

				// capacities
				if( capacity( x ) != n ) {
					return ILLEGAL;
				}
				if( capacity( r ) != n || capacity( u ) != n || capacity( temp ) != n ) {
					return ILLEGAL;
				}

				// others
				if( tol <= zero ) {
					std::cerr << "Error: tolerance input to CG must be strictly positive\n";
					return ILLEGAL;
				}
			}

			// make x and b structurally dense (if not already) so that the remainder
			// algorithm can safely use the dense descriptor for faster operations
			{
				RC rc = SUCCESS;
				if( nnz( x ) != n ) {
					rc = set< descriptors::invert_mask | descriptors::structural >(
						x, x, zero
					);
				}
				if( rc != SUCCESS ) {
					return rc;
				}
				assert( nnz( x ) == n );
			}

			Scalar< ResidualType > alpha, sigma, bnorm;

			// temp = 0
			RC ret = set( temp, zero );

			// temp += A * x
			ret = ret ? ret : mxv< descr_dense >( temp, A, x, ring );

			// r = b - temp;
			ret = ret ? ret : set( r, zero );
			ret = ret ? ret : foldl( r, b, ring.getAdditiveMonoid() );
			assert( nnz( r ) == n );
			assert( nnz( temp ) == n );
			ret = ret ? ret : foldl< descr_dense >( r, temp, minus );
			assert( ret == SUCCESS );
			assert( nnz( r ) == n );

			// u = r;
			ret = ret ? ret : set( u, r );

			// sigma = r' * r;
			sigma = zero;
			ret = ret ? ret : dot< descr_dense >( sigma, r, r, ring );

			// bnorm = b' * b;
			bnorm = zero;
			ret = ret ? ret : dot< descr_dense >( bnorm, b, b, ring );

			if( ret == SUCCESS ) {
				foldl(tol, tol, ring.getMultiplicativeMonoid() );
				foldl(tol, bnorm, ring.getMultiplicativeMonoid() );
			}

			size_t iter = 0;

			do {
				// temp = 0
				set( temp, zero );

				// temp += A * u;
				ret = ret ? ret : mxv< descr_dense >( temp, A, u, ring );

				// residual = u' * temp
				residual = zero;
				ret = ret ? ret : dot< descr_dense >( residual, temp, u, ring );

				// alpha = sigma / residual;
				ret = ret ? ret : apply( alpha, sigma, residual, divide );

				// x = x + alpha * u;
				ret = ret ? ret : eWiseMul< descr_dense >( x, alpha, u, ring );

				// temp = alpha .* temp
				// Warning: operator-based foldr requires temp be dense
				ret = ret ? ret : foldr( alpha, temp, ring.getMultiplicativeMonoid() );

				// r = r - temp;
				ret = ret ? ret : foldl< descr_dense >( r, temp, minus );

				// residual = r' * r;
				residual = zero;
				ret = ret ? ret : dot< descr_dense >( residual, r, r, ring );

				if( ret == SUCCESS ) {
					if( residual < tol ) {
						break;
					}
				}

				// alpha = residual / sigma;
				ret = ret ? ret : apply( alpha, residual, sigma, divide );

				// temp = r + alpha * u;
				ret = ret ? ret : set( temp, r );
				ret = ret ? ret : eWiseMul< descr_dense >( temp, alpha, u, ring );
				assert( nnz( temp ) == size( temp ) );

				// u = temp
				std::swap( u, temp ); 

				// sigma = residual;
				sigma = residual;

			} while( iter++ < max_iterations && ret == SUCCESS );

			// output
			iterations = iter;

			if( ret != SUCCESS ) {
				return FAILED;
			} else {
				return SUCCESS;
			}
		}

	} // namespace algorithms

} // end namespace alp

#endif // end _H_GRB_ALGORITHMS_CONJUGATE_GRADIENT_ALP