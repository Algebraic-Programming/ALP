
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
 * @file
 *
 * Implements the CG algorithm
 *
 * @author Aristeidis Mastoras
 */

#ifndef _H_GRB_ALGORITHMS_CONJUGATE_GRADIENT
#define _H_GRB_ALGORITHMS_CONJUGATE_GRADIENT

#include <cstdio>
#include <cmath>

#include <graphblas.hpp>
#include <graphblas/utils/iscomplex.hpp>


namespace grb {

	namespace algorithms {

		/**
		 * Solves a preconditioned linear system \f$ b = M^{-1}Ax \f$ with \f$ x \f$
		 * unknown by the Conjugate Gradients (CG) method on general fields.
		 *
		 * The preconditioner \a M, alike to the system matrix \a A, must be symmetric
		 * positive definite.
		 *
		 * @tparam descr          The user descriptor
		 * @tparam preconditioned Whether to apply any given preconditioners.
		 *
		 * \note The default value for \a preconditioned is <tt>true</tt> and it is
		 *       normally not necessary to override this value: if wishing to call a
		 *       non-preconditioned CG, please see #conjugate_gradient.
		 *
		 * @tparam IOType         The input/output vector nonzero type
		 * @tparam ResidualType   The type of the residual
		 * @tparam NonzeroType    The matrix nonzero type
		 * @tparam InputType      The right-hand side vector nonzero type
		 * @tparam Ring           The semiring under which to perform CG
		 * @tparam Minus          The minus operator corresponding to the inverse of
		 *                        the additive operator of the given \a Ring.
		 * @tparam Divide         The division operator corresponding to the inverse
		 *                        of the multiplicative operator of the given \a Ring.
		 *
		 * Valid descriptors to this algorithm are:
		 *   -# descriptors::no_casting
		 *   -# descriptors::transpose
		 *
		 * By default, i.e., if none of \a ring, \a minus, or \a divide (nor their
		 * types) are explicitly provided by the user, the natural field on double
		 * data types will be assumed.
		 *
		 * \note An abstraction of a field that encapsulates \a Ring, \a Minus, and
		 *       \a Divide may be more appropriate. This will also naturally ensure
		 *       that demands on domain types are met.
		 *
		 * \todo There is a sqrt(...) operator that lives outside of the current
		 *       algebraic abstractions. Would be great if that could be eliminated;
		 *       see, e.g., the approach taken by BiCGstab implementation. An
		 *       alternative solution is sketched in internal issue #89.
		 *
		 * @param[in,out] x              On input: an initial guess to the solution.
		 *                               On output: the last computed approximation.
		 * @param[in]     A              The (square) positive semi-definite system
		 *                               matrix.
		 * @param[in]     b              The known right-hand side in \f$ Ax = b \f$.
		 *                               Must be structurally dense.
		 *
		 * The preconditioner action \f$ M^{-1}r \f$ is supplied as a standard
		 * <tt>std::function</tt> that outputs a #grb::RC and takes two arguments:
		 *  -# the output vector of the preconditioning action as a reference to a
		 *     #grb::Vector with element type <tt>IOType</tt>, and
		 *  -# the input vector to the preconditioner as a const-reference to a
		 *     #grb::Vector with the same element type.
		 * Both vectors are guaranteed to be dense when given to \a Minv.
		 *
		 * A good preconditioner action \a Minv is both efficient to apply as well as
		 * drastrically reduces the number of CG iterations required. The latter is
		 * achieved by constructing \a Minv so that the condition number of
		 * \f$ M^{-1}A \f$ is much smaller than that of \f$ A \f$.
		 *
		 * @param[in]     Minv           The preconditioner action if
		 *                               \a preconditioned equals <tt>true</tt>.
		 *
		 * If \a A is \f$ n \times n \f$, then \a x and \a b must have matching length
		 * \f$ n \f$. The vector \a x furthermore must have a capacity of \f$ n \f$.
		 *
		 * CG algorithm inputs:
		 *
		 * @param[in]     max_iterations The maximum number of CG iterations. Must be
		 *                               larger than zero.
		 * @param[in]     tol            The requested relative tolerance.
		 *
		 * Additional outputs (besides \a x):
		 *
		 * @param[out]    iterations     The number of iterations the algorithm has
		 *                               started.
		 * @param[out]    residual       The residual corresponding to output \a x.
		 *
		 * The CG algorithm requires three workspace buffers with capacity \f$ n \f$:
		 *
		 * @param[in,out] r              A temporary vector of the same size as \a x.
		 * @param[in,out] u              A temporary vector of the same size as \a x.
		 * @param[in,out] temp           A temporary vector of the same size as \a x.
		 * @param[in,out] temp_precond   A temporary vector of the same size as \a x.
		 *
		 * \note If \a preconditioned is <tt>false</tt>, then both \a Minv and
		 *       \a temp_precond are ignored. In this case, \a temp_precond need not
		 *       have the same length as \a x, nor need it have full capacity.
		 *
		 * Finally, the algebraic structures over which the CG is executed are given:
		 *
		 * @param[in]     ring           The semiring under which to perform the CG.
		 * @param[in]     minus          The inverse of the additive operator of
		 *                               \a ring.
		 * @param[in]     divide         The inverse of the multiplicative operator
		 *                               of \a ring.
		 *
		 * This algorithm may return one of the following error codes:
		 *
		 * @returns #grb::SUCCESS  When the algorithm has converged to a solution
		 *                         within the given \a max_iterations and \a tol.
		 * @returns #grb::FAILED   When the algorithm did not converge within the
		 *                         given \a max_iterations.
		 * @returns #grb::ILLEGAL  When \a A is not square.
		 * @returns #grb::MISMATCH When \a x or \a b does not match the size of \a A.
		 * @returns #grb::ILLEGAL  When \a x does not have capacity \f$ n \f$.
		 * @returns #grb::ILLEGAL  When at least one of the required workspace vectors
		 *                         does not have capacity \f$ n \f$.
		 * @returns #grb::ILLEGAL  If \a tol is not strictly positive.
		 * @returns #grb::PANIC    If an unrecoverable error has been encountered. The
		 *                         output as well as the state of ALP/GraphBLAS is
		 *                         undefined.
		 *
		 * On output, the contents of the workspace \a r, \a u, and \a temp are
		 * always undefined. For non-#grb::SUCCESS error codes, additional containers
		 * or states may be left undefined:
		 * -# when #grb::PANIC is returned, the entire program state, including the
		 *    contents of all containers, become undefined;
		 * -# when #grb::ILLEGAL or #grb::MISMATCH are returned and \a iterations
		 *    equals zero, then all outputs are left unmodified compared to their
		 *    contents at function entry;
		 * -# when #grb::ILLEGAL or #grb::MISMATCH are returned and \a iterations is
		 *    nonzero, then the contents of \a x are undefined.
		 *
		 * \par Performance semantics
		 *
		 *   -# This function does not allocate nor free dynamic memory, nor shall it
		 *      make any system calls.
		 *
		 * For performance semantics regarding work, inter-process data movement,
		 * intra-process data movement, synchronisations, and memory use, please see
		 * the specification of the ALP primitives this function relies on. These
		 * performance semantics, with the exception of getters such as #grb::nnz, are
		 * specific to the backend selected during compilation.
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			bool preconditioned = true,
			typename IOType,
			typename ResidualType,
			typename NonzeroType,
			typename InputType,
			class Ring = Semiring<
				grb::operators::add< IOType >, grb::operators::mul< IOType >,
				grb::identities::zero, grb::identities::one
			>,
			class Minus = operators::subtract< IOType >,
			class Divide = operators::divide< IOType >,
			typename RSI, typename NZI, Backend backend
		>
		grb::RC preconditioned_conjugate_gradient(
			grb::Vector< IOType, backend > &x,
			const grb::Matrix< NonzeroType, backend, RSI, RSI, NZI > &A,
			const grb::Vector< InputType, backend > &b,
			const std::function<
					grb::RC(grb::Vector< IOType > &, const grb::Vector< IOType > &)
				> &Minv,
			const size_t max_iterations,
			ResidualType tol,
			size_t &iterations,
			ResidualType &residual,
			grb::Vector< IOType, backend > &r,
			grb::Vector< IOType, backend > &u,
			grb::Vector< IOType, backend > &temp,
			grb::Vector< IOType, backend > &temp_precond,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
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
			const ResidualType zero_residual = ring.template getZero< ResidualType >();
			const IOType zero = ring.template getZero< IOType >();
			const size_t n = grb::ncols( A );

			// retrieve conditional buffers
			grb::Vector< IOType > &z = preconditioned ? temp_precond : r;

			// dynamic checks
			{
				const size_t m = grb::nrows( A );
				if( grb::size( x ) != n ) {
					std::cerr << "Error: initial solution guess and output vector length ("
						<< size( x ) << ") does not match matrix size (" << m << ").\n";
					return grb::MISMATCH;
				}
				if( grb::size( b ) != m ) {
					std::cerr << "Error: right-hand side size (" << grb::size( b ) << ") does "
						<< "not match matrix size (" << m << ").\n";
					return grb::MISMATCH;
				}
				if( grb::size( r ) != n || grb::size( u ) != n || grb::size( temp ) != n ) {
					std::cerr << "Error: provided workspace vectors are not of the correct "
						<< "length.\n";
					return grb::MISMATCH;
				}
				if( preconditioned && grb::size( temp_precond ) != n ) {
					std::cerr << "Error: (left) preconditioner workspace vector does not have "
						<< "the correct length.\n";
					return grb::MISMATCH;
				}
				if( m != n ) {
					std::cerr << "Warning: grb::algorithms::conjugate_gradient requires "
						<< "square input matrices, but a non-square input matrix was "
						<< "given instead.\n";
					return grb::ILLEGAL;
				}

				// capacities
				if( grb::capacity( x ) != n ) {
					return grb::ILLEGAL;
				}
				if( grb::capacity( r ) != n || grb::capacity( u ) != n ||
					grb::capacity( temp ) != n
				) {
					return grb::ILLEGAL;
				}
				if( preconditioned && grb::capacity( temp_precond ) != n ) {
					return grb::ILLEGAL;
				}

				// others
				if( tol <= zero_residual ) {
					std::cerr << "Error: tolerance input to CG must be strictly positive\n";
					return grb::ILLEGAL;
				}
				if( max_iterations == 0 ) {
					std::cerr << "Error: at least one CG iteration must be requested\n";
					return grb::ILLEGAL;
				}
			}

			// set pure output fields to neutral defaults
			iterations = 0;
			residual = std::numeric_limits< double >::infinity();

			// make x and b structurally dense (if not already) so that the remainder
			// algorithm can safely use the dense descriptor for faster operations
			{
				RC rc = grb::SUCCESS;
				if( nnz( x ) != n ) {
					rc = grb::set< descriptors::invert_mask | descriptors::structural >(
						x, x, zero
					);
				}
				if( rc != grb::SUCCESS ) { return rc; }
				assert( nnz( x ) == n );
			}

			IOType sigma, bnorm, alpha, beta;

			// r = b - temp;
			grb::RC ret = grb::set( temp, 0 ); assert( ret == grb::SUCCESS );
			ret = ret ? ret : grb::mxv< descr_dense >( temp, A, x, ring );
			assert( ret == grb::SUCCESS );
			ret = ret ? ret : grb::set( r, zero ); assert( ret == grb::SUCCESS );
			// note: no dense descriptor since we actually allow sparse b
			ret = ret ? ret : grb::foldl( r, b, ring.getAdditiveMonoid() );
			// from here onwards, r, temp, x are dense and will remain so
			assert( nnz( r ) == n );
			assert( nnz( temp ) == n );
			ret = ret ? ret : grb::foldl< descr_dense >( r, temp, minus );
			assert( ret == SUCCESS );

			// z = M^-1r
			if( preconditioned ) {
				ret = ret ? ret : grb::set( z, 0 ); // also ensures z is dense, henceforth
				ret = ret ? ret : Minv( z, r );
			} // else, z equals r (by reference)

			// u = z;
			ret = ret ? ret : grb::set( u, z );
			assert( ret == grb::SUCCESS );
			// from here onwards, u is dense; i.e., all vectors are dense from now on,
			// and we can freely use the dense descriptor in the subsequent

			// sigma = r' * z;
			sigma = zero;
			ret = ret ? ret : grb::dot< descr_dense >(
					sigma,
					r, z,
					ring.getAdditiveMonoid(),
					grb::operators::conjugate_right_mul< IOType >()
				);

			assert( ret == grb::SUCCESS );

			// bnorm = b' * b;
			bnorm = zero;
			ret = ret ? ret : grb::dot< descr_dense >(
				bnorm,
				b, b,
				ring.getAdditiveMonoid(),
				grb::operators::conjugate_left_mul< IOType >() );
			assert( ret == grb::SUCCESS );

			// get effective tolerance and exit on any error during prelude
			if( ret == grb::SUCCESS ) {
				tol *= std::sqrt( grb::utils::is_complex< IOType >::modulus( bnorm ) );
			} else {
				std::cerr << "Warning: preconditioned CG caught error during prelude ("
					<< grb::toString( ret ) << ")\n";
				return ret;
			}

			// all OK; perform main iterations
			size_t iter = 0;
			do {
				assert( iter < max_iterations );
				(void) ++iter;

				// temp = A * u;
				ret = ret ? ret : grb::set< descr_dense >( temp, 0 );
				assert( ret == grb::SUCCESS );
				ret = ret ? ret : grb::mxv< descr_dense >( temp, A, u, ring );
				assert( ret == grb::SUCCESS );

				// beta = (A * u)' * u;
				beta = zero;
				ret = ret ? ret : grb::dot< descr_dense >(
						beta,
						temp, u,
						ring.getAdditiveMonoid(),
						grb::operators::conjugate_right_mul< IOType >()
					);
				assert( ret == grb::SUCCESS );

				// alpha = sigma / beta;
				ret = ret ? ret : grb::apply( alpha, sigma, beta, divide );
				assert( ret == grb::SUCCESS );

				// x = x + alpha * u;
				ret = ret ? ret : grb::eWiseMul< descr_dense >( x, alpha, u, ring );
				assert( ret == grb::SUCCESS );

				// r = r - alpha .* temp = r - alpha .* (A * u);
				ret = ret ? ret : grb::foldr< descr_dense >( alpha, temp,
					ring.getMultiplicativeMonoid() );
				assert( ret == grb::SUCCESS );
				ret = ret ? ret : grb::foldl< descr_dense >( r, temp, minus );
				assert( ret == SUCCESS );

				// get residual. In the preconditioned case, the resulting scalar is *not*
				// used for subsequent operations. Therefore, we first compute the residual
				// using alpha as a temporary scalar
				alpha = zero;
				ret = ret ? ret : grb::dot< descr_dense >(
						alpha,
						r, r,
						ring.getAdditiveMonoid(),
						grb::operators::conjugate_left_mul< IOType >()
					);
				assert( ret == grb::SUCCESS );
				residual = grb::utils::is_complex< IOType >::modulus( alpha );

				// check residual
				if( ret == grb::SUCCESS ) {
					if( sqrt( residual ) < tol || iter >= max_iterations ) { break; }
				}

				// apply preconditioner action (if required), and compute beta for the
				// preconditioned case
				// z = M^-1r
				// beta = r' * z
				if( preconditioned ) {
					beta = zero;
					ret = ret ? ret : Minv( z, r ); assert( ret == grb::SUCCESS );
					ret = ret ? ret : grb::dot< descr_dense >(
							beta,
							r, z,
							ring.getAdditiveMonoid(),
							grb::operators::conjugate_right_mul< IOType >()
						);
					assert( ret == grb::SUCCESS );
				} else {
					beta = alpha;
				}

				// alpha = beta / sigma;
				ret = ret ? ret : grb::apply( alpha, beta, sigma, divide );
				assert( ret == grb::SUCCESS );

				// u_next = z + beta * u_previous;
				ret = ret ? ret : grb::foldr< descr_dense >( alpha, u,
					ring.getMultiplicativeMonoid() );
				assert( ret == grb::SUCCESS );
				ret = ret ? ret : grb::foldr< descr_dense >( z, u,
					ring.getAdditiveMonoid() );
				assert( ret == grb::SUCCESS );

				sigma = beta;
			} while( ret == grb::SUCCESS );

			// output that is independent of error code
			iterations = iter;

			// return correct error code
			if( ret == grb::SUCCESS ) {
				if( std::sqrt( residual ) >= tol ) {
					// did not converge within iterations
					return grb::FAILED;
				}
			}
			return ret;
		}

		/**
		 * Solves a linear system \f$ b = Ax \f$ with \f$ x \f$ unknown by the
		 * Conjugate Gradients (CG) method on general fields.
		 *
		 * Does not perform any preconditioning-- for preconditioned CG and full
		 * documentation, please see #preconditioned_conjugate_gradient.
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename IOType,
			typename ResidualType,
			typename NonzeroType,
			typename InputType,
			class Ring = Semiring<
				grb::operators::add< IOType >, grb::operators::mul< IOType >,
				grb::identities::zero, grb::identities::one
			>,
			class Minus = operators::subtract< IOType >,
			class Divide = operators::divide< IOType >,
			typename RSI, typename NZI, Backend backend
		>
		grb::RC conjugate_gradient(
			grb::Vector< IOType, backend > &x,
			const grb::Matrix< NonzeroType, backend, RSI, RSI, NZI > &A,
			const grb::Vector< InputType, backend > &b,
			const size_t max_iterations,
			ResidualType tol,
			size_t &iterations,
			ResidualType &residual,
			grb::Vector< IOType, backend > &r,
			grb::Vector< IOType, backend > &u,
			grb::Vector< IOType, backend > &temp,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
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

			// create a dummy preconditioner and buffer that will never be used
			std::function<
				grb::RC( grb::Vector< IOType >&, const grb::Vector< IOType >& )
			> dummy_preconditioner =
				[](grb::Vector< IOType >&, const grb::Vector< IOType >&) -> grb::RC {
					return grb::FAILED;
				};
			grb::Vector< IOType > dummy_buffer( 0 );

			// call PCG with preconditioning disabled
			return preconditioned_conjugate_gradient< descr, false >(
					x, A, b,
					dummy_preconditioner,
					max_iterations, tol,
					iterations, residual,
					r, u, temp, dummy_buffer,
					ring, minus, divide
				);
		}

	} // namespace algorithms

} // end namespace grb

#endif // end _H_GRB_ALGORITHMS_CONJUGATE_GRADIENT

