
/*
 *   Copyright 2023 Huawei Technologies Co., Ltd.
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
 * @author Denis Jelovina
 */

#ifndef _H_GRB_ALGORITHMS_GMRES
#define _H_GRB_ALGORITHMS_GMRES

#include <cstdio>
#include <complex>

#include <graphblas.hpp>
#include <graphblas/utils/iscomplex.hpp>


namespace grb {

	namespace algorithms {

		/**
		 * Solves a linear system \f$ b = Ax \f$ with \f$ x \f$ unknown
		 * by the GMRES method on general fields.
		 *
		 * Preconditioning is possible by providing
		 * an initialised matrix \f$ M \f$ of matching size.
		 *
		 * @tparam descr        The user descriptor
		 * @tparam IOType       The input/output vector nonzero type
		 * @tparam ResidualType The type of the residual
		 * @tparam NonzeroType  The matrix nonzero type
		 * @tparam InputType    The right-hand side vector nonzero type
		 * @tparam Ring         The semiring under which to perform GMRES
		 * @tparam Minus        The minus operator corresponding to the inverse of the
		 *                      additive operator of the given \a Ring.
		 * @tparam Divide       The division operator corresponding to the inverse of
		 *                      the multiplicative operator of the given \a Ring.
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
		 *
		 * @param[in]     x              On input: an initial guess to the solution.
		 * @param[in]     A              The (square) positive semi-definite system
		 *                               matrix.
		 * @param[in]     b              The known right-hand side in \f$ Ax = b \f$.
		 *                               Must be structurally dense.
		 *
		 * If \a A is \f$ n \times n \f$, then \a x and \a b must have matching length
		 * \f$ n \f$. The vector \a x furthermore must have a capacity of \f$ n \f$.
		 *
		 * GMRES algorithm inputs:
		 *
		 * @param[in]     max_iterations The maximum number of GMRES iterations.
		 * @param[in]     n_restart      The number of GMRES restart iterations.
		 * @param[in]     tol            The requested relative tolerance.
		 * @param[out]    HMatrix        Upper-Hessenberg matrix of size \a n_restart x
		 *                               \a n_restart with row-major orientation.
		 *                               On output only fist \a iterations
		 *                               columns and \a iterations rows have meaning.
		 * @param[out]    Q              std::vector of length \a n_restart + 1.
		 *                               Each element of Q is grb::Vector of size
		 *                               \f$ n \f$. On output only fist \a n_restart
		 *                               vectors have meaning.
		 *
		 * Additional outputs (besides \a x):
		 *
		 * @param[out]    iterations     The number of iterations the algorithm has
		 *                               performed.
		 * @param[out]    residual       The residual corresponding to output \a x.
		 *
		 * The GMRES algorithm requires three workspace buffers with capacity \f$ n \f$:
		 *
		 * @param[in,out] r              A temporary vector of the same size as \a x.
		 * @param[in,out] u              A temporary vector of the same size as \a x.
		 * @param[in,out] temp           A temporary vector of the same size as \a x.
		 *
		 * Finally, the algebraic structures over which the GMRES is executed are given:
		 *
		 * @param[in]     ring           The semiring under which to perform the GMRES.
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
		 * @returns #grb::ILLEGAL  When at least one of the workspace vectors does not
		 *                         have capacity \f$ n \f$.
		 * @returns #grb::ILLEGAL  If \a tol is not strictly positive.
		 * @returns #grb::PANIC    If an unrecoverable error has been encountered. The
		 *                         output as well as the state of ALP/GraphBLAS is
		 *                         undefined.
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
			typename IOType,
			typename ResidualType,
			typename NonzeroType,
			typename InputType,
			class Ring = Semiring<
				grb::operators::add< IOType >, grb::operators::mul< IOType >,
				grb::identities::zero, grb::identities::one
			>,
			class Minus = operators::subtract< IOType >,
			class Divide = operators::divide< IOType >
		>
		grb::RC gmres(
			const grb::Vector< IOType > &x,
			const grb::Matrix< NonzeroType > &A,
			const grb::Vector< InputType > &b,
			std::vector< NonzeroType > &Hmatrix,
			std::vector< grb::Vector< NonzeroType > > &Q,
			const size_t n_restart,
			ResidualType tol,
			size_t &iterations,
			grb::Vector< IOType > &temp,
			const grb::Matrix< NonzeroType > &M = grb::Matrix< NonzeroType >( 0, 0 ),
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			// static checks
			static_assert( std::is_floating_point< ResidualType >::value,
				"Can only use the GMRES algorithm with floating-point residual "
				"types." );
			static_assert(
				!( descr & descriptors::no_casting ) ||
				(
					std::is_same< IOType, ResidualType >::value &&
					std::is_same< IOType, NonzeroType >::value &&
					std::is_same< IOType, InputType >::value
				),
				"One or more of the provided containers have differing element types "
				"while the no-casting descriptor has been supplied"
			);
			static_assert(
				!( descr & descriptors::no_casting ) ||
				(
					std::is_same< NonzeroType, typename Ring::D1 >::value &&
					std::is_same< IOType, typename Ring::D2 >::value &&
					std::is_same< InputType, typename Ring::D3 >::value &&
					std::is_same< InputType, typename Ring::D4 >::value
				),
				"no_casting descriptor was set, but semiring has incompatible domains "
				"with the given containers."
			);
			static_assert(
				!( descr & descriptors::no_casting ) ||
				(
					std::is_same< InputType, typename Minus::D1 >::value &&
					std::is_same< InputType, typename Minus::D2 >::value &&
					std::is_same< InputType, typename Minus::D3 >::value
				),
				"no_casting descriptor was set, but given minus operator has "
				"incompatible domains with the given containers."
			);
			static_assert(
				!( descr & descriptors::no_casting ) ||
				(
					std::is_same< ResidualType, typename Divide::D1 >::value &&
					std::is_same< ResidualType, typename Divide::D2 >::value &&
					std::is_same< ResidualType, typename Divide::D3 >::value
				),
				"no_casting descriptor was set, but given divide operator has "
				"incompatible domains with the given tolerance type."
			);
			static_assert(
				std::is_floating_point< ResidualType >::value,
				"Require floating-point residual type."
			);

			bool useprecond = false;
			if( ( nrows( M ) != 0 ) && ( ncols( M ) != 0 ) ) {
				useprecond = true;
			}

			constexpr const Descriptor descr_dense = descr | descriptors::dense;
			const ResidualType zero = ring.template getZero< ResidualType >();
			const size_t n = grb::ncols( A );

			// dynamic checks
			{
				const size_t m = grb::nrows( A );
				if( size( x ) != n ) {
					return MISMATCH;
				}
				if( size( b ) != m ) {
					return MISMATCH;
				}
				if( size( Q[ 0 ] ) != n || size( temp ) != n ) {
					std::cerr << "Error: provided workspace vectors are not of the correct "
						<< "length.\n";
					return MISMATCH;
				}
				if( m != n ) {
					std::cerr << "Warning: grb::algorithms::conjugate_gradient requires "
						<< "square input matrices, but a non-square input matrix was "
						<< "given instead.\n";
					return ILLEGAL;
				}

				// capacities
				if( capacity( x ) != n ) {
					return ILLEGAL;
				}
				if( capacity( Q[ 0 ] ) != n || capacity( temp ) != n ) {
					return ILLEGAL;
				}

				// others
				if( tol <= zero ) {
					std::cerr << "Error: tolerance input to GMRES must be strictly positive\n";
					return ILLEGAL;
				}
			}

			ResidualType rho, tau;
			NonzeroType alpha;

			// (re)set Hmatrix to zero
			std::fill( Hmatrix.begin(), Hmatrix.end(), zero );

			//Q[:,0]=b-A.dot(x) ;
			// temp = 0
			grb::RC ret = grb::set( temp, zero );
			assert( ret == SUCCESS );

			// temp = A * x
			ret = ret ? ret : grb::mxv< descr_dense >( temp, A, x, ring );
			assert( ret == SUCCESS );

			// Q[ 0 ] = b - temp;
			ret = ret ? ret : grb::set( Q[ 0 ], zero );
			ret = ret ? ret : grb::foldl( Q[ 0 ], b, ring.getAdditiveMonoid() );
			assert( nnz( Q[ 0 ] ) == n );
			assert( nnz( temp ) == n );
			ret = ret ? ret : grb::foldl< descr_dense >( Q[ 0 ], temp, minus );
			assert( ret == SUCCESS );

			// precond
			if( useprecond ) {
				// Q[ 0 ] = M * Q[ 0 ]
				ret = grb::set( temp, Q[ 0 ] );
				assert( ret == SUCCESS );

				ret = grb::set( Q[ 0 ], zero );
				assert( ret == SUCCESS );

				ret = ret ? ret : grb::mxv< descr_dense >( Q[ 0 ], M, temp, ring );
				assert( ret == SUCCESS );
			}

			//rho=norm(Q[:,0])
			alpha = zero;
			if( grb::utils::is_complex< IOType >::value ) {
				ret = ret ? ret : grb::eWiseLambda( [&,Q]( const size_t i ) {
					temp[ i ] = grb::utils::is_complex< IOType >::conjugate( Q[ 0 ] [ i ] );
					}, temp
				);
				ret = ret ? ret : grb::dot< descr_dense >( alpha, temp, Q[ 0 ], ring );
			} else {
				ret = ret ? ret : grb::dot< descr_dense >( alpha, Q[ 0 ], Q[ 0 ], ring );
			}
			assert( ret == SUCCESS );

			rho = sqrt( grb::utils::is_complex< IOType >::modulus( alpha ) );
			Hmatrix[ 0 ] = rho;

			tau = tol * rho;

			size_t k = 0;
			while( ( rho > tau ) && ( k < n_restart ) ) {
				// alpha = r' * r;
				alpha = Hmatrix[ k * ( n_restart + 1 ) + k ];

				if( std::abs( alpha ) < tol ) {
					break;
				}

				// Q[k] = Q[k] / alpha
				ret = ret ? ret : grb::foldl( Q[ k ], alpha, divide );
				assert( ret == SUCCESS );

				// Q[k+1]=0
				ret = ret ? ret : grb::set( Q[ k + 1 ], zero );
				assert( ret == SUCCESS );

				// Q[k+1]= A * Q[k+1]
				ret = ret ? ret : grb::mxv< descr_dense >( Q[ k + 1 ], A, Q[ k ], ring );
				assert( ret == SUCCESS );

				// precond
				if( useprecond ) {
					// Q[k+1]= M * Q[k+1]
					ret = grb::set( temp, Q[ k+1 ] );
					assert( ret == SUCCESS );

					ret = grb::set( Q[ k+1 ], zero );
					assert( ret == SUCCESS );

					ret = ret ? ret : grb::mxv< descr_dense >( Q[ k+1 ], M, temp, ring );
					assert( ret == SUCCESS );
				}

				k++;

				for( size_t j = 0; j < std::min( k, n_restart ); j++ ) {
					//H[j,k]=Q[:,j].dot(Q[:,k])
					Hmatrix[ k * ( n_restart + 1 ) + j ] = zero;
					if( grb::utils::is_complex< IOType >::value ) {
						ret = ret ? ret : grb::eWiseLambda( [&,Q]( const size_t i ) {
							temp[ i ] = grb::utils::is_complex< IOType >::conjugate( Q[ j ] [ i ] );
						},
							temp
						);
						ret = ret ? ret : grb::dot< descr_dense >( Hmatrix[ k * ( n_restart + 1 ) + j ], Q[ k ], temp, ring );
					}
					else {
						ret = ret ? ret : grb::dot< descr_dense >( Hmatrix[ k * ( n_restart + 1 ) + j ], Q[ k ], Q[ j ], ring );
					}
					assert( ret == SUCCESS );

					//Q[:,k]=Q[:,k]-H[j,k]*Q[:,i]
					grb::RC ret = grb::set( temp, zero );
					assert( ret == SUCCESS );

					NonzeroType alpha1 = Hmatrix[ k * ( n_restart + 1 ) + j ];
					ret = ret ? ret : grb::eWiseMul< descr_dense >( temp, alpha1, Q[ j ], ring );
					assert( ret == SUCCESS );

					ret = ret ? ret : grb::foldl< descr_dense >( Q[ k ], temp, minus );
					assert( ret == SUCCESS );

				} // while

				//rho=norm(Q[:,k])
				alpha = zero;
				if( grb::utils::is_complex< IOType >::value ) {
					grb::RC ret = grb::set( temp, zero );
					assert( ret == SUCCESS );

					ret = ret ? ret : grb::eWiseLambda( [&,Q,k]( const size_t i ) {
						temp[ i ] = grb::utils::is_complex< IOType >::conjugate( Q[ k ] [ i ] );
					}, temp
						);
					ret = ret ? ret : grb::dot< descr_dense >( alpha, temp, Q[ k ], ring );
				} else {
					ret = ret ? ret : grb::dot< descr_dense >( alpha, Q[ k ], Q[ k ], ring );
				}
				assert( ret == SUCCESS );

				//H[k,k]=rho
				Hmatrix[ k * ( n_restart + 1 ) + k ] = sqrt(
					grb::utils::is_complex< IOType >::modulus( alpha )
				) ;

			}

			iterations = iterations + k;

			if( ret != SUCCESS ) {
				return FAILED;
			} else {
				return SUCCESS;
			}
		}

	} // namespace algorithms

} // end namespace grb

#endif // end _H_GRB_ALGORITHMS_GMRES

