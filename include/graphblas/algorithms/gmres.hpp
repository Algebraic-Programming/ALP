
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

/**
 * @file
 *
 * Provides the GMRES algorithm
 *
 * @author Denis Jelovina
 * @date 6th of December, 2022
 */

#ifndef _H_GRB_ALGORITHMS_GMRES
#define _H_GRB_ALGORITHMS_GMRES

#include <cstdio>
#include <complex>
#include <functional> //function

#include <graphblas.hpp>
#include <graphblas/algorithms/norm.hpp>
#include <graphblas/utils/iscomplex.hpp>

namespace grb {

	namespace algorithms {

		/**
		 * Solves the least linear square problem defined by the vector
		 * \f$ H[1:n] \f$, which is to solve for \f$ x \f$ the following equation:
		 *
		 * \f$ H[1:n] x =  H[ 0 ], \f$
		 *
		 * using Givens rotations and back-substitution.
		 *
		 * The results are stored in H[ 0 ], which is used to update the GMRES
		 * solution vector.
		 *
		 * \todo Replace by ALP/Dense calls once available. At present, this
		 *       implementation performs dense computations using standard STL
		 *       containers.
		 *
		 * \todo The below code does not use any algebraic annotations. It may be
		 *       sensible to hold the required modifications until ALP/Dense is
		 *       available, however.
		 *
		 * \todo There is a sqrt(...) operator that lives outside of the current
		 *       algebraic abstractions. Would be great if that could be eliminated;
		 *       see, e.g., the approach taken by BiCGstab implementation. An
		 *       alternative solution is sketched in internal issue #89.
		 */
		template<
			typename NonzeroType,
			typename DimensionType,
			typename ResidualType,
			class Ring = Semiring<
				grb::operators::add< NonzeroType >, grb::operators::mul< NonzeroType >,
				grb::identities::zero, grb::identities::one
			>,
			class Minus = operators::subtract< NonzeroType >,
			class Divide = operators::divide< NonzeroType >
		>
		grb::RC hessolve(
			std::vector< NonzeroType > &H,
			const DimensionType n,
			const DimensionType &kspspacesize,
			const ResidualType tol,
			std::vector< NonzeroType > &rhs,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide(),
			const std::function< ResidualType( ResidualType ) > &sqrtX = std_sqrt< ResidualType, ResidualType >
		) {
			RC rc = grb::SUCCESS;

			if( n < 1 ) {
				return ILLEGAL;
			}
			if( H.size() <= n ) {
				std::cerr << "Error: algorithms::hessolve requires input parameter H to "
					<< "have a number of entries greater-than the given parameter n. "
					<< "However, " << H.size() << " is smaller-than or equal-to "
					<< n << ".\n";
				return ILLEGAL;
			}
			if( kspspacesize < 1 ) {
				return ILLEGAL;
			}
			if( kspspacesize > n ) {
				return ILLEGAL;
			}
			if( tol <= 0 ) {
				return ILLEGAL;
			}
			if( rhs.size() <= n ) {
				std::cerr << "Error: algorithms::hessolve requires a given workspace rhs "
					<< "that has a number of entries greater-than or equal-to the given "
					<< "parameter n. However, " << rhs.size() << " is strictly smaller-than "
					<< "or equal-to " << n << ".\n";
				return MISMATCH;
			}

			// rhs = H
			for( size_t i = 0; i < n; ++i ) {
				rhs[ i ] = H[ i ];
			}

			size_t n_ksp = std::min( kspspacesize, n - 1 );

			// for i in range(n):
			for( size_t i = 0; i < n_ksp; ++i ) {
				// Givens rotation parameters
				NonzeroType c, s;

				// this scope is is using real ring
				{
					Semiring<
						grb::operators::add< ResidualType >, grb::operators::mul< ResidualType >,
						grb::identities::zero, grb::identities::one
					> ring_rtype;

					// a, b = H[ i:i+2, i ]
					NonzeroType a = H[ ( i + 1 ) * n + i ];
					NonzeroType b = H[ ( i + 1 ) * n + i + 1 ];

					ResidualType a_mod = grb::utils::is_complex< NonzeroType >::modulus( a );
					ResidualType b_mod = grb::utils::is_complex< NonzeroType >::modulus( b );
					NonzeroType b_conj = grb::utils::is_complex< NonzeroType >::conjugate( b );
					ResidualType a_mod2 = a_mod;
					ResidualType b_mod2 = b_mod;
					rc = rc ? rc : grb::foldl( a_mod2, a_mod, ring_rtype.getMultiplicativeOperator() );
					rc = rc ? rc : grb::foldl( b_mod2, b_mod, ring_rtype.getMultiplicativeOperator() );

					// tmp1=sqrt(norm(a)**2+norm(b)**2)
					ResidualType tmp1 = a_mod2;
					rc = rc ? rc : grb::foldl( tmp1, b_mod2, ring_rtype.getAdditiveOperator() );
					tmp1 = sqrtX( tmp1 );

					//c = a_mod / tmp1 ;
					c = a_mod;
					rc = rc ? rc : grb::foldl( c, tmp1, divide );
					if( a_mod != 0 ) {
						// s = a / std::norm(a) * std::conj(b) / tmp1;
						s = a;
						rc = rc ? rc : grb::foldl( s, a_mod, divide );
						rc = rc ? rc : grb::foldl( s, b_conj, ring.getMultiplicativeOperator() );
						rc = rc ? rc : grb::foldl( s, tmp1, divide );
					}
					else {
						// s = std::conj(b) / tmp1;
						s = b_conj;
						rc = rc ? rc : grb::foldl( s, tmp1, divide );
					}
				}

				// for k in range(i,n):
				for( size_t k = i; k < n_ksp; ++k ) {
					// tmp2 = s * H[ i+1, k ]
					NonzeroType tmp2 = H[ ( k + 1 ) * n + i + 1 ];
					rc = rc ? rc : grb::foldl( tmp2, s, ring.getMultiplicativeOperator() );

					// H[i+1,k] = -conjugate(s) * H[i,k] + c * H[i+1,k]
					NonzeroType tmp4 = H[ ( k + 1 ) * n + i ];
					rc = rc ? rc : grb::foldl( H[ ( k + 1 ) * n + i + 1 ], c, ring.getMultiplicativeOperator() );
					rc = rc ? rc : grb::foldl(
						tmp4,
						grb::utils::is_complex< NonzeroType >::conjugate( s ),
						ring.getMultiplicativeOperator()
					);
					rc = rc ? rc : grb::foldl( H[ ( k + 1 ) * n + i + 1 ], tmp4, minus );

					// H[i,k]   = c * H[i,k] + tmp2
					rc = rc ? rc : grb::foldl( H[ ( k + 1 ) * n + i ], c, ring.getMultiplicativeOperator() );
					rc = rc ? rc : grb::foldl( H[ ( k + 1 ) * n + i ], tmp2, ring.getAdditiveOperator() );
				}

				// tmp3 = rhs[i]
				NonzeroType tmp3 = rhs[ i ];
				NonzeroType tmp5 = rhs[ i + 1 ];

				// rhs[i] =  c * tmp3 + s * rhs[i+1]
				rc = rc ? rc : grb::foldl( rhs[ i ], c, ring.getMultiplicativeOperator() );
				rc = rc ? rc : grb::foldl( tmp5, s, ring.getMultiplicativeOperator() );
				rc = rc ? rc : grb::foldl( rhs[ i ], tmp5, ring.getAdditiveOperator() );


				// rhs[i+1]  =  -conjugate(s) * tmp3 + c * rhs[i+1]
				rc = rc ? rc : grb::foldl( rhs[ i + 1 ], c, ring.getMultiplicativeOperator() );
				rc = rc ? rc : grb::foldl(
					tmp3,
					grb::utils::is_complex< NonzeroType >::conjugate( s ),
					ring.getMultiplicativeOperator()
				);
				rc = rc ? rc : grb::foldl( rhs[ i + 1 ], tmp3, minus );
			}

#ifdef _DEBUG
			std::cout << "hessolve rhs vector before inversion, vector = ";
			for( size_t k = 0; k < n_ksp; ++k ) {
				std::cout << rhs[ k ] << " ";
			}
			std::cout << "\n";
#endif

			// for i in range(n-1,-1,-1):
			for( size_t m = 0; m < n_ksp; ++m ) {
				size_t i = n_ksp - 1 - m;
				// for j in range(i+1,n):
				for( size_t j = i + 1; j < n_ksp; ++j ) {
					// rhs[i]=rhs[i]-rhs[j]*H[i,j]
					NonzeroType tmp6 = rhs[ j ];
					rc = rc ? rc : grb::foldl( tmp6, H[ ( j + 1 ) * n + i ], ring.getMultiplicativeOperator() );
					rc = rc ? rc : grb::foldl( rhs[ i ], tmp6, minus );
				}
				// rhs[i]=rhs[i]/H[i,i]
				if( grb::utils::is_complex< NonzeroType >::modulus( H[ ( i + 1 ) * n + i ] ) < tol ) {
					std::cerr << "Warning: small number in algorithms::hessolve\n";
				}
				rc = rc ? rc : grb::foldl( rhs[ i ], H[ ( i + 1 ) * n + i ], divide );
			}

			// H = rhs
			for( size_t i = 0; i < n; ++i ) {
				H[ i ] = rhs[ i ];
			}

			// done
			return rc;
		}

		namespace internal {

			/**
			 * Performs Arnoldi iterations for a GMRES solving of a linear system
			 * \f$ b = Ax \f$ with \f$ x \f$ unknown.
			 *
			 * Preconditioning is possible by providing an initialised matrix \f$ M \f$ of
			 * matching size.
			 *
			 * @tparam descr        The user descriptor
			 * @tparam NonzeroType  The input/output vector/matrix nonzero type
			 * @tparam ResidualType The type of the residual norm
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
				typename NonzeroType,
				typename ResidualType,
				class Ring = Semiring<
					grb::operators::add< NonzeroType >, grb::operators::mul< NonzeroType >,
					grb::identities::zero, grb::identities::one
				>,
				class Minus = operators::subtract< NonzeroType >,
				class Divide = operators::divide< NonzeroType >
			>
			grb::RC gmres_step(
				const grb::Vector< NonzeroType > &x,
				const grb::Matrix< NonzeroType > &A,
				const grb::Vector< NonzeroType > &b,
				std::vector< NonzeroType > &Hmatrix,
				std::vector< grb::Vector< NonzeroType > > &Q,
				const size_t n_restart,
				ResidualType tol,
				size_t &iterations,
				grb::Vector< NonzeroType > &temp,
				const grb::Matrix< NonzeroType > &M = grb::Matrix< NonzeroType >( 0, 0 ),
				const Ring &ring = Ring(),
				const Minus &minus = Minus(),
				const Divide &divide = Divide(),
				const std::function< ResidualType( ResidualType ) > &sqrtX = std_sqrt< ResidualType, ResidualType >
			) {
				// static checks
				static_assert( std::is_floating_point< ResidualType >::value,
					"Can only use the GMRES algorithm with floating-point residual "
					"types." );

				bool useprecond = false;
				if( (nrows( M ) != 0) && (ncols( M ) != 0) ) {
					useprecond = true;
				}

				constexpr const Descriptor descr_dense = descr | descriptors::dense;
				const ResidualType zero = ring.template getZero< ResidualType >();

				// dynamic checks, main error checking done in GMRES main function
#ifndef NDEBUG
				const size_t n = grb::ncols( A );
				{
					const size_t m = grb::nrows( A );
					assert( m == n );
					assert( size( x ) == n );
					assert( size( b ) == m );
					assert( size( Q[ 0 ] ) == n );
					assert( size( temp ) == n );
					assert( capacity( x ) == n );
					assert( capacity( Q[ 0 ] ) == n );
					assert( capacity( temp ) == n );
					assert( tol > zero );
				}
#endif

				ResidualType rho, tau;

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

				//rho = norm(Q[:,0])
				rho = zero;
				ret = ret ? ret : grb::algorithms::norm2< descr_dense >( rho, Q[ 0 ], ring, sqrtX );
				assert( ret == SUCCESS );

				Hmatrix[ 0 ] = rho;

				tau = tol * rho;

				size_t k = 0;
				while( ( rho > tau ) && ( k < n_restart ) ) {
					// alpha = r' * r;

					if( grb::utils::is_complex< NonzeroType >::modulus( Hmatrix[ k * ( n_restart + 1 ) + k ] ) < tol ) {
						break;
					}

					// Q[k] = Q[k] / alpha
					ret = ret ? ret : grb::foldl( Q[ k ], Hmatrix[ k * ( n_restart + 1 ) + k ], divide );
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

					(void) ++k;

					for( size_t j = 0; j < std::min( k, n_restart ); j++ ) {
						//H[j,k]=Q[:,j].dot(Q[:,k])
						Hmatrix[ k * ( n_restart + 1 ) + j ] = zero;
						ret = ret ? ret : grb::dot< descr_dense >(
							Hmatrix[ k * ( n_restart + 1 ) + j ],
							Q[ k ], Q[ j ],
							ring.getAdditiveMonoid(),
							conjuagte_mul< NonzeroType, NonzeroType, NonzeroType >()
						);
						assert( ret == SUCCESS );

						//Q[:,k]=Q[:,k]-H[j,k]*Q[:,i]
						grb::RC ret = grb::set( temp, zero );
						assert( ret == SUCCESS );

						NonzeroType alpha1 = Hmatrix[ k * ( n_restart + 1 ) + j ];
						ret = ret ? ret : grb::eWiseMul< descr_dense >( temp, alpha1, Q[ j ],
							ring );
						assert( ret == SUCCESS );

						ret = ret ? ret : grb::foldl< descr_dense >( Q[ k ], temp, minus );
						assert( ret == SUCCESS );
					} // while

					//alpha=norm(Q[:,k])
					ResidualType alpha = zero;
					ret = ret ? ret : grb::algorithms::norm2< descr_dense >( alpha, Q[ k ], ring, sqrtX );
					assert( ret == SUCCESS );

					//H[k,k]=alpha
					Hmatrix[ k * ( n_restart + 1 ) + k ] = alpha;
				}

				iterations += k;

				if( ret != SUCCESS ) {
					return FAILED;
				} else {
					return SUCCESS;
				}
			}

			template<
				Descriptor descr = descriptors::no_operation,
				bool no_preconditioning = true,
				typename NonzeroType,
				typename ResidualType,
				class Ring = Semiring<
					grb::operators::add< NonzeroType >, grb::operators::mul< NonzeroType >,
					grb::identities::zero, grb::identities::one
				>,
				class Minus = operators::subtract< NonzeroType >,
				class Divide = operators::divide< NonzeroType >
			>
			grb::RC gmres_dispatch(
				grb::Vector< NonzeroType > &x,
				const grb::Matrix< NonzeroType > &A,
				const grb::Vector< NonzeroType > &b,
				const size_t n_restart,
				const size_t max_iterations,
				const ResidualType max_residual_norm,
				const ResidualType tol,
				size_t &iterations,
				size_t &iterations_gmres,
				size_t &iterations_arnoldi,
				ResidualType &residual,
				ResidualType &residual_relative,
				std::vector< grb::Vector< NonzeroType > > &Q,
				std::vector< NonzeroType > &Hmatrix,
				grb::Vector< NonzeroType > &temp,
				std::vector< NonzeroType > &temp3,
				const grb::Matrix< NonzeroType > &M,
				const Ring &ring = Ring(),
				const Minus &minus = Minus(),
				const Divide &divide = Divide(),
				const std::function< ResidualType( ResidualType ) > &sqrtX = std_sqrt< ResidualType, ResidualType >
			) {
				grb::RC rc = grb::SUCCESS;
				constexpr const Descriptor descr_dense = descr | descriptors::dense;
				const ResidualType zero = ring.template getZero< ResidualType >();

				// dynamic checks
				{
					// mismatches
					const size_t n = grb::ncols( A );
					const size_t m = grb::nrows( A );
					if( grb::size( x ) != n ) {
						return MISMATCH;
					}
					if( grb::size( b ) != m ) {
						return MISMATCH;
					}
					if( !no_preconditioning ) {
						if( grb::ncols( M ) != n || grb::nrows( M ) != m ) {
							return MISMATCH;
						}
					}

					// illegal inputs
					if( capacity( x ) != n ) {
						return ILLEGAL;
					}
					if( m != n ) {
						std::cerr << "Warning: grb::algorithms::conjugate_gradient requires "
							<< "square input matrices, but a non-square input matrix was "
							<< "given instead.\n";
						return ILLEGAL;
					}
					if( n_restart == 0 && max_iterations > 0 ) {
						return ILLEGAL;
					}
					if( max_residual_norm <= zero ) {
						std::cerr << "Error: max residual norm to GMRES must be strictly "
							<< "positive\n";
						return ILLEGAL;
					}
					if( tol <= zero ) {
						std::cerr << "Error: tolerance input to GMRES must be strictly positive\n";
						return ILLEGAL;
					}

					// workspace
					if( Q.size() <= n_restart ) {
						std::cerr << "Error: expected n_restart + 1 (" << (n_restart+1) << ") "
							<< "columns in the given Q, but only " << Q.size() << " were given.\n";
						// FIXME this should become a MISMATCH once ALP/Dense is up
						return ILLEGAL;
					}
					for( size_t i = 0; i <= n_restart; ++i ) {
						if( grb::size( Q[ i ] ) != n || grb::capacity( Q[ i ] ) != n ) {
							std::cerr << "Error: provided workspace vectors in Q are not of the "
								<< "correct length and/or capacity.\n";
							return ILLEGAL;
						}
					}
					if( Hmatrix.size() <= n_restart ) {
						std::cerr << "Error: expected (n_restart + 1)^2 entries in H ("
							<< ((n_restart+1)*(n_restart+1)) << "), but only " << Hmatrix.size()
							<< " were given.\n";
						// FIXME H should become a structured matrix and this code should return
						//       MISMATCH if dimension check fails, once ALP/Dense is up
						return ILLEGAL;
					}
					if( grb::size( temp ) < n || grb::capacity( temp ) < n ) {
						std::cerr << "Error: provided temp workspace vector is not of the correct "
							<< "length and/or capacity.\n";
						return ILLEGAL;
					}
					if( temp3.size() < n ) {
						std::cerr << "Error: provided temp3 workspace vector (STL) is not of the "
							<< "correct length.\n";
						return ILLEGAL;
					}
				}

				// no side effects: set initial values to outputs only after error checking
				iterations = iterations_gmres = iterations_arnoldi = 0;
				residual = residual_relative = 0;

				// get RHS vector norm
				ResidualType bnorm = zero;
				rc = rc ? rc : grb::algorithms::norm2< descr_dense >( bnorm, b, ring, sqrtX );

#ifdef DEBUG
				{
					std::cout << "RHS norm = " << bnorm << " \n";
					PinnedVector< NonzeroType > pinnedVector( b, SEQUENTIAL );
					std::cout << "RHS vector = ";
					for( size_t k = 0; k < 10; ++k ) {
						const NonzeroType &nonzeroValue = pinnedVector.getNonzeroValue( k );
						std::cout << nonzeroValue << " ";
					}
					std::cout << " ...  ";
					for( size_t k = n - 10; k < n; ++k ) {
						const NonzeroType &nonzeroValue = pinnedVector.getNonzeroValue( k );
						std::cout << nonzeroValue << " ";
					}
					std::cout << "\n";
				}
#endif
				// guard against a trivial call
				if( max_iterations == 0 ) {
					rc = rc ? rc : grb::algorithms::norm2< descr_dense >( residual, b, ring, sqrtX );
					if( rc != grb::SUCCESS ) {
						std::cerr << "Error: residual norm not calculated properly.\n";
						return rc;
					}
					residual_relative = residual / bnorm;
					if( residual_relative < max_residual_norm ) {
						return rc;
					} else {
						return FAILED;
					}
				}

				// perform gmres iterations
				for( size_t gmres_iter = 0; gmres_iter < max_iterations; ++gmres_iter ) {
					(void) ++iterations;
					(void) ++iterations_gmres;
					size_t kspspacesize = 0;
					if( no_preconditioning ) {
#ifdef DEBUG
						std::cout << "Call gmres without preconditioner.\n";
#endif
						rc = rc ? rc : gmres_step(
							x, A, b,
							Hmatrix, Q,
							n_restart, tol,
							kspspacesize,
							temp, grb::Matrix< NonzeroType >( 0, 0 ),
							ring, minus, divide, sqrtX
						);
					} else {
#ifdef DEBUG
						std::cout << "Call gmres with preconditioner.\n";
#endif
						rc = rc ? rc : gmres_step(
							x, A, b,
							Hmatrix, Q,
							n_restart, tol,
							kspspacesize,
							temp, M,
							ring, minus, divide, sqrtX
						);
					}
#ifdef DEBUG
					if( rc == grb::SUCCESS ) {
						std::cout << "gmres iteration finished successfully, kspspacesize = "
							<< kspspacesize << "  \n";
					}
#endif
					if( rc != grb::SUCCESS ) {
						std::cerr << "Error: gmres_step failed.\n";
						return rc;
					}

					iterations_arnoldi += kspspacesize;

					rc = rc ? rc : hessolve( Hmatrix, n_restart + 1, kspspacesize, tol, temp3 );
					if( rc != grb::SUCCESS ) {
						std::cerr << "Error: hessolve failed.\n";
						return rc;
					}

					// update x
					for( size_t i = 0; rc == grb::SUCCESS && i < kspspacesize; ++i ) {
						rc = rc ? rc : grb::eWiseMul( x, Hmatrix[ i ], Q[ i ], ring );
#ifdef DEBUG
						if( rc != grb::SUCCESS ) {
							std::cout << "grb::eWiseMul( x, Hmatrix[ " << i << " ], Q [ " << i
								<< " ], ring ); failed\n";
						}
#endif
					}

#ifdef DEBUG
					if( rc == grb::SUCCESS ) {
						std::cout << "vector x updated successfully\n";
						PinnedVector< NonzeroType > pinnedVector( x, SEQUENTIAL );
						std::cout << "x vector = ";
						for( size_t k = 0; k < 10; ++k ) {
							const NonzeroType &nonzeroValue = pinnedVector.getNonzeroValue( k );
							std::cout << nonzeroValue << " ";
						}
						std::cout << " ...  ";
						for( size_t k = n-10; k < n; ++k ) {
							const NonzeroType &nonzeroValue = pinnedVector.getNonzeroValue( k );
							std::cout << nonzeroValue << " ";
						}
						std::cout << "\n";
					}
#endif
					if( rc != grb::SUCCESS ) {
						std::cerr << "Error: update solution vector failed.\n";
						return rc;
					}

					// calculate residual
					rc = grb::set( temp, zero );
					rc = rc ? rc : grb::mxv( temp, A, x, ring );
					rc = rc ? rc : grb::foldl( temp, b, minus );
					rc = rc ? rc : grb::algorithms::norm2< descr_dense >( residual, temp, ring, sqrtX );
					if( rc != grb::SUCCESS ) {
						std::cerr << "Error: residual norm not calculated properly.\n";
						return rc;
					}
					residual_relative = residual / bnorm;

#ifdef DEBUG
					std::cout << "Residual norm = " << residual << " \n";
					std::cout << "Residual norm (relative) = " << residual_relative << " \n";
#endif

					if( residual_relative < max_residual_norm ) {
#ifdef DEBUG
						std::cout << "Convergence reached\n";
#endif
						break;
					}
				} // gmres iterations

				if( rc == SUCCESS && residual_relative > max_residual_norm ) {
					return FAILED;
				} else {
					return rc;
				}
			}

		} // end namespace grb::algorithms::internal

		/**
		 * Solves a linear system \f$ b = Ax \f$ with \f$ x \f$ unknown using the
		 * Generalised Minimal Residual (GMRES) method on general fields.
		 *
		 * Explicit preconditioning is possible by providing an initialised matrix
		 * \f$ M \f$ of matching size.
		 *
		 * @tparam descr        The user descriptor
		 * @tparam NonzeroType  The input/output vector/matrix nonzero type
		 * @tparam ResidualType The type of the residual norm
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
		 * @param[in]	  no_preconditioning   Diables preconditioner.
		 * @param[in]	  max_residual_norm    Iterations stop after
		 *                                     max_residual_norm iterations.
		 * @param[out]    iterations           Total number of interactions.
		 * @param[out]    iterations_gmres     Number of GMRES interactions performed.
		 * @param[out]    iterations_arnoldi   Number of Arnoldi interactions performed.
		 * @param[out]    residual             Residual norm.
		 * @param[out]    residual_relative    Relative residual norm.
		 * @param[in,out] temp           A temporary vector of the same size as \a x.
		 * @param[out]    Q              std::vector of length \a n_restart + 1.
		 *                               Each element of Q is grb::Vector of size
		 *                               \f$ n \f$. On output only fist \a n_restart
		 *                               vectors have meaning.
		 *
		 * Additional outputs (besides \a x):
		 *
		 * The GMRES algorithm requires three workspace buffers with capacity \f$ n \f$:
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
			typename NonzeroType,
			typename ResidualType,
			class Ring = Semiring<
				grb::operators::add< NonzeroType >, grb::operators::mul< NonzeroType >,
				grb::identities::zero, grb::identities::one
			>,
			class Minus = operators::subtract< NonzeroType >,
			class Divide = operators::divide< NonzeroType >
		>
		grb::RC gmres(
			grb::Vector< NonzeroType > &x,
			const grb::Matrix< NonzeroType > &A,
			const grb::Vector< NonzeroType > &b,
			const size_t n_restart,
			const size_t max_iterations,
			const ResidualType max_residual_norm,
			const ResidualType tol,
			size_t &iterations,
			size_t &iterations_gmres,
			size_t &iterations_arnoldi,
			ResidualType &residual,
			ResidualType &residual_relative,
			std::vector< grb::Vector< NonzeroType > > &Q,
			std::vector< NonzeroType > &Hmatrix,
			grb::Vector< NonzeroType > &temp,
			std::vector< NonzeroType > &temp3,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide(),
			const std::function< ResidualType( ResidualType ) > &sqrtX = std_sqrt< ResidualType, ResidualType >
		) {
			grb::Matrix< NonzeroType > dummy( 0, 0 );
			return internal::gmres_dispatch< descr, true >(
					x, A, b,
					n_restart, max_iterations,
					max_residual_norm, tol,
					iterations, iterations_gmres, iterations_arnoldi,
					residual, residual_relative,
					Q, Hmatrix, temp, temp3,
					dummy,
					ring, minus, divide, sqrtX
				);
		}

		/** TODO documentation */
		template<
			Descriptor descr = descriptors::no_operation,
			typename NonzeroType,
			typename ResidualType,
			class Ring = Semiring<
				grb::operators::add< NonzeroType >, grb::operators::mul< NonzeroType >,
				grb::identities::zero, grb::identities::one
			>,
			class Minus = operators::subtract< NonzeroType >,
			class Divide = operators::divide< NonzeroType >
		>
		grb::RC preconditioned_gmres(
			grb::Vector< NonzeroType > &x,
			const grb::Matrix< NonzeroType > &M,
			const grb::Matrix< NonzeroType > &A,
			const grb::Vector< NonzeroType > &b,
			const size_t n_restart,
			const size_t max_iterations,
			const ResidualType max_residual_norm,
			const ResidualType tol,
			size_t &iterations,
			size_t &iterations_gmres,
			size_t &iterations_arnoldi,
			ResidualType &residual,
			ResidualType &residual_relative,
			std::vector< grb::Vector< NonzeroType > > &Q,
			std::vector< NonzeroType > &Hmatrix,
			grb::Vector< NonzeroType > &temp,
			std::vector< NonzeroType > &temp3,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide(),
			const std::function< ResidualType( ResidualType ) > &sqrtX = std_sqrt< ResidualType, ResidualType >
		) {
			return internal::gmres_dispatch< descr, false >(
					x, A, b,
					n_restart, max_iterations,
					max_residual_norm, tol,
					iterations, iterations_gmres, iterations_arnoldi,
					residual, residual_relative,
					Q, Hmatrix, temp, temp3,
					M,
					ring, minus, divide, sqrtX
				);
		}


	} // namespace algorithms

} // end namespace grb

#endif // end _H_GRB_ALGORITHMS_GMRES

