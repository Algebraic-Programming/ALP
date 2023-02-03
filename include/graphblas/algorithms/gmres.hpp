
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
		 * Solves the  least linear square problem defined by vector H[1:n] x =  H[ 0 ],
		 * using Givens rotations and backsubstitution. The results is stored in H[ 0 ],
		 * which is sused to update GMRES solution, vector x.
		 * \todo: Replace by ALP/Dense calls once available.
		 */
		template<
			typename NonzeroType,
			typename DimensionType,
			typename ResidualType
			>
		void hessolve(
			std::vector< NonzeroType > &H,
			const DimensionType n,
			const DimensionType &kspspacesize,
			const ResidualType tol
		) {
			std::vector< NonzeroType > rhs( H.begin(),  H.begin() + n );

			size_t n_ksp = std::min( kspspacesize, n - 1 );

			// for i in range(n):
			for( size_t i = 0; i < n_ksp; ++i ) {
				NonzeroType a, b, c, s;

				// a,b=H[i:i+2,i]
				a = H[ ( i + 1 ) * n + i ];
				b = H[ ( i + 1 ) * n + i + 1 ];
				// tmp1=sqrt(norm(a)**2+norm(b)**2)
				NonzeroType tmp1 = std::sqrt(
					std::norm( a ) +
					std::norm( b )
				);
				c = grb::utils::is_complex< NonzeroType >::modulus( a ) / tmp1 ;
				if( std::norm( a ) != 0 ) {
					// s = a / std::norm(a) * std::conj(b) / tmp1;
					s = a / grb::utils::is_complex< NonzeroType >::modulus( a )
						* grb::utils::is_complex< NonzeroType >::conjugate( b ) / tmp1;
				}
				else {
					// s = std::conj(b) / tmp1;
					s = grb::utils::is_complex< NonzeroType >::conjugate( b ) / tmp1;
				}

				NonzeroType tmp2;
				// for k in range(i,n):
				for( size_t k = i; k < n_ksp; ++k ) {
					// tmp2       =   s * H[i+1,k]
					tmp2 = s * H[ ( k + 1 ) * n + i + 1 ];
					// H[i+1,k] = -conjugate(s) * H[i,k] + c * H[i+1,k]
					H[ ( k + 1 ) * n + i + 1 ] = - grb::utils::is_complex< NonzeroType >::conjugate( s )
						* H[ ( k + 1 ) * n + i ] + c * H[ ( k + 1 ) * n + i + 1 ];
					// H[i,k]   = c * H[i,k] + tmp2
					H[ ( k + 1 ) * n + i ] = c * H[ ( k + 1 ) * n + i ] + tmp2;
				}

				// tmp3 = rhs[i]
				NonzeroType tmp3;
				tmp3 = rhs[ i ];
				// rhs[i] =  c * tmp3 + s * rhs[i+1]
				rhs[ i ]  =  c * tmp3 + s * rhs[ i + 1 ] ;
				// rhs[i+1]  =  -conjugate(s) * tmp3 + c * rhs[i+1]
				rhs[ i + 1 ]  =  - grb::utils::is_complex< NonzeroType >::conjugate( s )
					* tmp3 + c * rhs[ i + 1 ];
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
					rhs[ i ] = rhs[ i ] - rhs[ j ] * H[ ( j + 1 ) * n + i ];
				}
				// rhs[i]=rhs[i]/H[i,i]
				if( std::abs( H[ ( i + 1 ) * n + i ] ) < tol ) {
					std::cout << "---> small number in hessolve\n";
				}
				rhs[ i ] = rhs[ i ] / H[ ( i + 1 ) * n + i ];
			}

			std::copy( rhs.begin(), rhs.end(), H.begin() );
		}


		/**
		 * Performes Arnoldi iterations in GMRES solver.
		 * for a linear system \f$ b = Ax \f$ with \f$ x \f$ unknown
		 *
		 * Preconditioning is possible by providing
		 * an initialised matrix \f$ M \f$ of matching size.
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
			const Divide &divide = Divide()
		) {
			// static checks
			static_assert( std::is_floating_point< ResidualType >::value,
				"Can only use the GMRES algorithm with floating-point residual "
				"types." );


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
			if( grb::utils::is_complex< NonzeroType >::value ) {
				ret = ret ? ret : grb::eWiseLambda( [&,Q]( const size_t i ) {
					temp[ i ] = grb::utils::is_complex< NonzeroType >::conjugate( Q[ 0 ] [ i ] );
					}, temp
				);
				ret = ret ? ret : grb::dot< descr_dense >( alpha, temp, Q[ 0 ], ring );
			} else {
				ret = ret ? ret : grb::dot< descr_dense >( alpha, Q[ 0 ], Q[ 0 ], ring );
			}
			assert( ret == SUCCESS );

			rho = sqrt( grb::utils::is_complex< NonzeroType >::modulus( alpha ) );
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
					if( grb::utils::is_complex< NonzeroType >::value ) {
						ret = ret ? ret : grb::eWiseLambda( [&,Q]( const size_t i ) {
							temp[ i ] = grb::utils::is_complex< NonzeroType >::conjugate( Q[ j ] [ i ] );
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
				if( grb::utils::is_complex< NonzeroType >::value ) {
					grb::RC ret = grb::set( temp, zero );
					assert( ret == SUCCESS );

					ret = ret ? ret : grb::eWiseLambda( [&,Q,k]( const size_t i ) {
						temp[ i ] = grb::utils::is_complex< NonzeroType >::conjugate( Q[ k ] [ i ] );
					}, temp
						);
					ret = ret ? ret : grb::dot< descr_dense >( alpha, temp, Q[ k ], ring );
				} else {
					ret = ret ? ret : grb::dot< descr_dense >( alpha, Q[ k ], Q[ k ], ring );
				}
				assert( ret == SUCCESS );

				//H[k,k]=rho
				Hmatrix[ k * ( n_restart + 1 ) + k ] = sqrt(
					grb::utils::is_complex< NonzeroType >::modulus( alpha )
				) ;

			}

			iterations = iterations + k;

			if( ret != SUCCESS ) {
				return FAILED;
			} else {
				return SUCCESS;
			}
		}

		/**
		 * Solves a linear system \f$ b = Ax \f$ with \f$ x \f$ unknown
		 * by the GMRES method on general fields.
		 *
		 * Preconditioning is possible by providing
		 * an initialised matrix \f$ M \f$ of matching size.
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
			std::vector< grb::Vector< NonzeroType > > &Q,
			const size_t n_restart,
			const size_t max_iterations,
			const bool no_preconditioning,
			const ResidualType max_residual_norm,
			const ResidualType tol,
			size_t &iterations,
			size_t &iterations_gmres,
			size_t &iterations_arnoldi,
			ResidualType &residual,
			ResidualType &residual_relative,
			grb::Vector< NonzeroType > &temp,
			grb::Vector< NonzeroType > &temp2,
			std::vector< NonzeroType > &Hmatrix,
			const grb::Matrix< NonzeroType > &M = grb::Matrix< NonzeroType >( 0, 0 ),
			const Ring &ring = Ring(),
			const Minus &minus = Minus()
		) {

			grb::RC rc = grb::SUCCESS;
			const ResidualType zero = ring.template getZero< ResidualType >();

			// get RHS vector norm
			NonzeroType bnorm = ring.template getZero< NonzeroType >();
			rc = grb::set( temp, b );
			if( grb::utils::is_complex< NonzeroType >::value ) {
				rc = grb::set( temp2, zero );
				rc = rc ? rc : grb::eWiseLambda(
					[ &, temp ] ( const size_t i ) {
						temp2[ i ] = grb::utils::is_complex< NonzeroType >::conjugate( temp [ i ] );
					}, temp
				);
				rc = rc ? rc : grb::dot( bnorm, temp, temp2, ring );
			} else {
				rc = rc ? rc : grb::dot( bnorm, temp, temp, ring );
			}
			bnorm =std::sqrt( bnorm );

#ifdef DEBUG
			std::cout << "RHS norm = " << std::abs( bnorm ) << " \n";

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
#endif

			// gmres iterations
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
						temp
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
						temp,
						M
					);
				}
#ifdef DEBUG
				if( rc == grb::SUCCESS ) {
					std::cout << "gmres iteration finished successfully, kspspacesize = " << kspspacesize << "  \n";
				}
#endif
				iterations_arnoldi += kspspacesize;

				hessolve( Hmatrix, n_restart + 1, kspspacesize, tol );
				// update x
				for( size_t i = 0; i < kspspacesize; ++i ) {
					rc = rc ? rc : grb::eWiseMul( x, Hmatrix[ i ], Q[ i ], ring );
#ifdef DEBUG
					if( rc != grb::SUCCESS ) {
						std::cout << "grb::eWiseMul( x, Hmatrix[ " << i << " ], Q [ " << i << " ], ring ); failed\n";
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

				// calculate residual
				rc = rc ? rc : grb::set( temp, zero );
				rc = rc ? rc : grb::mxv( temp, A, x, ring );
				rc = rc ? rc : grb::foldl( temp, b, minus );
				NonzeroType residualnorm = zero;
				if( grb::utils::is_complex< NonzeroType >::value ) {
					rc = grb::set( temp2, zero );
					rc = rc ? rc : grb::eWiseLambda(
						[ &, temp ] ( const size_t i ) {
							temp2[ i ] = grb::utils::is_complex< NonzeroType >::conjugate( temp [ i ] );
						}, temp
					);
					rc = rc ? rc : grb::dot( residualnorm, temp, temp2, ring );
				} else {
					rc = rc ? rc : grb::dot( residualnorm, temp, temp, ring );
				}
				if( rc != grb::SUCCESS ) {
					std::cout << "Residual norm not calculated properly.\n";
				}
				residualnorm = std::sqrt( residualnorm );

				residual = std::abs( residualnorm );
				residual_relative = residual / std::abs( bnorm );

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

			return rc;
		}

	} // namespace algorithms

} // end namespace grb

#endif // end _H_GRB_ALGORITHMS_GMRES

