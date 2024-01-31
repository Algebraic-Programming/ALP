
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
 * Implements the canonical PageRank algorithm by Brin and Page.
 *
 * @author A. N. Yzelman
 * @date: 21st of March, 2017
 */

#ifndef _H_GRB_PAGERANK
#define _H_GRB_PAGERANK

#include <graphblas.hpp>

#ifndef _GRB_NO_STDIO
#include <iostream>
#endif


namespace grb {

	namespace algorithms {

		/**
		 * The canonical PageRank algorithm.
		 *
		 * @tparam descr    The descriptor under which to perform the computation.
		 * @tparam IOType   The value type of the pagerank vector. This will
		 *                  determine the precision of all computations this
		 *                  algorithm performs.
		 * @tparam NonzeroT The type of the elements of the nonzero matrix.
		 *
		 * @param[in,out] pr  Vector of size and capacity \f$ n \f$, where \f$ n \f$
		 *                    is the vertex size of the input graph \a L.
		 *                    On input, the contents of this vector will be taken as
		 *                    the initial guess to the final result, but only if the
		 *                    vector is dense; if it is, all entries of the initial
		 *                    guess must be nonzero, while it is not, this algorithm
		 *                    will make an initial guess.
		 *                    On output, if #grb::SUCCESS is returned, the PageRank
		 *                    vector corresponding to \a L.
		 * @param[in]    L    The input graph as a square link matrix of size
		 *                    \f$ n \f$.
		 *
		 * To operate, this algorithm requires a workspace of three vectors. The size
		 * \em and capacities of these must equal \f$ n \f$. The contents on input are
		 * ignored, and the contents on output are undefined.
		 *
		 * This algorithm does not explicitly materialise the Google matrix
		 * \f$ G = \alpha L + (1-\alpha)ee^T \f$ over which the power iterations are
		 * exectuted.
		 *
		 * @param[in,out] pr_next     Buffer for the PageRank algorithm.
		 * @param[in,out] pr_nextnext Buffer for the PageRank algorithm.
		 * @param[in,out] row_sum     Buffer for the PageRank algorithm.
		 *
		 * The PageRank algorithm holds the following \em optional parameters:
		 *
		 * @param[in]  alpha  The scaling factor. The default value is 0.85.
		 *                    This value must be smaller than 1, and larger than 0.
		 *
		 * @param[in]  conv   If the difference between two successive iterations, in
		 *                    terms of its one-norm, is less than this value, then the
		 *                    PageRank vector is considered converged and this
		 *                    algorithm exits successfully. The default value is
		 *                    \f$ 10^{-8} \f$. If this value is set to zero, then the
		 *                    algorithm will continue until \a max iterations are
		 *                    reached. May not be negative.
		 * @param[in]  max    The maximum number of power iterations. The default
		 *                    value is 1000. This value must be larger than 0.
		 *
		 * The PageRank algorithm reports the following \em optional output:
		 *
		 * @param[out] iterations If not <tt>nullptr</tt>, the number of iterations
		 *                        the call to this algorithm took will be written to
		 *                        the location pointed to.
		 * @param[out] quality    If not <tt>nullptr</tt>, the last computed residual
		 *                        will be written to the location pointed to.
		 *
		 * @returns #grb::SUCCESS  If the computation converged within \a max
		 *                         iterations.
		 * @returns #grb::ILLEGAL  If \a L is not square. All outputs are left
		 *                         untouched.
		 * @returns #grb::MISMATCH If the dimensions of \a pr and \a L do not match.
		 *                         All outputs are left untouched.
		 * @returns #grb::ILLEGAL  If an invalid value for \a alpha, \a conv, or
		 *                         \a max was given. All outputs are left untouched.
		 * @returns #grb::ILLEGAL  If the capacity of one or more of \a pr,
		 *                         \a pr_next, \a pr_nextnext, or \a row_sum is less
		 *                         than \f$ n \f$. All outputs are left untouched.
		 * @returns #grb::FAILED   If the PageRank method did not converge to within
		 *                         the given tolerance \a conv within the given
		 *                         maximum number of iterations \a max. The output
		 *                         \a pr is the last computed approximation, while
		 *                         \a iterations and \a quality are likewise set to
		 *                         \a max and the last computed residual,
		 *                         respectively.
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
			typename IOType, typename NonzeroT
		>
		RC simple_pagerank(
			Vector< IOType > &pr,
			const Matrix< NonzeroT > &L,
			Vector< IOType > &pr_next,
			Vector< IOType > &pr_nextnext,
			Vector< IOType > &row_sum,
			const IOType alpha = 0.85,
			const IOType conv = 0.0000001,
			const size_t max = 1000,
			size_t * const iterations = nullptr,
			double * const quality = nullptr
		) {
			grb::Monoid<
				grb::operators::add< IOType >,
				grb::identities::zero
			> addM;
			grb::Semiring<
				grb::operators::add< IOType >, grb::operators::mul< IOType >,
				grb::identities::zero, grb::identities::one
			> realRing;
#ifdef _DEBUG
			const auto s = spmd<>::pid();
#endif
			const size_t n = nrows( L );
			const IOType zero = realRing.template getZero< IOType >();

			// runtime sanity checks
			{
				if( n != ncols( L ) ) {
					return ILLEGAL;
				}
				if( size( pr ) != n ) {
					return MISMATCH;
				}
				if( size( pr ) != n ) {
					return MISMATCH;
				}
				if( size( pr_next ) != n ||
					size( pr_nextnext ) != n ||
					size( row_sum ) != n
				) {
					return MISMATCH;
				}
				if( capacity( pr ) != n ) {
					return ILLEGAL;
				}
				if( capacity( pr_next ) != n ||
					capacity( pr_nextnext ) != n ||
					capacity( row_sum ) != n
				) {
					return ILLEGAL;
				}
				// alpha must be within 0 and 1 (both exclusive)
				if( alpha <= 0 || alpha >= 1 ) {
					return ILLEGAL;
				}
				// max must be larger than 0
				if( max <= 0 ) {
					return ILLEGAL;
				}
			}

			// running error code
			RC ret = SUCCESS;

			// make initial guess if the user did not make one
			if( nnz( pr ) != n ) {
				ret = set( pr, static_cast< IOType >( 1 ) / static_cast< IOType >( n ) );
				assert( ret == SUCCESS );
			}

			// initialise all temporary vectors to default dense values
			ret = ret ? ret : set( pr_nextnext, zero );
			assert( ret == SUCCESS );

			// calculate row sums
			Semiring<
				operators::add< IOType >,
				operators::left_assign_if< IOType, bool, IOType >,
				identities::zero,
				identities::logical_true
			> pattern_ring;

			ret = ret ? ret : set( pr_next, 1 ); // abuses pr_next as temporary vector
			ret = ret ? ret : set( row_sum, 0 );
			ret = ret ? ret :
				vxm< descr | descriptors::dense | descriptors::transpose_matrix >(
					row_sum, pr_next, L, pattern_ring
				);
			// pr_next is now free for further use
			assert( ret == SUCCESS );

#ifdef _DEBUG
			std::cout << "Prelude to iteration 0:\n";
			(void) eWiseLambda< descriptors::dense >(
				[ &row_sum, &pr_next, &pr ]( const size_t i ) {
					#pragma omp critical
					{
						std::cout << i << ": " << row_sum[ i ] << "\t" << pr[ i ] << "\t"
							<< pr_next[ i ] << "\n";
					}
				},
				pr, pr_next, row_sum );
#endif

			// calculate in-place the inverse of row sums, and keep zero if dangling
			// this eWiseLambda is always supported since alpha is read-only
			ret = ret ? ret : eWiseLambda< descriptors::dense >(
					[ &row_sum, &alpha, &zero ]( const size_t i ) {
						assert( row_sum[ i ] >= zero );
						if( row_sum[ i ] > zero ) {
							row_sum[ i ] = alpha / row_sum[ i ];
						}
					},
					row_sum
				);
				assert( ret == SUCCESS );

#ifdef _DEBUG
			std::cout << "Row sum array:\n";
			(void) eWiseLambda< descriptors::dense >(
				[ &row_sum ]( const size_t i ) {
					#pragma omp critical
					std::cout << i << ": " << row_sum[ i ] << "\n";
				}, row_sum
			);
#endif

			// some control variables
			size_t iter = 0; //#iterations, initalise to zero
			IOType dangling = zero; // used for caching the contribution of random jumps
			                        // from dangling nodes
			IOType residual = zero; // declare residual (computed inside the do-while
			                        // loop)

			// main loop
			do {
				// reset iteration-local values
				residual = dangling = 0;

#ifdef _DEBUG
				std::cout << "Current PR array:\n";
				(void) eWiseLambda< descriptors::dense >(
						[ &pr ]( const size_t i ) {
							#pragma omp critical
							if( i < 8 ) {
								std::cout << i << ": " << pr[ i ] << "\n";
							}
						}, pr
					);
#endif

				// calculate dangling factor and do scaling
				if( ret == SUCCESS ) {
					// can we reduce via lambdas?
					if( Properties<>::writableCaptured ) {
						// yes we can, so save one unnecessary stream on pr
						ret = eWiseLambda< descriptors::dense >(
								[ &pr_next, &row_sum, &dangling, &pr ]( const size_t i ) {
									// calculate dangling contribution
									if( row_sum[ i ] == 0 ) {
										dangling += pr[ i ];
										pr_next[ i ] = 0;
									} else {
										// pre-scale input
										pr_next[ i ] = pr[ i ] * row_sum[ i ];
									}
								}, row_sum, pr, pr_next
							);
						// allreduce dangling factor
						assert( ret == SUCCESS );
						ret = ret ? ret : grb::collectives<>::allreduce(
							dangling,
							grb::operators::add< double >()
						);
						assert( ret == SUCCESS );
					} else {
						// otherwise we have to handle the reduction separately
						ret = foldl< descriptors::dense | grb::descriptors::invert_mask >(
							dangling, pr, row_sum, addM
						);
						assert( ret == SUCCESS );

						// separately from the element-wise multiplication here
						ret = ret ? ret : eWiseApply< descriptors::dense >(
								pr_next, pr, row_sum,
								grb::operators::mul< double >()
							);
						assert( ret == SUCCESS );
					}
				}

#if defined _DEBUG && defined _GRB_WITH_LPF
				for( size_t dbg = 0; dbg < spmd<>::nprocs(); ++dbg ) {
					const auto s = spmd<>::pid();
					if( dbg == s ) {
						std::cout << "Next PR array (under construction):\n";
						eWiseLambda< descriptors::dense >(
							[ &pr_next, s ]( const size_t i ) {
								#pragma omp critical
								if( i < 10 ) {
									std::cout << i << ", " << s << ": " << pr_next[ i ] << "\n";
								}
							},
							pr_next );
					}
					spmd<>::sync();
				}
#endif

				if( ret == SUCCESS ) {
#ifdef _DEBUG
					std::cout << s << ": dangling (1) = " << dangling << "\n";
#endif

					// complete dangling factor
					dangling = ( alpha * dangling + 1 - alpha ) / static_cast< IOType >( n );

#ifdef _DEBUG
					std::cout << s << ": dangling (2) = " << dangling << "\n";
#endif
				}

				// multiply with row-normalised link matrix (no change to dangling rows)
				// note that the later eWiseLambda requires the output be dense
				ret = ret ? ret : set< descriptors::dense >( pr_nextnext, 0 ); assert( ret == SUCCESS );
				ret = ret ? ret : vxm< descriptors::dense | descr >( pr_nextnext, pr_next, L, realRing );
				assert( ret == SUCCESS );
				assert( n == grb::nnz( pr_nextnext ) );

#if defined _DEBUG && defined _GRB_WITH_LPF
				for( size_t dbg = 0; dbg < spmd<>::nprocs(); ++dbg ) {
					if( dbg == s ) {
						std::cout << s << ": nextnext PR array (after vxm):\n";
						(void) eWiseLambda< descriptors::dense >(
							[ &pr_nextnext, s ]( const size_t i ) {
								#pragma omp critical
								if( i < 10 )
									std::cout << i << ", " << s << ": " << pr_nextnext[ i ] << "\n";
							}, pr_nextnext
						);
					}
					(void) spmd<>::sync();
				}
				for( size_t k = 0; k < spmd<>::nprocs(); ++k ) {
					if( spmd<>::pid() == k ) {
						std::cout << "old pr \t scaled input \t alpha * pr * H at PID "
							<< k << "\n";
						(void) eWiseLambda< descriptors::dense >(
							[ &pr, &pr_next, &pr_nextnext ]( const size_t i ) {
								#pragma omp critical
								{
									std::cout << pr[ i ] << "\t" << pr_next[ i ] << "\t"
										<< pr_nextnext[ i ] << "\n";
								}
							}, pr, pr_next, pr_nextnext
						);
					}
					(void) spmd<>::sync();
				}
#endif

				// calculate & normalise new pr and calculate residual
				if( ret == SUCCESS ) {
					// can we reduce via lambdas?
					if( grb::Properties<>::writableCaptured ) {
						// yes, we can. So update pr[ i ] and calculate residual simultaneously
						ret = eWiseLambda< descriptors::dense >(
							[ &pr, &pr_nextnext, &dangling, &residual, &zero ]( const size_t i ) {
								// cache old pagerank vector
								const IOType oldval = pr[ i ];
								// set new pagerank vector
								pr[ i ] = pr_nextnext[ i ] + dangling;
								// update residual
								if( oldval > pr[ i ] ) {
									residual += oldval - pr[ i ];
								} else {
									residual += pr[ i ] - oldval;
								}
								pr_nextnext[ i ] = zero;
							}, pr, pr_nextnext
						);
						// reduce process-local residual
						assert( ret == SUCCESS );
						if( ret == SUCCESS ) {
							ret = grb::collectives<>::allreduce(
								residual,
								grb::operators::add< double >()
							);
						}
						assert( ret == SUCCESS );
					} else {
						// we cannot reduce via lambdas, so calculate new pr vector
						ret = foldl< descriptors::dense >( pr_nextnext, dangling, addM );
						assert( ret == SUCCESS );
						// do a dot product under the one-norm ``ring''
						if( ret == SUCCESS ) {
							residual = zero;
							ret = dot< descriptors::dense >(
								residual,
								pr, pr_nextnext,
								addM, grb::operators::abs_diff< IOType >()
							);
							assert( ret == SUCCESS );
						}
						if( ret == SUCCESS ) {
							// next pr vector becomes current pr vector
							std::swap( pr, pr_nextnext );
						}
					}
				}

				// update iteration count
				(void) ++iter;

				// check convergence
				if( conv != zero && residual <= conv ) { break; }

#ifdef _DEBUG
				if( grb::spmd<>::pid() == 0 ) {
					std::cout << "Iteration " << iter << ", "
						<< "residual = " << residual << std::endl;
				}
#endif

			} while( ret == SUCCESS && iter < max );

			// check if the user requested any stats, and output if yes
			if( iterations != nullptr ) {
				*iterations = iter;
			}
			if( quality != nullptr ) {
				*quality = residual;
			}

			// return the appropriate exit code
			if( ret != SUCCESS ) {
				if( spmd<>::pid() == 0 ) {
					std::cerr << "Error while running simple pagerank algorithm: "
						<< toString( ret ) << "\n";
				}
				return ret;
			} else if( residual <= conv ) {
#ifdef _DEBUG
				if( spmd<>::pid() == 0 ) {
					std::cerr << "Info: simple pagerank converged after " << iter
						<< " iterations.\n";
				}
#endif
				return SUCCESS; // converged!
			} else {
#ifdef _DEBUG
				if( spmd<>::pid() == 0 ) {
					std::cout << "Info: simple pagerank did not converge after "
						<< iter << " iterations.\n";
				}
#endif
				return FAILED; // not converged
			}
		}

	} // namespace algorithms

} // namespace grb

#endif // end _H_GRB_PAGERANK

