
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
		 * The PageRank algorithm, including some optimisations to speed up
		 * convergence.
		 *
		 * This implementation always makes at least one iteration, even if the
		 * initial guess was spot-on.
		 *
		 * @tparam descr    The descriptor under which to perform the computation.
		 * @tparam IOType   The value type of the pagerank vector. This will
		 *                  determine the precision of all computations this
		 *                  algorithm performs.
		 * @tparam NonzeroT The type of the elements of the nonzero matrix.
		 *
		 * @param[in,out] pr  Vector of size \a n where the PageRank vectors will
		 *                    be stored after function exit. On function entry, the
		 *                    contents of this vector will be taken as the initial
		 *                    guess to the final result, but only if the size is
		 *                    nonzero. If the size on input is zero, this algorithm
		 *                    will instead set the initial guess will be the
		 *                    one-vector. The vector on input need not be normalised,
		 *                    this implementation will take care of this.
		 * @param[in]    L    The input link matrix. How its elements are interpreted
		 *                    depends on the given \a ring. It is recommended that
		 *                    \a NonzeroT is \a bool and that a multiplication with
		 *                    \a true results in a copy of the input vector element
		 *                    as output, which is handled under regular addition.
		 * @param[in]  alpha  The scaling factor. Optional; the default value is 0.85.
		 *                    This value must be smaller than 1, and more than 0.
		 * @param[in]  conv   If the difference between two successive iterations, in
		 *                    terms of its one-norm, is less than this value, then the
		 *                    PageRank vector is considered converged and this
		 *                    algorithm exits successfully. Optional; the default
		 *                    value is \f$ 10^{-8} \f$. If this value is set to zero,
		 *                    the algorithm will continue until \a max iterations are
		 *                    reached or until the iterates have bit-wise converged
		 *                    (the latter is very unlikely and does \em not imply
		 *                     perfect convergence in the mathematical sense).
		 * @param[in]  max    The maximum number of power iterations. Optional; the
		 *                    default value is 1000. This value must be larger than 0.
		 *
		 * @param[out] iterations (Optional, default is \a NULL) A pointer where to
		 *                   store the number of iterations the call to this
		 *                   algorithm took. If a \a NULL pointer is passed, this
		 *                   information will not be returned.
		 * @param[out] quality (Optional, default is \a NULL) A pointer where to store
		 *                   the last computed residual. If a \a NULL pointer is
		 *                   passed, this information will not be returned.
		 *
		 * This algorithm requires three buffer vectors \a pr_next, \a pr_nextnext,
		 * and \a row_sum. These vectors must be of the same size as \a pr.
		 *
		 * @returns SUCCESS  If the computation converged within \a max iterations.
		 * @returns MISMATCH If the dimensions of \a pr and \a L do not match, or if
		 *                   the matrix \a L is not square. When this error code is
		 *                   returned, this function will not have any other effects.
		 * @returns ILLEGAL  If one of the arguments passed to this function is
		 *                   illegal.
		 * @returns FAILED   If the PageRank method did not converge using the given
		 *                   values. The last computed PageRank iterate will be
		 *                   stored in \a pr on output, but should not be used.
		 * @returns PANIC    If the underlying GraphBLAS implementation has failed.
		 */
		template< Descriptor descr = descriptors::no_operation, typename IOType, typename NonzeroT >
		RC simple_pagerank( Vector< IOType > & pr,
			const Matrix< NonzeroT > & L, // PageRank input and output
			Vector< IOType > & pr_next,   // Three buffers
			Vector< IOType > & pr_nextnext,
			Vector< IOType > & row_sum,
			const IOType alpha = 0.85, // PageRank parameters
			const IOType conv = 0.0000001,
			const size_t max = 1000,          // Power method arguments
			size_t * const iterations = NULL, // Optional algo call stats
			double * const quality = NULL ) {
			grb::Monoid< grb::operators::add< IOType >, grb::identities::zero > addM;
			grb::Semiring< grb::operators::add< IOType >, grb::operators::mul< IOType >, grb::identities::zero, grb::identities::one > realRing;
#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
			const auto s = spmd<>::pid();
#endif
#endif
			// runtime sanity checks: the output dimension should match the number of rows in L
			if( size( pr ) != nrows( L ) ) {
				return MISMATCH;
			}
			// the input dimension should match the number of columns in L
			//(and the matrix L must be square)
			if( size( pr ) != ncols( L ) ) {
				return MISMATCH;
			}
			if( ( size( pr_next ) != size( pr ) ) || ( size( pr_nextnext ) != size( pr ) ) || ( size( row_sum ) != size( pr ) ) ) {
				return MISMATCH;
			}
			// alpha must be within 0 and 1 (both exclusive)
			if( alpha <= 0 || alpha >= 1 ) {
				return ILLEGAL;
			}
			// max must be larger than 0
			if( max <= 0 ) {
				return ILLEGAL;
			}

			// get problem dimension
			const size_t n = nrows( L );

			// running error code
			RC ret = SUCCESS;

			// make initial guess
			ret = set( pr, static_cast< IOType >( 1 ) / static_cast< IOType >( n ) );
			assert( ret == SUCCESS );

			// initialise all temporary vectors to default dense values
			if( ret == SUCCESS ) {
				ret = set( row_sum, 0 );
				assert( ret == SUCCESS );
			}
			if( ret == SUCCESS ) {
				ret = set( pr_nextnext, 0 );
				assert( ret == SUCCESS );
			}

			// calculate row sums
			Semiring< operators::add< IOType, IOType, IOType >, operators::left_assign_if< IOType, bool, IOType >, identities::zero, identities::logical_true > pattern_ring;

			if( ret == SUCCESS ) {
				ret = set( pr_next, 1 ); // abuses pr_next as temporary vector
				assert( ret == SUCCESS );
			}
			if( ret == SUCCESS ) {
				ret = vxm< descr | grb::descriptors::dense | grb::descriptors::transpose_matrix >( row_sum, pr_next, L,
					pattern_ring ); // pr_next is now free for further use
				assert( ret == SUCCESS );
			}

#ifdef _DEBUG
			std::cout << "Prelude to iteration 0:\n";
			eWiseLambda(
				[ &row_sum, &pr_next, &pr ]( const size_t i ) {
					#pragma omp critical
					std::cout << i << ": " << row_sum[ i ] << "\t" << pr[ i ] << "\t" << pr_next[ i ] << "\n";
				},
				pr, pr_next, row_sum );
#endif

			// calculate in-place the inverse of row sums, and keep zero if dangling
			if( ret == SUCCESS ) {
				// this eWiseLambda is always supported since alpha is a non-GraphBLAS object used exclusively in read-only mode
				ret = eWiseLambda(
					[ &row_sum, &alpha ]( const size_t i ) {
						assert( row_sum[ i ] >= 0 );
						if( row_sum[ i ] > 0 ) {
							row_sum[ i ] = alpha / row_sum[ i ];
						}
					},
					row_sum );
				assert( ret == SUCCESS );
			}

#ifdef _DEBUG
			std::cout << "Row sum array:\n";
			eWiseLambda(
				[ &row_sum ]( const size_t i ) {
					#pragma omp critical
					std::cout << i << ": " << row_sum[ i ] << "\n";
				},
				row_sum );
#endif

			// some control variables
			size_t iter = 0; //#iterations, initalise to zero
			IOType dangling; // used for caching the contribution of random jumps from dangling nodes
			IOType residual; // declare residual (will be computed inside the do-while loop)

			// main loop
			do {
				// reset iteration-local values
				residual = dangling = 0;

#ifdef _DEBUG
				std::cout << "Current PR array:\n";
				eWiseLambda(
					[ &pr ]( const size_t i ) {
						#pragma omp critical
						if( i < 8 ) {
							std::cout << i << ": " << pr[ i ] << "\n";
						}
					},
					pr );
#endif

				// calculate dangling factor and do scaling
				if( ret == SUCCESS ) {
					// can we reduce via lambdas?
					if( Properties<>::writableCaptured ) {
						// yes we can, so save one unnecessary stream on pr
						ret = eWiseLambda(
							[ &pr_next, &row_sum, &dangling, &pr ]( const size_t i ) {
								// calculate dangling contribution
								if( row_sum[ i ] == 0 ) {
									dangling += pr[ i ];
									pr_next[ i ] = 0;
								} else {
									// pre-scale input
									pr_next[ i ] = pr[ i ] * row_sum[ i ];
								}
							},
							row_sum, pr, pr_next );
						// allreduce dangling factor
						assert( ret == SUCCESS );
						if( ret == SUCCESS ) {
							ret = grb::collectives<>::allreduce( dangling, grb::operators::add< double >() );
							assert( ret == SUCCESS );
						}
					} else {
						// otherwise we have to handle the reduction separately
						ret = foldl< grb::descriptors::invert_mask >( dangling, pr, row_sum, addM );
						assert( ret == SUCCESS );

						// separately from the element-wise multiplication here
						if( ret == SUCCESS ) {
							set( pr_next, 0 );
							ret = eWiseApply( pr_next, pr, row_sum, grb::operators::mul< double >() );
							assert( ret == SUCCESS );
						}
					}
				}

#if defined _DEBUG && defined _GRB_WITH_LPF
				for( size_t dbg = 0; dbg < spmd<>::nprocs(); ++dbg ) {
					const auto s = spmd<>::pid();
					if( dbg == s ) {
						std::cout << "Next PR array (under construction):\n";
						eWiseLambda(
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

#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
				std::cout << s << ": dangling (1) = " << dangling << "\n";
#endif
#endif

				// complete dangling factor
				dangling = ( alpha * dangling + 1 - alpha ) / static_cast< IOType >( n );

#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
				std::cout << s << ": dangling (2) = " << dangling << "\n";
#endif
#endif

				// multiply with row-normalised link matrix (no change to dangling rows)
				if( ret == SUCCESS ) {
					// note that the following eWiseLambda requires the output be dense
					ret = set( pr_nextnext, 0 );
					assert( ret == SUCCESS );
					if( ret == SUCCESS ) {
						ret = vxm< descr >( pr_nextnext, pr_next, L, realRing );
					}
					assert( ret == SUCCESS );
					assert( grb::size( pr_nextnext ) == grb::nnz( pr_nextnext ) );
				}

#if defined _DEBUG && defined _GRB_WITH_LPF
				for( size_t dbg = 0; dbg < spmd<>::nprocs(); ++dbg ) {
					if( dbg == s ) {
						std::cout << s << ": nextnext PR array (after vxm):\n";
						eWiseLambda(
							[ &pr_nextnext, s ]( const size_t i ) {
								#pragma omp critical
								if( i < 10 )
									std::cout << i << ", " << s << ": " << pr_nextnext[ i ] << "\n";
							},
							pr_nextnext );
					}
					spmd<>::sync();
				}
				for( size_t k = 0; k < spmd<>::nprocs(); ++k ) {
					if( spmd<>::pid() == k ) {
						std::cout << "old pr \t scaled input \t alpha * pr * H at PID " << k << "\n";
						eWiseLambda(
							[ &pr, &pr_next, &pr_nextnext ]( const size_t i ) {
								std::cout << pr[ i ] << "\t" << pr_next[ i ] << "\t" << pr_nextnext[ i ] << "\n";
							},
							pr, pr_next, pr_nextnext );
					}
				}
#endif

				// calculate & normalise new pr and calculate residual
				if( ret == SUCCESS ) {
					// can we reduce via lambdas?
					if( grb::Properties<>::writableCaptured ) {
						// yes, we can. So update pr[ i ] and calculate residual simultaneously
						ret = eWiseLambda(
							[ &pr, &pr_nextnext, &dangling, &residual ]( const size_t i ) {
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
								pr_nextnext[ i ] = 0;
							},
							pr, pr_nextnext );
						// reduce process-local residual
						assert( ret == SUCCESS );
						if( ret == SUCCESS ) {
							ret = grb::collectives<>::allreduce( residual, grb::operators::add< double >() );
						}
						assert( ret == SUCCESS );
					} else {
						// we cannot reduce via lambdas, so calculate new pr vector
						ret = foldl< descriptors::dense >( pr_nextnext, dangling, addM );
						// do a dot product under the one-norm ``ring''
						assert( ret == SUCCESS );
						if( ret == SUCCESS ) {
							residual = 0;
							ret = dot< descriptors::dense >( residual, pr, pr_nextnext, addM, grb::operators::abs_diff< IOType >() );
							assert( ret == SUCCESS );
						}
						// next pr vector becomes current pr vector
						std::swap( pr, pr_nextnext );
					}
				}
				// update iteration count
				++iter;

#ifdef _DEBUG
				if( grb::spmd<>::pid() == 0 ) {
					std::cout << "Iteration " << iter << ", residual = " << residual << std::endl;
				}
#endif

			} while( residual > conv && iter < max && ret == SUCCESS );

			// check if the user requested any stats
			if( iterations != NULL ) {
				*iterations = iter;
			}
			if( quality != NULL ) {
				*quality = residual;
			}

			// return the appropriate exit code
			if( ret != SUCCESS ) {
#ifndef NDEBUG
				std::cerr << "Error (" << ret << ") while running simple pagerank algorithm" << std::endl;
#endif
				return ret;
			} else if( residual <= conv ) {
#ifndef NDEBUG
				if( spmd<>::pid() == 0 ) {
					std::cout << "Info: simple pagerank converged after " << iter << " iterations." << std::endl;
				}
#endif
				return SUCCESS; // converged!
			} else {
#ifndef NDEBUG
				if( spmd<>::pid() == 0 ) {
					std::cout << "Did not converge after " << iter << " iterations." << std::endl;
				}
#endif
				return FAILED; // not converged
			}
		}

	} // namespace algorithms

} // namespace grb

#endif // end _H_GRB_PAGERANK
