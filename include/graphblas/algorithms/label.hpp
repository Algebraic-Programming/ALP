
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
 * @author J. M. Nash
 * @date 21st of March, 2017
 */

#ifndef _H_GRB_LABEL
#define _H_GRB_LABEL

#include <iostream>

#include <graphblas.hpp>

namespace grb {

	namespace algorithms {

#ifdef _DEBUG
		constexpr size_t MaxPrinting = 20;
		constexpr size_t MaxAnyPrinting = 100;

		// take a vector and display with a message
		static void printVector( Vector< double > & v, std::string message ) {
			size_t zeros = 0;
			size_t ones = 0;
			size_t size = grb::size( v );
			if( size > MaxAnyPrinting )
				return;
			printf( "%s \n", message.c_str() );
			for( Vector< double >::const_iterator it = v.begin(); it != v.end(); ++it ) {
				const std::pair< size_t, double > iter = *it;
				const double val = iter.second;
				if( val < INFINITY ) {
					if( size > MaxPrinting ) {
						zeros += ( val == 0 ) ? 1 : 0;
						ones += ( val == 1 ) ? 1 : 0;
					} else
						printf( "%lf ", val );
				}
			}
			if( size > MaxPrinting )
				printf( "%zd zeros; %zd ones;", zeros, ones );
			printf( "\n" );
		}
#endif

		/**
		 * The label propagation algorithm.
		 *
		 * @tparam IOType   The value type of the label vector. This will
		 *                  determine the precision of all computations this
		 *                  algorithm performs.
		 * @param[in] y     Vector holding the initial labels from a total set of \a n vertices.
		 *
		 * @param[in] W     Sparse symmetric matrix of size \a n*n, holding the weights between the n vertices.
		 *
		 * @param[in] n     The total number of vertices.
		 *
		 * @param[in] l     The number of vertices with an initial label.
		 *
		 * @param[in] out   The resulting labelled vector representing the n vertices.
		 *
		 * @param[in] maxIterations The maximum number of iterations this algorithm may execute.
		 *                          Optional. Default value: 1000.
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
		 *
		 * [1] Kamvar, Haveliwala, Manning, Golub; `Extrapolation methods for
		 *     accelerating the PageRank computation', ACM Press, 2003.
		 */
		template< typename IOType >
		RC label( const Vector< IOType > & y, const Matrix< IOType > & W, size_t n, size_t l, Vector< IOType > & out, const size_t MaxIterations = 1000 ) {
			// label propagation vectors and matrices operate over the real domain
			Semiring< grb::operators::add< IOType >, grb::operators::mul< IOType >, grb::identities::zero, grb::identities::one > reals;
			grb::operators::not_equal< IOType, IOType, bool > notEqualOp;
			grb::Monoid< grb::operators::logical_or< bool >, grb::identities::logical_false > orMonoid;

			const size_t s = spmd<>::pid();

			// compute the diagonal matrix D from the weight matrix W
			// we represent D as a vector so we can use it to generate the probabilities matrix P
			Vector< IOType > multiplier( n );
			RC ret = set( multiplier, static_cast< IOType >( 1 ) ); // a vector of 1's

			Vector< IOType > diagonals( n );
			assert( ret == SUCCESS );
			ret = ret ? ret : mxv( diagonals, W, multiplier, reals ); // W*multiplier will sum each row
#ifdef _DEBUG
			printVector( diagonals, "diagonals matrix in vector form" );
#endif

			// compute the probabilistic transition matrix P as inverse of D * W
			// use diagonals vector to directly compute probabilistic transition matrix P
			// only the existing non-zero elements in W will map to P
			// the inverse of D is represented via the inverse element in the diagonals vector
			// update: the application of Dinv is now done within a lambda following the mxv
			//         on the original matrix P

			// make diagonals equal its inverse
			ret = ret ? ret : eWiseLambda(
							[ &diagonals ]( const size_t i ) {
								diagonals[ i ] = 1.0 / diagonals[ i ];
							},
							diagonals );

			// set up current and new solution functions
			Vector< IOType > f( n );
			Vector< IOType > fNext( n );
			Vector< bool > mask( n );
			for( size_t i = 0; ret == SUCCESS && i < l; ++i ) {
				ret = setElement( mask, true, i );
			}

			// fix f = y for the input set of labels
			ret = ret ? ret : set( f, y );

			// whether two successive solutions are different
			// initially set to true so that the computation may begin
			bool different = true;
			// compute f as P*f
			// main loop completes when function f is stable
			size_t iter = 1;
			while( ret == SUCCESS && different && iter < MaxIterations ) {

#ifdef _DEBUG
				if( n < MaxAnyPrinting ) {
					printf( ">>> Iteration %zd \n", iter );
				}

				// fNext = P*f
				printf( "*** PRE  f = %zd, fNext = %zd\n", nnz( f ), nnz( fNext ) );
#endif
				ret = mxv( fNext, W, f, reals );
#ifdef _DEBUG
				printf( "*** POST f = %zd, fNext = %zd\n", nnz( f ), nnz( fNext ) );
				printVector( f, "Previous iteration solution" );
				printVector( fNext, "New iteration solution" );
#endif

				// maintain the solution function in domain {0,1}
				// clamps the first l labelled nodes
				// can use the masked variant of vector assign when available ?
				ret = ret ? ret : eWiseLambda(
								[ &fNext, &diagonals, &l ]( const size_t i ) {
									fNext[ i ] = ( fNext[ i ] * diagonals[ i ] < 0.5 ? 0 : 1 );
								},
								fNext, diagonals );
#ifdef _DEBUG
				printVector( fNext, "New iteration solution after threshold cutoff" );
				printf( "*** PRE  fNext = %zd, mask = %zd\n", nnz( fNext ), nnz( mask ) );
#endif
				ret = ret ? ret : set( fNext, mask, f );
#ifdef _DEBUG
				printf( "*** POST fNext = %zd\n", nnz( fNext ) );
				printVector( fNext, "New iteration solution after threshold cutoff and clamping" );
#endif
				// test for stability
				ret = ret ? ret : dot( different, f, fNext, orMonoid, notEqualOp );

				// update f for the next iteration
#ifdef _DEBUG
				printf( "*** PRE  f = %zd\n", nnz( f ) );
#endif
				ret = ret ? ret : set( f, fNext );
#ifdef _DEBUG
				printf( "*** POST f = %zd\n", nnz( f ) );
#endif
				(void)++iter;
			}

			// output the result
			if( ret == SUCCESS ) {
				ret = set( out, f );
			}
			if( s == 0 ) {
				printf( ">>> %zd total iterations \n", iter - 1 );
			}

			// done
			return ret;
		}

	} // namespace algorithms

} // namespace grb

#endif // end _H_GRB_LABEL
