
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
 * Implements label propagation.
 *
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
		static void printVector(
			const Vector< double > &v, const std::string message
		) {
			size_t zeros = 0;
			size_t ones = 0;
			size_t size = grb::size( v );
			if( size > MaxAnyPrinting ) {
				return;
			}
			std::cerr << "\t " << message << ": ";
			for( Vector< double >::const_iterator it = v.begin(); it != v.end(); ++it ) {
				const std::pair< size_t, double > iter = *it;
				const double val = iter.second;
				if( val < INFINITY ) {
					if( size > MaxPrinting ) {
						zeros += ( val == 0 ) ? 1 : 0;
						ones += ( val == 1 ) ? 1 : 0;
					} else {
						std::cerr << val << " ";
					}
				}
			}
			if( size > MaxPrinting ) {
				std::cerr << zeros << " zeros; " << ones << " ones.";
			}
			std::cerr << "\n";
		}
#endif

		/**
		 * The label propagation algorithm.
		 *
		 * @tparam IOType   The value type of the label vector. This will
		 *                  determine the precision of all computations this
		 *                  algorithm performs.
		 *
		 * @param[out] out The resulting labelled vector representing the n vertices.
		 *
		 * @param[in]  y   Vector holding the initial labels from a total set of \a n
		 *                 vertices. The initial labels are assumed to correspond to
		 *                 the vertices corresponding to the first \a l entries of
		 *                 this vector. The labels must be either 0 ar 1.
		 *
		 * @param[in]  W   Sparse symmetric matrix of size \a n by \a n, holding the
		 *                 weights between the n vertices. The weights must be
		 *                 positive (larger than 0). The matrix may be defective while
		 *                 the corresponding graph may not be connected.
		 *
		 * @param[in]  n   The total number of vertices. If zero, and all of \a out,
		 *                 \a y, and \a W are empty, calling this function is
		 *                 equivalent to a no-op.
		 *
		 * @param[in]  l   The number of vertices with an initial label. Must be
		 *                 larger than zero.
		 *
		 * @param[in] maxIterations The maximum number of iterations this algorithm
		 *                          may execute. Optional. Default value: 1000.
		 *
		 * \note If the underlying graph is not connected then some components may
		 *       be rendered immutable by this algorithm.
		 *
		 * @returns #grb::ILLEGAL  If one of the arguments passed to this function is
		 *                         illegal. The output will be left unmodified.
		 * @returns #grb::ILLEGAL  The vector \a out did not have full capacity
		 *                         available. The output will be left unmodified.
		 * @returns #grb::ILLEGAL  If \a n was nonzero but \a l was zero.
		 * @returns #grb::SUCCESS  If the computation converged within \a max
		 *                         iterations, or if \a n was zero.
		 * @returns #grb::FAILED   If the method did not converge within \a max
		 *                         iterations. The output will contain the latest
		 *                         iterand.
		 * @returns #grb::PANIC    If an unrecoverable error has been encountered. The
		 *                         output as well as the state of ALP/GraphBLAS is
		 *                         undefined.
		 *
		 * [1] Kamvar, Haveliwala, Manning, Golub; `Extrapolation methods for
		 *     accelerating the PageRank computation', ACM Press, 2003.
		 */
		template< typename IOType >
		RC label(
			Vector< IOType > &out,
			const Vector< IOType > &y, const Matrix< IOType > &W,
			const size_t n, const size_t l,
			const size_t MaxIterations = 1000
		) {
			// label propagation vectors and matrices operate over the real domain
			Semiring<
				grb::operators::add< IOType >, grb::operators::mul< IOType >,
				grb::identities::zero, grb::identities::one
			> reals;
			grb::operators::not_equal< IOType, IOType, bool > notEqualOp;
			grb::Monoid<
				grb::operators::logical_or< bool >,
				grb::identities::logical_false
			> orMonoid;
			const IOType zero = reals.template getZero< IOType >();

			// dynamic checks
			if( nrows( W ) != n || ncols( W ) != n ||
				size( y ) != n || size( out ) != n
			) {
				return ILLEGAL;
			}
			if( capacity( out ) != n ) {
				return ILLEGAL;
			}
			if( n == 0 ) {
				return SUCCESS;
			}
			if( l == 0 ) {
				return ILLEGAL;
			}

			const size_t s = spmd<>::pid();

			// compute the diagonal matrix D from the weight matrix W
			// we represent D as a vector so we can use it to generate the probabilities
			// matrix P
			Vector< IOType > multiplier( n );
			RC ret = set( multiplier, static_cast< IOType >(1) ); // a vector of 1's

			Vector< IOType > diagonals( n );
			ret = ret ? ret : set( diagonals, zero );
			ret = ret ? ret : mxv< descriptors::dense >(
				diagonals, W, multiplier, reals
			); // W*multiplier will sum each row
#ifdef _DEBUG
			printVector( diagonals, "diagonals matrix in vector form" );
#endif

			// compute the probabilistic transition matrix P as inverse of D * W
			// use diagonals vector to directly compute probabilistic transition matrix P
			// only the existing non-zero elements in W will map to P
			// the inverse of D is represented via the inverse element in the diagonals
			// vector
			//
			// update: the application of Dinv is now done within a lambda following the
			//         mxv on the original matrix P

			// make diagonals equal its inverse
			ret = ret ? ret : eWiseLambda( [ &diagonals ]( const size_t i ) {
					diagonals[ i ] = 1.0 / diagonals[ i ];
				}, diagonals
			);

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
					std::cerr << "\t iteration " << iter << "\n";
				}

				// fNext = P*f
				std::cerr << "\t pre- set/mxv nnz( f ) = " << nnz( f ) << ", "
					<< "fNext = " << nnz( fNext ) << "\n";
#endif
				ret = ret ? ret : set( fNext, zero );
				ret = ret ? ret : mxv( fNext, W, f, reals );
#ifdef _DEBUG
				std::cerr << "\t post-set/mxv nnz( f ) = " << nnz( f ) << ", "
					<< "nnz( fNext ) = " << nnz( fNext ) << "\n";
				printVector( f, "Previous iteration solution" );
				printVector( fNext, "New iteration solution" );
#endif

				// maintain the solution function in domain {0,1}
				// can use the masked variant of vector assign when available ?
				ret = ret ? ret : eWiseLambda( [ &fNext, &diagonals ]( const size_t i ) {
						fNext[ i ] = ( fNext[ i ] * diagonals[ i ] < 0.5 ? 0 : 1 );
					}, diagonals, fNext
				);
#ifdef _DEBUG
				printVector( fNext, "New iteration solution after threshold cutoff" );
				std::cerr << "\t pre-set  nnz( fNext ) = " << nnz( fNext ) << ", "
					<< "nnz( mask ) = " << nnz( mask ) << "\n";
#endif
				// clamps the first l labelled nodes
				ret = ret ? ret : foldl(
					fNext, mask,
					f,
					grb::operators::right_assign< IOType >()
				);
				assert( ret == SUCCESS );
#ifdef _DEBUG
				std::cerr << "\t post-set nnz( fNext ) = " << nnz( fNext ) << "\n";
				printVector(
					fNext,
					"New iteration solution after threshold cutoff and clamping"
				);
#endif
				// test for stability
				different = false;
				ret = ret ? ret : dot( different, f, fNext, orMonoid, notEqualOp );

				// update f for the next iteration
#ifdef _DEBUG
				std::cerr << "\t pre-set  nnz(f) = " << nnz( f ) << "\n";
#endif
				std::swap( f, fNext );
#ifdef _DEBUG
				std::cerr << "\t post-set nnz(f) = " << nnz( f ) << "\n";
#endif
				// go to next iteration
				(void) ++iter;
			}

			if( ret == SUCCESS ) {
				if( different ) {
					if( s == 0 ) {
						std::cerr << "Info: label propagation did not converge after "
							<< (iter-1) << " iterations\n";
					}
					return FAILED;
				} else {
					if( s == 0 ) {
						std::cerr << "Info: label propagation converged in "
							<< (iter-1) << " iterations\n";
					}
					std::swap( out, f );
					return SUCCESS;
				}
			}

			// done
			if( s == 0 ) {
				std::cerr << "Warning: label propagation exiting with " << toString(ret)
					<< "\n";
			}
			return ret;
		}

	} // namespace algorithms

} // namespace grb

#endif // end _H_GRB_LABEL

