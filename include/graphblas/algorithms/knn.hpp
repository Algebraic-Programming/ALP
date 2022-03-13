
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
 * @date: 27th of April, 2017
 */

#ifndef _H_GRB_KNN
#define _H_GRB_KNN

#include "graphblas/algorithms/mpv.hpp"

#include <graphblas.hpp>

namespace grb {

	namespace algorithms {

		/**
		 * Given a graph and a source vertex, indicates which vertices are contained
		 * within \a k hops.
		 *
		 * This implementation is based on the matrix powers kernel over a Boolean
		 * semiring.
		 *
		 * @param[out]    u    The distance-k neighbourhood. Any prior contents will
		 *                     be ignored.
		 * @param[in]     A    The input graph in (square) matrix form
		 * @param[in]  source  The source vertex index.
		 * @param[in]     k    The neighbourhood distance, or the maximum number of
		 *                     hops in a breadth-first search.
		 *
		 * This algorithm requires the following workspace:
		 *
		 * @param[in,out] buf1 A buffer vector. Must match the size of \a A.
		 * @param[in,out] buf2 A buffer vector. Must match the size of \a A.
		 *
		 * For \f$ n \times n \f$ matrices \a A, the capacity of \a u, \a buf1, and
		 * \a buf2 must equal \f$ n \f$.
		 *
		 * @returns #grb::SUCCESS  When the computation completes successfully.
		 * @returns #grb::MISMATCH When the dimensions of \a u do not match that of
		 *                         \a A.
		 * @returns #grb::MISMATCH If \a source is not in range of \a A.
		 * @returns #grb::ILLEGAL  If one or more of \a u, \a buf1, or \a buf2 has
		 *                         insufficient capacity.
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
		template< Descriptor descr, typename OutputType, typename InputType >
		RC knn(
			Vector< OutputType > &u, const Matrix< InputType > &A,
			const size_t source, const size_t k,
			Vector< bool > &buf1
		) {
			// the nearest-neighbourhood ring
			Semiring<
				operators::logical_or< bool >, operators::logical_and< bool >,
				identities::logical_false, identities::logical_true
			> ring;

			// check input
			const size_t n = nrows( A );
			if( n != ncols( A ) ) {
				return MISMATCH;
			}
			if( size( buf1 ) != n ) {
				return MISMATCH;
			}
			if( size( u ) != n ) {
				return MISMATCH;
			}
			if( capacity( u ) != n ) {
				return ILLEGAL;
			}
			if( capacity( buf1 ) != n ) {
				return ILLEGAL;
			}

			// prepare
			RC ret = SUCCESS;
			if( nnz( u ) != 0 ) {
				ret = clear( u );
			}
			if( nnz( buf1 ) != 0 ) {
				ret = ret ? ret : clear( buf1 );
			}
#ifdef _DEBUG
			std::cout << "grb::algorithms::knn called with source " << source << " "
				<< "and k " << k << ".\n";
#endif
			ret = ret ? ret : setElement( buf1, true, source );

			// do sparse matrix powers on the given ring
			if( ret == SUCCESS ) {
				if( descr & descriptors::transpose_matrix ) {
					ret = mpv< (descr | descriptors::add_identity) &
						~( descriptors::transpose_matrix )
					>( u, A, k, buf1, buf1, ring );
				} else {
					ret = mpv< descr | descriptors::add_identity |
						descriptors::transpose_matrix
					>( u, A, k, buf1, buf1, ring );
				}
			}

			// done
			return ret;
		}

	} // namespace algorithms

} // namespace grb

#endif

