
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
		 * within k hops.
		 *
		 * This implementation is based on the matrix powers kernel over a Boolean
		 * semiring.
		 *
		 * @param[out] u The distance-k neighbourhood. Any prior contents will be
		 *               ignored.
		 * @param[in]  A The input graph.
		 * @param[in]  source The source vertex index.
		 * @param[in]  k The neighbourhood distance, or the maximum number of
		 *               hops in a breadth-first search.
		 * @param[in,out] buf1 A buffer vector used internally. Must match the
		 *                     number of columns of \a A.
		 * @param[in,out] buf2 A buffer vector used internally. Must match the
		 *                     number of columns of \a A.
		 *
		 * @returns #SUCCESS If the computation completes successfully.
		 * @returns #MISMATCH If the dimensions of \a u do not match that of \a A.
		 * @returns #MISMATCH If \a source is not in range of \a A.
		 */
		template< Descriptor descr, typename OutputType, typename InputType >
		RC knn( Vector< OutputType > & u, const Matrix< InputType > & A, const size_t source, const size_t k, Vector< bool > & buf1, Vector< bool > & buf2 ) {
			// the nearest-neighbourhood ring
			Semiring< operators::logical_or< bool >, operators::logical_and< bool >, identities::logical_false, identities::logical_true > ring;

			// check input
			if( nrows( A ) != ncols( A ) ) {
				return MISMATCH;
			}
			if( size( buf1 ) != ncols( A ) ) {
				return MISMATCH;
			}
			if( size( u ) != nrows( A ) ) {
				return MISMATCH;
			}
			if( size( buf2 ) != nrows( A ) ) {
				return MISMATCH;
			}
			if( nnz( u ) != 0 ) {
				clear( u );
			}
			if( nnz( buf1 ) != 0 ) {
				clear( buf1 );
			}
#ifdef _DEBUG
			std::cout << "grb::algorithms::knn called with source " << source << " and k " << k << ". In vector has size " << size( buffer ) << " and " << nnz( buffer ) << " nonzeroes.\n";
#endif
			RC ret = setElement( buf1, true, source );

			// do sparse matrix powers on the given ring
			if( ret == SUCCESS ) {
				if( descr & descriptors::transpose_matrix ) {
					ret = mpv< ( descr | descriptors::add_identity ) & ~( descriptors::transpose_matrix ) >( u, A, k, buf1, ring, buf2 );
				} else {
					ret = mpv< descr | descriptors::add_identity | descriptors::transpose_matrix >( u, A, k, buf1, ring, buf2 );
				}
			}

			// done
			return ret;
		}

	} // namespace algorithms

} // namespace grb

#endif
