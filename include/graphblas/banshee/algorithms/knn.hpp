
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
 * @date 27th of April, 2017
 */

#ifndef _H_GRB_KNN
#define _H_GRB_KNN

#include <graphblas.hpp>

namespace grb {

	namespace algorithms {

		/**
		 * Given a graph and a source vertex, indicates which vertices are contained
		 * within k hops.
		 *
		 * @param[out] u The distance-k neighbourhood. Any prior contents will be
		 *               ignored.
		 * @param[in]  A The input graph.
		 * @param[in]  source The source vertex index.
		 * @param[in]  k The distance of the neighbourhood.
		 *
		 * @returns #SUCCESS If the computation completes successfully.
		 * @returns #MISMATCH If the dimensions of \a u do not match that of \a A.
		 * @returns #MISMATCH If \a source is not in range of \a A.
		 *
		 * \note The banshee version differs from the regular version in that (S)SSR
		 *       does not support boolean data types. Therefore the k-NN is re-cast
		 *       using doubles instead.
		 */
		template< Descriptor descr, typename OutputType, typename InputType >
		RC knn( Vector< OutputType > & u, const Matrix< InputType > & A, const size_t source, const size_t k ) {
			// the nearest-neighbourhood ring
#ifndef SSR
			Semiring< operators::logical_or< bool >, operators::logical_and< bool >, identities::logical_false, identities::logical_true > ring;
#else
			Semiring< operators::logical_or< double >, operators::logical_and< double >, identities::logical_false, identities::logical_true > ring;
#endif

			// check input
			if( nrows( A ) != ncols( A ) ) {
				return MISMATCH;
			}
			if( nnz( u ) != 0 ) {
				clear( u );
			}

			// the initial input vector
#ifndef SSR
			Vector< bool > in( ncols( A ) );
#else
			Vector< double > in( ncols( A ) );
#endif
#ifdef _DEBUG
			printf( "grb::algorithms::knn called with source %d and k %d. In "
					"vector has size %d and %d nonzeroes.\n",
				(int)source, (int)k, (int)size( in ), (int)nnz( in ) );
#endif

#ifndef SSR
			RC ret = setElement( in, true, source );
#else
			RC ret = setElement( in, 1.0, source );
#endif

			// do sparse matrix powers on the given ring
			if( ret == SUCCESS ) {
				if( descr & descriptors::transpose_matrix ) {
					ret = mpv< ( descr | descriptors::add_identity ) & ~( descriptors::transpose_matrix ) >( u, A, k, in, ring );
				} else {
					ret = mpv< descr | descriptors::add_identity | descriptors::transpose_matrix >( u, A, k, in, ring );
				}
			}

			// done
			return ret;
		}

	} // namespace algorithms

} // namespace grb

#endif
