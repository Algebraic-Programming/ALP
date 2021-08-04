
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
 * @date 30th of March 2017
 */

#ifndef _H_GRB_ALGORITHMS_MPV
#define _H_GRB_ALGORITHMS_MPV

#include <graphblas.hpp>


namespace grb {
	namespace algorithms {

		/**
		 * The matrix powers kernel.
		 *
		 * Calculates \f$ y = A^k x \f$ for some integer \f$ k \geq 0 \f$ using the
		 * given semiring.
		 *
		 * @tparam descr          The descriptor used to perform this operation.
		 * @tparam Ring           The semiring used.
		 * @tparam IOType         The output vector type.
		 * @tparam InputType      The nonzero type of matrix elements.
		 * @tparam implementation Which implementation to use.
		 *
		 * @param[out] u    The output vector. Contents shall be overwritten. The
		 *                  supplied vector must match the row dimension size of \a A.
		 * @param[in]  A    The input matrix A. The supplied matrix must match the
		 *                  dimensions of \a u and \a v.
		 * @param[in]  v    The input vector v. The supplied vector must match the
		 *                  column dimension size of \a A. It may not equal \a u.
		 * @param[in]  ring The semiring to be used. This defines the additive and
		 *                  multiplicative monoids to be used.
		 *
		 * \parblock
		 * \par Performance guarantees
		 *      -# This call takes \f$ k\Theta(\mathit{nz}) + \mathcal{O}(m+n)\f$
		 *         work, where \f$ nz \f$ equals the number of nonzeroes in the
		 *         matrix, and \f$ m, n \f$ the dimensions of the matrix.
		 *
		 *      -# This call takes \f$ \mathcal{O}(\mathit{m+n}) \f$ memory beyond the
		 *         memory already used by the application when this function is called.
		 *
		 *      -# This call incurs at most
		 *         \f$ k\cdot\mathit{nz}(
		 *                 \mathit{sizeof}(\mathit{D1} + \mathit{sizeof}(\mathit{D2}) +
		 *                 \mathit{sizeof}(\mathit{D3} + \mathit{sizeof}(\mathit{D4}) +
		 *                 \mathit{sizeof}(\mathit{RI} + \mathit{sizeof}(\mathit{CI})
		 *         ) + \mathcal{O}(1) \f$
		 *         bytes of data movement, where RI is the row index data type
		 *         used by the input matrix \a A, and CI is the column index data
		 *         type used by the input matrix \a A.
		 * \endparblock
		 */
		template< Descriptor descr, class Ring, typename IOType, typename InputType >
		RC mpv( Vector< IOType > & u, const Matrix< InputType > & A, const size_t k, const Vector< IOType > & v, const Ring & ring, Vector< IOType > & temp ) {
			static_assert( ! ( descr & descriptors::no_casting ) ||
					( std::is_same< IOType, typename Ring::D4 >::value && std::is_same< InputType, typename Ring::D2 >::value && std::is_same< IOType, typename Ring::D1 >::value &&
						std::is_same< IOType, typename Ring::D3 >::value ),
				"grb::mpv : some containers were passed with element types that do not match the given semiring domains." );

			// runtime check
			if( size( u ) != size( v ) || nrows( A ) != ncols( A ) ) {
				return MISMATCH;
			}
			if( static_cast< const void * >( &u ) == static_cast< const void * >( &v ) ) {
				return ILLEGAL;
			}
			// catch trivial case
			if( k == 0 ) {
				return set< descr >( u, v );
			}
			// otherwise, do at least one multiplication
#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
			std::cout << "init: input vector nonzeroes is " << grb::nnz( v ) << ".\n";
#else
			printf( "init: input vector nonzeroes is %d\n", (int)grb::nnz( v ) );
#endif
#endif
			RC ret = mxv< descr >( u, A, v, ring );
			if( k == 1 ) {
				return ret;
			}
			// do any remaining multiplications using a temporary output vector
			bool copy;
			ret = ret ? ret : clear( temp );
			for( size_t iterate = 1; ret == SUCCESS && iterate < k; iterate += 2 ) {
				// multiply with output into temporary
				copy = true;
#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
				std::cout << "up: input vector nonzeroes is " << grb::nnz( u ) << "\n";
#else
				printf( "up: input vector nonzeroes is %d\n", (int)grb::nnz( u ) );
#endif
#endif
				ret = mxv< descr >( temp, A, u, ring );
				// check if this was the final multiplication
				assert( iterate <= k );
				if( iterate == k || ret != SUCCESS ) {
					break;
				}
				// multiply with output into u
				copy = false;
#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
				std::cout << "down: input vector nonzeroes is " << grb::nnz( temp ) << "\n";
#else
				printf( "down: input vector nonzeroes is %d\n", (int)grb::nnz( temp ) );
#endif
#endif
				ret = mxv< descr >( u, A, temp, ring );
			}
			// swap u and temp, if required
			if( ret == SUCCESS && copy ) {
				std::swap( u, temp );
			}
			// done
			return ret;
		}

	} // namespace algorithms

} // namespace grb

#endif // end ``_H_GRB_ALGORITHMS_MPV''
