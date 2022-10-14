
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
		 * @param[in]  A    The square input matrix A. The supplied matrix must match
		 *                  the dimensions of \a u and \a v.
		 * @param[in]  v    The input vector v. The supplied vector must match the
		 *                  column dimension size of \a A. It may not be the same
		 *                  vector as \a u.
		 * @param[in]  ring The semiring to be used. This defines the additive and
		 *                  multiplicative monoids to be used.
		 *
		 * This algorithm requires workspace:
		 *
		 * @param[in,out] temp A workspace buffer of matching size to the row
		 *                     dimension of \a A. May be the same vector as \a v,
		 *                     though note that the contents of \a temp on output are
		 *                     undefined.
		 *
		 * This algorithm assumes that \a u and \a temp have full capacity. If this
		 * assumption does not hold, then a two-stage mpv must be employed instead.
		 *
		 * @returns #grb::SUCCESS  If the computation completed successfully.
		 * @returns #grb::ILLEGAL  If \a A is not square.
		 * @returns #grb::MISMATCH If one or more of \a u, \a v, or \a temp has an
		 *                         incompatible size with \a A.
		 * @returns #grb::ILLEGAL  If one or more of \a u or \a temp does not have a
		 *                         full capacity.
		 * @returns #grb::PANIC    If an unrecoverable error has been encountered. The
		 *                         output as well as the state of ALP/GraphBLAS is
		 *                         undefined.
		 * @returns #grb::OVERLAP  If one or more of \a v or \a temp is the same
		 *                         vector as \a u.
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
		template< Descriptor descr, class Ring, typename IOType, typename InputType >
		RC mpv(
			Vector< IOType > &u,
			const Matrix< InputType > &A, const size_t k,
			const Vector< IOType > &v,
			Vector< IOType > &temp,
			const Ring &ring
		) {
			static_assert( !(descr & descriptors::no_casting) ||
					(std::is_same< IOType, typename Ring::D4 >::value &&
						std::is_same< InputType, typename Ring::D2 >::value &&
						std::is_same< IOType, typename Ring::D1 >::value &&
						std::is_same< IOType, typename Ring::D3 >::value
					),
				"grb::mpv : some containers were passed with element types that do not"
				"match the given semiring domains."
			);

			// runtime check
			const size_t n = nrows( A );
			if( n != ncols( A ) ) {
				return ILLEGAL;
			}
			if( size( u ) != n || n != size( v ) ) {
				return MISMATCH;
			}
			if( size( temp ) != n ) {
				return MISMATCH;
			}
			if( capacity( u ) != n ) {
				return ILLEGAL;
			}
			if( capacity( temp ) != n ) {
				return ILLEGAL;
			}
			// catch trivial case
			if( k == 0 ) {
				return set< descr >( u, v );
			}
			// otherwise, do at least one multiplication
#ifdef _DEBUG
			std::cout << "init: input vector nonzeroes is " << grb::nnz( v ) << ".\n";
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
				std::cout << "up: input vector nonzeroes is " << grb::nnz( u ) << "\n";
#endif
				ret = mxv< descr >( temp, A, u, ring );
				// check if this was the final multiplication
				assert( iterate <= k );
				if( iterate + 1 == k || ret != SUCCESS ) {
					break;
				}
				// multiply with output into u
				copy = false;
#ifdef _DEBUG
				std::cout << "down: input vector nonzeroes is " << grb::nnz( temp ) << "\n";
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

