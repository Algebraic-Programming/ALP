
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
 * Implements cosine simularity
 *
 * @author: A. N. Yzelman.
 * @date: 13th of December, 2017.
 */

#ifndef _H_GRB_COSSIM
#define _H_GRB_COSSIM

#include <graphblas.hpp>
#include <graphblas/algorithms/norm.hpp>


#define NO_CAST_ASSERT( x, y, z )                                                                       \
	static_assert( x,                                                                                   \
		"\n\n"                                                                                          \
		"*******************************************************************************************\n" \
		"*     ERROR      | " y " " z ".\n"                                                             \
		"*******************************************************************************************\n" \
		"* Possible fix 1 | Remove no_casting from the template parameters in this call to " y ".   \n" \
		"* Possible fix 2 | For all mismatches in the domains of input and output parameters w.r.t. \n" \
		"*                  the semiring domains, as specified in the documentation of the function.\n" \
		"*                  supply an input argument of the expected type instead.\n"                   \
		"* Possible fix 3 | Provide a compatible semiring where all domains match those of the input\n" \
		"*                  parameters, as specified in the documentation of the function.\n"           \
		"*******************************************************************************************\n" );


namespace grb {

	namespace algorithms {

		/**
		 * Computes the cosine similarity.
		 *
		 * Given two vectors \f$ x, y \f$ of equal size \f$ n \f$, this function
		 * computes \f$ \alpha = \frac{ (x,y) }{ ||x||_2\cdot||y||_2 } \f$.
		 *
		 * The 2-norms and inner products are computed according to the given semi-
		 * ring. However, the norms make use of the standard <tt>sqrt</tt> and so the
		 * algorithm assumes a regular field is used. Effectively, hence, the semiring
		 * controls the precision / data types under which the computation is
		 * performed.
		 *
		 * @tparam descr      The descriptor under which to perform the computation.
		 * @tparam OutputType The type of the output element (scalar).
		 * @tparam InputType1 The type of the first vector.
		 * @tparam InputType2 The type of the second vector.
		 * @tparam Ring       The semiring used.
		 * @tparam Division   Which binary operator correspond to division
		 *                    corresponding to the given \a Ring.
		 *
		 * @param[out] similarity Where to fold the result into.
		 * @param[in]  x          The non-zero left-hand input vector.
		 * @param[in]  y          The non-zero right-hand input vector.
		 * @param[in]  ring       The semiring to compute over.
		 * @param[in]  div        The division operator corresponding to \a ring.
		 *
		 * \note The vectors \a x and/or \a y may be sparse or dense.
		 *
		 * The argument \a div is optional. It will map to grb::operators::divide by
		 * default.
		 *
		 * @returns #grb::SUCCESS  If the computation was successful.
		 * @returns #grb::MISMATCH If the vector sizes do not match. The output
		 *                         \a similarity is untouched -- the call to this
		 *                         algorithm will have no other effects than returning
		 *                         #grb::MISMATCH.
		 * @returns #grb::ILLEGAL  In case \a x is all zero, and/or when \a y is all zero.
		 *                         The output \a similarity is undefined.
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
			typename OutputType,
			typename InputType1,
			typename InputType2,
			class Ring,
			class Division = grb::operators::divide<
				typename Ring::D3, typename Ring::D3, typename Ring::D4
			>
		>
		RC cosine_similarity(
			OutputType &similarity,
			const Vector< InputType1 > &x, const Vector< InputType2 > &y,
			const Ring &ring = Ring(), const Division &div = Division()
		) {
			static_assert( std::is_floating_point< OutputType >::value,
				"Cosine similarity requires a floating-point output type." );

			// static sanity checks
			NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
					std::is_same< InputType1, typename Ring::D1 >::value
				), "grb::algorithms::cosine_similarity",
				"called with a left-hand vector value type that does not match the "
				"first domain of the given semiring" );
			NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
					std::is_same< InputType2, typename Ring::D2 >::value
				), "grb::algorithms::cosine_similarity",
				"called with a right-hand vector value type that does not match "
				"the second domain of the given semiring" );
			NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
					std::is_same< OutputType, typename Ring::D4 >::value
				), "grb::algorithms::cosine_similarity",
				"called with an output vector value type that does not match the "
				"fourth domain of the given semiring" );
			NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
					std::is_same< typename Ring::D3, typename Ring::D4 >::value
				), "grb::algorithms::cosine_similarity",
				"called with a semiring that has unequal additive input domains" );

			const size_t n = size( x );

			// run-time sanity checks
			if( n != size( y ) ) {
				return MISMATCH;
			}

			// check whether inputs are dense
			const bool dense = nnz( x ) == n && nnz( y ) == n;

			// set return code
			RC rc = SUCCESS;

			// compute-- choose method depending on we can stream once or need to stream
			// multiple times
			OutputType nominator, denominator;
			nominator = denominator = ring.template getZero< OutputType >();
			if( dense && grb::Properties<>::writableCaptured ) {
				// lambda works, so we can stream each vector precisely once:
				OutputType norm1, norm2;
				norm1 = norm2 = ring.template getZero< OutputType >();
				rc = grb::eWiseLambda(
					[ &x, &y, &nominator, &norm1, &norm2, &ring ]( const size_t i ) {
						const auto &mul = ring.getMultiplicativeOperator();
						const auto &add = ring.getAdditiveOperator();
						OutputType temp;
						(void) grb::apply( temp, x[ i ], y[ i ], mul );
						(void) grb::foldl( nominator, temp, add );
						(void) grb::apply( temp, x[ i ], x[ i ], mul );
						(void) grb::foldl( norm1, temp, add );
						(void) grb::apply( temp, y[ i ], y[ i ], mul );
						(void) grb::foldl( norm2, temp, add );
					}, x, y
				);
				denominator = sqrt( norm1 ) * sqrt( norm2 );
			} else {
				// cannot stream each vector once, stream each one twice instead using
				// standard grb functions
				rc = grb::norm2( nominator, x, ring );
				if( rc == SUCCESS ) {
					rc = grb::norm2( denominator, y, ring );
				}
				if( rc == SUCCESS ) {
					rc = grb::foldl( denominator, nominator,
						ring.getMultiplicativeOperator() );
				}
				if( rc == SUCCESS ) {
					rc = grb::dot( nominator, x, y, ring );
				}
			}

			// accumulate
			if( rc == SUCCESS ) {
				// catch zeroes
				if( denominator == ring.template getZero() ) {
					return ILLEGAL;
				}
				if( nominator == ring.template getZero() ) {
					return ILLEGAL;
				}
				rc = grb::apply( similarity, nominator, denominator, div );
			}

			// done
			return rc;
		}

	} // end namespace algorithms

} // end namespace grb

#undef NO_CAST_ASSERT

#endif // end _H_GRB_COSSIM

