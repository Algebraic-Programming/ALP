
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
 * Implements the 2-norm.
 *
 * @author A. N. Yzelman
 * @date 17th of March 2022
 *
 * \internal
 * Factored out of graphblas/blas1.hpp, promoted to a (simple) algorithm since
 * semiring structures are insufficient to capture <tt>sqrt</tt>.
 *
 * \todo Provide implementations of other norms.
 * \endinternal
 */

#ifndef _H_GRB_ALGORITHMS_NORM
#define _H_GRB_ALGORITHMS_NORM

#include <graphblas.hpp>

#include <cmath> // for std::sqrt


namespace grb {

	namespace algorithms {

		/**
		 * A wrapper type for a partial sum of squared norms.
		 *
		 * This type is used as D1=D2=D3 of a monoid so that a single \a foldl call
		 * can compute the sum of squared norms directly from a vector, without
		 * allocating a temporary vector. The two distinct constructors — one from
		 * \a InputType (which computes the squared norm) and one that copies an
		 * existing partial sum (no re-squaring) — allow the framework's foldl
		 * machinery to distinguish the accumulation step from the combining step:
		 *
		 *  - Accumulation: <tt>static_cast<NormPartial>(y[i])</tt> calls
		 *    <tt>NormPartial(InputType)</tt>, which computes |y[i]|^2.
		 *  - Combining:    <tt>static_cast<NormPartial>(local_partial)</tt> calls
		 *    the copy constructor, which simply copies the value (no re-squaring).
		 *
		 * @tparam OutputType The real-valued output type of the norm computation.
		 * @tparam InputType  The type of the vector elements (real or complex).
		 */
		template< typename OutputType, typename InputType >
		struct NormPartial {

			/** The accumulated sum of squared norms. */
			OutputType value;

			/** Default constructor: initialises to zero (the additive identity). */
			NormPartial() noexcept : value( OutputType{} ) {}

			/** Copy constructor: copies the partial sum without re-squaring. */
			NormPartial( const NormPartial & ) = default;

			/** Copy assignment. */
			NormPartial & operator=( const NormPartial & ) = default;

			/**
			 * Converting constructor from a vector element: computes |x|^2.
			 *
			 * This is intentionally separate from the copy constructor so that the
			 * framework's cast <tt>static_cast<D2>(vectorElement)</tt> computes the
			 * squared norm, while <tt>static_cast<D3>(partialSum)</tt> (where the
			 * argument is already a NormPartial) uses the copy constructor and does
			 * not re-square.
			 */
			NormPartial( const InputType &x ) :
				value( static_cast< OutputType >(
					grb::utils::is_complex< InputType >::norm( x )
				) ) {}

			/** Element-wise addition (required by operators::add). */
			NormPartial operator+( const NormPartial &other ) const noexcept {
				NormPartial result;
				result.value = value + other.value;
				return result;
			}

			/** In-place addition (required by operators::add for foldl/foldr). */
			NormPartial & operator+=( const NormPartial &other ) noexcept {
				value += other.value;
				return *this;
			}

		};

	} // namespace algorithms

} // namespace grb


// Specialise grb::identities::zero for NormPartial so that it compiles for
// complex InputType (the default zero<T> requires is_convertible<int,T> which
// fails when there are two chained user-defined conversions int->InputType->NP).
namespace grb {
	namespace identities {

		template< typename OutputType, typename InputType >
		class zero< grb::algorithms::NormPartial< OutputType, InputType > > {
			public:
				static grb::algorithms::NormPartial< OutputType, InputType > value() {
					return grb::algorithms::NormPartial< OutputType, InputType >();
				}
		};

	} // namespace identities
} // namespace grb


namespace grb {

	namespace algorithms {

		/**
		 * An alias of std::sqrt where the input and output types are templated
		 * separately.
		 *
		 * @tparam OutputType The output type of the square-root operation.
		 * @tparam InputType The input type of the square-root operation.
		 *
		 * @param[in] x The value to take the square root of.
		 *
		 * @returns The square root of \a x, cast to \a OutputType.
		 *
		 * Relies on the standard std::sqrt implementation. If this is not available
		 * for \a InputType, the use of this operation will result in a compile-time
		 * error.
		 *
		 * This operation is used as a default to the #norm2 algorithm, as well as a
		 * default to algorithms that depend on it.
		 */
		template< typename OutputType, typename InputType >
		OutputType std_sqrt( const InputType x ) {
			return( static_cast< OutputType >( std::sqrt( x ) ) );
		};

		/**
		 * Provides a generic implementation of the 2-norm computation.
		 *
		 * Proceeds by computing a dot-product on itself and then taking the square
		 * root of the result.
		 *
		 * This function is only available when the output type is floating point.
		 *
		 * For return codes, exception behaviour, performance semantics, template
		 * and non-template arguments, @see grb::dot.
		 *
		 * @param[in,out] x On successful execution of this algorithm, on output,
		 *                  the 2-norm of \a y will have been added to \a x.
		 * @param[in]   y   The vector to compute the norm of.
		 * @param[in] ring  The Semiring under which the 2-norm is to be computed.
		 * @param[in] sqrtX The square root function which operates on real data
		 *                  type, as both input and output. If not explicitly
		 *                  provided, the std::sqrt() is used.
		 */
		template<
			Descriptor descr = descriptors::no_operation, class Ring,
			typename InputType, typename OutputType,
			Backend backend, typename Coords
		>
		RC norm2(
			OutputType &x,
			const Vector< InputType, backend, Coords > &y,
			const Ring &ring = Ring(),
			const std::function< OutputType( OutputType ) > sqrtX =
				[]( const OutputType val ) -> OutputType {
					return static_cast< OutputType >( std::sqrt( val ) );
				},
			const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< InputType >::value &&
				grb::is_semiring< Ring >::value &&
				std::is_floating_point< OutputType >::value,
			void >::type * = nullptr
		) {
			// ring is not used in the computation; accepted for API compatibility.
			(void)ring;

			// NormPartial<OutputType, InputType> is a wrapper whose constructor from
			// InputType computes |x|^2, while its copy constructor copies without
			// re-squaring.  Using this as D1=D2=D3 of a monoid allows a single foldl
			// to compute sum(|y[i]|^2) in one pass without a temporary vector.
			typedef NormPartial< OutputType, InputType > NP;

			// Build the addition monoid on NormPartial.
			// The identities::zero specialisation above provides the zero element.
			const Monoid< operators::add< NP >, identities::zero > normMonoid;

			// Reduction: for each y[i], static_cast<NP>(y[i]) calls NP(InputType)
			// which computes |y[i]|^2.  The combining step casts NP->NP via the copy
			// constructor (no re-squaring).  This is handled inside foldl safely and
			// in parallel in all backends.
			NP yyt;
			RC ret = foldl< descr >( yyt, y, normMonoid );

			// Take square root and accumulate into output.
			if( ret == SUCCESS ) {
				const OutputType sqrtYyt = sqrtX( yyt.value );
				ret = foldl( x, sqrtYyt, operators::add< OutputType >() );
			}
			return ret;
		}
	}
}

#endif // end ``_H_GRB_ALGORITHMS_NORM''

