
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
 * Implements some utilities that may be useful outside of the ALP context also.
 *
 * @author A. N. Yzelman
 * @date 8th of August, 2016
 */

#ifndef _H_GRB_UTILITIES
#define _H_GRB_UTILITIES

#include <assert.h>

#include <cmath>  //fabs
#include <limits> //numeric_limits
#include <type_traits>

#include <graphblas/descriptors.hpp>
#include <graphblas/utils/iscomplex.hpp>


namespace grb {

	/**
	 * Utility functions and classes used throughout the ALP implementation that
	 * could be useful outside of ALP also.
	 *
	 * \internal
	 * Utilities that rely on external libraries or system calls should \em not be
	 * added here-- those should reside in their own compilation units so that
	 * backends can decide on an individual basis whether or not to include them.
	 *
	 * This is especially important considering backends for architectures without
	 * extensive coverage of standard extensions or libraries.
	 * \endinternal
	 */
	namespace utils {

		/**
		 * Checks whether two values are equal.
		 *
		 * This function simply performs a bit-wise comparison on native \em integral
		 * data types, or this function assumes a properly overloaded == operator.
		 *
		 * @\tparam T The numerical type.
		 *
		 * @param a One of the two values to comare against.
		 * @param b One of the two values to comare against.
		 * @returns Whether a == b.
		 */
		template< typename T >
		static bool equals(
			const T &a, const T &b,
			typename std::enable_if<
				!std::is_floating_point< T >::value
			>::type * = nullptr
		) {
			// simply do standard compare
			return a == b;
		}

		/**
		 * Checks whether two floating point values, \a a and \a b, are equal.
		 *
		 * The comparison is relative in the following sense.
		 *
		 * It is mandatory to supply an integer, \a epsilons, that indicates a
		 * relative error. Here, \a epsilon represents the number of errors
		 * accumulated during their computation, assuming that all arithmetic that
		 * produced the two floating point arguments to this function are bounded by
		 * the magnitudes of \a a and \a b.
		 *
		 * Intuitively, one may take \a epsilons as the sum of the number of
		 * operations that produced \a a and \a b, if the above assumption holds.
		 *
		 * The resulting bound can be tightened if the magnitudes encountered during
		 * their computation are much smaller, and are (likely) too tight if those
		 * magnitudes were much larger instead. In such cases, one should obtain or
		 * compute a more appropriate error bound, and check whether the difference
		 * is within such a more appropriate, absolute, error bound.
		 *
		 * \warning When comparing for equality within a known absolute error bound,
		 *          <b>do not this function</b>.
		 *
		 * \note If such a function is desired, please submit an issue on GitHub or
		 *       Gitee. The absolute error bound can be provided again via a
		 *       parameter \a epsilons, but then of a floating-point type instead of
		 *       the current integer variant.
		 *
		 * @tparam T The numerical type.
		 * @tparam U The integer type used for \a epsilons.
		 *
		 * @param[in] a        One of the two values to compare.
		 * @param[in] b        One of the two values to compare.
		 * @param[in] epsilons How many floating point errors may have accumulated;
		 *                     must be chosen larger or equal to one.
		 *
		 * This function automatically adapts to the floating-point type used, and
		 * takes into account the border cases where one or more of \a a and \a b may
		 * be zero or subnormal. It also guards against overflow of the normalisation
		 * strategy employed in its implementation.
		 *
		 * If one of \a a or \a b is zero, then an absolute acceptable error bound is
		 * computed as \a epsilons times \f$ \epsilon \f$, where the latter is the
		 * machine precision.
		 *
		 * \note This behaviour is consistent with the intuitive usage of this
		 *       function as described above, assuming that the computations that led
		 *       to zero involved numbers of zero orders of magnitude.
		 *
		 * \warning Typically it is hence \em not correct to rely on this function to
		 *          compare some value \a a to zero by passing a zero \a b explicitly,
		 *          and vice versa-- the magnitude of the values used while computing
		 *          the nonzero matter.
		 *
		 * \note If a version of this function with an explicit relative error bound
		 *       is desired, please submit an issue on GitHub or Gitee.
		 *
		 * @returns Whether \a a equals \a b, taking into account numerical drift
		 *          within the effective range derived from \a epsilons and the
		 *          magnitudes of \a a and \a b.
		 *
		 * This utility function is chiefly used by the ALP/GraphBLAS unit, smoke, and
		 * performance tests.
		 */
		template< typename T, typename U >
		static bool equals(
			const T &a, const T &b,
			const U epsilons,
			typename std::enable_if<
				std::is_floating_point< T >::value &&
				std::is_integral< U >::value
			>::type * = nullptr
		) {
			assert( epsilons >= 1 );
#ifdef _DEBUG
			std::cout << "Comparing " << a << " to " << b << " with relative tolerance "
				<< "of " << epsilons << "\n";
#endif

			// if they are bit-wise equal, it's easy
			if( a == b ) {
#ifdef _DEBUG
				std::cout << "\t Bit-wise equal\n";
#endif
				return true;
			}

			// find the effective epsilon and absolute difference
			const T eps = static_cast< T >( epsilons ) *
				std::numeric_limits< T >::epsilon();
			const T absDiff = fabs( a - b );

			// If either a or b is zero, then we use the relative epsilons as an absolute
			// error on the nonzero argument
			if( a == 0 || b == 0 ) {
				assert( a != 0 || b != 0 );
#ifdef _DEBUG
				std::cout << "\t One of the arguments is zero\n";
#endif
				return absDiff < eps;
			}


			// if not, we need to look at the absolute values
			const T absA = fabs( a );
			const T absB = fabs( b );
			const T absPlus = absA + absB;

			// find the minimum and maximum *normal* values.
			const T min = std::numeric_limits< T >::min();

			// If the combined magnitudes of a or b are subnormal, then scale by the
			// smallest normal rather than the combined magnitude
			if( absPlus < min ) {
#ifdef _DEBUG
				std::cout << "\t Subnormal comparison requested. Scaling the tolerance by "
					<< "the smallest normal number rather than to the subnormal absolute "
					<< "difference\n";
#endif
				return absDiff / min < eps;
			}

			// we wish to normalise absDiff by (absA + absB), however, absA + absB might
			// overflow. If it does, we normalise by the smallest of absA and absB
			// instead of attempting to average.
			const T max = std::numeric_limits< T >::max();
			if( absA > absB ) {
				if( absB > max - absA ) {
#ifdef _DEBUG
					std::cout << "\t Normalising absolute difference by smallest magnitude "
						<< "(I)\n";
#endif
					return absDiff / absB < eps;
				}
			} else {
				if( absA > max - absB ) {
#ifdef _DEBUG
					std::cout << "\t Normalising absolute difference by smallest magnitude "
						<< "(II)\n";
#endif
					return absDiff / absA < eps;
				}
			}

			// at this point, the use of relative error vs. 0.5*absPlus should be safe
#ifdef _DEBUG
			std::cout << "\t Using relative error\n";
#endif
			return (static_cast< T >(2) * (absDiff / absPlus)) < eps;
		}

		/**
		 * A templated max function that is different from std::max in that the
		 * return value is a constexpr. (This was fixed in C++14.)
		 */
		template< typename T >
		constexpr const T & static_max( const T &a, const T &b ) {
			return a > b ? a : b;
		}

		/**
		 * A templated min function that is different from std::min in that the
		 * return value is a constexpr. (This was fixed in C++14.)
		 */
		template< typename T >
		constexpr const T & static_min( const T &a, const T &b ) {
			return a < b ? a : b;
		}

		/**
		 * A sizeof that is safe with respect to <tt>void</tt> types.
		 *
		 * The byte size of <tt>void</tt> types returns zero. The byte size of any
		 * other type <tt>T</tt> returns <tt>sizeof(T)</tt>.
		 */
		template< typename T >
		class SizeOf {

			public:

				/**
				 * If \a T is <tt>void</tt>, this value equals 0 and
				 * equal to <tt>sizeof(T)</tt> otherwise.
				 */
				static constexpr const size_t value = sizeof( T );

		};

		/** \internal void-specialisation of the above class */
		template<>
		class SizeOf< void > {

			public:

				static constexpr const size_t value = 0;

		};

		/**
		 * Given a (combination) of descriptors, evaluates a mask at a given
		 * position.
		 *
		 * @tparam descriptor The given descriptor.
		 * @tparam T          The type of an element of the mask.
		 *
		 * @param[in] assigned Whether there was an element in the mask.
		 * @param[in] val      Pointer to memory area where mask elements reside.
		 * @param[in] offset   Offset in \a val to the mask value to be interpreted.
		 *
		 * The memory area pointed to by \a val shall not be dereferenced if
		 * \a assigned is false.
		 *
		 * @return If the descriptor includes grb::descriptors::structoral,
		 *         returns \a assigned. If additionally grb::descriptors::invert_mask
		 *         was defined, instead returns the negation of \a assigned.
		 * @return If the descriptor includes grb::descriptors::structural_complement,
		 *         returns the negation of \a assigned. If additionally
		 *         grb::descriptors::invert_mask was defined, instead returns
		 *         \a assigned.
		 * @return If the descriptor does not include grb::descriptors::structural
		 *         nor grb::descriptors::structural_complement and if \a assigned
		 *         is false, then the entry is ignored and uninterpreted, thus
		 *         returning \a false.
		 * @return If the descriptor includes grb::descriptors::invert_mask,
		 *         returns the negation of the dereferenced value of \a val
		 *         which is first cast to a \a bool.
		 * @return Otherwise, returns the dereferenced value of \a val,
		 *         cast to a \a bool.
		 *
		 * If \a descriptor contains both grb::descriptors::structural and
		 * grb::descriptors::structural_complement, the code shall not
		 * compile.
		 */
		template< Descriptor descriptor, typename T >
		static bool interpretMask(
			const bool &assigned,
			const T * const val,
			const size_t offset
		) {
			// set default mask to false
			bool ret = false;
			// if we request a structural mask, decide only on passed assigned variable
			if( descriptor & descriptors::structural ) {
				ret = assigned;
			} else {
				// if based on value, if there is a value, cast it to bool
				if( !assigned ) {
					return false;
				}
				else {
					ret = static_cast< bool >( val[ offset ] );
				}
				// otherwise there is no value and false is assumed
			}
			// check whether we should return the inverted value
			if( descriptor & descriptors::invert_mask ) {
				return !ret;
			} else {
				return ret;
			}
		}

		/** Specialisation for complex-valued masks */
		template< Descriptor descriptor, typename T >
		static bool interpretMask(
				const bool &assigned,
				const std::complex<T> * const val,
				const size_t offset
		) {
			// set default mask to false
			bool ret = false;
			// if we request a structural mask, decide only on passed assigned variable
			if( descriptor & descriptors::structural ) {
				ret = assigned;
			} else {
				// if based on value, if there is a value, cast it to bool
				if( !assigned ) {
					return false;
				}
				else {
					ret = static_cast< bool >( real( val [ offset ] ) ) ||
					       static_cast< bool >( imag( val [ offset ] ) );
				}
				// otherwise there is no value and false is assumed
			}
			// check whether we should return the inverted value
			if( descriptor & descriptors::invert_mask ) {
				return !ret;
			} else {
				return ret;
			}
		}

		/** Specialisation for void-valued masks */
		template< Descriptor descriptor >
		static bool interpretMask(
			const bool &assigned,
			const void * const,
			const size_t
		) {
			// set default mask to false
			bool ret = assigned;
			// check whether we should return the inverted value
			if( ( descriptor & descriptors::structural_complement ) == descriptors::structural_complement ) {
				return !ret;
			} else {
				return ret;
			}
		}

		/** Specialisation for void-valued matrice's masks */
		template< Descriptor descriptor, typename MatrixDataType, typename ValuesType >
		static bool interpretMatrixMask(
			const bool &assigned,
			const ValuesType * const values,
			const size_t k,
			typename std::enable_if<
				!std::is_void< MatrixDataType >::value
			>::type * = nullptr
		) {
			return interpretMask< descriptor, ValuesType >( assigned, values, k );
		}

		/** Specialisation for void-valued matrice's masks */
		template< Descriptor descriptor, typename MatrixDataType, typename ValuesType >
		static bool interpretMatrixMask(
			const bool &assigned,
			const ValuesType * const,
			const size_t,
			typename std::enable_if<
				std::is_void< MatrixDataType >::value
			>::type * = nullptr
		) {
			// set default mask to false
			bool ret = assigned;
			// check whether we should return the inverted value
			if( ( descriptor & descriptors::structural_complement ) == descriptors::structural_complement ) {
				return !ret;
			} else {
				return ret;
			}
		}

	} // namespace utils

} // namespace grb

#endif

