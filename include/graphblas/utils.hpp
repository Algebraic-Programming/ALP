
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
 * @date 8th of August, 2016
 */

#ifndef _H_GRB_UTILITIES
#define _H_GRB_UTILITIES

#include <cmath>  //fabs
#include <limits> //numeric_limits
#include <type_traits>

#include <graphblas/descriptors.hpp>

namespace grb {

	/**
	 * Some utility classes used that may be used throughout this GraphBLAS
	 * implementation.
	 *
	 * Utilities that rely on external libraries or system calls should \em not be
	 * added here-- those should reside in their own compilation units so that
	 * backends can decide on an individual basis whether or not to include them.
	 * This is especially useful when writing a backend for an architecture without
	 * extensive coverage of standard extensions or libraries.
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
		static bool equals( const T & a, const T & b, typename std::enable_if< ! std::is_floating_point< T >::value >::type * = NULL ) {
			// simply do standard compare
			return a == b;
		}

		/**
		 * Checks whether two floating point values are equal.
		 *
		 * This function takes into account round-off errors due to machine precision.
		 *
		 * \warning This does not take into account accumulated numerical errors
		 *          due to previous operations on the given values.
		 *
		 * @\tparam T The numerical type.
		 *
		 * @param a One of the two values to comare against.
		 * @param b One of the two values to comare against.
		 * @param epsilons How many floating point errors may have accumulated.
		 *                 Should be chosen larger or equal to one.
		 *
		 * @returns Whether a == b.
		 */
		template< typename T, typename U >
		static bool equals( const T & a, const T & b, const U epsilons, typename std::enable_if< std::is_floating_point< T >::value >::type * = NULL

		) {
			assert( epsilons >= 1 );

			// if they are bit-wise equal, it's easy
			if( a == b ) {
#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
				std::cout << "\t Bit-wise equal\n";
#else
				printf( "\t Bit-wise equal\n" );
#endif
#endif
				return true;
			}

			// if not, we need to look at the absolute values
			const T absA = fabs( a );
			const T absB = fabs( b );
			const T absDiff = fabs( a - b );
			const T absPlus = absA + absB;

			// find the effective epsilon
			const T eps = static_cast< T >( epsilons ) * std::numeric_limits< T >::epsilon();

			// find the minimum and maximum *normal* values.
			const T min = std::numeric_limits< T >::min();
			const T max = std::numeric_limits< T >::max();

			// if the difference is a subnormal number, it should be smaller than machine epsilon times min;
			// if this is not the case, then we cannot safely conclude anything from this small a difference.
			// The same is true if a or b are zero.
			if( a == 0 || b == 0 || absPlus < min ) {
#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
				std::cout << "\t Zero or close to zero difference\n";
#else
				printf( "\t Zero or close to zero difference\n" );
#endif
#endif
				return absDiff < eps * min;
			}

			// we wish to normalise absDiff by (absA + absB),
			// However, absA + absB might overflow.
			if( absA > absB ) {
				if( absB > max - absA ) {
#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
					std::cout << "\t Normalising absolute difference by max "
								 "(I)\n";
#else
					printf( "\t Normalising absolute difference by max (I)\n" );
#endif
#endif
					return absDiff / max < eps;
				}
			} else {
				if( absA > max - absB ) {
#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
					std::cout << "\t Normalising absolute difference by max "
								 "(II)\n";
#else
					printf( "\t Normalising absolute difference by max "
							"(II)\n" );
#endif
#endif
					return absDiff / max < eps;
				}
			}
			// use of relative error should be safe
#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
			std::cout << "\t Using relative error\n";
#else
			printf( "\t Using relative error\n" );
#endif
#endif
			return absDiff / absPlus < eps;
		}

		/**
		 * A templated max function that is different from std::max in that the
		 * return value is a constexpr. (This was fixed in C++14.)
		 */
		template< typename T >
		constexpr const T & static_max( const T & a, const T & b ) {
			return a > b ? a : b;
		}

		/**
		 * A templated min function that is different from std::min in that the
		 * return value is a constexpr. (This was fixed in C++14.)
		 */
		template< typename T >
		constexpr const T & static_min( const T & a, const T & b ) {
			return a < b ? a : b;
		}

		/**
		 * A sizeof that is safe w.r.t. void types.
		 *
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

		// void-specialisation of the above class
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
		static bool interpretMask( const bool & assigned, const T * const val, const size_t offset ) {
			// set default mask to false
			bool ret = false;
			// if we request a structural mask, decide only on passed assigned variable
			if( descriptor & descriptors::structural ) {
				ret = assigned;
			} else {
				// if based on value, if there is a value, cast it to bool
				if( assigned ) {
					ret = static_cast< bool >( val[ offset ] );
				}
				// otherwise there is no value and false is assumed
			}
			// check whether we should return the inverted value
			if( descriptor & descriptors::invert_mask ) {
				return ! ret;
			} else {
				return ret;
			}
		}

		/** Specialisation for void-valued masks */
		template< Descriptor descriptor >
		static bool interpretMask( const bool & assigned, const void * const, const size_t ) {
			// set default mask to false
			bool ret = assigned;
			// check whether we should return the inverted value
			if( descriptor & descriptors::invert_mask ) {
				return ! ret;
			} else {
				return ret;
			}
		}

	} // namespace utils

} // namespace grb

#endif
