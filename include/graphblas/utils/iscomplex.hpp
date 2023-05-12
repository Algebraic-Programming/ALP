
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
 * @date 12th of April, 2022
 */

#ifndef _H_GRB_UTILS_ISCOMPLEX
#define _H_GRB_UTILS_ISCOMPLEX

#include <complex>

namespace grb {

	namespace utils {

		/**
		 * A template class used for inspecting whether a given class is of type
		 * std::complex.
		 *
		 * @tparam C The type to check.
		 *
		 * This class also mirrors a couple of standard functions that translate to
		 * no-ops in case \a C is not complex. The following functions are provided:
		 *  - is_complex::conjugate
		 *  - is_complex::modulus
		 *
		 * \internal This is the base implementation which assumes \a C is not of
		 * type std::complex.
		 */
		template< typename C >
		class is_complex {

			static_assert(
				std::is_floating_point< C >::value,
				"is_complex: C is not a floating point type"
			);

			public:

				/**
				 * If \a value is <tt>false</tt>, the type will be \a C.
				 * If \a value is <tt>true</tt>, the type will be C::value_type.
				 */
				typedef C type;

				/** Whether the type \a C is std::complex */
				static constexpr const bool value = false;

				/**
				 * @returns The conjugate of a given value if \a C is a complex type, or
				 *          the given value if \a C is not complex.
				 */
				static C conjugate( const C &x ) noexcept {
					return x;
				}

				/**
				 * @returns The absolute value of a given value if \a C is a complex type,
				 *          or the given value if \a C is not complex.
				 */
				static C modulus( const C &x ) noexcept {
					return( x > 0 ? x : -x );
				}

				/**
				 * @returns The absolute value squared of a given value.
				 */
				static C norm( const C &x ) noexcept {
					return x * x;
				}

		};

		/** \internal The specialisation for std::complex types. */
		template< typename T >
		class is_complex< std::complex< T > >
		{

			static_assert(
				std::is_floating_point< T >::value,
				"is_complex: T is not a floating point type"
			);

			public:
				typedef T type;
				static constexpr const bool value = true;
				static std::complex< T > conjugate( const std::complex< T > &x ) {
					return std::conj( x );
				}
				static T modulus( const std::complex< T > &x ) {
					return std::abs( x );
				}
				static T norm( const std::complex< T > &x ) {
					return std::norm( x );
				}

		};


	} // end namespace utils

} // end namespace grb

#endif // end _H_GRB_UTILS_ISCOMPLEX

