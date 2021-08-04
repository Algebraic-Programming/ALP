
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
 * @date 5th of April, 2017
 */

#ifndef _H_GRB_UTILS_RANGES
#define _H_GRB_UTILS_RANGES

#include <type_traits>

namespace grb {

	namespace utils {

		/**
		 * Checks whether a given value is larger or equal to zero.
		 *
		 * @tparam   T   The type of the value to check.
		 * @tparam guard Guard pointer to a value of type \a void when referring to
		 *               this default implementation of this function.
		 *
		 * This implementation will use the standard operator
		 * \code
		 * >=
		 * \endcode
		 *
		 * \warning This operator should be defined on the type \a T or the code will
		 *          not compile.
		 *
		 * @param[in] x The value to check whether it is larger or equal to zero.
		 *
		 * @return Whether \f$ x \geq 0 \f$ returns \a true.
		 */
		template< typename T, typename std::enable_if< ! std::is_unsigned< T >::value, T >::type guard = 0 >
		bool is_geq( const T x ) {
			if( x >= 0 ) {
				return true;
			} else {
				return false;
			}
		}

		/**
		 * Checks whether a given unsigned integral value is larger or equal to zero.
		 *
		 * Always returns true.
		 *
		 * @tparam T The unsigned type to check the value of.
		 *
		 * This function is only enabled for types \a T which are integral and
		 * unsigned.
		 *
		 * @param[in] x The value to check whether it is larger or equal to zero.
		 *
		 * @return This function always returns \a true.
		 */
		template< typename T, typename std::enable_if< std::is_unsigned< T >::value, T >::type = 0 >
		bool is_geq_zero( const T ) {
			return true;
		}

		/**
		 * Checks whether a given value \a x is inside a given range.
		 *
		 * @tparam T The type of \a x.
		 *
		 * @param[in] x The value to check for whether it is inside a given range.
		 *
		 * @param[in] inclusive_lower_bound The inclusive lower bound of the range.
		 * @param[in] exclusive_upper_bound The exclusive upper bound of the range.
		 *
		 * @return Whether \f$ x \geq \mathit{inclusive\_lower\_bound} \land
		 *          x < \mathit{exclusive\_upper\_bound} \f$ returns \a true.
		 *
		 * @see is_in_normalized_range This function normalises the given range and
		 *                             then delegates to this function.
		 */
		template< typename T >
		bool is_in_range( const T x, const T inclusive_lower_bound, const T exclusive_upper_bound ) {
			return is_in_normalized_range( x - inclusive_lower_bound, exclusive_upper_bound - inclusive_lower_bound );
		}

		/**
		 * Checks whether a given value is inside the given normalised range.
		 *
		 * @tparam   T   The type of the value to check.
		 * @tparam guard Guard pointer to a value of type \a void when referring
		 *               to this default implementation of this function.
		 *
		 * This implementation will use the standard operators
		 * \code
		 * >=, <
		 * \endcode
		 *
		 * \warning These operator should be defined on the type \a T or the code
		 *          will not compile.
		 *
		 * @param[in] x     The value to check whether it is inside the given range.
		 * @param[in] exclusive_upper_bound The value \a x may be less than of.
		 *
		 * @return Whether
		 *           \f$ x \geq 0 \land x < \mathit{exclusive\_upper\_bound} \f$
		 *         returns \a true.
		 */
		template< typename T, typename std::enable_if< ! std::is_unsigned< T >::value, T >::type guard = 0 >
		bool is_in_normalized_range( const T x, const T exclusive_upper_bound ) {
			if( x >= 0 && x < exclusive_upper_bound ) {
				return true;
			} else {
				return false;
			}
		}

		/**
		 * Checks whether a given unsigned integral value is in the given normalised
		 * range, i.e., whether \a x is strictly less than \a exclusive_upper_bound.
		 *
		 * @tparam T The unsigned type to check the value of.
		 *
		 * This function is only enabled for types \a T which are integral and
		 * unsigned.
		 *
		 * @param[in] x The value to check the value of.
		 * @param[in] exclusive_upper_bound The upper bound \a x may be less than.
		 *
		 * @return Whether \f$ x < \mathit{exclusive\_upper\_bound} \f$ is \a true.
		 */
		template< typename T, typename std::enable_if< std::is_unsigned< T >::value, T >::type = 0 >
		bool is_in_normalized_range( const T x, const T exclusive_upper_bound ) {
			return x < exclusive_upper_bound;
		}

	} // namespace utils

} // namespace grb

#endif
