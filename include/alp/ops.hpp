
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

#ifndef _H_ALP_OPERATORS
#define _H_ALP_OPERATORS

#include "internalops.hpp"
#include "type_traits.hpp"
#include <graphblas/ops.hpp>

namespace alp {

	/**
	 * This namespace holds various standard operators such as #alp::operators::add
	 * and #alp::operators::mul.
	 */
	namespace operators {

		/** @see grb::operators::left_assign */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using left_assign = grb::operators::left_assign< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::left_assign_if */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using left_assign_if = grb::operators::left_assign_if< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::right_assign */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using right_assign = grb::operators::right_assign< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::right_assign_if */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using right_assign_if = grb::operators::right_assign_if< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::add */
		// [Operator Wrapping]
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using add = grb::operators::add< D1, D2, D3, grb::Backend::reference >;
		// [Operator Wrapping]

		/** @see grb::operators::mul */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using mul = grb::operators::mul< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::max */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using max = grb::operators::max< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::min */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using min = grb::operators::min< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::subtract */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using subtract = grb::operators::subtract< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::divide */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using divide = grb::operators::divide< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::divide_reverse */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using divide_reverse = grb::operators::divide_reverse< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::equal */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using equal = grb::operators::equal< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::not_equal */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using not_equal = grb::operators::not_equal< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::any_or */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using any_or = grb::operators::any_or< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::logical_or */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using logical_or = grb::operators::logical_or< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::logical_and */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using logical_and = grb::operators::logical_and< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::relu */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using relu = grb::operators::relu< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::abs_diff */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using abs_diff = grb::operators::abs_diff< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::argmin */
		template< typename IType, typename VType >
		using argmin = grb::operators::argmin< IType, VType >;

		/** @see grb::operators::argmax */
		template< typename IType, typename VType >
		using argmax = grb::operators::argmax< IType, VType >;

		/** @see grb::operators::square_diff */
		template< typename D1, typename D2, typename D3, enum Backend implementation = config::default_backend >
		using square_diff = grb::operators::square_diff< D1, D2, D3, grb::Backend::reference >;

		/** @see grb::operators::zip */
		template< typename IN1, typename IN2, enum Backend implementation = config::default_backend >
		using zip = grb::operators::zip< IN1, IN2, grb::Backend::reference >;

		/** @see grb::operators::equal_first */
		template< typename D1, typename D2 = D1, typename D3 = D2, enum Backend implementation = config::default_backend >
		using equal_first = grb::operators::equal_first< D1, D2, D3, grb::Backend::reference >;

	} // namespace operators

	template< typename OP >
	struct is_associative {
		static constexpr const bool value = is_operator< OP >::value && OP::is_associative();
	};

	template< typename OP >
	struct is_commutative {
		static constexpr const bool value = is_operator< OP >::value && OP::is_commutative();
	};

} // namespace alp

#ifdef __DOXYGEN__
 /**
  * Macro that disables the definition of an operator<< overload for
  * instances of std::pair. This overload is only active when the _DEBUG
  * macro is defined, but may clash with user-defined overloads.
  */
 #define _DEBUG_NO_IOSTREAM_PAIR_CONVERTER
#endif
#ifdef _DEBUG
 #ifndef _DEBUG_NO_IOSTREAM_PAIR_CONVERTER
	template< typename U, typename V >
	std::ostream & operator<<( std::ostream & out, const std::pair< U, V > & pair ) {
		out << "( " << pair.first << ", " << pair.second << " )";
		return out;
	}
 #endif
#endif

#endif // end ``_H_ALP_OPERATORS''

