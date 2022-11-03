
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

#ifndef _H_ALP_INTERNAL_OPERATORS_BASE
#define _H_ALP_INTERNAL_OPERATORS_BASE

#include <alp/utils/suppressions.h>

#include <type_traits>
#include <utility>

#include <alp/type_traits.hpp>
#include <alp/utils.hpp>

#include <graphblas/internalops.hpp>

#include "config.hpp"


namespace alp {

	namespace operators {

		/**
		 * Core implementations of the standard operators in #alp::operators.
		 * All ALP operators use the GraphBLAS implementation of the
		 * corresponding types specialized for grb::reference backend.
		 * For documentation, see corresponding GraphBLAS documentation.
		 */
		namespace internal {

			/**
			 * @see grb::operators::internal::argmin
			 */
			template< typename IType, typename VType >
			using argmin = grb::operators::internal::argmin< IType, VType >;

			/**
			 * @see grb::operators::internal::argmax
			 */
			template< typename IType, typename VType >
			using argmax = grb::operators::internal::argmax< IType, VType >;

			/**
			 * @see grb::operators::internal::left_assign
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using left_assign = grb::operators::internal::left_assign< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::right_assign
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using right_assign = grb::operators::internal::right_assign< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::left_assign_if
			 */
			template< typename D1, typename D2, typename D3, enum Backend implementation = config::default_backend >
			using left_assign_if = grb::operators::internal::left_assign_if< D1, D2, D3, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::right_assign_if
			 */
			template< typename D1, typename D2, typename D3, enum Backend implementation = config::default_backend >
			using right_assign_if = grb::operators::internal::left_assign_if< D1, D2, D3, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::add
			 */
			// [Example Base Operator Implementation]
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using add = grb::operators::internal::add< IN1, IN2, OUT, grb::Backend::reference >;
			// [Example Base Operator Implementation]

			/**
			 * @see grb::operators::internal::mul
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using mul = grb::operators::internal::mul< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::max
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using max = grb::operators::internal::max< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::min
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using min = grb::operators::internal::min< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::substract
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using substract = grb::operators::internal::substract< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::divide
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using divide = grb::operators::internal::divide< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::divide_reverse
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using divide_reverse = grb::operators::internal::divide_reverse< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::equal
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using equal = grb::operators::internal::equal< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::not_equal
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using not_equal = grb::operators::internal::not_equal< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::any_or
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using any_or = grb::operators::internal::any_or< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::logical_or
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using logical_or = grb::operators::internal::logical_or< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::logical_and
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using logical_and = grb::operators::internal::logical_and< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::abs_diff
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using abs_diff = grb::operators::internal::abs_diff< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::relu
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using relu = grb::operators::internal::relu< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::square_diff
			 */
			template< typename D1, typename D2, typename D3, enum Backend implementation = config::default_backend >
			using square_diff = grb::operators::internal::square_diff< D1, D2, D3, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::zip
			 */
			template< typename IN1, typename IN2, enum Backend implementation = config::default_backend >
			using zip = grb::operators::internal::zip< IN1, IN2, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::equal_first
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			using equal_first = grb::operators::internal::equal_first< IN1, IN2, OUT, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::OperatorBase
			 */
			template< typename OP, enum Backend implementation = config::default_backend >
			using OperatorBase = grb::operators::internal::OperatorBase< OP, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::OperatorFR
			 */
			template< typename OP, typename guard = void, enum Backend implementation = config::default_backend >
			using OperatorFR = grb::operators::internal::OperatorFR< OP, guard, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::OperatorFL
			 */
			template< typename OP, typename guard = void, enum Backend implementation = config::default_backend >
			using OperatorFL = grb::operators::internal::OperatorFL< OP, guard, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::OperatorNoFR
			 */
			template< typename OP, typename guard = void, enum Backend implementation = config::default_backend >
			using OperatorNoFR = grb::operators::internal::OperatorNoFR< OP, guard, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::OperatorNoFRFL
			 */
			template< typename OP, typename guard = void, enum Backend implementation = config::default_backend >
			using OperatorNoFRFL = grb::operators::internal::OperatorNoFRFL< OP, guard, grb::Backend::reference >;

			/**
			 * @see grb::operators::internal::Operator
			 */
			template< typename OP, enum Backend implementation = config::default_backend >
			using Operator = grb::operators::internal::Operator< OP, grb::Backend::reference >;

		} // namespace internal

	} // namespace operators

} // namespace alp

#endif // _H_ALP_INTERNAL_OPERATORS_BASE

