
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
 * Provides an ALP monoid.
 *
 * @author A. N. Yzelman
 * @date 15 March, 2016
 */

#ifndef _H_GRB_MONOID
#define _H_GRB_MONOID

#ifdef _DEBUG
#include <cstdio>
#endif

#include <cstddef> //size_t
#include <cstdlib> //posix_memalign, rand
#include <type_traits>

#include <assert.h>

#include <graphblas/identities.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/type_traits.hpp>


namespace grb {

	/**
	 * A generalised monoid.
	 *
	 * @tparam _OP The monoid operator.
	 * @tparam _ID The monoid identity (the `0').
	 */
	template< class _OP, template< typename > class _ID >
	class Monoid {

		static_assert( grb::is_operator< _OP >::value,
			"First template argument to Monoid must be a GraphBLAS operator" );

		static_assert( grb::is_associative< _OP >::value,
			"Cannot form a monoid using the given operator since it is not "
			"associative" );

		static_assert(
			std::is_same< typename _OP::D1, typename _OP::D3 >::value ||
				std::is_same< typename _OP::D2, typename _OP::D3 >::value,
			"Cannot form a monoid when the output domain does not match at least "
			"one of its input domains" );

	public:

		/** The left-hand side input domain. */
		typedef typename _OP::D1 D1;

		/** The right-hand side input domain. */
		typedef typename _OP::D2 D2;

		/** The output domain. */
		typedef typename _OP::D3 D3;

		/** The type of the underlying operator. */
		typedef _OP Operator;

		/** The underlying identity. */
		template< typename IdentityType >
		using Identity = _ID< IdentityType >;


	private:

		/**
		 * The underlying binary operator.
		 *
		 * For stateless operators, this field corresponds to empty storage.
		 */
		Operator op;


	public:

		/**
		 * Constructor that infers a default operator, given the operator type.
		 * Useful for stateless operators.
		 */
		Monoid() : op() {}

		/**
		 * Retrieves the identity corresponding to this monoid. The identity value
		 * will be cast to the requested domain.
		 *
		 * @tparam D The requested domain of the identity.
		 *
		 * @returns The identity corresponding to this monoid, cast to the requested
		 *          domain.
		 */
		template< typename D >
		constexpr D getIdentity() const {
			return Identity< D >::value();
		}

		/**
		 * Retrieves the underlying operator.
		 *
		 * @return The underlying operator. Any state is copied.
		 */
		Operator getOperator() const {
			return op;
		}

	};

	// type traits
	template< class _OP, template< typename > class _ID >
	struct is_monoid< Monoid< _OP, _ID > > {
		/** This is a GraphBLAS monoid. */
		static const constexpr bool value = true;
	};

	template< class OP, template< typename > class ID >
	struct has_immutable_nonzeroes< Monoid< OP, ID > > {
		static const constexpr bool value =
			grb::is_monoid< Monoid< OP, ID > >::value &&
			std::is_same< OP, typename grb::operators::logical_or<
				typename OP::D1, typename OP::D2, typename OP::D3
			> >::value;
	};

} // namespace grb

#endif

