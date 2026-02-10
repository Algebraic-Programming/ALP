
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

	// after all of the standard definitions, declare some standard monoids

	/**
	 * A name space that contains a set of standard monoids.
	 *
	 * Standard monoids include:
	 *  - #plus, for numerical addition
	 *  - #times, for numerical multiplication
	 *  - #min, for the minimum relation
	 *  - #max, for the maximum relation
	 *  - #lor, for the logical-or relation
	 *  - #land, for the logical-and relation
	 *  - #lxor, for the exclusive-or relation
	 *  - #lxnor, for the negated exclusive-or relation.
	 *
	 * \note In the above, the prefix letter <tt>l</tt> stands for \em logical,
	 *       e.g., <tt>lor</tt> stands for logical-or.
	 *
	 * There are also a couple of aliases to match different preferences:
	 *  - #add (same as #plus),
	 *  - #mul (same as #times),
	 *  - #lneq (same as #lxor), and
	 *  - #leq (same as #lxnor).
	 *
	 * \note The #min and #max monoids have different identities depending on the
	 *       domain. The standard monoids defined here auto-adapt to the correct
	 *       identity.
	 */
	namespace monoids {

		/**
		 * The plus monoid.
		 *
		 * Uses \em addition (plus) as the operator, and zero as its identity.
		 *
		 * The three domains of the monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the operator
		 * @tparam D2 The right-hand input domain of the operator
		 * @tparam D3 The output domain of the operator
		 *
		 * The types \a D2 and \a D3 are optional. If \a D3 is not explicitly given,
		 * it will be set to \a D2. If \a D2 is not explicitly given, it will be set
		 * to \a D1.
		 *
		 * This is a commutative monoid (assuming \a D1 equals \a D2).
		 *
		 * @see #add.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2 >
		using plus = grb::Monoid<
			grb::operators::add< D1, D2, D3 >,
			grb::identities::zero
		>;

		/**
		 * The times monoid.
		 *
		 * Uses \em multiplication (times) as the operator, and one as its identity.
		 *
		 * The three domains of the monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the operator
		 * @tparam D2 The right-hand input domain of the operator
		 * @tparam D3 The output domain of the operator
		 *
		 * The types \a D2 and \a D3 are optional. If \a D3 is not explicitly given,
		 * it will be set to \a D2. If \a D2 is not explicitly given, it will be set
		 * to \a D1.
		 *
		 * @see #mul.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2 >
		using times = grb::Monoid<
			grb::operators::mul< D1, D2, D3 >,
			grb::identities::one
		>;

		/**
		 * This is an alias of #plus.
		 *
		 * @see #add.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2 >
		using add = plus< D1, D2, D3 >;

		/**
		 * This is an alias of #times.
		 *
		 * @see #times.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2 >
		using mul = times< D1, D2, D3 >;

		/**
		 * The min monoid.
		 *
		 * Uses \em min as the operator. If the domain is floating-point, uses
		 * infinity as its identity; if the domain is integer, uses its maximum
		 * representable value as the identity of this monoid.
		 *
		 * The three domains of the monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the operator
		 * @tparam D2 The right-hand input domain of the operator
		 * @tparam D3 The output domain of the operator
		 *
		 * The types \a D2 and \a D3 are optional. If \a D3 is not explicitly given,
		 * it will be set to \a D2. If \a D2 is not explicitly given, it will be set
		 * to \a D1.
		 *
		 * This is a commutative monoid (assuming \a D1 equals \a D2).
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2 >
		using min = grb::Monoid<
			grb::operators::min< D1, D2, D3 >,
			grb::identities::infinity
		>;

		/**
		 * The max monoid.
		 *
		 * Uses \em max as the operator. If the domain is floating-point, uses
		 * negative infinity (\f$ -\infty \f$) as its identity; if the domain is
		 * integer, uses its minimum representable value as the identity of this
		 * monoid.
		 *
		 * The three domains of the monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the operator
		 * @tparam D2 The right-hand input domain of the operator
		 * @tparam D3 The output domain of the operator
		 *
		 * The types \a D2 and \a D3 are optional. If \a D3 is not explicitly given,
		 * it will be set to \a D2. If \a D2 is not explicitly given, it will be set
		 * to \a D1.
		 *
		 * This is a commutative monoid (assuming \a D1 equals \a D2).
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2 >
		using max = grb::Monoid<
			grb::operators::max< D1, D2, D3 >,
			grb::identities::negative_infinity
		>;

		/**
		 * The logical-or monoid.
		 *
		 * Uses \em logical-or as the operator and <tt>false</tt> as its identity.
		 *
		 * If the domain is non-boolean, inputs will be cast to a Boolean before the
		 * operator is invoked, while the result will be cast to the target domain on
		 * output.
		 *
		 * The three domains of the monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the operator
		 * @tparam D2 The right-hand input domain of the operator
		 * @tparam D3 The output domain of the operator
		 *
		 * The types \a D2 and \a D3 are optional. If \a D3 is not explicitly given,
		 * it will be set to \a D2. If \a D2 is not explicitly given, it will be set
		 * to \a D1.
		 *
		 * This is a commutative monoid (assuming \a D1 equals \a D2).
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2 >
		using lor = grb::Monoid<
			grb::operators::logical_or< D1, D2, D3 >,
			grb::identities::logical_false
		>;

		/**
		 * The logical-and monoid.
		 *
		 * Uses \em logical-and as the operator and <tt>true</tt> as its identity.
		 *
		 * If the domain is non-boolean, inputs will be cast to a Boolean before the
		 * operator is invoked, while the result will be cast to the target domain on
		 * output.
		 *
		 * The three domains of the monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the operator
		 * @tparam D2 The right-hand input domain of the operator
		 * @tparam D3 The output domain of the operator
		 *
		 * The types \a D2 and \a D3 are optional. If \a D3 is not explicitly given,
		 * it will be set to \a D2. If \a D2 is not explicitly given, it will be set
		 * to \a D1.
		 *
		 * This is a commutative monoid (assuming \a D1 equals \a D2).
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2 >
		using land = grb::Monoid<
			grb::operators::logical_and< D1, D2, D3 >,
			grb::identities::logical_true
		>;

		/**
		 * The logical-exclusive-or monoid.
		 *
		 * Uses \em logical-exclusive-or as the operator and <tt>false</tt> as its
		 * identity.
		 *
		 * If the domain is non-boolean, inputs will be cast to a Boolean before the
		 * operator is invoked, while the result will be cast to the target domain on
		 * output.
		 *
		 * The three domains of the monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the operator
		 * @tparam D2 The right-hand input domain of the operator
		 * @tparam D3 The output domain of the operator
		 *
		 * The types \a D2 and \a D3 are optional. If \a D3 is not explicitly given,
		 * it will be set to \a D2. If \a D2 is not explicitly given, it will be set
		 * to \a D1.
		 *
		 * This is a commutative monoid (assuming \a D1 equals \a D2).
		 *
		 * @see #lneq.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2 >
		using lxor = grb::Monoid<
			grb::operators::not_equal< D1, D2, D3 >,
			grb::identities::logical_false
		>;

		/**
		 * The logical-not-equals monoid.
		 *
		 * Uses \em logical-not-equals as the operator and <tt>false</tt> as its
		 * identity.
		 *
		 * If the domain is non-boolean, inputs will be cast to a Boolean before the
		 * operator is invoked, while the result will be cast to the target domain on
		 * output.
		 *
		 * The three domains of the monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the operator
		 * @tparam D2 The right-hand input domain of the operator
		 * @tparam D3 The output domain of the operator
		 *
		 * The types \a D2 and \a D3 are optional. If \a D3 is not explicitly given,
		 * it will be set to \a D2. If \a D2 is not explicitly given, it will be set
		 * to \a D1.
		 *
		 * This is a commutative monoid (assuming \a D1 equals \a D2).
		 *
		 * @see #lxor.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2 >
		using lneq = lxor< D1, D2, D3 >;

		/**
		 * The logical-negated-exclusive-or monoid.
		 *
		 * Uses \em logical-negated-exclusive-or as the operator and <tt>true</tt> as
		 * its identity.
		 *
		 * If the domain is non-boolean, inputs will be cast to a Boolean before the
		 * operator is invoked, while the result will be cast to the target domain on
		 * output.
		 *
		 * The three domains of the monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the operator
		 * @tparam D2 The right-hand input domain of the operator
		 * @tparam D3 The output domain of the operator
		 *
		 * The types \a D2 and \a D3 are optional. If \a D3 is not explicitly given,
		 * it will be set to \a D2. If \a D2 is not explicitly given, it will be set
		 * to \a D1.
		 *
		 * This is a commutative monoid (assuming \a D1 equals \a D2).
		 *
		 * @see #leq.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2 >
		using lxnor = grb::Monoid<
			grb::operators::equal< D1, D2, D3 >,
			grb::identities::logical_true
		>;

		/**
		 * The logical-equals monoid.
		 *
		 * Uses \em logical-equals as the operator and <tt>true</tt> as its identity.
		 *
		 * If the domain is non-boolean, inputs will be cast to a Boolean before the
		 * operator is invoked, while the result will be cast to the target domain on
		 * output.
		 *
		 * The three domains of the monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the operator
		 * @tparam D2 The right-hand input domain of the operator
		 * @tparam D3 The output domain of the operator
		 *
		 * The types \a D2 and \a D3 are optional. If \a D3 is not explicitly given,
		 * it will be set to \a D2. If \a D2 is not explicitly given, it will be set
		 * to \a D1.
		 *
		 * This is a commutative monoid (assuming \a D1 equals \a D2).
		 *
		 * @see #lxnor.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2 >
		using leq = lxnor< D1, D2, D3 >;

	}

} // namespace grb

#endif

