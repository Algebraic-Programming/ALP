
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
 * Specifies the ALP algebraic type traits
 *
 * @author A. N. Yzelman
 * @date 25th of March, 2019
 */

#ifndef _H_GRB_TYPE_TRAITS
#define _H_GRB_TYPE_TRAITS

#include <type_traits>


namespace grb {

	/**
	 * Used to inspect whether a given type is an ALP/GraphBLAS container.
	 *
	 * @tparam T The type to inspect.
	 *
	 * There are only two ALP/GraphBLAS containers:
	 *  -# grb::Vector, and
	 *  -# grb::Matrix.
	 *
	 * \ingroup typeTraits
	 */
	template< typename T >
	struct is_container {

		/**
		 * Whether \a T is an ALP/GraphBLAS container.
		 *
		 * \internal Base case: an arbitrary type is not an ALP/GraphBLAS object.
		 */
		static const constexpr bool value = false;

	};

	/**
	 * Used to inspect whether a given type is an ALP semiring.
	 *
	 * @tparam T The type to inspect.
	 *
	 * \ingroup typeTraits
	 */
	template< typename T >
	struct is_semiring {

		/**
		 * Whether \a T is an ALP semiring.
		 *
		 * \internal Base case: an arbitrary type is not a semiring.
		 */
		static const constexpr bool value = false;

	};

	/**
	 * Used to inspect whether a given type is an ALP monoid.
	 *
	 * @tparam T The type to inspect.
	 *
	 * \ingroup typeTraits
	 */
	template< typename T >
	struct is_monoid {

		/**
		 * Whether \a T is an ALP monoid.
		 *
		 * \internal Base case: an arbitrary type is not an ALP monoid.
		 */
		static const constexpr bool value = false;

	};

	/**
	 * Used to inspect whether a given type is an ALP operator.
	 *
	 * @tparam T The type to inspect.
	 *
	 * \ingroup typeTraits
	 */
	template< typename T >
	struct is_operator {

		/**
		 * Whether \a T is an ALP operator.
		 *
		 * \internal Base case: an arbitrary type is not an ALP operator.
		 */
		static const constexpr bool value = false;
	};

	/**
	 * Used to inspect whether a given type is an ALP matrix selection operator.
	 *
	 * @tparam T The type to inspect.
	 *
	 * \ingroup typeTraits
	 */
	template< typename T >
	struct is_matrix_selection_operator {

		/**
		 * Whether \a T is an ALP operator.
		 *
		 * \internal Base case: an arbitrary type is not an ALP operator.
		 */
		static constexpr bool value = false;
	};

	/**
	 * Used to inspect whether a given type is an ALP/GraphBLAS object.
	 *
	 * @tparam T The type to inspect.
	 *
	 * An ALP/GraphBLAS object is either an ALP/GraphBLAS container or an ALP
	 * semiring, monoid, or operator.
	 *
	 * @see #grb::is_monoid
	 * @see #grb::is_semiring
	 * @see #grb::is_operator
	 * @see #grb::is_container
	 *
	 * \ingroup typeTraits
	 */
	template< typename T >
	struct is_object {

		/**
		 * Whether the given time is an ALP/GraphBLAS object.
		 */
		static const constexpr bool value = is_container< T >::value ||
			is_semiring< T >::value ||
			is_monoid< T >::value ||
			is_operator< T >::value;
	};

	/**
	 * Used to inspect whether a given operator or monoid is idempotent.
	 *
	 * @tparam T The operator or monoid to inspect.
	 *
	 * An example of an idempotent operator is the logical OR,
	 * #grb::operators::logical_or.
	 *
	 * \internal
	 * Since there are relatively few idempotent operators we rely on explicitly
	 * overriding the default <tt>false</tt> type trait.
	 *
	 * \todo This has the disadvantage that user-defined operators do not easily
	 *       embed an idempotent trait. This should hence be re-written to use the
	 *       same mechanism as for #grb::is_associative and #grb::is_idempotent.
	 * \endinternal
	 *
	 * \ingroup typeTraits
	 */
	template< typename T, typename = void >
	struct is_idempotent {

		static_assert( is_operator< T >::value || is_monoid< T >::value,
			"Template argument to grb::is_idempotent must be an operator or a monoid!" );

		/** Wheter \a T is idempotent. */
		static const constexpr bool value = false;

	};

	/**
	 * \internal
	 * Specialisation for ALP monoids.
	 * \endinternal
	 *
	 * \ingroup typeTraits
	 */
	template< typename Monoid >
	struct is_idempotent<
		Monoid,
		typename std::enable_if< is_monoid< Monoid >::value, void >::type
	> {
		static const constexpr bool value =
			is_idempotent< typename Monoid::Operator >::value;
	};

	/**
	 * Used to inspect whether a given operator or monoid is associative.
	 *
	 * \note Monoids are associative by definition, but this type traits is
	 *       nonetheless defined for them so as to preserve symmetry in the API;
	 *       see, e.g., #grb::is_commutative or #grb::is_idempotent.
	 *
	 * @tparam T The operator or monoid to inspect.
	 *
	 * An example of an associative operator is the logical or,
	 * #grb::operators::logical_or.
	 *
	 * \ingroup typeTraits
	 */
	template< typename T, typename = void >
	struct is_associative {

		static_assert( is_operator< T >::value || is_monoid< T >::value,
			"Template argument should be an ALP binary operator or monoid." );

		/** Whether \a T is associative. */
		static const constexpr bool value = false;

	};

	/**
	 * \internal
	 * Specialisation for ALP monoids.
	 * \endinternal
	 *
	 * \ingroup typeTraits
	 */
	template< typename Monoid >
	struct is_associative<
		Monoid,
		typename std::enable_if< is_monoid< Monoid >::value, void >::type
	> {
		static_assert( is_associative< typename Monoid::Operator >::value,
			"Malformed ALP monoid encountered" );
		static const constexpr bool value = true;
	};

	/**
	 * Used to inspect whether a given operator or monoid is commutative.
	 *
	 * @tparam T The operator or monoid to inspect.
	 *
	 * An example of a commutative operator is numerical addition,
	 * #grb::operators::add.
	 *
	 * \ingroup typeTraits
	 */
	template< typename T, typename = void >
	struct is_commutative {

		static_assert( is_operator< T >::value || is_monoid< T >::value,
			"Template argument should be an ALP binary operator or monoid." );

		/** Whether \a T is commutative. */
		static const constexpr bool value = false;

	};

	/**
	 * \internal
	 * Specialisation for ALP monoids.
	 * \endinternal
	 *
	 * \ingroup typeTraits
	 */
	template< typename Monoid >
	struct is_commutative<
		Monoid,
		typename std::enable_if< is_monoid< Monoid >::value, void >::type
	> {
		/** \internal Simply inherit from underlying operator */
		static const constexpr bool value =
			is_commutative< typename Monoid::Operator >::value;
	};

	/**
	 * Used to inspect whether a given semiring has immutable nonzeroes under
	 * addition.
	 *
	 * @tparam T The semiring to inspect.
	 *
	 * An example of a monoid with an immutable identity is the logical OR,
	 * #grb::operators::logical_or.
	 *
	 * \ingroup typeTraits
	 */
	template< typename T >
	struct has_immutable_nonzeroes {

		static_assert( is_semiring< T >::value,
			"Template argument to grb::has_immutable_nonzeroes must be a "
			"semiring!" );

		/** Whether \a T a semiring where nonzeroes are immutable. */
		static const constexpr bool value = false;

	};

	namespace internal {

		/**
		 * Whether or not a given operator could translate to a no-op;
		 * i.e., leave its outputs unmodified. This can be relevant
		 * because it indicates situations where grb::apply could leave
		 * the output uninitialised, which may well not be as intended.
		 *
		 * An example of an operator that non-trivially may result in a
		 * no-op is grb::operators::left_assign_if. Such operators must
		 * overload this internal type trait.
		 *
		 * \ingroup typeTraits
		 */
		template< typename OP >
		struct maybe_noop {
			static_assert( is_operator< OP >::value,
				"Argument to internal::maybe_noop must be an operator."
			);
			static const constexpr bool value = false;
		};

	} // end namespace grb::internal

} // namespace grb

#endif // end _H_GRB_TYPE_TRAITS

