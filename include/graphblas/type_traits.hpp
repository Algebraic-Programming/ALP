
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
 * @date 25th of March, 2019
 */

#ifndef _H_GRB_TYPE_TRAITS
#define _H_GRB_TYPE_TRAITS

namespace grb {

	/**
	 * Used to inspect whether a given type is a GraphBLAS container.
	 *
	 * @tparam T The type to inspect.
	 *
	 * There are only two GraphBLAS containers:
	 *  -# grb::Vector, and
	 *  -# grb::Matrix.
	 */
	template< typename T >
	struct is_container {
		/** Base case: an arbitrary type is not a GraphBLAS object. */
		static const constexpr bool value = false;
	};

	/**
	 * Used to inspect whether a given type is a GraphBLAS semiring.
	 *
	 * @tparam T The type to inspect.
	 */
	template< typename T >
	struct is_semiring {
		/** Base case: an arbitrary type is not a semiring. */
		static const constexpr bool value = false;
	};

	/**
	 * Used to inspect whether a given type is a GraphBLAS monoid.
	 *
	 * @tparam T The type to inspect.
	 */
	template< typename T >
	struct is_monoid {
		/** Base case: an arbitrary type is not a monoid. */
		static const constexpr bool value = false;
	};

	/**
	 * Used to inspect whether a given type is a GraphBLAS operator.
	 *
	 * @tparam T The type to inspect.
	 */
	template< typename T >
	struct is_operator {
		/** Base case: an arbitrary type is not an operator. */
		static const constexpr bool value = false;
	};

	/**
	 * Used to inspect whether a given type is a GraphBLAS object.
	 *
	 * @tparam T The type to inspect.
	 *
	 * A GraphBLAS object is either a container, a semiring, a monoid, or an
	 * operator.
	 *
	 * @see #is_monoid
	 * @see #is_semiring
	 * @see #is_operator
	 * @see #is_container
	 */
	template< typename T >
	struct is_object {
		/** A GraphBLAS object is either a container, a semiring, a monoid, or an operator. */
		static const constexpr bool value = is_container< T >::value ||
			is_semiring< T >::value ||
			is_monoid< T >::value ||
			is_operator< T >::value;
	};

	/**
	 * Used to inspect whether a given operator is idempotent.
	 *
	 * @tparam T The operator to inspect.
	 *
	 * An example of an idempotent operator is the logical OR,
	 * #grb::operators::logical_or.
	 */
	template< typename T >
	struct is_idempotent {
		static_assert( is_operator< T >::value, "Template argument to grb::is_idempotent must be an operator!" );
		static const constexpr bool value = false;
	};

	/**
	 * Used to inspect whether a given semiring has immutable nonzeroes under
	 * addition.
	 *
	 * @tparam T The semiring to inspect.
	 *
	 * An example of a monoid with an immutable identity is the logical OR,
	 * #grb::operators::logical_or.
	 */
	template< typename T >
	struct has_immutable_nonzeroes {
		static_assert( is_semiring< T >::value,
			"Template argument to grb::has_immutable_nonzeroes must be a "
			"semiring!" );
		static const constexpr bool value = false;
	};

	/**
	 * @brief Used to select the iterator tag: if no IterT::iterator_category field, it assumes
	 * 		std::forward_iterator_tag
	 * 
	 * @tparam IterT iterator type
	 */
	template< typename IterT > class iterator_tag_selector {

		template< typename It> static typename std::iterator_traits<It>::iterator_category select( int ) {
			return typename std::iterator_traits<IterT>::iterator_category();
		}

		template< typename It> static typename std::forward_iterator_tag select( ... ) {
			return typename std::forward_iterator_tag();
		}

	public:
		using iterator_category = decltype( select< IterT >( 0 ) );

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

#endif
