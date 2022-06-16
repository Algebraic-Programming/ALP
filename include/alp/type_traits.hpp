
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

#ifndef _H_ALP_TYPE_TRAITS
#define _H_ALP_TYPE_TRAITS

#include <type_traits>
#include <alp/views.hpp>

namespace alp {

	/**
	 * Used to inspect whether a given type is an ALP scalar.
	 *
	 * @tparam T The type to inspect.
	 *
	 * \note An arbitrary type is not an ALP scalar.
	 *
	 */
	template< typename T >
	struct is_scalar : std::false_type {};

	/**
	 * Used to inspect whether a given type is an ALP vector.
	 *
	 * @tparam T The type to inspect.
	 *
	 * \note An arbitrary type is not an ALP vector.
	 *
	 */
	template< typename T >
	struct is_vector : std::false_type {};

	/**
	 * Used to inspect whether a given type is an ALP matrix.
	 *
	 * @tparam T The type to inspect.
	 *
	 * \note An arbitrary type is not an ALP matrix.
	 *
	 */
	template< typename T >
	struct is_matrix : std::false_type {};

	/**
	 * Used to inspect whether a given type is an ALP container.
	 *
	 * @tparam T The type to inspect.
	 *
	 * There are only three ALP containers:
	 *  -# alp::Scalar,
	 *  -# alp::Vector, and
	 *  -# alp::Matrix.
	 */
	template< typename T >
	struct is_container : std::integral_constant<
		bool,
		is_scalar< T >::value || is_vector< T >::value || is_matrix< T >::value
	> {};

	namespace internal {

		/**
		 * Used to inspect whether a given type is an internal container.
		 *
		 * @tparam T The type to inspect.
		 *
		 * There are only two internal containers:
		 *  -# alp::internal::Vector, and
		 *  -# alp::internal::Matrix.
		 */
		template< typename T >
		struct is_container : std::false_type {};

	} // namespace internal

	/**
	 * Used to inspect whether a given type is an ALP semiring.
	 *
	 * @tparam T The type to inspect.
	 */
	template< typename T >
	struct is_semiring {
		/** Base case: an arbitrary type is not a semiring. */
		static const constexpr bool value = false;
	};

	/**
	 * Used to inspect whether a given type is an ALP monoid.
	 *
	 * @tparam T The type to inspect.
	 */
	template< typename T >
	struct is_monoid {
		/** Base case: an arbitrary type is not a monoid. */
		static const constexpr bool value = false;
	};

	/**
	 * Used to inspect whether a given type is an ALP operator.
	 *
	 * @tparam T The type to inspect.
	 */
	template< typename T >
	struct is_operator {
		/** Base case: an arbitrary type is not an operator. */
		static const constexpr bool value = false;
	};

	/**
	 * Used to inspect whether a given type is an ALP object.
	 *
	 * @tparam T The type to inspect.
	 *
	 * A ALP object is either a container, a semiring, a monoid, or an
	 * operator.
	 *
	 * @see #is_monoid
	 * @see #is_semiring
	 * @see #is_operator
	 * @see #is_container
	 */
	template< typename T >
	struct is_object {
		/** A ALP object is either a container, a semiring, a monoid, or an operator. */
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
	 * #alp::operators::logical_or.
	 */
	template< typename T >
	struct is_idempotent {
		static_assert( is_operator< T >::value, "Template argument to alp::is_idempotent must be an operator!" );
		static const constexpr bool value = false;
	};

	/**
	 * Used to inspect whether a given semiring has immutable nonzeroes under
	 * addition.
	 *
	 * @tparam T The semiring to inspect.
	 *
	 * An example of a monoid with an immutable identity is the logical OR,
	 * #alp::operators::logical_or.
	 */
	template< typename T >
	struct has_immutable_nonzeroes {
		static_assert( is_semiring< T >::value,
			"Template argument to alp::has_immutable_nonzeroes must be a "
			"semiring!" );
		static const constexpr bool value = false;
	};

	namespace internal {

		/**
		 * Whether or not a given operator could translate to a no-op;
		 * i.e., leave its outputs unmodified. This can be relevant
		 * because it indicates situations where alp::apply could leave
		 * the output uninitialised, which may well not be as intended.
		 *
		 * An example of an operator that non-trivially may result in a
		 * no-op is alp::operators::left_assign_if. Such operators must
		 * overload this internal type trait.
		 */
		template< typename OP >
		struct maybe_noop {
			static_assert( is_operator< OP >::value,
				"Argument to internal::maybe_noop must be an operator."
			);
			static const constexpr bool value = false;
		};

	} // end namespace alp::internal

	namespace internal {

		/**
		 * Used to get a structure type of the given ALP container
		 *
		 * @tparam T The ALP container to inspect.
		 *
		 */
		template< typename Container >
		struct inspect_structure {};

		/**
		 * Used to get a View type of the given ALP container
		 *
		 * @tparam T The ALP container to inspect.
		 *
		 */
		template< typename Container >
		struct inspect_view {};


		template< typename T >
		struct is_view_over_concrete_container : is_view_over_concrete_container< typename inspect_view< T >::type::applied_to > {
			static_assert( is_vector< T >::value || is_matrix< T >::value, "Argument to internal::is_view_over_concrete_container must be an ALP vector or an ALP matrix.");
		};

		template<>
		struct is_view_over_concrete_container< void > : std::true_type {};

		/**
		 * Inspects whether a view corresponds to a storage-based ALP container.
		 *
		 * ALP containers can either be storage-based or functor-based.
		 *
		 * @tparam T The view to inspect.
		 *
		 * \note A Matrix is storage-based if it has
		 *       - an original view over void, or
		 *       - any type of view over another storage-based matrix.
		 *
		 */
		template< typename View >
		struct is_storage_based : is_storage_based<
			typename inspect_view< typename View::applied_to >::type
		> {};

		/** Original view over void is by definition storage based ALP container. */
		template<>
		struct is_storage_based< view::Original< void > > : std::true_type {};

		/** Functor views are not storage-based ALP containers. */
		template< typename LambdaType >
		struct is_storage_based< view::Functor< LambdaType > > : std::false_type {};

		/**
		 * A helper type trait for \a is_functor_based.
		 *
		 * @tparam View       The view to inspect.
		 * @tparam AppliedTo  The type that View is applied to.
		 *
		 * @see is_functor_based
		 *
		 */
		template< typename View, typename AppliedTo >
		struct is_functor_based_helper : is_functor_based_helper<
			/** The view of the ALP container this view is applied to */
			typename inspect_view< typename View::applied_to >::type,
			/** What the above view is applied to */
			typename inspect_view< typename View::applied_to >::type::applied_to
		> {};

		/** Functor view over a lambda type is by definition functor-based ALP container. */
		template< typename AppliedTo >
		struct is_functor_based_helper< view::Functor< AppliedTo >, AppliedTo > : std::true_type {};

		template< typename AppliedTo >
		struct is_functor_based_helper< view::Original< void >, AppliedTo > : std::false_type {};

		/**
		 * Inspects whether a view corresponds to a functor-based ALP container.
		 *
		 * ALP containers can either be storage-based or functor-based.
		 *
		 * @tparam View  The view to inspect.
		 *
		 * \note A Matrix is functor-based if it has
		 *       - a functor view over a lambda type, or
		 *       - any type of view over another functor-based matrix.
		 *
		 * @see is_functor_based_helper
		 *
		 */
		template< typename View >
		struct is_functor_based : is_functor_based_helper< View, typename View::applied_to > {};

		/**
		 * Inspects whether a provided view is associated with an ALP container
		 * that allocates the container data-related memory (either the storage
		 * or the functor), or, in other words,
		 * whether it is a view over another ALP container.
		 *
		 * @tparam T The view type to inspect.
		 *
		 * The value is true if the provided view corresponds to an ALP container that
		 * - allocates memory for container storage, or
		 * - allocates memory for a functor
		 * The value is false otherwise, i.e., if the provided view type corresponds
		 * to a view over another ALP container, and, therefore, does not need to
		 * allocate memory for storage/functor.
		 *
		 */
		template< typename View >
		struct allocates_memory : std::integral_constant<
			bool,
			std::is_same< view::Original< void >, View >::value ||
			std::is_same< view::Functor< typename View::applied_to >, View >::value
		> {};
	} // namespace internal

} // namespace alp

#endif
