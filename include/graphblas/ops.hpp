
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
 * Provides a set of standard binary operators.
 *
 * @author A. N. Yzelman
 * @date 8th of August, 2016
 */

#ifndef _H_GRB_OPERATORS
#define _H_GRB_OPERATORS

#include "internalops.hpp"
#include "type_traits.hpp"


namespace grb {

	/**
	 * This namespace holds various standard operators such as #grb::operators::add
	 * and #grb::operators::mul.
	 */
	namespace operators {

		/**
		 * This operator discards all right-hand side input and simply copies the
		 * left-hand side input to the output variable. It exposes the complete
		 * interface detailed in grb::operators::internal::Operator. This operator
		 * can be passed to any GraphBLAS function or object constructor.
		 *
		 * Mathematical notation: \f$ \odot(x,y)\ \to\ x \f$.
		 *
		 * @tparam D1 The left-hand side input domain.
		 * @tparam D2 The right-hand side input domain.
		 * @tparam D3 The output domain.
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class left_assign :
			public internal::Operator<
				internal::left_assign< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = left_assign< A, B, C, D >;

				left_assign() {}
		};

		/**
		 * This operator assigns the left-hand input if the right-hand input
		 * evaluates <tt>true</tt>. If the right-hand input does not evaluate
		 * <tt>true</tt>, then the output field is unmodified.
		 *
		 * \warning Therefore, this operator may propagate the use of uninitialised
		 *          values if not used with care. Ensuring its use with in-place
		 *          primitives is recommended.
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class left_assign_if :
			 public internal::Operator<
					internal::left_assign_if< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = left_assign_if< A, B, C, D >;

				left_assign_if() {}
		};

		/**
		 * This operator discards all left-hand side input and simply copies the
		 * right-hand side input to the output variable. It exposes the complete
		 * interface detailed in grb::operators::internal::Operator. This operator
		 * can be passed to any GraphBLAS function or object constructor.
		 *
		 * Mathematical notation: \f$ \odot(x,y)\ \to\ y \f$.
		 *
		 * @tparam D1 The left-hand side input domain.
		 * @tparam D2 The right-hand side input domain.
		 * @tparam D3 The output domain.
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class right_assign : public internal::Operator<
				internal::right_assign< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = right_assign< A, B, C, D >;

				right_assign() {}
		};

		/**
		 * This operator assigns the right-hand input if the left-hand input
		 * evaluates <tt>true</tt>. If the left-hand input does not evaluate
		 * <tt>true</tt>, then the output field is unmodified.
		 *
		 * \warning Therefore, this operator may propagate the use of uninitialised
		 *          values if not used with care. Ensuring its use with in-place
		 *          primitives is recommended.
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class right_assign_if : public internal::Operator<
				internal::right_assign_if< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = right_assign_if< A, B, C, D >;

				right_assign_if() {}
		};

		/**
		 * This operator takes the sum of the two input parameters and writes it to
		 * the output variable. It exposes the complete interface detailed in
		 * grb::operators::internal::Operator. This operator can be passed to any
		 * GraphBLAS function or object constructor.
		 *
		 * Mathematical notation: \f$ \odot(x,y)\ \to\ x + y \f$.
		 *
		 * @tparam D1 The left-hand side input domain.
		 * @tparam D2 The right-hand side input domain.
		 * @tparam D3 The output domain.
		 *
		 * \warning This operator expects numerical types for \a D1, \a D2, and
		 *          \a D3, or types that have the appropriate operator+-functions
		 *          available.
		 */
		// [Operator Wrapping]
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class add : public internal::Operator<
				internal::add< D1, D2, D3, implementation >
			>
		{

			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = add< A, B, C, D >;

				add() {}
		};
		// [Operator Wrapping]

		/**
		 * This operator multiplies the two input parameters and writes the result to
		 * the output variable. It exposes the complete interface detailed in
		 * grb::operators::internal::Operator. This operator can be passed to any
		 * GraphBLAS function or object constructor.
		 *
		 * Mathematical notation: \f$ \odot(x,y)\ \to\ x \cdot y \f$.
		 *
		 * @tparam D1 The left-hand side input domain.
		 * @tparam D2 The right-hand side input domain.
		 * @tparam D3 The output domain.
		 *
		 * \warning This operator expects numerical types for \a D1, \a D2, and
		 *          \a D3, or types that have the appropriate operator*-functions
		 *          available.
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class mul : public internal::Operator<
				internal::mul< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = mul< A, B, C, D >;

				mul() {}
		};

		/**
		 * This operator takes the maximum of the two input parameters and writes
		 * the result to the output variable. It exposes the complete interface
		 * detailed in grb::operators::internal::Operator. This operator can be
		 * passed to any GraphBLAS function or object constructor.
		 *
		 * Mathematical notation: \f$ \max(x,y)\ \to\ \begin{cases}
		 *    x \text{ if } x > y \\
		 *    y \text{ otherwise} \end{cases} \f$.
		 *
		 * @tparam D1 The left-hand side input domain.
		 * @tparam D2 The right-hand side input domain.
		 * @tparam D3 The output domain.
		 *
		 * \warning This operator expects objects with a partial ordering defined on
		 *          and between elements of types \a D1, \a D2, and \a D3.
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class max : public internal::Operator<
				internal::max< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = max< A, B, C, D >;

				max() {}
		};

		/**
		 * This operator takes the minimum of the two input parameters and writes
		 * the result to the output variable. It exposes the complete interface
		 * detailed in grb::operators::internal::Operator. This operator can be
		 * passed to any GraphBLAS function or object constructor.
		 *
		 * Mathematical notation: \f$ \max(x,y)\ \to\ \begin{cases}
		 *    x \text{ if } x < y \\
		 *    y \text{ otherwise} \end{cases} \f$.
		 *
		 * @tparam D1 The left-hand side input domain.
		 * @tparam D2 The right-hand side input domain.
		 * @tparam D3 The output domain.
		 *
		 * \warning This operator expects objects with a partial ordering defined on
		 *          and between elements of types \a D1, \a D2, and \a D3.
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class min : public internal::Operator<
				internal::min< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = min< A, B, C, D >;

				min() {}
		};

		/**
		 * Numerical substraction of two numbers.
		 *
		 * Mathematical notation: \f$ \odot(x,y)\ \to\ x - y \f$.
		 *
		 * \note This is the inverse to #grb::operators::add.
		 *
		 * \warning This operator expects numerical types for \a D1, \a D2, and
		 *          \a D3, or types that have the appropriate operator- overloads 
		 *          available.
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class subtract : public internal::Operator<
				internal::substract< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = subtract< A, B, C, D >;

				subtract() {}
		};

		/**
		 * Numerical division of two numbers.
		 *
		 * Mathematical notation: \f$ \odot(x,y)\ \to\ x / y \f$.
		 *
		 * \note This is the inverse to #grb::operators::mul.
		 *
		 * \warning This operator expects numerical types for \a D1, \a D2, and
		 *          \a D3, or types that have the appropriate operator/-functions
		 *          available.
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class divide : public internal::Operator<
				internal::divide< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = divide< A, B, C, D >;

				divide() {}
		};

		/**
		 * Reversed division of two numbers.
		 *
		 * Mathematical notation: \f$ \odot(x,y)\ \to\ y / x \f$.
		 *
		 * \warning This operator expects numerical types for \a D1, \a D2, and
		 *          \a D3, or types that have the appropriate operator/-functions
		 *          available.
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class divide_reverse : public internal::Operator<
				internal::divide_reverse< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = divide_reverse< A, B, C, D >;

				divide_reverse() {}
		};

		/**
		 * Operator which returns <tt>true</tt> if its inputs compare equal, and
		 * <tt>false</tt> otherwise.
		 *
		 * \note This operator is the inverse of #grb::operators::not_equal.
		 *
		 * \warning This operator expects numerical types for \a D1, \a D2, and
		 *          \a D3, or types that have the appropriate operator=-functions
		 *          available.
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class equal : public internal::Operator<
				internal::equal< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = equal< A, B, C, D >;

				equal() {}
		};

		/**
		 * Operator that returns <tt>false</tt> whenever its inputs compare equal,
		 * and <tt>true</tt> otherwise.
		 *
		 * \note This operator is the inverse of #grb::operators::equal.
		 *
		 * \warning This operator expects numerical types for \a D1, \a D2, and
		 *          \a D3, or types that have the appropriate operator=-functions
		 *          available.
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class not_equal : public internal::Operator<
				internal::not_equal< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = not_equal< A, B, C, D >;

				not_equal() {}
		};

		/**
		 * This operator is a generalisation of the logical or.
		 *
		 * It assigns to the output any input which evaluates <tt>true</tt>. If there
		 * is no such input, it assigns any input that evaluates <tt>false</tt>.
		 *
		 * \note The main difference is that the output is never cast from a Boolean
		 *       <tt>true</tt> or <tt>false</tt>.
		 *
		 * The input domains must be <em>castable</em> to <tt>bool</tt>.
		 *
		 * The input domains must furthermore be \em castable to the output domain.
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class any_or : public internal::Operator<
				internal::any_or< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = any_or< A, B, C, D >;

				any_or() {}
		};

		/**
		 * The logical or.
		 *
		 * It returns <tt>true</tt> whenever any of its inputs evaluate <tt>true</tt>,
		 * and returns <tt>false</tt> otherwise.
		 *
		 * If the output domain is not Boolean, then the returned value is
		 * <tt>true</tt> or <tt>false</tt> cast to the output domain.
		 *
		 * \warning Thus both input domains and the output domain must be \em castable
		 *          to <tt>bool</tt>.
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class logical_or : public internal::Operator<
				internal::logical_or< D1, D2, D3, implementation >
			>
		{

			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = logical_or< A, B, C, D >;

				logical_or() {}
		};

		/**
		 * The logical and.
		 *
		 * It returns <tt>true</tt> when both of its inputs evaluate <tt>true</tt>,
		 * and returns <tt>false</tt> otherwise.
		 *
		 * If the output domain is not Boolean, then the returned value is
		 * <tt>true</tt> or <tt>false</tt> cast to the output domain.
		 *
		 * \warning Thus both input domains and the output domain must be \em castable
		 *          to <tt>bool</tt>.
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class logical_and : public internal::Operator<
				internal::logical_and< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = logical_and< A, B, C, D >;

				logical_and() {}
		};

		/**
		 * This operation is equivalent to #grb::operators::min.
		 *
		 * It assumes that the right-hand input is the bias, while the left-hand
		 * input is the signal.
		 *
		 * @see min
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class relu : public internal::Operator<
				internal::relu< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = relu< A, B, C, D >;

				relu() {}
		};

		/**
		 * This operator returns the absolute difference between two numbers.
		 *
		 * Mathematical notation: \f$ \odot(x,y) \to |x-y| \f$.
		 *
		 * \warning This operator expects numerical types for \a D1, \a D2, and
		 *          \a D3, or types that have the appropriate operator- and
		 *          <tt>std::abs</tt> overloads available.
		 *
		 * @see square_diff
		 */
		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class abs_diff : public internal::Operator<
				internal::abs_diff< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = abs_diff< A, B, C, D >;

				abs_diff() {}
		};

		/**
		 * The argmin operator on key-value pairs.
		 *
		 * @tparam IType The key type.
		 * @tparam VType The value type.
		 *
		 * This operator is only defined for key-value pairs encapsulated in the
		 * STL standard <tt>std::pair</tt>. The return type equals that of the
		 * key type.
		 *
		 * This operator returns the key corresponding to the key-value pair whose
		 * value evaluates less than the other.
		 *
		 * \warning If both values are equal, any key may be returned.
		 *
		 * @see argmax
		 * @see equal_first
		 */
		template< typename IType, typename VType >
		class argmin : public internal::Operator< internal::argmin< IType, VType > > {
			public:
				argmin() {}
		};

		/**
		 * The argmax operator on key-value pairs.
		 *
		 * @tparam IType The key type.
		 * @tparam VType The value type.
		 *
		 * This operator is only defined for key-value pairs encapsulated in the
		 * STL standard <tt>std::pair</tt>. The return type equals that of the
		 * key type.
		 *
		 * This operator returns the key corresponding to the key-value pair whose
		 * value evaluates greater than the other.
		 *
		 * \warning If both values are equal, any key may be returned.
		 *
		 * @see argmin
		 * @see equal_first
		 */
		template< typename IType, typename VType >
		class argmax : public internal::Operator< internal::argmax< IType, VType > > {
			public:
				argmax() {}
		};

		/**
		 * This operation returns the squared difference between two numbers.
		 *
		 * Mathematical notation: \f$ \odot(x,y) \to (x-y)^2 \f$.
		 *
		 * \warning This operator expects numerical types for \a D1, \a D2, and
		 *          \a D3, or types that have the appropriate operator- and
		 *          operator* overloads available.
		 *
		 * @see abs_diff
		 */
		template<
			typename D1, typename D2, typename D3,
			enum Backend implementation = config::default_backend
		>
		class square_diff : public internal::Operator<
				internal::square_diff< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = square_diff< A, B, C, D >;

				square_diff() {}
		};

		/**
		 * The zip operator that operators on keys as a left-hand input and values as
		 * a right hand input, producing a key-value <tt>std::pair</tt>.
		 *
		 * @tparam IN1 The key type.
		 * @tparam IN2 The value type.
		 *
		 * The output domain is fixed to <tt>std::pair< IN1, IN2 ></tt>.
		 */
		template<
			typename IN1, typename IN2,
			enum Backend implementation = config::default_backend
		>
		class zip : public internal::Operator<
				internal::zip< IN1, IN2, implementation >
			>
		{
			public:

				template< typename A, typename B, enum Backend D >
				using GenericOperator = zip< A, B, D >;

				zip() {}
		};

		/**
		 * Compares <tt>std::pair</tt> inputs taking the first entry in every pair
		 * as the comparison key, and returns <tt>true</tt> or <tt>false</tt>
		 * accordingly.
		 *
		 * The input domains must both be <tt>std::pair</tt>.
		 *
		 * \note If the output type is not Boolean, the output is cast from Boolean
		 *       to the output domain.
		 *
		 * The output domain must hence be \em castable from <tt>bool</tt>.
		 */
 		template<
			typename D1, typename D2 = D1, typename D3 = D2,
			enum Backend implementation = config::default_backend
		>
		class equal_first : public internal::Operator<
				internal::equal_first< D1, D2, D3, implementation >
			>
		{
			public:

				template< typename A, typename B, typename C, enum Backend D >
				using GenericOperator = equal_first< A, B, C, D >;

				equal_first() {}
		};

	} // namespace operators

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::left_assign_if< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::right_assign_if< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::left_assign< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::right_assign< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	// [Operator Type Traits]
	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::add< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};
	// [Operator Type Traits]

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::mul< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::max< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::min< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::subtract< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::divide< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::divide_reverse< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::equal< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::not_equal< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::any_or< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::logical_or< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::logical_and< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::abs_diff< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::relu< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename IType, typename VType >
	struct is_operator< operators::argmin< IType, VType > > {
		static const constexpr bool value = true;
	};

	template< typename IType, typename VType >
	struct is_operator< operators::argmax< IType, VType > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::square_diff< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename IN1, typename IN2, enum Backend implementation >
	struct is_operator< operators::zip< IN1, IN2, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3, enum Backend implementation >
	struct is_operator< operators::equal_first< D1, D2, D3, implementation > > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3 >
	struct is_idempotent< operators::min< D1, D2, D3 >, void > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3 >
	struct is_idempotent< operators::max< D1, D2, D3 >, void > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3 >
	struct is_idempotent< operators::any_or< D1, D2, D3 >, void > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3 >
	struct is_idempotent< operators::logical_or< D1, D2, D3 >, void > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3 >
	struct is_idempotent< operators::logical_and< D1, D2, D3 >, void > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3 >
	struct is_idempotent< operators::relu< D1, D2, D3 >, void > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3 >
	struct is_idempotent< operators::left_assign_if< D1, D2, D3 >, void > {
		static const constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3 >
	struct is_idempotent< operators::right_assign_if< D1, D2, D3 >, void > {
		static const constexpr bool value = true;
	};

	template< typename IType, typename VType >
	struct is_idempotent< operators::argmin< IType, VType >, void > {
		static const constexpr bool value = true;
	};

	template< typename IType, typename VType >
	struct is_idempotent< operators::argmax< IType, VType >, void > {
		static const constexpr bool value = true;
	};

	template< typename OP >
	struct is_associative<
		OP,
		typename std::enable_if< is_operator< OP >::value, void >::type
	> {
		static constexpr const bool value = OP::is_associative();
	};

	template< typename OP >
	struct is_commutative<
		OP,
		typename std::enable_if< is_operator< OP >::value, void >::type
	> {
		static constexpr const bool value = OP::is_commutative();
	};

	// internal type traits follow

	namespace internal {

		template< typename D1, typename D2, typename D3, enum Backend implementation >
		struct maybe_noop< operators::left_assign_if< D1, D2, D3, implementation > > {
			static const constexpr bool value = true;
		};

		template< typename D1, typename D2, typename D3, enum Backend implementation >
		struct maybe_noop< operators::right_assign_if< D1, D2, D3, implementation > > {
			static const constexpr bool value = true;
		};

	} // end namespace grb::internal

} // end namespace grb

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
namespace std {
	template< typename U, typename V >
	std::ostream & operator<<( std::ostream &out, const std::pair< U, V > &pair ) {
		out << "( " << pair.first << ", " << pair.second << " )";
		return out;
	}
} // end namespace std
 #endif
#endif

#endif // end ``_H_GRB_OPERATORS''

