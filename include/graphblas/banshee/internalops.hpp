
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
 */

#ifndef _H_GRB_BANSHEE_INTERNAL_OPERATORS
#define _H_GRB_BANSHEE_INTERNAL_OPERATORS

#include <snrt.h>

#include <graphblas/base/internalops.hpp>
#include <graphblas/config.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/utils.hpp>

namespace grb {

	namespace operators {

		/** Core implementations of the standard operators in #grb::operators. */
		namespace internal {

			/**
			 * Standard additive operator.
			 *
			 * Assumes native availability of + on the given data types or assumes that
			 * the relevant operators are properly overloaded.
			 *
			 * Assumes that the + operator is associative \em and commutative when
			 * assuming perfect arithmetic and equal data types for \a IN1, \a IN2, and
			 * \a OUT.
			 *
			 * Non-standard or non-matching data types or non-standard (overloaded) +
			 * operators, should therefore be used with caution and may necessitate an
			 * explicit definition as a GraphBLAS operator with the #is_associative and
			 * #is_commutative fields, and others, set as required.
			 *
			 * @tparam IN1 The left-hand input data type.
			 * @tparam IN2 The right-hand input data type.
			 * @tparam OUT The output data type.
			 */
			// [Example Base Operator Implementation]
			template< typename IN1, typename IN2, typename OUT >
			class add< IN1, IN2, OUT, banshee_ssr > {

			public:
				/** Alias to the left-hand input data type. */
				typedef IN1 left_type;

				/** Alias to the right-hand input data type. */
				typedef IN2 right_type;

				/** Alias to the output data type. */
				typedef OUT result_type;

				/** Whether this operator has an inplace foldl. */
				static constexpr bool has_foldl = true;

				/** Whether this operator has an inplace foldr. */
				static constexpr bool has_foldr = true;

				/**
				 * Whether this operator is \em mathematically associative; that is,
				 * associative when assuming equivalent data types for \a IN1, \a IN2,
				 * and \a OUT, as well as assuming exact arithmetic, no overflows, etc.
				 */
				static constexpr bool is_associative = true;

				/**
				 * Whether this operator is \em mathematically commutative; that is,
				 * commutative when assuming equivalent data types for \a IN1, \a IN2,
				 * and \a OUT, as well as assuming exact arithmetic, no overflows, etc.
				 */
				static constexpr bool is_commutative = true;

				/**
				 * Out-of-place application of the addition c = a + b.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 * @param[out] c The output. Must be pre-allocated.
				 */
				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					(void)a;
					(void)b;
					asm volatile( "fadd.d %[result], ft0, ft1" : [result] "=f"( *c ) : : "ft0", "ft1" );
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					(void)a;
					asm volatile( "fadd.d %[result], ft0, %[result]" : [result] "+f"( *c ) : : "ft0", "ft1" );
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					(void)b;
					asm volatile( "fadd.d %[result], %[result], ft1" : [result] "+f"( *c ) : : "ft0", "ft1" );
				}
			};

			/** \todo add documentation */
			template< typename IN1, typename IN2, typename OUT >
			class subtract< IN1, IN2, OUT, banshee_ssr > {
			public:
				/** Alias to the left-hand input data type. */
				typedef IN1 left_type;

				/** Alias to the right-hand input data type. */
				typedef IN2 right_type;

				/** Alias to the output data type. */
				typedef OUT result_type;

				/** Whether this operator has an in-place foldl. */
				static constexpr bool has_foldl = true;

				/** Whether this operator has an in-place foldr. */
				static constexpr bool has_foldr = true;

				/**
				 * Whether this operator is \em mathematically associative; that is,
				 * associative when assuming equivalent data types for \a IN1, \a IN2,
				 * and \a OUT, as well as assuming exact arithmetic, no overflows, etc.
				 */
				static constexpr bool is_associative = false;

				/**
				 * Whether this operator is \em mathematically commutative; that is,
				 * commutative when assuming equivalent data types for \a IN1, \a IN2,
				 * and \a OUT, as well as assuming exact arithmetic, no overflows, etc.
				 */
				static constexpr bool is_commutative = false;

				/**
				 * Out-of-place application of this operator.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 * @param[out] c The output. Must be pre-allocated.
				 *
				 * At the end of the operation, \f$ c = \min\{a,b\} \f$.
				 */
				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					(void)a;
					(void)b;
					asm volatile( "fsub.d %[result], ft0, ft1" : [result] "=f"( *c ) : : "ft0", "ft1" );
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					*c = *a - *c;
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					*c -= *b;
				}
			};
			/**
			 * Standard multiplicative operator.
			 *
			 * Assumes native availability * on the given data types, or assumes
			 * the relevant operators are properly overloaded.
			 *
			 * Assumes that the * operator is associative \em and commutative when
			 * assuming perfect arithmetic and equal data types for \a IN1, \a IN2, and
			 * \a OUT.
			 *
			 * Non-standard or non-matching data types or non-standard (overloaded) *
			 * operators, should therefore be used with caution and may necessitate an
			 * explicit definition as a GraphBLAS operator with the #is_associative and
			 * #is_commutative fields, and others, set as required.
			 *
			 * @tparam IN1 The left-hand input data type.
			 * @tparam IN2 The right-hand input data type.
			 * @tparam OUT The output data type.
			 */
			template< typename IN1, typename IN2, typename OUT >
			class mul< IN1, IN2, OUT, banshee_ssr > {

			public:
				/** Alias to the left-hand input data type. */
				typedef IN1 left_type;

				/** Alias to the right-hand input data type. */
				typedef IN2 right_type;

				/** Alias to the output data type. */
				typedef OUT result_type;

				/** Whether this operator has an in-place foldl. */
				static constexpr bool has_foldl = true;

				/** Whether this operator has an in-place foldr. */
				static constexpr bool has_foldr = true;

				/**
				 * Whether this operator is \em mathematically associative; that is,
				 * associative when assuming equivalent data types for \a IN1, \a IN2,
				 * and \a OUT, as well as assuming exact arithmetic, no overflows, etc.
				 */
				static constexpr bool is_associative = true;

				/**
				 * Whether this operator is \em mathematically commutative; that is,
				 * commutative when assuming equivalent data types for \a IN1, \a IN2,
				 * and \a OUT, as well as assuming exact arithmetic, no overflows, etc.
				 */
				static constexpr bool is_commutative = true;

				/**
				 * Out-of-place application of the multiplication c = a * b.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 * @param[out] c The output. Must be pre-allocated.
				 */
				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					(void)a;
					(void)b;
					asm volatile( "fmul.d %[result], ft0, ft1" : [result] "=f"( *c ) : : "ft0", "ft1" );
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					(void)a;
					asm volatile( "fmul.d %[result], ft0, %[result]" : [result] "+f"( *c ) : : "ft0", "ft1" );
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					(void)b;
					asm volatile( "fadd.d %[result], %[result], ft1" : [result] "+f"( *c ) : : "ft0", "ft1" );
				}
			};

			/**
			 * Left-sided operator that combines an indicator and an identity function
			 * as follows:
			 *
			 * \f$ z = x \odot y = x \text{ if } y \text{evaluates true}. \f$
			 *
			 * If \f$ x \f$ does not evaluate true the operator shall have no effect.
			 */
			template< typename D1, typename D2, typename D3 >
			class left_assign_if< D1, D2, D3, banshee_ssr > {

			public:
				/** Alias to the left-hand input data type. */
				typedef D1 left_type;

				/** Alias to the right-hand input data type. */
				typedef D2 right_type;

				/** Alias to the output data type. */
				typedef D3 result_type;

				/** Whether this operator has an inplace foldl. */
				static constexpr bool has_foldl = true;

				/** Whether this operator has an inplace foldr. */
				static constexpr bool has_foldr = true;

				/**
				 * Whether this operator is \em mathematically associative; that is,
				 * associative when assuming equivalent data types for \a IN1, \a IN2,
				 * and \a OUT, as well as assuming exact arithmetic, no overflows, etc.
				 */
				static constexpr bool is_associative = true;

				/**
				 * Whether this operator is \em mathematically commutative; that is,
				 * commutative when assuming equivalent data types for \a IN1, \a IN2,
				 * and \a OUT, as well as assuming exact arithmetic, no overflows, etc.
				 */
				static constexpr bool is_commutative = true;

				/**
				 * Out-of-place application of the addition c = a.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 * @param[out] c The output. Must be pre-allocated.
				 */
				static void apply( const D1 * __restrict__ const a, const D2 * __restrict__ const b, D3 * __restrict__ const c ) {
					(void)a;
					(void)b;
					asm volatile( "fmv.d %[result], ft0" : [result] "=f"( *c ) : : "ft0", "ft1" );
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const D1 * __restrict__ const a, D3 * __restrict__ const c ) {
					(void)a;
					asm volatile( "fmv.d %[result], ft0" : [result] "=f"( *c ) : : "ft0", "ft1" );
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( D3 * __restrict__ const c, const D2 * __restrict__ const b ) {
					(void)b;
					asm volatile( "fmv.d %[result], ft1" : [result] "=f"( *c ) : : "ft0", "ft1" );
				}
			};

			/** \todo add documentation */
			template< typename IN1, typename IN2, typename OUT >
			class logical_or< IN1, IN2, OUT, banshee_ssr > {
			public:
				/** Alias to the left-hand input data type. */
				typedef IN1 left_type;

				/** Alias to the right-hand input data type. */
				typedef IN2 right_type;

				/** Alias to the output data type. */
				typedef OUT result_type;

				/** Whether this operator has an in-place foldl. */
				static constexpr bool has_foldl = true;

				/** Whether this operator has an in-place foldr. */
				static constexpr bool has_foldr = true;

				/**
				 * Whether this operator is \em mathematically associative; that is,
				 * associative when assuming equivalent data types for \a IN1, \a IN2,
				 * and \a OUT, as well as assuming exact arithmetic, no overflows, etc.
				 */
				static constexpr bool is_associative = true;

				/**
				 * Whether this operator is \em mathematically commutative; that is,
				 * commutative when assuming equivalent data types for \a IN1, \a IN2,
				 * and \a OUT, as well as assuming exact arithmetic, no overflows, etc.
				 */
				static constexpr bool is_commutative = true;

				/**
				 * Out-of-place application of this operator.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 * @param[out] c The output. Must be pre-allocated.
				 *
				 * At the end of the operation, \f$ c = \min\{a,b\} \f$.
				 */
				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					(void)a;
					(void)b;

					asm volatile( "fadd.d %[result], ft0, ft1" : [result] "=f"( *c ) : : "ft0", "ft1" );
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					(void)a;

					asm volatile( "fadd.d %[result], ft0, %[result]" : [result] "+f"( *c ) : : "ft0", "ft1" );
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					(void)b;

					asm volatile( "fadd.d %[result], %[result], ft0" : [result] "+f"( *c ) : : "ft0", "ft1" );
				}
			};

			/** \todo add documentation */
			template< typename IN1, typename IN2, typename OUT >
			class logical_and< IN1, IN2, OUT, banshee_ssr > {
			public:
				/** Alias to the left-hand input data type. */
				typedef IN1 left_type;

				/** Alias to the right-hand input data type. */
				typedef IN2 right_type;

				/** Alias to the output data type. */
				typedef OUT result_type;

				/** Whether this operator has an in-place foldl. */
				static constexpr bool has_foldl = true;

				/** Whether this operator has an in-place foldr. */
				static constexpr bool has_foldr = true;

				/**
				 * Whether this operator is \em mathematically associative; that is,
				 * associative when assuming equivalent data types for \a IN1, \a IN2,
				 * and \a OUT, as well as assuming exact arithmetic, no overflows, etc.
				 */
				static constexpr bool is_associative = true;

				/**
				 * Whether this operator is \em mathematically commutative; that is,
				 * commutative when assuming equivalent data types for \a IN1, \a IN2,
				 * and \a OUT, as well as assuming exact arithmetic, no overflows, etc.
				 */
				static constexpr bool is_commutative = true;

				/**
				 * Out-of-place application of this operator.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 * @param[out] c The output. Must be pre-allocated.
				 *
				 * At the end of the operation, \f$ c = \min\{a,b\} \f$.
				 */
				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					(void)a;
					(void)b;

					asm volatile( "fmul.d %[result], ft0, ft1" : [result] "=f"( *c ) : : "ft0", "ft1" );
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					(void)a;

					asm volatile( "fmul.d %[result], ft0, %[result]" : [result] "+f"( *c ) : : "ft0", "ft1" );
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					(void)b;

					asm volatile( "fmul.d %[result], ft1, %[result]" : [result] "+f"( *c ) : : "ft0", "ft1" );
				}
			};

			/**
			 * This class takes a generic operator implementation and exposes a more
			 * convenient apply() function based on it. This function allows arbitrary
			 * data types being passed as parameters, and automatically handles any
			 * casting required for the raw operator.
			 *
			 * @tparam OP The generic operator implementation.
			 *
			 * @see Operator for full details.
			 */
			template< typename OP >
			class OperatorBase< OP, banshee_ssr > {

			protected:
				/** The left-hand side input domain. */
				typedef typename OP::left_type D1;

				/** The right-hand side input domain. */
				typedef typename OP::right_type D2;

				/** The output domain. */
				typedef typename OP::result_type D3;

			public:
				/** @return Whether this operator is mathematically associative. */
				static constexpr bool is_associative() {
					return OP::is_associative;
				}

				/** @return Whether this operator is mathematically commutative. */
				static constexpr bool is_commutative() {
					return OP::is_commutative;
				}

				/**
				 * This is the high-performance version of apply() in the sense that no
				 * casting is required. This version will be automatically caled whenever
				 * possible.
				 */
				static void apply( const D1 & x, const D2 & y, D3 & out ) {
					OP::apply( &x, &y, &out );
				}
			};

			/**
			 * This is the operator interface exposed to the GraphBLAS implementation.
			 *
			 * \warning Note that most GraphBLAS usage requires associative operators.
			 *          While very easily possible to create non-associative operators
			 *          using this interface, passing them to GraphBLAS functions,
			 *          either explicitly or indirectly (by, e.g., including them in a
			 *          grb::Monoid or grb::Semiring), will lead to undefined
			 *          behaviour.
			 *
			 * This class wraps around a base operator of type \a OP we denote by
			 *        \f$ \odot:\ D_1\times D_2 \to D_3 \f$.
			 *
			 * \parblock
			 * \par Base Operators
			 *
			 * The class \a OP is expected to define the following public function:
			 *   - \a apply, which takes three pointers to parameters \f$ x \in D_1 \f$
			 *      \f$ y \in D_2 \f$, and \f$ z \in D_3 \f$ and computes
			 *      \f$ z = x \odot y \f$.
			 *
			 * It is also expected to define the following types:
			 *   - \a left_type, which corresponds to \f$ D_1 \f$,
			 *   - \a right_type, which corresponds to \f$ D_2 \f$,
			 *   - \a result_type, which corresponds to \f$ D_3 \f$.
			 *
			 * It is also expected to define the following two public boolean fields:
			 *   - \a has_foldr
			 *   - \a has_foldl
			 *
			 * If \a has_foldr is \a true, then the class \a OP is expected to also
			 * define the function
			 *   - foldr, which takes two pointers to parameters \f$ x \in D_1 \f$
			 *      and \f$ z \in D_2 \subseteq D_3 \f$ and stores in \a z the result of
			 *      \f$ x \odot z \f$.
			 *
			 * If \a has_foldl is \a true, the the class \a OP is expected to also
			 * define the function
			 *   - foldl, which takes two pointers to parameters
			 *      \f$ z \in D_1 \subseteq D_3 \f$ and \f$ y \in D_2 \f$ and stores in
			 *      \a z the result of \f$ z \odot y \f$.
			 *
			 * For examples of these base operators, see grb::operators::internal::max
			 * or grb::operators::internal::mul. An example of a full implementation,
			 * in this case for numerical addition, is the following:
			 *
			 * \snippet internalops.hpp Example Base Operator Implementation
			 *
			 * \note GraphBLAS users should never call these functions directly. This
			 *       documentation is provided for developers to understand or extend
			 *       the current implementation, for example to include new operators.
			 *
			 * \warning When calling these functions directly, note that the pointers
			 *          to the memory areas are declared using the \em restrict key
			 *          word. One of the consequences is that all pointers given in a
			 *          single call <em>may never refer to the same memory area, or
			 *          undefined behaviour is invoked</em>.
			 *
			 * \endparblock
			 *
			 * \parblock
			 * \par The exposed GraphBLAS Operator Interface
			 *
			 * The Base Operators as illustrated above are wrapped by this class to
			 * provide a more convient API. It translates the functionality of any Base
			 * Operator and exposes the following interface instead:
			 *
			 *   -# apply, which takes three parameters \f$ x, y, z \f$ of arbitrary
			 *      types and computes \f$ z = x \odot y \f$ after performing any
			 *      casting if required.
			 *   -# foldr, which takes two parameters \f$ x, z \f$ of arbitrary types
			 *      and computes \f$ z = x \odot z \f$ after performing any casting if
			 *      required.
			 *   -# foldl, which takes two parameters \f$ z, y \f$ of arbitrary types
			 *      and computes \f$ z = z \odot y \f$ after performing any casting if
			 *      required.
			 *   -# eWiseApply, which takes three pointers to arrays \f$ x, y, z \f$
			 *      and a size \a n. The arrays can correspond to elements of any type,
			 *      all three with length at least \a n. For every i-th element of the
			 *      three arrays, on the values \f$ x_i, y_i, z_i \f$, \f$ z_i \f$ will
			 *      be set to \f$ x_i \odot y_i \f$.
			 *   -# foldrArray, which takes a pointer to an array \f$ x \f$, a
			 *      parameter \f$ z \f$ of arbitrary type, and a size \n as parameters.
			 *      The value \f$ z \f$ will be overwritten to \f$ x_i \odot z \f$ for
			 *      each of the \f$ i \in \{ 0, 1, \ldots, n-1 \} \f$. The order of
			 *      application, in the sense of which \f$ i \f$ are processed first,
			 *      is undefined.
			 *   -# foldlArray, which takes as parameters: \f$ z \f$ of arbitrary type,
			 *      an array \f$ y \f$, and a size \n. The value \f$ z \f$ will be
			 *      overwritten to \f$ z \odot y_i \f$ for each of the
			 *      \f$ i \in \{ 0, 1, \ldots, n-1 \} \f$. The order of application, in
			 *      the sense of which \f$ i \f$ are processed first, is undefined.
			 * \endparblock
			 *
			 * \note This class only allows wrapping of stateless base operators. This
			 *       GraphBLAS implementation in principle allows for stateful
			 *       operators, though they must be provided by a specialised class
			 *       which directly implements the above public interface.
			 *
			 * @see OperatorBase::apply
			 * @see OperatorFR::foldr
			 * @see OperatorFL::foldl
			 * @see \ref OperatorNoFRFLeWiseApply
			 * @see Operator::foldrArray
			 * @see Operator::foldlArray
			 *
			 * \parblock
			 * \par Providing New Operators
			 *
			 * New operators are easily added to this
			 * GraphBLAS implementation by providing a base operator and wrapping this
			 * class around it, as illustrated, e.g., by grb::operators::add as follows:
			 *
			 * \snippet ops.hpp Operator Wrapping
			 *
			 * This need to be compatible with the GraphBLAS type traits, specifically,
			 * the #is_operator template. To ensure this, a specialisation of it must be
			 * privided:
			 *
			 * \snippet ops.hpp Operator Type Traits
			 * \endparblock
			 */
			template< typename OP >
			class Operator< OP, banshee_ssr > : public OperatorBase< OP, banshee_ssr > {

			private:
			public:
				/** The left-hand side input domain of this operator. */
				typedef typename OperatorBase< OP >::D1 D1;

				/** The right-hand side input domain of this operator. */
				typedef typename OperatorBase< OP >::D2 D2;

				/** The output domain of this operator. */
				typedef typename OperatorBase< OP >::D3 D3;

				template< typename InputType1, typename InputType2, typename OutputType >
				static void eWiseApply( const InputType1 * x, const InputType2 * y, OutputType * __restrict__ z, const size_t n ) {
#ifdef _DEBUG
					printf( "In SSR wise apply\n" );
#endif
					register volatile double ft0 asm( "ft0" );
					register volatile double ft1 asm( "ft1" );
					asm volatile( "" : "=f"( ft0 ), "=f"( ft1 ) );

					snrt_ssr_loop_1d( SNRT_SSR_DM0, n, sizeof( InputType1 ) );
					snrt_ssr_loop_1d( SNRT_SSR_DM1, n, sizeof( InputType2 ) );

					snrt_ssr_read( SNRT_SSR_DM0, SNRT_SSR_1D, x );
					snrt_ssr_read( SNRT_SSR_DM1, SNRT_SSR_1D, y );
					snrt_ssr_enable();

					// direct application for remainder
					for( int i = 0; i < n; i++ ) {
						OP::apply( x[ i ], y[ i ], z[ i ] );
					}
					snrt_ssr_disable();
					asm volatile( "" ::"f"( ft0 ), "f"( ft1 ) );
				}

				/**
				 * Reduces a vector of type \a InputType into a value in \a IOType
				 * by repeated application of this operator. The \a IOType is cast
				 * into \a D3 prior reduction. The \a InputType is cast into \a D1
				 * during reduction. The final result is cast to IOType after
				 * reduction. The reduction happens `right-to-left'.
				 *
				 * This implementation relies on the \a foldr, whether it be an
				 * true in-place or emulated version.
				 *
				 * @param[in,out] out On input, the initial value to be used for
				 *                    reduction. On output, all elements of \a x
				 *                    have been applied to \a out.
				 * @param[in] x A vector of size \a n with elements of type \a left_type.
				 * @param[in] n A positive integer (can be 0).
				 */
				template< typename IOType, typename InputType >
				static void foldrArray( const InputType * __restrict__ const x, IOType & out, const size_t n ) {
					(void)x;
					(void)out;
					(void)n;
					return;
				}

				/**
				 * Reduces a vector of type \a InputType into a value in \a IOType
				 * by repeated application of this operator. The \a IOType is cast
				 * into \a D3 prior reduction. The \a InputType is cast into \a D2
				 * during reduction. The final result is cast to IOType after
				 * reduction. The reduction happens `left-to-right'.
				 *
				 * This implementation relies on the \a foldr, whether it be an
				 * true in-place or emulated version.
				 *
				 * @param[in,out] out On input, the initial value to be used for
				 *                    reduction. On output, all elements of \a x
				 *                    have been applied to \a out.
				 * @param[in] x A vector of size \a n with elements of type \a left_type.
				 * @param[in] n A positive integer (can be 0).
				 */
				template< typename IOType, typename InputType >
				static void foldlArray( IOType & out, const InputType * __restrict__ const x, const size_t n ) {
					(void)x;
					(void)out;
					(void)n;
					return;
				}
			};

		} // namespace internal

	} // namespace operators

} // namespace grb

#endif
