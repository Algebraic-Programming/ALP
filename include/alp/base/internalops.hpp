
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

#include "config.hpp"


namespace alp {

	namespace operators {

		/** Core implementations of the standard operators in #alp::operators. */
		namespace internal {

			/**
			 * Standard argmin operator.
			 *
			 * Takes std::pair< index, value > domains only.
			 *
			 * Given two pairs (i1,v1), (i2,v2)
			 *  - returns (i1,v1) if v1<v2, OR
			 *  - returns (i2,v2) otherwise.
			 */
			template< typename IType, typename VType >
			class argmin {

				static_assert( std::is_integral< IType >::value,
					"Argmin operator may only be constructed using integral index "
					"types." );

			public:
				/** Alias to the left-hand input data type. */
				typedef std::pair< IType, VType > left_type;

				/** Alias to the right-hand input data type. */
				typedef std::pair< IType, VType > right_type;

				/** Alias to the output data type. */
				typedef std::pair< IType, VType > result_type;

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
				 * Out-of-place application of the operator.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 * @param[out] c The output. Must be pre-allocated.
				 */
				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					if( a->second < b->second ) {
						c->first = a->first;
						c->second = a->second;
					}
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					if( a->second < c->second ) {
						c->first = a->first;
						c->second = a->second;
					}
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					if( b->second <= c->second ) {
						c->first = b->first;
						c->second = b->second;
					}
				}
			};

			/**
			 * Standard argmax operator.
			 *
			 * Takes std::pair< index, value > domains only.
			 *
			 * Given two pairs (i1,v1), (i2,v2)
			 *  - returns (i1,v1) if v1>v2, OR
			 *  - returns (i2,v2) otherwise.
			 */
			template< typename IType, typename VType >
			class argmax {

				static_assert( std::is_integral< IType >::value,
					"Argmin operator may only be constructed using integral index "
					"types." );

			public:
				/** Alias to the left-hand input data type. */
				typedef std::pair< IType, VType > left_type;

				/** Alias to the right-hand input data type. */
				typedef std::pair< IType, VType > right_type;

				/** Alias to the output data type. */
				typedef std::pair< IType, VType > result_type;

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
				 * Out-of-place application of the operator.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 * @param[out] c The output. Must be pre-allocated.
				 */
				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					if( a->second > b->second ) {
						c->first = a->first;
						c->second = a->second;
					}
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					if( a->second > c->second ) {
						c->first = a->first;
						c->second = a->second;
					}
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					if( b->second >= c->second ) {
						c->first = b->first;
						c->second = b->second;
					}
				}
			};

			/**
			 * Standard left-hand side assignment operator.
			 *
			 * Takes binary input, but ignores the right-hand side input and simply
			 * assigns the left-hand side input to the output variable.
			 *
			 * Assumes native availability of = on the given data types, or assumes
			 * the relevant operators are properly overloaded.
			 *
			 * Assumes a binary operator defined using the =-operator in the following
			 * way, is \em associative:
			 * \code
			 * void left_assign( const IN1 x, const IN2 y, OUT &out ) {
			 *     (void)y;
			 *     out = x;
			 * }
			 * \endcode
			 *
			 * Non-standard or non-matching data types, or non-standard (overloaded) =
			 * operators should be used with caution and may necessitate an explicit
			 * definition as a GraphBLAS operator with the #has_foldl, #has_foldr, and
			 * the other fields, set as required.
			 *
			 * @tparam IN1 The left-hand input data type.
			 * @tparam IN2 The right-hand input data type.
			 * @tparam OUT The output data type.
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class left_assign {

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
				static constexpr bool is_commutative = false;

				/**
				 * Out-of-place application of the addition c = a.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 * @param[out] c The output. Must be pre-allocated.
				 */
				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					(void)b;
					*c = static_cast< result_type >( *a );
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					*c = static_cast< result_type >( *a );
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
					(void)c;
				}
			};

			/**
			 * Standard right-hand side assignment operator.
			 *
			 * Takes binary input, but ignores the right-hand side input and simply
			 * assigns the left-hand side input to the output variable.
			 *
			 * Assumes native availability of = on the given data types, or assumes
			 * the relevant operators are properly overloaded.
			 *
			 * Assumes a binary operator defined using the =-operator in the following
			 * way, is \em associative:
			 * \code
			 * void right_assign( const IN1 x, const IN2 y, OUT &out ) {
			 *     (void)x;
			 *     out = y;
			 * }
			 * \endcode
			 *
			 * Non-standard or non-matching data types, or non-standard (overloaded) =
			 * operators should be used with caution and may necessitate an explicit
			 * definition as a GraphBLAS operator with the #has_foldl, #has_foldr, and
			 * the other fields, set as required.
			 *
			 * @tparam IN1 The left-hand input data type.
			 * @tparam IN2 The right-hand input data type.
			 * @tparam OUT The output data type.
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class right_assign {

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
				static constexpr bool is_commutative = false;

				/**
				 * Out-of-place application of the addition c = a.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 * @param[out] c The output. Must be pre-allocated.
				 */
				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					(void)a;
					*c = *b;
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
					(void)c;
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					*c = static_cast< result_type >( *b );
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
			template< typename D1, typename D2, typename D3, enum Backend implementation = config::default_backend >
			class left_assign_if {

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
					if( static_cast< const bool >( *b ) ) {
						*c = *a;
					}
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const D1 * __restrict__ const a, D3 * __restrict__ const c ) {
					if( static_cast< const bool >( *c ) ) {
						*c = static_cast< D3 >( *a );
					}
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( D3 * __restrict__ const c, const D2 * __restrict__ const b ) {
					if( static_cast< bool >( *b ) ) {
						*c = static_cast< D3 >( static_cast< D1 >( *c ) );
					}
				}
			};

			/**
			 * Right-sided operator that combines an indicator and an identity function
			 * as follows:
			 *
			 * \f$ z = x \odot y = y \text{ if } x \text{evaluates true}. \f$
			 *
			 * If \f$ x \f$ does not evaluate true the operator shall have no effect.
			 */
			template< typename D1, typename D2, typename D3, enum Backend implementation = config::default_backend >
			class right_assign_if {

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
					if( static_cast< const bool >( *a ) ) {
						*c = *b;
					}
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const D1 * __restrict__ const a, D3 * __restrict__ const c ) {
					if( static_cast< const bool >( *a ) ) {
						*c = static_cast< D3 >( static_cast< D2 >( *c ) );
					}
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( D3 * __restrict__ const c, const D2 * __restrict__ const b ) {
					if( static_cast< bool >( *c ) ) {
						*c = static_cast< D3 >( *b );
					}
				}
			};

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
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class add {

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
				 *
				 * \warning Passing invalid pointers will result in UB.
				 */
				static void apply( const left_type * __restrict__ const a,
					const right_type * __restrict__ const b,
					result_type * __restrict__ const c
				) {
					ALP_UTIL_IGNORE_MAYBE_UNINITIALIZED // this is a (too) broad suppression--
					                                    // see internal issue 306 for rationale
					*c = *a + *b;
					ALP_UTIL_RESTORE_WARNINGS
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 *
				 * \warning Passing invalid pointers will result in UB.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					*c += *a;
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 *
				 * \warning Passing invalid pointers will result in UB.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					*c += *b;
				}
			};
			// [Example Base Operator Implementation]

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
			template<
				typename IN1, typename IN2, typename OUT,
				enum Backend implementation = config::default_backend
			>
			class mul {

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
				 * @param[in]  a Pointer to the left-hand side input. Must be initialised.
				 * @param[in]  b Pointer to the right-hand side input. Must be initialised.
				 * @param[out] c Pointer to where to compute the output.
				 *
				 * \warning All pointers must be valid or UB occurs.
				 */
				static void apply(
						const left_type * __restrict__ const a,
						const right_type * __restrict__ const b,
						result_type * __restrict__ const c
				) {
					ALP_UTIL_IGNORE_MAYBE_UNINITIALIZED // this is a (too) broad suppression--
					                                    // see internal issue 306 for rationale
					*c = *a * *b;
					ALP_UTIL_RESTORE_WARNINGS
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					*c *= *a;
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					*c *= *b;
				}
			};

			/**
			 * Standard max operator.
			 *
			 * Assumes native availability of < on the given data types, or assumes
			 * the relevant operators are properly overloaded.
			 *
			 * Non-standard or non-matching data types, or non-standard (overloaded) <
			 * operators, should be used with caution and may necessitate an explicit
			 * definition as a GraphBLAS operator with the #is_associative and
			 * #is_commutative fields, and others, set as required.
			 *
			 * @tparam IN1 The left-hand input data type.
			 * @tparam IN2 The right-hand input data type.
			 * @tparam OUT The output data type.
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class max {
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
				 * Out-of-place application of the max operator.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 * @param[out] c The output. Must be pre-allocated.
				 *
				 * At the end of the operation, \f$ c = \max\{a,b\} \f$.
				 */
				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					if( *a < *b ) {
						*c = static_cast< OUT >( *b );
					} else {
						*c = static_cast< OUT >( *a );
					}
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					if( *a > *c ) {
						*c = *a;
					}
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					if( *b > *c ) {
						*c = *b;
					}
				}
			};

			/**
			 * Standard min operator.
			 *
			 * Assumes native availability of > on the given data types, or assumes
			 * the relevant operators are properly overloaded.
			 *
			 * Non-standard or non-matching data types, or non-standard (overloaded) >
			 * operators, should be used with caution and may necessitate an explicit
			 * definition as a GraphBLAS operator with the #is_associative and
			 * #is_commutative fields, and others, set as required.
			 *
			 * @tparam IN1 The left-hand input data type.
			 * @tparam IN2 The right-hand input data type.
			 * @tparam OUT The output data type.
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class min {
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
				 * Out-of-place application of the min operator.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 * @param[out] c The output. Must be pre-allocated.
				 *
				 * At the end of the operation, \f$ c = \min\{a,b\} \f$.
				 */
				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					if( *a > *b ) {
						*c = static_cast< OUT >( *b );
					} else {
						*c = static_cast< OUT >( *a );
					}
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					if( *a < *c ) {
						*c = *a;
					}
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					if( *b < *c ) {
						*c = *b;
					}
				}
			};

			/** \todo add documentation */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class substract {
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
					*c = *a - *b;
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

			/** \todo add documentation */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class divide {
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
				 * At the end of the operation, \f$ c = a/b \f$.
				 */
				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					*c = *a / *b;
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					*c = *a / *c;
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					*c /= *b;
				}
			};

			/** \todo add documentation */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class divide_reverse {
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
				 * At the end of the operation, \f$ c = b/a \f$.
				 */
				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					*c = *b / *a;
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					*c /= *a;
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					*c = *b / *c;
				}
			};

			/** \todo add documentation */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class equal {
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
					if( *a == *b ) {
						*c = static_cast< OUT >( true );
					} else {
						*c = static_cast< OUT >( false );
					}
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					if( *a == *c ) {
						*c = static_cast< result_type >( true );
					} else {
						*c = static_cast< result_type >( false );
					}
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					if( *b == *c ) {
						*c = static_cast< result_type >( true );
					} else {
						*c = static_cast< result_type >( false );
					}
				}
			};

			/** \todo add documentation */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class not_equal {
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
					ALP_UTIL_IGNORE_MAYBE_UNINITIALIZED // this is a (too) broad suppression--
					                                    // see internal issue 306 for rationale
					if( *a != *b ) {
						*c = static_cast< OUT >( true );
					} else {
						*c = static_cast< OUT >( false );
					}
					ALP_UTIL_RESTORE_WARNINGS
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					if( *a != *c ) {
						*c = static_cast< result_type >( true );
					} else {
						*c = static_cast< result_type >( false );
					}
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					if( *b != *c ) {
						*c = static_cast< result_type >( true );
					} else {
						*c = static_cast< result_type >( false );
					}
				}
			};

			/** \todo add documentation */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class any_or {
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
					if( *a ) {
						*c = static_cast< OUT >( *a );
					} else if( *b ) {
						*c = static_cast< OUT >( *b );
					} else {
						assert( ! ( *a ) );
						*c = static_cast< OUT >( *a );
					}
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					if( *a ) {
						*c = static_cast< result_type >( *a );
					}
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					if( *b ) {
						*c = static_cast< result_type >( *b );
					}
				}
			};

			/** \todo add documentation */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class logical_or {
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
					if( *a || *b ) {
						*c = static_cast< OUT >( true );
					} else {
						*c = static_cast< OUT >( false );
					}
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					if( *a || *c ) {
						*c = static_cast< result_type >( true );
					} else {
						*c = static_cast< result_type >( false );
					}
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					if( *b || *c ) {
						*c = static_cast< result_type >( true );
					} else {
						*c = static_cast< result_type >( false );
					}
				}
			};

			/** \todo add documentation */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class logical_and {
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
					if( *a && *b ) {
						*c = static_cast< OUT >( true );
					} else {
						*c = static_cast< OUT >( false );
					}
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					if( *a && *c ) {
						*c = static_cast< result_type >( true );
					} else {
						*c = static_cast< result_type >( false );
					}
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					if( *b && *c ) {
						*c = static_cast< result_type >( true );
					} else {
						*c = static_cast< result_type >( false );
					}
				}
			};

			/** \todo add documentation */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class abs_diff {

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
					if( *a < *b ) {
						*c = static_cast< OUT >( *b - *a );
					} else {
						*c = static_cast< OUT >( *a - *b );
					}
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					if( *a < *c ) {
						*c -= *a;
					} else {
						*c = static_cast< OUT >( *a - *c );
					}
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					if( *b < *c ) {
						*c -= *b;
					} else {
						*c = static_cast< OUT >( *b - *c );
					}
				}
			};

			/** \todo add documentation */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class relu {
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
				 * At the end of the operation, \f$ c = ReLU\{a,b\} = \begin{cases}
				 *   a \text{, if } a>b \\
				 *   b \text{, otherwise}
				 * \end{cases}\f$.
				 */
				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					if( *a < *b ) {
						*c = static_cast< OUT >( *b );
					} else {
						*c = static_cast< OUT >( *a );
					}
				}

				/**
				 * In-place left-to-right folding.
				 *
				 * @param[in]     a Pointer to the left-hand side input data.
				 * @param[in,out] c Pointer to the right-hand side input data. This also
				 *                  dubs as the output memory area.
				 */
				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					if( *a > *c ) {
						*c = *a;
					}
				}

				/**
				 * In-place right-to-left folding.
				 *
				 * @param[in,out] c Pointer to the left-hand side input data. This also
				 *                  dubs as the output memory area.
				 * @param[in]     b Pointer to the right-hand side input data.
				 */
				static void foldl( result_type * __restrict__ const c, const right_type * __restrict__ const b ) {
					if( *b > *c ) {
						*c = *b;
					}
				}
			};

			template< typename D1, typename D2, typename D3, enum Backend implementation = config::default_backend >
			class square_diff {
			public:
				typedef D1 left_type;
				typedef D2 right_type;
				typedef D3 result_type;

				static constexpr bool has_foldl = true;
				static constexpr bool has_foldr = true;
				static constexpr bool is_associative = false;
				static constexpr bool is_commutative = true;

				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					*c = ( *a - *b ) * ( *a - *b );
				}

				static void foldr( const left_type * __restrict__ const a, result_type * __restrict__ const c ) {
					*c = ( *a - *c ) * ( *a - *c );
				}

				static void foldl( const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					*c = ( *c - *b ) * ( *c - *b );
				}
			};

			/**
			 * left operand of type IN1,
			 * right operand of type IN2
			 * result of type std::pair< IN1, IN2 >
			 *
			 * for use together with argmin
			 */
			template< typename IN1, typename IN2, enum Backend implementation = config::default_backend >
			class zip {
			public:
				typedef IN1 left_type;
				typedef IN2 right_type;
				typedef std::pair< IN1, IN2 > result_type;

				static constexpr bool has_foldl = false;
				static constexpr bool has_foldr = false;
				static constexpr bool is_associative = false;
				static constexpr bool is_commutative = false;

				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					*c = std::make_pair( *a, *b );
				}
			};

			/**
			 * compares the first argument of a pair
			 */
			template< typename IN1, typename IN2, typename OUT, enum Backend implementation = config::default_backend >
			class equal_first {
			public:
				typedef IN1 left_type;

				typedef IN2 right_type;

				typedef OUT result_type;

				static constexpr bool has_foldl = false;
				static constexpr bool has_foldr = false;
				static constexpr bool is_associative = false;
				static constexpr bool is_commutative = false;

				/**
				 * Out-of-place application of this operator.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 * @param[out] c The output. Must be pre-allocated.
				 *
				 * At the end of the operation, \f$ c = a->first == b->first \f$.
				 */
				static void apply( const left_type * __restrict__ const a, const right_type * __restrict__ const b, result_type * __restrict__ const c ) {
					if( a->first == b->first ) {
						*c = static_cast< OUT >( true );
					} else {
						*c = static_cast< OUT >( false );
					}
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
			template< typename OP, enum Backend implementation = config::default_backend >
			class OperatorBase {

			protected:
				/** The block size that should be used during map-like operations. */
				static constexpr size_t blocksize = alp::utils::static_min( alp::config::SIMD_BLOCKSIZE< typename OP::left_type >::value(),
					alp::utils::static_min( alp::config::SIMD_BLOCKSIZE< typename OP::right_type >::value(), alp::config::SIMD_BLOCKSIZE< typename OP::result_type >::value() ) );

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
				 * Straightforward application of this operator. Computes \f$ x \odot y \f$
				 * and stores the result in \a z.
				 *
				 * @tparam InputType1 The type of the input parameter \a x.
				 * @tparam InputType2 The type of the input parameter \a y.
				 * @tparam OutputType The type of the output parameter \a z.
				 *
				 * \warning If \a InputType1 does not match \a D! \em or \a InputType2 does
				 *          not match \a D2 \em or \a OutputType does not match \a D3, then
				 *          the input will be cast into temporary variables of the correct
				 *          types, while the output will be cast from a temporary variable,
				 *
				 * \note Best performance is thus only guaranteed when all domains match.
				 *
				 * @param[in]  x The left-hand side input.
				 * @param[in]  y The right-hand side input.
				 * @param[out] z The output element.
				 */
				template< typename InputType1, typename InputType2, typename OutputType >
				static void apply( const InputType1 & x, const InputType2 & y, OutputType & z ) {
					const D1 a = static_cast< D1 >( x );
					const D2 b = static_cast< D2 >( y );
					D3 temp;
					OP::apply( &a, &b, &temp );
					z = static_cast< OutputType >( temp );
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
			 * A class capable of adding an out-of-place \a foldr function for an
			 * operator that is not fold-right capable, or capable of adding an in-
			 * place foldr function for an operator that is fold-right capable. For
			 * fold-right capable operators, this class is also capable of adding
			 * an efficient eWiseApply function.
			 *
			 * An operator is fold-right capable when the Base Operator \a OP
			 * provides an in-place foldr implementation, \em and whenever \a D1
			 * equals \a D3. If one of either requirements is not met, then \a OP
			 * is not fold-right capable and this class is selected to add an out-
			 * of-place foldr function.
			 *
			 * @tparam OP    The generic operator implementation.
			 * @tparam guard This typename is void if and only if \a OP is not fold-
			 *               right capable. In this case, this class adds an out-of-
			 *               place foldr implementation to the operator.
			 *               If it is not void, then this class defines an
			 *               in-place foldr implementation instead.
			 *
			 * \note This specific class corresponds to the \a guard variable equal to
			 *       \a void.
			 *
			 * @see Operator for full details.
			 * @see OperatorBase for additional functions exposed to the final operator.
			 */
			template< typename OP, typename guard = void, enum Backend implementation = config::default_backend >
			class OperatorFR : public OperatorBase< OP > {

			public:
				/**
				 * Emulated in-place application of this operator on two data elements.
				 *
				 * Computes \f$ x \odot y \f$ and writes the result into \f$ y \f$.
				 *
				 * We wish to call this in-place variant internally for brevity. However,
				 * if \a OP has no in-place variant, then we must cache the previous
				 * value of the output element or otherwise we will breach the
				 * __restrict__ contract of OP::apply.
				 * The caller must ensure the appropriate domains and casting behaviour
				 * is applicable. Note that a user is never to call these functions
				 * explicitly.
				 *
				 * @tparam InputType The type of the parameter \a x.
				 * @tparam IOType    The type of the parameter \a y.
				 *
				 * \warning Additional casting and use of temporary variables may occur
				 *          when \a InputType does not match \a D1 \em or \a IOType
				 *          does not match \a D3.
				 *
				 * \note This implementation relies on apply().
				 *
				 * @param[in]     x The value that is to be applied to \a y.
				 * @param[in,out] y The value \a x is to be applied against.
				 */
				template< typename InputType, typename IOType >
				static void foldr( const InputType & x, IOType & y ) {
					typedef typename OperatorBase< OP >::D2 D2;
					const D2 cache = static_cast< D2 >( y );
					OperatorBase< OP >::apply( x, cache, y );
				}

				/**
				 * Out-of-place element-wise foldr function. Calculates
				 * \f$\forall\ i \in \{ 0, 1, \ldots, n - 1 \}, \f$
				 * \f$ x_i \odot z_i \f$ and stores the result into
				 * \f$ z_i \f$.
				 *
				 * @tparam InputType The type of elements in \a x.
				 * @tparam IOType    The type of elements in \a z.
				 *
				 * @param x The left-hand side input data.
				 * @param z Where \a x shall be mapped into.
				 * @param n How many data elements \a x and \a z contain.
				 *
				 * This version requires three buffers, streams \a x once,
				 * and streams \a z twice (once for reading, once for
				 * writing.
				 */
				template< typename InputType, typename IOType >
				static void eWiseFoldrAA( const InputType * __restrict__ const x, IOType * __restrict__ const z, const size_t n ) {
					// local buffers
					typedef typename OperatorBase< OP >::D1 D1;
					typedef typename OperatorBase< OP >::D2 D2;
					typedef typename OperatorBase< OP >::D3 D3;
					D1 left_buffer[ OperatorBase< OP >::blocksize ];
					D2 right_buffer[ OperatorBase< OP >::blocksize ];
					D3 result_buffer[ OperatorBase< OP >::blocksize ];

					// blockwise application
					size_t i = 0;
					while( i + OperatorBase< OP >::blocksize <= n ) {
						// load into buffers
						for( size_t b = 0; b < OperatorBase< OP >::blocksize; ++i, ++b ) {
							left_buffer[ b ] = static_cast< D1 >( x[ i ] );
							right_buffer[ b ] = static_cast< D2 >( z[ i ] );
						}

						// rewind source and output
						i -= OperatorBase< OP >::blocksize;

						// operate within buffer
						for( size_t b = 0; b < OperatorBase< OP >::blocksize; ++b ) {
							OP::apply( &( left_buffer[ b ] ), &( right_buffer[ b ] ), &( result_buffer[ b ] ) );
						}

						// write back result
						for( size_t b = 0; b < OperatorBase< OP >::blocksize; ++i, ++b ) {
							z[ i ] = static_cast< IOType >( result_buffer[ b ] );
						}
					}

					// direct application for remainder
					for( ; i < n; ++i ) {
						left_buffer[ 0 ] = static_cast< D1 >( x[ i ] );
						right_buffer[ 0 ] = static_cast< D2 >( z[ i ] );
						OP::apply( left_buffer, right_buffer, result_buffer );
						z[ i ] = static_cast< IOType >( result_buffer[ 0 ] );
					}
				}

				/**
				 * Out-of-place element-wise foldr function. Calculates
				 * \f$ \forall\ i \in \{ 0, 1, \ldots, n - 1 \}, \f$
				 * \f$ x \odot z_i \f$ and stores the result into
				 * \f$ z_i \f$.
				 *
				 * @tparam InputType The type of elements in \a x.
				 * @tparam IOType    The type of elements in \a z.
				 *
				 * @param x The left-hand side input value.
				 * @param z Where \a x shall be mapped into.
				 * @param n How many data elements \a z contains.
				 *
				 * This version requires two buffers and streams \a z
				 * twice (once for reading, once for writing).
				 */
				template< typename InputType, typename IOType >
				static void eWiseFoldrSA( const InputType x, IOType * __restrict__ const z, const size_t n ) {
					// local buffers
					typedef typename OperatorBase< OP >::D1 D1;
					typedef typename OperatorBase< OP >::D2 D2;
					typedef typename OperatorBase< OP >::D3 D3;
					const D1 left_buffer = x; // this is actually mandatory in case x is a temporary
					D2 right_buffer[ OperatorBase< OP >::blocksize ];
					D3 result_buffer[ OperatorBase< OP >::blocksize ];

					// blockwise application
					size_t i = 0;
					while( i + OperatorBase< OP >::blocksize <= n ) {
						// load into buffers
						for( size_t b = 0; b < OperatorBase< OP >::blocksize; ++i, ++b ) {
							right_buffer[ b ] = static_cast< D2 >( z[ i ] );
						}

						// rewind source and output
						i -= OperatorBase< OP >::blocksize;

						// operate within buffer
						for( size_t b = 0; b < OperatorBase< OP >::blocksize; ++b ) {
							OP::apply( &left_buffer, &( right_buffer[ b ] ), &( result_buffer[ b ] ) );
						}

						// write back result
						for( size_t b = 0; b < OperatorBase< OP >::blocksize; ++i, ++b ) {
							z[ i ] = static_cast< IOType >( result_buffer[ b ] );
						}
					}

					// direct application for remainder
					for( ; i < n; ++i ) {
						right_buffer[ 0 ] = static_cast< D2 >( z[ i ] );
						OP::apply( &left_buffer, right_buffer, result_buffer );
						z[ i ] = static_cast< IOType >( result_buffer[ 0 ] );
					}
				}
			};

			/**
			 * This class provides an in-place foldr implementation for Base Operators
			 * that are fold-right capable given its provided domains. It also implements
			 * an eWiseApply function that requires two buffers by exploiting the
			 * in-place foldr operator. Without an in-place foldr, it is still possible
			 * to implement an eWiseApply using two buffers if there is an in-place foldl
			 * added via OperatorFL. If that also fails, the eWiseApply function will be
			 * implemented using three buffers via OperatorNoFRFL.
			 *
			 * @tparam OP The generic operator implementation.
			 *
			 * @see Operator for full details.
			 * @see OperatorFR for details on fold-right capable operators and behaviour
			 *                 for non fold-right capable operators.
			 * @see OperatorBase for additional functions exposed to the final operator.
			 */
			template< typename OP >
			class OperatorFR< OP, typename std::enable_if< OP::has_foldr && std::is_same< typename OP::right_type, typename OP::result_type >::value >::type > : public OperatorBase< OP > {

			private:
				typedef typename OperatorBase< OP >::D1 D1;
				typedef typename OperatorBase< OP >::D3 D3;
				static constexpr size_t blocksize = OperatorBase< OP >::blocksize;

			public:
				/**
				 * In-place application of this operator on two data elements.
				 *
				 * Computes \f$ x \odot y \f$ and writes the result into \f$ y \f$.
				 *
				 * \note This variant is only called when the underlying raw operator
				 *       supports in-place operations.
				 *
				 * The caller must ensure the appropriate domains and casting behaviour
				 * is applicable. Note that a user is never to call these functions
				 * explicitly.
				 *
				 * @param[in]     x The value that is to be applied to \a y.
				 * @param[in,out] y The value \a x is to be applied against.
				 */
				static void foldr( const D1 & x, D3 & y ) {
					OP::foldr( &x, &y );
				}

				/**
				 * In-place element-wise foldr function. Calculates
				 * \f$\forall\ i \in \{ 0, 1, \ldots, n - 1 \}, \f$
				 * \f$ x \odot z_i \f$ and stores the result into \f$ z_i \f$.
				 *
				 * @tparam InputType The type of \a x.
				 * @tparam IOType    The type of elements in \a z.
				 *
				 * @param[in]     x The left-hand side input value.
				 * @param[in,out] z Where \a x shall be mapped into.
				 * @param[in]     n How many data elements \a z contains.
				 *
				 * This implementation requires one buffers only. It streams \a z twice,
				 * once for reading, once for writing. This function should vectorise.
				 */
				template< typename InputType, typename IOType >
				static void eWiseFoldrSA( const InputType x, IOType * __restrict__ const z, const size_t n ) {
					// local buffers
					const D1 left_buffer = static_cast< D1 >( x );
					D3 result_buffer[ blocksize ];

					// blockwise application
					size_t i = 0;
					while( i + blocksize <= n ) {
						// load into buffers
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							result_buffer[ b ] = static_cast< D3 >( z[ i ] );
						}

						// rewind source and output
						i -= blocksize;

						// operate within buffer
						for( size_t b = 0; b < blocksize; ++b ) {
							OP::foldr( &left_buffer, &( result_buffer[ b ] ) );
						}

						// write back result
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							z[ i ] = static_cast< IOType >( result_buffer[ b ] );
						}
					}

					// direct application for remainder
					for( ; i < n; ++i ) {
						result_buffer[ 0 ] = static_cast< D3 >( z[ i ] );
						OP::foldr( &left_buffer, result_buffer );
						z[ i ] = static_cast< IOType >( result_buffer[ 0 ] );
					}
				}

				/**
				 * In-place element-wise foldr function. Calculates
				 * \f$\forall\ i \in \{ 0, 1, \ldots, n - 1 \}, \f$
				 * \f$ x_i \odot z_i \f$ and stores the result into \f$ z_i \f$.
				 *
				 * @tparam InputType The type of elements in \a x.
				 * @tparam IOType    The type of elements in \a z.
				 *
				 * @param[in]     x The left-hand side input data.
				 * @param[in,out] z Where \a x shall be mapped into.
				 * @param[in]     n How many data elements \a x and \a z contain.
				 *
				 * This implementation requires two buffers only. It streams \a x once,
				 * while streaming \a z twice (once for reading, once for writing). This
				 * function should vectorise.
				 */
				template< typename InputType, typename IOType >
				static void eWiseFoldrAA( const InputType * __restrict__ const x, IOType * __restrict__ const z, const size_t n ) {
					// local buffers
					D1 left_buffer[ blocksize ];
					D3 result_buffer[ blocksize ];

					// blockwise application
					size_t i = 0;
					while( i + blocksize <= n ) {
						// load into buffers
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							left_buffer[ b ] = static_cast< D1 >( x[ i ] );
							result_buffer[ b ] = static_cast< D3 >( z[ i ] );
						}

						// rewind source and output
						i -= blocksize;

						// operate within buffer
						for( size_t b = 0; b < blocksize; ++b ) {
							OP::foldr( &( left_buffer[ b ] ), &( result_buffer[ b ] ) );
						}

						// write back result
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							z[ i ] = static_cast< IOType >( result_buffer[ b ] );
						}
					}

					// direct application for remainder
					for( ; i < n; ++i ) {
						left_buffer[ 0 ] = static_cast< D1 >( x[ i ] );
						result_buffer[ 0 ] = static_cast< D3 >( z[ i ] );
						OP::foldr( left_buffer, result_buffer );
						z[ i ] = static_cast< IOType >( result_buffer[ 0 ] );
					}
				}

				/**
				 * In-place element-wise apply function. Calculates
				 * \f$\forall\ i \in \{ 0, 1, \ldots, n - 1 \}, \f$
				 * \f$ z_i = x_i \odot y_i \f$.
				 *
				 * @tparam InputType1 The type of elements in \a x.
				 * @tparam InputType2 The type of elements in \a y.
				 * @tparam OutputType The type of elements in \a z.
				 *
				 * If \a InputType2 and \a D3 are not the same, then the existing data in
				 * \a y is cast to \a D3 prior to application of this in-place operator.
				 * If \a InputType1 and \a D1 are not the same, then the existing data in
				 * \a x are cast to \a D1 prior to application of this in-place operator.
				 * If \a OutputType and \a D3 are not the same, then the results of
				 * applying this operator are cast to \a OutputType prior to writing back
				 * the results.
				 *
				 * \warning The first casting behaviour may not be what you want. The two
				 *          other casting behaviours are allowed by the GraphBLAS unless
				 *          the alp::descriptor::no_casting is given.
				 *
				 * \note By default, this GraphBLAS implementation will only use this
				 *       code when \a D2 matches \a D3 and OP::has_foldr is \a true.
				 *
				 * This implementation relies on an in-place foldr().
				 *
				 * @param[in]  x The left-hand side input data. The memory range starting
				 *               at \a x and ending at \a x + n (exclusive) may not
				 *               overlap with the memory area starting at \a z and ending
				 *               at \a z + n (exclusive).
				 * @param[in]  y The right-hand side input data. The memory range starting
				 *               at \a y and ending at \a y + n (exclusive) may not
				 *               overlap with the memory area starting at \a z and ending
				 *               at \a z + n.
				 * @param[out] z Where the map of \a x into \a y must be stored. This
				 *               pointer is restricted in the sense that its memory may
				 *               never overlap with those pointed to by \a x or \y, as
				 *               detailed above.
				 * @param[in]  n How many data elements \a x, \a y, and \a z contain.
				 */
				template< typename InputType1, typename InputType2, typename OutputType >
				static void eWiseApply( const InputType1 * x, const InputType2 * y, OutputType * __restrict__ z, const size_t n ) {
#ifdef _DEBUG
#ifdef D_ALP_NO_STDIO
					std::cout << "In OperatorFR::eWiseApply\n";
#endif
#endif
					// NOTE: this variant is only active when the computation can be done using two buffers only

					// local buffers
					D1 left_buffer[ blocksize ];
					D3 result_buffer[ blocksize ];

					// blockwise application
					size_t i = 0;
					while( i + blocksize <= n ) {

						// load into buffers
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							left_buffer[ b ] = static_cast< D1 >( x[ i ] );
							result_buffer[ b ] = static_cast< D3 >( y[ i ] );
						}

						// rewind source and output
						i -= blocksize;

						// operate within buffer
						for( size_t b = 0; b < blocksize; ++b ) {
							OP::foldr( &( left_buffer[ b ] ), &( result_buffer[ b ] ) );
						}

						// write back result
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							z[ i ] = static_cast< OutputType >( result_buffer[ b ] );
						}
					}

					// direct application for remainder
					for( ; i < n; ++i ) {
						left_buffer[ 0 ] = static_cast< typename OP::left_type >( x[ i ] );
						result_buffer[ 0 ] = static_cast< typename OP::result_type >( y[ i ] );
						OP::foldr( left_buffer, result_buffer );
						z[ i ] = static_cast< OutputType >( result_buffer[ 0 ] );
					}
				}
			};

			/**
			 * A class capable of adding an out-of-place \a foldl function for an
			 * operator that is not fold-left capable, or capable of adding an in-
			 * place foldl function for an operator that is fold-left capable.
			 *
			 * An operator is fold-left capable when the Base Operator \a OP provides
			 * an in-place foldl implementation, \em and whenever \a D2 equals \a D3.
			 * If one of either requirements is not met, then \a OP is not fold-left
			 * capable and this class is selected to add an out-of-place foldl function.
			 *
			 * @tparam OP    The generic operator implementation.
			 * @tparam guard This typename is void if and only if \a OP is not fold-
			 *               left capable. In this case, this class adds an
			 *               out-of-place foldl implementation to the operator.
			 *               If \a guard is not void, then this class defines an
			 *               in-place foldr implementation instead.
			 *
			 * \note This specific class corresponds to the \a guard variable equal to
			 *       \a void.
			 *
			 * @see Operator for full details.
			 * @see OperatorFR for additional functions exposed to the resulting
			 *                 operator.
			 * @see OperatorBase for additional functions exposed to the resulting
			 *                   operator.
			 */
			template< typename OP, typename guard = void, enum Backend implementation = config::default_backend >
			class OperatorFL : public OperatorFR< OP > {

			private:
			public:
				typedef typename OperatorBase< OP >::D1 D1;
				typedef typename OperatorBase< OP >::D2 D2;
				typedef typename OperatorBase< OP >::D3 D3;
				static constexpr size_t blocksize = OperatorBase< OP >::blocksize;

				/**
				 * Emulated in-place application of this operator on two data elements.
				 *
				 * Computes \f$ x \odot y \f$ and writes the result into \f$ x \f$.
				 *
				 * We wish to call this in-place variant internally for brevity. However,
				 * if \a OP has no in-place variant, then we must cache the previous
				 * value of the output element or otherwise we will breach the
				 * __restrict__ contract of OP::apply.
				 * The caller must ensure the appropriate domains and casting behaviour
				 * is applicable. Note that a user is never to call these functions
				 * explicitly.
				 *
				 * @tparam InputType The type of the parameter \a x.
				 * @tparam IOType    The type of the parameter \a y.
				 *
				 * \warning Additional casting and use of temporary variables may occur
				 *          when \a InputType does not match \a D2 \em or \a IOType
				 *          does not match \a D3.
				 *
				 * \note This implementation relies on apply().
				 *
				 * @param[in,out] x The value \a y is to be applied against.
				 * @param[in]     y The value that is to be applied to \a x.
				 */
				template< typename InputType, typename IOType >
				static void foldl( IOType & x, const InputType & y ) {
					const D1 cache = static_cast< D1 >( x );
					OperatorBase< OP >::apply( cache, y, x );
				}

				/**
				 * Out-of-place element-wise foldl function. Calculates
				 * \f$\forall\ i \in \{ 0, 1, \ldots, n - 1 \}, \f$
				 * \f$ x_i \odot y \f$ and stores the result into \f$ x_i \f$.
				 *
				 * @tparam IOType    The type of elements in \a x.
				 * @tparam InputType The type of \a y.
				 *
				 * @param[in, out] x At function entry, the left-hand side input data.
				 *                   At function exit, the output data as defined above.
				 * @param[in]      y The right-hand side input value.
				 * @param[in]      n How many data elements \a x contains.
				 *
				 * This version requires two buffers and streams \a x twice (once for
				 * reading, once for writing). This function should vectorise its
				 * out-of-place operations.
				 */
				template< typename IOType, typename InputType >
				static void eWiseFoldlAS( IOType * __restrict__ const x, const InputType y, const size_t n ) {
					// local buffers
					D1 left_buffer[ blocksize ];
					const D2 right_buffer = y;
					D3 result_buffer[ blocksize ];

					// blockwise application
					size_t i = 0;
					while( i + blocksize <= n ) {
						// load into buffers
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							left_buffer[ b ] = static_cast< D1 >( x[ i ] );
						}

						// rewind source and output
						i -= blocksize;

						// operate within buffer
						for( size_t b = 0; b < blocksize; ++b ) {
							OP::apply( &( left_buffer[ b ] ), &right_buffer, &( result_buffer[ b ] ) );
						}

						// write back result
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							x[ i ] = static_cast< IOType >( result_buffer[ b ] );
						}
					}

					// direct application for remainder
					for( ; i < n; ++i ) {
						left_buffer[ 0 ] = static_cast< D1 >( x[ i ] );
						OP::apply( left_buffer, &right_buffer, result_buffer );
						x[ i ] = static_cast< IOType >( result_buffer[ 0 ] );
					}
				}

				/**
				 * Out-of-place element-wise foldl function. Calculates
				 * \f$\forall\ i \in \{ 0, 1, \ldots, n - 1 \}, \f$
				 * \f$ x_i \odot y_i \f$ and stores the result into \f$ x_i \f$.
				 *
				 * @tparam IOType    The type of elements in \a x.
				 * @tparam InputType The type of elements in \a y.
				 *
				 * @param[in, out] x At function entry, the left-hand side input data.
				 *                   At function exit, the output data as defined above.
				 * @param[in]      y The right-hand side input.
				 * @param[in]      n How many data elements \a x and \a y contain.
				 *
				 * This version requires three buffers, streams \a y once, and streams
				 * \a x twice (once for reading, once for writing). This function should
				 * vectorise its out-of-place operations.
				 */
				template< typename IOType, typename InputType >
				static void eWiseFoldlAA( IOType * __restrict__ const x, const InputType * __restrict__ const y, const size_t n ) {
					// local buffers
					D1 left_buffer[ blocksize ];
					D2 right_buffer[ blocksize ];
					D3 result_buffer[ blocksize ];

					// blockwise application
					size_t i = 0;
					while( i + blocksize <= n ) {
						// load into buffers
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							left_buffer[ b ] = static_cast< D1 >( x[ i ] );
							right_buffer[ b ] = static_cast< D2 >( y[ i ] );
						}

						// rewind source and output
						i -= blocksize;

						// operate within buffer
						for( size_t b = 0; b < blocksize; ++b ) {
							OP::apply( &( left_buffer[ b ] ), &( right_buffer[ b ] ), &( result_buffer[ b ] ) );
						}

						// write back result
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							x[ i ] = static_cast< IOType >( result_buffer[ b ] );
						}
					}

					// direct application for remainder
					for( ; i < n; ++i ) {
						left_buffer[ 0 ] = static_cast< D1 >( x[ i ] );
						right_buffer[ 0 ] = static_cast< D2 >( y[ i ] );
						OP::apply( left_buffer, right_buffer, result_buffer );
						x[ i ] = static_cast< IOType >( result_buffer[ 0 ] );
					}
				}
			};

			/**
			 * This class provides an in-place foldl implementation for Base Operators
			 * that are fold-left capable given its provided domains.
			 *
			 * @tparam OP The generic operator implementation.
			 *
			 * @see Operator for full details.
			 * @see OperatorFL for details on fold-right capable operators and behaviour
			 *                 for non fold-right capable operators.
			 * @see OperatorFR for additional functions exposed to the resulting
			 *                 operator.
			 * @see OperatorBase for additional functions exposed to the resulting
			 *                   operator.
			 */
			template< typename OP >
			class OperatorFL< OP, typename std::enable_if< OP::has_foldl && std::is_same< typename OP::left_type, typename OP::result_type >::value >::type > : public OperatorFR< OP > {

			private:
			public:
				typedef typename OperatorBase< OP >::D2 D2;
				typedef typename OperatorBase< OP >::D3 D3;
				static constexpr size_t blocksize = OperatorBase< OP >::blocksize;

				/**
				 * In-place application of this operator on two data elements.
				 *
				 * Computes \f$ x \odot y \f$ and writes the result into \f$ x \f$.
				 *
				 * \note This variant is only called when the underlying raw operator
				 *       supports in-place operations.
				 *
				 * The caller must ensure the appropriate domains and casting behaviour
				 * is applicable. Note that a user is never to call these functions
				 * explicitly.
				 *
				 * @param[in,out] x The value \a y is to be applied against.
				 * @param[in]     y The value that is to be applied to \a x.
				 */
				static void foldl( D3 & x, const D2 & y ) {
					OP::foldl( &x, &y );
				}

				/**
				 * In-place element-wise foldl function. Calculates
				 * \f$\forall\ i \in \{ 0, 1, \ldots, n - 1 \}, \f$
				 * \f$ x_i \odot y_i \f$ and stores the result into \f$ x_i \f$.
				 *
				 * @tparam IOType    The type of elements in \a x.
				 * @tparam InputType The type of elements in \a y.
				 *
				 * @param[in,out] x At function extry: the left-hand side input data.
				 *                  At function exit: the result data.
				 * @param[in]     y The right-hand side input data.
				 * @param[in]     n How many data elements \a x and \a y contain.
				 *
				 * This implementation requires two buffers only. It streams \a y once,
				 * while streaming \a x twice (once for reading, once for writing). This
				 * function should vectorise.
				 */
				template< typename InputType, typename IOType >
				static void eWiseFoldlAA( IOType * __restrict__ const x, const InputType * __restrict__ const y, const size_t n ) {
					// local buffers
					D2 right_buffer[ blocksize ];
					D3 result_buffer[ blocksize ];

					// blockwise application
					size_t i = 0;
					while( i + blocksize <= n ) {
						// load into buffers
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							right_buffer[ b ] = static_cast< D2 >( y[ i ] );
							result_buffer[ b ] = static_cast< D3 >( x[ i ] );
						}

						// rewind source and output
						i -= blocksize;

						// operate within buffer
						for( size_t b = 0; b < blocksize; ++b ) {
							OP::foldl( &( result_buffer[ b ] ), &( right_buffer[ b ] ) );
						}

						// write back result
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							x[ i ] = static_cast< IOType >( result_buffer[ b ] );
						}
					}

					// direct application for remainder
					for( ; i < n; ++i ) {
						right_buffer[ 0 ] = static_cast< D2 >( y[ i ] );
						result_buffer[ 0 ] = static_cast< D3 >( x[ i ] );
						OP::foldl( result_buffer, right_buffer );
						x[ i ] = static_cast< IOType >( result_buffer[ 0 ] );
					}
				}

				/**
				 * In-place element-wise foldl function. Calculates
				 * \f$ \forall\ i \in \{ 0, 1, \ldots, n - 1 \}, \f$
				 * \f$ x_i \odot y \f$ and stores the result into \f$ x_i \f$.
				 *
				 * @tparam IOType    The type of elements in \a x.
				 * @tparam InputType The type of \a y.
				 *
				 * @param[in,out] x At function extry: the left-hand side input data.
				 *                  At function exit: the result data.
				 * @param[in]     y The right-hand side input value.
				 * @param[in]     n How many data elements \a x contains.
				 *
				 * This implementation requires one buffers only. It streams \a x twice
				 * (once for reading, once for writing). This function should vectorise.
				 */
				template< typename InputType, typename IOType >
				static void eWiseFoldlAS( IOType * __restrict__ const x, const InputType y, const size_t n ) {
					// local buffers
					const D2 right_buffer = static_cast< D2 >( y );
					D3 result_buffer[ blocksize ];

					// blockwise application
					size_t i = 0;
					while( i + blocksize <= n ) {
						// load into buffers
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							result_buffer[ b ] = static_cast< D3 >( x[ i ] );
						}

						// rewind source and output
						i -= blocksize;

						// operate within buffer
						for( size_t b = 0; b < blocksize; ++b ) {
							OP::foldl( &( result_buffer[ b ] ), &right_buffer );
						}

						// write back result
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							x[ i ] = static_cast< IOType >( result_buffer[ b ] );
						}
					}

					// direct application for remainder
					for( ; i < n; ++i ) {
						result_buffer[ 0 ] = static_cast< D3 >( x[ i ] );
						OP::foldl( result_buffer, &right_buffer );
						x[ i ] = static_cast< IOType >( result_buffer[ 0 ] );
					}
				}
			};

			/**
			 * A class capable of adding an in-place \a eWiseApply function for an
			 * operator that is fold-left capable but not fold-right capable.
			 *
			 * Like OperatorFR on an fold-right capable operator, this class is
			 * capable of providing an eWiseApply function that requires only two
			 * internal buffers by making use of the in-place foldl.
			 *
			 * @tparam OP The generic operator implementation.
			 * @tparam guard This typename is void if and only if \a OP is fold-left
			 *               capable but \em not fold-right capable. In this case,
			 *               this class adds nothing to the resulting operator.
			 *               If \a guard is not void, however, then this class adds an
			 *               in-place eWiseApply implementation to this operator
			 *               instead.
			 *
			 * @see Operator for full details.
			 * @see OperatorFL for additional functions exposed to the resulting
			 *                 operator.
			 * @see OperatorFR for additional functions exposed to the resulting
			 *                 operator and an alternative way of providing a more
			 *                 efficient eWiseApply.
			 * @see OperatorBase for additional functions exposed to the resulting
			 *                   operator.
			 */
			template< typename OP, typename guard = void, enum Backend implementation = config::default_backend >
			class OperatorNoFR : public OperatorFL< OP > {};

			/**
			 * This class provides an in-place eWiseApply implementation for Base
			 * Operators that are fold-left capable given its provided domains, but not
			 * fold-right capable. This implementation uses two internal buffers and
			 * relies on an in-place foldl. If this were not possible, then the
			 * eWiseApply will be provided by OperatorNoFRFL in an implementation that
			 * requires three buffers and out-of-place operations instead.
			 *
			 * @tparam OP The generic operator implementation.
			 *
			 * @see Operator for full details.
			 * @see OperatorFL for additional functions exposed to the resulting
			 *                 operator.
			 * @see OperatorFR for additional functions exposed to the resulting
			 *                 operator.
			 * @see OperatorBase for additional functions exposed to the resulting
			 *                   operator.
			 */
			template< typename OP >
			class OperatorNoFR< OP, typename std::enable_if< OP::has_foldl && ! ( OP::has_foldr ) && std::is_same< typename OP::left_type, typename OP::result_type >::value >::type > :
				public OperatorFL< OP > {

			private:
			public:
				typedef typename OperatorBase< OP >::D2 D2;
				typedef typename OperatorBase< OP >::D3 D3;
				static constexpr size_t blocksize = OperatorBase< OP >::blocksize;

				/**
				 * In-place element-wise apply function. Calculates
				 * \f$\forall\ i \in \{ 0, 1, \ldots, n - 1 \}, \f$
				 * \f$ z_i = x_i \odot y_i \f$.
				 *
				 * @tparam InputType1 The type of elements in \a x.
				 * @tparam InputType2 The type of elements in \a y.
				 * @tparam OutputType The type of elements in \a z.
				 *
				 * If the \a InputType1 and \a D3 are not the same, then the existing data
				 * in \a x is cast to \a D3 prior to application of this operator.
				 * If \a InputType2 and \a D2 are not the same, then the existing data in
				 * \a y is cast to \a D2 prior to application of this operator.
				 * If \a OutputType and \a D3 are not the same, then the result of
				 * applications of this operator are cast to \a OutputType prior to
				 * writing it back to \a z.
				 *
				 * \warning The first casting behaviour may not be what you want. The two
				 *          other casting behaviours are allowed by the GraphBLAS unless
				 *          the alp::descriptor::no_casting is given.
				 *
				 * \note By default, this GraphBLAS implementation will only use this
				 *       code when \a D1 matches \a D3 and OP::has_foldr is \a true.
				 *       However, this implementation will never be enabled if \a D2
				 *       equals \a D3 and OP::has_foldl is \a true.
				 *
				 * This implementation relies on an in-place foldl().
				 *
				 * @param[in]  x The left-hand side input data. The memory range starting
				 *               at \a x and ending at \a x + n (exclusive) may not
				 *               overlap with the memory area starting at \a z and ending
				 *               at \a z + n (exclusive).
				 * @param[in]  y The right-hand side input data. The memory range starting
				 *               at \a y and ending at \a y + n (exclusive) may not
				 *               overlap with the memory area starting at \a z and ending
				 *               at \a z + n.
				 * @param[out] z Where the map of \a x into \a y must be stored. This
				 *               pointer is restricted in the sense that its memory may
				 *               never overlap with those pointed to by \a x or \y, as
				 *               detailed above.
				 * @param[in]  n How many data elements \a x, \a y, and \a z contain.
				 */
				template< typename InputType1, typename InputType2, typename OutputType >
				static void eWiseApply( const InputType1 * x, const InputType2 * y, OutputType * __restrict__ z, const size_t n ) {
#ifdef _DEBUG
#ifdef D_ALP_NO_STDIO
					std::cout << "In OperatorNoFR::eWiseApply\n";
#endif
#endif
					// NOTE: this variant is only active when the computation can be done using two buffers only

					// local buffers
					D2 right_buffer[ blocksize ];
					D3 result_buffer[ blocksize ];

					// blockwise application
					size_t i = 0;
					while( i + blocksize <= n ) {

						// load into buffers
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							right_buffer[ b ] = static_cast< D2 >( y[ i ] );
							result_buffer[ b ] = static_cast< D3 >( x[ i ] );
						}

						// rewind source and output
						i -= blocksize;

						// operate within buffer
						for( size_t b = 0; b < blocksize; ++b ) {
							OP::foldl( &( result_buffer[ b ] ), &( right_buffer[ b ] ) );
						}

						// write back result
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							z[ i ] = static_cast< OutputType >( result_buffer[ b ] );
						}
					}

					// direct application for remainder
					for( ; i < n; ++i ) {
						right_buffer[ 0 ] = static_cast< D2 >( y[ i ] );
						result_buffer[ 0 ] = static_cast< D3 >( x[ i ] );
						OP::foldl( result_buffer, right_buffer );
						z[ i ] = static_cast< OutputType >( result_buffer[ 0 ] );
					}
				}
			};

			/**
			 * A class capable of adding an out-of-place \a eWiseApply function for an
			 * operator that, given its domains, is not fold-left capable \em an not
			 * fold-right capable.
			 *
			 * If the given operator is not fold-left and not fold-right capable, then
			 * both OperatorFR and OperatorNoFR have not yet added an eWiseApply
			 * implementation. However, if there was already an in-place foldr or an
			 * in-place foldl available, then this class will add no new functions to
			 * the resulting operator.
			 * A class capable of adding an out-of-place eWiseApply function for an
			 * operator that is not fold-left capable \em and not fold-right capable.
			 *
			 * @tparam OP    The generic operator implementation.
			 * @tparam guard This typename is void if and only if there is already an
			 *               in-place eWiseApply defined by the base OperatorNoFR
			 *               class or by the OperatorFR class. In this case, this
			 *               class does not add any new public methods.
			 *               If it is not void, then this class defines an
			 *               out-of-place eWiseApply function.
			 *
			 * \note This specific class corresponds to the \a guard variable equal to
			 *       \a void.
			 *
			 * @see Operator for full details.
			 * @see OperatorNoFR for additional functions exposed to the resulting
			 *                   operator.
			 * @see OperatorFL for additional functions exposed to the resulting
			 *                 operator.
			 * @see OperatorFR for additional functions exposed to the resulting
			 *                 operator and an alternative way of providing a more
			 *                 efficient eWiseApply.
			 * @see OperatorBase for additional functions exposed to the resulting
			 *                   operator.
			 */
			template< typename OP, typename guard = void, enum Backend implementation = config::default_backend >
			class OperatorNoFRFL : public OperatorNoFR< OP > {};

			/**
			 * A class that adds an out-of-place \a eWiseApply function for an operator
			 * that, given its domains, is not fold-left capable \em and not fold-right
			 * capable.
			 *
			 * Contains further specialisations for an operator that is not fold-left,
			 * capable \em and not fold-right capable. This means we have to supply an
			 * eWiseApply function that uses the normal OperatorBase::apply function,
			 * and thus uses three buffers instead of the two buffers required by its
			 * in-place counterparts.
			 *
			 * @tparam OP The generic operator implementation.
			 *
			 * @see Operator for full details.
			 * @see OperatorNoFR for additional functions exposed to the resulting
			 *                   operator.
			 * @see OperatorFL for additional functions exposed to the resulting
			 *                 operator.
			 * @see OperatorFR for additional functions exposed to the resulting
			 *                 operator and an alternative way of providing a more
			 *                 efficient eWiseApply.
			 * @see OperatorBase for additional functions exposed to the resulting
			 *                   operator and the OperatorBase::apply function this
			 *                   class will use.
			 */
			template< typename OP >
			class OperatorNoFRFL< OP,
				typename std::enable_if< ( ! ( OP::has_foldl ) || ! ( std::is_same< typename OP::left_type, typename OP::result_type >::value ) ) &&
					( ! ( OP::has_foldr ) || ! ( std::is_same< typename OP::right_type, typename OP::result_type >::value ) ) >::type > : public OperatorNoFR< OP > {

			private:
			public:
				typedef typename OperatorBase< OP >::D1 D1;
				typedef typename OperatorBase< OP >::D2 D2;
				typedef typename OperatorBase< OP >::D3 D3;
				static constexpr size_t blocksize = OperatorBase< OP >::blocksize;

				/** \anchor OperatorNoFRFLeWiseApply
				 *
				 * Standard out-of-place element-wise apply function. Calculates
				 * \f$\forall\ i \in \{ 0, 1, \ldots, n - 1 \}, \f$
				 * \f$ z_i = x_i \odot y_i \f$.
				 *
				 * This is the non-public variant that operates on raw arrays.
				 *
				 * @tparam InputType1 The type of elements in \a x.
				 * @tparam InputType2 The type of elements in \a y.
				 * @tparam OutputType The type of elements in \a z.
				 *
				 * If \a InputType1 and \a D1 are not the same, then the existing data in
				 * \a x will be cast to \a D1 prior to application of this operator.
				 * If \a InputType2 and \a D2 are not the same, then the existing data in
				 * \a y will be cast to \a D2 prior to application of this operator.
				 * If \a OutputType and \a D3 are not the same, then the results of
				 * applications of this operator are cast to \a OutputType prior to
				 * writing them back to \a z.
				 *
				 * \note The GraphBLAS can explicitly control all \em three of this
				 *       casting behaviours via alp::descriptors::no_casting.
				 *
				 * \warning With the in-place variants of this code, unwanted behaviour
				 *          cannot be prevented by use of alp::descriptors::no_casting.
				 *          Therefore the current implementation only calls the in-place
				 *          variants when \a D1 equals \a D3 (for foldl-based in-place),
				 *          or when \a D2 equals \a D3 (for foldr-based ones).
				 *
				 * @param[in]  x The left-hand side input data. The memory range starting
				 *               at \a x and ending at \a x + n (exclusive) may not
				 *               overlap with the memory area starting at \a z and ending
				 *               at \a z + n (exclusive).
				 * @param[in]  y The right-hand side input data. The memory range starting
				 *               at \a y and ending at \a y + n (exclusive) may not
				 *               overlap with the memory area starting at \a z and ending
				 *               at \a z + n.
				 * @param[out] z Where the map of \a x into \a y must be stored. This
				 *               pointer is restricted in the sense that its memory may
				 *               never overlap with those pointed to by \a x or \y, as
				 *               detailed above.
				 * @param[in]  n How many data elements \a x, \a y, and \a z contain.
				 */
				template< typename InputType1, typename InputType2, typename OutputType >
				static void eWiseApply( const InputType1 * x, const InputType2 * y, OutputType * __restrict__ z, const size_t n ) {
#ifdef _DEBUG
#ifdef D_ALP_NO_STDIO
					std::cout << "In OperatorNoFRFL::eWiseApply\n";
#endif
#endif
					// NOTE: this variant is only active when the computation can NOT be done using two buffers only

					// local buffers
					D1 left_buffer[ blocksize ];
					D2 right_buffer[ blocksize ];
					D3 result_buffer[ blocksize ];

					// blockwise application
					size_t i = 0;
					while( i + blocksize <= n ) {

						// load into buffers
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							left_buffer[ b ] = static_cast< D1 >( x[ i ] );
							right_buffer[ b ] = static_cast< D2 >( y[ i ] );
						}

						// rewind source and output
						i -= blocksize;

						// operate within buffer
						for( size_t b = 0; b < blocksize; ++b ) {
							OP::apply( &( left_buffer[ b ] ), &( right_buffer[ b ] ), &( result_buffer[ b ] ) );
						}

						// write back result
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							z[ i ] = static_cast< OutputType >( result_buffer[ b ] );
						}
					}

					// direct application for remainder
					for( ; i < n; ++i ) {
						left_buffer[ 0 ] = static_cast< D1 >( x[ i ] );
						right_buffer[ 0 ] = static_cast< D2 >( y[ i ] );
						OP::apply( left_buffer, right_buffer, result_buffer );
						z[ i ] = static_cast< OutputType >( result_buffer[ 0 ] );
					}
				}
			};

			/**
			 * This is the operator interface exposed to the GraphBLAS implementation.
			 *
			 * \warning Note that most GraphBLAS usage requires associative operators.
			 *          While very easily possible to create non-associative operators
			 *          using this interface, passing them to GraphBLAS functions,
			 *          either explicitly or indirectly (by, e.g., including them in a
			 *          alp::Monoid or alp::Semiring), will lead to undefined
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
			 * For examples of these base operators, see alp::operators::internal::max
			 * or alp::operators::internal::mul. An example of a full implementation,
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
			 * class around it, as illustrated, e.g., by alp::operators::add as follows:
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
			template< typename OP, enum Backend implementation = config::default_backend >
			class Operator : public OperatorNoFRFL< OP > {

			private:
			public:
				/** The maximum block size when vectorising this operation. */
				static constexpr size_t blocksize = OperatorBase< OP >::blocksize;

				/** The left-hand side input domain of this operator. */
				typedef typename OperatorBase< OP >::D1 D1;

				/** The right-hand side input domain of this operator. */
				typedef typename OperatorBase< OP >::D2 D2;

				/** The output domain of this operator. */
				typedef typename OperatorBase< OP >::D3 D3;

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
					// prepare scalar buffer
					D3 reduced = static_cast< D3 >( out );
					// prepare vectorisation buffer
					D1 left_buffer[ blocksize ];
					// blockwise application
					size_t i = n - 1;
					while( i - blocksize + 1 < n ) {
						// load into buffer
						for( size_t b = 0; b < blocksize; --i, ++b ) {
							left_buffer[ b ] = static_cast< D1 >( x[ i ] );
						}
						// do reduce
						for( size_t b = 0; b < blocksize; ++b ) {
							OP::foldr( &( left_buffer[ b ] ), &reduced );
						}
					}
					// direct application for remainder
					for( ; i < n; --i ) {
						left_buffer[ 0 ] = static_cast< D1 >( x[ i ] );
						OP::foldr( left_buffer, &reduced );
					}
					// write out
					out = static_cast< IOType >( reduced );
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
					// prepare scalar buffer
					D3 reduced = static_cast< D3 >( out );
					// prepare vectorisation buffer
					D2 right_buffer[ blocksize ];
					// blockwise application
					size_t i = 0;
					while( i + blocksize <= n ) {
						// load into buffer
						for( size_t b = 0; b < blocksize; ++i, ++b ) {
							right_buffer[ b ] = static_cast< D2 >( x[ i ] );
						}
						// do reduce
						for( size_t b = 0; b < blocksize; ++b ) {
							OP::foldl( &reduced, &( right_buffer[ b ] ) );
						}
					}
					// direct application for remainder
					for( ; i < n; ++i ) {
						right_buffer[ 0 ] = static_cast< D2 >( x[ i ] );
						OP::foldl( &reduced, right_buffer );
					}
					// write out
					out = static_cast< IOType >( reduced );
				}
			};

		} // namespace internal

	} // namespace operators

} // namespace alp

#endif // _H_ALP_INTERNAL_OPERATORS_BASE

