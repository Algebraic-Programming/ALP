
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
 * @author D. G. Spampinato
 * @date 2nd of November, 2022
 */

#ifndef _H_ALP_INTERNAL_RELATIONS_BASE
#define _H_ALP_INTERNAL_RELATIONS_BASE

#include <type_traits>

#include <alp/backends.hpp>

namespace alp {

	namespace relations {

		/** Core implementations of the standard relations in #alp::relations. */
		namespace internal {

			/**
			 * Standard less then or equal (le) operator.
			 *
			 * Assumes native availability of operator <= on the given data types 
			 * or assumes that the relevant operators are properly overloaded.
			 *
			 * Assumes that the <= operator is a total order. Non-standard/non-matching 
			 * data types or non-standard (overloaded) <= operators should 
			 * therefore be used with caution.
			 *
			 * @tparam SET The input data type.
			 */
			template< typename SET, enum Backend implementation = config::default_backend >
			class le {

			public:
				/** Alias to the left-hand input data type. */
				typedef SET left_type;

				/** Alias to the right-hand input data type. */
				typedef SET right_type;

				/**
				 * Whether this relation is \em reflexive; that is,
				 * for all \a a in \a SET, \f$ a \le a \f$.
				 */
				static constexpr bool is_reflexive = true;

				/**
				 * Whether this relation is \em irreflexive; that is,
				 * for all \a a in \a SET, not \f$ a \le a \f$.
				 */
				static constexpr bool is_irreflexive = false;

				/**
				 * Whether this relation is \em symmetric; that is,
				 * for all \a a, \a b in \a SET, 
				 * if \f$ a \le b \f$ then \f$ b \le a \f$.
				 */
				static constexpr bool is_symmetric = false;

				/**
				 * Whether this relation is \em antisymmetric; that is,
				 * for all \a a, \a b in \a SET, if \f$ a \le b \f$ and 
				 * \f$ b \le a \f$ then \f$ a = b \f$.
				 */
				static constexpr bool is_antisymmetric = true;

				/**
				 * Whether this relation is \em transitive; that is,
				 * for all \a a, \a b, \a c in \a SET, if \f$ a \le b \f$ and
				 * \f$ b \le c \f$ then \f$ a \le c \f$.
				 */
				static constexpr bool is_transitive = true;

				/**
				 * Whether this relation is \em connected; that is,
				 * for all \a a, \a b in \a SET, if \f$ a \neq b \f$ then
				 * either \f$ a \le b \f$ or \f$ b \le a \f$.
				 */
				static constexpr bool is_connected = true;

				/**
				 * Whether this relation is <em> strongly connected </em> (or total); 
				 * that is,
				 * for all \a a, \a b in \a SET, 
				 * either \f$ a \le b \f$ or \f$ b \le a \f$.
				 */
				static constexpr bool is_strongly_connected = true;

				/**
				 * Check if a <= b.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 *
				 * \warning Passing invalid pointers will result in UB.
				 */
				static bool test( const left_type * __restrict__ const a,
					const right_type * __restrict__ const b
				) {
					return *a <= *b;
				}
			};

			/**
			 * Standard less then (lt) operator.
			 *
			 * Assumes native availability of operator < on the given data types 
			 * or assumes that the relevant operators are properly overloaded.
			 *
			 * Assumes that the < operator is a strict total order. Non-standard/non-matching 
			 * data types or non-standard (overloaded) < operators should 
			 * therefore be used with caution.
			 *
			 * @tparam SET The input data type.
			 */
			template< typename SET, enum Backend implementation = config::default_backend >
			class lt {

			public:
				/** Alias to the left-hand input data type. */
				typedef SET left_type;

				/** Alias to the right-hand input data type. */
				typedef SET right_type;

				/**
				 * Whether this relation is \em reflexive; that is,
				 * for all \a a in \a SET, \f$ a < a \f$.
				 */
				static constexpr bool is_reflexive = false;

				/**
				 * Whether this relation is \em irreflexive; that is,
				 * for all \a a in \a SET, not \f$ a < a \f$.
				 */
				static constexpr bool is_irreflexive = true;

				/**
				 * Whether this relation is \em symmetric; that is,
				 * for all \a a, \a b in \a SET, 
				 * if \f$ a < b \f$ then \f$ b < a \f$.
				 */
				static constexpr bool is_symmetric = false;

				/**
				 * Whether this relation is \em antisymmetric; that is,
				 * for all \a a, \a b in \a SET, if \f$ a < b \f$ and 
				 * \f$ b < a \f$ then \f$ a = b \f$.
				 */
				static constexpr bool is_antisymmetric = true;

				/**
				 * Whether this relation is \em transitive; that is,
				 * for all \a a, \a b, \a c in \a SET, if \f$ a < b \f$ and
				 * \f$ b < c \f$ then \f$ a < c \f$.
				 */
				static constexpr bool is_transitive = true;

				/**
				 * Whether this relation is \em connected (or total); that is,
				 * for all \a a, \a b in \a SET, if \f$ a \neq b \f$ then
				 * either \f$ a < b \f$ or \f$ b < a \f$.
				 */
				static constexpr bool is_connected = true;

				/**
				 * Whether this relation is <em> strongly connected </em>; 
				 * that is,
				 * for all \a a, \a b in \a SET, 
				 * either \f$ a < b \f$ or \f$ b < a \f$.
				 */
				static constexpr bool is_strongly_connected = false;

				/**
				 * Check if a < b.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 *
				 * \warning Passing invalid pointers will result in UB.
				 */
				static bool test( const left_type * __restrict__ const a,
					const right_type * __restrict__ const b
				) {
					return *a < *b;
				}
			};

			/**
			 * Standard equal (eq) operator.
			 *
			 * Assumes native availability of operator == on the given data types 
			 * or assumes that the relevant operators are properly overloaded.
			 *
			 * Assumes that the == operator is an equivalence relation. 
			 * Non-standard/non-matching data types or non-standard (overloaded) 
			 * == operators should therefore be used with caution.
			 *
			 * @tparam SET The input data type.
			 */
			template< typename SET, enum Backend implementation = config::default_backend >
			class eq {

			public:
				/** Alias to the left-hand input data type. */
				typedef SET left_type;

				/** Alias to the right-hand input data type. */
				typedef SET right_type;

				/**
				 * Whether this relation is \em reflexive; that is,
				 * for all \a a in \a SET, \f$ a = a \f$.
				 */
				static constexpr bool is_reflexive = true;

				/**
				 * Whether this relation is \em irreflexive; that is,
				 * for all \a a in \a SET, not \f$ a = a \f$.
				 */
				static constexpr bool is_irreflexive = false;

				/**
				 * Whether this relation is \em symmetric; that is,
				 * for all \a a, \a b in \a SET, 
				 * if \f$ a = b \f$ then \f$ b = a \f$.
				 */
				static constexpr bool is_symmetric = true;

				/**
				 * Whether this relation is \em antisymmetric; that is,
				 * for all \a a, \a b in \a SET, if \f$ a = b \f$ and 
				 * \f$ b = a \f$ then \f$ a = b \f$.
				 */
				static constexpr bool is_antisymmetric = true;

				/**
				 * Whether this relation is \em transitive; that is,
				 * for all \a a, \a b, \a c in \a SET, if \f$ a = b \f$ and
				 * \f$ b = c \f$ then \f$ a = c \f$.
				 */
				static constexpr bool is_transitive = true;

				/**
				 * Whether this relation is \em connected; that is,
				 * for all \a a, \a b in \a SET, if \f$ a \neq b \f$ then
				 * either \f$ a = b \f$ or \f$ b = a \f$.
				 */
				static constexpr bool is_connected = false;

				/**
				 * Whether this relation is <em> strongly connected </em> (or total); 
				 * that is,
				 * for all \a a, \a b in \a SET, 
				 * either \f$ a = b \f$ or \f$ b = a \f$.
				 */
				static constexpr bool is_strongly_connected = false;

				/**
				 * Check if a == b.
				 *
				 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
				 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
				 *
				 * \warning Passing invalid pointers will result in UB.
				 */
				static bool test( const left_type * __restrict__ const a,
					const right_type * __restrict__ const b
				) {
					return *a == *b;
				}
			};


			/**
			 * This class takes a generic operator implementation and exposes a more
			 * convenient test() function based on it. This function allows arbitrary
			 * data types being passed as parameters, and automatically handles any
			 * casting required for the raw operator.
			 *
			 * @tparam REL The generic operator implementation.
			 *
			 */
			template< typename REL, enum Backend implementation = config::default_backend >
			class RelationBase {

			protected:

				/** The left-hand input domain. */
				typedef typename REL::left_type D1;

				/** The right-hand input domain. */
				typedef typename REL::right_type D2;

			public:
				/** @return Whether this relation is reflexive. */
				static constexpr bool is_reflexive() {
					return REL::is_reflexive;
				}

				/** @return Whether this relation is irreflexive. */
				static constexpr bool is_irreflexive() {
					return REL::is_irreflexive;
				}

				/** @return Whether this relation is symmetric. */
				static constexpr bool is_symmetric() {
					return REL::is_symmetric;
				}

				/** @return Whether this relation is antisymmetric. */
				static constexpr bool is_antisymmetric() {
					return REL::is_antisymmetric;
				}

				/** @return Whether this relation is transitive. */
				static constexpr bool is_transitive() {
					return REL::is_transitive;
				}

				/** @return Whether this relation is connected. */
				static constexpr bool is_connected() {
					return REL::is_connected;
				}

				/** @return Whether this relation is strongly connected. */
				static constexpr bool is_strongly_connected() {
					return REL::is_strongly_connected;
				}

				/**
				 * Straightforward test of this relation. Returns if \f$ x REL y \f$.
				 *
				 * @tparam InputType1 The type of the input parameter \a x.
				 * @tparam InputType2 The type of the input parameter \a y.
				 *
				 * \warning If \a InputType1 does not match \a D1 \em or \a InputType2 does
				 *          not match \a D2, then input will be cast into temporary 
				 *          variables of the correct types.
				 *
				 * \note Best performance is thus only guaranteed when all domains match.
				 *
				 * @param[in]  x The left-hand side input.
				 * @param[in]  y The right-hand side input.
				 */
				template< typename InputType1, typename InputType2 >
				static bool test( const InputType1 & x, const InputType2 & y ) {
					const D1 a = static_cast< D1 >( x );
					const D2 b = static_cast< D2 >( y );
					return REL::test( &a, &b );
				}

				/**
				 * This is the high-performance version of apply() in the sense that no
				 * casting is required. This version will be automatically caled whenever
				 * possible.
				 */
				static bool test( const D1 & x, const D2 & y ) {
					return REL::test( &x, &y );
				}
			};

			/**
			 * TODO: Update for Relation
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
			template< typename REL, enum Backend implementation = config::default_backend >
			class Relation : public RelationBase< REL > {

				public:
					typedef typename RelationBase< REL >::D1 D1;
					typedef typename RelationBase< REL >::D2 D2;

			};

			/**
			 *
			 * @tparam REL   The generic homogeneous relation.
			 *
			 * @see Relation
			 * @see RelationBase for additional functions exposed to the final relation.
			 */
			template< 
				typename REL, 
				enum Backend implementation = config::default_backend,
				std::enable_if_t< 
					std::is_same< 
						typename REL::left_type, 
						typename REL::right_type 
					>::value 
				> * = nullptr
			>
			class HomogeneousRelation : public RelationBase< REL > {
			};

			template< 
				typename REL, 
				enum Backend implementation = config::default_backend,
				std::enable_if_t< 
					REL::is_reflexive
					&& REL::is_transitive
					&& REL::is_antisymmetric
				> * = nullptr
			>
			class PartialOrder : public HomogeneousRelation< REL > {
			};

			template< 
				typename REL, 
				enum Backend implementation = config::default_backend,
				std::enable_if_t< 
					REL::is_irreflexive
					&& REL::is_transitive
					&& REL::is_antisymmetric
				> * = nullptr
			>
			class StrictPartialOrder : public HomogeneousRelation< REL > {
			};

			template< 
				typename REL, 
				enum Backend implementation = config::default_backend,
				std::enable_if_t< 
					REL::is_strongly_connected 
				> * = nullptr
			>
			class TotalOrder : public PartialOrder< REL > {
			};

			template< 
				typename REL, 
				enum Backend implementation = config::default_backend,
				std::enable_if_t< 
					REL::is_connected 
				> * = nullptr
			>
			class StrictTotalOrder : public StrictPartialOrder< REL > {
			};

		} // namespace internal

	} // namespace relations

} // namespace alp

#endif // _H_ALP_INTERNAL_RELATIONS_BASE

