
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
#include <alp/type_traits.hpp>


namespace alp {

	namespace relations {

		/** Core implementations of the standard relations in #alp::relations. */
		namespace internal {

			/**
			 * Standard less-than (\a lt) operator.
			 *
			 * Assumes native availability of operator< on the given data types 
			 * or assumes that the relevant operators are properly overloaded.
			 *
			 * Assumes that \a lt is a strict total order. Non-standard/non-matching 
			 * data types or non-standard (overloaded) \a operator< should 
			 * therefore be used with caution.
			 *
			 * @tparam SET The input data type.
			 */
			template< typename SET, enum Backend implementation = config::default_backend >
			class lt {

				public:
					/** Alias to the domain data type. */
					typedef SET domain;

					/** Alias to the codomain data type. */
					typedef SET codomain;

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
					 * This function checks if <em> a < b </em>.
					 *
					 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
					 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
					 *
					 * \warning Passing invalid pointers will result in UB.
					 */
					static bool check( const domain * const a,
						const codomain * const b
					) {
						return *a < *b;
					}
			};

			/**
			 * Standard greater-than (\a gt) operator.
			 *
			 * Assumes native availability of \a operator> on the given data types 
			 * or assumes that the relevant operators are properly overloaded.
			 *
			 * Assumes that \a gt is a strict total order. Non-standard/non-matching 
			 * data types or non-standard (overloaded) \a operator> should 
			 * therefore be used with caution.
			 *
			 * @tparam SET The input data type.
			 */
			template< typename SET, enum Backend implementation = config::default_backend >
			class gt {

				public:
					/** Alias to the domain data type. */
					typedef SET domain;

					/** Alias to the codomain data type. */
					typedef SET codomain;

					/**
					 * Whether this relation is \em reflexive; that is,
					 * for all \a a in \a SET, \f$ a > a \f$.
					 */
					static constexpr bool is_reflexive = false;

					/**
					 * Whether this relation is \em irreflexive; that is,
					 * for all \a a in \a SET, not \f$ a > a \f$.
					 */
					static constexpr bool is_irreflexive = true;

					/**
					 * Whether this relation is \em symmetric; that is,
					 * for all \a a, \a b in \a SET, 
					 * if \f$ a > b \f$ then \f$ b > a \f$.
					 */
					static constexpr bool is_symmetric = false;

					/**
					 * Whether this relation is \em antisymmetric; that is,
					 * for all \a a, \a b in \a SET, if \f$ a > b \f$ and 
					 * \f$ b > a \f$ then \f$ a = b \f$.
					 */
					static constexpr bool is_antisymmetric = true;

					/**
					 * Whether this relation is \em transitive; that is,
					 * for all \a a, \a b, \a c in \a SET, if \f$ a > b \f$ and
					 * \f$ b > c \f$ then \f$ a > c \f$.
					 */
					static constexpr bool is_transitive = true;

					/**
					 * Whether this relation is \em connected (or total); that is,
					 * for all \a a, \a b in \a SET, if \f$ a \neq b \f$ then
					 * either \f$ a > b \f$ or \f$ b > a \f$.
					 */
					static constexpr bool is_connected = true;

					/**
					 * Whether this relation is <em> strongly connected </em>; 
					 * that is,
					 * for all \a a, \a b in \a SET, 
					 * either \f$ a > b \f$ or \f$ b > a \f$.
					 */
					static constexpr bool is_strongly_connected = false;

					/**
					 * This function checks if <em> a > b </em>.
					 *
					 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
					 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
					 *
					 * \warning Passing invalid pointers will result in UB.
					 */
					static bool check( const domain * const a,
						const codomain * const b
					) {
						return *a > *b;
					}
			};

			/**
			 * Standard equal (\a eq) relation.
			 *
			 * Assumes native availability of \a operator== on the given data types 
			 * or assumes that the relevant operators are properly overloaded.
			 *
			 * Assumes that \a eq is an equivalence relation. 
			 * Non-standard/non-matching data types or non-standard (overloaded) 
			 * \a operator== should therefore be used with caution.
			 *
			 * @tparam SET The input data type.
			 */
			template< typename SET, enum Backend implementation = config::default_backend >
			class eq {

				public:
					/** Alias to the domain data type. */
					typedef SET domain;

					/** Alias to the codomain data type. */
					typedef SET codomain;

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
					 * This function checks if <em> a == b </em>.
					 *
					 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
					 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
					 *
					 * \warning Passing invalid pointers will result in UB.
					 */
					static bool check( const domain * const a,
						const codomain * const b
					) {
						return *a == *b;
					}
			};

			/**
			 * Standard not-equal (\a neq) operator.
			 *
			 * Assumes native availability of \a operator!= on the given data types 
			 * or assumes that the relevant operators are properly overloaded.
			 *
			 * While \a neq does not require two values to be members of
			 * an ordered set, the relation is still assumed to be irreflexive, 
			 * symmetric and connected. 
			 * Non-standard/non-matching data types or non-standard (overloaded) 
			 * \a operator!= should therefore be used with caution.
			 *
			 * @tparam SET The input data type.
			 */
			template< typename SET, enum Backend implementation = config::default_backend >
			class neq {

				public:
					/** Alias to the domain data type. */
					typedef SET domain;

					/** Alias to the codomain data type. */
					typedef SET codomain;

					/**
					 * Whether this relation is \em reflexive; that is,
					 * for all \a a in \a SET, \f$ a \neq a \f$.
					 */
					static constexpr bool is_reflexive = false;

					/**
					 * Whether this relation is \em irreflexive; that is,
					 * for all \a a in \a SET, not \f$ a \neq a \f$.
					 */
					static constexpr bool is_irreflexive = true;

					/**
					 * Whether this relation is \em symmetric; that is,
					 * for all \a a, \a b in \a SET, 
					 * if \f$ a \neq b \f$ then \f$ b \neq a \f$.
					 */
					static constexpr bool is_symmetric = true;

					/**
					 * Whether this relation is \em antisymmetric; that is,
					 * for all \a a, \a b in \a SET, if \f$ a \neq b \f$ and 
					 * \f$ b \neq a \f$ then \f$ a = b \f$.
					 */
					static constexpr bool is_antisymmetric = false;

					/**
					 * Whether this relation is \em transitive; that is,
					 * for all \a a, \a b, \a c in \a SET, if \f$ a \neq b \f$ and
					 * \f$ b \neq c \f$ then \f$ a \neq c \f$.
					 */
					static constexpr bool is_transitive = false;

					/**
					 * Whether this relation is \em connected; that is,
					 * for all \a a, \a b in \a SET, if \f$ a \neq b \f$ then
					 * either \f$ a \neq b \f$ or \f$ b \neq a \f$.
					 */
					static constexpr bool is_connected = true;

					/**
					 * Whether this relation is <em> strongly connected </em> (or total); 
					 * that is,
					 * for all \a a, \a b in \a SET, 
					 * either \f$ a \neq b \f$ or \f$ b \neq a \f$.
					 */
					static constexpr bool is_strongly_connected = false;

					/**
					 * This function checks if <em> a != b </em>.
					 *
					 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
					 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
					 *
					 * \warning Passing invalid pointers will result in UB.
					 */
					static bool check( const domain * const a,
						const codomain * const b
					) {
						return *a != *b;
					}
			};

			/**
			 * Standard less-than-or-equal (\a le) operator.
			 *
			 * Assumes native availability of \a operator<= on the given data types 
			 * or assumes that the relevant operators are properly overloaded.
			 *
			 * Assumes that \a le is a total order. Non-standard/non-matching 
			 * data types or non-standard (overloaded) \a operator<= should 
			 * therefore be used with caution.
			 *
			 * @tparam SET The input data type.
			 */
			template< typename SET, enum Backend implementation = config::default_backend >
			class le {

				public:
					/** Alias to the domain data type. */
					typedef SET domain;

					/** Alias to the codomain data type. */
					typedef SET codomain;

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
					 * This function checks if <em> a <= b </em>.
					 *
					 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
					 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
					 *
					 * \warning Passing invalid pointers will result in UB.
					 */
					static bool check( const domain * const a,
						const codomain * const b
					) {
						return *a <= *b;
					}
			};

			/**
			 * Standard greater-than-or-equal (\a ge) operator.
			 *
			 * Assumes native availability of \a operator>= on the given data types 
			 * or assumes that the relevant operators are properly overloaded.
			 *
			 * Assumes that \a ge is a total order. Non-standard/non-matching 
			 * data types or non-standard (overloaded) \a operator>= should 
			 * therefore be used with caution.
			 *
			 * @tparam SET The input data type.
			 */
			template< typename SET, enum Backend implementation = config::default_backend >
			class ge {

				public:
					/** Alias to the domain data type. */
					typedef SET domain;

					/** Alias to the codomain data type. */
					typedef SET codomain;

					/**
					 * Whether this relation is \em reflexive; that is,
					 * for all \a a in \a SET, \f$ a \ge a \f$.
					 */
					static constexpr bool is_reflexive = true;

					/**
					 * Whether this relation is \em irreflexive; that is,
					 * for all \a a in \a SET, not \f$ a \ge a \f$.
					 */
					static constexpr bool is_irreflexive = false;

					/**
					 * Whether this relation is \em symmetric; that is,
					 * for all \a a, \a b in \a SET, 
					 * if \f$ a \ge b \f$ then \f$ b \ge a \f$.
					 */
					static constexpr bool is_symmetric = false;

					/**
					 * Whether this relation is \em antisymmetric; that is,
					 * for all \a a, \a b in \a SET, if \f$ a \ge b \f$ and 
					 * \f$ b \ge a \f$ then \f$ a = b \f$.
					 */
					static constexpr bool is_antisymmetric = true;

					/**
					 * Whether this relation is \em transitive; that is,
					 * for all \a a, \a b, \a c in \a SET, if \f$ a \ge b \f$ and
					 * \f$ b \ge c \f$ then \f$ a \ge c \f$.
					 */
					static constexpr bool is_transitive = true;

					/**
					 * Whether this relation is \em connected; that is,
					 * for all \a a, \a b in \a SET, if \f$ a \neq b \f$ then
					 * either \f$ a \ge b \f$ or \f$ b \ge a \f$.
					 */
					static constexpr bool is_connected = true;

					/**
					 * Whether this relation is <em> strongly connected </em> (or total); 
					 * that is,
					 * for all \a a, \a b in \a SET, 
					 * either \f$ a \ge b \f$ or \f$ b \ge a \f$.
					 */
					static constexpr bool is_strongly_connected = true;

					/**
					 * This function checks if <em> a >= b </em>.
					 *
					 * @param[in]  a The left-hand side input. Must be pre-allocated and initialised.
					 * @param[in]  b The right-hand side input. Must be pre-allocated and initialised.
					 *
					 * \warning Passing invalid pointers will result in UB.
					 */
					static bool check( const domain * const a,
						const codomain * const b
					) {
						return *a >= *b;
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

				public:

					/** The domain type. */
					typedef typename REL::domain D1;

					/** The codomain type. */
					typedef typename REL::codomain D2;

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
					 * This function checks if \f$ x REL y \f$.
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
					static bool check( const InputType1 & x, const InputType2 & y ) {
						const D1 a = static_cast< D1 >( x );
						const D2 b = static_cast< D2 >( y );
						return REL::check( &a, &b );
					}

					/**
					 * This is the high-performance version of check() in the sense that no
					 * casting is required. This version will be automatically called whenever
					 * possible.
					 */
					static bool check( const D1 & x, const D2 & y ) {
						return REL::check( &x, &y );
					}
			};

			/**
			 * This is the relation interface exposed to the ALP implementation.
			 *
			 * This class wraps around a base relation of type \a REL we denote by
			 *        \f$ REL \subseteq D_1\times D_2 \f$.
			 *
			 * \parblock
			 * \par Base Operators
			 *
			 * The class \a REL is expected to define the following public function:
			 *   - \a check, which takes two pointers to parameters \f$ a \in D_1 \f$
			 *      and \f$ b \in D_2 \f$ and checks if 
			 *      \f$ a REL b \f$.
			 *
			 * It is also expected to define the following types:
			 *   - \a domain, which corresponds to \f$ D_1 \f$,
			 *   - \a codomain, which corresponds to \f$ D_2 \f$.
			 *
			 * It is also expected to define the following public boolean fields:
			 *   - \a is_reflexive
			 *   - \a is_irreflexive
			 *   - \a is_symmetric
			 *   - \a is_antisymmetric
			 *   - \a is_transitive
			 *   - \a is_connected
			 *   - \a is_strongly_connected
			 *
			 * For an example of base relation, see alp::relations::internal::lt.
			 *
			 * \note ALP users should never access these classes directly. This
			 *       documentation is provided for developers to understand or extend
			 *       the current implementation, for example to include new relations.
			 *
			 * \endparblock
			 *
			 * \parblock
			 * \par The exposed GraphBLAS Relation Interface
			 *
			 * The Base Relations as illustrated above are wrapped by this class to
			 * provide a more convient API. It translates the functionality of any Base
			 * Relation and exposes the following interface instead:
			 *
			 *   -# check, which takes two parameters \f$ a, b \f$ of arbitrary
			 *      types and checks \f$ a REL b \f$ after performing any
			 *      casting if required.
			 * \endparblock
			 *
			 * \note This class only allows wrapping of stateless base relations. This
			 *       ALP implementation in principle allows for stateful
			 *       relations, though they must be provided by a specialised class
			 *       which directly implements the above public interface.
			 *
			 * @see RelationBase::check
			 *
			 * \parblock
			 * \par Providing New Relations
			 *
			 * New relations are easily added to this
			 * ALP implementation by providing a base relation and wrapping this
			 * class around it, as illustrated, e.g., by alp::relations::lt as follows:
			 *
			 * \snippet rels.hpp Relation Wrapping
			 *
			 * This need to be compatible with the ALP type traits, specifically,
			 * the #is_relation template. To ensure this, a specialisation of it must be
			 * privided:
			 *
			 * \snippet rels.hpp Relation Type Traits
			 * \endparblock
			 */
			template< typename REL, enum Backend implementation = config::default_backend >
			class Relation : public RelationBase< REL, implementation > {

				// public:
				// 	typedef typename RelationBase< REL, implementation >::D1 D1;
				// 	typedef typename RelationBase< REL, implementation >::D2 D2;

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
						typename REL::domain, 
						typename REL::codomain 
					>::value 
				> * = nullptr
			>
			class HomogeneousRelation : public Relation< REL, implementation > {
			};

		} // namespace internal

	} // namespace relations

	template< typename Rel >
	struct is_homogeneous_relation {
		static const constexpr bool value = is_relation< Rel >::value
			and std::is_same< typename Rel::D1, typename Rel::D2 >::value;
	};

	template< typename Rel >
	struct is_reflexive {
		static const constexpr bool value = is_homogeneous_relation< Rel >::value
			and Rel::is_reflexive();
	};

	template< typename Rel >
	struct is_irreflexive {
		static const constexpr bool value = is_homogeneous_relation< Rel >::value
			and Rel::is_irreflexive();
	};

	template< typename Rel >
	struct is_symmetric {
		static const constexpr bool value = is_homogeneous_relation< Rel >::value
			and Rel::is_symmetric();
	};

	template< typename Rel >
	struct is_antisymmetric {
		static const constexpr bool value = is_homogeneous_relation< Rel >::value
			and Rel::is_antisymmetric();
	};

	template< typename Rel >
	struct is_transitive {
		static const constexpr bool value = is_homogeneous_relation< Rel >::value
			and Rel::is_transitive();
	};

	template< typename Rel >
	struct is_connected {
		static const constexpr bool value = is_homogeneous_relation< Rel >::value
			and Rel::is_connected();
	};

	template< typename Rel >
	struct is_strongly_connected {
		static const constexpr bool value = is_homogeneous_relation< Rel >::value
			and Rel::is_strongly_connected();
	};

	template< typename Rel >
	struct is_asymmetric {
		static const constexpr bool value = is_irreflexive< Rel >::value
			and is_antisymmetric< Rel >::value;
	};

	template< typename Rel >
	struct is_partial_order {
		static const constexpr bool value = is_reflexive< Rel >::value
			and is_antisymmetric< Rel >::value
			and is_transitive< Rel >::value;
	};

	template< typename Rel >
	struct is_strict_partial_order {
		static const constexpr bool value = is_asymmetric< Rel >::value
			and is_transitive< Rel >::value;
	};

	template< typename Rel >
	struct is_total_order {
		static const constexpr bool value = is_partial_order< Rel >::value
			and is_strongly_connected< Rel >::value;
	};

	template< typename Rel >
	struct is_strict_total_order {
		static const constexpr bool value = is_strict_partial_order< Rel >::value
			and is_connected< Rel >::value;
	};

	template< typename Rel >
	struct is_equivalence_relation {
		static const constexpr bool value = is_reflexive< Rel >::value
			and is_symmetric< Rel >::value
			and is_transitive< Rel >::value;
	};

} // namespace alp

#endif // _H_ALP_INTERNAL_RELATIONS_BASE

