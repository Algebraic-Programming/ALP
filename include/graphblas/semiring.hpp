
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
 * Provides an ALP semiring
 *
 * @author A. N. Yzelman
 * @date 15th of March, 2016
 */

#ifndef _H_GRB_SEMIRING
#define _H_GRB_SEMIRING

#include <graphblas/identities.hpp>
#include <graphblas/monoid.hpp>
#include <graphblas/ops.hpp>


namespace grb {

	/**
	 * A generalised semiring.
	 *
	 * This semiring works with the standard operators provided in grb::operators
	 * as well as with standard identities provided in grb::identities.
	 *
	 * \par Operators
	 *
	 * An operator \a OP here is of the form \f$ f:\ D_1 \times D_2 \to D_3 \f$;
	 * i.e., it has a fixed left-hand input domain, a fixed right-hand input
	 * domain, and a fixed output domain.
	 *
	 * A generalised semiring must include two operators; an additive operator,
	 * and a multiplicative one:
	 *   -# \f$ \oplus: \ D_1 \times D_2 \to D_3 \f$, and
	 *   -# \f$ \otimes:\ D_4 \times D_5 \to D_6 \f$.
	 *
	 * By convention, primitives such as grb::mxv will feed the output of the
	 * multiplicative operation to the additive operator as left-hand side input;
	 * hence, a valid semiring must have \f$ D_6 = D_1 \f$. Should the additive
	 * operator reduce several multiplicative outputs, the thus-far accumulated
	 * value will thus be passed as right-hand input to the additive operator;
	 * hence, a valid semiring must also have \f$ D_2 = D_3 \f$.
	 *
	 * If these constraints on the domains do not hold, attempted compilation will
	 * result in a clear error message.
	 *
	 * A semiring, in our definition here, thus in fact only defines four domains.
	 * We may thus rewrite the above definitions of the additive and multiplicative
	 * operators as:
	 *   -# \f$ \otimes:\ D_1 \times D_2 \to D_3 \f$, and
	 *   -# \f$ \oplus: \ D_3 \times D_4 \to D_4 \f$.
	 *
	 * \par Identities
	 *
	 * There are two identities that make up a generalised semiring: the zero-
	 * identity and the one-identity. These identities must be able to instantiate
	 * values for different domains, should indeed the four domains a generalised
	 * semiring operates on differ.
	 *
	 * Specifically, the zero-identity may be required for any of the domains the
	 * additive and multiplicative operators employ, whereas the one-identity may
	 * only be required for the domains the multiplicative operator employs.
	 *
	 * \par Standard examples
	 *
	 * An example of the standard semiring would be:
	 *    grb::Semiring<
	 *        grb::operators::add< double, double, double >,
	 *        grb::operators::mul< double, double, double >,
	 *        grb::identities::zero,
	 *        grb::identitites::one
	 *    > realSemiring;
	 * In this standard case, all domains the operators the semiring comprises are
	 * equal to one another. GraphBLAS supports the following shorthand for this
	 * special case:
	 *    grb::Semiring<
	 *        grb::operators::add< double >,
	 *        grb::operators::mul< double >,
	 *        grb::identities::zero,
	 *        grb::identities::one
	 *    > realSemiring;
	 *
	 * As another example, consider min-plus algebras. These may be used, for
	 * example, for deriving shortest paths through an edge-weighted graph:
	 *    grb::Semiring<
	 *        grb::operators::min< unsigned int >,
	 *        grb::operators::add< unsigned int >,
	 *        grb::identities::negative_infinity,
	 *        grb::identities::zero
	 *    > minPlus;
	 *
	 * \par CMonoid-categories
	 *
	 * While in these standard examples the relation to standard semirings as
	 * defined in mathematics apply, the possiblity of having differing domains
	 * that may not even be subsets of one another makes the above sketch
	 * generalisation incompatible with the standard notion of semirings.
	 *
	 * Our notion of a generalised semiring indeed is closer to what one might call
	 * CMonoid-categories, i.e. categories enriched in commutative monoids. Such
	 * CMonoid-categories are specified by some data, and are required to satisfy
	 * certain algebraic (equational) laws, thus being well-specified mathematical
	 * objects.
	 *
	 * Additionally, such CMonoid-categories encapsulate the definition of
	 * semirings, vector spaces, left modules and right modules.
	 *
	 * The full structure of a CMonoid-category C is specified by the data:
	 *
	 *  -# a set ob(C) of so-called objects,
	 *  -# for each pair of objects a,b in ob(C), a commutative monoid
	 *     (C(a,b), 0_{a,b}, +_{a,b}),
	 *  -# for each triple of objects a,b,c in ob(C), a multiplication operation
	 *     ._{a,b,c} : C(b,c) x C(a,b) -> C(a,c), and
	 *  -# for each object a in ob(C), a multiplicative identity 1_a in C(a,a).
	 *
	 * This data is then required to specify a list of algebraic laws that
	 * essentially capture:
	 *  -# (that the (C(a,b), 0_{a,b}, +_{a,b}) are commutative monoids)
	 *  -# joint associativity of the family of multiplication operators,
	 *  -# that the multiplicative identities 1_a are multiplicative identities,
	 *  -# that the family of multiplication operators ._{a,b,c} distributes over
	 *     the family of addition operators +_{a,b} on the left and on the right
	 *     in an appropriate sense, and
	 *  -# left and right annihilativity of the family of additive zeros 0_{a,b}.
	 *
	 * \par Generalised semirings in terms of CMonoid-categories
	 *
	 * The current notion of generalised semiring is specified by the following
	 * data:
	 *  -# operators OP1, OP2,
	 *  -# the four domains those operators are defined on,
	 *  -# an additive identity ID1, and
	 *  -# a multiplicative identity ID2.
	 *
	 * The four domains correspond to the choice of a CMonoid-category with two
	 * objects; e.g., \f$ ob(C)=\{a,b\} \f$. This gives rise to four possible
	 * pairings of the objects, including self-pairs, that correspond to the
	 * four different domains.
	 *
	 * CMonoid-categories then demand an additive operator must exist that
	 * operates purely within each of the four domains, when combined with a zero
	 * identity that likewise must exist in each of the four domains. None of
	 * these additive operators in fact matches with the generalised semiring's
	 * additive operator.
	 *
	 * CMonoid-categories also demand the existance of six different
	 * multiplicative operators that operate on three different domains each, that
	 * the composition of these operators is associative, that these operators
	 * distribute over the appropriate additive operators, and that there exists
	 * an multiplicative identity over at least one of the input domains.
	 *
	 * One of these six multiplicative operators is what appears in our generalised
	 * semiring. We seem to select exactly that multiplicative operator for which
	 * both input domains have an multiplicative identity.
	 *
	 * Finally, the identities corresponding to additive operators must act as
	 * annihilators over the matching multiplicative operators.
	 *
	 * Full details can be found in the git repository located here:
	 * https://gitlab.huaweirc.ch/abooij/semirings
	 *
	 * @tparam _OP1 The addition operator.
	 * @tparam _OP2 The multiplication operator.
	 * @tparam _ID1 The identity under addition (the `0').
	 * @tparam _ID2 The identity under multiplication (the `1').
	 */
	template<
		class _OP1, class _OP2,
		template< typename > class _ID1,
		template< typename > class _ID2
	>
	class Semiring {

		static_assert( std::is_same< typename _OP2::D3, typename _OP1::D1 >::value,
			"The multiplicative output type must match the left-hand additive "
			"input type" );

		static_assert( std::is_same< typename _OP1::D2, typename _OP1::D3 >::value,
			"The right-hand input type of the additive operator must match its "
			"output type" );

		static_assert( grb::is_associative< _OP1 >::value,
			"Cannot construct a semiring using a non-associative additive "
			"operator" );

		static_assert( grb::is_associative< _OP2 >::value,
			"Cannot construct a semiring using a non-associative multiplicative "
			"operator" );

		static_assert( grb::is_commutative< _OP1 >::value,
			"Cannot construct a semiring using a non-commutative additive "
			"operator" );

	public:

		/** The first input domain of the multiplicative operator. */
		typedef typename _OP2::D1 D1;

		/** The second input domain of the multiplicative operator. */
		typedef typename _OP2::D2 D2;

		/**
		 * The output domain of the multiplicative operator.
		 * The first input domain of the additive operator.
		 */
		typedef typename _OP2::D3 D3;

		/**
		 * The second input domain of the additive operator.
		 * The output domain of the additive operator.
		 */
		typedef typename _OP1::D2 D4;

		/** The additive operator type. */
		typedef _OP1 AdditiveOperator;

		/** The multiplicative operator type. */
		typedef _OP2 MultiplicativeOperator;

		/** The additive monoid type. */
		typedef Monoid< _OP1, _ID1 > AdditiveMonoid;

		/** The multiplicative monoid type. */
		typedef Monoid< _OP2, _ID2 > MultiplicativeMonoid;

		/** The identity under addition. */
		template< typename ZeroType >
		using Zero = _ID1< ZeroType >;

		/** The identity under multiplication. */
		template< typename OneType >
		using One = _ID2< OneType >;


	private:

		static constexpr size_t D1_bsz = grb::config::SIMD_BLOCKSIZE< D1 >::value();
		static constexpr size_t D2_bsz = grb::config::SIMD_BLOCKSIZE< D2 >::value();
		static constexpr size_t D3_bsz = grb::config::SIMD_BLOCKSIZE< D3 >::value();
		static constexpr size_t D4_bsz = grb::config::SIMD_BLOCKSIZE< D4 >::value();
		static constexpr size_t mul_input_bsz = D1_bsz < D2_bsz ? D1_bsz : D2_bsz;

		/** The additive monoid. */
		AdditiveMonoid additiveMonoid;

		/** The multiplicative monoid. */
		MultiplicativeMonoid multiplicativeMonoid;


	public:

		/** Blocksize for element-wise addition. */
		static constexpr size_t blocksize_add = D3_bsz < D4_bsz
			? D3_bsz
			: D4_bsz;

		/** Blocksize for element-wise multiplication. */
		static constexpr size_t blocksize_mul = mul_input_bsz < D3_bsz
			? mul_input_bsz
			: D3_bsz;

		/** Blocksize for element-wise multiply-adds. */
		static constexpr size_t blocksize = blocksize_mul < blocksize_add
			? blocksize_mul
			: blocksize_add;

		/**
		 * Retrieves the zero corresponding to this semiring. The zero value will be
		 * cast to the requested domain.
		 *
		 * @tparam D The requested domain of the zero. The arbitrary choice for the
		 *           default return type is \a D1-- inspired by the regularly
		 *           occurring expression \f$ a_{ij}x_j \f$ where often the left-
		 *           hand side is zero.
		 *
		 * @returns The zero corresponding to this semiring, cast to the requested
		 *          domain.
		 */
		template< typename D >
		D getZero() const {
			return additiveMonoid.template getIdentity< D >();
		}

		/**
		 * Sets the given value equal to one, corresponding to this semiring.
		 * The identity value will be cast to the requested domain.
		 *
		 * @tparam D The requested domain of the one. The arbitrary choice for the
		 *           default return type is \a D1-- the reasoning being to simply
		 *           have the same default type as getZero().
		 *
		 * @return The one corresponding to this semiring, cast to the requested
		 *         domain.
		 */
		template< typename D >
		D getOne() const {
			return multiplicativeMonoid.template getIdentity< D >();
		}

		/**
		 * Retrieves the underlying additive monoid.
		 *
		 * @return The underlying monoid. Any state is copied.
		 */
		AdditiveMonoid getAdditiveMonoid() const {
			return additiveMonoid;
		}

		/**
		 * Retrieves the underlying multiplicative monoid.
		 *
		 * @return The underlying monoid. Any state is copied.
		 */
		MultiplicativeMonoid getMultiplicativeMonoid() const {
			return multiplicativeMonoid;
		}

		/**
		 * Retrieves the underlying additive operator.
		 *
		 * @return The underlying operator. Any state is copied.
		 */
		AdditiveOperator getAdditiveOperator() const {
			return additiveMonoid.getOperator();
		}

		/**
		 * Retrieves the underlying multiplicative operator.
		 *
		 * @return The underlying operator. Any state is copied.
		 */
		MultiplicativeOperator getMultiplicativeOperator() const {
			return multiplicativeMonoid.getOperator();
		}

	};

	// overload for GraphBLAS type traits.
	template<
		class _OP1, class _OP2,
		template< typename > class _ID1,
		template< typename > class _ID2
	>
	struct is_semiring<
		Semiring< _OP1, _OP2, _ID1, _ID2 >
	> {
		/** This is a GraphBLAS semiring. */
		static const constexpr bool value = true;
	};

	template<
		class _OP1, class _OP2,
		template< typename > class _ID1,
		template< typename > class _ID2
	>
	struct has_immutable_nonzeroes<
		Semiring< _OP1, _OP2, _ID1, _ID2 >
	> {
		static const constexpr bool value = grb::is_semiring<
			Semiring< _OP1, _OP2, _ID1, _ID2 > >::value &&
			std::is_same<
				_OP1, typename grb::operators::logical_or< typename _OP1::D1,
				typename _OP1::D2, typename _OP1::D3
			> >::value;

	};

	// after all of the standard definitions, declare some standard semirings

	/**
	 * A name space that contains a set of standard semirings.
	 *
	 * Standard semirings include:
	 *  - #plusTimes, for numerical linear algebra
	 *  - #minPlus, for, e.g., shortest-path graph queries
	 *  - #maxPlus, for, e.g., longest-path graph queries
	 *  - #minTimes, for, e.g., least-reliable-path graph queries
	 *  - #maxTimes, for, e.g., most-reliable-path graph queries,
	 *  - #boolean, for, e.g., reachability graph queries.
	 *  - etc.
	 *
	 * A list of all pre-defined semirings, in addition to the above, follows:
	 * #minMax, #maxMin, #plusMin, #lorLand, #landLor, #lxorLand, #lxnorLor,
	 * #lneqLand, and #leqLor.
	 *
	 * \note Here, lor stands for logical-or and land stands for logical-and, while
	 *       ne stands for not-equal and eq for equal.
	 *
	 * \note The #lorLand semiring over the Boolean domains is the same as the
	 *       #boolean semiring.
	 *
	 * \note The #lxorLand semiring is the same as the #lneqLand semiring.
	 *
	 * \note The #lxnorLor semiring is the same as the #leqLor semiring.
	 *
	 * \warning Some of these pre-defined semirings are not proper semirings over
	 *          all domains. For example, the #maxPlus semiring over unsigned
	 *          integers would have both max and + identities be zero, and thus
	 *          cannot act as an annihilator over +.
	 *
	 * \warning While ALP does a best-effort in catching erroneous semirings, by
	 *          virtue of templates it cannot catch all erroneous semirings. E.g.,
	 *          continuing the above #maxPlus semiring example: even if ALP
	 *          prevents the definition of #maxPlus semirings over unsigned types
	 *          by relying on the <tt>std::is_unsigned</tt> type trait, a user
	 *          could still define their own unsigned integer type that erroneously
	 *          overloads this type trait to <tt>false</tt>. We cannot catch such
	 *          errors and consider those programming errors.
	 *
	 * \note We do not pre-define any improper semiring, such as plusMin, that do
	 *       appear in the GraphBLAS C specification. Instead, ALP has, for every
	 *       primitive that takes a semiring, a variant of that primitive that
	 *       instead of a semiring, takes 1) a cummutative monoid as an additive
	 *       operator, and 2) any binary operator as the multiplicative operator.
	 *       These variants do not (and may not) rely on the additive identity
	 *       being an annihilator over the multiplicative operation, and do not
	 *       (may not) rely on any distributive property over the two operations.
	 *
	 * Each semiring except #boolean takes up to four domains as template
	 * arguments, while semirings as a pure mathematical concept take only a single
	 * domain. The first three domains indicate the left-hand input domain, the
	 * right-hand input domain, and the output domain of the multiplicative monoid,
	 * respectively. The third and fourth domains indicate the left-hand and right-
	 * hand input domain of the additive monoid. The fourth domain also indicates
	 * the output domain of the additive monoid.
	 *
	 * \note This particular extension of semirings to four domains is rooted in
	 *       C-Monoid categories. All useful mixed-domain semirings ALP has
	 *       presently been applied with are C-Monoid categories, while since
	 *       assuming this underlying algebra, the ALP code base that relates to
	 *       algebraic structures, algebraic type traits, and their application,
	 *       has simplified significantly.
	 */
	namespace semirings {

		/**
		 * The plusTimes semiring.
		 *
		 * Uses \em addition (plus) as the additive commutative monoid and
		 * \em multiplication (times) as the multiplicative monoid. The identities
		 * for each monoid are zero and one, respectively.
		 *
		 * The three domains of the multiplicative monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the multiplicative monoid
		 * @tparam D2 The right-hand input domain of the multiplicative monoid
		 * @tparam D3 The output domain of the multiplicative monoid
		 *
		 * The domains of the additive monoid are \a D3 and:
		 *
		 * @tparam D4 The right-hand input domain of the additive monoid, as well as
		 *            the output domain of the additive monoid.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2, typename D4 = D3 >
		using plusTimes = grb::Semiring<
			grb::operators::add< D3, D4, D4 >,
			grb::operators::mul< D1, D2, D3 >,
			grb::identities::zero, grb::identities::one
		>;

		/**
		 * The minPlus semiring.
		 *
		 * Uses \em min as the additive commutative monoid and \em addition as the
		 * multiplicative monoid. The identities for each monoid are \f$ \infty \f$
		 * and zero, respectively.
		 *
		 * The three domains of the multiplicative monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the multiplicative monoid
		 * @tparam D2 The right-hand input domain of the multiplicative monoid
		 * @tparam D3 The output domain of the multiplicative monoid
		 *
		 * The domains of the additive monoid are \a D3 and:
		 *
		 * @tparam D4 The right-hand input domain of the additive monoid, as well as
		 *            the output domain of the additive monoid.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2, typename D4 = D3 >
		using minPlus = grb::Semiring<
			grb::operators::min< D3, D4, D4 >,
			grb::operators::add< D1, D2, D3 >,
			grb::identities::infinity, grb::identities::zero
		>;

		/**
		 * The maxPlus semiring.
		 *
		 * Uses \em max as the additive commutative monoid and \em addition as the
		 * multiplicative monoid. The identities for each monoid are \f$ -\infty \f$
		 * and zero, respectively.
		 *
		 * The three domains of the multiplicative monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the multiplicative monoid
		 * @tparam D2 The right-hand input domain of the multiplicative monoid
		 * @tparam D3 The output domain of the multiplicative monoid
		 *
		 * The domains of the additive monoid are \a D3 and:
		 *
		 * @tparam D4 The right-hand input domain of the additive monoid, as well as
		 *            the output domain of the additive monoid.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2, typename D4 = D3 >
		using maxPlus = grb::Semiring<
			grb::operators::max< D3, D4, D4 >,
			grb::operators::add< D1, D2, D3 >,
			grb::identities::negative_infinity, grb::identities::zero
		>;

		/**
		 * The minTimes semiring.
		 *
		 * Uses \em min as the additive commutative monoid and \em multiplication as
		 * the multiplicative monoid. The identities for each monoid are
		 * \f$ \infty \f$ and one, respectively.
		 *
		 * The three domains of the multiplicative monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the multiplicative monoid
		 * @tparam D2 The right-hand input domain of the multiplicative monoid
		 * @tparam D3 The output domain of the multiplicative monoid
		 *
		 * The domains of the additive monoid are \a D3 and:
		 *
		 * @tparam D4 The right-hand input domain of the additive monoid, as well as
		 *            the output domain of the additive monoid.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2, typename D4 = D3 >
		using minTimes = grb::Semiring<
			grb::operators::min< D3, D4, D4 >,
			grb::operators::mul< D1, D2, D3 >,
			grb::identities::infinity, grb::identities::one
		>;

		/**
		 * The maxTimes semiring.
		 *
		 * Uses \em max as the additive commutative monoid and \em multiplication as
		 * the multiplicative monoid. The identities for each monoid are
		 * \f$ -infty \f$ and one, respectively.
		 *
		 * The three domains of the multiplicative monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the multiplicative monoid
		 * @tparam D2 The right-hand input domain of the multiplicative monoid
		 * @tparam D3 The output domain of the multiplicative monoid
		 *
		 * The domains of the additive monoid are \a D3 and:
		 *
		 * @tparam D4 The right-hand input domain of the additive monoid, as well as
		 *            the output domain of the additive monoid.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2, typename D4 = D3 >
		using maxTimes = grb::Semiring<
			grb::operators::max< D3, D4, D4 >,
			grb::operators::mul< D1, D2, D3 >,
			grb::identities::negative_infinity, grb::identities::one
		>;

		/**
		 * The minMax semiring.
		 *
		 * Uses \em min as the additive commutative monoid and \em max as the
		 * multiplicative monoid. The identities for each monoid are \f$ \infty \f$
		 * and \f$ -\infty \f$, respectively.
		 *
		 * The three domains of the multiplicative monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the multiplicative monoid
		 * @tparam D2 The right-hand input domain of the multiplicative monoid
		 * @tparam D3 The output domain of the multiplicative monoid
		 *
		 * The domains of the additive monoid are \a D3 and:
		 *
		 * @tparam D4 The right-hand input domain of the additive monoid, as well as
		 *            the output domain of the additive monoid.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2, typename D4 = D3 >
		using minMax = grb::Semiring<
			grb::operators::min< D3, D4, D4 >,
			grb::operators::max< D1, D2, D3 >,
			grb::identities::infinity, grb::identities::negative_infinity
		>;

		/**
		 * The maxMin semiring.
		 *
		 * Uses \em max as the additive commutative monoid and \em min as the
		 * multiplicative monoid. The identities for each monoid are \f$ -\infty \f$
		 * and \f$ \infty \f$, respectively.
		 *
		 * The three domains of the multiplicative monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the multiplicative monoid
		 * @tparam D2 The right-hand input domain of the multiplicative monoid
		 * @tparam D3 The output domain of the multiplicative monoid
		 *
		 * The domains of the additive monoid are \a D3 and:
		 *
		 * @tparam D4 The right-hand input domain of the additive monoid, as well as
		 *            the output domain of the additive monoid.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2, typename D4 = D3 >
		using maxMin = grb::Semiring<
			grb::operators::max< D3, D4, D4 >,
			grb::operators::min< D1, D2, D3 >,
			grb::identities::negative_infinity, grb::identities::infinity
		>;

		/**
		 * The plusMin semiring.
		 *
		 * Uses \em plus as the additive commutative monoid and \em min as the
		 * multiplicative monoid. The identities for each monoid are \f$ 0 \f$ and
		 * \f$ \infty \f$, respectively.
		 *
		 * The three domains of the multiplicative monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the multiplicative monoid
		 * @tparam D2 The right-hand input domain of the multiplicative monoid
		 * @tparam D3 The output domain of the multiplicative monoid
		 *
		 * The domains of the additive monoid are \a D3 and:
		 *
		 * @tparam D4 The right-hand input domain of the additive monoid, as well as
		 *            the output domain of the additive monoid.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2, typename D4 = D3 >
		using plusMin = grb::Semiring<
			grb::operators::add< D3, D4, D4 >,
			grb::operators::min< D1, D2, D3 >,
			grb::identities::zero, grb::identities::infinity
		>;

		/**
		 * The logical-or logical-and semiring.
		 *
		 * Uses \em or as the additive commutative monoid and \em and as the
		 * multiplicative monoid. The identities for each monoid are \em false and
		 * \em true, respectively.
		 *
		 * The three domains of the multiplicative monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the multiplicative monoid
		 * @tparam D2 The right-hand input domain of the multiplicative monoid
		 * @tparam D3 The output domain of the multiplicative monoid
		 *
		 * The domains of the additive monoid are \a D3 and:
		 *
		 * @tparam D4 The right-hand input domain of the additive monoid, as well as
		 *            the output domain of the additive monoid.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2, typename D4 = D3 >
		using lorLand = grb::Semiring<
			grb::operators::logical_or< D3, D4, D4 >,
			grb::operators::logical_and< D1, D2, D3 >,
			grb::identities::logical_false, grb::identities::logical_true
		>;

		/**
		 * The Boolean semiring.
		 *
		 * Uses \em or as the additive commutative monoid and \em and as the
		 * multiplicative monoid. The identities for each monoid are \em false and
		 * \em true, respectively. All domains are fixed to <tt>bool</tt>.
		 */
		using boolean = lorLand< bool >;

		/**
		 * The logical-and logical-or semiring.
		 *
		 * Uses \em and as the additive commutative monoid and \em or as the
		 * multiplicative monoid. The identities for each monoid are \em true and
		 * em false, respectively.
		 *
		 * The three domains of the multiplicative monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the multiplicative monoid
		 * @tparam D2 The right-hand input domain of the multiplicative monoid
		 * @tparam D3 The output domain of the multiplicative monoid
		 *
		 * The domains of the additive monoid are \a D3 and:
		 *
		 * @tparam D4 The right-hand input domain of the additive monoid, as well as
		 *            the output domain of the additive monoid.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2, typename D4 = D3 >
		using landLor = grb::Semiring<
			grb::operators::logical_and< D3, D3, D4 >,
			grb::operators::logical_or< D1, D2, D3 >,
			grb::identities::logical_true, grb::identities::logical_false
		>;

		/**
		 * The exclusive-logical-or logical-and semiring.
		 *
		 * Uses <em>not-equals</em> as the additive commutative monoid and logical-and
		 * as the multiplicative monoid. The identities for each monoid are \em false
		 * and \em true, respectively.
		 *
		 * The three domains of the multiplicative monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the multiplicative monoid
		 * @tparam D2 The right-hand input domain of the multiplicative monoid
		 * @tparam D3 The output domain of the multiplicative monoid
		 *
		 * The domains of the additive monoid are \a D3 and:
		 *
		 * @tparam D4 The right-hand input domain of the additive monoid, as well as
		 *            the output domain of the additive monoid.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2, typename D4 = D3 >
		using lxorLand = grb::Semiring<
			grb::operators::not_equal< D3, D3, D4 >,
			grb::operators::logical_and< D1, D2, D3 >,
			grb::identities::logical_false, grb::identities::logical_true
		>;

		/**
		 * The not-equals logical-and semiring.
		 *
		 * Uses <em>not-equal</em> as the additive commutative monoid and \em and as
		 * the multiplicative monoid. The identities for each monoid are \em false and
		 * \em true, respectively.
		 *
		 * The three domains of the multiplicative monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the multiplicative monoid
		 * @tparam D2 The right-hand input domain of the multiplicative monoid
		 * @tparam D3 The output domain of the multiplicative monoid
		 *
		 * The domains of the additive monoid are \a D3 and:
		 *
		 * @tparam D4 The right-hand input domain of the additive monoid, as well as
		 *            the output domain of the additive monoid.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2, typename D4 = D3 >
		using lneqLand = lxorLand< D1, D2, D3, D4 >;

		/**
		 * The negated-exclusive-or logical-or semring.
		 *
		 * Uses <em>negated exclusive or</em> as the additive commutative monoid and
		 * \em or as the multiplicative monoid. The identities for each monoid are
		 * \em true and \em false, respectively.
		 *
		 * The three domains of the multiplicative monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the multiplicative monoid
		 * @tparam D2 The right-hand input domain of the multiplicative monoid
		 * @tparam D3 The output domain of the multiplicative monoid
		 *
		 * The domains of the additive monoid are \a D3 and:
		 *
		 * @tparam D4 The right-hand input domain of the additive monoid, as well as
		 *            the output domain of the additive monoid.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2, typename D4 = D3 >
		using lxnorLor = grb::Semiring<
			grb::operators::equal< D3, D4, D4 >,
			grb::operators::logical_or< D1, D2, D3 >,
			grb::identities::logical_true, grb::identities::logical_false
		>;

		/**
		 * The equals logical-or semiring.
		 *
		 * Uses \em equals as the additive commutative monoid and \em or as the
		 * multiplicative monoid. The identities for each monoid are \em true and
		 * \em false, respectively.
		 *
		 * The three domains of the multiplicative monoid are:
		 *
		 * @tparam D1 The left-hand input domain of the multiplicative monoid
		 * @tparam D2 The right-hand input domain of the multiplicative monoid
		 * @tparam D3 The output domain of the multiplicative monoid
		 *
		 * The domains of the additive monoid are \a D3 and:
		 *
		 * @tparam D4 The right-hand input domain of the additive monoid, as well as
		 *            the output domain of the additive monoid.
		 */
		template< typename D1, typename D2 = D1, typename D3 = D2, typename D4 = D3 >
		using leqLor = lxnorLor< D1, D2, D3, D4 >;

	}

} // namespace grb

#endif

