
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
 * @date 15th of March, 2016
 */

#ifndef _H_ALP_SEMIRING
#define _H_ALP_SEMIRING

#include <alp/identities.hpp>
#include <alp/monoid.hpp>
#include <alp/ops.hpp>

/**
 * The main GraphBLAS namespace.
 */
namespace alp {

	/**
	 * A generalised semiring.
	 *
	 * This semiring works with the standard operators provided in alp::operators
	 * as well as with standard identities provided in alp::identities.
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
	 * By convention, primitives such as alp::mxv will feed the output of the
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
	 *    alp::Semiring<
	 *        alp::operators::add< double, double, double >,
	 *        alp::operators::mul< double, double, double >,
	 *        alp::identities::zero,
	 *        alp::identitites::one
	 *    > realSemiring;
	 * In this standard case, all domains the operators the semiring comprises are
	 * equal to one another. GraphBLAS supports the following shorthand for this
	 * special case:
	 *    alp::Semiring<
	 *        alp::operators::add< double >,
	 *        alp::operators::mul< double >,
	 *        alp::identities::zero,
	 *        alp::identities::one
	 *    > realSemiring;
	 *
	 * As another example, consider min-plus algebras. These may be used, for
	 * example, for deriving shortest paths through an edge-weighted graph:
	 *    alp::Semiring<
	 *        alp::operators::min< unsigned int >,
	 *        alp::operators::add< unsigned int >,
	 *        alp::identities::negative_infinity,
	 *        alp::identities::zero
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
	template< class _OP1, class _OP2, template< typename > class _ID1, template< typename > class _ID2 >
	class Semiring {

		static_assert( std::is_same< typename _OP2::D3, typename _OP1::D1 >::value,
			"The multiplicative output type must match the left-hand additive "
			"input type" );

		static_assert( std::is_same< typename _OP1::D2, typename _OP1::D3 >::value,
			"The right-hand input type of the additive operator must match its "
			"output type" );

		static_assert( alp::is_associative< _OP1 >::value,
			"Cannot construct a semiring using a non-associative additive "
			"operator" );

		static_assert( alp::is_associative< _OP2 >::value,
			"Cannot construct a semiring using a non-associative multiplicative "
			"operator" );

		static_assert( alp::is_commutative< _OP1 >::value,
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
		static constexpr size_t D1_bsz = alp::config::SIMD_BLOCKSIZE< D1 >::value();
		static constexpr size_t D2_bsz = alp::config::SIMD_BLOCKSIZE< D2 >::value();
		static constexpr size_t D3_bsz = alp::config::SIMD_BLOCKSIZE< D3 >::value();
		static constexpr size_t D4_bsz = alp::config::SIMD_BLOCKSIZE< D4 >::value();
		static constexpr size_t mul_input_bsz = D1_bsz < D2_bsz ? D1_bsz : D2_bsz;

		/** The additive monoid. */
		AdditiveMonoid additiveMonoid;

		/** The multiplicative monoid. */
		MultiplicativeMonoid multiplicativeMonoid;

	public:
		/** Blocksize for element-wise addition. */
		static constexpr size_t blocksize_add = D3_bsz < D4_bsz ? D3_bsz : D4_bsz;

		/** Blocksize for element-wise multiplication. */
		static constexpr size_t blocksize_mul = mul_input_bsz < D3_bsz ? mul_input_bsz : D3_bsz;

		/** Blocksize for element-wise multiply-adds. */
		static constexpr size_t blocksize = blocksize_mul < blocksize_add ? blocksize_mul : blocksize_add;

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
		constexpr D getZero() const {
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
		constexpr D getOne() const {
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
	template< class _OP1, class _OP2, template< typename > class _ID1, template< typename > class _ID2 >
	struct is_semiring< Semiring< _OP1, _OP2, _ID1, _ID2 > > {
		/** This is a GraphBLAS semiring. */
		static const constexpr bool value = true;
	};

	template< class _OP1, class _OP2, template< typename > class _ID1, template< typename > class _ID2 >
	struct has_immutable_nonzeroes< Semiring< _OP1, _OP2, _ID1, _ID2 > > {
		static const constexpr bool value = alp::is_semiring< Semiring< _OP1, _OP2, _ID1, _ID2 > >::value &&
			std::is_same< _OP1, typename alp::operators::logical_or< typename _OP1::D1, typename _OP1::D2, typename _OP1::D3 > >::value;
	};

} // namespace alp

#endif
