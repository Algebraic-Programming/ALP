
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
 * @date 14th of January 2022
 */

#ifndef _H_ALP_DISPATCH_BLAS1
#define _H_ALP_DISPATCH_BLAS1

#include <functional>
#include <alp/backends.hpp>
#include <alp/config.hpp>
#include <alp/rc.hpp>
#include <alp/density.hpp>

#include "scalar.hpp"
#include "matrix.hpp"
#include "vector.hpp"
#include "blas0.hpp"
#include "blas2.hpp"

#include <alp/utils/iscomplex.hpp>

#define NO_CAST_ASSERT( x, y, z )                                              \
	static_assert( x,                                                          \
		"\n\n"                                                                 \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"*     ERROR      | " y " " z ".\n"                                    \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"* Possible fix 1 | Remove no_casting from the template parameters "   \
		"in this call to " y ".\n"                                             \
		"* Possible fix 2 | Provide a value that matches the expected type.\n" \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );

#define NO_CAST_OP_ASSERT( x, y, z )                                           \
	static_assert( x,                                                          \
		"\n\n"                                                                 \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"*     ERROR      | " y " " z ".\n"                                    \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"* Possible fix 1 | Remove no_casting from the template parameters "   \
		"in this call to " y ".\n"                                             \
		"* Possible fix 2 | For all mismatches in the domains of input "       \
		"parameters and the operator domains, as specified in the "            \
		"documentation of the function " y ", supply an input argument of "    \
		"the expected type instead.\n"                                         \
		"* Possible fix 3 | Provide a compatible operator where all domains "  \
		"match those of the input parameters, as specified in the "            \
		"documentation of the function " y ".\n"                               \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );


namespace alp {

	/**
	 * Calculates the dot product, \f$ \alpha = (x,y) \f$, under a given additive
	 * monoid and multiplicative operator.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class AddMonoid, class AnyOp
	>
	RC dot(
		Scalar< OutputType, OutputStructure, dispatch > &z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, dispatch > &x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, dispatch > &y,
		const AddMonoid &addMonoid = AddMonoid(),
		const AnyOp &anyOp = AnyOp(),
		const std::enable_if_t< !alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_monoid< AddMonoid >::value &&
			alp::is_operator< AnyOp >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< InputType1, typename AnyOp::D1 >::value ), "alp::dot",
			"called with a left-hand vector value type that does not match the first "
			"domain of the given multiplicative operator" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< InputType2, typename AnyOp::D2 >::value ), "alp::dot",
			"called with a right-hand vector value type that does not match the second "
			"domain of the given multiplicative operator" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< typename AddMonoid::D3, typename AnyOp::D1 >::value ), "alp::dot",
			"called with a multiplicative operator output domain that does not match "
			"the first domain of the given additive operator" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< OutputType, typename AddMonoid::D2 >::value ), "alp::dot",
			"called with an output vector value type that does not match the second "
			"domain of the given additive operator" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< typename AddMonoid::D3, typename AddMonoid::D2 >::value ), "alp::dot",
			"called with an additive operator whose output domain does not match its "
			"second input domain" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< OutputType, typename AddMonoid::D3 >::value ), "alp::dot",
			"called with an output vector value type that does not match the third "
			"domain of the given additive operator" );
		(void)z;
		if( size( x ) != size( y ) ) {
			return MISMATCH;
		}

		if( !( internal::getInitialized( z ) && internal::getInitialized( x ) && internal::getInitialized( y ) ) ) {
#ifdef _DEBUG
			std::cout << "dot(): one of input vectors or scalar are not initialized: do noting!\n";
#endif
			return SUCCESS;
		}

		std::function< void( typename AddMonoid::D3 &, const size_t, const size_t ) > data_lambda =
			[ &x, &y, &anyOp ]( typename AddMonoid::D3 &result, const size_t i, const size_t j ) {
				(void) j;
				internal::apply(
					result, x[ i ],
					alp::utils::is_complex< InputType2 >::conjugate( y[ i ] ),
					anyOp
				);
			};

		std::function< bool() > init_lambda =
			[ &x ]() -> bool {
				return internal::getInitialized( x );
			};

		Vector<
			typename AddMonoid::D3,
			structures::General,
			Density::Dense,
			view::Functor< std::function< void( typename AddMonoid::D3 &, const size_t, const size_t ) > >,
			imf::Id, imf::Id,
			dispatch
		> temp(
			init_lambda,
			getLength( x ),
			data_lambda
		);
		RC rc = foldl( z, temp, addMonoid );
		return rc;
	}

	/**
	 * Provides a generic implementation of the dot computation on semirings by
	 * translating it into a dot computation on an additive commutative monoid
	 * with any multiplicative operator.
	 *
	 * For return codes, exception behaviour, performance semantics, template
	 * and non-template arguments, @see alp::dot.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Ring
	>
	RC dot( Scalar< IOType, IOStructure, dispatch > &x,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, dispatch > &left,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, dispatch > &right,
		const Ring &ring = Ring(),
		const typename std::enable_if<
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			!alp::is_object< IOType >::value &&
			alp::is_semiring< Ring >::value,
		void >::type * const = NULL
	) {
		return alp::dot< descr >( x,
			left, right,
			ring.getAdditiveMonoid(),
			ring.getMultiplicativeOperator()
		);
	}

	/**
	 * This is the eWiseLambda that performs length checking by recursion.
	 *
	 * in the reference implementation all vectors are distributed equally, so no
	 * need to synchronise any data structures. We do need to do error checking
	 * though, to see when to return alp::MISMATCH. That's this function.
	 *
	 * @see Vector::operator[]()
	 * @see Vector::lambda_reference
	 */
	template<
		typename Func,
		typename DataType1, typename DataStructure1, typename DataView1, typename InputImfR1, typename InputImfC1,
		typename DataType2, typename DataStructure2, typename DataView2, typename InputImfR2, typename InputImfC2,
		typename... Args
	>
	RC eWiseLambda(
		const Func f,
		Vector< DataType1, DataStructure1, Density::Dense, DataView1, InputImfR1, InputImfC1, dispatch > &x,
		const Vector< DataType2, DataStructure2, Density::Dense, DataView2, InputImfR2, InputImfC2, dispatch > &y,
		Args const &... args
	) {
		// catch mismatch
		if( getLength( x ) != getLength( y ) ) {
			return MISMATCH;
		}
		// continue
		return eWiseLambda( f, x, args... );
	}

	/**
	 * No implementation notes. This is the `real' implementation on dispatch
	 * vectors.
	 *
	 * @see Vector::operator[]()
	 * @see Vector::lambda_reference
	 */
	template<
		typename Func,
		typename DataType, typename DataStructure, typename DataView, typename DataImfR, typename DataImfC
	>
	RC eWiseLambda( const Func f, Vector< DataType, DataStructure, Density::Dense, DataView, DataImfR, DataImfC, dispatch > &x ) {
#ifdef _DEBUG
		std::cout << "Info: entering eWiseLambda function on vectors.\n";
#endif
		auto x_as_matrix = get_view< view::matrix >( x );
		return eWiseLambda(
			[ &f ]( const size_t i, const size_t j, DataType &val ) {
				(void)j;
				f( i, val );
			},
			x_as_matrix
		);
	}

	/**
	 * Reduces a vector into a scalar. Reduction takes place according a monoid
	 * \f$ (\oplus,1) \f$, where \f$ \oplus:\ D_1 \times D_2 \to D_3 \f$ with an
	 * associated identity \f$ 1 \in \{D_1,D_2,D_3\} \f$. Elements from the given
	 * vector \f$ y \in \{D_1,D_2\} \f$ will be applied at the left-hand or right-
	 * hand side of \f$ \oplus \f$; which, exactly, is implementation-dependent
	 * but should not matter since \f$ \oplus \f$ should be associative.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		class Monoid
	>
	RC foldl(
		Scalar< IOType, IOStructure, dispatch > &alpha,
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, dispatch > &y,
		const Monoid &monoid = Monoid(),
		const std::enable_if_t<
			! alp::is_object< IOType >::value && ! alp::is_object< InputType >::value && alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {

		// static sanity checks
		NO_CAST_ASSERT(
			( ! ( descr & descriptors::no_casting ) || std::is_same< IOType, InputType >::value ),
			"alp::reduce",
			"called with a scalar IO type that does not match the input vector type"
		);
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, typename Monoid::D1 >::value ), "alp::reduce",
			"called with an input vector value type that does not match the first "
			"domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, typename Monoid::D2 >::value ), "alp::reduce",
			"called with an input vector type that does not match the second domain of "
			"the given monoid" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, typename Monoid::D3 >::value ), "alp::reduce",
			"called with an input vector type that does not match the third domain of "
			"the given monoid" );

#ifdef _DEBUG
		std::cout << "foldl(Scalar,Vector,Monoid) called. Vector has size " << getLength( y ) << " .\n";
#endif

		internal::setInitialized(
			alpha,
			internal::getInitialized( alpha ) && internal::getInitialized( y )
		);

		if( !internal::getInitialized( alpha ) ) {
			return SUCCESS;
		}

		const size_t n = getLength( y );
		for ( size_t i = 0; i < n; ++i ) {
			(void) internal::foldl( *alpha, y[ i ], monoid.getOperator() );
		}
		return SUCCESS;
	}
	/**
	 * For all elements in a ALP Vector \a x, fold the value \f$ \beta \f$
	 * into each element.
	 *
	 * The original value of \f$ \beta \f$ is used as the right-hand side input
	 * of the operator \a op. The left-hand side inputs for \a op are retrieved
	 * from the input vector \a x. The result of the operation is stored in
	 * \f$ \beta \f$, thus overwriting its previous value. This process is
	 * repeated for every element in \a y.
	 *
	 * The value of \f$ x_i \f$ after a call to thus function thus equals
	 * \f$ x_i \odot \beta \f$, for all \f$ i \in \{ 0, 1, \dots, n - 1 \} \f$.
	 *
	 * @tparam descr       The descriptor used for evaluating this function. By
	 *                     default, this is alp::descriptors::no_operation.
	 * @tparam OP          The type of the operator to be applied.
	 * @tparam IOType      The type of the value \a beta.
	 * @tparam InputType   The type of the elements of \a x.
	 * @tparam IOStructure The structure of the vector \a x.
	 * @tparam IOView      The view type applied to the vector \a x.
	 *
	 * @param[in,out] x    On function entry: the initial values to be applied as
	 *                     the left-hand side input to \a op. The input vector must
	 *                     be dense.
	 *                     On function exit: the output data.
	 * @param[in]     beta The input value to apply as the right-hand side input
	 *                     to \a op.
	 * @param[in]     op   The operator under which to perform this left-folding.
	 *
	 * @returns alp::SUCCESS This function always succeeds.
	 *
	 * \note This function is also defined for monoids.
	 *
	 * \warning If \a x is sparse and this operation is requested, a monoid instead
	 *          of an operator is required!
	 *
	 * \parblock
	 * \par Valid descriptors
	 * alp::descriptors::no_operation, alp::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If alp::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a IOType, 2) the second domain of \a op must match
	 * \a InputType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# (no requirement).
	 * \endparblock
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	//  *         the size of the vector \a x. The constant factor depends on the
	//  *         cost of evaluating the underlying binary operator. A good
	//  *         implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the operator used allow for this.
	//  *
	//  *      -# This call will not result in additional dynamic memory allocations.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ 2n \cdot \mathit{sizeof}(\mathit{IOType}) + \mathcal{O}(1) \f$
	//  *         bytes of data movement.
	//  * \endparblock
	 *
	 * @see alp::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType, typename InputStructure,
		class Op
	>
	RC foldl(
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, dispatch > &x,
		const Scalar< InputType, InputStructure, dispatch > beta,
		const Op &op = Op(),
		const std::enable_if_t<
			! alp::is_object< IOType >::value && ! alp::is_object< InputType >::value && alp::is_operator< Op >::value
		> * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT(
			( ! ( descr & descriptors::no_casting )	|| std::is_same< typename Op::D1, IOType >::value ),
			"alp::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( ! ( descr & descriptors::no_casting )	|| std::is_same< typename Op::D2, InputType >::value ),
			"alp::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( ! ( descr & descriptors::no_casting )	|| std::is_same< typename Op::D3, IOType >::value ),
			"alp::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator"
		);

#ifdef _DEBUG
		std::cout << "foldl(Vector,Scalar,Op) called. Vector has size " << getLength( x ) << " .\n";
#endif

		internal::setInitialized(
			x,
			internal::getInitialized( x ) && internal::getInitialized( beta )
		);

		if( !internal::getInitialized( x ) ) {
			return SUCCESS;
		}

		const size_t n = getLength( x );
		for ( size_t i = 0; i < n; ++i ) {
			(void) internal::foldl( x[ i ], *beta, op );
		}
		return SUCCESS;
	}

	/**
	 * Returns a view over the input vector returning conjugate of the accessed element.
	 * This avoids materializing the resulting container.
	 * The elements are calculated lazily on access.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename Structure, typename View, typename ImfR, typename ImfC
	>
	Vector<
		DataType, Structure, Density::Dense,
		view::Functor< std::function< void( DataType &, const size_t, const size_t ) > >,
		imf::Id, imf::Id,
		dispatch
	>
	conjugate(
		const Vector< DataType, Structure, Density::Dense, View, ImfR, ImfC, dispatch > &x,
		const std::enable_if_t<
			!alp::is_object< DataType >::value
		> * const = nullptr
	) {

		std::function< void( DataType &, const size_t, const size_t ) > data_lambda =
			[ &x ]( DataType &result, const size_t i, const size_t j ) {
				(void) j;
				result = alp::utils::is_complex< DataType >::conjugate( x[ i ] );
			};

		std::function< bool() > init_lambda =
			[ &x ]() -> bool {
				return internal::getInitialized( x );
			};

		return Vector<
			DataType,
			Structure,
			Density::Dense,
			view::Functor< std::function< void( DataType &, const size_t, const size_t ) > >,
			imf::Id, imf::Id,
			dispatch
		>( init_lambda,	getLength( x ),	data_lambda );

	}

} // end namespace ``alp''

#undef NO_CAST_ASSERT

#endif // end ``_H_ALP_DISPATCH_BLAS1''

