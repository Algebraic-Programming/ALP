
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
 * Implements the level-3 primitives for the nonblocking backend
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_BLAS3
#define _H_GRB_NONBLOCKING_BLAS3

#include <type_traits> //for std::enable_if

#include <graphblas/base/blas3.hpp>
#include <graphblas/utils/iterators/matrixVectorIterator.hpp>

#include "io.hpp"
#include "matrix.hpp"

#include <omp.h>

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
		"* Possible fix 2 | For all mismatches in the domains of input "       \
		"parameters and the semiring domains, as specified in the "            \
		"documentation of the function " y ", supply a container argument of " \
		"the expected type instead.\n"                                         \
		"* Possible fix 3 | Provide a compatible semiring where all domains "  \
		"match those of the container arguments, as specified in the "         \
		"documentation of the function " y ".\n"                               \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );


namespace grb {

	namespace internal {

		extern LazyEvaluation le;

	}

}

namespace grb {

	namespace internal {

		template<
			bool allow_void,
			Descriptor descr,
			class Monoid,
			class Operator,
			class MulMonoid,
			typename OutputType, typename InputType1, typename InputType2,
			typename RIT1, typename CIT1, typename NIT1,
			typename RIT2, typename CIT2, typename NIT2,
			typename RIT3, typename CIT3, typename NIT3
		>
		RC mxm_generic(
			Matrix< OutputType, nonblocking, RIT1, CIT1, NIT1 > &C,
			const Matrix< InputType1, nonblocking, RIT2, CIT2, NIT2 > &A,
			const Matrix< InputType2, nonblocking, RIT3, CIT3, NIT3 > &B,
			const Operator &oper,
			const Monoid &monoid,
			const MulMonoid &mulMonoid,
			const Phase &phase,
			const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_operator< Operator >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = nullptr
		) {
			// nonblocking execution is not supported
			// first, execute any computation that is not completed
			le.execution();

			// second, delegate to the reference backend
			return mxm_generic< allow_void, descr >(
					getRefMatrix( C ), getRefMatrix( A ), getRefMatrix( B ),
					oper, monoid, mulMonoid, phase
				);
		}

	} // end namespace grb::internal

	template<
		Descriptor descr = descriptors::no_operation,
		class Semiring,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2,
		typename RIT3, typename CIT3, typename NIT3
	>
	RC mxm(
		Matrix< OutputType, nonblocking, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType1, nonblocking, RIT2, CIT2, NIT2 > &A,
		const Matrix< InputType2, nonblocking, RIT3, CIT3, NIT3 > &B,
		const Semiring &ring = Semiring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Semiring >::value,
		void >::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Semiring::D1, InputType1 >::value
			), "grb::mxm",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Semiring::D2, InputType2 >::value ), "grb::mxm",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Semiring::D4, OutputType >::value
			), "grb::mxm",
			"called with an output matrix C that does not match the output domain "
			"of the given operator" );

#ifdef _DEBUG
		std::cout << "In grb::mxm (nonblocking, unmasked, semiring)\n";
#endif

		if( internal::NONBLOCKING::warn_if_not_native &&
			config::PIPELINE::warn_if_not_native
		) {
			std::cerr << "Warning: mxm (nonblocking, unmasked, semiring) currently "
				<< "delegates to a blocking implementation\n"
				<< "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}

		return internal::mxm_generic< true, descr >(
			C, A, B,
			ring.getMultiplicativeOperator(),
			ring.getAdditiveMonoid(),
			ring.getMultiplicativeMonoid(),
			phase
		);
	}

	template<
		Descriptor descr = grb::descriptors::no_operation,
		class Operator,
		class Monoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2,
		typename RIT3, typename CIT3, typename NIT3
	>
	RC mxm(
		Matrix< OutputType, nonblocking, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType1, nonblocking, RIT2, CIT2, NIT2 > &A,
		const Matrix< InputType2, nonblocking, RIT3, CIT3, NIT3 > &B,
		const Monoid &addM,
		const Operator &mulOp,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Operator::D1, InputType1 >::value
			), "grb::mxm",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the given multiplication operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Operator::D2, InputType2 >::value
			), "grb::mxm",
			"called with a postfactor input matrix B that does not match the first "
			"domain of the given multiplication operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Operator::D3, OutputType >::value ),
			"grb::mxm",
			"called with an output matrix C that does not match the output domain "
			"of the given multiplication operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Monoid::D1, typename Operator::D3 >::value
			), "grb::mxm",
			"the output domain of the multiplication operator does not match the "
			"first domain of the given addition monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Monoid::D2, OutputType >::value
			), "grb::mxm",
			"the second domain of the given addition monoid does not match the "
			"type of the output matrix C" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Monoid::D3, OutputType >::value
			), "grb::mxm",
			"the output type of the given addition monoid does not match the type "
			"of the output matrix C" );
		static_assert( ( !(
				std::is_same< InputType1, void >::value ||
				std::is_same< InputType2, void >::value
			) ),
			"grb::mxm: the operator-monoid version of mxm cannot be used if either "
			"of the input matrices is a pattern matrix (of type void)" );

		if( internal::NONBLOCKING::warn_if_not_native &&
			config::PIPELINE::warn_if_not_native
		) {
			std::cerr << "Warning: mxm (nonblocking, unmasked, monoid-op) currently "
				<< "delegates to a blocking implementation\n"
				<< "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}

		return internal::mxm_generic< false, descr >(
			C, A, B, mulOp, addM, Monoid(), phase
		);
	}

	namespace internal {

		template<
			Descriptor descr = descriptors::no_operation,
			bool matrix_is_void,
			typename OutputType,
			typename InputType1, typename InputType2, typename InputType3,
			typename RIT, typename CIT, typename NIT,
			typename Coords
		>
		RC matrix_zip_generic(
			Matrix< OutputType, nonblocking, RIT, CIT, NIT > &A,
			const Vector< InputType1, nonblocking, Coords > &x,
			const Vector< InputType2, nonblocking, Coords > &y,
			const Vector< InputType3, nonblocking, Coords > &z,
			const Phase &phase
		) {
			if( internal::NONBLOCKING::warn_if_not_native &&
				config::PIPELINE::warn_if_not_native
			) {
				std::cerr << "Warning: zip (matrix<-vector<-vector<-vector, nonblocking) "
					<< "currently delegates to a blocking implementation.\n"
					<< "         Further similar such warnings will be suppressed.\n";
				internal::NONBLOCKING::warn_if_not_native = false;
			}

			// nonblocking execution is not supported
			// first, execute any computation that is not completed
			le.execution();

			// second, delegate to the reference backend
			return matrix_zip_generic< descr, matrix_is_void >(
					getRefMatrix( A ), getRefVector( x ), getRefVector( y ), getRefVector( z ),
					phase
				);
		}

	} // namespace internal

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType,
		typename InputType1, typename InputType2, typename InputType3,
		typename RIT, typename CIT, typename NIT,
		typename Coords
	>
	RC zip(
		Matrix< OutputType, nonblocking, RIT, CIT, NIT > &A,
		const Vector< InputType1, nonblocking, Coords > &x,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Vector< InputType3, nonblocking, Coords > &z,
		const Phase &phase = EXECUTE
	) {
		static_assert( !(descr & descriptors::no_casting) ||
				std::is_integral< InputType1 >::value,
			"grb::zip (two vectors to matrix) called "
			"using non-integral left-hand vector elements" );
		static_assert( !(descr & descriptors::no_casting) ||
				std::is_integral< InputType2 >::value,
			"grb::zip (two vectors to matrix) called "
			"using non-integral right-hand vector elements" );
		static_assert( !(descr & descriptors::no_casting) ||
				std::is_same< OutputType, InputType3 >::value,
			"grb::zip (two vectors to matrix) called "
			"with differing vector nonzero and output matrix domains" );

		const size_t n = grb::size( x );
		const size_t nz = grb::nnz( x );
		const RC ret = grb::clear( A );
		if( ret != SUCCESS ) {
			return ret;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( n != grb::size( z ) ) {
			return MISMATCH;
		}
		if( nz != grb::nnz( y ) ) {
			return ILLEGAL;
		}
		if( nz != grb::nnz( z ) ) {
			return ILLEGAL;
		}

		return internal::matrix_zip_generic< descr, false >( A, x, y, z, phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType1, typename InputType2,
		typename RIT, typename CIT, typename NIT,
		typename Coords
	>
	RC zip(
		Matrix< void, nonblocking, RIT, CIT, NIT > &A,
		const Vector< InputType1, nonblocking, Coords > &x,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Phase &phase = EXECUTE
	) {
		static_assert( !(descr & descriptors::no_casting) ||
				std::is_integral< InputType1 >::value,
			"grb::zip (two vectors to void matrix) called using non-integral "
			"left-hand vector elements" );
		static_assert( !(descr & descriptors::no_casting) ||
				std::is_integral< InputType2 >::value,
			"grb::zip (two vectors to void matrix) called using non-integral "
			"right-hand vector elements" );

		const size_t n = grb::size( x );
		const size_t nz = grb::nnz( x );
		const RC ret = grb::clear( A );
		if( ret != SUCCESS ) {
			return ret;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( nz != grb::nnz( y ) ) {
			return ILLEGAL;
		}

		return internal::matrix_zip_generic< descr, true >( A, x, y, x, phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords,
		typename RIT, typename CIT, typename NIT
	>
	RC outer(
		Matrix< OutputType, nonblocking, RIT, CIT, NIT > &A,
		const Vector< InputType1, nonblocking, Coords > &u,
		const Vector< InputType2, nonblocking, Coords > &v,
		const Operator &mul = Operator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_operator< Operator >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< OutputType >::value,
		void >::type * const = nullptr
	) {
		if( internal::NONBLOCKING::warn_if_not_native &&
			config::PIPELINE::warn_if_not_native
		) {
			std::cerr << "Warning: outer (nonblocking) currently delegates to a "
				<< "blocking implementation.\n"
				<< "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}

		// nonblocking execution is not supported
		// first, execute any computation that is not completed
		internal::le.execution();

		// second, delegate to the reference backend
		return outer< descr, Operator >(
			internal::getRefMatrix( A ),
			internal::getRefVector( u ), internal::getRefVector( v ),
			mul, phase
		);
	}

	namespace internal {

		template<
			bool allow_void,
			Descriptor descr,
			class MulMonoid, class Operator,
			typename OutputType, typename InputType1, typename InputType2,
			typename RIT1, typename CIT1, typename NIT1,
			typename RIT2, typename CIT2, typename NIT2,
			typename RIT3, typename CIT3, typename NIT3
		>
		RC eWiseApply_matrix_generic(
			Matrix< OutputType, nonblocking, RIT1, CIT1, NIT1 > &C,
			const Matrix< InputType1, nonblocking, RIT2, CIT2, NIT2 > &A,
			const Matrix< InputType2, nonblocking, RIT3, CIT3, NIT3 > &B,
			const Operator &oper,
			const MulMonoid &mulMonoid,
			const Phase &phase,
			const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_operator< Operator >::value,
			void >::type * const = nullptr
		) {
			if( internal::NONBLOCKING::warn_if_not_native &&
				config::PIPELINE::warn_if_not_native
			) {
				std::cerr << "Warning: eWiseApply (nonblocking) currently delegates to a "
					<< "blocking implementation.\n"
					<< "         Further similar such warnings will be suppressed.\n";
				internal::NONBLOCKING::warn_if_not_native = false;
			}

			// nonblocking execution is not supported
			// first, execute any computation that is not completed
			le.execution();

			// second, delegate to the reference backend
			return eWiseApply_matrix_generic<
					allow_void, descr,
					MulMonoid, Operator
				>(
					getRefMatrix( C ), getRefMatrix( A ), getRefMatrix( B ),
					oper, mulMonoid, phase
				);
		}

	} // namespace internal

	template<
		Descriptor descr = descriptors::no_operation,
		class MulMonoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2,
		typename RIT3, typename CIT3, typename NIT3
	>
	RC eWiseApply(
		Matrix< OutputType, nonblocking, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType1, nonblocking, RIT2, CIT2, NIT2 > &A,
		const Matrix< InputType2, nonblocking, RIT3, CIT3, NIT3 > &B,
		const MulMonoid &mulmono,
		const Phase phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< MulMonoid >::value,
		void >::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D1, InputType1 >::value ),
			"grb::eWiseApply (nonblocking, matrix <- matrix x matrix, monoid)",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D2, InputType2 >::value ),
			"grb::eWiseApply (nonblocking, matrix <- matrix x matrix, monoid)",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D3, OutputType >::value ),
			"grb::eWiseApply (nonblocking, matrix <- matrix x matrix, monoid)",
			"called with an output matrix C that does not match the output domain "
			"of the monoid operator"
		);

#ifdef _DEBUG
		std::cout << "In grb::eWiseApply_matrix_generic (nonblocking, monoid)\n";
#endif

		return internal::eWiseApply_matrix_generic< true, descr >(
			C, A, B, mulmono.getOperator(), mulmono, phase
		);
	}

	template<
		Descriptor descr = grb::descriptors::no_operation,
		class Operator,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2,
		typename RIT3, typename CIT3, typename NIT3
	>
	RC eWiseApply(
		Matrix< OutputType, nonblocking, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType1, nonblocking, RIT2, CIT2, NIT2 > &A,
		const Matrix< InputType2, nonblocking, RIT3, CIT3, NIT3 > &B,
		const Operator &mulOp,
		const Phase phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value,
		void >::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename Operator::D1, InputType1 >::value ),
			"grb::eWiseApply (nonblocking, matrix <- matrix x matrix, operator)",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the given multiplication operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename Operator::D2, InputType2 >::value ),
			"grb::eWiseApply (nonblocking, matrix <- matrix x matrix, operator)",
			"called with a postfactor input matrix B that does not match the first "
			"domain of the given multiplication operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename Operator::D3, OutputType >::value ),
			"grb::eWiseApply (nonblocking, matrix <- matrix x matrix, operator)",
			"called with an output matrix C that does not match the output domain "
			"of the given multiplication operator"
		);
		static_assert( ( !(
				std::is_same< InputType1, void >::value ||
				std::is_same< InputType2, void >::value )
			), "grb::eWiseApply (nonblocking, matrix <- matrix x matrix, operator): "
			"the operator version of eWiseApply cannot be used if either of the "
			"input matrices is a pattern matrix (of type void)"
		);

		typename grb::Monoid<
			grb::operators::mul< double >,
			grb::identities::one
		> dummyMonoid;
		return internal::eWiseApply_matrix_generic< false, descr >(
			C, A, B, mulOp, dummyMonoid, phase
		);
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename Tin,
		typename RITin, typename CITin, typename NITin,
		typename Tout,
		typename RITout, typename CITout, typename NITout
	>
	RC select(
		Matrix< Tout, nonblocking, RITout, CITout, NITout > &out,
		const Matrix< Tin, nonblocking, RITin, CITin, NITin > &in,
		const Operator &op = Operator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!is_object< Tin >::value &&
			!is_object< Tout >::value
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In grb::select( nonblocking )\n";
#endif
		if( internal::NONBLOCKING::warn_if_not_native &&
			config::PIPELINE::warn_if_not_native
		) {
			std::cerr << "Warning: select (nonblocking) currently delegates to a "
				<< "blocking implementation.\n"
				<< "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}

		// nonblocking execution is not supported
		// first, execute any computation that is not completed
		internal::le.execution();

		// second, delegate to the reference backend
		return select< descr, Operator >(
			internal::getRefMatrix( out ),
			internal::getRefMatrix( in ),
			op,
			phase
		);
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType, typename MaskType,
		typename RIT_A, typename CIT_A, typename NIT_A,
		typename RIT_M, typename CIT_M, typename NIT_M
	>
	RC foldr(
		const Matrix< InputType, nonblocking, RIT_A, CIT_A, NIT_A > &A,
		const Matrix< MaskType, nonblocking, RIT_M, CIT_M, NIT_M > &mask,
		IOType &x,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( !std::is_void< InputType >::value,
			"grb::foldr( nonblocking, IOType <- [[IOType]], monoid, masked ): "
			"the provided matrix may not be a pattern matrix."
			"Possible fix: provide a semiring instead of a ring"
		);
		static_assert( grb::is_commutative< Monoid >::value,
			"grb::foldr( nonblocking, IOType <- [[IOType]], monoid, masked ): "
			"the provided monoid must be commutative (but is not)"
		);
		static_assert( (std::is_same< typename Monoid::D1, InputType >::value),
			"grb::foldr( nonblocking, IOType <- [[IOType]], monoid, masked ): "
			"called with a prefactor input type that does not match the first domain of "
			"the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D2, IOType >::value),
			"grb::foldr( nonblocking, IOType <- [[IOType]], monoid, masked ): "
			"called with a postfactor input type that does not match the second domain "
			"of the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldr( nonblocking, IOType <- [[IOType]], monoid, masked ): "
			"called with an output type that does not match the output domain of the "
			"given monoid"
		);
		static_assert( !(descr & descriptors::add_identity),
			"grb::foldr( nonblocking, IOType <- [[IOType]], monoid, masked ): "
			"the use of the add_identity descriptor requires a semiring, but a monoid "
			"was given"
		);
		static_assert( !(
				(descr & descriptors::invert_mask) &&
				(descr & descriptors::structural)
			), "grb::foldr( nonblocking, IOType <- [[IOType]], monoid, masked ): "
			"may not select an inverted structural mask for matrices"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldr( nonblocking, matrix, mask, monoid )\n";
#endif

		if( internal::NONBLOCKING::warn_if_not_native &&
			config::PIPELINE::warn_if_not_native
		) {
			std::cerr << "Warning: grb::foldr( nonblocking, matrix, mask, monoid )"
				<< " currently delegates to a blocking implementation\n"
				<< "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}

		// nonblocking execution is not supported
		// first, execute any computation that is not completed
		internal::le.execution();

		// second, delegate to the reference backend
		return foldr< descr, Monoid >( internal::getRefMatrix( A ),
			internal::getRefMatrix( mask ), x, monoid );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType,
		typename RIT, typename CIT, typename NIT
	>
	RC foldr(
		const Matrix< InputType, nonblocking, RIT, CIT, NIT > &A,
		IOType &x,
		const Monoid &monoid,
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( !std::is_void< InputType >::value,
			"grb::foldr( nonblocking, IOType <- [[IOType]], monoid ): "
			"the provided matrix may not be a pattern matrix."
			"Possible fix: provide a semiring instead of a ring"
		);
		static_assert( grb::is_commutative< Monoid >::value,
			"grb::foldr( nonblocking, IOType <- [[IOType]], monoid ): "
			"the provided monoid must be commutative (but is not)"
		);
		static_assert( (std::is_same< typename Monoid::D1, InputType >::value),
			"grb::foldr( nonblocking, IOType <- [[IOType]], monoid ): "
			"called with a prefactor input type that does not match the first domain of "
			"the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D2, IOType >::value),
			"grb::foldr( nonblocking, IOType <- [[IOType]], monoid ): "
			"called with a postfactor input type that does not match the second domain "
			"of the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldr( nonblocking, IOType <- [[IOType]], monoid ): "
			"called with an output type that does not match the output domain of the "
			"given monoid"
		);
		static_assert( !(descr & descriptors::add_identity),
			"grb::foldr( nonblocking, IOType <- [[IOType]], monoid ): "
			"the use of the add_identity descriptor requires a semiring, but a monoid "
			"was given"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldr( nonblocking, matrix, monoid )\n";
#endif

		if( internal::NONBLOCKING::warn_if_not_native &&
			config::PIPELINE::warn_if_not_native
		) {
			std::cerr << "Warning: grb::foldr( nonblocking, matrix, monoid )"
				<< " currently delegates to a blocking implementation\n"
				<< "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}

		// nonblocking execution is not supported
		// first, execute any computation that is not completed
		internal::le.execution();

		// second, delegate to the reference backend
		return foldr< descr, Monoid >( internal::getRefMatrix( A ), x, monoid );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Semiring,
		typename InputType, typename IOType,
		typename RIT, typename CIT, typename NIT
	>
	RC foldr(
		const Matrix< InputType, nonblocking, RIT, CIT, NIT > &A,
		IOType &x,
		const Semiring &semiring = Semiring(),
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_semiring< Semiring >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( (std::is_same< typename Semiring::D3, InputType >::value),
			"grb::foldr( nonblocking, IOType <- [[IOType]], semiring ): "
			"called with a prefactor input type that does not match the third domain of "
			"the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, IOType >::value),
			"grb::foldr( nonblocking, IOType <- [[IOType]], semiring ): "
			"called with a postfactor input type that does not match the fourth domain "
			"of the given semiring"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldr( nonblocking, matrix, semiring )\n";
#endif

		if( internal::NONBLOCKING::warn_if_not_native &&
			config::PIPELINE::warn_if_not_native
		) {
			std::cerr << "Warning: grb::foldr( nonblocking, matrix, semiring )"
				<< " currently delegates to a blocking implementation\n"
				<< "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}

		// nonblocking execution is not supported
		// first, execute any computation that is not completed
		internal::le.execution();

		// second, delegate to the reference backend
		return foldr< descr, Semiring >( internal::getRefMatrix( A ), x, semiring );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Semiring,
		typename InputType, typename IOType, typename MaskType,
		typename RIT_A, typename CIT_A, typename NIT_A,
		typename RIT_M, typename CIT_M, typename NIT_M
	>
	RC foldr(
		const Matrix< InputType, nonblocking, RIT_A, CIT_A, NIT_A > &A,
		const Matrix< MaskType, nonblocking, RIT_M, CIT_M, NIT_M > &mask,
		IOType &x,
		const Semiring &semiring = Semiring(),
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Semiring >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( (std::is_same< typename Semiring::D3, InputType >::value),
			"grb::foldr( nonblocking, IOType <- [[IOType]], semiring, masked ): "
			"called with a prefactor input type that does not match the third domain "
			"of the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, IOType >::value),
			"grb::foldr( nonblocking, IOType <- [[IOType]], semiring, masked ): "
			"called with a postfactor input type that does not match the fourth domain "
			"of the given semiring"
		);
		static_assert( !(
				(descr & descriptors::invert_mask) &&
				(descr & descriptors::structural)
			), "grb::foldr( nonblocking, IOType <- [[IOType]], semiring, masked ): "
			"may not select an inverted structural mask for matrices"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldr( nonblocking, matrix, mask, semiring )\n";
#endif

		if( internal::NONBLOCKING::warn_if_not_native &&
			config::PIPELINE::warn_if_not_native
		) {
			std::cerr << "Warning: grb::foldr( nonblocking, matrix, mask, semiring )"
				<< " currently delegates to a blocking implementation\n"
				<< "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}

		// nonblocking execution is not supported
		// first, execute any computation that is not completed
		internal::le.execution();

		// second, delegate to the reference backend
		return foldr< descr, Semiring >( internal::getRefMatrix( A ),
			internal::getRefMatrix( mask ), x, semiring );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType, typename MaskType,
		typename RIT_A, typename CIT_A, typename NIT_A,
		typename RIT_M, typename CIT_M, typename NIT_M
	>
	RC foldl(
		IOType &x,
		const Matrix< InputType, nonblocking, RIT_A, CIT_A, NIT_A > &A,
		const Matrix< MaskType, nonblocking, RIT_M, CIT_M, NIT_M > &mask,
		const Monoid &monoid,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( !std::is_void< InputType >::value,
			"grb::foldl( nonblocking, IOType <- [[IOType]], monoid ): "
			"the provided matrix may not be a pattern matrix."
			"Possible fix: provide a semiring instead of a ring"
		);
		static_assert( grb::is_commutative< Monoid >::value,
			"grb::foldl( nonblocking, IOType <- [[IOType]], monoid ): "
			"the provided monoid must be commutative (but is not)"
		);
		static_assert( (std::is_same< typename Monoid::D1, IOType >::value),
			"grb::foldl( nonblocking, IOType <- [[IOType]], monoid, masked ): "
			"called with a prefactor input type that does not match the first domain of "
			"the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D2, InputType >::value),
			"grb::foldl( nonblocking, IOType <- [[IOType]], monoid, masked ): "
			"called with a postfactor input type that does not match the second domain "
			"of the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldl( nonblocking, IOType <- [[IOType]], monoid, masked ): "
			"called with an output type that does not match the output domain of the "
			"given monoid"
		);
		static_assert( !(descr & descriptors::add_identity),
			"grb::foldl( nonblocking, IOType <- [[IOType]], monoid, masked ): "
			"the use of the add_identity descriptor requires a semiring, but a monoid "
			"was given"
		);
		static_assert( !(
				(descr & descriptors::invert_mask) &&
				(descr & descriptors::structural)
			), "grb::foldl( nonblocking, IOType <- [[IOType]], monoid, masked ): "
			"may not select an inverted structural mask for matrices"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldl( nonblocking, matrix, mask, monoid )\n";
#endif

		if( internal::NONBLOCKING::warn_if_not_native &&
			config::PIPELINE::warn_if_not_native
		) {
			std::cerr << "Warning: grb::foldl( nonblocking, matrix, mask, monoid )"
				<< " currently delegates to a blocking implementation\n"
				<< "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}

		// nonblocking execution is not supported
		// first, execute any computation that is not completed
		internal::le.execution();

		// second, delegate to the reference backend
		return foldl< descr, Monoid >( x, internal::getRefMatrix( A ),
			internal::getRefMatrix( mask ), monoid );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType,
		typename RIT, typename CIT, typename NIT
	>
	RC foldl(
		IOType &x,
		const Matrix< InputType, nonblocking, RIT, CIT, NIT > &A,
		const Monoid &monoid,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( !std::is_void< InputType >::value,
			"grb::foldl( nonblocking, IOType <- [[IOType]], monoid ): "
			"the provided matrix may not be a pattern matrix."
			"Possible fix: provide a semiring instead of a ring"
		);
		static_assert( grb::is_commutative< Monoid >::value,
			"grb::foldl( nonblocking, IOType <- [[IOType]], monoid ): "
			"the provided monoid must be commutative (but is not)"
		);
		static_assert( (std::is_same< typename Monoid::D1, IOType >::value),
			"grb::foldl( nonblocking, IOType <- [[IOType]], monoid ): "
			"called with a prefactor input type that does not match the first domain of "
			"the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D2, InputType >::value),
			"grb::foldl( nonblocking, IOType <- [[IOType]], monoid ): "
			"called with a postfactor input type that does not match the second domain "
			"of the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldl( nonblocking, IOType <- [[IOType]], monoid ): "
			"called with an output type that does not match the output domain of the "
			"given monoid"
		);
		static_assert( !(descr & descriptors::add_identity),
			"grb::foldl( nonblocking, IOType <- [[IOType]], monoid ): "
			"the use of the add_identity descriptor requires a semiring, but a monoid "
			"was given"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldl( nonblocking, matrix, monoid )\n";
#endif

		if( internal::NONBLOCKING::warn_if_not_native &&
			config::PIPELINE::warn_if_not_native
		) {
			std::cerr << "Warning: grb::foldl( nonblocking, matrix, monoid )"
				<< " currently delegates to a blocking implementation\n"
				<< "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}
		// nonblocking execution is not supported
		// first, execute any computation that is not completed
		internal::le.execution();

		// second, delegate to the reference backend
		return foldl< descr, Monoid >( x, internal::getRefMatrix( A ), monoid );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Semiring,
		typename InputType, typename IOType, typename MaskType,
		typename RIT_A, typename CIT_A, typename NIT_A,
		typename RIT_M, typename CIT_M, typename NIT_M
	>
	RC foldl(
		IOType &x,
		const Matrix< InputType, nonblocking, RIT_A, CIT_A, NIT_A > &A,
		const Matrix< MaskType, nonblocking, RIT_M, CIT_M, NIT_M > &mask,
		const Semiring &semiring = Semiring(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Semiring >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( (std::is_same< typename Semiring::D3, IOType >::value),
			"grb::foldl( nonblocking, IOType <- [[IOType]], semiring, masked ): "
			"called with a prefactor input type that does not match the third domain "
			"of the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, InputType >::value),
			"grb::foldl( nonblocking, IOType <- [[IOType]], semiring, masked ): "
			"called with a postfactor input type that does not match the fourth domain "
			"of the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, IOType >::value),
			"grb::foldl( nonblocking, IOType <- [[IOType]], semiring, masked ): "
			"called with an output type that does not match the fourth domain of the "
			"given semiring"
		);
		static_assert( !(
				(descr & descriptors::invert_mask) &&
				(descr & descriptors::structural)
			), "grb::foldl( nonblocking, IOType <- [[IOType]], semiring, masked ): "
			"may not select an inverted structural mask for matrices"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldl( nonblocking, matrix, mask, semiring )\n";
#endif

		if( internal::NONBLOCKING::warn_if_not_native &&
			config::PIPELINE::warn_if_not_native
		) {
			std::cerr << "Warning: grb::foldl( nonblocking, matrix, mask, semiring )"
				<< " currently delegates to a blocking implementation\n"
				<< "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}

		// nonblocking execution is not supported
		// first, execute any computation that is not completed
		internal::le.execution();

		// second, delegate to the reference backend
		return foldl< descr, Semiring >( x, internal::getRefMatrix( A ),
			internal::getRefMatrix( mask ), semiring );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Semiring,
		typename InputType, typename IOType,
		typename RIT, typename CIT, typename NIT
	>
	RC foldl(
		IOType &x,
		const Matrix< InputType, nonblocking, RIT, CIT, NIT > &A,
		const Semiring &semiring = Semiring(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_semiring< Semiring >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( (std::is_same< typename Semiring::D3, IOType >::value),
			"grb::foldl( nonblocking, IOType <- [[IOType]], semiring ): "
			"called with a prefactor input type that does not match the third domain "
			"of the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, InputType >::value),
			"grb::foldl( nonblocking, IOType <- [[IOType]], semiring ): "
			"called with a postfactor input type that does not match the fourth domain "
			"of the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, IOType >::value),
			"grb::foldl( nonblocking, IOType <- [[IOType]], semiring ): "
			"called with an output type that does not match the fourth domain of the "
			"given semiring"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldl( nonblocking, matrix, semiring )\n";
#endif

		if( internal::NONBLOCKING::warn_if_not_native &&
			config::PIPELINE::warn_if_not_native
		) {
			std::cerr << "Warning: grb::foldl( nonblocking, matrix, semiring )"
				<< " currently delegates to a blocking implementation\n"
				<< "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}
		// nonblocking execution is not supported
		// first, execute any computation that is not completed
		internal::le.execution();

		// second, delegate to the reference backend
		return foldl< descr, Semiring >( x, internal::getRefMatrix( A ), semiring );
	}

} // namespace grb

#undef NO_CAST_ASSERT

#endif // ``_H_GRB_NONBLOCKING_BLAS3''

