
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

#ifndef _H_GRB_BSP1D_BLAS3
#define _H_GRB_BSP1D_BLAS3

#include <graphblas/backends.hpp>
#include <graphblas/base/blas3.hpp>

#include "matrix.hpp"


namespace grb {

	namespace internal {

		/**
		 * \internal
		 * Given an output container \a A and a local error code \a local_rc,
		 * will check the global error state. If there was any process with an error,
		 * one of the raised errors will be returned while making sure that \a A is
		 * cleared.
		 * \endinternal
		 */
		template<
			typename DataType, Backend backend,
			typename RIT, typename CIT, typename NIT
		>
		RC checkGlobalErrorStateOrClear(
			Matrix< DataType, backend, RIT, CIT, NIT > &A,
			const RC local_rc
		) noexcept {
			RC global_rc = local_rc;
			if( collectives<>::allreduce( global_rc,
				operators::any_or< RC >() ) != SUCCESS
			) {
				return PANIC;
			}
			if( global_rc != SUCCESS && local_rc == SUCCESS ) {
				// a remote user process failed while we did not --
				//   we need to clear the output and return
				if( clear( internal::getLocal( A ) ) != SUCCESS ) {
					return PANIC;
				}
			}
			return global_rc;
		}

	} // end namespace grb::internal

	// we keep the definition of set here, rather than in bsp1d/io.hpp, because
	// of the use of the above internal convenience function

	/** \internal No implementation details; simply delegates */
	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType1, typename DataType2,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2
	>
	RC set(
		Matrix< DataType1, BSP1D, RIT1, CIT1, NIT1 > &out,
		const Matrix< DataType2, BSP1D, RIT2, CIT2, NIT2 > &in,
		const Phase &phase = EXECUTE
	) noexcept {
		assert( phase != TRY );
		RC local_rc = SUCCESS;
		if( phase == RESIZE ) {
			return resize( out, nnz( in ) );
		} else {
			local_rc = grb::set< descr >( internal::getLocal( out ),
				internal::getLocal( in ) );
		}
		return internal::checkGlobalErrorStateOrClear( out, local_rc );
	}

	/** \internal Simply delegates to process-local backend. */
	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType1, typename DataType2, typename DataType3,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2
	>
	RC set(
		Matrix< DataType1, BSP1D, RIT1, CIT1, NIT1 > &out,
		const Matrix< DataType2, BSP1D, RIT2, CIT2, NIT2 > &mask,
		const DataType3 &val,
		const Phase &phase = EXECUTE
	) noexcept {
		assert( phase != TRY );
		RC local_rc = SUCCESS;
		if( phase == RESIZE ) {
			return resize( out, nnz( mask ) );
		} else {
			local_rc = grb::set< descr >(
				internal::getLocal( out ), internal::getLocal( mask ), val
			);
		}
		return internal::checkGlobalErrorStateOrClear( out, local_rc );
	}

	/** \internal Simply delegates to process-local backend */
	template<
		Descriptor descr = descriptors::no_operation,
		class MulMonoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2,
		typename RIT3, typename CIT3, typename NIT3
	>
	RC eWiseApply(
		Matrix< OutputType, BSP1D, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType1, BSP1D, RIT2, CIT2, NIT2 > &A,
		const Matrix< InputType2, BSP1D, RIT3, CIT3, NIT3 > &B,
		const MulMonoid &mul,
		const Phase phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< MulMonoid >::value,
		void >::type * const = nullptr
	) {
		assert( phase != TRY );
		RC local_rc = SUCCESS;
		if( phase == RESIZE ) {
			RC ret = eWiseApply< descr >(
				internal::getLocal( C ),
				internal::getLocal( A ), internal::getLocal( B ),
				mul,
				RESIZE
			);
			if( collectives<>::allreduce( ret, operators::any_or< RC >() ) != SUCCESS ) {
				return PANIC;
			} else {
				return ret;
			}
		} else {
			assert( phase == EXECUTE );
			local_rc = eWiseApply< descr >(
				internal::getLocal( C ),
				internal::getLocal( A ), internal::getLocal( B ),
				mul,
				EXECUTE
			);
		}
		return internal::checkGlobalErrorStateOrClear( C, local_rc );
	}

	/** \internal Simply delegates to process-local backend */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2,
		typename RIT3, typename CIT3, typename NIT3
	>
	RC eWiseApply(
		Matrix< OutputType, BSP1D, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType1, BSP1D, RIT2, CIT2, NIT2 > &A,
		const Matrix< InputType2, BSP1D, RIT3, CIT3, NIT3 > &B,
		const Operator &op,
		const Phase phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value,
		void >::type * const = nullptr
	) {
		assert( phase != TRY );
		RC ret = eWiseApply< descr >(
			internal::getLocal( C ),
			internal::getLocal( A ), internal::getLocal( B ),
			op,
			phase
		);
		if( phase == RESIZE ) {
			if( collectives<>::allreduce( ret, operators::any_or< RC >() ) != SUCCESS ) {
				return PANIC;
			} else {
				return SUCCESS;
			}
		}
		assert( phase == EXECUTE );
		return internal::checkGlobalErrorStateOrClear( C, ret );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType, typename MaskType,
		typename RIT_A, typename CIT_A, typename NIT_A,
		typename RIT_M, typename CIT_M, typename NIT_M
	>
	RC foldr(
		IOType &x,
		const Matrix< InputType, BSP1D, RIT_A, CIT_A, NIT_A > &A,
		const Matrix< MaskType, BSP1D, RIT_M, CIT_M, NIT_M > &mask,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( !std::is_same< InputType, void >::value,
			"grb::foldr( BSP1D, IOType <- op( InputType, IOType ): "
			"the operator version of foldr cannot be used if the "
			"input matrix is a pattern matrix (of type void)"
		);
		static_assert( !std::is_same< IOType, void >::value,
			"grb::foldr( BSP1D, IOType <- op( InputType, IOType ): "
			"the operator version of foldr cannot be used if the "
			"result is of type void"
		);
		static_assert( (std::is_same< typename Monoid::D1, InputType >::value),
			"grb::foldr( BSP1D, IOType <- op( InputType, IOType ): "
			"called with a prefactor input type that does not match the first domain of the given operator"
		);
		static_assert( (std::is_same< typename Monoid::D2, IOType >::value),
			"grb::foldr( BSP1D, IOType <- op( InputType, IOType ): "
			"called with a postfactor input type that does not match the first domain of the given operator"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldr( BSP1D, IOType <- op( InputType, IOType ): "
			"called with an output type that does not match the output domain of the given operator"
		);

#ifdef _DEBUG
		std::cout << "In grb::foldr( BSP1D, matrix, mask, monoid )\n";
#endif

		if( grb::nnz( A ) == 0 ) {
			return RC::SUCCESS;
		}

		RC rc = SUCCESS;

		// Do local folding
		IOType local = monoid.template getIdentity< IOType >();
		rc = foldr< descr >( local, internal::getLocal( A ), internal::getLocal( mask ), monoid );

#ifdef _DEBUG
		std::cout << "After process-local delegation, local value has become "
			<< local << ". Entering allreduce..." << std::endl;
#endif

		// All-reduce using \a op
		rc = rc ? rc : collectives< BSP1D >::allreduce< descr >( local, monoid.getOperator() );

		// Accumulate end result
		rc = rc ? rc : foldr( x, local, monoid.getOperator() );

		return rc;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType,
		typename RIT, typename CIT, typename NIT
	>
	RC foldr(
		IOType &x,
		const Matrix< InputType, BSP1D, RIT, CIT, NIT > &A,
		const Monoid &monoid,
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( !std::is_same< InputType, void >::value,
			"grb::foldr( BSP1D, IOType <- op( IOType, InputType ): "
			"the operator version of foldr cannot be used if the "
			"input matrix is a pattern matrix (of type void)"
		);
		static_assert( !std::is_same< IOType, void >::value,
			"grb::foldr( BSP1D, IOType <- op( IOType, InputType ): "
			"the operator version of foldr cannot be used if the "
			"result is of type void"
		);
		static_assert( (std::is_same< typename Monoid::D1, InputType >::value),
			"grb::foldr( BSP1D, IOType <- op( IOType, InputType ): "
			"called with a prefactor input type that does not match the first domain of the given operator"
		);
		static_assert( (std::is_same< typename Monoid::D2, IOType >::value),
			"grb::foldr( BSP1D, IOType <- op( IOType, InputType ): "
			"called with a postfactor input type that does not match the first domain of the given operator"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldr( BSP1D, IOType <- op( IOType, InputType ): "
			"called with an output type that does not match the output domain of the given operator"
		);

#ifdef _DEBUG
		std::cout << "In grb::foldr( BSP1D, matrix, monoid )\n";
#endif

		if( grb::nnz( A ) == 0 ) {
			return RC::SUCCESS;
		}

		RC rc = SUCCESS;

		// Do local folding
		IOType local = monoid.template getIdentity< IOType >();
		rc = foldr< descr >( local, internal::getLocal( A ), monoid );

#ifdef _DEBUG
		std::cout << "After process-local delegation, local value has become "
			<< local << ". Entering allreduce..." << std::endl;
#endif

		// All-reduce using \a op
		rc = rc ? rc : collectives< BSP1D >::allreduce< descr >( local, monoid.getOperator() );

		// Accumulate end result
		rc = rc ? rc : foldr( x, local, monoid.getOperator() );

		return rc;
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
		const Matrix< InputType, BSP1D, RIT_A, CIT_A, NIT_A > &A,
		const Matrix< MaskType, BSP1D, RIT_M, CIT_M, NIT_M > &mask,
		const Monoid &monoid,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( !std::is_same< InputType, void >::value,
			"grb::foldl( BSP1D, IOType <- op( IOType, InputType ): "
			"the operator version of foldl cannot be used if the "
			"input matrix is a pattern matrix (of type void)"
		);
		static_assert( !std::is_same< IOType, void >::value,
			"grb::foldl( BSP1D, IOType <- op( IOType, InputType ): "
			"the operator version of foldl cannot be used if the "
			"result is of type void"
		);
		static_assert( (std::is_same< typename Monoid::D1, IOType >::value),
			"grb::foldl( BSP1D, IOType <- op( IOType, InputType ): "
			"called with a prefactor input type that does not match the first domain of the given operator"
		);
		static_assert( (std::is_same< typename Monoid::D2, InputType >::value),
			"grb::foldl( BSP1D, IOType <- op( IOType, InputType ): "
			"called with a postfactor input type that does not match the first domain of the given operator"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldl( BSP1D, IOType <- op( IOType, InputType ): "
			"called with an output type that does not match the output domain of the given operator"
		);

#ifdef _DEBUG
		std::cout << "In grb::foldl( BSP1D, matrix, mask, monoid )\n";
#endif

		if( grb::nnz( A ) == 0 ) {
			return RC::SUCESS;
		}
		
		RC rc = SUCCESS;

		// Do local folding
		IOType local = monoid.template getIdentity< IOType >();
		rc = foldl< descr >( local, internal::getLocal( A ), internal::getLocal( mask ), monoid );

#ifdef _DEBUG
		std::cout << "After process-local delegation, local value has become "
			<< local << ". Entering allreduce..." << std::endl;
#endif

		// All-reduce using \a op
		rc = rc ? rc : collectives< BSP1D >::allreduce< descr >( local, monoid.getOperator() );

		// Accumulate end result
		rc = rc ? rc : foldl( x, local, monoid.getOperator() );

		return rc;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType,
		typename RIT, typename CIT, typename NIT
	>
	RC foldl(
		IOType &x,
		const Matrix< InputType, BSP1D, RIT, CIT, NIT > &A,
		const Monoid &monoid,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( !std::is_same< InputType, void >::value,
			"grb::foldl( BSP1D, IOType <- op( IOType, InputType ): "
			"the operator version of foldl cannot be used if the "
			"input matrix is a pattern matrix (of type void)"
		);
		static_assert( !std::is_same< IOType, void >::value,
			"grb::foldl( BSP1D, IOType <- op( IOType, InputType ): "
			"the operator version of foldl cannot be used if the "
			"result is of type void"
		);
		static_assert( (std::is_same< typename Monoid::D1, IOType >::value),
			"grb::foldl( BSP1D, IOType <- op( IOType, InputType ): "
			"called with a prefactor input type that does not match the first domain of the given operator"
		);
		static_assert( (std::is_same< typename Monoid::D2, InputType >::value),
			"grb::foldl( BSP1D, IOType <- op( IOType, InputType ): "
			"called with a postfactor input type that does not match the first domain of the given operator"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldl( BSP1D, IOType <- op( IOType, InputType ): "
			"called with an output type that does not match the output domain of the given operator"
		);

#ifdef _DEBUG
		std::cout << "In grb::foldl( BSP1D, matrix, monoid )\n";
#endif

		if( grb::nnz( A ) == 0 ) {
			return RC::SUCCESS;
		}

		RC rc = SUCCESS;

		// Do local folding
		IOType local = monoid.template getIdentity< IOType >();
		rc = foldl< descr >( local, internal::getLocal( A ), monoid );

#ifdef _DEBUG
		std::cout << "After process-local delegation, local value has become "
			<< local << ". Entering allreduce..." << std::endl;
#endif

		// All-reduce using \a op
		rc = rc ? rc : collectives< BSP1D >::allreduce< descr >( local, monoid.getOperator() );

		// Accumulate end result
		rc = rc ? rc : foldl( x, local, monoid.getOperator() );

		return rc;
	}

} // namespace grb

#endif

