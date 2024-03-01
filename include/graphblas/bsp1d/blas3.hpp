
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

#include "graphblas/blas3.hpp"
#include "graphblas/backends.hpp"

#include "graphblas/bsp/utils.hpp"

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
				return ret;
			}
		}
		assert( phase == EXECUTE );
		return internal::checkGlobalErrorStateOrClear( C, ret );
	}

	/** \internal Simply delegates to process-local backend */
	template<
		Descriptor descr = descriptors::no_operation,
		class SelectionOperator,
		typename Tin,
		typename RITin, typename CITin, typename NITin,
		typename Tout,
		typename RITout, typename CITout, typename NITout
	>
	RC select(
		Matrix< Tout, BSP1D, RITout, CITout, NITout > &out,
		const Matrix< Tin, BSP1D, RITin, CITin, NITin > &in,
		const SelectionOperator &op,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!is_object< Tin >::value &&
			!is_object< Tout >::value
		>::type * const = nullptr
	) {
		assert( phase != TRY );

		const auto coordinatesTranslationFunctions =
			in.getLocalToGlobalCoordinatesTranslationFunctions();

		RC ret = internal::select_generic< descr >(
			internal::getLocal( out ),
			internal::getLocal( in ),
			op,
			std::get<0>(coordinatesTranslationFunctions),
			std::get<1>(coordinatesTranslationFunctions),
			phase
		);

		if( phase == RESIZE ) {
			if( collectives<>::allreduce( ret, operators::any_or< RC >() ) != SUCCESS ) {
				return PANIC;
			} else {
				return ret;
			}
		}
		assert( phase == EXECUTE );
		return internal::checkGlobalErrorStateOrClear( out, ret );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType, typename MaskType,
		typename RIT_A, typename CIT_A, typename NIT_A,
		typename RIT_M, typename CIT_M, typename NIT_M
	>
	RC foldr(
		const Matrix< InputType, BSP1D, RIT_A, CIT_A, NIT_A > &A,
		const Matrix< MaskType, BSP1D, RIT_M, CIT_M, NIT_M > &mask,
		IOType &x,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( !std::is_void< InputType >::value,
			"grb::foldr( BSP1D, IOType <- [[IOType]], monoid, masked ): "
			"the provided matrix may not be a pattern matrix."
			"Possible fix: provide a semiring instead of a ring"
		);
		static_assert( grb::is_commutative< Monoid >::value,
			"grb::foldr( BSP1D, IOType <- [[IOType]], monoid, masked ): "
			"the provided monoid must be commutative (but is not)"
		);
		static_assert( (std::is_same< typename Monoid::D1, InputType >::value),
			"grb::foldr( BSP1D, IOType <- [[IOType]], monoid, masked ): "
			"called with a prefactor input type that does not match the first domain of "
			"the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D2, IOType >::value),
			"grb::foldr( BSP1D, IOType <- [[IOType]], monoid, masked ): "
			"called with a postfactor input type that does not match the first domain "
			"of the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldr( BSP1D, IOType <- [[IOType]], monoid, masked ): "
			"called with an output type that does not match the output domain of the "
			"given monoid"
		);
		static_assert( !(descr & descriptors::add_identity),
			"grb::foldr( BSP1D, IOType <- [[IOType]], monoid, masked ): "
			"the use of the add_identity descriptor requires a semiring, but a monoid "
			"was given"
		);
		static_assert( !(
				(descr & descriptors::invert_mask) &&
				(descr & descriptors::structural)
			), "grb::foldr( BSP1D, IOType <- [[IOType]], monoid, masked ): "
			"may not select an inverted structural mask for matrices"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldr( BSP1D, matrix, mask, monoid )\n";
#endif
		// first check whether we can dispatch
		if( grb::nrows( mask ) == 0 && grb::ncols( mask ) == 0 ) {
			return foldr< descr >( A, x, monoid );
		}

		// dynamic checks
		if( grb::nrows( A ) != grb::nrows( mask ) ||
			grb::ncols( A ) != grb::ncols( mask )
		) {
			return RC::MISMATCH;
		}

		// check for trivial op
		if( grb::nnz( A ) == 0 || grb::nnz( mask ) == 0 ||
			grb::nrows( A ) == 0 || grb::ncols( A ) == 0
		) {
			return RC::SUCCESS;
		}

		// do local folding
		RC rc = SUCCESS;
		IOType local = monoid.template getIdentity< IOType >();
		rc = foldr< descr >( internal::getLocal( A ),
			internal::getLocal( mask ), local, monoid );

		// note we are not synchronising the error code, since by this point any non-
		// success error-code will always collective. However, in the spirit of
		// defensive programming, we will assert this when in debug mode:
#ifndef NDEBUG
		rc = internal::assertSyncedRC( rc );
#endif
#ifdef _DEBUG
		std::cout << "After process-local delegation, local value has become "
			<< local << ". Entering allreduce..." << std::endl;
#endif

		// All-reduce using \a op
		rc = rc ? rc :
			collectives< BSP1D >::allreduce< descr >( local, monoid.getOperator() );

		// Accumulate end result
		rc = rc ? rc : foldr( x, local, monoid.getOperator() );

		// done
		return rc;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Semiring,
		typename InputType, typename IOType, typename MaskType,
		typename RIT_A, typename CIT_A, typename NIT_A,
		typename RIT_M, typename CIT_M, typename NIT_M
	>
	RC foldr(
		const Matrix< InputType, BSP1D, RIT_A, CIT_A, NIT_A > &A,
		const Matrix< MaskType, BSP1D, RIT_M, CIT_M, NIT_M > &mask,
		IOType &x,
		const Semiring &semiring = Semiring(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Semiring >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( (std::is_same< typename Semiring::D3, InputType >::value),
			"grb::foldr( BSP1D, IOType <- [[IOType]], semiring, masked ): "
			"called with a prefactor input type that does not match the third domain "
			"of the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, IOType >::value),
			"grb::foldr( BSP1D, IOType <- [[IOType]], semiring, masked ): "
			"called with a postfactor input type that does not match the fourth domain "
			"of the given semiring"
		);
		static_assert( !(
				(descr & descriptors::invert_mask) &&
				(descr & descriptors::structural)
			), "grb::foldr( BSP1D, IOType <- [[IOType]], semiring, masked ): "
			"may not select an inverted structural mask for matrices"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldr( BSP1D, matrix, mask, semiring )\n";
#endif
		// first check whether we can dispatch
		if( grb::nrows( mask ) == 0 && grb::ncols( mask ) == 0 ) {
			return foldr< descr >( A, x, semiring );
		}

		// dynamic checks
		if( grb::nrows( A ) != grb::nrows( mask ) ||
			grb::ncols( A ) != grb::ncols( mask )
		) {
			return RC::MISMATCH;
		}

		// check for trivial op
		if( grb::nnz( A ) == 0 || grb::nnz( mask ) == 0 ||
			grb::nrows( A ) == 0 || grb::ncols( A ) == 0
		) {
			return RC::SUCCESS;
		}

		// do local folding
		RC rc = SUCCESS;
		IOType local = semiring.template getZero< IOType >();
		rc = foldr< descr >( internal::getLocal( A ),
			internal::getLocal( mask ), local, semiring );

		// note we are not synchronising the error code, since by this point any non-
		// success error-code will always collective. However, in the spirit of
		// defensive programming, we will assert this when in debug mode:
#ifndef NDEBUG
		rc = internal::assertSyncedRC( rc );
#endif
#ifdef _DEBUG
		std::cout << "After process-local delegation, local value has become "
			<< local << ". Entering allreduce..." << std::endl;
#endif

		// All-reduce using \a op
		rc = rc ? rc :
			collectives< BSP1D >::allreduce< descr >( local,
				semiring.getAdditiveOperator() );

		// Accumulate end result
		rc = rc ? rc : foldr( x, local, semiring.getAdditiveOperator() );

		// done
		return rc;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType,
		typename RIT, typename CIT, typename NIT
	>
	RC foldr(
		const Matrix< InputType, BSP1D, RIT, CIT, NIT > &A,
		IOType &x,
		const Monoid &monoid,
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( !std::is_void< InputType >::value,
			"grb::foldr( BSP1D, IOType <- [[IOType]], monoid ): "
			"the provided matrix may not be a pattern matrix."
			"Possible fix: provide a semiring instead of a ring"
		);
		static_assert( grb::is_commutative< Monoid >::value,
			"grb::foldr( BSP1D, IOType <- [[IOType]], monoid ): "
			"the provided monoid must be commutative (but is not)"
		);
		static_assert( (std::is_same< typename Monoid::D1, InputType >::value),
			"grb::foldr( BSP1D, IOType <- [[IOType]], monoid ): "
			"called with a prefactor input type that does not match the first domain of "
			"the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D2, IOType >::value),
			"grb::foldr( BSP1D, IOType <- [[IOType]], monoid ): "
			"called with a postfactor input type that does not match the first domain "
			"of the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldr( BSP1D, IOType <- [[IOType]], monoid ): "
			"called with an output type that does not match the output domain of the "
			"given monoid"
		);
		static_assert( !(descr & descriptors::add_identity),
			"grb::foldr( BSP1D, IOType <- [[IOType]], monoid ): "
			"the use of the add_identity descriptor requires a semiring, but a monoid "
			"was given"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldr( BSP1D, matrix, monoid )\n";
#endif

		// check for trivial op
		if( grb::nnz( A ) == 0 || grb::ncols( A ) == 0 || grb::nrows( A ) == 0 ) {
			return RC::SUCCESS;
		}

		// do local folding
		RC rc = SUCCESS;
		IOType local = monoid.template getIdentity< IOType >();
		rc = foldr< descr >( internal::getLocal( A ), local, monoid );

		// note we are not synchronising the error code, since by this point any non-
		// success error-code will always collective. However, in the spirit of
		// defensive programming, we will assert this when in debug mode:
#ifndef NDEBUG
		rc = internal::assertSyncedRC( rc );
#endif
#ifdef _DEBUG
		std::cout << "After process-local delegation, local value has become "
			<< local << ". Entering allreduce..." << std::endl;
#endif

		// All-reduce using \a op
		rc = rc ? rc :
			collectives< BSP1D >::allreduce< descr >( local, monoid.getOperator() );

		// Accumulate end result
		rc = rc ? rc : foldr( x, local, monoid.getOperator() );

		// done
		return rc;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Semiring,
		typename InputType, typename IOType,
		typename RIT, typename CIT, typename NIT
	>
	RC foldr(
		const Matrix< InputType, BSP1D, RIT, CIT, NIT > &A,
		IOType &x,
		const Semiring &semiring = Semiring(),
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_semiring< Semiring >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( (std::is_same< typename Semiring::D3, InputType >::value),
			"grb::foldr( BSP1D, IOType <- [[IOType]], semiring ): "
			"called with a prefactor input type that does not match the third domain "
			"the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, IOType >::value),
			"grb::foldr( BSP1D, IOType <- [[IOType]], semiring ): "
			"called with a postfactor input type that does not match the fourth domain "
			"of the given semiring"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldr( BSP1D, matrix, semiring )\n";
#endif

		// check for trivial op
		if( grb::nnz( A ) == 0 || grb::ncols( A ) == 0 || grb::nrows( A ) == 0 ) {
			return RC::SUCCESS;
		}

		// do local folding
		RC rc = SUCCESS;
		IOType local = semiring.template getZero< IOType >();
		rc = foldr< descr >( internal::getLocal( A ), local, semiring );

		// note we are not synchronising the error code, since by this point any non-
		// success error-code will always collective. However, in the spirit of
		// defensive programming, we will assert this when in debug mode:
#ifndef NDEBUG
		rc = internal::assertSyncedRC( rc );
#endif
#ifdef _DEBUG
		std::cout << "After process-local delegation, local value has become "
			<< local << ". Entering allreduce..." << std::endl;
#endif

		// All-reduce using \a op
		rc = rc ? rc :
			collectives< BSP1D >::allreduce< descr >( local,
				semiring.getAdditiveOperator() );

		// Accumulate end result
		rc = rc ? rc : foldr( x, local, semiring.getAdditiveOperator() );

		// done
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
		static_assert( !std::is_void< InputType >::value,
			"grb::foldl( BSP1D, IOType <- [[IOType]], monoid, masked ): "
			"the provided matrix may not be a pattern matrix."
			"Possible fix: provide a semiring instead of a ring"
		);
		static_assert( grb::is_commutative< Monoid >::value,
			"grb::foldl( BSP1D, IOType <- [[IOType]], monoid, masked ): "
			"the provided monoid must be commutative (but is not)"
		);
		static_assert( (std::is_same< typename Monoid::D1, IOType >::value),
			"grb::foldl( BSP1D, IOType <- [[IOType]], monoid, masked ): "
			"called with a prefactor input type that does not match the first domain of "
			"the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D2, InputType >::value),
			"grb::foldl( BSP1D, IOType <- [[IOType]], monoid, masked ): "
			"called with a postfactor input type that does not match the first domain "
			"of the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldl( BSP1D, IOType <- [[IOType]], monoid, masked ): "
			"called with an output type that does not match the output domain of the "
			"given monoid"
		);
		static_assert( !(descr & descriptors::add_identity),
			"grb::foldl( BSP1D, IOType <- [[IOType]], monoid, masked ): "
			"the use of the add_identity descriptor requires a semiring, but a monoid "
			"was given"
		);
		static_assert( !(
				(descr & descriptors::invert_mask) &&
				(descr & descriptors::structural)
			), "grb::foldl( BSP1D, IOType <- [[IOType]], monoid, masked ): "
			"may not select an inverted structural mask for matrices"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldl( BSP1D, matrix, mask, monoid )\n";
#endif
		// first check whether we can dispatch
		if( grb::nrows( mask ) == 0 && grb::ncols( mask ) == 0 ) {
			return foldl< descr >( x, A, monoid );
		}

		// dynamic checks
		if( grb::nrows( A ) != grb::nrows( mask ) ||
			grb::ncols( A ) != grb::ncols( mask )
		) {
			return RC::MISMATCH;
		}

		// check for trivial op
		if( nnz( A ) == 0 || grb::nnz( mask ) == 0 ||
			grb::nrows( A ) == 0 || grb::ncols( A ) == 0
		) {
#ifdef _DEBUG
			std::cout << "Input matrix has no entries; returning identity" << std::endl;
#endif
			return SUCCESS;
		}

		// do local folding
		RC rc = SUCCESS;
		IOType local = monoid.template getIdentity< IOType >();
		rc = foldl< descr >( local, internal::getLocal( A ),
			internal::getLocal( mask ), monoid );

		// note we are not synchronising the error code, since by this point any non-
		// success error-code will always collective. However, in the spirit of
		// defensive programming, we will assert this when in debug mode:
#ifndef NDEBUG
		rc = internal::assertSyncedRC( rc );
#endif
#ifdef _DEBUG
		std::cout << "After process-local delegation, local value has become "
			<< local << ". Entering allreduce..." << std::endl;
#endif

		// All-reduce using \a op
		rc = rc ? rc : collectives< BSP1D >::allreduce< descr >( local,
			monoid.getOperator() );

		// Accumulate end result
		rc = rc ? rc : foldl( x, local, monoid.getOperator() );

		// done
		return rc;
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
		const Matrix< InputType, BSP1D, RIT_A, CIT_A, NIT_A > &A,
		const Matrix< MaskType, BSP1D, RIT_M, CIT_M, NIT_M > &mask,
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
			"grb::foldl( BSP1D, IOType <- [[IOType]], semiring, masked ): "
			"called with a prefactor input type that does not match the third domain "
			"the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, InputType >::value),
			"grb::foldl( BSP1D, IOType <- [[IOType]], semiring, masked ): "
			"called with a postfactor input type that does not match the fourth domain "
			"of the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, IOType >::value),
			"grb::foldl( BSP1D, IOType <- [[IOType]], semiring, masked ): "
			"called with an output type that does not match the fourth domain of the "
			"given semiring"
		);
		static_assert( !(
				(descr & descriptors::invert_mask) &&
				(descr & descriptors::structural)
			), "grb::foldl( BSP1D, IOType <- [[IOType]], semiring, masked ): "
			"may not select an inverted structural mask for matrices"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldl( BSP1D, matrix, mask, semiring )\n";
#endif
		// first check whether we can dispatch
		if( grb::nrows( mask ) == 0 && grb::ncols( mask ) == 0 ) {
			return foldl< descr >( x, A, semiring );
		}

		// dynamic checks
		if( grb::nrows( A ) != grb::nrows( mask ) ||
			grb::ncols( A ) != grb::ncols( mask )
		) {
			return RC::MISMATCH;
		}

		// check for trivial op
		if( nnz( A ) == 0 || grb::nnz( mask ) == 0 ||
			grb::nrows( A ) == 0 || grb::ncols( A ) == 0
		) {
#ifdef _DEBUG
			std::cout << "Input matrix has no entries; returning identity" << std::endl;
#endif
			return SUCCESS;
		}

		// do local folding
		RC rc = SUCCESS;
		IOType local = semiring.template getZero< IOType >();
		if( descr & descriptors::add_identity ) {
			const auto& union_to_global_coordinates_translators =
				A.unionToGlobalCoordinatesTranslators();
			rc = internal::fold_masked_generic__add_identity<
					descr, true, Semiring
				>(
					local,
					internal::getLocal( A ),
					internal::getLocal( mask ),
					std::get<0>(union_to_global_coordinates_translators),
					std::get<1>(union_to_global_coordinates_translators),
					semiring
				);
		} else {
			rc = foldl< descr & (~descriptors::add_identity) >(
				local,
				internal::getLocal( A ),
				internal::getLocal( mask ),
				semiring
			);
		}

		// note we are not synchronising the error code, since by this point any non-
		// success error-code will always collective. However, in the spirit of
		// defensive programming, we will assert this when in debug mode:
#ifndef NDEBUG
		rc = internal::assertSyncedRC( rc );
#endif
#ifdef _DEBUG
		std::cout << "After process-local delegation, local value has become "
			<< local << ". Entering allreduce..." << std::endl;
#endif

		// All-reduce using \a op
		rc = rc ? rc : collectives< BSP1D >::allreduce< descr >( local,
			semiring.getAdditiveOperator() );

		// Accumulate end result
		rc = rc ? rc : foldl( x, local, semiring.getAdditiveOperator() );

		// done
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
		static_assert( !std::is_void< InputType >::value,
			"grb::foldl( BSP1D, IOType <- [[IOType]], monoid ): "
			"the provided matrix may not be a pattern matrix."
			"Possible fix: provide a semiring instead of a ring"
		);
		static_assert( grb::is_commutative< Monoid >::value,
			"grb::foldl( BSP1D, IOType <- [[IOType]], monoid ): "
			"the provided monoid must be commutative (but is not)"
		);
		static_assert( (std::is_same< typename Monoid::D1, IOType >::value),
			"grb::foldl( BSP1D, IOType <- [[IOType]], monoid ): "
			"called with a prefactor input type that does not match the first domain of "
			"the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D2, InputType >::value),
			"grb::foldl( BSP1D, IOType <- [[IOType]], monoid ): "
			"called with a postfactor input type that does not match the first domain "
			"of the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldl( BSP1D, IOType <- [[IOType]], monoid ): "
			"called with an output type that does not match the output domain of the "
			"given monoid"
		);
		static_assert( !(descr & descriptors::add_identity),
			"grb::foldl( BSP1D, IOType <- [[IOType]], monoid ): "
			"the use of the add_identity descriptor requires a semiring, but a monoid "
			"was given"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldl( BSP1D, matrix, monoid )\n";
#endif

		// check for trivial op
		if( nnz( A ) == 0 || nrows( A ) == 0 || ncols( A ) == 0 ) {
#ifdef _DEBUG
			std::cout << "Input matrix has no entries; returning identity" << std::endl;
#endif
			return SUCCESS;
		}

		// do local folding
		RC rc = SUCCESS;
		IOType local = monoid.template getIdentity< IOType >();
		rc = foldl< descr >( local, internal::getLocal( A ), monoid );

		// note we are not synchronising the error code, since by this point any non-
		// success error-code will always collective. However, in the spirit of
		// defensive programming, we will assert this when in debug mode:
#ifndef NDEBUG
		rc = internal::assertSyncedRC( rc );
#endif
#ifdef _DEBUG
		std::cout << "After process-local delegation, local value has become "
			<< local << ". Entering allreduce..." << std::endl;
#endif

		// All-reduce using \a op
		rc = rc ? rc : collectives< BSP1D >::allreduce< descr >( local,
			monoid.getOperator() );

		// Accumulate end result
		rc = rc ? rc : foldl( x, local, monoid.getOperator() );

		// done
		return rc;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Semiring,
		typename InputType, typename IOType,
		typename RIT, typename CIT, typename NIT
	>
	RC foldl(
		IOType &x,
		const Matrix< InputType, BSP1D, RIT, CIT, NIT > &A,
		const Semiring &semiring = Semiring(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_semiring< Semiring >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( (std::is_same< typename Semiring::D3, IOType >::value),
			"grb::foldl( BSP1D, IOType <- [[IOType]], semiring ): "
			"called with a prefactor input type that does not match the third domain "
			"of the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, InputType >::value),
			"grb::foldl( BSP1D, IOType <- [[IOType]], semiring ): "
			"called with a postfactor input type that does not match the fourth domain "
			"of the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, IOType >::value),
			"grb::foldl( BSP1D, IOType <- [[IOType]], semiring ): "
			"called with an output type that does not match the fourth domain of the "
			"given semiring"
		);
#ifdef _DEBUG
		std::cout << "In grb::foldl( BSP1D, matrix, semiring )\n";
#endif

		// check for trivial op
		if( nnz( A ) == 0 || nrows( A ) == 0 || ncols( A ) == 0 ) {
#ifdef _DEBUG
			std::cout << "Input matrix has no entries; returning identity" << std::endl;
#endif
			return SUCCESS;
		}

		// do local folding
		RC rc = SUCCESS;
		IOType local = semiring.template getZero< IOType >();
		rc = foldl< descr >( local, internal::getLocal( A ), semiring );

		// note we are not synchronising the error code, since by this point any non-
		// success error-code will always collective. However, in the spirit of
		// defensive programming, we will assert this when in debug mode:
#ifndef NDEBUG
		rc = internal::assertSyncedRC( rc );
#endif
#ifdef _DEBUG
		std::cout << "After process-local delegation, local value has become "
			<< local << ". Entering allreduce..." << std::endl;
#endif

		// All-reduce using \a op
		rc = rc ? rc : collectives< BSP1D >::allreduce< descr >( local,
			semiring.getAdditiveOperator() );

		// Accumulate end result
		rc = rc ? rc : foldl( x, local, semiring.getAdditiveOperator() );

		// done
		return rc;
	}

} // namespace grb

#endif

