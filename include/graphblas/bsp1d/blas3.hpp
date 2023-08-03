
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
			if( collectives<>::allreduce(
					ret, operators::any_or< RC >()
				) != SUCCESS
			) {
				return PANIC;
			}
			return SUCCESS;
		}
		assert( phase == EXECUTE );
		return internal::checkGlobalErrorStateOrClear( C, ret );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename OutputType,
		typename RIT_L, typename CIT_L, typename NIT_L,
		typename RIT_A, typename CIT_A, typename NIT_A
	>
	RC tril(
		Matrix< OutputType, BSP1D, RIT_L, CIT_L, NIT_L > &L,
		const Matrix< InputType, BSP1D, RIT_A, CIT_A, NIT_A > &A,
		const long int k,
		const Phase &phase = Phase::EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType >::value &&
			std::is_convertible< InputType, OutputType >::value
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In grb::tril( BSP1D )\n";
#endif
		assert( phase != TRY );
		const std::pair<size_t, size_t> anchor = internal::getGlobalAnchor( A );

		const RC ret = tril< descr >(
			internal::getLocal( L ),
			internal::getLocal( A ),
			k,
			phase,
			anchor.first,
			anchor.second
		);

		return internal::checkGlobalErrorStateOrClear( L, ret );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename OutputType,
		typename RIT_L, typename CIT_L, typename NIT_L,
		typename RIT_A, typename CIT_A, typename NIT_A
	>
	RC tril(
		Matrix< OutputType, BSP1D, RIT_L, CIT_L, NIT_L > &L,
		const Matrix< InputType, BSP1D, RIT_A, CIT_A, NIT_A > &A,
		const Phase &phase = Phase::EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType >::value &&
			std::is_convertible< InputType, OutputType >::value
		>::type * const = nullptr
	) {
		return tril< descr >( L, A, 0, phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename OutputType,
		typename RIT_U, typename CIT_U, typename NIT_U,
		typename RIT_A, typename CIT_A, typename NIT_A
	>
	RC triu(
		Matrix< OutputType, BSP1D, RIT_U, CIT_U, NIT_U > &U,
		const Matrix< InputType, BSP1D, RIT_A, CIT_A, NIT_A > &A,
		const long int k,
		const Phase &phase = Phase::EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType >::value &&
			std::is_convertible< InputType, OutputType >::value
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In grb::triu( BSP1D )\n";
#endif
		assert( phase != TRY );
		const RC ret = triu< descr >(
			internal::getLocal( U ),
			internal::getLocal( A ),
			k,
			phase
		);

		return internal::checkGlobalErrorStateOrClear( U, ret );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename OutputType,
		typename RIT_U, typename CIT_U, typename NIT_U,
		typename RIT_A, typename CIT_A, typename NIT_A
	>
	RC triu(
		Matrix< OutputType, BSP1D, RIT_U, CIT_U, NIT_U > &U,
		const Matrix< InputType, BSP1D, RIT_A, CIT_A, NIT_A > &A,
		const Phase &phase = Phase::EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType >::value &&
			std::is_convertible< InputType, OutputType >::value
		>::type * const = nullptr
	) {
		return triu< descr >( U, A, 0, phase );
	}

} // namespace grb

#endif

