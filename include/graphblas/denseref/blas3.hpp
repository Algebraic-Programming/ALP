
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

#ifndef _H_GRB_DENSEREF_BLAS3
#define _H_GRB_DENSEREF_BLAS3

#include <type_traits> //for std::enable_if

#include <graphblas/base/blas3.hpp>

#include "io.hpp"
#include "matrix.hpp"

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

		/**
		 * \internal general mxm implementation that all mxm variants refer to
		 */
		template<
			bool allow_void,
			class MulMonoid,
			typename OutputType, typename InputType1, typename InputType2,
			class Operator, class Monoid
		>
		RC mxm_generic( Matrix< OutputType, reference_dense > &C,
			const Matrix< InputType1, reference_dense > &A,
			const Matrix< InputType2, reference_dense > &B,
			const Operator &oper,
			const Monoid &monoid,
			const MulMonoid &mulMonoid,
			const typename std::enable_if< !grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value && !
				grb::is_object< InputType2 >::value &&
				grb::is_operator< Operator >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = NULL
		) {
			(void)oper;
			(void)monoid;
			(void)mulMonoid;
			static_assert( allow_void ||
				( !(
					std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value
				) ),
				"grb::mxm_generic: the operator-monoid version of mxm cannot be "
				"used if either of the input matrices is a pattern matrix (of type "
				"void)"
			);

#ifdef _DEBUG
			std::cout << "In grb::internal::mxm_generic (reference_dense, unmasked)\n";
#endif

			// run-time checks
			const size_t m = grb::nrows( C );
			const size_t n = grb::ncols( C );
			const size_t m_A = grb::nrows( A );
			const size_t k = grb::ncols( A );
			const size_t k_B = grb::nrows( B );
			const size_t n_B = grb::ncols( B );

			if( m != m_A || k != k_B || n != n_B ) {
				return MISMATCH;
			}

			const auto A_raw = grb::getRaw( A );
			const auto B_raw = grb::getRaw( B );
			auto C_raw = grb::getRaw( C );

			std::cout << "Multiplying dense matrices.\n";

			for( size_t row = 0; row < m; ++row ) {
				for( size_t col = 0; col < n; ++col ) {
					C_raw[ row * k + col] = monoid.template getIdentity< OutputType >();
					for( size_t i = 0; i < k; ++ i ) {
						OutputType temp = monoid.template getIdentity< OutputType >();
						(void)grb::apply( temp, A_raw[ row * k + i ], B_raw[ i * n_B + col ], oper );
						(void)grb::foldl( C_raw[ row * k + col], temp, monoid.getOperator() );
					}
				}
			}
			grb::internal::setInitialized( C, true );
			// done
			return SUCCESS;
		}

		/**
		 * \internal general mxm implementation that all mxm variants using structured matrices refer to
		 */
		template<
			bool allow_void,
			class MulMonoid,
			typename OutputType, typename InputType1, typename InputType2,
			class Operator, class Monoid,
			typename OutputStructure, typename InputStructure1, typename InputStructure2,
			typename OutputView, typename InputView1, typename InputView2
		>
		RC mxm_generic( StructuredMatrix< OutputType, OutputStructure, storage::Dense, OutputView, reference_dense > &C,
			const StructuredMatrix< InputType1, InputStructure1, storage::Dense, InputView1, reference_dense > &A,
			const StructuredMatrix< InputType2, InputStructure2, storage::Dense, InputView2, reference_dense > &B,
			const Operator &oper,
			const Monoid &monoid,
			const MulMonoid &mulMonoid,
			const typename std::enable_if< !grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value && !
				grb::is_object< InputType2 >::value &&
				grb::is_operator< Operator >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = NULL
		) {
			// TODO; Implement this as a backup version that works for any structure and storage.
			// Even though the performance does not have to be optimal, we guarantee that any two matrices can be multiplied
			// To provide better performing mxm, one should implement a function with specialized template elements
			// This currently cannot work as we do not have a generic way to access elements in a given StructuredMatrix
			return UNSUPPORTED;
		}

		/**
		 * \internal mxm specialized to StructuredMatrices having general structure and full dense storage scheme
		 */
		template<
			bool allow_void,
			class MulMonoid,
			typename OutputType, typename InputType1, typename InputType2,
			class Operator, class Monoid,
			typename OutputView = view::Identity< void >, typename InputView1 = view::Identity< void >, typename InputView2 = view::Identity< void >
		>
		RC mxm_generic( StructuredMatrix< OutputType, structures::General, storage::Dense, OutputView, reference_dense > &C,
			const StructuredMatrix< InputType1, structures::General, storage::Dense, InputView1, reference_dense > &A,
			const StructuredMatrix< InputType2, structures::General, storage::Dense, InputView2, reference_dense > &B,
			const Operator &oper,
			const Monoid &monoid,
			const MulMonoid &mulMonoid,
			const typename std::enable_if< !grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value && !
				grb::is_object< InputType2 >::value &&
				grb::is_operator< Operator >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = NULL
		) {
			(void)oper;
			(void)monoid;
			(void)mulMonoid;
			static_assert( allow_void ||
				( !(
					std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value
				) ),
				"grb::mxm_generic: the operator-monoid version of mxm cannot be "
				"used if either of the input matrices is a pattern matrix (of type "
				"void)"
			);

#ifdef _DEBUG
			std::cout << "In grb::internal::mxm_generic (reference_dense, unmasked)\n";
#endif

			// How to handle combinations of different storage schemes?
			// For example for C<dense:full> = A<dense::full> * B<dense:full> we can directly call mxm for Matrix<> containers
			// run-time checks
			const size_t m = grb::nrows( C );
			const size_t n = grb::ncols( C );
			const size_t m_A = grb::nrows( A );
			const size_t k = grb::ncols( A );
			const size_t k_B = grb::nrows( B );
			const size_t n_B = grb::ncols( B );
			// Maybe we can offload these checks to mxm call later in this function

			if( m != m_A || k != k_B || n != n_B ) {
				return MISMATCH;
			}

			const auto A_container = grb::internal::getContainer( A );
			const auto B_container = grb::internal::getContainer( B );
			auto C_container = grb::internal::getContainer( C );

			std::cout << "Multiplying dense matrices.\n";

			RC rc = mxm_generic< true >( C_container, A_container, B_container, oper, monoid, mulMonoid );

			if ( rc == SUCCESS ) {
				// grb::internal::setInitialized( C, true );
			}

			// done
			return rc;
		}

	} // namespace internal

	/**
	 * \internal grb::mxm, semiring version.
	 * Dispatches to internal::mxm_generic
	 */
	template< typename OutputType, typename InputType1, typename InputType2, class Semiring >
	RC mxm( Matrix< OutputType, reference_dense > & C,
		const Matrix< InputType1, reference_dense > & A,
		const Matrix< InputType2, reference_dense > & B,
		const Semiring & ring = Semiring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_semiring< Semiring >::value,
			void >::type * const = NULL ) {

#ifdef _DEBUG
		std::cout << "In grb::mxm (reference_dense, unmasked, semiring)\n";
#endif

		return internal::mxm_generic< true >( C, A, B, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid(), ring.getMultiplicativeMonoid() );
	}

	/**
	 * \internal grb::mxm semiring version for dense Structured Matrices
	 */
	template< typename OutputType, typename InputType1, typename InputType2, class Semiring,
		typename OutputStructure, typename OutputView = view::Identity< void >,
		typename InputStructure1, typename InputView1 = view::Identity< void >,
		typename InputStructure2, typename InputView2 = view::Identity< void > >
	RC mxm( StructuredMatrix< OutputType, OutputStructure, storage::Dense, OutputView, reference_dense > & C,
		const StructuredMatrix< InputType1, InputStructure1, storage::Dense, InputView1, reference_dense > & A,
		const StructuredMatrix< InputType2, InputStructure2, storage::Dense, InputView2, reference_dense > & B,
		const Semiring & ring = Semiring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_semiring< Semiring >::value,
			void >::type * const = NULL ) {
		// TODO: How should we handle multiplication of combinations of Structures and Storage schemes?
		return internal::mxm_generic< true >( C, A, B, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid(), ring.getMultiplicativeMonoid() );
	}

	namespace internal {

		/**
		 * \internal general elementwise matrix application that all eWiseApply variants refer to.
		 */

		template<
			bool allow_void,
			bool left_scalar, bool right_scalar,
			Descriptor descr,
			class MulMonoid,
			typename OutputType, typename InputType1, typename InputType2,
			typename OutputStructure, typename InputStructure1, typename InputStructure2,
			typename OutputStorage, typename InputStorage1, typename InputStorage2,
			typename OutputView, typename InputView1, typename InputView2,
			class Operator
		>
		RC eWiseApply_matrix_generic( StructuredMatrix< OutputType, OutputStructure, OutputStorage, OutputView, reference_dense > *C,
			const StructuredMatrix< InputType1, InputStructure1, InputStorage1, InputView1, reference_dense > *A,
			const InputType1 *alpha,
			const StructuredMatrix< InputType2, InputStructure2, InputStorage2, InputView2, reference_dense > *B,
			const InputType1 *beta,
			const Operator &oper,
			const MulMonoid &mulMonoid,
			const PHASE &phase,
			const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_operator< Operator >::value,
			void >::type * const = NULL
		) {
			(void)C;
			(void)A;
			(void)alpha;
			(void)B;
			(void)beta;
			(void)oper;
			(void)mulMonoid;
			static_assert( allow_void ||
				( !(
				     std::is_same< InputType1, void >::value ||
				     std::is_same< InputType2, void >::value
				) ),
				"grb::internal::eWiseApply_matrix_generic: the non-monoid version of "
				"elementwise mxm can only be used if neither of the input matrices "
				"is a pattern matrix (of type void)" );

#ifdef _DEBUG
			std::cout << "In grb::internal::eWiseApply_matrix_generic\n";
#endif

			// get whether the matrices should be transposed prior to execution
			// constexpr bool trans_left = descr & descriptors::transpose_left;
			// constexpr bool trans_right = descr & descriptors::transpose_right;

			// run-time checks
			// TODO: support left/right_scalar
			// const size_t m = grb::nrows( *C );
			// const size_t n = grb::ncols( *C );
			// const size_t m_A = !trans_left ? grb::nrows( *A ) : grb::ncols( *A );
			// const size_t n_A = !trans_left ? grb::ncols( *A ) : grb::nrows( *A );
			// const size_t m_B = !trans_right ? grb::nrows( *B ) : grb::ncols( *B );
			// const size_t n_B = !trans_right ? grb::ncols( *B ) : grb::nrows( *B );

			// if( m != m_A || m != m_B || n != n_A || n != n_B ) {
			// 	return MISMATCH;
			// }


			// retrieve buffers
			// end buffer retrieval

			// initialisations
			// end initialisations

			// symbolic phase
			if( phase == SYMBOLIC ) {
			}

			// computational phase
			if( phase == NUMERICAL ) {
			}

			// done
			return SUCCESS;
		}

	} // namespace internal

	/**
	 * Computes \f$ C = A . B \f$ for a given monoid.
	 *
	 * \internal Allows pattern matrix inputs.
	 *
	 * \internal Dispatches to internal::eWiseApply_matrix_generic
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		typename OutputStructure, typename InputStructure1, typename InputStructure2,
		typename OutputStorage, typename InputStorage1, typename InputStorage2,
		typename OutputView, typename InputView1, typename InputView2,
		class MulMonoid
	>
	RC eWiseApply( StructuredMatrix< OutputType, OutputStructure, OutputStorage, OutputView, reference_dense > &C,
		const StructuredMatrix< InputType1, InputStructure1, InputStorage1, InputView1, reference_dense > &A,
		const StructuredMatrix< InputType2, InputStructure2, InputStorage2, InputView2, reference_dense > &B,
		const MulMonoid &mulmono,
		const PHASE phase = NUMERICAL,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< MulMonoid >::value,
		void >::type * const = NULL
	) {
		// static checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D1, InputType1 >::value ),
			"grb::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D2, InputType2 >::value ),
			"grb::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D3, OutputType >::value ),
			"grb::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with an output matrix C that does not match the output domain "
			"of the monoid operator"
		);

#ifdef _DEBUG
		std::cout << "In grb::eWiseApply_matrix_generic (reference, monoid)\n";
#endif

		return internal::eWiseApply_matrix_generic< true, false, false, descr >(
			&C, &A, static_cast< const InputType1 * >( nullptr ), &B, static_cast< const InputType2 * >( nullptr ), mulmono.getOperator(), mulmono, phase
		);
	}

	/**
	 * Computes \f$ C = alpha . B \f$ for a given monoid.
	 *
	 * \internal Allows pattern matrix inputs.
	 *
	 * \internal Dispatches to internal::eWiseApply_matrix_generic
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		typename OutputStructure, typename InputStructure2,
		typename OutputStorage, typename InputStorage2,
		typename OutputView, typename InputView2,
		class MulMonoid
	>
	RC eWiseApply( StructuredMatrix< OutputType, OutputStructure, OutputStorage, OutputView, reference_dense > &C,
		const InputType1 &alpha,
		const StructuredMatrix< InputType2, InputStructure2, InputStorage2, InputView2, reference_dense > &B,
		const MulMonoid &mulmono,
		const PHASE phase = NUMERICAL,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< MulMonoid >::value,
		void >::type * const = NULL
	) {
		// static checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D1, InputType1 >::value ),
			"grb::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D2, InputType2 >::value ),
			"grb::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D3, OutputType >::value ),
			"grb::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with an output matrix C that does not match the output domain "
			"of the monoid operator"
		);

#ifdef _DEBUG
		std::cout << "In grb::eWiseApply_matrix_generic (reference_dense, monoid)\n";
#endif

		const StructuredMatrix< InputType1, structures::General, storage::Dense, view::Identity< void >, reference_dense> * no_matrix = nullptr;
		return internal::eWiseApply_matrix_generic< true, true, false, descr >(
			&C,
			no_matrix,
			&alpha,
			&B,
			static_cast< const InputType2 * >( nullptr ),
			mulmono.getOperator(), mulmono, phase
		);
	}

/**
	 * Computes \f$ C = A . beta \f$ for a given monoid.
	 *
	 * \internal Allows pattern matrix inputs.
	 *
	 * \internal Dispatches to internal::eWiseApply_matrix_generic
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		typename OutputStructure, typename InputStructure1,
		typename OutputStorage, typename InputStorage1,
		typename OutputView, typename InputView1,
		class MulMonoid
	>
	RC eWiseApply( StructuredMatrix< OutputType, OutputStructure, OutputStorage, OutputView, reference_dense > &C,
		const StructuredMatrix< InputType1, InputStructure1, InputStorage1, InputView1, reference_dense > &A,
		const InputType2 &beta,
		const MulMonoid &mulmono,
		const PHASE phase = NUMERICAL,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< MulMonoid >::value,
		void >::type * const = NULL
	) {
		// static checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D1, InputType1 >::value ),
			"grb::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D2, InputType2 >::value ),
			"grb::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D3, OutputType >::value ),
			"grb::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with an output matrix C that does not match the output domain "
			"of the monoid operator"
		);

#ifdef _DEBUG
		std::cout << "In grb::eWiseApply_matrix_generic (reference_dense, monoid)\n";
#endif

		const StructuredMatrix< InputType2, structures::General, storage::Dense, view::Identity< void >, reference_dense> * no_matrix = nullptr;
		return internal::eWiseApply_matrix_generic< true, false, true, descr >(
			&C,
			&A,
			static_cast< const InputType1 * >( nullptr ),
			no_matrix,
			&beta,
			mulmono.getOperator(), mulmono, phase
		);
	}

} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_BLAS3''

