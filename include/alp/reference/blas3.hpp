
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

#ifndef _H_ALP_REFERENCE_BLAS3
#define _H_ALP_REFERENCE_BLAS3

#include <type_traits> //for std::enable_if

#include <alp/base/blas3.hpp>
#include <alp/descriptors.hpp>

#include "io.hpp"
#include "matrix.hpp"
#include "vector.hpp"

#ifndef NO_CAST_ASSERT
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
#endif

namespace alp {
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
		RC mxm_generic( Matrix< OutputType, reference > &C,
			const Matrix< InputType1, reference > &A,
			const Matrix< InputType2, reference > &B,
			const Operator &oper,
			const Monoid &monoid,
			const MulMonoid &mulMonoid,
			const typename std::enable_if< !alp::is_object< OutputType >::value &&
				!alp::is_object< InputType1 >::value && !
				alp::is_object< InputType2 >::value &&
				alp::is_operator< Operator >::value &&
				alp::is_monoid< Monoid >::value,
			void >::type * const = NULL
		) {
			(void)oper;
			(void)monoid;
			(void)mulMonoid;
			static_assert( allow_void ||
				( !(
					std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value
				) ),
				"alp::mxm_generic: the operator-monoid version of mxm cannot be "
				"used if either of the input matrices is a pattern matrix (of type "
				"void)"
			);

#ifdef _DEBUG
			std::cout << "In alp::internal::mxm_generic (reference, unmasked)\n";
#endif

			// run-time checks
			const size_t m = nrows( C );
			const size_t n = ncols( C );
			const size_t m_A = nrows( A );
			const size_t k = ncols( A );
			const size_t k_B = nrows( B );
			const size_t n_B = ncols( B );

			if( m != m_A || k != k_B || n != n_B ) {
				return MISMATCH;
			}

			const auto A_raw = getRaw( A );
			const auto B_raw = getRaw( B );
			auto C_raw = getRaw( C );

			std::cout << "Multiplying dense matrices.\n";

			for( size_t row = 0; row < m; ++row ) {
				for( size_t col = 0; col < n; ++col ) {
					C_raw[ row * k + col] = monoid.template getIdentity< OutputType >();
					for( size_t i = 0; i < k; ++ i ) {
						OutputType temp = monoid.template getIdentity< OutputType >();
						(void)alp::apply( temp, A_raw[ row * k + i ], B_raw[ i * n_B + col ], oper );
						(void)alp::foldl( C_raw[ row * k + col], temp, monoid.getOperator() );
					}
				}
			}
			alp::internal::setInitialized( C, true );
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
			typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
			typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
			typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2
		>
		RC mxm_generic( alp::Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
			const alp::Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &A,
			const alp::Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
			const Operator &oper,
			const Monoid &monoid,
			const MulMonoid &mulMonoid,
			const typename std::enable_if< !alp::is_object< OutputType >::value &&
				!alp::is_object< InputType1 >::value && !
				alp::is_object< InputType2 >::value &&
				alp::is_operator< Operator >::value &&
				alp::is_monoid< Monoid >::value,
			void >::type * const = NULL
		) {
			(void)C;
			(void)A;
			(void)B;
			(void)oper;
			(void)monoid;
			(void)mulMonoid;
			// TODO; Implement this as a backup version that works for any structure and storage.
			// Even though the performance does not have to be optimal, we guarantee that any two matrices can be multiplied
			// To provide better performing mxm, one should implement a function with specialized template elements
			// This currently cannot work as we do not have a generic way to access elements in a given Matrix
			return UNSUPPORTED;
		}

		/**
		 * \internal mxm specialized to StructuredMatrices having general structure and full dense storage scheme
		 */
		template<
			bool allow_void,
			class MulMonoid,
			class Operator, class Monoid,
			typename OutputType, typename OutputView = view::Original< void >, typename OutputImfR, typename OutputImfC,
			typename InputType1, typename InputView1 = view::Original< void >, typename InputImfR1, typename InputImfC1,
			typename InputType2, typename InputView2 = view::Original< void >, typename InputImfR2, typename InputImfC2
		>
		RC mxm_generic( alp::Matrix< OutputType, structures::General, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
			const alp::Matrix< InputType1, structures::General, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &A,
			const alp::Matrix< InputType2, structures::General, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
			const Operator &oper,
			const Monoid &monoid,
			const MulMonoid &mulMonoid,
			const typename std::enable_if< !alp::is_object< OutputType >::value &&
				!alp::is_object< InputType1 >::value && !
				alp::is_object< InputType2 >::value &&
				alp::is_operator< Operator >::value &&
				alp::is_monoid< Monoid >::value,
			void >::type * const = NULL
		) {
			(void)oper;
			(void)monoid;
			(void)mulMonoid;
			static_assert( allow_void ||
				( !(
					std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value
				) ),
				"alp::mxm_generic: the operator-monoid version of mxm cannot be "
				"used if either of the input matrices is a pattern matrix (of type "
				"void)"
			);

#ifdef _DEBUG
			std::cout << "In alp::internal::mxm_generic (reference, unmasked)\n";
#endif

			// How to handle combinations of different storage schemes?
			// For example for C<dense:full> = A<dense::full> * B<dense:full> we can directly call mxm for Matrix<> containers
			// run-time checks
			const size_t m = alp::nrows( C );
			const size_t n = alp::ncols( C );
			const size_t m_A = alp::nrows( A );
			const size_t k = alp::ncols( A );
			const size_t k_B = alp::nrows( B );
			const size_t n_B = alp::ncols( B );
			// Maybe we can offload these checks to mxm call later in this function

			if( m != m_A || k != k_B || n != n_B ) {
				return MISMATCH;
			}

			const auto A_container = alp::internal::getContainer( A );
			const auto B_container = alp::internal::getContainer( B );
			auto C_container = alp::internal::getContainer( C );

			std::cout << "Multiplying dense matrices.\n";

			RC rc = mxm_generic< true >( C_container, A_container, B_container, oper, monoid, mulMonoid );

			if ( rc == SUCCESS ) {
				// alp::internal::setInitialized( C, true );
			}

			// done
			return rc;
		}

	} // namespace internal

	/**
	 * Dense Matrix-Matrix multiply between unstructured containers.
	 * Version with semiring parameter.
	 *
	 * @tparam descr      The descriptors under which to perform the computation.
	 * @tparam OutputType The type of elements in the output matrix.
	 * @tparam InputType1 The type of elements in the left-hand side input
	 *                    matrix.
	 * @tparam InputType2 The type of elements in the right-hand side input
	 *                    matrix.
	 * @tparam Semiring   The semiring under which to perform the
	 *                    multiplication.
	 * @tparam Backend    The backend that should perform the computation.
	 *
	 * @returns SUCCESS If the computation completed as intended.
	 * @returns FAILED  If the call was not not preceded by one to
	 *                  #alp::resize( C, A, B ); \em and the current capacity of
	 *                  \a C was insufficient to store the multiplication of \a A
	 *                  and \a B. The contents of \a C shall be undefined (which
	 *                  is why #FAILED is returned instead of #ILLEGAL-- this
	 *                  error has side effects).
	 *
	 * @param[out] C 	The output matrix \f$ C = AB \f$ when the function returns
	 *               	#SUCCESS.
	 * @param[in]  A 	The left-hand side input matrix \f$ A \f$.
	 * @param[in]  B 	The left-hand side input matrix \f$ B \f$.
	 * @param[in] ring  (Optional.) The semiring under which the computation should
	 *                             proceed.
	 * @param phase 	The execution phase.
	 */
	template< Descriptor descr = descriptors::no_operation,
			  typename OutputType, typename InputType1, typename InputType2, 
			  class Semiring >
	RC mxm( internal::Matrix< OutputType, reference > & C,
		const internal::Matrix< InputType1, reference > & A,
		const internal::Matrix< InputType2, reference > & B,
		const Semiring & ring = Semiring(),
		const PHASE &phase = NUMERICAL,
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && alp::is_semiring< Semiring >::value,
			void >::type * const = NULL ) {

#ifdef _DEBUG
		std::cout << "In alp::mxm (reference, unmasked, semiring)\n";
#endif

		return internal::mxm_generic< true >( C, A, B, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid(), ring.getMultiplicativeMonoid() );
	}

	/**
	 * Dense Matrix-Matrix multiply between structured matrices.
	 * Version with semiring parameter.
	 *
	 * @tparam descr      		The descriptors under which to perform the computation.
	 * @tparam OutputStructMatT The structured matrix type of the output matrix.
	 * @tparam InputStructMatT1 The structured matrix type of the the left-hand side input
	 *                    		matrix.
	 * @tparam InputStructMatT2 The structured matrix type of the right-hand side input
	 *                    		matrix.
	 * @tparam Semiring   		The semiring under which to perform the
	 *                    		multiplication.
	 *
	 * @returns SUCCESS  If the computation completed as intended.
	 * @returns MISMATCH Whenever the structures or dimensions of \a A, \a B, and \a C do
 *                       not match. All input data containers are left
 *                       untouched if this exit code is returned; it will be
 *                       as though this call was never made.
	 *
	 * @param[out] C 	The output matrix \f$ C = AB \f$ when the function returns
	 *               	#SUCCESS.
	 * @param[in]  A 	The left-hand side input matrix \f$ A \f$.
	 * @param[in]  B 	The left-hand side input matrix \f$ B \f$.
	 * @param[in] ring  (Optional.) The semiring under which the computation should
	 *                             proceed.
	 * @param phase 	The execution phase.
	 */
	template< Descriptor descr = descriptors::no_operation, 
			  typename OutputStructMatT, 
			  typename InputStructMatT1, 
			  typename InputStructMatT2, 
			  class Semiring>
	RC mxm( OutputStructMatT & C,
		const InputStructMatT1 & A,
		const InputStructMatT2 & B,
		const Semiring & ring = Semiring(),
		const PHASE &phase = NUMERICAL,
		const typename std::enable_if< ! alp::is_object< typename OutputStructMatT::value_type >::value && ! alp::is_object< typename InputStructMatT1::value_type >::value && ! alp::is_object< typename InputStructMatT2::value_type >::value && alp::is_semiring< Semiring >::value,
			void >::type * const = NULL ) {
		(void)phase;
		// TODO: How should we handle multiplication of combinations of Structures and Storage schemes?
		return internal::mxm_generic< true >( C, A, B, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid(), ring.getMultiplicativeMonoid() );
	}

	/**
	 * Dense Matrix-Matrix multiply between structured matrices.
	 * Version with additive monoid and multiplicative operator
	 */
	template< typename OutputStructMatT, 
			  typename InputStructMatT1, 
			  typename InputStructMatT2, 
			  class Operator, class Monoid >
	RC mxm( OutputStructMatT & C,
		const InputStructMatT1 & A,
		const InputStructMatT2 & B,
		const Operator & mulOp,
		const Monoid & addM,
		const PHASE &phase = NUMERICAL,
		const typename std::enable_if< ! alp::is_object< typename OutputStructMatT::value_type >::value && ! alp::is_object< typename InputStructMatT1::value_type >::value && ! alp::is_object< typename InputStructMatT2::value_type >::value &&
		                               alp::is_operator< Operator >::value && alp::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
		(void)phase;
		// TODO: How should we handle multiplication of combinations of Structures and Storage schemes?
		return internal::mxm_generic< true >( C, A, B, mulOp, addM, Monoid() );
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
			typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
			typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
			typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
			class Operator
		>
		RC eWiseApply_matrix_generic(
			alp::Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > *C,
			const alp::Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > *A,
			const InputType1 *alpha,
			const alp::Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > *B,
			const InputType1 *beta,
			const Operator &oper,
			const MulMonoid &mulMonoid,
			const PHASE &phase,
			const typename std::enable_if<
				!alp::is_object< OutputType >::value &&
				!alp::is_object< InputType1 >::value &&
				!alp::is_object< InputType2 >::value &&
				alp::is_operator< Operator >::value,
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
				"alp::internal::eWiseApply_matrix_generic: the non-monoid version of "
				"elementwise mxm can only be used if neither of the input matrices "
				"is a pattern matrix (of type void)" );

#ifdef _DEBUG
			std::cout << "In alp::internal::eWiseApply_matrix_generic\n";
#endif

			// get whether the matrices should be transposed prior to execution
			// constexpr bool trans_left = descr & descriptors::transpose_left;
			// constexpr bool trans_right = descr & descriptors::transpose_right;

			// run-time checks
			// TODO: support left/right_scalar
			// const size_t m = alp::nrows( *C );
			// const size_t n = alp::ncols( *C );
			// const size_t m_A = !trans_left ? alp::nrows( *A ) : alp::ncols( *A );
			// const size_t n_A = !trans_left ? alp::ncols( *A ) : alp::nrows( *A );
			// const size_t m_B = !trans_right ? alp::nrows( *B ) : alp::ncols( *B );
			// const size_t n_B = !trans_right ? alp::ncols( *B ) : alp::nrows( *B );

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
	 * @brief Computes \f$ C = A . B \f$ for a given monoid.
	 * 
	 * @tparam descr      		The descriptor to be used (descriptors::no_operation
	 *                    		if left unspecified).
	 * @tparam OutputType 		The element type of the output matrix
	 * @tparam InputType1 		The element type of the left-hand side matrix
	 * @tparam InputType2 		The element type of the right-hand side matrix
	 * @tparam OutputStructure 	The structure of the output matrix
	 * @tparam InputStructure1 	The structure of the left-hand side matrix
	 * @tparam InputStructure2  The structure of the right-hand matrix
	 * @tparam OutputView 		The type of view of the output matrix
	 * @tparam InputView1 		The type of view of the left-hand matrix
	 * @tparam InputView2 		The type of view of the right-hand matrix
	 * @tparam MulMonoid 		The type of monoid used for this element-wise operation
	 * 
	 * @param C 		The output structured matrix
	 * @param A 		The left-hand side structured matrix
	 * @param B 		The right-hand side structured matrix
	 * @param mulmono 	The monoid used in the element-wise operation
	 * @param phase 	The execution phase 
	 * 
	 * @return alp::MISMATCH Whenever the structures or dimensions of \a A, \a B, and \a C do
	 *                       not match. All input data containers are left
	 *                       untouched if this exit code is returned; it will be
	 *                       as though this call was never made.
	 * @return alp::SUCCESS  On successful completion of this call.
 	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class MulMonoid >
	RC eWiseApply( Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &A,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
		const MulMonoid &mulmono,
		const PHASE phase = NUMERICAL,
		const typename std::enable_if< !alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_monoid< MulMonoid >::value,
		void >::type * const = NULL
	) {
		// static checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D1, InputType1 >::value ),
			"alp::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D2, InputType2 >::value ),
			"alp::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D3, OutputType >::value ),
			"alp::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with an output matrix C that does not match the output domain "
			"of the monoid operator"
		);

#ifdef _DEBUG
		std::cout << "In alp::eWiseApply_matrix_generic (reference, monoid)\n";
#endif

		return internal::eWiseApply_matrix_generic< true, false, false, descr >(
			&C, &A, static_cast< const InputType1 * >( nullptr ), &B, static_cast< const InputType2 * >( nullptr ), mulmono.getOperator(), mulmono, phase
		);
	}


	/**
	 * Computes \f$ C = alpha . B \f$ for a given monoid.
	 *
	 * Case where \a A is a scalar.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class MulMonoid >
	RC eWiseApply( Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const InputType1 &alpha,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
		const MulMonoid &mulmono,
		const PHASE phase = NUMERICAL,
		const typename std::enable_if< !alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_monoid< MulMonoid >::value,
		void >::type * const = NULL
	) {
		// static checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D1, InputType1 >::value ),
			"alp::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D2, InputType2 >::value ),
			"alp::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D3, OutputType >::value ),
			"alp::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with an output matrix C that does not match the output domain "
			"of the monoid operator"
		);

#ifdef _DEBUG
		std::cout << "In alp::eWiseApply_matrix_generic (reference, monoid)\n";
#endif

		const Matrix< InputType1, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > * no_matrix = nullptr;
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
	 * Case where \a B is a scalar.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2,
		class MulMonoid >
	RC eWiseApply( Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &A,
		const InputType2 &beta,
		const MulMonoid &mulmono,
		const PHASE phase = NUMERICAL,
		const typename std::enable_if< !alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_monoid< MulMonoid >::value,
		void >::type * const = NULL
	) {
		// static checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D1, InputType1 >::value ),
			"alp::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D2, InputType2 >::value ),
			"alp::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D3, OutputType >::value ),
			"alp::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with an output matrix C that does not match the output domain "
			"of the monoid operator"
		);

#ifdef _DEBUG
		std::cout << "In alp::eWiseApply_matrix_generic (reference, monoid)\n";
#endif

		const Matrix< InputType2, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > * no_matrix = nullptr;
		return internal::eWiseApply_matrix_generic< true, false, true, descr >(
			&C,
			&A,
			static_cast< const InputType1 * >( nullptr ),
			no_matrix,
			&beta,
			mulmono.getOperator(), mulmono, phase
		);
	}

	/**
	 * Calculates the element-wise addition of two structured matrices, \f$ C = A + B \f$,
	 * under the selected semiring.
	 *
	 * @tparam descr      		The descriptor to be used (descriptors::no_operation
	 *                    		if left unspecified).
	 * @tparam OutputType 		The element type of the output matrix
	 * @tparam InputType1 		The element type of the left-hand side matrix
	 * @tparam InputType2 		The element type of the right-hand side matrix
	 * @tparam OutputStructure 	The structure of the output matrix
	 * @tparam InputStructure1 	The structure of the left-hand side matrix
	 * @tparam InputStructure2  The structure of the right-hand matrix
	 * @tparam OutputView 		The type of view of the output matrix
	 * @tparam InputView1 		The type of view of the left-hand matrix
	 * @tparam InputView2 		The type of view of the right-hand matrix
	 * @tparam Ring       		The semiring type to perform the element-wise addition
	 *                    		on.
	 *
	 * @param[out]  C  The output vector of type \a OutputType. This may be a
	 *                 sparse vector.
	 * @param[in]   x  The left-hand input vector of type \a InputType1. This may
	 *                 be a sparse vector.
	 * @param[in]   y  The right-hand input vector of type \a InputType2. This may
	 *                 be a sparse vector.
	 * @param[in] ring The generalized semiring under which to perform this
	 *                 element-wise operation.
	 *
	 * @return alp::MISMATCH Whenever the dimensions of \a x, \a y, and \a z do
	 *                       not match. All input data containers are left
	 *                       untouched; it will be as though this call was never
	 *                       made.
	 * @return alp::SUCCESS  On successful completion of this call.
	 *
	 * \note Invalid descriptors will be ignored.
	 * 
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call takes \f$ \Theta(n) \f$ work, where \f$ n \f$ equals the
	//  *         size of the vectors \a x, \a y, and \a z. The constant factor
	//  *         depends on the cost of evaluating the addition operator. A good
	//  *         implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the the additive operator used
	//  *         allow for this.
	//  *
	//  *      -# This call will not result in additional dynamic memory allocations.
	//  *         No system calls will be made.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ n( \mathit{sizeof}(
	//  *             \mathit{InputType1} +
	//  *             \mathit{InputType2} +
	//  *             \mathit{OutputType}
	//  *           ) + \mathcal{O}(1) \f$
	//  *         bytes of data movement. A good implementation will stream \a x or
	//  *         \a y into \a z to apply the additive operator in-place, whenever
	//  *         the input domains, the output domain, and the operator used allow
	//  *         for this.
	//  * \endparblock
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		typename Ring>
	RC eWiseAdd( Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &A,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && alp::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D4, OutputType >::value ), "alp::eWiseAdd",
			"called with an output vector with element type that does not match the "
			"fourth domain of the given semiring" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D3, InputType1 >::value ), "alp::eWiseAdd",
			"called with a left-hand side input vector with element type that does not "
			"match the third domain of the given semiring" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D4, OutputType >::value ), "alp::eWiseAdd",
			"called with a right-hand side input vector with element type that does "
			"not match the fourth domain of the given semiring" );
	#ifdef _DEBUG
		std::cout << "eWiseAdd (reference, StrMat <- StrMat + StrMat) dispatches to eWiseApply( reference, StrMat <- StrMat . StrMat ) using additive monoid\n";
	#endif
		return eWiseApply< descr >( C, A, B, ring.getAdditiveMonoid() );
	}

	/**
	 * Calculates the element-wise multiplication of two matrices,
	 *     \f$ C = C + A .* B \f$,
	 * under a given semiring.
	 *
	 * @tparam descr      The descriptor to be used (descriptors::no_operation
	 *                    if left unspecified).
	 * @tparam Ring       The semiring type to perform the element-wise multiply
	 *                    on.
	 * @tparam InputType1 The left-hand side input type to the multiplicative
	 *                    operator of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the multiplicative
	 *                    operator of the \a ring.
	 * @tparam OutputType The the result type of the multiplicative operator of
	 *                    the \a ring.
	 * @tparam InputStructure1  The structure of the left-hand side input to
	 *                          the multiplicative operator of the \a ring.
	 * @tparam InputStructure2  The structure of the right-hand side input
	 *                          to the multiplicative operator of the \a ring.
	 * @tparam OutputStructure1 The structure of the output to the
	 *                          multiplicative operator of the \a ring.
	 * @tparam InputView1       The view type applied to the left-hand side
	 *                          input to the multiplicative operator
	 *                          of the \a ring.
	 * @tparam InputView2       The view type applied to the right-hand side
	 *                          input to the multiplicative operator
	 *                          of the \a ring.
	 * @tparam OutputView1      The view type applied to the output to the
	 *                          multiplicative operator of the \a ring.
	 *
	 * @param[out]  z  The output vector of type \a OutputType.
	 * @param[in]   x  The left-hand input vector of type \a InputType1.
	 * @param[in]   y  The right-hand input vector of type \a InputType2.
	 * @param[in] ring The generalized semiring under which to perform this
	 *                 element-wise multiplication.
	 *
	 * @return alp::MISMATCH Whenever the dimensions of \a x, \a y, and \a z do
	 *                       not match. All input data containers are left
	 *                       untouched if this exit code is returned; it will be
	 *                       as though this call was never made.
	 * @return alp::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * alp::descriptors::no_operation, alp::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If alp::descriptors::no_casting is specified, then 1) the first domain of
	 * \a ring must match \a InputType1, 2) the second domain of \a ring must match
	 * \a InputType2, 3) the third domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call takes \f$ \Theta(n) \f$ work, where \f$ n \f$ equals the
	//  *         size of the vectors \a x, \a y, and \a z. The constant factor
	//  *         depends on the cost of evaluating the multiplication operator. A
	//  *         good implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the multiplicative operator used
	//  *         allow for this.
	//  *
	//  *      -# This call will not result in additional dynamic memory allocations.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most \f$ n( \mathit{sizeof}(\mathit{D1}) +
	//  *         \mathit{sizeof}(\mathit{D2}) + \mathit{sizeof}(\mathit{D3})) +
	//  *         \mathcal{O}(1) \f$ bytes of data movement. A good implementation
	//  *         will stream \a x or \a y into \a z to apply the multiplication
	//  *         operator in-place, whenever the input domains, the output domain,
	//  *         and the operator used allow for this.
	//  * \endparblock
	 *
	 * \warning When given sparse vectors, the zero now annihilates instead of
	 *       acting as an identity. Thus the eWiseMul cannot simply map to an
	 *       eWiseApply of the multiplicative operator.
	 *
	 * @see This is a specialised form of eWiseMulAdd.
	 */
	template< Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2 >
	RC eWiseMul( Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &A,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && alp::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D1, InputType1 >::value ), "alp::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D2, InputType2 >::value ), "alp::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D3, OutputType >::value ), "alp::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
	#ifdef _DEBUG
		std::cout << "eWiseMul (reference, vector <- vector x vector) dispatches to eWiseMulAdd (vector <- vector x vector + 0)\n";
	#endif
		// return eWiseMulAdd< descr >( z, x, y, ring.template getZero< Ring::D4 >(), ring );
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * eWiseMul, version where A is a scalar.
	 */
	template< Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, 
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2
	>
	RC eWiseMul( Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Scalar< InputType1, InputStructure1, reference > &alpha,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && alp::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		(void)C;
		(void)alpha;
		(void)B;
		(void)ring;
		// static sanity checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D1, InputType1 >::value ), "alp::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D2, InputType2 >::value ), "alp::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D3, OutputType >::value ), "alp::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
	#ifdef _DEBUG
		std::cout << "eWiseMul (reference, vector <- vector x vector) dispatches to eWiseMulAdd (vector <- vector x vector + 0)\n";
	#endif
		// return eWiseMulAdd< descr >( z, x, y, ring.template getZero< Ring::D4 >(), ring );
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * @brief  Outer product of two vectors. The result matrix \a A will contain \f$ uv^T \f$.
	 * 
	 * @tparam descr      	The descriptor to be used (descriptors::no_operation
	 *                    	if left unspecified).
	 * @tparam InputType1 	The value type of the left-hand vector.
	 * @tparam InputType2 	The value type of the right-hand scalar.
	 * @tparam OutputType 	The value type of the ouput vector.
	 * @tparam InputStructure1  The Structure type applied to the left-hand vector.
	 * @tparam InputStructure2  The Structure type applied to the right-hand vector.
	 * @tparam OutputStructure1 The Structure type applied to the output vector.
	 * @tparam InputView1       The view type applied to the left-hand vector.
	 * @tparam InputView2       The view type applied to the right-hand vector.
	 * @tparam OutputView1      The view type applied to the output vector.
	 * @tparam Operator         The operator type used for this element-wise operation.
	 *  
	 * @param A      The output structured matrix 
	 * @param u      The left-hand side vector view
	 * @param v 	 The right-hand side vector view
	 * @param mul 	 The operator
	 * @param phase  The execution phase 
	 * 
	 * @return alp::MISMATCH Whenever the structures or dimensions of \a A, \a u, and \a v do
	 *                       not match. All input data containers are left
	 *                       untouched if this exit code is returned; it will be
	 *                       as though this call was never made.
	 * @return alp::SUCCESS  On successful completion of this call.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1,
		typename InputType2, typename InputStructure2, typename InputView2,
		class Operator >
	RC outer( Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, reference > & u,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, reference > & v,
		const Operator & mul = Operator(),
		const typename std::enable_if< alp::is_operator< Operator >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && ! alp::is_object< OutputType >::value,
			void >::type * const = NULL ) {
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D1, InputType1 >::value ), "alp::outerProduct",
			"called with a prefactor vector that does not match the first domain "
			"of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D2, InputType2 >::value ), "alp::outerProduct",
			"called with a postfactor vector that does not match the first domain "
			"of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D3, OutputType >::value ), "alp::outerProduct",
			"called with an output matrix that does not match the output domain of "
			"the given multiplication operator" );

		const size_t nrows = getLength( u );
		const size_t ncols = getLength( v );

		if( nrows != alp::nrows( A ) ) {
			return MISMATCH;
		}

		if( ncols != alp::ncols( A ) ) {
			return MISMATCH;
		}

		alp::Matrix< InputType1, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > u_matrix( nrows, 1 );
		alp::Matrix< InputType2, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > v_matrix( 1, ncols );

		// auto u_converter = alp::utils::makeVectorToMatrixConverter< InputType1 >( u, []( const size_t & ind, const InputType1 & val ) {
		// 	return std::make_pair( std::make_pair( ind, 0 ), val );
		// } );

		// alp::buildMatrixUnique( u_matrix, u_converter.begin(), u_converter.end(), PARALLEL );

		// auto v_converter = alp::utils::makeVectorToMatrixConverter< InputType2 >( v, []( const size_t & ind, const InputType2 & val ) {
		// 	return std::make_pair( std::make_pair( 0, ind ), val );
		// } );
		// alp::buildMatrixUnique( v_matrix, v_converter.begin(), v_converter.end(), PARALLEL );

		alp::Monoid< alp::operators::left_assign< OutputType >, alp::identities::zero > mono;

		return alp::mxm( A, u_matrix, v_matrix, mul, mono );
	}

	/**
	 * Returns a view over the general rank-1 matrix computed with the outer product.
	 * This avoids creating the resulting container. The elements are calculated lazily on access.
	 *
	 * @tparam descr      	    The descriptor to be used (descriptors::no_operation
	 *                    	    if left unspecified).
	 * @tparam InputType1 	    The value type of the left-hand vector.
	 * @tparam InputType2 	    The value type of the right-hand scalar.
	 * @tparam InputStructure1  The Structure type applied to the left-hand vector.
	 * @tparam InputStructure2  The Structure type applied to the right-hand vector.
	 * @tparam InputView1       The view type applied to the left-hand vector.
	 * @tparam InputView2       The view type applied to the right-hand vector.
	 * @tparam Operator         The operator type used for this element-wise operation.
	 *
	 * @param x      The left-hand side vector view
	 * @param y 	 The right-hand side vector view
	 * @param mul 	 The operator
	 * @param phase  The execution phase
	 *
	 * @return Matrix view over a lambda function defined in this function.
	 *         The data type of the matrix equals to the return type of the provided operator.
	 *         The structure of this matrix is General.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType1, typename InputStructure1, typename InputView1,
		typename InputType2, typename InputStructure2, typename InputView2,
		class Operator
		>
	Matrix< typename Operator::D3, structures::General, Density::Dense,
		view::Functor< std::function< void( InputType1 &, const size_t &, const size_t & ) > >,
		imf::Id, imf::Id,
		reference >
	outer(
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, reference > &x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, reference > &y,
		const Operator &mul = Operator(),
		const typename std::enable_if< alp::is_operator< Operator >::value &&
			! alp::is_object< InputType1 >::value &&
			! alp::is_object< InputType2 >::value,
			void >::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D1, InputType1 >::value ), "alp::outerProduct",
			"called with a prefactor vector that does not match the first domain "
			"of the given multiplication operator" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D2, InputType2 >::value ), "alp::outerProduct",
			"called with a postfactor vector that does not match the first domain "
			"of the given multiplication operator" );

		std::function< void( typename Operator::D3 &, const size_t &, const size_t & ) > lambda =
			[ &x, &y, &mul ]( typename Operator::D3 &result, const size_t &i, const size_t &j ) {
				//set( ret, alp::identities::zero );
				apply( result, x[ i ], y[ j ], mul );
			};
		
		return Matrix<
			typename Operator::D3,
			structures::General,
			Density::Dense,
			view::Functor< std::function< void( typename Operator::D3 &, const size_t &, const size_t & ) > >,
			imf::Id, imf::Id,
			reference
			>(
				lambda,
				imf::Id( getLength( x ) ),
				imf::Id( getLength( y ) )
			);

	}

	/**
	 * Returns a view over the general rank-1 matrix computed with the outer product.
	 * Version for the case when input vectors are the same vector,
	 * which results in a symmetric matrix.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView,
		class Operator
		>
	Matrix< typename Operator::D3, structures::Symmetric, Density::Dense,
		view::Functor< std::function< void( typename Operator::D3 &, const size_t &, const size_t & ) > >,
		imf::Id, imf::Id,
		reference >
	outer(
		const Vector< InputType, InputStructure, Density::Dense, InputView, reference > &x,
		const Operator &mul = Operator(),
		const typename std::enable_if< alp::is_operator< Operator >::value &&
			! alp::is_object< InputType >::value,
			void >::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D1, InputType >::value ), "alp::outerProduct",
			"called with a prefactor vector that does not match the first domain "
			"of the given multiplication operator" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D2, InputType >::value ), "alp::outerProduct",
			"called with a prefactor vector that does not match the first domain "
			"of the given multiplication operator" );

		std::function< void( typename Operator::D3 &, const size_t &, const size_t & ) > lambda =
			[ &x, &mul ]( typename Operator::D3 &result, const size_t &i, const size_t &j ) {
				//set( ret, alp::identities::zero );
				apply( result, x[ i ], x[ j ], mul );
			};
		
		return Matrix<
			typename Operator::D3,
			structures::Symmetric,
			Density::Dense,
			view::Functor< std::function< void( typename Operator::D3 &, const size_t &, const size_t & ) > >,
			imf::Id, imf::Id,
			reference
			>(
				lambda,
				imf::Id( getLength( x ) )
			);

	}

	/**
	 * Sets all elements of the output matrix to the values of the input matrix. Unmasked version.
	 * C = A
	 * 
	 * @tparam descr
	 * @tparam OutputType      Data type of the output matrix C
	 * @tparam OutputStructure Structure of the matrix C
	 * @tparam OutputView      View type applied to the matrix C
	 * @tparam InputType       Data type of the scalar a
	 *
	 * @param C    Matrix whose values are to be set
	 * @param A    The input matrix
	 *
	 * @return RC  SUCCESS on the successful execution of the set
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC >
	RC set( Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &A ) noexcept {
		(void)C;
		(void)A;
		static_assert( ! std::is_same< OutputType, void >::value,
			"alp::set (set to value): cannot have a pattern "
			"matrix as output" );
#ifdef _DEBUG
		std::cout << "Called alp::set (matrix-to-value, reference)" << std::endl;
#endif
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, OutputType >::value ), "alp::set", "called with non-matching value types" );

		assert( false ); // "Needs an implementation."
		return SUCCESS;
	}

	/**
	 * Sets all elements of the given matrix to the value of the given scalar. Unmasked version.
	 * C = val
	 * 
	 * @tparam descr
	 * @tparam OutputType      Data type of the output matrix C
	 * @tparam OutputStructure Structure of the matrix C
	 * @tparam OutputView      View type applied to the matrix C
	 * @tparam InputType       Data type of the scalar a
	 *
	 * @param C    Matrix whose values are to be set
	 * @param val  The value to set the elements of the matrix C
	 *
	 * @return RC  SUCCESS on the successful execution of the set
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType, typename InputStructure >
	RC set( Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Scalar< InputType, InputStructure, reference > &val ) noexcept {
		(void)C;
		(void)val;
		static_assert( ! std::is_same< OutputType, void >::value,
			"alp::set (set to value): cannot have a pattern "
			"matrix as output" );
#ifdef _DEBUG
		std::cout << "Called alp::set (matrix-to-value, reference)" << std::endl;
#endif
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, OutputType >::value ), "alp::set", "called with non-matching value types" );

		assert( false ); // "Needs an implementation."
		return SUCCESS;
	}

} // end namespace ``alp''

#undef NO_CAST_ASSERT

#endif // end ``_H_ALP_REFERENCE_BLAS3''

