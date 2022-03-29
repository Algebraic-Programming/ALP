
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
#include <graphblas/descriptors.hpp>

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

			const auto A_raw = grb::internal::getRaw( A );
			const auto B_raw = grb::internal::getRaw( B );
			auto C_raw = grb::internal::getRaw( C );

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
		RC mxm_generic( StructuredMatrix< OutputType, OutputStructure, Density::Dense, OutputView, reference_dense > &C,
			const StructuredMatrix< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > &A,
			const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > &B,
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
			(void)C;
			(void)A;
			(void)B;
			(void)oper;
			(void)monoid;
			(void)mulMonoid;
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
			typename OutputView = view::Original< void >, typename InputView1 = view::Original< void >, typename InputView2 = view::Original< void >
		>
		RC mxm_generic( StructuredMatrix< OutputType, structures::General, Density::Dense, OutputView, reference_dense > &C,
			const StructuredMatrix< InputType1, structures::General, Density::Dense, InputView1, reference_dense > &A,
			const StructuredMatrix< InputType2, structures::General, Density::Dense, InputView2, reference_dense > &B,
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
	 *                  #grb::resize( C, A, B ); \em and the current capacity of
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
	RC mxm( Matrix< OutputType, reference_dense > & C,
		const Matrix< InputType1, reference_dense > & A,
		const Matrix< InputType2, reference_dense > & B,
		const Semiring & ring = Semiring(),
		const PHASE &phase = NUMERICAL,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_semiring< Semiring >::value,
			void >::type * const = NULL ) {

#ifdef _DEBUG
		std::cout << "In grb::mxm (reference_dense, unmasked, semiring)\n";
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
		const typename std::enable_if< ! grb::is_object< typename OutputStructMatT::value_type >::value && ! grb::is_object< typename InputStructMatT1::value_type >::value && ! grb::is_object< typename InputStructMatT2::value_type >::value && grb::is_semiring< Semiring >::value,
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
		const typename std::enable_if< ! grb::is_object< typename OutputStructMatT::value_type >::value && ! grb::is_object< typename InputStructMatT1::value_type >::value && ! grb::is_object< typename InputStructMatT2::value_type >::value &&
		                               grb::is_operator< Operator >::value && grb::is_monoid< Monoid >::value,
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
			typename OutputType, typename InputType1, typename InputType2,
			typename OutputStructure, typename InputStructure1, typename InputStructure2,
			  
			typename OutputView, typename InputView1, typename InputView2,
			class Operator
		>
		RC eWiseApply_matrix_generic( StructuredMatrix< OutputType, OutputStructure, Density::Dense, OutputView, reference_dense > *C,
			const StructuredMatrix< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > *A,
			const InputType1 *alpha,
			const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > *B,
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
	 * @return grb::MISMATCH Whenever the structures or dimensions of \a A, \a B, and \a C do
	 *                       not match. All input data containers are left
	 *                       untouched if this exit code is returned; it will be
	 *                       as though this call was never made.
	 * @return grb::SUCCESS  On successful completion of this call.
 	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		typename OutputStructure, typename InputStructure1, typename InputStructure2,
		  
		typename OutputView, typename InputView1, typename InputView2,
		class MulMonoid >
	RC eWiseApply( StructuredMatrix< OutputType, OutputStructure, Density::Dense, OutputView, reference_dense > &C,
		const StructuredMatrix< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > &A,
		const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > &B,
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
			"grb::eWiseApply (reference_dense, matrix <- matrix x matrix, monoid)",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D2, InputType2 >::value ),
			"grb::eWiseApply (reference_dense, matrix <- matrix x matrix, monoid)",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D3, OutputType >::value ),
			"grb::eWiseApply (reference_dense, matrix <- matrix x matrix, monoid)",
			"called with an output matrix C that does not match the output domain "
			"of the monoid operator"
		);

#ifdef _DEBUG
		std::cout << "In grb::eWiseApply_matrix_generic (reference_dense, monoid)\n";
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
		typename OutputType, typename InputType1, typename InputType2,
		typename OutputStructure, typename InputStructure2,
		 
		typename OutputView, typename InputView2,
		class MulMonoid >
	RC eWiseApply( StructuredMatrix< OutputType, OutputStructure, Density::Dense, OutputView, reference_dense > &C,
		const InputType1 &alpha,
		const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > &B,
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

		const StructuredMatrix< InputType1, structures::General, Density::Dense, view::Original< void >, reference_dense > * no_matrix = nullptr;
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
		typename OutputType, typename InputType1, typename InputType2,
		typename OutputStructure, typename InputStructure1,
		 
		typename OutputView, typename InputView1,
		class MulMonoid >
	RC eWiseApply( StructuredMatrix< OutputType, OutputStructure, Density::Dense, OutputView, reference_dense > &C,
		const StructuredMatrix< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > &A,
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

		const StructuredMatrix< InputType2, structures::General, Density::Dense, view::Original< void >, reference_dense > * no_matrix = nullptr;
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
	 * @return grb::MISMATCH Whenever the dimensions of \a x, \a y, and \a z do
	 *                       not match. All input data containers are left
	 *                       untouched; it will be as though this call was never
	 *                       made.
	 * @return grb::SUCCESS  On successful completion of this call.
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
		typename OutputType, typename InputType1, typename InputType2,
		typename OutputStructure, typename InputStructure1, typename InputStructure2,
		  
		typename OutputView, typename InputView1, typename InputView2,
		typename Ring>
	RC eWiseAdd( StructuredMatrix< OutputType, OutputStructure, Density::Dense, OutputView, reference_dense > &C,
		const StructuredMatrix< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > &A,
		const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > &B,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D4, OutputType >::value ), "grb::eWiseAdd",
			"called with an output vector with element type that does not match the "
			"fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D3, InputType1 >::value ), "grb::eWiseAdd",
			"called with a left-hand side input vector with element type that does not "
			"match the third domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D4, OutputType >::value ), "grb::eWiseAdd",
			"called with a right-hand side input vector with element type that does "
			"not match the fourth domain of the given semiring" );
	#ifdef _DEBUG
		std::cout << "eWiseAdd (reference_dense, StrMat <- StrMat + StrMat) dispatches to eWiseApply( reference_dense, StrMat <- StrMat . StrMat ) using additive monoid\n";
	#endif
		return eWiseApply< descr >( C, A, B, ring.getAdditiveMonoid() );
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
	 * @return grb::MISMATCH Whenever the structures or dimensions of \a A, \a u, and \a v do
	 *                       not match. All input data containers are left
	 *                       untouched if this exit code is returned; it will be
	 *                       as though this call was never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		typename OutputStructure, typename InputStructure1, typename InputStructure2,
		  
		typename OutputView, typename InputView1, typename InputView2, class Operator >
	RC outer( StructuredMatrix< OutputType, OutputStructure, Density::Dense, OutputView, reference_dense > & A,
		const VectorView< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > & u,
		const VectorView< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > & v,
		const Operator & mul = Operator(),
		const typename std::enable_if< grb::is_operator< Operator >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! grb::is_object< OutputType >::value,
			void >::type * const = NULL ) {
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D1, InputType1 >::value ), "grb::outerProduct",
			"called with a prefactor vector that does not match the first domain "
			"of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D2, InputType2 >::value ), "grb::outerProduct",
			"called with a postfactor vector that does not match the first domain "
			"of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D3, OutputType >::value ), "grb::outerProduct",
			"called with an output matrix that does not match the output domain of "
			"the given multiplication operator" );

		const size_t nrows = getLength( u );
		const size_t ncols = getLength( v );

		if( nrows != grb::nrows( A ) ) {
			return MISMATCH;
		}

		if( ncols != grb::ncols( A ) ) {
			return MISMATCH;
		}

		grb::StructuredMatrix< InputType1, structures::General, Density::Dense, view::Original< void >, reference_dense > u_matrix( nrows, 1 );
		grb::StructuredMatrix< InputType2, structures::General, Density::Dense, view::Original< void >, reference_dense > v_matrix( 1, ncols );

		// auto u_converter = grb::utils::makeVectorToMatrixConverter< InputType1 >( u, []( const size_t & ind, const InputType1 & val ) {
		// 	return std::make_pair( std::make_pair( ind, 0 ), val );
		// } );

		// grb::buildMatrixUnique( u_matrix, u_converter.begin(), u_converter.end(), PARALLEL );

		// auto v_converter = grb::utils::makeVectorToMatrixConverter< InputType2 >( v, []( const size_t & ind, const InputType2 & val ) {
		// 	return std::make_pair( std::make_pair( 0, ind ), val );
		// } );
		// grb::buildMatrixUnique( v_matrix, v_converter.begin(), v_converter.end(), PARALLEL );

		grb::Monoid< grb::operators::left_assign< OutputType >, grb::identities::zero > mono;

		return grb::mxm( A, u_matrix, v_matrix, mul, mono );
	}

} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_BLAS3''

