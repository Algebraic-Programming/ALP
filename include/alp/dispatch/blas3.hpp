
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

#ifndef _H_ALP_DISPATCH_BLAS3
#define _H_ALP_DISPATCH_BLAS3

#include <algorithm>   // for std::min/max
#include <type_traits> // for std::enable_if
#include <cblas.h> // for gemm

#include <alp/base/blas3.hpp>
#include <alp/descriptors.hpp>
#include <alp/structures.hpp>
#include <alp/blas0.hpp>
#include <graphblas/utils/iscomplex.hpp> // use from grb

#include "io.hpp"
#include "matrix.hpp"
#include "vector.hpp"

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
	namespace internal {

		/**
		 * \internal generic band mxm implementation - forward declaration.
		 */
		template<
			size_t BandPos1, size_t BandPos2,
			class MulMonoid,
			typename OutputType, typename InputType1, typename InputType2,
			class Operator, class Monoid,
			typename OutputStructure, typename OutputView, 
			typename OutputImfR, typename OutputImfC,
			typename InputStructure1, typename InputView1, 
			typename InputImfR1, typename InputImfC1,
			typename InputStructure2, typename InputView2, 
			typename InputImfR2, typename InputImfC2
		>
		typename std::enable_if<
			( BandPos1 < std::tuple_size< typename InputStructure1::band_intervals >::value ) &&
			( BandPos2 < std::tuple_size< typename InputStructure2::band_intervals >::value ),
		RC >::type mxm_band_generic( 
			alp::Matrix< OutputType, OutputStructure, 
			Density::Dense, OutputView, OutputImfR, OutputImfC, dispatch > &C,
			const alp::Matrix< InputType1, InputStructure1, 
			Density::Dense, InputView1, InputImfR1, InputImfC1, dispatch > &A,
			const alp::Matrix< InputType2, InputStructure2, 
			Density::Dense, InputView2, InputImfR2, InputImfC2, dispatch > &B,
			const Operator &oper,
			const Monoid &monoid,
			const MulMonoid &mulMonoid
		);

		/**
		 * \internal generic band mxm implementation.
		 * Recursively enumerating the cartesian product of non-zero bands 
		 * (Base case).
		 */
		template<
			size_t BandPos1, size_t BandPos2,
			class MulMonoid,
			typename OutputType, typename InputType1, typename InputType2,
			class Operator, class Monoid,
			typename OutputStructure, typename OutputView, 
			typename OutputImfR, typename OutputImfC,
			typename InputStructure1, typename InputView1, 
			typename InputImfR1, typename InputImfC1,
			typename InputStructure2, typename InputView2, 
			typename InputImfR2, typename InputImfC2
		>
		typename std::enable_if<
			( BandPos1 == std::tuple_size< typename InputStructure1::band_intervals >::value ),
		RC >::type mxm_band_generic( 
			alp::Matrix< OutputType, OutputStructure, 
			Density::Dense, OutputView, OutputImfR, OutputImfC, dispatch > &C,
			const alp::Matrix< InputType1, InputStructure1, 
			Density::Dense, InputView1, InputImfR1, InputImfC1, dispatch > &A,
			const alp::Matrix< InputType2, InputStructure2, 
			Density::Dense, InputView2, InputImfR2, InputImfC2, dispatch > &B,
			const Operator &oper,
			const Monoid &monoid,
			const MulMonoid &mulMonoid
		) {
			(void)C;
			(void)A;
			(void)B;
			(void)oper;
			(void)monoid;
			(void)mulMonoid;

			return SUCCESS;
		}

		/**
		 * \internal generic band mxm implementation. 
		 * Recursively enumerating the cartesian product of non-zero bands.
		 * Move to next non-zero band of A.
		 */
		template<
			size_t BandPos1, size_t BandPos2,
			class MulMonoid,
			typename OutputType, typename InputType1, typename InputType2,
			class Operator, class Monoid,
			typename OutputStructure, typename OutputView, 
			typename OutputImfR, typename OutputImfC,
			typename InputStructure1, typename InputView1, 
			typename InputImfR1, typename InputImfC1,
			typename InputStructure2, typename InputView2, 
			typename InputImfR2, typename InputImfC2
		>
		typename std::enable_if<
			( BandPos1 < std::tuple_size< typename InputStructure1::band_intervals >::value ) &&
			( BandPos2 == std::tuple_size< typename InputStructure2::band_intervals >::value ),
		RC >::type mxm_band_generic( 
			alp::Matrix< OutputType, OutputStructure, 
			Density::Dense, OutputView, OutputImfR, OutputImfC, dispatch > &C,
			const alp::Matrix< InputType1, InputStructure1, 
			Density::Dense, InputView1, InputImfR1, InputImfC1, dispatch > &A,
			const alp::Matrix< InputType2, InputStructure2, 
			Density::Dense, InputView2, InputImfR2, InputImfC2, dispatch > &B,
			const Operator &oper,
			const Monoid &monoid,
			const MulMonoid &mulMonoid
		) {
			return mxm_band_generic< BandPos1 + 1, 0 >( C, A, B, oper, monoid, mulMonoid );
		}

		/**
		 * \internal generic band mxm implementation. 
		 * Recursively enumerating the cartesian product of non-zero bands.
		 * Compute and move to next non-zero band of B.
		 */
		template<
			size_t BandPos1, size_t BandPos2,
			class MulMonoid,
			typename OutputType, typename InputType1, typename InputType2,
			class Operator, class Monoid,
			typename OutputStructure, typename OutputView, 
			typename OutputImfR, typename OutputImfC,
			typename InputStructure1, typename InputView1, 
			typename InputImfR1, typename InputImfC1,
			typename InputStructure2, typename InputView2, 
			typename InputImfR2, typename InputImfC2
		>
		typename std::enable_if<
			( BandPos1 < std::tuple_size< typename InputStructure1::band_intervals >::value ) &&
			( BandPos2 < std::tuple_size< typename InputStructure2::band_intervals >::value ),
		RC >::type mxm_band_generic( 
			alp::Matrix< OutputType, OutputStructure, 
			Density::Dense, OutputView, OutputImfR, OutputImfC, dispatch > &C,
			const alp::Matrix< InputType1, InputStructure1, 
			Density::Dense, InputView1, InputImfR1, InputImfC1, dispatch > &A,
			const alp::Matrix< InputType2, InputStructure2, 
			Density::Dense, InputView2, InputImfR2, InputImfC2, dispatch > &B,
			const Operator &oper,
			const Monoid &monoid,
			const MulMonoid &mulMonoid
		) {

			OutputType temp;

			const std::ptrdiff_t M   { static_cast< std::ptrdiff_t >( nrows( C ) ) };
			const std::ptrdiff_t N   { static_cast< std::ptrdiff_t >( ncols( C ) ) };
			const std::ptrdiff_t K   { static_cast< std::ptrdiff_t >( ncols( A ) ) };

			const std::ptrdiff_t l_a { structures::get_lower_limit< BandPos1 >( A ) };
			const std::ptrdiff_t u_a { structures::get_upper_limit< BandPos1 >( A ) };

			const std::ptrdiff_t l_b { structures::get_lower_limit< BandPos2 >( B ) };
			const std::ptrdiff_t u_b { structures::get_upper_limit< BandPos2 >( B ) };
			
			// In case of symmetry the iteration domain intersects the the upper 
			// (or lower) domain of C
			constexpr bool is_sym_a { structures::is_a< InputStructure1, structures::Symmetric >::value };
			constexpr bool is_sym_b { structures::is_a< InputStructure2, structures::Symmetric >::value };
			constexpr bool is_sym_c { structures::is_a< OutputStructure, structures::Symmetric >::value };

			// Temporary until adding multiple symmetry directions
			constexpr bool sym_up_a { is_sym_a };
			constexpr bool sym_up_b { is_sym_b };
			constexpr bool sym_up_c { is_sym_c };

			// Intersecting potential symmetry of A and B, 
			// in which case consider case Up( A ) * Up( B )
			for( std::ptrdiff_t i = 0; i < M; ++i ) {
				// Size + Symmetry constraints
				//    sym_up_c * i   <= j < N
				// Band constraints
				// /\ i + l_a + l_b <= j < i + u_a + u_b - 1 (u is past-the-end)
				for( std::ptrdiff_t j = std::max( sym_up_c * i, i + l_a + l_b ); 
					 j < std::min( N, i + u_a + u_b - 1 ); 
					 ++j ) {
					
					auto &c_val = internal::access( C, internal::getStorageIndex( C, i, j ) );

					// Size + Symmetry constraints
					//    sym_up_a * i <= l < K * (!sym_up_b) + ( j + 1 ) * (sym_up_b)   
					// Band constraints
					// /\ i + l_a      <= l < i + u_a        
					// /\ j - u_b + 1  <= l < j - l_b + 1
					for( std::ptrdiff_t l = std::max( { sym_up_a * i, i + l_a, j - u_b + 1 } ); 
						 l < std::min( { K * ( !sym_up_b ) + ( j + 1 ) * sym_up_b, i + u_a, j - l_b + 1 } ); 
						 ++l ) {
						const auto ta { internal::access( A, internal::getStorageIndex( A, i, l ) ) };
						const auto tb { internal::access( B, internal::getStorageIndex( B, l, j ) ) };
						(void)internal::apply( temp, ta, tb, oper );
						// std::cout << c_val << " += " << temp << " = " << ta << " * " << tb << std::endl;
						(void)internal::foldl( c_val, temp, monoid.getOperator() );
						// std::cout << c_val << std::endl;
					}
				}
			}

			if ( sym_up_b ) {
				// Intersecting potential symmetry of A and B, 
				// in which case consider case Up( A ) * Lo( B )
				for( std::ptrdiff_t i = 0; i < M; ++i ) {
					// Size + Symmetry constraints
					//    sym_up_c * i   <= j < N - 1 
					// Band constraints
					// /\ i + l_a + l_b <= j < i + u_a + u_b - 1
					for( std::ptrdiff_t j = std::max( sym_up_c * i, i + l_a + l_b ); 
						j < std::min( N - 1, i + u_a + u_b - 1 ); 
						++j ) {
						
						auto &c_val = internal::access( C, internal::getStorageIndex( C, i, j ) );

						// Size + Symmetry constraints
						//    max(sym_up_a * i, j + 1 ) <= l < K
						// Band constraints
						// /\ i + l_a              <= l < i + u_a 
						// /\ j - u_b + 1          <= l < j - l_b + 1
						for( std::ptrdiff_t l = std::max( { sym_up_a * i, j + 1, i + l_a, j - u_b + 1 } ); 
							l < std::min( { K, i + u_a, j - l_b + 1 } ); 
							++l ) {
							const auto ta { internal::access( A, internal::getStorageIndex( A, i, l ) ) };
							// Access to B^T
							const auto tb { internal::access( B, internal::getStorageIndex( B, j, l ) ) };
							(void)internal::apply( temp, ta, tb, oper );
							(void)internal::foldl( c_val, temp, monoid.getOperator() );
						}
					}
				}
			}

			if ( sym_up_a ) {
				// Intersecting potential symmetry of A and B, 
				// in which case consider case Lo( A ) * Up( B )
				for( std::ptrdiff_t i = 0; i < M; ++i ) {
					// Size + Symmetry constraints
					//    sym_up_c * i   <= j < N
					// Band constraints
					// /\ i + l_a + l_b <= j < i + u_a + u_b - 1
					for( std::ptrdiff_t j = std::max( sym_up_c * i, i + l_a + l_b ); 
						j < std::min( N, i + u_a + u_b - 1 ); 
						++j ) {
						
						auto &c_val = internal::access( C, internal::getStorageIndex( C, i, j ) );

						// Size + Symmetry constraints
						//    0                    <= l < min(i, 
						//                                    K * ( !sym_up_b ) 
						//                                    + ( j + 1 ) * sym_up_b
						//                                    )
						// Band constraints
						// /\ i + l_a              <= l < i + u_a 
						// /\ j - u_b + 1          <= l < j - l_b + 1
						for( std::ptrdiff_t l = std::max( { ( std::ptrdiff_t )0, i + l_a, j - u_b + 1 } ); 
							l < std::min( { i, K * ( !sym_up_b ) + ( j + 1 ) * sym_up_b, i + u_a, j - l_b + 1 } ); 
							++l ) {
							// Access to A^T
							const auto ta { internal::access( A, internal::getStorageIndex( A, l, i ) ) };
							const auto tb { internal::access( B, internal::getStorageIndex( B, l, j ) ) };
							(void)internal::apply( temp, ta, tb, oper );
							(void)internal::foldl( c_val, temp, monoid.getOperator() );
						}
					}
				}

				if( ( !sym_up_c ) && sym_up_b ) {
					// Intersecting potential symmetry of A and B, 
					// in which case consider case Lo( A ) * Lo( B ).
					// Useful only if C is not sym
					for( std::ptrdiff_t i = 2; i < M; ++i ) {
						// Size + Symmetry constraints
						//    0             <= j < i - 1
						// Band constraints
						// /\ i + l_a + l_b <= j < i + u_a + u_b - 1
						for( std::ptrdiff_t j = std::max( ( std::ptrdiff_t )0, i + l_a + l_b ); 
							j < std::min( i - 1, i + u_a + u_b - 1 ); 
							++j ) {
							
							auto &c_val = internal::access( C, internal::getStorageIndex( C, i, j ) );

							// Size + Symmetry constraints
							//    j + 1                <= l < i
							// Band constraints
							// /\ i + l_a              <= l < i + u_a 
							// /\ j - u_b + 1          <= l < j - l_b + 1
							for( std::ptrdiff_t l = std::max( { j + 1, i + l_a, j - u_b + 1 } ); 
								l < std::min( { i, i + u_a, j - l_b + 1 } ); 
								++l ) {
								// Access to A^T
								const auto ta { internal::access( A, internal::getStorageIndex( A, l, i ) ) };
								// Access to B^T
								const auto tb { internal::access( B, internal::getStorageIndex( B, j, l ) ) };
								(void)internal::apply( temp, ta, tb, oper );
								(void)internal::foldl( c_val, temp, monoid.getOperator() );
							}
						}
					}
				}
			}


			return mxm_band_generic< BandPos1, BandPos2 + 1 >( C, A, B, oper, monoid, mulMonoid );
		}

		/**
		 * \internal general mxm implementation that all mxm variants using 
		 * structured matrices refer to.
		 */
		template<
			bool allow_void,
			class MulMonoid,
			typename OutputType, typename InputType1, typename InputType2,
			class Operator, class Monoid,
			typename OutputStructure, typename OutputView,
			typename OutputImfR, typename OutputImfC,
			typename InputStructure1, typename InputView1,
			typename InputImfR1, typename InputImfC1,
			typename InputStructure2, typename InputView2,
			typename InputImfR2, typename InputImfC2
		>
		RC mxm_generic(
			alp::Matrix< OutputType, OutputStructure,
			Density::Dense, OutputView, OutputImfR, OutputImfC, dispatch > &C,
			const alp::Matrix< InputType1, InputStructure1,
			Density::Dense, InputView1, InputImfR1, InputImfC1, dispatch > &A,
			const alp::Matrix< InputType2, InputStructure2,
			Density::Dense, InputView2, InputImfR2, InputImfC2, dispatch > &B,
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

			static_assert(
				!(
					std::is_same< InputType1, void >::value ||
					std::is_same< InputType2, void >::value
				),
				"alp::internal::mxm_generic: the operator-monoid version of mxm cannot be "
				"used if either of the input matrices is a pattern matrix (of type "
				"void)"
			);

#ifdef _DEBUG
			std::cout << "In alp::internal::mxm_generic (dispatch)\n";
#endif

			// Early exit checks
			if( ! internal::getInitialized( A ) ||
				! internal::getInitialized( B ) ||
				! internal::getInitialized( C )
			) {
				internal::setInitialized( C, false );
				return SUCCESS;
			}

			const std::ptrdiff_t m   { static_cast< std::ptrdiff_t >( nrows( C ) ) };
			const std::ptrdiff_t n   { static_cast< std::ptrdiff_t >( ncols( C ) ) };
			const std::ptrdiff_t m_a { static_cast< std::ptrdiff_t >( nrows( A ) ) };
			const std::ptrdiff_t k   { static_cast< std::ptrdiff_t >( ncols( A ) ) };
			const std::ptrdiff_t k_b { static_cast< std::ptrdiff_t >( nrows( B ) ) };
			const std::ptrdiff_t n_b { static_cast< std::ptrdiff_t >( ncols( B ) ) };

			if( m != m_a || k != k_b || n != n_b ) {
				return MISMATCH;
			}

			return mxm_band_generic< 0, 0 >( C, A, B, oper, monoid, mulMonoid );
		}

	} // namespace internal

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
	template<
		typename OutputStructMatT,
		typename InputStructMatT1,
		typename InputStructMatT2,
		class Semiring
	>
	RC mxm( OutputStructMatT & C,
		const InputStructMatT1 & A,
		const InputStructMatT2 & B,
		const Semiring & ring = Semiring(),
		const PHASE &phase = NUMERICAL,
		const typename std::enable_if< ! alp::is_object< typename OutputStructMatT::value_type >::value && ! alp::is_object< typename InputStructMatT1::value_type >::value && ! alp::is_object< typename InputStructMatT2::value_type >::value && alp::is_semiring< Semiring >::value,
			void >::type * const = NULL ) {
		(void)phase;

#if 1
		return internal::mxm_generic< false >( C, A, B, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid(), ring.getMultiplicativeMonoid() );

#else
		// \todo Check that the ring is using standard + and * operators
		(void) ring;

		// Offload to blas gemm
		const size_t m = nrows( C );
		const size_t n = ncols( C );
		const size_t k = ncols( A );
		
		cblas_dgemm(
			CblasRowMajor, CblasNoTrans, CblasNoTrans,
			m, n, k,
			1,
			internal::getRawPointerToFirstElement( A ), internal::getLeadingDimension( A ),
			internal::getRawPointerToFirstElement( B ), internal::getLeadingDimension( B ),
			1,
			internal::getRawPointerToFirstElement( C ), internal::getLeadingDimension( C )
		);
		return SUCCESS;
#endif
	}

	/**
	 * Dense Matrix-Matrix multiply between structured matrices.
	 * Version with additive monoid and multiplicative operator
	 */
	template< typename OutputStructMatT,
		typename InputStructMatT1,
		typename InputStructMatT2,
		class Operator, class Monoid
	>
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

		return internal::mxm_generic< false >( C, A, B, mulOp, addM, Monoid() );
	}

	namespace internal {

		/**
		 * Applies eWiseApply to all elements of the given band
		 * Forward declaration. Specialization handle bound-checking.
		 * Assumes compatible parameters:
		 *   - matching structures
		 *   - matching dynamic
		 */
		template<
			size_t band_index,
			bool left_scalar, bool right_scalar,
			Descriptor descr,
			class MulMonoid,
			typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
			typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
			typename InputTypeScalar1, typename InputStructureScalar1,
			typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
			typename InputTypeScalar2, typename InputStructureScalar2,
			class Operator,
			std::enable_if_t<
				band_index >= std::tuple_size< typename OutputStructure::band_intervals >::value
			> * = nullptr
		>
		RC eWiseApply_matrix_band_generic(
			alp::Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, dispatch > *C,
			const alp::Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, dispatch > *A,
			const alp::Scalar< InputTypeScalar1, InputStructureScalar1, dispatch > *alpha,
			const alp::Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, dispatch > *B,
			const alp::Scalar< InputTypeScalar2, InputStructureScalar2, dispatch > *beta,
			const Operator &oper,
			const MulMonoid &mulMonoid,
			const typename std::enable_if<
				!alp::is_object< OutputType >::value &&
				!alp::is_object< InputType1 >::value &&
				!alp::is_object< InputType2 >::value &&
				alp::is_operator< Operator >::value,
			void >::type * const = nullptr
		) {
			(void)C;
			(void)A;
			(void)alpha;
			(void)B;
			(void)beta;
			(void)oper;
			(void)mulMonoid;
			return SUCCESS;
		}

		/** Specialization for band index within the bounds */
		template<
			size_t band_index,
			bool left_scalar, bool right_scalar,
			Descriptor descr,
			class MulMonoid,
			typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
			typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
			typename InputTypeScalar1, typename InputStructureScalar1,
			typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
			typename InputTypeScalar2, typename InputStructureScalar2,
			class Operator,
			std::enable_if_t<
				band_index < std::tuple_size< typename OutputStructure::band_intervals >::value
			> * = nullptr
		>
		RC eWiseApply_matrix_band_generic(
			alp::Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, dispatch > *C,
			const alp::Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, dispatch > *A,
			const alp::Scalar< InputTypeScalar1, InputStructureScalar1, dispatch > *alpha,
			const alp::Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, dispatch > *B,
			const alp::Scalar< InputTypeScalar2, InputStructureScalar2, dispatch > *beta,
			const Operator &oper,
			const MulMonoid &mulMonoid,
			const typename std::enable_if<
				!alp::is_object< OutputType >::value &&
				!alp::is_object< InputType1 >::value &&
				!alp::is_object< InputType2 >::value &&
				alp::is_operator< Operator >::value,
			void >::type * const = nullptr
		) {
			(void)mulMonoid;
			assert( C != nullptr );
			// In case of symmetry the iteration domain intersects the the upper
			// (or lower) domain of A
			constexpr bool is_sym_c = structures::is_a< OutputStructure, structures::Symmetric >::value;
			constexpr bool is_sym_a = structures::is_a< InputStructure1, structures::Symmetric >::value;
			constexpr bool is_sym_b = structures::is_a< InputStructure2, structures::Symmetric >::value;

			// Temporary until adding multiple symmetry directions
			constexpr bool sym_up_c = is_sym_c;
			constexpr bool sym_up_a = is_sym_a;
			constexpr bool sym_up_b = is_sym_b;

			const auto i_limits = structures::calculate_row_coordinate_limits< band_index >( *C );

			for( size_t i = i_limits.first; i < i_limits.second; ++i ) {

				const auto j_limits = structures::calculate_column_coordinate_limits< band_index >( *C, i );

				for( size_t j = j_limits.first; j < j_limits.second; ++j ) {
					auto &C_val = internal::access( *C, internal::getStorageIndex( *C, i, j ) );

					// Calculate indices to A and B depending on matching symmetry with C
					const size_t A_i = ( sym_up_c == sym_up_a ) ? i : j;
					const size_t A_j = ( sym_up_c == sym_up_a ) ? j : i;
					const size_t B_i = ( sym_up_c == sym_up_b ) ? i : j;
					const size_t B_j = ( sym_up_c == sym_up_b ) ? j : i;

					if( left_scalar ) {
						if( right_scalar ) {
							// C = alpha . beta
							internal::apply( C_val, **alpha, **beta, oper );
						} else {
							// C = alpha . B
							const auto &B_val = internal::access( *B, internal::getStorageIndex( *B, B_i, B_j ) );
							internal::apply( C_val, **alpha, B_val, oper );
						}
					} else {
						if( right_scalar ) {
							// C = A . beta
							const auto &A_val = internal::access( *A, internal::getStorageIndex( *A, A_i, A_j ) );
							internal::apply( C_val, A_val, **beta, oper );
						} else {
							// C = A . B
							const auto &A_val = internal::access( *A, internal::getStorageIndex( *A, A_i, A_j ) );
							const auto &B_val = internal::access( *B, internal::getStorageIndex( *B, B_i, B_j ) );
							internal::apply( C_val, A_val, B_val, oper );
						}
					}
				}
			}
			return eWiseApply_matrix_band_generic<
				band_index + 1, left_scalar, right_scalar, descr
			>( C, A, alpha, B, beta, oper, mulMonoid );
		}

		/**
		 * \internal general elementwise matrix application that all eWiseApply variants refer to.
		 */
		template<
			bool left_scalar, bool right_scalar,
			Descriptor descr,
			class MulMonoid,
			typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
			typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
			typename InputTypeScalar1, typename InputStructureScalar1,
			typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
			typename InputTypeScalar2, typename InputStructureScalar2,
			class Operator
		>
		RC eWiseApply_matrix_generic(
			alp::Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, dispatch > *C,
			const alp::Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, dispatch > *A,
			const alp::Scalar< InputTypeScalar1, InputStructureScalar1, dispatch > *alpha,
			const alp::Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, dispatch > *B,
			const alp::Scalar< InputTypeScalar2, InputStructureScalar2, dispatch > *beta,
			const Operator &oper,
			const MulMonoid &mulMonoid,
			const typename std::enable_if<
				!alp::is_object< OutputType >::value &&
				!alp::is_object< InputType1 >::value &&
				!alp::is_object< InputType2 >::value &&
				alp::is_operator< Operator >::value,
			void >::type * const = NULL
		) {
			(void)alpha;
			(void)beta;
			(void)oper;
			(void)mulMonoid;

#ifdef _DEBUG
			std::cout << "In alp::internal::eWiseApply_matrix_generic\n";
#endif

			// run-time checks
			// TODO: support left/right_scalar
			const size_t m = alp::nrows( *C );
			const size_t n = alp::ncols( *C );

			if( !left_scalar ){
				assert( A != nullptr );
				if( m != nrows( *A ) || n != ncols( *A ) ) {
					return MISMATCH;
				}
			}
			if( !right_scalar ){
				assert( B != nullptr );
				if( m != nrows( *B ) || n != ncols( *B ) ) {
					return MISMATCH;
				}
			}

			// delegate to single-band variant
			return eWiseApply_matrix_band_generic< 0, left_scalar, right_scalar, descr >( C, A, alpha, B, beta, oper, mulMonoid );
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
		class MulMonoid
	>
	RC eWiseApply(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, dispatch > &C,
		const Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, dispatch > &A,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, dispatch > &B,
		const MulMonoid &mulmono,
		const typename std::enable_if< !alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_monoid< MulMonoid >::value,
		void >::type * const = NULL
	) {
		// static checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D1, InputType1 >::value ),
			"alp::eWiseApply (dispatch, matrix <- matrix x matrix, monoid)",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D2, InputType2 >::value ),
			"alp::eWiseApply (dispatch, matrix <- matrix x matrix, monoid)",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the monoid operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename MulMonoid::D3, OutputType >::value ),
			"alp::eWiseApply (dispatch, matrix <- matrix x matrix, monoid)",
			"called with an output matrix C that does not match the output domain "
			"of the monoid operator"
		);

#ifdef _DEBUG
		std::cout << "In alp::eWiseApply (dispatch, monoid)\n";
#endif
		constexpr Scalar< InputType1, structures::General, dispatch > *no_scalar = nullptr;
		constexpr bool left_scalar = false;
		constexpr bool right_scalar = false;

		return internal::eWiseApply_matrix_generic< left_scalar, right_scalar, descr >(
			&C, &A, no_scalar, &B, no_scalar, mulmono.getOperator(), mulmono
		);
	}

	/**
	 * Returns a view over the general rank-1 matrix computed with the outer product.
	 * Version for the case when input vectors are the same vector,
	 * which results in a symmetric matrix.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		class Operator
	>
	Matrix<
		typename Operator::D3,
		typename std::conditional<
			grb::utils::is_complex< typename Operator::D3 >::value,
			alp::structures::Hermitian,
			alp::structures::Symmetric
		>::type,
		Density::Dense,
		view::Functor< std::function< void( typename Operator::D3 &, const size_t, const size_t ) > >,
		imf::Id, imf::Id,
		dispatch
	>
	outer(
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, dispatch > &x,
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

		std::function< void( typename Operator::D3 &, const size_t, const size_t ) > data_lambda =
			[ &x, &mul ]( typename Operator::D3 &result, const size_t i, const size_t j ) {
				//set( ret, alp::identities::zero );
				internal::apply(
					result, x[ i ],
					grb::utils::is_complex< InputType >::conjugate( x[ j ] ),
					mul
				);
			};
		std::function< bool() > init_lambda =
			[ &x ]() -> bool {
				return internal::getInitialized( x );
			};

		return Matrix<
			typename Operator::D3,
			typename std::conditional<
				grb::utils::is_complex< typename Operator::D3 >::value,
				alp::structures::Hermitian,
				alp::structures::Symmetric
			>::type,
			Density::Dense,
			view::Functor< std::function< void( typename Operator::D3 &, const size_t, const size_t ) > >,
			imf::Id, imf::Id,
			dispatch
		>(
			init_lambda,
			getLength( x ),
			data_lambda
		);

	}

} // end namespace ``alp''

#undef NO_CAST_ASSERT
#undef NO_CAST_OP_ASSERT

#endif // end ``_H_ALP_DISPATCH_BLAS3''

