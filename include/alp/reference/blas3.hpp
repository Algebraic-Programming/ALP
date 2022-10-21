
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

#include <algorithm>   // for std::min/max
#include <type_traits> // for std::enable_if

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
			Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
			const alp::Matrix< InputType1, InputStructure1, 
			Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &A,
			const alp::Matrix< InputType2, InputStructure2, 
			Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
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
			Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
			const alp::Matrix< InputType1, InputStructure1, 
			Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &A,
			const alp::Matrix< InputType2, InputStructure2, 
			Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
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
			Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
			const alp::Matrix< InputType1, InputStructure1, 
			Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &A,
			const alp::Matrix< InputType2, InputStructure2, 
			Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
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
			Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
			const alp::Matrix< InputType1, InputStructure1, 
			Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &A,
			const alp::Matrix< InputType2, InputStructure2, 
			Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
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
			Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
			const alp::Matrix< InputType1, InputStructure1, 
			Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &A,
			const alp::Matrix< InputType2, InputStructure2, 
			Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
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
			std::cout << "In alp::internal::mxm_generic (reference)\n";
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

		return internal::mxm_generic< false >( C, A, B, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid(), ring.getMultiplicativeMonoid() );
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
			typename std::enable_if_t<
				band_index >= std::tuple_size< typename OutputStructure::band_intervals >::value
			> * = nullptr
		>
		RC eWiseApply_matrix_band_generic(
			alp::Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > *C,
			const alp::Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > *A,
			const alp::Scalar< InputTypeScalar1, InputStructureScalar1, reference > *alpha,
			const alp::Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > *B,
			const alp::Scalar< InputTypeScalar2, InputStructureScalar2, reference > *beta,
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
			typename std::enable_if_t<
				band_index < std::tuple_size< typename OutputStructure::band_intervals >::value
			> * = nullptr
		>
		RC eWiseApply_matrix_band_generic(
			alp::Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > *C,
			const alp::Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > *A,
			const alp::Scalar< InputTypeScalar1, InputStructureScalar1, reference > *alpha,
			const alp::Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > *B,
			const alp::Scalar< InputTypeScalar2, InputStructureScalar2, reference > *beta,
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
			alp::Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > *C,
			const alp::Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > *A,
			const alp::Scalar< InputTypeScalar1, InputStructureScalar1, reference > *alpha,
			const alp::Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > *B,
			const alp::Scalar< InputTypeScalar2, InputStructureScalar2, reference > *beta,
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
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &A,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
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
		std::cout << "In alp::eWiseApply (reference, monoid)\n";
#endif
		constexpr Scalar< InputType1, structures::General, reference > *no_scalar = nullptr;
		constexpr bool left_scalar = false;
		constexpr bool right_scalar = false;

		return internal::eWiseApply_matrix_generic< left_scalar, right_scalar, descr >(
			&C, &A, no_scalar, &B, no_scalar, mulmono.getOperator(), mulmono
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
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class MulMonoid
	>
	RC eWiseApply(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Scalar< InputType1, InputStructure1, reference > &alpha,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
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
		std::cout << "In alp::eWiseApply (reference, monoid)\n";
#endif

		constexpr Matrix< InputType1, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > * no_matrix = nullptr;
		constexpr Scalar< InputType2, structures::General, reference > *no_scalar = nullptr;
		constexpr bool left_scalar = true;
		constexpr bool right_scalar = false;

		return internal::eWiseApply_matrix_generic< left_scalar, right_scalar, descr >(
			&C, no_matrix, &alpha, &B, no_scalar, mulmono.getOperator(), mulmono
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
		typename InputType2, typename InputStructure2,
		class MulMonoid
	>
	RC eWiseApply(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &A,
		const Scalar< InputType2, InputStructure2, reference > &beta,
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
		std::cout << "In alp::eWiseApply (reference, monoid)\n";
#endif

		constexpr Scalar< InputType1, structures::General, reference > *no_scalar = nullptr;
		constexpr Matrix< InputType2, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > *no_matrix = nullptr;
		constexpr bool left_scalar = false;
		constexpr bool right_scalar = true;

		return internal::eWiseApply_matrix_generic< left_scalar, right_scalar, descr >(
			&C, &A, no_scalar, no_matrix, &beta, mulmono.getOperator(), mulmono
		);
	}

	namespace internal {

		/**
		 * Applies eWiseMul to all elements of the given band
		 * Specialization handle bound-checking.
		 * Assumes compatible parameters:
		 *   - matching structures
		 *   - matching dynamic sizes
		 */
		template<
			size_t band_index,
			bool left_scalar, bool right_scalar,
			Descriptor descr,
			class Ring,
			typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
			typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
			typename InputTypeScalar1, typename InputStructureScalar1,
			typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
			typename InputTypeScalar2, typename InputStructureScalar2,
			typename std::enable_if_t<
				band_index >= std::tuple_size< typename OutputStructure::band_intervals >::value
			> * = nullptr
		>
		RC eWiseMul_matrix_band_generic(
			alp::Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > *C,
			const alp::Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > *A,
			const alp::Scalar< InputTypeScalar1, InputStructureScalar1, reference > *alpha,
			const alp::Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > *B,
			const alp::Scalar< InputTypeScalar2, InputStructureScalar2, reference > *beta,
			const Ring &ring,
			const std::enable_if_t<
				!alp::is_object< OutputType >::value &&
				!alp::is_object< InputType1 >::value &&
				!alp::is_object< InputType2 >::value &&
				alp::is_semiring< Ring >::value
			> * const = nullptr
		) {
			(void) C;
			(void) A;
			(void) alpha;
			(void) B;
			(void) beta;
			(void) ring;
			return SUCCESS;
		}

		/** Specialization for band index within the bounds */
		template<
			size_t band_index,
			bool left_scalar, bool right_scalar,
			Descriptor descr,
			class Ring,
			typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
			typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
			typename InputTypeScalar1, typename InputStructureScalar1,
			typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
			typename InputTypeScalar2, typename InputStructureScalar2,
			typename std::enable_if_t<
				band_index < std::tuple_size< typename OutputStructure::band_intervals >::value
			> * = nullptr
		>
		RC eWiseMul_matrix_band_generic(
			alp::Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > *C,
			const alp::Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > *A,
			const alp::Scalar< InputTypeScalar1, InputStructureScalar1, reference > *alpha,
			const alp::Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > *B,
			const alp::Scalar< InputTypeScalar2, InputStructureScalar2, reference > *beta,
			const Ring &ring,
			const std::enable_if_t<
				!alp::is_object< OutputType >::value &&
				!alp::is_object< InputType1 >::value &&
				!alp::is_object< InputType2 >::value &&
				alp::is_semiring< Ring >::value
			> * const = nullptr
		) {
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
					OutputType C_tmp;

					// Calculate indices to A and B depending on matching symmetry with C
					const size_t A_i = ( sym_up_c == sym_up_a ) ? i : j;
					const size_t A_j = ( sym_up_c == sym_up_a ) ? j : i;
					const size_t B_i = ( sym_up_c == sym_up_b ) ? i : j;
					const size_t B_j = ( sym_up_c == sym_up_b ) ? j : i;

					if( left_scalar ) {
						if( right_scalar ) {
							// C = alpha . beta
							internal::apply( C_tmp, **alpha, **beta, ring.getMultiplicativeOperator() );
						} else {
							// C = alpha . B
							const auto &B_val = internal::access( *B, internal::getStorageIndex( *B, B_i, B_j ) );
							internal::apply( C_tmp, **alpha, B_val, ring.getMultiplicativeOperator() );
						}
					} else {
						if( right_scalar ) {
							// C = A . beta
							const auto &A_val = internal::access( *A, internal::getStorageIndex( *A, A_i, A_j ) );
							internal::apply( C_tmp, A_val, **beta, ring.getMultiplicativeOperator() );
						} else {
							// C = A . B
							const auto &A_val = internal::access( *A, internal::getStorageIndex( *A, A_i, A_j ) );
							const auto &B_val = internal::access( *B, internal::getStorageIndex( *B, B_i, B_j ) );
							internal::apply( C_tmp, A_val, B_val, ring.getMultiplicativeOperator() );
						}
					}

					auto &C_val = internal::access( *C, internal::getStorageIndex( *C, i, j ) );
					internal::foldl( C_val, C_tmp, ring.getAdditiveOperator() );
				}
			}
			return eWiseMul_matrix_band_generic<
				band_index + 1, left_scalar, right_scalar, descr
			>( C, A, alpha, B, beta, ring );
		}

		/**
		 * \internal general elementwise matrix application that all eWiseApply variants refer to.
		 */
		template<
			bool left_scalar, bool right_scalar,
			Descriptor descr,
			class Ring,
			typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
			typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
			typename InputTypeScalar1, typename InputStructureScalar1,
			typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
			typename InputTypeScalar2, typename InputStructureScalar2
		>
		RC eWiseMul_matrix_generic(
			alp::Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > *C,
			const alp::Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > *A,
			const alp::Scalar< InputTypeScalar1, InputStructureScalar1, reference > *alpha,
			const alp::Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > *B,
			const alp::Scalar< InputTypeScalar2, InputStructureScalar2, reference > *beta,
			const Ring &ring = Ring(),
			const std::enable_if_t<
				!alp::is_object< OutputType >::value &&
				!alp::is_object< InputType1 >::value &&
				!alp::is_object< InputType2 >::value &&
				alp::is_semiring< Ring >::value
			> * const = nullptr
		) {

#ifdef _DEBUG
			std::cout << "In alp::internal::eWiseMul_matrix_generic\n";
#endif

			// run-time checks
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
			return eWiseMul_matrix_band_generic< 0, left_scalar, right_scalar, descr >( C, A, alpha, B, beta, ring );
		}

	} // namespace internal

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
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2
	>
	RC eWiseMul(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &A,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
		const Ring &ring = Ring(),
		const std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_semiring< Ring >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Ring::D1, InputType1 >::value ),
			"alp::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Ring::D2, InputType2 >::value ),
			"alp::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Ring::D3, OutputType >::value ),
			"alp::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring"
		);
#ifdef _DEBUG
		std::cout << "eWiseMul (reference, matrix <- matrix x matrix) dispatches to internal::eWiseMul_matrix_generic (matrix <- matrix x matrix)\n";
#endif
		constexpr Scalar< InputType1, structures::General, reference > *no_scalar = nullptr;
		constexpr bool left_scalar = false;
		constexpr bool right_scalar = false;
		return internal::eWiseMul_matrix_generic< left_scalar, right_scalar, descr >( &C, &A, no_scalar, &B, no_scalar, ring );
	}

	/**
	 * eWiseMul, version where A is a scalar.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, 
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2
	>
	RC eWiseMul(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Scalar< InputType1, InputStructure1, reference > &alpha,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &B,
		const Ring &ring = Ring(),
		const std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_semiring< Ring >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Ring::D1, InputType1 >::value ),
			"alp::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring"
		);
		NO_CAST_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Ring::D2, InputType2 >::value ),
			"alp::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring"
		);
		NO_CAST_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Ring::D3, OutputType >::value ),
			"alp::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring"
		);
#ifdef _DEBUG
		std::cout << "eWiseMul (reference, matrix <- scalar x matrix) dispatches to internal::eWiseMul_matrix_generic (matrix <- scalar x matrix)\n";
#endif
		constexpr Matrix< InputType1, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > *no_matrix = nullptr;
		constexpr Scalar< InputType2, structures::General, reference > *no_scalar = nullptr;
		constexpr bool left_scalar = true;
		constexpr bool right_scalar = false;
		return internal::eWiseMul_matrix_generic< left_scalar, right_scalar, descr >( &C, no_matrix, &alpha, &B, no_scalar, ring );
	}

	/**
	 * eWiseMul, version where B is a scalar.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2
	>
	RC eWiseMul(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &A,
		const Scalar< InputType2, InputStructure2, reference > &beta,
		const Ring &ring = Ring(),
		const std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_semiring< Ring >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Ring::D1, InputType1 >::value ),
			"alp::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring"
		);
		NO_CAST_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Ring::D2, InputType2 >::value ),
			"alp::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring"
		);
		NO_CAST_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Ring::D3, OutputType >::value ),
			"alp::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring"
		);
#ifdef _DEBUG
		std::cout << "eWiseMul (reference, matrix <- matrix x scalar) dispatches to internal::eWiseMul_matrix_generic (matrix <- matrix x scalar)\n";
#endif
		constexpr Scalar< InputType1, structures::General, reference > *no_scalar = nullptr;
		constexpr Matrix< InputType2, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > *no_matrix = nullptr;
		constexpr bool left_scalar = false;
		constexpr bool right_scalar = true;
		return internal::eWiseMul_matrix_generic< left_scalar, right_scalar, descr >( &C, &A, no_scalar, no_matrix, &beta, ring );
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
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Operator
	>
	RC outer( Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & u,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & v,
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
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Operator
	>
	Matrix< typename Operator::D3, structures::General, Density::Dense,
		view::Functor< std::function< void( InputType1 &, const size_t, const size_t ) > >,
		imf::Id, imf::Id,
		reference
	>
	outer(
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &y,
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

		std::function< void( typename Operator::D3 &, const size_t, const size_t ) > data_lambda =
			[ &x, &y, &mul ]( typename Operator::D3 &result, const size_t i, const size_t j ) {
				//set( ret, alp::identities::zero );
				internal::apply(
					result,
					x[ i ],
					grb::utils::is_complex< InputType2 >::conjugate( y[ j ] ),
					mul
				);
			};
		std::function< bool() > init_lambda =
			[ &x, &y ]() -> bool {
				return internal::getInitialized( x ) && internal::getInitialized( y );
			};

		return Matrix<
			typename Operator::D3,
			structures::General,
			Density::Dense,
			view::Functor< std::function< void( typename Operator::D3 &, const size_t, const size_t ) > >,
			imf::Id, imf::Id,
			reference
		>(
			init_lambda,
			getLength( x ),
			getLength( y ),
			data_lambda
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
		reference
	>
	outer(
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &x,
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
			reference
		>(
			init_lambda,
			getLength( x ),
			data_lambda
		);

	}

	/**
	 * Sets all elements of the output matrix to the values of the input matrix.
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
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC
	>
	RC set(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &A
	) noexcept {
		static_assert(
			!std::is_same< OutputType, void >::value,
			"alp::set (set to value): cannot have a pattern matrix as output"
		);
#ifdef _DEBUG
		std::cout << "Called alp::set (matrix-to-matrix, reference)" << std::endl;
#endif
		// static checks
		NO_CAST_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< InputType, OutputType >::value ),
			"alp::set", "called with non-matching value types"
		);

		static_assert(
			!internal::is_functor_based<
				Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference >
			>::value,
			"alp::set cannot be called with a functor-based matrix as a destination."
		);

		// TODO: Improve this check to account for non-zero structrue (i.e., bands)
		//       and algebraic properties (e.g., symmetry)
		static_assert(
			std::is_same< OutputStructure, InputStructure >::value,
			"alp::set cannot be called for containers with different structures."
		);

		if( ( nrows( C ) != nrows( A ) ) || ( ncols( C ) != ncols( A ) ) ) {
			return MISMATCH;
		}

		if( !internal::getInitialized( A ) ) {
			internal::setInitialized( C, false );
			return SUCCESS;
		}

		internal::setInitialized( C, true );
		return foldl( C, A, alp::operators::right_assign< OutputType >() );
	}

	/**
	 * Sets all elements of the given matrix to the value of the given scalar.
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
		typename InputType, typename InputStructure
	>
	RC set(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Scalar< InputType, InputStructure, reference > &val
	) noexcept {

		static_assert(
			!std::is_same< OutputType, void >::value,
			"alp::set (set to matrix): cannot have a pattern matrix as output"
		);
#ifdef _DEBUG
		std::cout << "Called alp::set (matrix-to-value, reference)" << std::endl;
#endif
		// static checks
		NO_CAST_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< InputType, OutputType >::value ),
			"alp::set", "called with non-matching value types"
		);

		static_assert(
			!internal::is_functor_based<
				Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference >
			>::value,
			"alp::set cannot be called with a functor-based matrix as a destination."
		);

		if( !internal::getInitialized( val ) ) {
			internal::setInitialized( C, false );
			return SUCCESS;
		}

		internal::setInitialized( C, true );
		return foldl( C, val, alp::operators::right_assign< OutputType >() );
	}

} // end namespace ``alp''

#undef NO_CAST_ASSERT
#undef NO_CAST_OP_ASSERT

#endif // end ``_H_ALP_REFERENCE_BLAS3''

