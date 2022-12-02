
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

#ifndef _H_ALP_OMP_BLAS3
#define _H_ALP_OMP_BLAS3

#include <algorithm>   // for std::min/max
#include <type_traits> // for std::enable_if

#include <alp/base/blas3.hpp>
#include <alp/descriptors.hpp>
#include <alp/structures.hpp>

// Include backend to which sequential work is delegated
#ifdef _ALP_OMP_WITH_REFERENCE
 #include <alp/reference/blas3.hpp>
 #include <alp/reference/io.hpp>
#endif

#include "matrix.hpp"
#include "storage.hpp"


namespace alp {

	namespace internal {

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
			Density::Dense, OutputView, OutputImfR, OutputImfC, omp > &C,
			const alp::Matrix< InputType1, InputStructure1, 
			Density::Dense, InputView1, InputImfR1, InputImfC1, omp > &A,
			const alp::Matrix< InputType2, InputStructure2, 
			Density::Dense, InputView2, InputImfR2, InputImfC2, omp > &B,
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
			std::cout << "In alp::internal::mxm_generic (omp)\n";
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

			const Distribution &da = internal::getAmf( A ).getDistribution();
			const Distribution &db = internal::getAmf( B ).getDistribution();
			const Distribution &dc = internal::getAmf( C ).getDistribution();

			RC rc = SUCCESS;

			#pragma omp parallel for
			for( size_t thread = 0; thread < config::OMP::current_threads(); ++thread ) {

				const auto th_ijk_a = da.getThreadCoords( thread );
				const auto th_ijk_b = da.getThreadCoords( thread );
				const auto th_ijk_c = da.getThreadCoords( thread );

				const auto block_grid_dims_a = d.getLocalBlockGridDims( t_coords_a );
				const auto block_grid_dims_b = d.getLocalBlockGridDims( t_coords_b );
				const auto block_grid_dims_c = d.getLocalBlockGridDims( t_coords_c );

				RC local_rc = SUCCESS;

				// Broadcast Aij and Bij to all c layers
				if( t_coords.rt > 0 ) {

					typename Distribution::ThreadCoord th_ij0_a( t_coords_a.tr, t_coords_a.tc, 0 );

					for( size_t br = 0; br < block_grid_dims_a.first; ++br ) {
						for( size_t bc = 0; bc < block_grid_dims_a.second; ++bc ) {

							auto refAij0 = internal::get_view( A, th_ij0_a, br, bc );
							auto refAijk = internal::get_view( A, th_ijk_a, br, bc );

							local_rc = local_rc ? local_rc : set( refAijk, refAij0 );


						}
					}

					if( local_rc != SUCCESS ) {

						typename Distribution::ThreadCoord th_ij0_b( t_coords_b.tr, t_coords_b.tc, 0 );

						for( size_t br = 0; br < block_grid_dims_b.first; ++br ) {
							for( size_t bc = 0; bc < block_grid_dims_b.second; ++bc ) {

								auto refBij0 = internal::get_view( B, th_ij0_b, br, bc );
								auto refBijk = internal::get_view( B, th_ijk_b, br, bc );

								local_rc = local_rc ? local_rc : set( refBijk, refBij0 );
							}
						}
					}

				}
				// End Broadcast of Aij and Bij
				#pragma omp barrier


				// for( size_t br = 0; br < block_grid_dims.first; ++br ) {
				// 	for( size_t bc = 0; bc < block_grid_dims.second; ++bc ) {

				// 		// Get a sequential matrix view over the block
				// 		auto refC = internal::get_view( C, tr, tc, 1 /* rt */, br, bc );

				// 		// Construct a sequential Scalar container from the input Scalar
				// 		Scalar< InputType, InputStructure, config::default_sequential_backend > ref_val( *val );

				// 		// Delegate the call to the sequential set implementation
				// 		local_rc = local_rc ? local_rc : set( refC, ref_val );

				// 		if( local_rc != SUCCESS ) {
				// 			rc = local_rc;
				// 		}
				// 	}
				// }
			}

			internal::setInitialized( C, true );
			return rc;

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

} // end namespace ``alp''

#endif // end ``_H_ALP_OMP_BLAS3''

