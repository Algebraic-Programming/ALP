
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

#include <alp/descriptors.hpp>
#include <alp/structures.hpp>

#include <alp/base/blas3.hpp>

#include "matrix.hpp"
#include "storage.hpp"

// Include backend to which sequential work is delegated
#ifdef _ALP_OMP_WITH_REFERENCE
 #include <alp/reference/blas2.hpp>
 #include <alp/reference/blas3.hpp>
 #include <alp/reference/io.hpp>
#endif

#ifndef NDEBUG
#include "../../../tests/utils/print_alp_containers.hpp"
#endif


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
			alp::Matrix< InputType1, InputStructure1, 
			Density::Dense, InputView1, InputImfR1, InputImfC1, omp > &A,
			alp::Matrix< InputType2, InputStructure2, 
			Density::Dense, InputView2, InputImfR2, InputImfC2, omp > &B,
			const Operator &oper,
			const Monoid &monoid,
			const MulMonoid &mulMonoid,
			const std::enable_if_t< !alp::is_object< OutputType >::value &&
				!alp::is_object< InputType1 >::value && !
				alp::is_object< InputType2 >::value &&
				alp::is_operator< Operator >::value &&
				alp::is_monoid< Monoid >::value,
			void > * const = NULL
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

#ifndef NDEBUG
			std::cout << "In alp::internal::mxm_generic (omp)\n";
#endif

			// Early exit checks 
			if( ! internal::getInitialized( A ) || ! internal::getInitialized( B ) ||
				! internal::getInitialized( C )
			) {
				internal::setInitialized( C, false );
				return SUCCESS;
			}

			const std::ptrdiff_t m { static_cast< std::ptrdiff_t >( nrows( C ) ) };
			const std::ptrdiff_t n { static_cast< std::ptrdiff_t >( ncols( C ) ) };
			const std::ptrdiff_t m_a { static_cast< std::ptrdiff_t >( nrows( A ) ) };
			const std::ptrdiff_t k { static_cast< std::ptrdiff_t >( ncols( A ) ) };
			const std::ptrdiff_t k_b { static_cast< std::ptrdiff_t >( nrows( B ) ) };
			const std::ptrdiff_t n_b { static_cast< std::ptrdiff_t >( ncols( B ) ) };

			if( m != m_a || k != k_b || n != n_b ) {
				return MISMATCH;
			}

			const Distribution_2_5D &da = internal::getAmf( A ).getDistribution();
			const Distribution_2_5D &db = internal::getAmf( B ).getDistribution();
			const Distribution_2_5D &dc = internal::getAmf( C ).getDistribution();

			const auto tg_a = da.getThreadGridDims();
			const auto tg_b = db.getThreadGridDims();
			const auto tg_c = dc.getThreadGridDims();
			
			// 2.5D factor ranges within 3D limits
			if( tg_c.rt != tg_a.rt || tg_a.rt != tg_b.rt || ( tg_a.tc % tg_a.rt ) != 0 ) {
				return MISMATCH;
			}

			if( tg_c.tr != tg_a.tr || tg_c.tc != tg_b.tc || tg_a.tc != tg_b.tr ) {
				return MISMATCH;
			}

			using th_coord_t = typename Distribution_2_5D::ThreadCoords;

			RC rc = SUCCESS;

#ifndef NDEBUG
			#pragma omp critical
			std::cerr << "In alp::internal::mxm_generic (omp)\n"
				"\tBeginning parallel region" << std::endl;
#endif

			#pragma omp parallel
			{			
				const size_t thread = config::OMP::current_thread_ID();

				const th_coord_t th_ijk_a = da.getThreadCoords( thread );
				const th_coord_t th_ijk_b = db.getThreadCoords( thread );
				const th_coord_t th_ijk_c = dc.getThreadCoords( thread );

				RC local_rc = SUCCESS;

				// Broadcast A and B to all c-dimensional layers
				if( local_rc == SUCCESS && da.isActiveThread( th_ijk_a ) && th_ijk_a.rt > 0 ) {

#ifndef NDEBUG
					#pragma omp critical
					std::cerr << "Thread " << thread << " in alp::internal::mxm_generic (omp)\n"
						"\tCopying A." << std::endl;
#endif

					const auto set_block_grid_dims_a = da.getLocalBlockGridDims( th_ijk_a );

					th_coord_t th_ij0_a( th_ijk_a.tr, th_ijk_a.tc, 0 );

					for( size_t br = 0; br < set_block_grid_dims_a.first; ++br ) {
						for( size_t bc = 0; bc < set_block_grid_dims_a.second; ++bc ) {

							auto refAij0 = get_view( A, th_ij0_a, br, bc );
							auto refAijk = get_view( A, th_ijk_a, br, bc );

							local_rc = local_rc ? local_rc : set( refAijk, refAij0 );

						}
					}
				
				} // End Broadcast of A

				if( local_rc == SUCCESS && db.isActiveThread( th_ijk_b ) && th_ijk_b.rt > 0 ) {

#ifndef NDEBUG
					#pragma omp critical
					std::cerr << "Thread " << thread << " in alp::internal::mxm_generic (omp)\n"
						"\tCopying B." << std::endl;
#endif

					const auto set_block_grid_dims_b = db.getLocalBlockGridDims( th_ijk_b );

					th_coord_t th_ij0_b( th_ijk_b.tr, th_ijk_b.tc, 0 );

					for( size_t br = 0; br < set_block_grid_dims_b.first; ++br ) {
						for( size_t bc = 0; bc < set_block_grid_dims_b.second; ++bc ) {

							auto refBij0 = get_view( B, th_ij0_b, br, bc );
							auto refBijk = get_view( B, th_ijk_b, br, bc );

							local_rc = local_rc ? local_rc : set( refBijk, refBij0 );
						}
					}
				} // End Broadcast of B

				if( local_rc == SUCCESS && dc.isActiveThread( th_ijk_c ) && th_ijk_c.rt > 0 ) {

#ifndef NDEBUG
					#pragma omp critical
					std::cerr << "Thread " << thread << " in alp::internal::mxm_generic (omp)\n"
						"\tZeroing C." << std::endl;
#endif

					const auto block_grid_dims_c = dc.getLocalBlockGridDims( th_ijk_c );

					alp::Scalar< 
						OutputType, alp::structures::General, config::default_sequential_backend 
					> zero( monoid.template getIdentity< OutputType >() );

					for( size_t br = 0; br < block_grid_dims_c.first; ++br ) {
						for( size_t bc = 0; bc < block_grid_dims_c.second; ++bc ) {
							auto refCijk = get_view( C, th_ijk_c, br, bc );
							local_rc = local_rc ? local_rc : set( refCijk, zero );
						}
					}
				} // End Zero-ing of C

				// Different values for rc could converge here (eg, MISMATCH, FAILED).
				if( local_rc != SUCCESS ) {
#ifndef NDEBUG
					#pragma omp critical
					std::cerr << "Thread " << thread << " in alp::internal::mxm_generic (omp)\n"
						"\tIssues replicating input matrices." << std::endl;
#endif
					rc = local_rc;
				}

				// End Broadcast of A and B and zero-ing of C
				#pragma omp barrier

#ifndef NDEBUG
				#pragma omp critical
				std::cerr << "Thread " << thread << " in alp::internal::mxm_generic (omp)\n"
					"\tPassing barrier" << std::endl;
#endif

				if( rc == SUCCESS && dc.isActiveThread( th_ijk_c ) ) {

					const auto block_grid_dims_c = dc.getLocalBlockGridDims( th_ijk_c );

					// Initialize circular shifts at stride of Rt
					size_t c_a = utils::modulus( th_ijk_a.tc + th_ijk_a.tr + th_ijk_a.rt * tg_a.tc / tg_a.rt, tg_a.tc );
					size_t r_b = utils::modulus( th_ijk_b.tr + th_ijk_b.tc + th_ijk_b.rt * tg_b.tr / tg_b.rt, tg_b.tr );

					// Per-c-dimensional-layer partial computation
					for( size_t r = 0; r < tg_a.tc / tg_a.rt; ++r ) {

						const th_coord_t th_isk_a( th_ijk_a.tr, c_a, th_ijk_a.rt );
						const th_coord_t th_sjk_b( r_b, th_ijk_b.tc, th_ijk_b.rt );

#ifndef NDEBUG
						if( thread == 1 ) {
							#pragma omp critical
							{
								std::cerr << "Thread " << thread << " in alp::internal::mxm_generic (omp)\n"
									"\tCompute iteration r=" << r << "\n"
									"\tStarting from A( " << th_ijk_a.tr << ", " << c_a << ", " << th_ijk_a.rt << " )\n"
									"\tStarting from B( " << r_b << ", " << th_ijk_b.tc << ", " << th_ijk_b.rt << " )" <<std::endl;
							}
						}
#endif

						const auto mxm_block_grid_dims_a = da.getLocalBlockGridDims( th_isk_a );
						const auto mxm_block_grid_dims_b = db.getLocalBlockGridDims( th_sjk_b );

						if( block_grid_dims_c.first != mxm_block_grid_dims_a.first 
							|| block_grid_dims_c.second != mxm_block_grid_dims_b.second 
							|| mxm_block_grid_dims_a.second != mxm_block_grid_dims_b.first 
						) {
#ifndef NDEBUG
							#pragma omp critical
							std::cerr << "Thread " << thread << " in alp::internal::mxm_generic (omp)\n"
								"\tMismatching local block grid size on mxm." << std::endl;
#endif
							local_rc = MISMATCH;
						}

						if( local_rc == SUCCESS ) {

							for( size_t bk = 0; bk < mxm_block_grid_dims_a.second; ++bk ) {
								for( size_t br = 0; br < block_grid_dims_c.first; ++br ) {
	
									const auto refA_loc = get_view( A, th_isk_a, br, bk );

									for( size_t bc = 0; bc < block_grid_dims_c.second; ++bc ) {

										const auto refB_loc = get_view( B, th_sjk_b, bk, bc );
										auto refC_ijk = get_view( C, th_ijk_c, br, bc );

#ifndef NDEBUG
										if( thread == 1 ) {
											#pragma omp critical 
											{
												std::cerr << "Thread " << thread << " in alp::internal::mxm_generic (omp)\n"
													"\tIteration r=" << r << " computing: " << bk << " " << br << " " << bc << std::endl;
												print_matrix("Aref pre", refA_loc );
												print_matrix("Bref pre", refB_loc );
												print_matrix("Cref pre", refC_ijk );
											}
										}
#endif

										// Delegate the call to the sequential mxm implementation
										local_rc = local_rc ? local_rc : mxm_generic< allow_void >( refC_ijk, refA_loc, refB_loc, oper, monoid, mulMonoid );

#ifndef NDEBUG
										if( thread == 1 ) {
											#pragma omp critical 
											{
												print_matrix("Cref post", refC_ijk );
											}
										}
#endif

									}
								}
							}

							// Circular shift rightwards for A
							c_a = utils::modulus( c_a + 1, tg_a.tc );
							// Circular shift downwards for B
							r_b = utils::modulus( r_b + 1, tg_b.tr );

						} 

					} 

				} // End computation per layer of c-dimension

				// Different values for rc could converge here (eg, MISMATCH, FAILED).
				if( local_rc != SUCCESS ) {
#ifndef NDEBUG
					#pragma omp critical
					std::cerr << "Thread " << thread << " in alp::internal::mxm_generic (omp)\n"
						"\tIssues with local mxm computations." << std::endl;
#endif
					rc = local_rc;
				}

				// End layer-by-layer partial computation
				#pragma omp barrier

				// Final c-dimension reduction
				if( rc == SUCCESS && dc.isActiveThread( th_ijk_c ) && th_ijk_c.rt == 0 ) {
					
#ifndef NDEBUG
					#pragma omp critical
					std::cerr << "Thread " << thread << " in alp::internal::mxm_generic (omp)\n"
						"\tEntering reduction." << std::endl;
#endif

					const auto block_grid_dims_c = dc.getLocalBlockGridDims( th_ijk_c );

					for( size_t r = 1; r < tg_c.rt; ++r ) {

#ifndef NDEBUG
						#pragma omp critical
						std::cerr << "Thread " << thread << " in alp::internal::mxm_generic (omp)\n"
							"\tReduction iteration r=" << r << std::endl;
#endif

						const th_coord_t th_ijr_c( th_ijk_c.tr, th_ijk_c.tc, r );

						for( size_t br = 0; br < block_grid_dims_c.first; ++br ) {
							for( size_t bc = 0; bc < block_grid_dims_c.second; ++bc ) {

								auto refCij0 = internal::get_view( C, th_ijk_c, br, bc ); // k == 0 
								auto refCijr = internal::get_view( C, th_ijr_c, br, bc );

								// Final result in C at layer 0
								local_rc = local_rc ? local_rc : foldl( refCij0, refCijr, monoid );
							}
						}

					}
				
				} // End of final reduction along c-dimension
				
				// Different values for rc could converge here (eg, MISMATCH, FAILED).
				if( local_rc != SUCCESS ) {
#ifndef NDEBUG
					#pragma omp critical
					std::cerr << "Thread " << thread << " in alp::internal::mxm_generic (omp)\n"
						"\tIssues with final reduction." << std::endl;
#endif
					rc = local_rc;
				}

			}

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
		typename OutputType, typename InputType1, typename InputType2,
		typename OutputStructure, typename OutputView, 
		typename OutputImfR, typename OutputImfC,
		typename InputStructure1, typename InputView1, 
		typename InputImfR1, typename InputImfC1,
		typename InputStructure2, typename InputView2, 
		typename InputImfR2, typename InputImfC2,
		class Semiring
	>
	RC mxm( 
		alp::Matrix< OutputType, OutputStructure, 
		Density::Dense, OutputView, OutputImfR, OutputImfC, omp > &C,
		alp::Matrix< InputType1, InputStructure1, 
		Density::Dense, InputView1, InputImfR1, InputImfC1, omp > &A,
		alp::Matrix< InputType2, InputStructure2, 
		Density::Dense, InputView2, InputImfR2, InputImfC2, omp > &B,
		const Semiring & ring = Semiring(),
		const PHASE &phase = NUMERICAL,
		const std::enable_if_t< 
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_semiring< Semiring >::value,
			void 
		> * const = NULL 
	) {
		(void)phase;

		return internal::mxm_generic< false >( 
			C, A, B,
			ring.getMultiplicativeOperator(), ring.getAdditiveMonoid(), ring.getMultiplicativeMonoid()
		);

	}

	/**
	 * Dense Matrix-Matrix multiply between structured matrices.
	 * Version with additive monoid and multiplicative operator
	 */
	template< 
		typename OutputType, typename InputType1, typename InputType2,
		typename OutputStructure, typename OutputView, 
		typename OutputImfR, typename OutputImfC,
		typename InputStructure1, typename InputView1, 
		typename InputImfR1, typename InputImfC1,
		typename InputStructure2, typename InputView2, 
		typename InputImfR2, typename InputImfC2,
		class Operator, class Monoid
	>
	RC mxm( 
		alp::Matrix< OutputType, OutputStructure, 
		Density::Dense, OutputView, OutputImfR, OutputImfC, omp > &C,
		alp::Matrix< InputType1, InputStructure1, 
		Density::Dense, InputView1, InputImfR1, InputImfC1, omp > &A,
		alp::Matrix< InputType2, InputStructure2, 
		Density::Dense, InputView2, InputImfR2, InputImfC2, omp > &B,
		const Operator & mulOp,
		const Monoid & addM,
		const PHASE &phase = NUMERICAL,
		const std::enable_if_t< 
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_operator< Operator >::value && alp::is_monoid< Monoid >::value,
			void 
		> * const = NULL 
	) {
		(void)phase;

		return internal::mxm_generic< false >( C, A, B, mulOp, addM, Monoid() );
	
	}

} // end namespace ``alp''

#endif // end ``_H_ALP_OMP_BLAS3''

