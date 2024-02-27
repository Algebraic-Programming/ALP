
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

#if ! defined _H_GRB_REFERENCE_BLAS3 || defined _H_GRB_REFERENCE_OMP_BLAS3
#define _H_GRB_REFERENCE_BLAS3

#include <type_traits> //for std::enable_if

#include <graphblas/base/blas3.hpp>
#include <graphblas/utils/iterators/matrixVectorIterator.hpp>

#include "io.hpp"
#include "matrix.hpp"

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
 #include <omp.h>
#endif

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
			Descriptor descr,
			class Monoid,
			class Operator,
			class MulMonoid,
			typename OutputType, typename InputType1, typename InputType2,
			typename RIT1, typename CIT1, typename NIT1,
			typename RIT2, typename CIT2, typename NIT2,
			typename RIT3, typename CIT3, typename NIT3
		>
		RC mxm_generic(
			Matrix< OutputType, reference, RIT1, CIT1, NIT1 > &C,
			const Matrix< InputType1, reference, RIT2, CIT2, NIT2 > &A,
			const Matrix< InputType2, reference, RIT3, CIT3, NIT3 > &B,
			const Operator &oper,
			const Monoid &monoid,
			const MulMonoid &mulMonoid,
			const Phase &phase,
			const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_operator< Operator >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = nullptr
		) {
			static_assert( allow_void ||
				( !(
					std::is_same< InputType1, void >::value ||
					std::is_same< InputType2, void >::value
				) ),
				"grb::mxm_generic: the operator-monoid version of mxm cannot be "
				"used if either of the input matrices is a pattern matrix (of type "
				"void)"
			);

#ifdef _DEBUG
			std::cout << "In grb::internal::mxm_generic (reference, unmasked)\n";
#endif

			// get whether the matrices should be transposed prior to execution
			constexpr bool trans_left = descr & descriptors::transpose_left;
			constexpr bool trans_right = descr & descriptors::transpose_right;

			// get whether we are required to stick to CRS
			constexpr bool crs_only = descr & descriptors::force_row_major;

			// static checks
			static_assert( !(crs_only && trans_left), "Cannot (presently) transpose A "
				"and force the use of CRS" );
			static_assert( !(crs_only && trans_right), "Cannot (presently) transpose B "
				"and force the use of CRS" );

			// run-time checks
			const size_t m = grb::nrows( C );
			const size_t n = grb::ncols( C );
			const size_t m_A = !trans_left ? grb::nrows( A ) : grb::ncols( A );
			const size_t k = !trans_left ? grb::ncols( A ) : grb::nrows( A );
			const size_t k_B = !trans_right ? grb::nrows( B ) : grb::ncols( B );
			const size_t n_B = !trans_right ? grb::ncols( B ) : grb::nrows( B );
			assert( phase != TRY );

			if( m != m_A || k != k_B || n != n_B ) {
				return MISMATCH;
			}

			const auto &A_raw = !trans_left
				? internal::getCRS( A )
				: internal::getCCS( A );
			const auto &B_raw = !trans_right
				? internal::getCRS( B )
				: internal::getCCS( B );
			auto &C_raw = internal::getCRS( C );
			auto &CCS_raw = internal::getCCS( C );

			char * arr = nullptr;
			char * buf = nullptr;
			OutputType * valbuf = nullptr;
			internal::getMatrixBuffers( arr, buf, valbuf, 1, C );
			config::NonzeroIndexType * C_col_index = internal::template
				getReferenceBuffer< typename config::NonzeroIndexType >( n + 1 );

			// initialisations
			internal::Coordinates< reference > coors;
			coors.set( arr, false, buf, n );

			if( !crs_only ) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp parallel
				{
					size_t start, end;
					config::OMP::localRange( start, end, 0, n + 1 );
#else
					const size_t start = 0;
					const size_t end = n + 1;
#endif
					for( size_t j = start; j < end; ++j ) {
						CCS_raw.col_start[ j ] = 0;
					}
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				}
#endif
			}
			// end initialisations

			// symbolic phase (counting sort, step 1)
			size_t nzc = 0; // output nonzero count
			if( crs_only && phase == RESIZE ) {
				// we are using an auxialiary CRS that we cannot resize ourselves
				// instead, we update the offset array only
				C_raw.col_start[ 0 ] = 0;
			}
			// if crs_only, then the below implements its resize phase
			// if not crs_only, then the below is both crucial for the resize phase,
			// as well as for enabling the insertions of output values in the output CCS
			if( (crs_only && phase == RESIZE) || !crs_only ) {
				for( size_t i = 0; i < m; ++i ) {
					coors.clear();
					for( auto k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
						const size_t k_col = A_raw.row_index[ k ];
						for(
							auto l = B_raw.col_start[ k_col ];
							l < B_raw.col_start[ k_col + 1 ];
							++l
						) {
							const size_t l_col = B_raw.row_index[ l ];
							if( !coors.assign( l_col ) ) {
								(void) ++nzc;
								if( !crs_only ) {
									(void) ++CCS_raw.col_start[ l_col + 1 ];
								}
							}
						}
					}
					if( crs_only && phase == RESIZE ) {
						// we are using an auxialiary CRS that we cannot resize ourselves
						// instead, we update the offset array only
						C_raw.col_start[ i + 1 ] = nzc;
					}
				}
			}

			if( phase == RESIZE ) {
				if( !crs_only ) {
					// do final resize
					const RC ret = grb::resize( C, nzc );
					return ret;
				} else {
					// we are using an auxiliary CRS that we cannot resize
					// instead, we updated the offset array in the above and can now exit
					return SUCCESS;
				}
			}

			// computational phase
			assert( phase == EXECUTE );
			if( grb::capacity( C ) < nzc ) {
#ifdef _DEBUG
				std::cerr << "\t not enough capacity to execute requested operation\n";
#endif
				const RC clear_rc = grb::clear( C );
				if( clear_rc != SUCCESS ) {
					return PANIC;
				} else {
					return FAILED;
				}
			}

			// prefix sum for C_col_index,
			// set CCS_raw.col_start to all zero
#ifndef NDEBUG
			if( !crs_only ) {
				assert( CCS_raw.col_start[ 0 ] == 0 );
			}
#endif
			C_col_index[ 0 ] = 0;
			for( size_t j = 1; j < n; ++j ) {
				if( !crs_only ) {
					CCS_raw.col_start[ j + 1 ] += CCS_raw.col_start[ j ];
				}
				C_col_index[ j ] = 0;
			}
#ifndef NDEBUG
			if( !crs_only ) {
				assert( CCS_raw.col_start[ n ] == nzc );
			}
#endif

#ifndef NDEBUG
			const size_t old_nzc = nzc;
#endif
			// use previously computed CCS offset array to update CCS during the
			// computational phase
			nzc = 0;
			C_raw.col_start[ 0 ] = 0;
			for( size_t i = 0; i < m; ++i ) {
				coors.clear();
				for( auto k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
					const size_t k_col = A_raw.row_index[ k ];
					for( auto l = B_raw.col_start[ k_col ];
						l < B_raw.col_start[ k_col + 1 ];
						++l
					) {
						const size_t l_col = B_raw.row_index[ l ];
#ifdef _DEBUG
						std::cout << "\t A( " << i << ", " << k_col << " ) = "
							<< A_raw.getValue( k,
								mulMonoid.template getIdentity< typename Operator::D1 >() )
							<< " will be multiplied with B( " << k_col << ", " << l_col << " ) = "
							<< B_raw.getValue( l,
								mulMonoid.template getIdentity< typename Operator::D2 >() )
							<< " to accumulate into C( " << i << ", " << l_col << " )\n";
#endif
						if( !coors.assign( l_col ) ) {
							valbuf[ l_col ] = monoid.template getIdentity< OutputType >();
							(void) grb::apply( valbuf[ l_col ],
								A_raw.getValue( k,
									mulMonoid.template getIdentity< typename Operator::D1 >() ),
								B_raw.getValue( l,
									mulMonoid.template getIdentity< typename Operator::D2 >() ),
								oper );
						} else {
							OutputType temp = monoid.template getIdentity< OutputType >();
							(void) grb::apply( temp,
								A_raw.getValue( k,
									mulMonoid.template getIdentity< typename Operator::D1 >() ),
								B_raw.getValue( l,
									mulMonoid.template getIdentity< typename Operator::D2 >() ),
								oper );
							(void) grb::foldl( valbuf[ l_col ], temp, monoid.getOperator() );
						}
					}
				}
				for( size_t k = 0; k < coors.nonzeroes(); ++k ) {
					assert( nzc < old_nzc );
					const size_t j = coors.index( k );
					// update CRS
					C_raw.row_index[ nzc ] = j;
					C_raw.setValue( nzc, valbuf[ j ] );
					// update CCS
					if( !crs_only ) {
						const size_t CCS_index = C_col_index[ j ]++ + CCS_raw.col_start[ j ];
						CCS_raw.row_index[ CCS_index ] = i;
						CCS_raw.setValue( CCS_index, valbuf[ j ] );
					}
					// update count
					(void) ++nzc;
				}
				C_raw.col_start[ i + 1 ] = nzc;
			}

#ifndef NDEBUG
			if( !crs_only ) {
				for( size_t j = 0; j < n; ++j ) {
					assert( CCS_raw.col_start[ j + 1 ] - CCS_raw.col_start[ j ] ==
						C_col_index[ j ] );
				}
			}
			assert( nzc == old_nzc );
#endif

			// set final number of nonzeroes in output matrix
			internal::setCurrentNonzeroes( C, nzc );

			// done
			return SUCCESS;
		}

	} // end namespace grb::internal

	/**
	 * \internal grb::mxm, semiring version.
	 * Dispatches to internal::mxm_generic
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2,
		typename RIT3, typename CIT3, typename NIT3,
		class Semiring
	>
	RC mxm(
		Matrix< OutputType, reference, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType1, reference, RIT2, CIT2, NIT2 > &A,
		const Matrix< InputType2, reference, RIT3, CIT3, NIT3 > &B,
		const Semiring &ring = Semiring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Semiring >::value,
		void >::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Semiring::D1, InputType1 >::value
			), "grb::mxm",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Semiring::D2, InputType2 >::value ), "grb::mxm",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Semiring::D4, OutputType >::value
			), "grb::mxm",
			"called with an output matrix C that does not match the output domain "
			"of the given operator" );

#ifdef _DEBUG
		std::cout << "In grb::mxm (reference, unmasked, semiring)\n";
#endif

		return internal::mxm_generic< true, descr >(
			C, A, B,
			ring.getMultiplicativeOperator(),
			ring.getAdditiveMonoid(),
			ring.getMultiplicativeMonoid(),
			phase
		);
	}

	/**
	 * \internal mxm implementation with additive monoid and multiplicative operator
	 * Dispatches to internal::mxm_generic
	 */
	template<
		Descriptor descr = grb::descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2,
		typename RIT3, typename CIT3, typename NIT3,
		class Operator, class Monoid
	>
	RC mxm(
		Matrix< OutputType, reference, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType1, reference, RIT2, CIT2, NIT2 > &A,
		const Matrix< InputType2, reference, RIT3, CIT3, NIT3 > &B,
		const Monoid &addM,
		const Operator &mulOp,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Operator::D1, InputType1 >::value
			), "grb::mxm",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the given multiplication operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Operator::D2, InputType2 >::value
			), "grb::mxm",
			"called with a postfactor input matrix B that does not match the first "
			"domain of the given multiplication operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Operator::D3, OutputType >::value ),
			"grb::mxm",
			"called with an output matrix C that does not match the output domain "
			"of the given multiplication operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Monoid::D1, typename Operator::D3 >::value
			), "grb::mxm",
			"the output domain of the multiplication operator does not match the "
			"first domain of the given addition monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Monoid::D2, OutputType >::value
			), "grb::mxm",
			"the second domain of the given addition monoid does not match the "
			"type of the output matrix C" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Monoid::D3, OutputType >::value
			), "grb::mxm",
			"the output type of the given addition monoid does not match the type "
			"of the output matrix C" );
		static_assert( ( !(
				std::is_same< InputType1, void >::value ||
				std::is_same< InputType2, void >::value
			) ),
			"grb::mxm: the operator-monoid version of mxm cannot be used if either "
			"of the input matrices is a pattern matrix (of type void)" );

		return internal::mxm_generic< false, descr >(
			C, A, B, mulOp, addM, Monoid(), phase
		);
	}

	namespace internal {

		template<
			Descriptor descr = descriptors::no_operation,
			bool matrix_is_void,
			typename OutputType, typename InputType1,
			typename InputType2, typename InputType3,
			typename RIT, typename CIT, typename NIT,
			typename Coords
		>
		RC matrix_zip_generic(
			Matrix< OutputType, reference, RIT, CIT, NIT > &A,
			const Vector< InputType1, reference, Coords > &x,
			const Vector< InputType2, reference, Coords > &y,
			const Vector< InputType3, reference, Coords > &z,
			const Phase &phase
		) {
			assert( !(descr & descriptors::force_row_major) );
#ifdef _DEBUG
			std::cout << "In matrix_zip_generic (reference, vectors-to-matrix)\n";
#endif
			assert( phase != TRY );
			assert( nnz( x ) == nnz( y ) );
			assert( nnz( x ) == nnz( z ) );
			if( phase == RESIZE ) {
				return resize( A, nnz( x ) );
			}
			assert( phase == EXECUTE );
			const RC clear_rc = clear( A );
			if( nnz( x ) > capacity( A ) ) {
#ifdef _DEBUG
				std::cout << "\t output matrix did not have sufficient capacity to "
					<< "complete the requested computation\n";
#endif
				if( clear_rc == SUCCESS ) {
					return FAILED;
				} else {
					return PANIC;
				}
			} else if( clear_rc != SUCCESS ) {
				return clear_rc;
			}

			auto x_it = x.cbegin();
			auto y_it = y.cbegin();
			auto z_it = z.cbegin();
			const auto x_end = x.cend();
			const auto y_end = y.cend();
			const auto z_end = z.cend();
			const size_t nrows = grb::nrows( A );
			const size_t ncols = grb::ncols( A );
			const size_t nmins = nrows < ncols ? nrows : ncols;

			assert( grb::nnz( A ) == 0 );

			auto &crs = internal::getCRS( A );
			auto &ccs = internal::getCCS( A );
			auto * __restrict__ crs_offsets = crs.getOffsets();
			auto * __restrict__ crs_indices = crs.getIndices();
			auto * __restrict__ ccs_offsets = ccs.getOffsets();
			auto * __restrict__ ccs_indices = ccs.getIndices();
			auto * __restrict__ crs_values = crs.getValues();
			auto * __restrict__ ccs_values = ccs.getValues();

			RC ret = SUCCESS;

			// step 1: reset matrix storage

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp parallel
#endif
			{
				size_t start, end;
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				config::OMP::localRange( start, end, 0, nmins );
#else
				start = 0;
				end = nmins;
#endif
				for( size_t i = start; i < end; ++i ) {
					crs_offsets[ i ] = ccs_offsets[ i ] = 0;
				}
				assert( nrows >= nmins );
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				config::OMP::localRange( start, end, 0, nrows - nmins );
#else
				start = 0;
				end = nrows - nmins;
#endif
				for( size_t i = nmins + start; i < nmins + end; ++i ) {
					crs_offsets[ i ] = 0;
				}
				assert( ncols >= nmins );
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				config::OMP::localRange( start, end, 0, ncols - nmins );
#else
				start = 0;
				end = ncols - nmins;
#endif
				for( size_t i = nmins + start; i < nmins + end; ++i ) {
					ccs_offsets[ i ] = 0;
				}
			}

			// step 2: counting sort, phase one

			// TODO internal issue #64
			for( ; x_it != x_end; ++x_it ) {
				assert( x_it->second < nrows );
				(void) ++( crs_offsets[ x_it->second ] );
			}
			// TODO internal issue #64
			for( ; y_it != y_end; ++y_it ) {
				assert( y_it->second < ncols );
				(void) ++( ccs_offsets[ y_it->second ] );
			}

			// step 3: perform prefix-sum on row- and column-counts

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
	#pragma omp parallel
	{
			const size_t T = omp_get_num_threads();
#else
			const size_t T = 1;
#endif
			assert( nmins > 0 );
			size_t start, end;
			config::OMP::localRange( start, end, 0, nmins );
			(void) ++start;
			for( size_t i = start; i < end; ++i ) {
				crs_offsets[ i ] += crs_offsets[ i - 1 ];
				ccs_offsets[ i ] += ccs_offsets[ i - 1 ];
			}
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp barrier
#endif
			assert( nrows >= nmins );
			config::OMP::localRange( start, end, 0, nrows - nmins );
			for( size_t i = nmins + start; i < nmins + end; ++i ) {
				crs_offsets[ i ] += crs_offsets[ i - 1 ];
			}
			assert( ncols >= nmins );
			config::OMP::localRange( start, end, 0, ncols - nmins );
			for( size_t i = nmins + start; i < nmins + end; ++i ) {
				ccs_offsets[ i ] += ccs_offsets[ i - 1 ];
			}
			assert( T > 0 );
			for( size_t k = T - 1; k > 0; --k ) {
				config::OMP::localRange( start, end, 0, nrows,
					config::CACHE_LINE_SIZE::value(), k, T
				);
				assert( start > 0 );
				// note: in the below, the end of the subloop is indeed nrows, not end(!)
				size_t subloop_start, subloop_end;
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				config::OMP::localRange( subloop_start, subloop_end, start, nrows );
				#pragma omp barrier
#else
				subloop_start = start;
				subloop_end = nrows;
#endif
				for( size_t i = subloop_start; i < subloop_end; ++i ) {
					crs_offsets[ i ] += crs_offsets[ start - 1 ];
				}
				config::OMP::localRange( start, end, 0, ncols,
					config::CACHE_LINE_SIZE::value(), k, T
				);
				assert( start > 0 );
				// note: in the below, the end of the subloop is indeed ncols, not end(!)
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				config::OMP::localRange( subloop_start, subloop_end, start, ncols );
#else
				subloop_start = start;
				subloop_end = ncols;
#endif
				for( size_t i = subloop_start; i < subloop_end; ++i ) {
					ccs_offsets[ i ] += ccs_offsets[ start - 1 ];
				}
			}
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
	} // end parallel
#endif

			crs_offsets[ nrows ] = crs_offsets[ nrows - 1 ];
			ccs_offsets[ ncols ] = ccs_offsets[ ncols - 1 ];

			// step 4, check nonzero capacity
			assert( crs_offsets[ nrows ] == ccs_offsets[ ncols ] );
			if( internal::getNonzeroCapacity( A ) < crs_offsets[ nrows ] ) {
				return FAILED;
			}

			// step 5, counting sort, second and final ingestion phase
			x_it = x.cbegin();
			y_it = y.cbegin();
			// TODO internal issue #64
			for( ; x_it != x_end; ++x_it, ++y_it ) {
				if( ret == SUCCESS && x_it->first != y_it->first ) {
					ret = ILLEGAL;
				}
				if( !matrix_is_void && ret == SUCCESS && ( x_it->first != z_it->first ) ) {
					ret = ILLEGAL;
				}
				assert( x_it->second < nrows );
				assert( y_it->second < ncols );
				const size_t crs_pos = --( crs_offsets[ x_it->second ] );
				const size_t ccs_pos = --( ccs_offsets[ y_it->second ] );
				assert( crs_pos < crs_offsets[ nrows ] );
				assert( ccs_pos < ccs_offsets[ ncols ] );
				crs_indices[ crs_pos ] = y_it->second;
				ccs_indices[ ccs_pos ] = x_it->second;
				if( !matrix_is_void ) {
					crs_values[ crs_pos ] = ccs_values[ ccs_pos ] = z_it->second;
					(void) ++z_it;
				}
			}

			if( ret == SUCCESS ) {
				internal::setCurrentNonzeroes( A, crs_offsets[ nrows ] );
			}

			// check all inputs are handled
			assert( x_it == x_end );
			assert( y_it == y_end );
			if( !matrix_is_void ) {
				assert( z_it == z_end );
			} else {
				(void) z_end;
			}

			// finally, some (expensive) debug checks on the output matrix
			assert( crs_offsets[ nrows ] == ccs_offsets[ ncols ] );
#ifndef NDEBUG
			for( size_t j = 0; j < ncols; ++j ) {
				for( size_t k = ccs_offsets[ j ]; k < ccs_offsets[ j + 1 ]; ++k ) {
					assert( k < ccs_offsets[ ncols ] );
					assert( ccs_indices[ k ] < nrows );
				}
			}
			for( size_t i = 0; i < nrows; ++i ) {
				for( size_t k = crs_offsets[ i ]; k < crs_offsets[ i + 1 ]; ++k ) {
					assert( k < crs_offsets[ nrows ] );
					assert( crs_indices[ k ] < ncols );
				}
			}
#endif

			// done
			return ret;
		}

	} // namespace internal

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1,
		typename InputType2, typename InputType3,
		typename RIT, typename CIT, typename NIT,
		typename Coords
	>
	RC zip(
		Matrix< OutputType, reference, RIT, CIT, NIT > &A,
		const Vector< InputType1, reference, Coords > &x,
		const Vector< InputType2, reference, Coords > &y,
		const Vector< InputType3, reference, Coords > &z,
		const Phase &phase = EXECUTE
	) {
		static_assert( !(descr & descriptors::no_casting) ||
				std::is_integral< InputType1 >::value,
			"grb::zip (two vectors to matrix) called "
			"using non-integral left-hand vector elements" );
		static_assert( !(descr & descriptors::no_casting) ||
				std::is_integral< InputType2 >::value,
			"grb::zip (two vectors to matrix) called "
			"using non-integral right-hand vector elements" );
		static_assert( !(descr & descriptors::no_casting) ||
				std::is_same< OutputType, InputType3 >::value,
			"grb::zip (two vectors to matrix) called "
			"with differing vector nonzero and output matrix domains" );

		const size_t n = grb::size( x );
		const size_t nz = grb::nnz( x );
		const RC ret = grb::clear( A );
		if( ret != SUCCESS ) {
			return ret;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( n != grb::size( z ) ) {
			return MISMATCH;
		}
		if( nz != grb::nnz( y ) ) {
			return ILLEGAL;
		}
		if( nz != grb::nnz( z ) ) {
			return ILLEGAL;
		}

		return internal::matrix_zip_generic< descr, false >( A, x, y, z, phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType1, typename InputType2,
		typename RIT, typename CIT, typename NIT,
		typename Coords
	>
	RC zip(
		Matrix< void, reference, RIT, CIT, NIT > &A,
		const Vector< InputType1, reference, Coords > &x,
		const Vector< InputType2, reference, Coords > &y,
		const Phase &phase = EXECUTE
	) {
		static_assert( !(descr & descriptors::no_casting) ||
				std::is_integral< InputType1 >::value,
			"grb::zip (two vectors to void matrix) called using non-integral "
			"left-hand vector elements" );
		static_assert( !(descr & descriptors::no_casting) ||
				std::is_integral< InputType2 >::value,
			"grb::zip (two vectors to void matrix) called using non-integral "
			"right-hand vector elements" );

		const size_t n = grb::size( x );
		const size_t nz = grb::nnz( x );
		const RC ret = grb::clear( A );
		if( ret != SUCCESS ) {
			return ret;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( nz != grb::nnz( y ) ) {
			return ILLEGAL;
		}

		return internal::matrix_zip_generic< descr, true >( A, x, y, x, phase );
	}

	/**
	 * Selecting a submatrix of matrix \a B based on the given vector of \a rows and \a cols
	 * Depends on the masked outer product and masked set. Currently, only the structural version is supported.
	 */

	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename InputType, typename MaskType, typename OutputType,
		typename Coords,
		typename RIT, typename CIT, typename NIT
	>
	RC selectSubmatrix(
		Matrix< OutputType, reference, RIT, CIT, NIT > &B,
		const Matrix< InputType, reference, RIT, CIT, NIT > &A,
		const Vector< MaskType, reference, Coords > &rows,
		const Vector< MaskType, reference, Coords > &cols,
		const typename std::enable_if<
			grb::is_operator< Operator >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< OutputType >::value,
			void >::type * const = nullptr
	) {
		//static asserts
		static_assert(
			! ( descr & descriptors::transpose_matrix ),
			"grb::selectSubmatrix can not be called with descriptors::transpose_matrix"
		);
		static_assert(
			( descr & descriptors::structural ),
			"Only the structural version is supported for grb::selectSubmatrix"
		);
#ifdef _DEBUG
		std::cout << "In grb::selectSubmatrix (reference)\n";
#endif

		const size_t nrows = size( rows );
		const size_t ncols = size( cols );
		if( nrows != grb::nrows( A ) || nrows != grb::nrows( B ) ) {
			return MISMATCH;
		}

		if( ncols != grb::ncols( A ) || ncols != grb::ncols( B ) ) {
			return MISMATCH;
		}

		//mask contains only those values that need to be selected from A
		Matrix< MaskType, reference, RIT, CIT, NIT > mask( nrows, ncols );

		RC ret = outer( mask, A, rows, cols, grb::operators::zip< MaskType, MaskType, reference >(), Phase::RESIZE );
		if( ret != SUCCESS ) {
			return ret;
		}
		ret = outer( mask, A, rows, cols, grb::operators::zip< MaskType, MaskType, reference >() );
		if( ret != SUCCESS ) {
			return ret;
		}

		ret = set( B, mask, A, Phase::RESIZE );
		if( ret != SUCCESS ) {
			return ret;
		}
		ret = set( B, mask, A );
		return ret;
	}



	/**
	 * Outer product of two vectors. Assuming vectors \a u and \a v are oriented
	 * column-wise, the result matrix \a A will contain \f$ uv^T \f$. This is an
	 * out-of-place function and will be updated soon to be in-place instead.
	 *
	 * \internal Implemented via mxm as a multiplication of a column vector with
	 *           a row vector.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords,
		typename RIT, typename CIT, typename NIT
	>
	RC outer(
		Matrix< OutputType, reference, RIT, CIT, NIT > &A,
		const Vector< InputType1, reference, Coords > &u,
		const Vector< InputType2, reference, Coords > &v,
		const Operator &mul = Operator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_operator< Operator >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< OutputType >::value,
			void >::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Operator::D1, InputType1 >::value
			), "grb::outerProduct",
			"called with a prefactor vector that does not match the first domain "
			"of the given multiplication operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Operator::D2, InputType2 >::value
			), "grb::outerProduct",
			"called with a postfactor vector that does not match the first domain "
			"of the given multiplication operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Operator::D3, OutputType >::value
			), "grb::outerProduct",
			"called with an output matrix that does not match the output domain of "
			"the given multiplication operator" );
#ifdef _DEBUG
		std::cout << "In grb::outer (reference)\n";
#endif

		const size_t nrows = size( u );
		const size_t ncols = size( v );

		assert( phase != TRY );
		if( nrows != grb::nrows( A ) ) {
			return MISMATCH;
		}

		if( ncols != grb::ncols( A ) ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return resize( A, nnz( u ) * nnz( v ) );
		}

		assert( phase == EXECUTE );
		if( capacity( A ) < nnz( u ) * nnz( v ) ) {
#ifdef _DEBUG
			std::cout << "\t insufficient capacity to complete "
				"requested outer-product computation\n";
#endif
			const RC clear_rc = clear( A );
			if( clear_rc != SUCCESS ) {
				return PANIC;
			} else {
				return FAILED;
			}
		}

		grb::Matrix< InputType1, reference > u_matrix( nrows, 1 );
		grb::Matrix< InputType2, reference > v_matrix( 1, ncols );

		auto u_converter = grb::utils::makeVectorToMatrixConverter< InputType1 >(
			u, []( const size_t &ind, const InputType1 &val ) {
				return std::make_pair( std::make_pair( ind, 0 ), val );
			} );

		grb::buildMatrixUnique(
			u_matrix,
			u_converter.begin(), u_converter.end(),
			PARALLEL
		);

		auto v_converter = grb::utils::makeVectorToMatrixConverter< InputType2 >(
			v, []( const size_t &ind, const InputType2 &val ) {
				return std::make_pair( std::make_pair( 0, ind ), val );
			} );

		grb::buildMatrixUnique(
			v_matrix,
			v_converter.begin(), v_converter.end(),
			PARALLEL
		);

		grb::Monoid<
			grb::operators::left_assign< OutputType >,
			grb::identities::zero
		> mono;

		RC ret = SUCCESS;
		if( phase == EXECUTE ) {
			ret = grb::clear( A );
		}
		assert( nnz( A ) == 0 );
		ret = ret ? ret : grb::mxm( A, u_matrix, v_matrix, mono, mul, phase );
		return ret;
	}


	/**
	 * A masked outer product of two vectors. Assuming vectors \a u and \a v are oriented
	 * column-wise, the result matrix \a A will contain \f$ uv^T \f$, masked to non-zero values from \a mask.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename InputType1, typename InputType2,
		typename MaskType, typename OutputType,
		typename Coords,
		typename RIT, typename CIT, typename NIT
	>
	RC outer(
		Matrix< OutputType, reference, RIT, CIT, NIT > &A,
		const Matrix< MaskType, reference, RIT, CIT, NIT > &mask,
		const Vector< InputType1, reference, Coords > &u,
		const Vector< InputType2, reference, Coords > &v,
		const Operator &mul = Operator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_operator< Operator >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< OutputType >::value,
			void >::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Operator::D1, InputType1 >::value
			), "grb::outer",
			"called with a prefactor vector that does not match the first domain "
			"of the given multiplication operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Operator::D2, InputType2 >::value
			), "grb::outer",
			"called with a postfactor vector that does not match the first domain "
			"of the given multiplication operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Operator::D3, OutputType >::value
			), "grb::outer",
			"called with an output matrix that does not match the output domain of "
			"the given multiplication operator" );
		static_assert(
			! ( descr & descriptors::structural && descr & descriptors::invert_mask ),
			"grb::outer can not be called with both descriptors::structural "
			"and descriptors::invert_mask in the masked variant"
		);
#ifdef _DEBUG
		std::cout << "In grb::outer (reference)\n";
#endif

		const size_t nrows = size( u );
		const size_t ncols = size( v );

		const size_t m = grb::nrows( mask );
		const size_t n = grb::ncols( mask );

		if( m == 0 || n == 0 ) {
			// If the mask has a null size, it will be ignored
			return outer< descr >( A, u, v, mul, phase );
		}

		constexpr bool crs_only = descr & descriptors::force_row_major;

		assert( phase != TRY );
		if( nrows != grb::nrows( A ) || nrows != m ) {
			return MISMATCH;
		}

		if( ncols != grb::ncols( A ) || ncols != n ) {
			return MISMATCH;
		}

		if( nnz( u ) == 0 || nnz( v ) == 0 ) {
			clear( A );
			return SUCCESS;
		}


		const auto &mask_raw = internal::getCRS( mask );

		char * mask_arr = nullptr;
		char * mask_buf = nullptr;
		MaskType * mask_valbuf = nullptr;
		internal::getMatrixBuffers( mask_arr, mask_buf, mask_valbuf, 1, mask );

		internal::Coordinates< reference > mask_coors;
		mask_coors.set( mask_arr, false, mask_buf, ncols );

		size_t nzc = 0;

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
		#pragma omp parallel for reduction(+:nzc)
#endif
		for( size_t i = 0; i < nrows; ++i ) {
			if( internal::getCoordinates( u ).assigned( i ) ) {
				for( auto k = mask_raw.col_start[ i ]; k < mask_raw.col_start[ i + 1 ]; ++k ) {
					const auto k_col = mask_raw.row_index[ k ];
					if( 
						internal::getCoordinates( v ).assigned( k_col ) && 
						utils::interpretMatrixMask< descr, MaskType >( true, mask_raw.getValues(), k ) 
					) {

						nzc++;
					}
				}
			}
		}

		if( phase == RESIZE ) {
			return resize( A, nzc );
		}

		assert( phase == EXECUTE );
		if( capacity( A ) < nzc ) {
#ifdef _DEBUG
			std::cout << "\t insufficient capacity to complete "
				"requested masked outer-product computation\n";
#endif
			const RC clear_rc = clear( A );
			if( clear_rc != SUCCESS ) {
				return PANIC;
			} else {
				return FAILED;
			}
		}

		RC ret = SUCCESS;
		if( phase == EXECUTE ) {
			ret = grb::clear( A );
		}
		assert( nnz( A ) == 0 );

		auto &CRS_raw = internal::getCRS( A );
		auto &CCS_raw = internal::getCCS( A );

		const InputType1 * __restrict__ const x = internal::getRaw( u );
		const InputType2 * __restrict__ const y = internal::getRaw( v );
		config::NonzeroIndexType * A_col_index = internal::template
			getReferenceBuffer< typename config::NonzeroIndexType >( ncols + 1 );

		CRS_raw.col_start[ 0 ] = 0;

		if( !crs_only ) {

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp parallel for simd
#endif
			for( size_t j = 0; j <= ncols; ++j ) {
				CCS_raw.col_start[ j ] = 0;
			}
		}


		nzc = 0;

		for( size_t i = 0; i < nrows; ++i ) {
			if( internal::getCoordinates( u ).assigned( i ) ) {
				for( auto k = mask_raw.col_start[ i ]; k < mask_raw.col_start[ i + 1 ]; ++k ) {
					const auto k_col = mask_raw.row_index[ k ];
					if( 
						internal::getCoordinates( v ).assigned( k_col ) && 
						utils::interpretMatrixMask< descr, MaskType >( true, mask_raw.getValues(), k )
					) {
						nzc++;
						if( !crs_only ) {
							CCS_raw.col_start[ k_col + 1 ]++;
						}
					}
				}
			}
			CRS_raw.col_start[ i + 1 ] = nzc;
		}

		if( !crs_only ) {
			for( size_t j = 1; j < ncols; ++j ) {
				CCS_raw.col_start[ j + 1 ] += CCS_raw.col_start[ j ];
			}


#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp parallel for simd
#endif
			for( size_t j = 0; j < ncols; ++j ) {
				A_col_index[ j ] = 0;
			}
		}

		// use previously computed CCS offset array to update CCS during the
		// computational phase
		nzc = 0;
		for( size_t i = 0; i < nrows; ++i ) {
			if( internal::getCoordinates( u ).assigned( i ) ) {
				for( auto k = mask_raw.col_start[ i ]; k < mask_raw.col_start[ i + 1 ]; ++k ) {
					const auto k_col = mask_raw.row_index[ k ];
					if( 
						internal::getCoordinates( v ).assigned( k_col ) &&
						utils::interpretMatrixMask< descr, MaskType >( true, mask_raw.getValues(), k )
					) {
						OutputType val;
						grb::apply( val,
							x[ i ],
							y[ k_col ],
							mul );
						CRS_raw.row_index[ nzc ] = k_col;
						CRS_raw.setValue( nzc, val );
						// update CCS
						if( !crs_only ) {
							const auto CCS_index = A_col_index[ k_col ] + CCS_raw.col_start[ k_col ];
							A_col_index[ k_col ]++;
							CCS_raw.row_index[ CCS_index ] = i;
							CCS_raw.setValue( CCS_index, val );
						}
						// update count
						nzc++;	
					}
				}
			}
			CRS_raw.col_start[ i + 1 ] = nzc;
		}

#ifndef NDEBUG
		if( !crs_only ) {
			for( size_t j = 0; j < ncols; ++j ) {
				assert( CCS_raw.col_start[ j + 1 ] - CCS_raw.col_start[ j ] ==
					A_col_index[ j ] );
			}
		}
#endif

		internal::setCurrentNonzeroes( A, nzc );

		return ret;
	}

	namespace internal {

		/**
		 * \internal general elementwise matrix application that all eWiseApply
		 *           variants refer to.
		 * @param[in] oper The operator corresponding to \a mulMonoid if
		 *                 \a allow_void is true; otherwise, an arbitrary operator
		 *                 under which to perform the eWiseApply.
		 * @param[in] mulMonoid The monoid under which to perform the eWiseApply if
		 *                      \a allow_void is true; otherwise, will be ignored.
		 * \endinternal
		 */

		template<
			bool allow_void,
			Descriptor descr,
			class MulMonoid, class Operator,
			typename OutputType, typename InputType1, typename InputType2,
			typename RIT1, typename CIT1, typename NIT1,
			typename RIT2, typename CIT2, typename NIT2,
			typename RIT3, typename CIT3, typename NIT3
		>
		RC eWiseApply_matrix_generic(
			Matrix< OutputType, reference, RIT1, CIT1, NIT1 > &C,
			const Matrix< InputType1, reference, RIT2, CIT2, NIT2 > &A,
			const Matrix< InputType2, reference, RIT3, CIT3, NIT3 > &B,
			const Operator &oper,
			const MulMonoid &mulMonoid,
			const Phase &phase,
			const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_operator< Operator >::value,
			void >::type * const = nullptr
		) {
			assert( !(descr & descriptors::force_row_major ) );
			static_assert( allow_void ||
				( !(
				     std::is_same< InputType1, void >::value ||
				     std::is_same< InputType2, void >::value
				) ),
				"grb::internal::eWiseApply_matrix_generic: the non-monoid version of "
				"elementwise mxm can only be used if neither of the input matrices "
				"is a pattern matrix (of type void)" );
			assert( phase != TRY );

#ifdef _DEBUG
			std::cout << "In grb::internal::eWiseApply_matrix_generic\n";
#endif

			// get whether the matrices should be transposed prior to execution
			constexpr bool trans_left = descr & descriptors::transpose_left;
			constexpr bool trans_right = descr & descriptors::transpose_right;

			// run-time checks
			const size_t m = grb::nrows( C );
			const size_t n = grb::ncols( C );
			const size_t m_A = !trans_left ? grb::nrows( A ) : grb::ncols( A );
			const size_t n_A = !trans_left ? grb::ncols( A ) : grb::nrows( A );
			const size_t m_B = !trans_right ? grb::nrows( B ) : grb::ncols( B );
			const size_t n_B = !trans_right ? grb::ncols( B ) : grb::nrows( B );

			if( m != m_A || m != m_B || n != n_A || n != n_B ) {
				return MISMATCH;
			}

			const auto &A_raw = !trans_left ?
				internal::getCRS( A ) :
				internal::getCCS( A );
			const auto &B_raw = !trans_right ?
				internal::getCRS( B ) :
				internal::getCCS( B );
			auto &C_raw = internal::getCRS( C );
			auto &CCS_raw = internal::getCCS( C );

#ifdef _DEBUG
			std::cout << "\t\t A offset array = { ";
			for( size_t i = 0; i <= m_A; ++i ) {
				std::cout << A_raw.col_start[ i ] << " ";
			}
			std::cout << "}\n";
			for( size_t i = 0; i < m_A; ++i ) {
				for( size_t k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
					std::cout << "\t\t ( " << i << ", " << A_raw.row_index[ k ] << " ) = "
						<< A_raw.getPrintValue( k ) << "\n";
				}
			}
			std::cout << "\t\t B offset array = { ";
			for( size_t j = 0; j <= m_B; ++j ) {
				std::cout << B_raw.col_start[ j ] << " ";
			}
			std::cout << "}\n";
			for( size_t j = 0; j < m_B; ++j ) {
				for( size_t k = B_raw.col_start[ j ]; k < B_raw.col_start[ j + 1 ]; ++k ) {
					std::cout << "\t\t ( " << B_raw.row_index[ k ] << ", " << j << " ) = "
						<< B_raw.getPrintValue( k ) << "\n";
				}
			}
#endif

			// retrieve buffers
			char * arr1, * arr2, * arr3, * buf1, * buf2, * buf3;
			arr1 = arr2 = buf1 = buf2 = nullptr;
			InputType1 * vbuf1 = nullptr;
			InputType2 * vbuf2 = nullptr;
			OutputType * valbuf = nullptr;
			internal::getMatrixBuffers( arr1, buf1, vbuf1, 1, A );
			internal::getMatrixBuffers( arr2, buf2, vbuf2, 1, B );
			internal::getMatrixBuffers( arr3, buf3, valbuf, 1, C );
			// end buffer retrieval

			// initialisations
			internal::Coordinates< reference > coors1, coors2;
			coors1.set( arr1, false, buf1, n );
			coors2.set( arr2, false, buf2, n );
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp parallel
			{
				size_t start, end;
				config::OMP::localRange( start, end, 0, n + 1 );
#else
				const size_t start = 0;
				const size_t end = n + 1;
#endif
				for( size_t j = start; j < end; ++j ) {
					CCS_raw.col_start[ j ] = 0;
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			}
#endif
			// end initialisations

			// nonzero count
			size_t nzc = 0;

			// symbolic phase
			if( phase == RESIZE ) {
				for( size_t i = 0; i < m; ++i ) {
					coors1.clear();
					for( size_t k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
						const size_t k_col = A_raw.row_index[ k ];
						coors1.assign( k_col );
					}
					for( size_t l = B_raw.col_start[ i ]; l < B_raw.col_start[ i + 1 ]; ++l ) {
						const size_t l_col = B_raw.row_index[ l ];
						if( coors1.assigned( l_col ) ) {
							(void)++nzc;
						}
					}
				}

				const RC ret = grb::resize( C, nzc );
				if( ret != SUCCESS ) {
					return ret;
				}
			}

			// computational phase
			if( phase == EXECUTE ) {
				// retrieve additional buffer
				config::NonzeroIndexType * const C_col_index = internal::template
					getReferenceBuffer< typename config::NonzeroIndexType >( n + 1 );

				// perform column-wise nonzero count
				for( size_t i = 0; i < m; ++i ) {
					coors1.clear();
					for( size_t k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
						const size_t k_col = A_raw.row_index[ k ];
						coors1.assign( k_col );
					}
					for( size_t l = B_raw.col_start[ i ]; l < B_raw.col_start[ i + 1 ]; ++l ) {
						const size_t l_col = B_raw.row_index[ l ];
						if( coors1.assigned( l_col ) ) {
							(void) ++nzc;
							(void) ++CCS_raw.col_start[ l_col + 1 ];
						}
					}
				}

				// check capacity
				if( nzc > capacity( C ) ) {
#ifdef _DEBUG
					std::cout << "\t detected insufficient capacity "
						<< "for requested operation\n";
#endif
					const RC clear_rc = clear( C );
					if( clear_rc != SUCCESS ) {
						return PANIC;
					} else {
						return FAILED;
					}
				}

				// prefix sum for CCS_raw.col_start
				assert( CCS_raw.col_start[ 0 ] == 0 );
				for( size_t j = 1; j < n; ++j ) {
					CCS_raw.col_start[ j + 1 ] += CCS_raw.col_start[ j ];
				}
				assert( CCS_raw.col_start[ n ] == nzc );

				// set C_col_index to all zero
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp parallel
				{
					size_t start, end;
					config::OMP::localRange( start, end, 0, n );
#else
					const size_t start = 0;
					const size_t end = n;
#endif
					for( size_t j = start; j < end; ++j ) {
						C_col_index[ j ] = 0;
					}
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				}
#endif

				// do computations
				size_t nzc = 0;
				C_raw.col_start[ 0 ] = 0;
				for( size_t i = 0; i < m; ++i ) {
					coors1.clear();
					coors2.clear();
#ifdef _DEBUG
					std::cout << "\t The elements ";
#endif
					for( size_t k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
						const size_t k_col = A_raw.row_index[ k ];
						coors1.assign( k_col );
						valbuf[ k_col ] = A_raw.getValue( k,
							mulMonoid.template getIdentity< typename Operator::D1 >() );
#ifdef _DEBUG
						std::cout << "A( " << i << ", " << k_col << " ) = " << A_raw.getValue( k,
							mulMonoid.template getIdentity< typename Operator::D1 >() ) << ", ";
#endif
					}
#ifdef _DEBUG
					std::cout << "are multiplied pairwise with ";
#endif
					for( size_t l = B_raw.col_start[ i ]; l < B_raw.col_start[ i + 1 ]; ++l ) {
						const size_t l_col = B_raw.row_index[ l ];
						if( coors1.assigned( l_col ) ) {
							coors2.assign( l_col );
							(void)grb::apply( valbuf[ l_col ], valbuf[ l_col ], B_raw.getValue( l,
								mulMonoid.template getIdentity< typename Operator::D2 >() ), oper );
#ifdef _DEBUG
							std::cout << "B( " << i << ", " << l_col << " ) = " << B_raw.getValue( l,
								mulMonoid.template getIdentity< typename Operator::D2 >() )
							<< " to yield C( " << i << ", " << l_col << " ), ";
#endif
						}
					}
#ifdef _DEBUG
					std::cout << "\n";
#endif
					for( size_t k = 0; k < coors2.nonzeroes(); ++k ) {
						const size_t j = coors2.index( k );
						// update CRS
						C_raw.row_index[ nzc ] = j;
						C_raw.setValue( nzc, valbuf[ j ] );
						// update CCS
						const size_t CCS_index = C_col_index[ j ]++ + CCS_raw.col_start[ j ];
						CCS_raw.row_index[ CCS_index ] = i;
						CCS_raw.setValue( CCS_index, valbuf[ j ] );
						// update count
						(void)++nzc;
					}
					C_raw.col_start[ i + 1 ] = nzc;
#ifdef _DEBUG
					std::cout << "\n";
#endif
				}

#ifndef NDEBUG
				for( size_t j = 0; j < n; ++j ) {
					assert( CCS_raw.col_start[ j + 1 ] - CCS_raw.col_start[ j ] == C_col_index[ j ] );
				}
#endif

				// set final number of nonzeroes in output matrix
				internal::setCurrentNonzeroes( C, nzc );
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
		class MulMonoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2,
		typename RIT3, typename CIT3, typename NIT3
	>
	RC eWiseApply(
		Matrix< OutputType, reference, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType1, reference, RIT2, CIT2, NIT2 > &A,
		const Matrix< InputType2, reference, RIT3, CIT3, NIT3 > &B,
		const MulMonoid &mulmono,
		const Phase phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< MulMonoid >::value,
		void >::type * const = nullptr
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

		return internal::eWiseApply_matrix_generic< true, descr >(
			C, A, B, mulmono.getOperator(), mulmono, phase
		);
	}

	/**
	 * Computes \f$ C = A . B \f$ for a given binary operator.
	 *
	 * \internal Pattern matrices not allowed
	 *
	 * \internal Dispatches to internal::eWiseApply_matrix_generic
	 */

	template<
		Descriptor descr = grb::descriptors::no_operation,
		class Operator,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2,
		typename RIT3, typename CIT3, typename NIT3
	>
	RC eWiseApply(
		Matrix< OutputType, reference, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType1, reference, RIT2, CIT2, NIT2 > &A,
		const Matrix< InputType2, reference, RIT3, CIT3, NIT3 > &B,
		const Operator &mulOp,
		const Phase phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value,
		void >::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename Operator::D1, InputType1 >::value ),
			"grb::eWiseApply (reference, matrix <- matrix x matrix, operator)",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the given multiplication operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename Operator::D2, InputType2 >::value ),
			"grb::eWiseApply (reference, matrix <- matrix x matrix, operator)",
			"called with a postfactor input matrix B that does not match the first "
			"domain of the given multiplication operator"
		);
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< typename Operator::D3, OutputType >::value ),
			"grb::eWiseApply (reference, matrix <- matrix x matrix, operator)",
			"called with an output matrix C that does not match the output domain "
			"of the given multiplication operator"
		);
		static_assert( ( !(
				std::is_same< InputType1, void >::value ||
				std::is_same< InputType2, void >::value )
			), "grb::eWiseApply (reference, matrix <- matrix x matrix, operator): "
			"the operator version of eWiseApply cannot be used if either of the "
			"input matrices is a pattern matrix (of type void)"
		);

		typename grb::Monoid<
			grb::operators::mul< double >,
			grb::identities::one
		> dummyMonoid;
		return internal::eWiseApply_matrix_generic< false, descr >(
			C, A, B, mulOp, dummyMonoid, phase
		);
	}

} // namespace grb

#undef NO_CAST_ASSERT

// parse this unit again for OpenMP support
#ifdef _GRB_WITH_OMP
 #ifndef _H_GRB_REFERENCE_OMP_BLAS3
  #define _H_GRB_REFERENCE_OMP_BLAS3
  #define reference reference_omp
  #include "graphblas/reference/blas3.hpp"
  #undef reference
  #undef _H_GRB_REFERENCE_OMP_BLAS3
 #endif
#endif

#endif // ``_H_GRB_REFERENCE_BLAS3''

