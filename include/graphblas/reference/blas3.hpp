
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
#include <graphblas/utils/MatrixVectorIterator.hpp>

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
			const PHASE &phase,
			const typename std::enable_if< !grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value && !
				grb::is_object< InputType2 >::value &&
				grb::is_operator< Operator >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = NULL
		) {
			static_assert( allow_void ||
				( !(
					std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value
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

			// run-time checks
			const size_t m = grb::nrows( C );
			const size_t n = grb::ncols( C );
			const size_t m_A = !trans_left ? grb::nrows( A ) : grb::ncols( A );
			const size_t k = !trans_left ? grb::ncols( A ) : grb::nrows( A );
			const size_t k_B = !trans_right ? grb::nrows( B ) : grb::ncols( B );
			const size_t n_B = !trans_right ? grb::ncols( B ) : grb::nrows( B );

			if( m != m_A || k != k_B || n != n_B ) {
				return MISMATCH;
			}

			const auto &A_raw = !trans_left ? internal::getCRS( A ) : internal::getCCS( A );
			const auto &B_raw = !trans_right ? internal::getCRS( B ) : internal::getCCS( B );
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
			for( size_t j = 0; j <= n; ++j ) {
				CCS_raw.col_start[ j ] = 0;
			}
			// end initialisations

			// symbolic phase (counting sort, step 1)
			size_t nzc = 0; // output nonzero count
			for( size_t i = 0; i < m; ++i ) {
				coors.clear();
				for( size_t k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
					const size_t k_col = A_raw.row_index[ k ];
					for( size_t l = B_raw.col_start[ k_col ]; l < B_raw.col_start[ k_col + 1 ]; ++l ) {
						const size_t l_col = B_raw.row_index[ l ];
						if( ! coors.assign( l_col ) ) {
							(void)++nzc;
							(void)++CCS_raw.col_start[ l_col + 1 ];
						}
					}
				}
			}

			if( phase == SYMBOLIC ) {
				// do final resize
				const RC ret = grb::resize( C, nzc );
				return ret;
			}

			// computational phase
			assert( phase == NUMERICAL );

			// prefix sum for C_col_index,
			// set CCS_raw.col_start to all zero
			assert( CCS_raw.col_start[ 0 ] == 0 );
			C_col_index[ 0 ] = 0;
			for( size_t j = 1; j < n; ++j ) {
				CCS_raw.col_start[ j + 1 ] += CCS_raw.col_start[ j ];
				C_col_index[ j ] = 0;
			}
			assert( CCS_raw.col_start[ n ] == nzc );

#ifndef NDEBUG
			const size_t old_nzc = nzc;
#endif
			// use prefix sum to perform computational phase
			nzc = 0;
			C_raw.col_start[ 0 ] = 0;
			for( size_t i = 0; i < m; ++i ) {
				coors.clear();
				for( size_t k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
					const size_t k_col = A_raw.row_index[ k ];
					for( size_t l = B_raw.col_start[ k_col ]; l < B_raw.col_start[ k_col + 1 ]; ++l ) {
						const size_t l_col = B_raw.row_index[ l ];
#ifdef _DEBUG
						std::cout << "\t A( " << i << ", " << k_col << " ) = " << A_raw.getValue( k, mulMonoid.template getIdentity< typename Operator::D1 >() ) << " will be multiplied with B( "
								  << k_col << ", " << l_col << " ) = " << B_raw.getValue( l, mulMonoid.template getIdentity< typename Operator::D2 >() ) << " to accumulate into C( " << i << ", "
								  << l_col << " )\n";
#endif
						if( ! coors.assign( l_col ) ) {
							valbuf[ l_col ] = monoid.template getIdentity< OutputType >();
							(void)grb::apply( valbuf[ l_col ], A_raw.getValue( k, mulMonoid.template getIdentity< typename Operator::D1 >() ),
								B_raw.getValue( l, mulMonoid.template getIdentity< typename Operator::D2 >() ), oper );
						} else {
							OutputType temp = monoid.template getIdentity< OutputType >();
							(void)grb::apply( temp, A_raw.getValue( k, mulMonoid.template getIdentity< typename Operator::D1 >() ),
								B_raw.getValue( l, mulMonoid.template getIdentity< typename Operator::D2 >() ), oper );
							(void)grb::foldl( valbuf[ l_col ], temp, monoid.getOperator() );
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
					const size_t CCS_index = C_col_index[ j ]++ + CCS_raw.col_start[ j ];
					CCS_raw.row_index[ CCS_index ] = i;
					CCS_raw.setValue( CCS_index, valbuf[ j ] );
					// update count
					(void)++nzc;
				}
				C_raw.col_start[ i + 1 ] = nzc;
			}

#ifndef NDEBUG
			for( size_t j = 0; j < n; ++j ) {
				assert( CCS_raw.col_start[ j + 1 ] - CCS_raw.col_start[ j ] == C_col_index[ j ] );
			}
			assert( nzc == old_nzc );
#endif

			// set final number of nonzeroes in output matrix
			internal::setCurrentNonzeroes( C, nzc );

			// done
			return SUCCESS;
		}

	} // namespace internal

	namespace internal {

		template<
			bool A_is_mask,
			Descriptor descr,
			typename OutputType, typename InputType1, typename InputType2 = const OutputType
		>
		RC set( Matrix< OutputType, reference > &C,
			const Matrix< InputType1, reference > &A,
			const InputType2 * __restrict__ id = NULL
		) noexcept {
#ifdef _DEBUG
			std::cout << "Called grb::set (matrices, reference)" << std::endl;
#endif
			// static checks
			NO_CAST_ASSERT(
				( !( descr & descriptors::no_casting ) ||
				( !A_is_mask && std::is_same< InputType1, OutputType >::value ) ),
				"internal::grb::set", "called with non-matching value types"
			);
			NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
				( A_is_mask && std::is_same< InputType2, OutputType >::value ) ),
				"internal::grb::set", "Called with non-matching value types"
			);

			// run-time checks
			const size_t m = nrows( A );
			const size_t n = ncols( A );
			if( nrows( C ) != m ) {
				return MISMATCH;
			}
			if( ncols( C ) != n ) {
				return MISMATCH;
			}
			if( A_is_mask ) {
				assert( id != NULL );
			}

			// catch trivial cases
			if( m == 0 || n == 0 ) {
				return SUCCESS;
			}
			if( nnz( A ) == 0 ) {
#ifdef _DEBUG
				std::cout << "\t input matrix has no nonzeroes, simply clearing output matrix...\n";
#endif
				return clear( C );
			}

			// symbolic phase (TODO: issue #7)
			const size_t nz = grb::nnz( A );
			RC ret = grb::resize( C, nz );

			// compute phase (TODO: issue #7)
			if( ret == SUCCESS ) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp parallel
#endif
				{
					size_t range = internal::getCRS( C ).copyFromRange( nz, m );
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
					size_t start, end;
					config::OMP::localRange( start, end, 0, range );
#else
					const size_t start = 0;
					size_t end = range;
#endif
					if( A_is_mask ) {
						internal::getCRS( C ).template copyFrom< true >( internal::getCRS( A ), nz, m, start, end, id );
					} else {
						internal::getCRS( C ).template copyFrom< false >( internal::getCRS( A ), nz, m, start, end );
					}
					range = internal::getCCS( C ).copyFromRange( nz, n );
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
					config::OMP::localRange( start, end, 0, range );
#else
					end = range;
#endif
					if( A_is_mask ) {
						internal::getCCS( C ).template copyFrom< true >( internal::getCCS( A ), nz, n, start, end, id );
					} else {
						internal::getCCS( C ).template copyFrom< false >( internal::getCCS( A ), nz, n, start, end );
					}
				}
				internal::setCurrentNonzeroes( C, nz );
			}

			// done
			return ret;
		}

	} // namespace internal

	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename InputType >
	RC set( Matrix< OutputType, reference > & C, const Matrix< InputType, reference > & A ) noexcept {
		static_assert( std::is_same< OutputType, void >::value || ! std::is_same< InputType, void >::value,
			"grb::set cannot interpret an input pattern matrix without a "
			"semiring or a monoid. This interpretation is needed for "
			"writing the non-pattern matrix output. Possible solutions: 1) "
			"use a (monoid-based) foldl / foldr, 2) use a masked set, or "
			"3) change the output of grb::set to a pattern matrix also." );
#ifdef _DEBUG
		std::cout << "Called grb::set (matrix-to-matrix, reference)" << std::endl;
#endif
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, OutputType >::value ), "grb::set", "called with non-matching value types" );

		// delegate
		return internal::set< false, descr >( C, A );
	}

	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2 >
	RC set( Matrix< OutputType, reference > & C, const Matrix< InputType1, reference > & A, const InputType2 & val ) noexcept {
		static_assert( ! std::is_same< OutputType, void >::value,
			"internal::grb::set (masked set to value): cannot have a pattern "
			"matrix as output" );
#ifdef _DEBUG
		std::cout << "Called grb::set (matrix-to-value-masked, reference)" << std::endl;
#endif
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType2, OutputType >::value ), "grb::set", "called with non-matching value types" );

		// delegate
		if( std::is_same< OutputType, void >::value ) {
			return internal::set< false, descr >( C, A );
		} else {
			return internal::set< true, descr >( C, A, &val );
		}
	}

	/**
	 * \internal grb::mxm, semiring version.
	 * Dispatches to internal::mxm_generic
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		class Semiring
	>
	RC mxm( Matrix< OutputType, reference > &C,
		const Matrix< InputType1, reference > &A,
		const Matrix< InputType2, reference > &B,
		const Semiring & ring = Semiring(),
		const PHASE & phase = NUMERICAL,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
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
		class Operator, class Monoid
	>
	RC mxm( Matrix< OutputType, reference > &C,
		const Matrix< InputType1, reference > &A,
		const Matrix< InputType2, reference > &B,
		const Operator &mulOp,
		const Monoid &addM,
		const PHASE &phase = NUMERICAL,
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
			typename Coords
		>
		RC matrix_zip_generic( Matrix< OutputType, reference > & A,
			const Vector< InputType1, reference, Coords > & x,
			const Vector< InputType2, reference, Coords > & y,
			const Vector< InputType3, reference, Coords > & z
		) {
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
			#pragma omp parallel for schedule( static, config::CACHE_LINE_SIZE::value() )
#endif
			for( size_t i = 0; i < nmins; ++i ) {
				crs_offsets[ i ] = ccs_offsets[ i ] = 0;
			}
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp parallel for schedule( static, config::CACHE_LINE_SIZE::value() )
#endif
			for( size_t i = nmins; i < nrows; ++i ) {
				crs_offsets[ i ] = 0;
			}
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp parallel for schedule( static, config::CACHE_LINE_SIZE::value() )
#endif
			for( size_t i = nmins; i < ncols; ++i ) {
				ccs_offsets[ i ] = 0;
			}

			// step 2: counting sort, phase one

			// TODO internal issue #64
			for( ; x_it != x_end; ++x_it ) {
				assert( x_it->second < nrows );
				(void)++( crs_offsets[ x_it->second ] );
			}
			// TODO internal issue #64
			for( ; y_it != y_end; ++y_it ) {
				assert( y_it->second < ncols );
				(void)++( ccs_offsets[ y_it->second ] );
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
			(void)++start;
			for( size_t i = start; i < end; ++i ) {
				crs_offsets[ i ] += crs_offsets[ i - 1 ];
				ccs_offsets[ i ] += ccs_offsets[ i - 1 ];
			}
			config::OMP::localRange( start, end, 0, nrows - nmins );
			for( size_t i = nmins + start; i < nmins + end; ++i ) {
				crs_offsets[ i ] += crs_offsets[ i - 1 ];
			}
			config::OMP::localRange( start, end, 0, ncols - nmins );
			for( size_t i = nmins + start; i < nmins + end; ++i ) {
				ccs_offsets[ i ] += ccs_offsets[ i - 1 ];
			}
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp barrier
#endif
			assert( T > 0 );
			for( size_t k = T - 1; k > 0; --k ) {
				config::OMP::localRange( start, end, 0, nrows,
					config::CACHE_LINE_SIZE::value(), k, T
				);
				assert( start > 0 );
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp for schedule( static, config::CACHE_LINE_SIZE::value() )
#endif
				for( size_t i = start; i < nrows; ++i ) { // note: nrows, not end(!)
					crs_offsets[ i ] += crs_offsets[ start - 1 ];
				}
				config::OMP::localRange( start, end, 0, ncols,
					config::CACHE_LINE_SIZE::value(), k, T
				);
				assert( start > 0 );
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp for schedule( static, config::CACHE_LINE_SIZE::value() )
#endif
				for( size_t i = start; i < ncols; ++i ) { // note: ncols, not end
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
				if( ! matrix_is_void && ret == SUCCESS && ( x_it->first != z_it->first ) ) {
					ret = ILLEGAL;
				}
				const size_t crs_pos = --( crs_offsets[ x_it->second ] );
				const size_t ccs_pos = --( ccs_offsets[ y_it->second ] );
				crs_indices[ crs_pos ] = y_it->second;
				ccs_indices[ ccs_pos ] = x_it->second;
				if( ! matrix_is_void ) {
					crs_values[ crs_pos ] = ccs_values[ ccs_pos ] = z_it->second;
					(void)++z_it;
				}
			}

			if( ret == SUCCESS ) {
				internal::setCurrentNonzeroes( A, crs_offsets[ nrows ] );
			}

			assert( x_it == x_end );
			assert( y_it == y_end );
			if( ! matrix_is_void ) {
				assert( z_it == z_end );
			}

			if( matrix_is_void ) {
				(void)z_end;
			}

			// done
			return ret;
		}

	} // namespace internal

	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2, typename InputType3, typename Coords >
	RC zip( Matrix< OutputType, reference > & A,
		const Vector< InputType1, reference, Coords > & x,
		const Vector< InputType2, reference, Coords > & y,
		const Vector< InputType3, reference, Coords > & z ) {
		static_assert( ! ( descr & descriptors::no_casting ) || std::is_integral< InputType1 >::value,
			"grb::zip (two vectors to matrix) called using non-integral left-hand vector elements" );
		static_assert( ! ( descr & descriptors::no_casting ) || std::is_integral< InputType2 >::value,
			"grb::zip (two vectors to matrix) called using non-integral right-hand vector elements" );
		static_assert( ! ( descr & descriptors::no_casting ) || std::is_same< OutputType, InputType3 >::value,
			"grb::zip (two vectors to matrix) called with differing vector nonzero and output matrix domains" );

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

		return internal::matrix_zip_generic< descr, false >( A, x, y, z );
	}

	template< Descriptor descr = descriptors::no_operation, typename InputType1, typename InputType2, typename Coords >
	RC zip( Matrix< void, reference > & A, const Vector< InputType1, reference, Coords > & x, const Vector< InputType2, reference, Coords > & y ) {
		static_assert( ! ( descr & descriptors::no_casting ) || std::is_integral< InputType1 >::value,
			"grb::zip (two vectors to void matrix) called using non-integral "
			"left-hand vector elements" );
		static_assert( ! ( descr & descriptors::no_casting ) || std::is_integral< InputType2 >::value,
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

		return internal::matrix_zip_generic< descr, true >( A, x, y, x );
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
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords, class Operator
	>
	RC outer( Matrix< OutputType, reference > &A,
		const Vector< InputType1, reference, Coords > &u,
		const Vector< InputType2, reference, Coords > &v,
		const Operator &mul = Operator(),
		const PHASE &phase = NUMERICAL,
		const typename std::enable_if<
			grb::is_operator< Operator >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				!grb::is_object< OutputType >::value,
			void
		>::type * const = nullptr
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

		const size_t nrows = size( u );
		const size_t ncols = size( v );

		if( nrows != grb::nrows( A ) ) {
			return MISMATCH;
		}

		if( ncols != grb::ncols( A ) ) {
			return MISMATCH;
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
		if( phase == NUMERICAL ) {
			ret = grb::clear( A );
		}
		assert( nnz( A ) == 0 );
		ret = ret ? ret : grb::mxm( A, u_matrix, v_matrix, mul, mono, phase );
		return ret;
	}

	namespace internal {

		/**
		 * \internal general elementwise matrix application that all eWiseApply
		 *           variants refer to.
		 */

		template<
			bool allow_void,
			Descriptor descr,
			class MulMonoid,
			typename OutputType, typename InputType1, typename InputType2,
			class Operator
		>
		RC eWiseApply_matrix_generic( Matrix< OutputType, reference > &C,
			const Matrix< InputType1, reference > &A,
			const Matrix< InputType2, reference > &B,
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

			const auto &A_raw = !trans_left ? internal::getCRS( A ) : internal::getCCS( A );
			const auto &B_raw = !trans_right ? internal::getCRS( B ) : internal::getCCS( B );
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
			#pragma omp parallel for schedule( static, config::CACHE_LINE_SIZE::value() )
#endif
			for( size_t j = 0; j <= n; ++j ) {
				CCS_raw.col_start[ j ] = 0;
			}
			// end initialisations

			// nonzero count
			size_t nzc = 0;

			// symbolic phase
			if( phase == SYMBOLIC ) {
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
			if( phase == NUMERICAL ) {
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
							(void)++nzc;
							(void)++CCS_raw.col_start[ l_col + 1 ];
						}
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
				#pragma omp parallel for schedule( static, config::CACHE_LINE_SIZE::value() )
#endif
				for( size_t j = 0; j < n; ++j ) {
					C_col_index[ j ] = 0;
				}

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
						valbuf[ k_col ] = A_raw.getValue( k, mulMonoid.template getIdentity< typename Operator::D1 >() );
#ifdef _DEBUG
						std::cout << "A( " << i << ", " << k_col << " ) = " << A_raw.getValue( k, mulMonoid.template getIdentity< typename Operator::D1 >() ) << ", ";
#endif
					}
#ifdef _DEBUG
					std::cout << "are multiplied pairwise with ";
#endif
					for( size_t l = B_raw.col_start[ i ]; l < B_raw.col_start[ i + 1 ]; ++l ) {
						const size_t l_col = B_raw.row_index[ l ];
						if( coors1.assigned( l_col ) ) {
							coors2.assign( l_col );
							(void)grb::apply( valbuf[ l_col ], valbuf[ l_col ], B_raw.getValue( l, mulMonoid.template getIdentity< typename Operator::D2 >() ), oper );
#ifdef _DEBUG
							std::cout << "B( " << i << ", " << l_col << " ) = " << B_raw.getValue( l, mulMonoid.template getIdentity< typename Operator::D2 >() ) <<
								" to yield C( " << i << ", " << l_col << " ), ";
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
		typename OutputType, typename InputType1, typename InputType2,
		class MulMonoid
	>
	RC eWiseApply( Matrix< OutputType, reference > &C,
		const Matrix< InputType1, reference > &A,
		const Matrix< InputType2, reference > &B,
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
		typename OutputType, typename InputType1, typename InputType2,
		class Operator
	>
	RC eWiseApply( Matrix< OutputType, reference > &C,
		const Matrix< InputType1, reference > &A,
		const Matrix< InputType2, reference > &B,
		const Operator &mulOp,
		const PHASE phase = NUMERICAL,
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

