
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
 * @author: A. N. Yzelman
 */

#if ! defined _H_GRB_BANSHEE_BLAS3
#define _H_GRB_BANSHEE_BLAS3

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

#include <type_traits> //for std::enable_if

#include <graphblas/utils/iterators/MatrixVectorIterator.hpp>

#include "matrix.hpp"


namespace grb {

	namespace internal {

		/**
		 * \internal general mxm implementation that all mxm variants refer to
		 */
		template< bool allow_void, Descriptor descr, class MulMonoid, typename OutputType, typename InputType1, typename InputType2, class Operator, class Monoid >
		RC mxm_generic( Matrix< OutputType, banshee > & C,
			const Matrix< InputType1, banshee > & A,
			const Matrix< InputType2, banshee > & B,
			const Operator & oper,
			const Monoid & monoid,
			const MulMonoid & mulMonoid = MulMonoid(),
			const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
					grb::is_operator< Operator >::value && grb::is_monoid< Monoid >::value,
				void >::type * const = NULL ) {
			static_assert( allow_void || ( ! ( std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value ) ),
				"grb::mxm_generic: the operator-monoid version of mxm cannot be "
				"used if either of the input matrices is a pattern matrix (of type "
				"void)" );

#ifdef _DEBUG
			printf( "In grb::internal::mxm_generic (banshee, unmasked)\n" );
#endif

			// get whether the matrices should be transposed prior to execution
			constexpr bool trans_left = descr & descriptors::transpose_left;
			constexpr bool trans_right = descr & descriptors::transpose_right;

			// run-time checks
			const size_t m = grb::nrows( C );
			const size_t n = grb::ncols( C );
			const size_t m_A = ( ! trans_left ) ? grb::nrows( A ) : grb::ncols( A );
			const size_t k = ( ! trans_left ) ? grb::ncols( A ) : grb::nrows( A );
			const size_t k_B = ( ! trans_right ) ? grb::nrows( B ) : grb::ncols( B );
			const size_t n_B = ( ! trans_right ) ? grb::ncols( B ) : grb::nrows( B );

			if( m != m_A || k != k_B || n != n_B ) {
				return MISMATCH;
			}

			const auto & A_raw = ( ! trans_left ) ? internal::getCRS( A ) : internal::getCCS( A );
			const auto & B_raw = ( ! trans_right ) ? internal::getCRS( B ) : internal::getCCS( B );
			auto & C_raw = internal::getCRS( C );
			auto & CCS_raw = internal::getCCS( C );

			// memory allocations
			char * const arr = new char[ internal::Coordinates< banshee >::arraySize( n ) ];
			char * const buf = new char[ internal::Coordinates< banshee >::bufferSize( n ) ];
			OutputType * const valbuf = new OutputType[ n ];
			grb::config::NonzeroIndexType C_col_index[ n + 1 ];
			// end memory allocations and initialisations

			// initialisations
			internal::Coordinates< banshee > coors;
			coors.set( arr, false, buf, n );
			for( size_t j = 0; j <= n; ++j ) {
				CCS_raw.col_start[ j ] = 0;
			}
			// end initialisations

			// symbolic phase
			size_t nzc = 0; // non-zero count
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

			// prefix sum for C_col_index,
			// set CCS_raw.col_start to all zero
			assert( CCS_raw.col_start[ 0 ] == 0 );
			C_col_index[ 0 ] = 0;
			for( size_t j = 1; j < n; ++j ) {
				CCS_raw.col_start[ j + 1 ] += CCS_raw.col_start[ j ];
				C_col_index[ j ] = 0;
			}
			assert( CCS_raw.col_start[ n ] == nzc );

			const RC ret = grb::resize( C, nzc );
			if( ret != SUCCESS ) {
				return ret;
			}

			// computational phase
#ifndef NDEBUG
			const size_t old_nzc = nzc;
#endif
			nzc = 0;
			C_raw.col_start[ 0 ] = 0;
			for( size_t i = 0; i < m; ++i ) {
				coors.clear();
				for( size_t k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
					const size_t k_col = A_raw.row_index[ k ];
					for( size_t l = B_raw.col_start[ k_col ]; l < B_raw.col_start[ k_col + 1 ]; ++l ) {
						const size_t l_col = B_raw.row_index[ l ];
#ifdef _DEBUG
						printf( "\t A( %d, %d) = %d will be multiplied with B( "
								"%d, %d) = %d to accumulate into C( %d, %d )",
							(int)i, (int)k_col, (int)A_raw.template getValue< typename Operator::D1 >( k, mulMonoid.getIdentity() ), (int)k_col, (int)l_col,
							(int)B_raw.template getValue< typename Operator::D2 >( l, mulMonoid.getIdentity() ), (int)i, (int)l_col );
#endif
						if( ! coors.assign( l_col ) ) {
							(void)grb::apply( valbuf[ l_col ], A_raw.template getValue< typename Operator::D1 >( k, mulMonoid.getIdentity() ),
								B_raw.template getValue< typename Operator::D2 >( l, mulMonoid.getIdentity() ), oper );
						} else {
							OutputType temp;
							(void)grb::apply( temp, A_raw.template getValue< typename Operator::D1 >( k, mulMonoid.getIdentity() ),
								B_raw.template getValue< typename Operator::D2 >( l, mulMonoid.getIdentity() ), oper );
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

			delete[] arr;
			delete[] buf;
			delete[] valbuf;

			// done
			return SUCCESS;
		}

	} // namespace internal

	/**
	 * Clears the matrix of all nonzeroes.
	 *
	 * On function exit, this matrix contains zero nonzeroes. The matrix
	 * dimensions remain unchanged (these cannot change).
	 *
	 * @return grb::SUCCESS This function cannot fail.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -# This function consitutes \f$ \mathcal{O}(m+n) \f$ work.
	 *        -# This function allocates no additional dynamic memory.
	 *        -# This function uses \f$ \mathcal{O}(1) \f$ memory
	 *           beyond that which was already used at function entry.
	 *        -# This function will move up to
	 *             \f$ (m+n)\mathit{sizeof}( size\_t ) \f$
	 *           bytes of memory.
	 *        -# This function \em may free up to
	 *           \f$ \mathcal{O} \left(
	 *               (m+n)\mathit{sizeof}( size\_t ) +
	 *               \mathit{nz}\mathit{sizeof}( \text{InputType} )
	 *           \right) \f$
	 *           bytes of dynamically allocated memory.
	 * \endparblock
	 *
	 * \warning Calling clear may not clear any dynamically allocated
	 *          memory. Only destruction of the container \a A would
	 *          ensure this.
	 */
	template< typename InputType >
	RC clear( Matrix< InputType, banshee > & A ) noexcept {
		// delegate
		return A.clear();
	}

	/**
	 * \internal grb::mxm, semiring version.
	 * Dispatches to internal::mxm_generic
	 */
	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2, class Semiring >
	RC mxm( Matrix< OutputType, banshee > & C,
		const Matrix< InputType1, banshee > & A,
		const Matrix< InputType2, banshee > & B,
		const Semiring & ring = Semiring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_semiring< Semiring >::value,
			void >::type * const = NULL ) {
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Semiring::D1, InputType1 >::value ), "grb::mxm",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Semiring::D2, InputType2 >::value ), "grb::mxm",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the given operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Semiring::D4, OutputType >::value ), "grb::mxm",
			"called with an output matrix C that does not match the output domain "
			"of the given operator" );

#ifdef _DEBUG
		printf( "In grb::mxm (banshee, unmasked, semiring)\n" );
#endif

		return internal::mxm_generic< true, descr >( C, A, B, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid(), ring.getMultiplicativeMonoid() );
	}

	/**
	 * \internal mxm implementation with additive monoid and multiplicative operator
	 * Dispatches to internal::mxm_generic
	 */
	template< Descriptor descr = grb::descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2, class Operator, class Monoid >
	RC mxm( Matrix< OutputType, banshee > & C,
		const Matrix< InputType1, banshee > & A,
		const Matrix< InputType2, banshee > & B,
		const Operator & mulOp,
		const Monoid & addM,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_operator< Operator >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D1, InputType1 >::value ), "grb::mxm",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D2, InputType2 >::value ), "grb::mxm",
			"called with a postfactor input matrix B that does not match the first "
			"domain of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D3, OutputType >::value ), "grb::mxm",
			"called with an output matrix C that does not match the output domain "
			"of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, typename Operator::D3 >::value ), "grb::mxm",
			"the output domain of the multiplication operator does not match the "
			"first domain of the given addition monoid" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, OutputType >::value ), "grb::mxm",
			"the second domain of the given addition monoid does not match the "
			"type of the output matrix C" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, OutputType >::value ), "grb::mxm",
			"the output type of the given addition monoid does not match the type "
			"of the output matrix C" );
		static_assert( ( ! ( std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value ) ),
			"grb::mxm: the operator-monoid version of mxm cannot be used if either "
			"of the input matrices is a pattern matrix (of type void)" );

		return internal::mxm_generic< false, descr, Monoid >( C, A, B, mulOp, addM );
	}

	/**
	 * \internal add implementation notes
	 */
	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2 >
	RC resize( Matrix< OutputType, banshee > & C, const Matrix< InputType1, banshee > & A, const Matrix< InputType2, banshee > & B ) {
		(void)A;
		(void)B;
		(void)C;
		return SUCCESS;
	}

	namespace internal {

		template<
			Descriptor descr = descriptors::no_operation,
			bool matrix_is_void,
			typename OutputType, typename InputType1, typename InputType2, typename InputType3,
			typename Coords
		>
		RC matrix_zip_generic(
			Matrix< OutputType, banshee > &A,
			const Vector< InputType1, banshee, Coords > &x,
			const Vector< InputType2, banshee, Coords > &y,
			const Vector< InputType3, banshee, Coords > &z
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

			auto & crs = internal::getCRS( A );
			auto & ccs = internal::getCCS( A );
			auto * __restrict__ crs_offsets = crs.getOffsets();
			auto * __restrict__ crs_indices = crs.getIndices();
			auto * __restrict__ ccs_offsets = ccs.getOffsets();
			auto * __restrict__ ccs_indices = ccs.getIndices();
			auto * __restrict__ crs_values = crs.getValues();
			auto * __restrict__ ccs_values = ccs.getValues();

			grb::RC ret = SUCCESS;

			for( size_t i = 0; i < nmins; ++i ) {
				crs_offsets[ i ] = ccs_offsets[ i ] = 0;
			}
			for( size_t i = nmins; i < nrows; ++i ) {
				crs_offsets[ i ] = 0;
			}
			for( size_t i = nmins; i < ncols; ++i ) {
				ccs_offsets[ i ] = 0;
			}
			// TODO issue #64
			for( ; x_it != x_end; ++x_it ) {
				assert( *x_it < nrows );
				(void) ++( crs_offsets[ x_it->second ] );
			}
			// TODO issue #64
			for( ; y_it != y_end; ++y_it ) {
				assert( *y_it < ncols );
				(void) ++( ccs_offsets[ y_it->second ] );
			}
			const size_t T = 1;
			const size_t t = 0;
			assert( nmins > 0 );
			constexpr const size_t max_blocksize = config::CACHE_LINE_SIZE::value();
			size_t loopsz = nmins;
			size_t blocks = loopsz / max_blocksize + ( loopsz % max_blocksize == 0 ? 0 : 1 );
			size_t blocks_per_thread = blocks / T + ( blocks % T == 0 ? 0 : 1 );
			size_t start = t * blocks_per_thread * max_blocksize;
			if( start > loopsz ) {
				start = loopsz - 1;
			}
			size_t end = start + blocks_per_thread * max_blocksize;
			if( end > loopsz ) {
				end = loopsz;
			}
			(void) ++start;
			for( size_t i = start; i < end; ++i ) {
				crs_offsets[ i ] += crs_offsets[ i - 1 ];
				ccs_offsets[ i ] += ccs_offsets[ i - 1 ];
			}
			loopsz = ( nrows - nmins );
			blocks = loopsz / max_blocksize + ( loopsz % max_blocksize == 0 ? 0 : 1 );
			blocks_per_thread = blocks / T + ( blocks % T == 0 ? 0 : 1 );
			start = t * blocks_per_thread * max_blocksize;
			if( start > loopsz ) {
				start = loopsz;
			}
			end = start + blocks_per_thread * max_blocksize;
			if( end > loopsz ) {
				end = loopsz;
			}
			for( size_t i = nmins + start; i < nmins + end; ++i ) {
				crs_offsets[ i ] += crs_offsets[ i - 1 ];
			}
			loopsz = ( ncols - nmins );
			blocks = loopsz / max_blocksize + ( loopsz % max_blocksize == 0 ? 0 : 1 );
			blocks_per_thread = blocks / T + ( blocks % T == 0 ? 0 : 1 );
			start = t * blocks_per_thread * max_blocksize;
			if( start > loopsz ) {
				start = loopsz;
			}
			end = start + blocks_per_thread * max_blocksize;
			if( end > loopsz ) {
				end = loopsz;
			}
			for( size_t i = nmins + start; i < nmins + end; ++i ) {
				ccs_offsets[ i ] += ccs_offsets[ i - 1 ];
			}
			assert( T > 0 );
			for( size_t k = T - 1; k > 0; --k ) {
				loopsz = nrows;
				blocks = loopsz / max_blocksize + ( loopsz % max_blocksize == 0 ? 0 : 1 );
				blocks_per_thread = blocks / T + ( blocks % T == 0 ? 0 : 1 );
				start = k * blocks_per_thread * max_blocksize;
				if( start > loopsz ) {
					start = loopsz;
				}
				end = loopsz;
				assert( start > 0 );
				for( size_t i = start; i < end; ++i ) {
					crs_offsets[ i ] += crs_offsets[ start - 1 ];
				}
				loopsz = ncols;
				blocks = loopsz / max_blocksize + ( loopsz % max_blocksize == 0 ? 0 : 1 );
				blocks_per_thread = blocks / T + ( blocks % T == 0 ? 0 : 1 );
				start = k * blocks_per_thread * max_blocksize;
				if( start > loopsz ) {
					start = loopsz;
				}
				end = loopsz;
				assert( start > 0 );
				for( size_t i = start; i < end; ++i ) {
					ccs_offsets[ i ] += ccs_offsets[ start - 1 ];
				}
			}
			crs_offsets[ nrows ] = crs_offsets[ nrows - 1 ];
			ccs_offsets[ ncols ] = ccs_offsets[ ncols - 1 ];

			// check capacity
			assert( crs_offsets[ nrows ] == ccs_offsets[ ncols ] );
			if( internal::getNonzeroCapacity( A ) < crs_offsets[ nrows ] ) {
				return FAILED;
			}

			// do ingest
			x_it = x.cbegin();
			y_it = y.cbegin();
			// TODO issue #64
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
				internal::getCurrentNonzeroes( A ) = crs_offsets[ nrows ];
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

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2, typename InputType3,
		typename Coords
	>
	RC zip(
		Matrix< OutputType, banshee > &A,
		const Vector< InputType1, banshee, Coords > &x,
		const Vector< InputType2, banshee, Coords > &y,
		const Vector< InputType3, banshee, Coords > &z
	) {
		static_assert( !(descr & descriptors::no_casting) || std::is_integral< InputType1 >::value,
			"grb::zip (two vectors to matrix) called using non-integral left-hand "
			"vector elements" );
		static_assert( !(descr & descriptors::no_casting) || std::is_integral< InputType2 >::value,
			"grb::zip (two vectors to matrix) called using non-integral right-hand "
			"vector elements" );
		static_assert( !(descr & descriptors::no_casting) || std::is_same< OutputType, InputType3 >::value,
			"grb::zip (two vectors to matrix) called with differing vector nonzero "
			"and output matrix domains" );

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

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType1, typename InputType2, typename Coords
	>
	RC zip(
		Matrix< void, banshee > &A,
		const Vector< InputType1, banshee, Coords > &x,
		const Vector< InputType2, banshee, Coords > &y
	) {
		static_assert( !(descr & descriptors::no_casting) || std::is_integral< InputType1 >::value,
			"grb::zip (two vectors to void matrix) called using non-integral "
			"left-hand vector elements" );
		static_assert( !(descr & descriptors::no_casting) || std::is_integral< InputType2 >::value,
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
	 * \internal outer product of two vectors, implemented via mxm as
	 * a multiplication of a column vector with a row vector
	 *
	 */
	template< Descriptor descr = descriptors::no_operation, typename InputType1, typename InputType2, typename Coords, typename OutputType, class Operator >
	RC outerProduct( Matrix< OutputType, banshee > & A,
		const Vector< InputType1, banshee, Coords > & u,
		const Vector< InputType2, banshee, Coords > & v,
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

		const size_t nrows = size( u );
		const size_t ncols = size( v );

		if( nrows != grb::nrows( A ) ) {
			return MISMATCH;
		}

		if( ncols != grb::ncols( A ) ) {
			return MISMATCH;
		}

		grb::Matrix< InputType1 > u_matrix( nrows, 1 );
		grb::Matrix< InputType2 > v_matrix( 1, ncols );

		auto u_converter = grb::utils::makeVectorToMatrixConverter< InputType1 >( u, []( const size_t &ind, const InputType1 &val ) {
			return std::make_pair( std::make_pair( ind, 0 ), val );
		} );

		buildMatrixUnique( u_matrix, u_converter.begin(), u_converter.end(), PARALLEL );

		auto v_converter = grb::utils::makeVectorToMatrixConverter< InputType2 >( v, []( const size_t &ind, const InputType2 &val ) {
			return std::make_pair( std::make_pair( 0, ind ), val );
		} );
		buildMatrixUnique( v_matrix, v_converter.begin(), v_converter.end(), PARALLEL );

		grb::Monoid< grb::operators::left_assign< OutputType >, grb::identities::zero > mono;

		return grb::mxm( A, u_matrix, v_matrix, mul, mono );
	}

	namespace internal {

		/**
		 * \internal general elementwise mxm implementation that all mxm variants refer to
		 */

		template< bool allow_void, Descriptor descr, class MulMonoid, typename OutputType, typename InputType1, typename InputType2, class Operator >
		RC mxm_elementwise_generic( Matrix< OutputType, banshee > & C,
			const Matrix< InputType1, banshee > & A,
			const Matrix< InputType2, banshee > & B,
			const Operator & oper,
			const MulMonoid & mulMonoid = MulMonoid(),
			const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
					grb::is_operator< Operator >::value,
				void >::type * const = NULL ) {
			static_assert( allow_void || ( ! ( std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value ) ),
				"grb::mxm_generic: the non-monoid version of elementwise mxm can "
				"only be used if neither of the input matrices is a pattern matrix "
				"(of type void)" );

#ifdef _DEBUG
			printf( "In grb::internal::mxm_elementwise\n" );
#endif

			// get whether the matrices should be transposed prior to execution
			constexpr bool trans_left = descr & descriptors::transpose_left;
			constexpr bool trans_right = descr & descriptors::transpose_right;

			// run-time checks
			const size_t m = grb::nrows( C );
			const size_t n = grb::ncols( C );
			const size_t m_A = ( ! trans_left ) ? grb::nrows( A ) : grb::ncols( A );
			const size_t n_A = ( ! trans_left ) ? grb::ncols( A ) : grb::nrows( A );
			const size_t m_B = ( ! trans_right ) ? grb::nrows( B ) : grb::ncols( B );
			const size_t n_B = ( ! trans_right ) ? grb::ncols( B ) : grb::nrows( B );

			if( m != m_A || m != m_B || n != n_A || n != n_B ) {
				return MISMATCH;
			}

			const auto & A_raw = ( ! trans_left ) ? internal::getCRS( A ) : internal::getCCS( A );
			const auto & B_raw = ( ! trans_right ) ? internal::getCRS( B ) : internal::getCCS( B );
			auto & C_raw = internal::getCRS( C );
			auto & CCS_raw = internal::getCCS( C );

			// memory allocations
			char * const arr1 = new char[ internal::Coordinates< banshee >::arraySize( n ) ];
			char * const arr2 = new char[ internal::Coordinates< banshee >::arraySize( n ) ];
			char * const buf1 = new char[ internal::Coordinates< banshee >::bufferSize( n ) ];
			char * const buf2 = new char[ internal::Coordinates< banshee >::bufferSize( n ) ];
			OutputType * const valbuf = new OutputType[ n ];
			grb::config::NonzeroIndexType C_col_index[ n + 1 ];
			// end memory allocations and initialisations

			// initialisations
			internal::Coordinates< banshee > coors1, coors2;
			coors1.set( arr1, false, buf1, n );
			coors2.set( arr2, false, buf2, n );
			for( size_t j = 0; j <= n; ++j ) {
				CCS_raw.col_start[ j ] = 0;
			}
			// end initialisations

			// symbolic phase
			size_t nzc = 0; // non-zero count
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

			// prefix sum for C_col_index,
			// set CCS_raw.col_start to all zero
			assert( CCS_raw.col_start[ 0 ] == 0 );
			C_col_index[ 0 ] = 0;
			for( size_t j = 1; j < n; ++j ) {
				CCS_raw.col_start[ j + 1 ] += CCS_raw.col_start[ j ];
				C_col_index[ j ] = 0;
			}
			assert( CCS_raw.col_start[ n ] == nzc );

			const RC ret = grb::resize( C, nzc );
			if( ret != SUCCESS ) {
				return ret;
			}

			// computational phase
#ifndef NDEBUG
			const size_t old_nzc = nzc;
#endif
			nzc = 0;
			C_raw.col_start[ 0 ] = 0;
			for( size_t i = 0; i < m; ++i ) {
				coors1.clear();
				coors2.clear();
#ifdef _DEBUG
				printf( "\t The elements " );
#endif
				for( size_t k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
					const size_t k_col = A_raw.row_index[ k ];
					coors1.assign( k_col );
					valbuf[ k_col ] = A_raw.template getValue< typename Operator::D1 >( k, mulMonoid.getIdentity() );
#ifdef _DEBUG
					printf( "A( %d, %d) = %d ,", i, k_col, (int)A_raw.template getValue< typename Operator::D1 >( k, mulMonoid.getIdentity() ) );
#endif
				}
#ifdef _DEBUG
				printf( "are multiplied pairwise with " );
#endif
				for( size_t l = B_raw.col_start[ i ]; l < B_raw.col_start[ i + 1 ]; ++l ) {
					const size_t l_col = B_raw.row_index[ l ];
					if( coors1.assigned( l_col ) ) {
						coors2.assign( l_col );
						(void)grb::apply( valbuf[ l_col ], valbuf[ l_col ], B_raw.template getValue< typename Operator::D2 >( l, mulMonoid.getIdentity() ), oper );
#ifdef _DEBUG
						printf( "B( %d, %d) = %d to yield C(%d, %d)\n", (int)i, (int)l_col, (int)B_raw.template getValue< typename Operator::D2 >( l, mulMonoid.getIdentity() ), (int)i, (int)l_col );
#endif
					}
				}
				for( size_t k = 0; k < coors2.nonzeroes(); ++k ) {
					assert( nzc < old_nzc );
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
				printf( "\n" );
#endif
			}

#ifndef NDEBUG
			for( size_t j = 0; j < n; ++j ) {
				assert( CCS_raw.col_start[ j + 1 ] - CCS_raw.col_start[ j ] == C_col_index[ j ] );
			}
			assert( nzc == old_nzc );
#endif

			// set final number of nonzeroes in output matrix
			internal::setCurrentNonzeroes( C, nzc );

			delete[] arr1;
			delete[] arr2;
			delete[] buf1;
			delete[] buf2;
			delete[] valbuf;

			// done
			return SUCCESS;
		}

	} // namespace internal

	/**
	 * \internal grb::mxm_elementwise, multiplicative monoid version, allows pattern matrix inputs
	 * Dispatches to internal::mxm_generic
	 */
	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2, class MulMonoid >
	RC mxm_elementwise( Matrix< OutputType, banshee > & C,
		const Matrix< InputType1, banshee > & A,
		const Matrix< InputType2, banshee > & B,
		const MulMonoid & mulmono = MulMonoid(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_monoid< MulMonoid >::value,
			void >::type * const = NULL ) {
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename MulMonoid::D1, InputType1 >::value ), "grb::mxm_elementwise",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the monoid operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename MulMonoid::D2, InputType2 >::value ), "grb::mxm_elementwise",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the monoid operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename MulMonoid::D3, OutputType >::value ), "grb::mxm_elementwise",
			"called with an output matrix C that does not match the output domain "
			"of the monoid operator" );

#ifdef _DEBUG
		printf( "In grb::mxm_elementwise (banshee, monoid)\n" );
#endif

		return internal::mxm_elementwise_generic< true, descr >( C, A, B, mulmono.getOperator(), mulmono );
	}

	/**
	 * \internal mxm_elementwise implementation with an operator that does not necessarily come from a monoid, pattern matrices not allowed
	 * Dispatches to internal::mxm_generic
	 */

	template< Descriptor descr = grb::descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2, class Operator >
	RC mxm_elementwise( Matrix< OutputType, banshee > & C,
		const Matrix< InputType1, banshee > & A,
		const Matrix< InputType2, banshee > & B,
		const Operator & mulOp,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_operator< Operator >::value,
			void >::type * const = NULL ) {
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D1, InputType1 >::value ), "grb::mxm_elementwise",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D2, InputType2 >::value ), "grb::mxm_elementwise",
			"called with a postfactor input matrix B that does not match the first "
			"domain of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D3, OutputType >::value ), "grb::mxm_elementwise",
			"called with an output matrix C that does not match the output domain "
			"of the given multiplication operator" );
		static_assert( ( ! ( std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value ) ),
			"grb::mxm_elementwise: the operator version of mxm_elementwise cannot "
			"be used if either of the input matrices is a pattern matrix (of type "
			"void)" );

		typedef grb::Monoid< grb::operators::mul< double >, grb::identities::one > DummyMono;
		return internal::mxm_elementwise_generic< false, descr, DummyMono >( C, A, B, mulOp );
	}

} // namespace grb

#undef NO_CAST_ASSERT

#endif // ``_H_GRB_BANSHEE_BLAS3''
