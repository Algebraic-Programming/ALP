
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

/**
 * @file
 *
 * @author A. N. Yzelman
 */

#if ! defined _H_GRB_REFERENCE_BLAS3 || defined _H_GRB_REFERENCE_OMP_BLAS3
#define _H_GRB_REFERENCE_BLAS3

#include <type_traits> //for std::enable_if

#include <graphblas/base/blas3.hpp>
#include <graphblas/base/final.hpp>

#include <graphblas/utils/unordered_memmove.hpp>
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

#ifdef _DEBUG
 #define _DEBUG_REFERENCE_BLAS3
#endif


namespace grb {

	namespace internal {

		/**
		 * \internal general select implementation that
		 * all select variants refer to
		 */
		template<
			Descriptor descr,
			class SelectionOperator,
			typename Tin,
			typename RITin, typename CITin, typename NITin,
			typename Tout,
			typename RITout, typename CITout, typename NITout
		>
		RC select_generic(
			Matrix< Tout, reference, RITout, CITout, NITout > &out,
			const Matrix< Tin, reference, RITin, CITin, NITin > &in,
			const SelectionOperator &op,
			const std::function< size_t( size_t ) > &row_l2g,
			const std::function< size_t( size_t ) > &col_l2g,
			const Phase &phase
		) {
			typedef typename std::conditional<
				std::is_void< Tin >::value,
				bool,
				Tin
			>::type InValuesType;
			// the identity will only be used for void input matrices
			// if unused, we initialise it via default-construction, which is always ok
			// to do since value types must be POD.
			const InValuesType identity = std::is_void< Tin >::value
				? true
				: InValuesType();

			constexpr bool crs_only = descr & descriptors::force_row_major;
			constexpr bool transpose_input = descr & descriptors::transpose_matrix;
			static_assert( !(crs_only && transpose_input),
				"The descriptors::force_row_major and descriptors::transpose_matrix "
				"flags cannot be used simultaneously" );

			const auto &in_raw = transpose_input
				? internal::getCCS( in )
				: internal::getCRS( in );
			auto &out_crs = internal::getCRS( out );
			auto &out_ccs = internal::getCCS( out );
			const size_t
				m = transpose_input ? ncols( in ) : nrows( in ),
				n = transpose_input ? nrows( in ) : ncols( in );
			const size_t m_out = nrows( out );
			const size_t n_out = ncols( out );
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			const size_t nz = nnz( in );
#endif

			typedef typename std::conditional< transpose_input, CITin, RITin >::type
				EffectiveRowType;
			typedef typename std::conditional< transpose_input, RITin, CITin >::type
				EffectiveColumnType;

			// Check if the dimensions fit
			if( m != m_out || n != n_out ) {
				return MISMATCH;
			}

			if( m == 0 || n == 0 || nnz(in ) == 0 ) {
				return SUCCESS;
			}

			if( phase == RESIZE ) {
				size_t nzc = 0;

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				const size_t nthreads = m < config::OMP::minLoopSize()
					? 1
					: std::min(
							config::OMP::threads(),
							nz / config::CACHE_LINE_SIZE::value()
						  );

				#pragma omp parallel reduction(+: nzc) num_threads( nthreads )
#endif
				{
					size_t start_row = 0;
					size_t end_row = m;
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
					config::OMP::localRange( start_row, end_row, 0, m );
#endif
					size_t local_nzc = 0;
					for( size_t i = start_row; i < end_row; ++i ) {
						for(
							size_t k = in_raw.col_start[ i ];
							k < in_raw.col_start[ i + 1 ];
							++k
						) {
							const auto j = in_raw.row_index[ k ];
							const auto global_row =
								static_cast< EffectiveRowType >( row_l2g( i ) );
							const auto global_col =
								static_cast< EffectiveColumnType >( col_l2g( j ) );
							const auto value = in_raw.getValue( k, identity );
							if( op( global_row, global_col, value ) ) {
								(void) ++local_nzc;
							}
						}
					}
					nzc += local_nzc;
				}
				return grb::resize( out, nzc );
			}

			// Execute phase only from here on
			assert( phase == EXECUTE );

			// Declare the column counter array
			config::NonzeroIndexType * col_counter = nullptr;

			if( !crs_only ) {
				// Allocate the column counter array
				char *arr = nullptr, *buf = nullptr;
				Tin *valbuf = nullptr;
				internal::getMatrixBuffers( arr, buf, valbuf, 1, out );
				col_counter =
					internal::getReferenceBuffer< config::NonzeroIndexType >( n + 1 );

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				const size_t nthreads = ( n + 1 ) < config::OMP::minLoopSize()
					? 1
					: std::min(
							config::OMP::threads(),
							nz / config::CACHE_LINE_SIZE::value()
						);
				#pragma omp parallel for simd num_threads( nthreads )
#endif
				for( size_t j = 0; j < n + 1; ++j ) {
					out_ccs.col_start[ j ] = 0;
				}

				// Fill CCS.col_start with the number of nonzeros in each column
				for( size_t i = 0; i < m; ++i ) {
					for( size_t k = in_raw.col_start[ i ]; k < in_raw.col_start[ i + 1 ]; ++k ) {
						const auto j = in_raw.row_index[ k ];
						const auto global_row = static_cast< EffectiveRowType >( row_l2g( i ) );
						const auto global_col =
							static_cast< EffectiveColumnType >( col_l2g( j ) );
						const auto value = in_raw.getValue( k, identity );
						if( op( global_row, global_col, value ) ) {
							(void) ++out_ccs.col_start[ j + 1 ];
						}
					}
				}

				// Prefix sum of CCS.col_start
				for( size_t j = 1; j < n + 1; ++j ) {
					out_ccs.col_start[ j ] += out_ccs.col_start[ j - 1 ];
				}

				{ // Initialise the column counter array with zeros
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
					const size_t nthreads = ( n + 1 ) < config::OMP::minLoopSize()
						? 1
						: std::min(
								config::OMP::threads(),
								n / config::CACHE_LINE_SIZE::value()
							);
					#pragma omp parallel for simd num_threads( nthreads )
#endif
					for( size_t j = 0; j < n + 1; ++j ) {
						col_counter[ j ] = 0;
					}
				}
			}

			out_crs.col_start[ 0 ] = 0;
			size_t nzc = 0;
			for( size_t i = 0; i < m; ++i ) {
				for( size_t k = in_raw.col_start[ i ]; k < in_raw.col_start[ i + 1 ];
					++k
				) {
					const auto j = in_raw.row_index[ k ];
					const auto global_row = static_cast< EffectiveRowType >( row_l2g( i ) );
					const auto global_col = static_cast< EffectiveColumnType >( col_l2g( j ) );
					const auto value = in_raw.getValue( k, identity );
					if( !op( global_row, global_col, value ) ) {
						continue;
					}
#ifdef _DEBUG_REFERENCE_BLAS3
					std::cout << "\tKeeping value at: " << row_l2g( i ) << ", "
						<< col_l2g( j ) << " -> idx=" << nzc << "\n";
#endif

					// Update CCS
					if( !crs_only ) {
						const auto idx = out_ccs.col_start[ j ] + col_counter[ j ];
						(void) ++col_counter[ j ];
						out_ccs.row_index[ idx ] = i;
						out_ccs.setValue( idx, value );
					}

					// Update CRS
					out_crs.row_index[ nzc ] = j;
					out_crs.setValue( nzc, value );
					(void) ++nzc;
				}
				out_crs.col_start[ i + 1 ] = nzc;
			}

			internal::setCurrentNonzeroes( out, nzc );

			return SUCCESS;
		}

		/**
		 * This is a inner-loop kernel definition that is taken outside in order to
		 * reduce code duplication
		 */
		template<
			bool crs_only, grb::Phase phase,
			typename D, typename IND, typename SIZE,
			typename AStorageT, typename BStorageT
		>
		inline void mxm_generic_ompPar_get_row_col_counts_kernel(
			size_t &nzc,
			Compressed_Storage< D, IND, SIZE > &CRS,
			Compressed_Storage< D, IND, SIZE > &CCS,
			const AStorageT &A_raw, const BStorageT &B_raw,
			const size_t n,
			const bool inplace,
			internal::Coordinates< reference > &coors,
			const size_t i, const size_t end_offset
		) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
//			const size_t dbgt = omp_get_num_threads(); // DBG
//			omp_set_num_threads( 1 ); // DBG
 #ifdef _DEBUG_REFERENCE_BLAS3
			#pragma omp critical
 #endif
#endif
			coors.clear_seq();
			if( inplace ) {
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp critical
 #endif
				std::cout << "\t\t\t looking for pre-existing nonzeroes on row " << i
					<< " in the CRS array range " << CRS.col_start[ i ] << " -- "
					<< end_offset << "\n";
#endif
				for(
					auto k = CRS.col_start[ i ];
					k < static_cast< SIZE >(end_offset);
					++k
				) {
					const auto index = CRS.row_index[ k ];
					if( static_cast< size_t >(index) == n ) {
						break;
					}
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
					#pragma omp critical
 #endif
					std::cout << "\t\t\t\t recording pre-existing nonzero ( " << i << ", "
						<< index << " ) [thread " << omp_get_thread_num() << "]\n";
#endif
					if( !coors.assign( index ) ) {
						(void) ++nzc;
						if( !crs_only && phase == EXECUTE ) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
							#pragma omp atomic update
#else
							(void)
#endif
								++CCS.col_start[ index + 1 ];
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
							#pragma omp critical
							std::cout << "\t\t\t updated col count at " << index << " to "
								<< CCS.col_start[ index + 1 ] << "\n"; // note that printed info may be
							                                               // stale due to data race
 #endif
#endif
						}
					}
				}
			}
			for(
				auto k = A_raw.col_start[ i ];
				k < A_raw.col_start[ i + 1 ];
				++k
			) {
				const size_t k_col = A_raw.row_index[ k ];
				for(
					auto l = B_raw.col_start[ k_col ];
					l < B_raw.col_start[ k_col + 1 ];
					++l
				) {
					const size_t l_col = B_raw.row_index[ l ];
					if( !coors.assign( l_col ) ) {
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
						#pragma omp critical
 #endif
						std::cout << "\t\t\t recording new nonzero ( " << i << ", " << l_col
							<< " )\n";
#endif
						(void) ++nzc;
						if( !crs_only && phase == EXECUTE ) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
							#pragma omp atomic update
#else
							(void)
#endif
								++CCS.col_start[ l_col + 1 ];
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
							#pragma omp critical
							std::cout << "\t\t\t updated col count at " << l_col << " to "
								<< CCS.col_start[ l_col + 1 ] << "\n"; // note that printed info may be
							                                               // stale due to data race
 #endif
#endif
						}
					}
				}
			}
			if( phase == EXECUTE ) {
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp critical
				std::cout << "\t\t\t recording row " << i << " start offset: "
					<< nzc << " [thread " << omp_get_thread_num() << "]\n";
 #endif
#endif
				CRS.col_start[ i ] = nzc;
			}
/*#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			omp_set_num_threads( dbgt ); // DBG
#endif*/
		}

		/**
		 * This is a rather specialised function for computing local nonzero counts,
		 * a cumulative row-count array, and a direct column-count array. Furthermore,
		 * the latter two are only computed when \a phase is EXECUTE, and if it is,
		 * the latter is only computed when \a crs_only is <tt>false</tt>.
		 */
		template<
			Descriptor descr, bool crs_only, grb::Phase phase,
			typename D,
			typename RIT, typename CIT, typename NIT,
			typename InputType1, typename InputType2, typename OutputType,
			typename IND, typename SIZE
		>
		RC mxm_generic_ompPar_get_row_col_counts(
			size_t &nzc,
			Compressed_Storage< D, IND, SIZE > &CRS,
			Compressed_Storage< D, IND, SIZE > &CCS,
			const size_t m, const size_t n,
			const Matrix< InputType1, reference, RIT, CIT, NIT > &A,
			const Matrix< InputType2, reference, RIT, CIT, NIT > &B,
			const Matrix< OutputType, reference, RIT, CIT, NIT > &C,
			char * const arr, char * const buf,
			bool inplace
		) {
			(void) C;

			// static sanity checks
			static_assert( phase == RESIZE || phase == EXECUTE, "Only resize and execute "
				"phases are currently supported." );

#ifdef _DEBUG_REFERENCE_BLAS3
			std::cout << "\t\t Entering mxm_generic_ompPar_get_row_col_counts, "
				<< "inplace = " << inplace << ", phase = ";
			if( phase == RESIZE ) { std::cout <<"RESIZE\n"; }
			else if( phase == EXECUTE ) { std::cout << "EXECUTE\n"; }
			else { std::cout << "UNKNOWN\n"; }
#endif
			constexpr bool trans_left = descr & descriptors::transpose_left;
			constexpr bool trans_right = descr & descriptors::transpose_right;
			const auto &A_raw = !trans_left
				? internal::getCRS( A )
				: internal::getCCS( A );
			const auto &B_raw = !trans_right
				? internal::getCRS( B )
				: internal::getCCS( B );

			// local loop bounds
			size_t start, end;

			// initialisations

			// local SPA
			internal::Coordinates< reference > coors;
			coors.set_seq( arr, false, buf, n );

			if( !crs_only && phase == EXECUTE ) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				config::OMP::localRange( start, end, 0, n + 1 );
#else
				start = 0;
				end = n + 1;
#endif
				for( size_t j = start; j < end; ++j ) {
					CCS.col_start[ j ] = 0;
				}
			}

			// loop over all rows + 1 (CRS start/offsets array range)
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			config::OMP::localRange( start, end, 0, m );
			/*if( omp_get_thread_num() == 0 ) {
				start = 0;
				end = m;
			} else {
				start = end = m;
			} // DBG <-- IF ENABLING THIS, IT WORKS!!!*/
#else
			start = 0;
			end = m;
#endif

			// keeps track of nonzero count for each row we process
			nzc = 0;

#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp critical
 #endif
			std::cout << "\t\t\t Thread "
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
				<< omp_get_thread_num()
 #else
				<< "0"
 #endif
				<< " has range " << start << "--" << end << "\n";
#endif

			// note: in the OpenMP case, the following statement must precede the barrier
			//       that follows it. We cache the upper bound that may be overwritten by
			//       a sibling thread.
			const auto &C_CRS = internal::getCRS( C );
			const IND cached_end_offset = C_CRS.col_start[ end ];

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			// the below for-loop may write to 1) non-local locations in CCS.col_start and
			// 2) C_CRS.col_start[ end ]. We must therefore sync.
			#pragma omp barrier
#endif

			// counting sort, first step
			//  - if crs_only, then the below implements its resize phase
			//  - if not crs_only, then the below is both crucial for the resize phase,
			//    as well as for enabling the insertions of output values in the output
			//    CCS
			for( size_t i = start; i < end - 1; ++i ) {
				mxm_generic_ompPar_get_row_col_counts_kernel< crs_only, phase >(
					nzc, CRS, CCS, A_raw, B_raw, n, inplace,
					coors, i, CRS.col_start[ i + 1 ]
				);
			}
			if( end > start ) {
				mxm_generic_ompPar_get_row_col_counts_kernel< crs_only, phase >(
					nzc, CRS, CCS, A_raw, B_raw, n, inplace,
					coors, end - 1, cached_end_offset
				);
			}

#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp barrier
			#pragma omp single
 #endif
			{
				if( phase == EXECUTE ) {
					std::cout << "\t\t\t CRS start array, locally prefixed = { "
						<< CRS.col_start[ 0 ];
					for( size_t i = 1; i <= m; ++i ) {
						std::cout << ", " << CRS.col_start[ i ];
					}
					std::cout << " }\n";
				}
			}
#endif
			// now do prefix sum phases 2 & 3
			// the above code does phase 1 in-place, but ends with the CRS.col_start
			// array shifted one position to the left (which should be corrected)
			if( phase == EXECUTE ) {
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp single
 #endif
				std::cout << "\t\t\t shifting the CRS start array...\n";
#endif
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
//#if 0 // DBG
				NIT ps_ws, cached_left = nzc;
				#pragma omp barrier
				if( start > 0 ) {
					cached_left = CRS.col_start[ start - 1 ];
				}
				utils::prefixSum_ompPar_phase2< false, NIT >( CRS.col_start, m, ps_ws );
				#pragma omp barrier
				// first shift right
				if( end > start ) {
					if( end == m ) {
						CRS.col_start[ end ] = CRS.col_start[ end - 1 ];
					}
					for( size_t i = end - 2; i >= start && i < end - 1; --i ) {
						CRS.col_start[ i + 1 ] = CRS.col_start[ i ];
					}
					if( start > 0 ) {
						CRS.col_start[ start ] = cached_left;
					} else {
						assert( start == 0 );
						CRS.col_start[ 0 ] = 0;
					}
				}
#ifdef _DEBUG_REFERENCE_BLAS3
				#pragma omp single
				{
					std::cout << "\t\t\t CRS start array after shift: " << CRS.col_start[ 0 ];
					for( size_t i = 1; i <= m; ++i ) {
						std::cout << ", " << CRS.col_start[ i ];
					}
					std::cout << "\n";
				}
#endif
				#pragma omp barrier // TODO FIXME DBG
				                    // this barrier prevents a segfault caused by element 64 being overwritten by a shift-to-right
				                    // a better fix that does not cost a barrier would be great
						    // in fact, the shift-to-right can probably be fused with the below phase 3
				// then finalise prefix-sum
				utils::prefixSum_ompPar_phase3< false, NIT >( CRS.col_start + 1, m, ps_ws );
#else
				// in the sequential case, only a shift is needed
 /*#ifdef _H_GRB_REFERENCE_OMP_BLAS3
	//			#pragma omp barrier // DBG
	//			#pragma omp single  // DBG
 #endif*/// DBG
				for( size_t i = end - 1; i < end; --i ) {
					CRS.col_start[ i + 1 ] = CRS.col_start[ i ];
				}
				CRS.col_start[ 0 ] = 0;
#endif
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp single
 #endif
				{
					std::cout << "\t\t\t final CRS start array: " << CRS.col_start[ 0 ];
					for( size_t i = 1; i <= m; ++i ) {
						std::cout << ", " << CRS.col_start[ i ];
					}
					std::cout << "\n";
				}
#endif
			}
			return SUCCESS;
		}

#ifndef _H_GRB_REFERENCE_OMP_BLAS3
		/**
		 * Meta-data for global buffer management for use with #grb::mxm.
		 *
		 * This meta-data is the same for both the sequential (reference) and shared-
		 * memory parallel (reference_omp) backends.
		 *
		 * This class contains all meta-data necessary to interpret the global buffer
		 * as an array of sparse accumulators (SPAs). The length of the array is given
		 * by a call to #threads(), minus one. It is called that since a call to
		 * #threads() retrieves how many threads can be used to process the call to
		 * #grb::mxm.
		 *
		 * Each SPA has the layout (bitarray, stack, valueArray). These are packed in a
		 * padded byte array, such that each bit array, stack, and value array is
		 * aligned on sizeof(int) bytes.
		 *
		 * @tparam NIT       The nonzero index type.
		 * @tparam ValueType The output matrix value type.
		 */
		template< typename NIT, typename ValueType >
		class MXM_BufferMetaData {

			static_assert( sizeof(NIT) % sizeof(int) == 0, "Unsupported type for NIT; "
				"please submit a bug report!" );

			private:

				/** The size of the offset array */
				size_t m;

				/** The size of the SPA */
				size_t n;

				/** The number of threads supported during a call to #grb::mxm */
				size_t nthreads;

				/** The initial buffer offset */
				size_t bufferOffset;

				/** The size of a single SPA, including bytes needed for padding */
				size_t paddedSPASize;

				/** The number of bytes to pad the SPA array with */
				size_t arrayShift;

				/** The number of bytes to pad the SPA stack with */
				size_t stackShift;

				/**
				 * Given a number of used bytes of the buffer, calculate the available
				 * remainder buffer and return it.
				 *
				 * @param[in]  osize     The size of the buffer (in bytes) that is already
				 *                       in use.
				 * @param[out] remainder Pointer to any remainder buffer.
				 * @param[out] rsize     The size of the remainder buffer.
				 *
				 * If no buffer space is left, \a remainder will be set to <tt>nullptr</tt>
				 * and \a size to <tt>0</tt>.
				 */
				void retrieveRemainderBuffer(
					const size_t osize,
					void * &remainder, size_t &rsize
				) const noexcept {
					const size_t size = internal::template getCurrentBufferSize< char >();
					char * rem = internal::template getReferenceBuffer< char >( size );
					size_t rsize_calc = size - osize;
					rem += osize;
					const size_t mod = reinterpret_cast< uintptr_t >(rem) % sizeof(int);
					if( mod ) {
						const size_t shift = sizeof(int) - mod;
						if( rsize_calc >= shift ) {
							rsize_calc -= shift;
							rem += rsize;
						} else {
							rsize_calc = 0;
							rem = nullptr;
						}
					}
					assert( !(reinterpret_cast< uintptr_t >(rem) % sizeof(int)) );
					// write out
					remainder = rem;
					rsize = rsize_calc;
				}


			public:

				/**
				 * Base constructor.
				 *
				 * @param[in] _m          The length of the offset array.
				 * @param[in] _n          The length of the SPA.
				 * @param[in] max_threads The maximum number of threads.
				 *
				 * \note \a max_threads is a separate input since there might be a need to
				 *       cap the maximum number of threads used based on some analytic
				 *       performance model. Rather than putting such a performance model
				 *       within this class, we make it an obligatory input parameter
				 *       instead.
				 *
				 * \note It is always valid to pass <tt>config::OMP::threads()</tt>.
				 *
				 * \note This class \em will, however, cap the number of threads returned
				 *       to \a _n.
				 */
				MXM_BufferMetaData(
					const size_t _m, const size_t _n,
					const size_t max_threads
				) : m( _m ), n( _n ), arrayShift( 0 ), stackShift( 0 ) {
 #ifdef _DEBUG_REFERENCE_BLAS3
					#pragma omp critical
					std::cout << "\t\t\t computing padded buffer size for a SPA of length "
						<< n << " while leaving space for an additional offset buffer of length "
						<< std::max( m, n ) << "...\n";
 #endif
					// compute bufferOffset
					bufferOffset = (std::max( m, n ) + 1) * sizeof( NIT );

					// compute value buffer size
					const size_t valBufSize = n * sizeof( ValueType );

 #ifdef _DEBUG_REFERENCE_BLAS3
					std::cout << "\t\t\t\t bit-array size has byte-size " <<
						internal::Coordinates< reference >::arraySize( n ) << "\n";
					std::cout << "\t\t\t\t stack has byte-size " <<
						internal::Coordinates< reference >::stackSize( n ) << "\n";
					std::cout << "\t\t\t\t value buffer has byte-size " << valBufSize << "\n";
 #endif

					// compute paddedSPASize
					paddedSPASize =
						internal::Coordinates< reference >::arraySize( n ) +
						internal::Coordinates< reference >::stackSize( n ) +
						valBufSize;
					size_t shift =
						internal::Coordinates< reference >::arraySize( n ) % sizeof(int);
					if( shift != 0 ) {
						arrayShift = sizeof(int) - shift;
						paddedSPASize += arrayShift;
					}
					shift = internal::Coordinates< reference >::stackSize( n ) % sizeof(int);
					if( shift != 0 ) {
						stackShift = sizeof(int) - shift;
						paddedSPASize += stackShift;
					}
					shift = valBufSize % sizeof(int);
					if( shift != 0 ) {
						paddedSPASize += (sizeof(int) - shift);
					}

					// pad bufferOffset
					shift = bufferOffset % sizeof(int);
					if( shift != 0 ) {
						bufferOffset += (sizeof(int) - shift);
					}

					// compute free buffer size
					const size_t freeBufferSize = internal::getCurrentBufferSize< char >() -
						bufferOffset;

					// compute max number of threads
					nthreads = 1 + freeBufferSize / paddedSPASize;
 #ifdef _DEBUG_REFERENCE_BLAS3
					#pragma omp critical
					std::cout << "\t\t\t free buffer size: " << freeBufferSize
						<< ", (padded) SPA size: " << paddedSPASize
						<< " -> supported #threads: " << nthreads << ". "
						<< " The shifts for the bit-array and the stack are " << arrayShift
						<< ", respectively, " << stackShift << "."
						<< "\n";
 #endif
					// cap the final number of selected threads
					if( nthreads > max_threads ) {
						nthreads = max_threads;
					}
					if( nthreads > n ) {
						nthreads = n;
					}
				}

				/** @returns The maximum number of supported threads during #grb::mxm */
				size_t threads() const noexcept {
					return nthreads;
				}

				/**
				 * Requests and returns a global buffer required for a thread-local SPA.
				 *
				 * @param[in] t The thread ID. Must be larger than 0.
				 *
				 * \note Thread 0 employs the SPA allocated with the output matrix.
				 *
				 * @returns Pointer into the global buffer starting at the area reserved for
				 *          the SPA of thread \a t.
				 */
				char * getSPABuffers( size_t t ) const noexcept {
					assert( t > 0 );
					(void) --t;
					char * raw = internal::template getReferenceBuffer< char >(
						bufferOffset + nthreads * paddedSPASize );
					assert( reinterpret_cast< uintptr_t >(raw) % sizeof(int) == 0 );
					raw += bufferOffset;
					assert( reinterpret_cast< uintptr_t >(raw) % sizeof(int) == 0 );
					raw += t * paddedSPASize;
					return raw;
				}

				/**
				 * Retrieves the column offset buffer.
				 *
				 * @param[out] remainder Returns any remainder buffer beyond that of the row
				 *                       offset buffer.
				 * @param[out] rsize     The remainder buffer size \a remainder points to.
				 *
				 * If \a remainder is not a <tt>nullptr</tt> then neither should \a rsize,
				 * and vice versa.
				 *
				 * Retrieving any remainder buffer is optional. The default is to not ask
				 * for them.
				 *
				 * \warning If all buffer memory is used for the column offsets, it may be
				 *          that \a remainder equals <tt>nullptr</tt> and <tt>rsize</tt>
				 *          zero.
				 *
				 * \warning This buffer is only guaranteed exclusive if only the retrieved
				 *          column buffer is used. In particular, if also requesting (and
				 *          using) SPA buffers, the remainder buffer area is shared with
				 *          those SPA buffers, and data races are likely to occur. In other
				 *          words: be very careful with any use of these remainder buffers.
				 *
				 * @returns The column offset buffer.
				 *
				 * \warning This buffer overlaps with the CRS offset buffer. The caller
				 *          must ensure to only ever use one at a time.
				 */
				NIT * getColOffsetBuffer(
					void * * const remainder = nullptr,
					size_t * const rsize = nullptr
				) const noexcept {
					NIT * const ret = internal::template getReferenceBuffer< NIT >( n + 1 );
					if( remainder != nullptr || rsize != nullptr ) {
						assert( remainder != nullptr && rsize != nullptr );
						retrieveRemainderBuffer( (m + 1) * sizeof(NIT), *remainder, *rsize );
					}
					return ret;
				}

				/**
				 * Retrieves the row offset buffer.
				 *
				 * @param[out] remainder Returns any remainder buffer beyond that of the row
				 *                       offset buffer.
				 * @param[out] rsize     The remainder buffer size \a remainder points to.
				 *
				 * If \a remainder is not a <tt>nullptr</tt> then neither should \a rsize,
				 * and vice versa.
				 *
				 * Retrieving any remainder buffer is optional. The default is to not ask
				 * for them.
				 *
				 * \warning If all buffer memory is used for the row offsets, it may be that
				 *          \a remainder equals <tt>nullptr</tt> and <tt>rsize</tt> zero.
				 *
				 * \warning This buffer is only guaranteed exclusive if only the retrieved
				 *          row buffer is used. In particular, if also requesting (and
				 *          using) SPA buffers, the remainder buffer area is shared with
				 *          those SPA buffers, and data races are likely to occur. In other
				 *          words: be very careful with any use of these remainder buffers.
				 *
				 * @returns The row offset buffer.
				 *
				 * \warning This buffer overlaps with the CCS offset buffer. The caller
				 *          must ensure to only ever use one at a time.
				 */
				NIT * getRowOffsetBuffer(
					void * * const remainder = nullptr,
					size_t * const rsize = nullptr
				) const noexcept {
					NIT * const ret = internal::template getReferenceBuffer< NIT >( m + 1 );
					if( remainder != nullptr || rsize != nullptr ) {
						assert( remainder != nullptr && rsize != nullptr );
						retrieveRemainderBuffer( (m + 1) * sizeof(NIT), *remainder, *rsize );
					}
					return ret;
				}

				/**
				 * Shifts a pointer into the global buffer by the bit-array size and its
				 * padding.
				 *
				 * @param[in,out] raw On input: an aligned pointer into the global buffer.
				 *                    On output: an aligned pointer past the bit-array
				 *                    position.
				 */
				void applyArrayShift( char * &raw ) const noexcept {
					const size_t totalShift =
						internal::Coordinates< reference >::arraySize( n ) +
						arrayShift;
 #ifdef _DEBUG_REFERENCE_BLAS3
					std::cout << "\t\t\t shifting input pointer with "
						<< internal::Coordinates< reference >::arraySize( n ) << " + "
						<< arrayShift << " = " << totalShift << "bytes \n";
 #endif
					raw += totalShift;
				}

				/**
				 * Shifts a pointer into the global buffer by the stack size and its
				 * padding.
				 *
				 * @param[in,out] raw On input: an aligned pointer into the global buffer.
				 *                    On output: an aligned pointer past the stack position.
				 */
				void applyStackShift( char * &raw ) const noexcept {
					const size_t totalShift =
						internal::Coordinates< reference >::stackSize( n ) +
						stackShift;
 #ifdef _DEBUG_REFERENCE_BLAS3
					std::cout << "\t\t\t shifting input pointer with "
						<< internal::Coordinates< reference >::arraySize( n ) << " + "
						<< stackShift << " = " << totalShift << "bytes \n";
 #endif
					raw += totalShift;
				}

		};
#endif

		/**
		 * Retrieves the SPA buffers for the calling thread.
		 *
		 * \warning This function must be called from within an OpenMP parallel
		 *          section.
		 *
		 * @param[out]    arr Where the bit-array may be located.
		 * @param[out]    buf Where the stack may be located.
		 * @param[out] valbuf Where the value buffer may be located.
		 *
		 * All above pointers are aligned on sizeof(int) bytes.
		 *
		 * @param[in] md Meta-data for global buffer management.
		 * @param[in]  C The output matrix.
		 *
		 * One thread uses the buffers pre-allocated with the matrix \a C, thus
		 * ensuring at least one thread may perform the #grb::mxm. Any remainder
		 * threads can only help process the #grb::mxm if there is enough global
		 * buffer memory available.
		 *
		 *
		 * \note The global memory has size \f$ \Omega( \mathit{nz} ) \f$, which may
		 *       be several factors (or even asymptotically greater than)
		 *       \f$ \max\{ m, n \} \f$.
		 *
		 * \note In case the application stores multiple matrices, the global buffer
		 *       may additionally be greater than the above note indicates if at least
		 *       one of the other matrices is significantly (or asymptotically) larger
		 *       than the one involved with the #grb::mxm.
		 */
		template<
			typename OutputType,
			typename RIT, typename CIT, typename NIT
		>
		void mxm_ompPar_getSPABuffers(
			char * &arr, char * &buf, OutputType * &valbuf,
			const struct MXM_BufferMetaData< NIT, OutputType > &md,
			Matrix< OutputType, reference, RIT, CIT, NIT > &C
		) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			// other threads use the global buffer to create additional SPAs
			{
				const size_t t = config::OMP::current_thread_ID();
 #ifndef NDEBUG
				const size_t T = config::OMP::current_threads();
				assert( t < T );
 #endif
				if( t > 0 ) {
 #ifdef _DEBUG_REFERENCE_BLAS3
					#pragma omp critical
					std::cout << "\t Thread " << t << " gets buffers from global buffer\n";
 #endif
					char * rawBuffer = md.getSPABuffers( t );
					assert( reinterpret_cast< uintptr_t >(rawBuffer) % sizeof(int) == 0 );
					arr = rawBuffer;
 #ifdef _DEBUG_REFERENCE_BLAS3
					#pragma omp critical
 #endif
					md.applyArrayShift( rawBuffer );
					assert( reinterpret_cast< uintptr_t >(rawBuffer) % sizeof(int) == 0 );
					buf = rawBuffer;
 #ifdef _DEBUG_REFERENCE_BLAS3
					#pragma omp critical
 #endif
					md.applyStackShift( rawBuffer );
					assert( reinterpret_cast< uintptr_t >(rawBuffer) % sizeof(int) == 0 );
					assert( buf != arr );
					valbuf = reinterpret_cast< OutputType * >(rawBuffer);
					assert( static_cast< void * >(valbuf) != static_cast< void * >(buf) );
				} else {
 #ifdef _DEBUG_REFERENCE_BLAS3
					#pragma omp critical
					std::cout << "\t Thread " << t << " gets buffers from matrix storage\n";
 #endif
					// one thread uses the standard matrix buffer
					internal::getMatrixBuffers( arr, buf, valbuf, 1, C );
				}
 #ifdef _DEBUG_REFERENCE_BLAS3
				#pragma omp critical
				{
					std::cout << "\t Thread " << t << " has SPA array @ "
						<< static_cast< void * >( arr ) << " and SPA stack @ "
						<< static_cast< void * >( buf ) << " and SPA values @ "
						<< static_cast< void * >( valbuf ) << "\n";
				}
 #endif
			}
#else
 #ifdef _DEBUG_REFERENCE_BLAS3
			std::cout << "\t Reference backend gets buffers from global buffer\n";
 #endif
			internal::getMatrixBuffers( arr, buf, valbuf, 1, C );
			(void) md;
#endif
		}

		/**
		 * Given a computed new row_start array, moves the old index and value arrays
		 * to new offsets. This leaves precisely enough space for the mxm algorithm to
		 * inject new nonzeroes at the right location for every row.
		 *
		 * The mxm algorithm must know the old bounds, but we do not want to keep the
		 * \a oldOffsets buffer around for too long-- and we do not have to. If there
		 * are gaps that should contain new nonzero entries, we can indicate the first
		 * such gap element in the index array by putting the corresponding matrix
		 * size there-- this value is an invalid index, and thus is taken as an
		 * encoding for the original boundary.
		 *
		 * \note This function is friendly towards sequential code flows.
		 */
		template<
			typename OutputType,
			typename RIT, typename CIT, typename NIT
		>
		void mxm_ompPar_shiftByOffset(
			Matrix< OutputType, reference, RIT, CIT, NIT > &C,
			const NIT *__restrict__ const oldOffsets,
			void * const buffer, const size_t bufferSize
		) {
/*#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			const size_t dbgt = omp_get_num_threads(); // DBG
			omp_set_num_threads( 1 ); // DBG
#endif*/
			const auto &CRS = internal::getCRS( C );
			const size_t m = grb::nrows( C );
			const size_t n = grb::ncols( C );
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
			for( size_t s = 0; s < omp_get_num_threads(); ++s ) {
				if( s == omp_get_thread_num() )
 #endif
				{
					std::cout << "\t\t\t column indices before shift: " << CRS.row_index[ 0 ];
					for( size_t i = 1; i < CRS.col_start[ m ]; ++i ) {
						std::cout << ", " << CRS.row_index[ i ];
					}
					std::cout << std::endl;
				}
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp barrier
			}
 #endif
#endif
			{
				RIT * const workspace = reinterpret_cast< RIT * >(buffer);
				const size_t workspaceSize = bufferSize / sizeof(RIT);
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				grb::utils::unordered_memmove_ompPar(
					CRS.row_index, oldOffsets, CRS.col_start, m, workspace, workspaceSize );
#else
				grb::utils::unordered_memmove_seq(
					CRS.row_index, oldOffsets, CRS.col_start, m, workspace, workspaceSize );
#endif
			}
			if( !std::is_void< OutputType >::value ) {
				OutputType * const workspace = reinterpret_cast< OutputType * >(buffer);
				const size_t workspaceSize = bufferSize / sizeof(OutputType);
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				grb::utils::unordered_memmove_ompPar(
					CRS.values, oldOffsets, CRS.col_start, m, workspace, workspaceSize );
#else
				grb::utils::unordered_memmove_seq(
					CRS.values, oldOffsets, CRS.col_start, m, workspace, workspaceSize );
#endif
			}
			size_t start, end;
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			config::OMP::localRange( start, end, 0, m );
#else
			start = 0;
			end = m;
#endif
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
			for( size_t s = 0; s < omp_get_num_threads(); ++s ) {
				if( s == omp_get_thread_num() )
 #endif
				{
					std::cout << "\t\t\t column indices after shift: "
						<< CRS.row_index[ 0 ];
					for( size_t i = 1; i < CRS.col_start[ m ]; ++i ) {
						std::cout << ", " << CRS.row_index[ i ];
					}
					std::cout << std::endl;
				}
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp barrier
			}
 #endif
#endif
			for( size_t i = start; i < end; ++i ) {
				const size_t nelems = oldOffsets[ i + 1 ] - oldOffsets[ i ];
				const size_t space  = CRS.col_start[ i + 1 ] - CRS.col_start[ i ];
				assert( oldOffsets[ i + 1 ] >= oldOffsets[ i ] );
				assert( CRS.col_start[ i + 1 ] >= CRS.col_start[ i ] );
#ifndef NDEBUG
				// in debug mode, set the entire range to be able to easier catch any
				// related memory issues
				for( size_t k = nelems; k < space; ++k ) {
					CRS.row_index[ CRS.col_start[ i ] + k ] = n;
				}
#else
				// in performance mode, set only the minimum number of elements
				if( space > nelems ) {
					CRS.row_index[ CRS.col_start[ i ] + nelems ] = n;
				}
#endif
			}
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
			for( size_t s = 0; s < omp_get_num_threads(); ++s ) {
				if( s == omp_get_thread_num() )
 #endif
				{
					std::cout << "\t\t\t column indices after shift and fill: "
						<< CRS.row_index[ 0 ];
					for( size_t i = 1; i < CRS.col_start[ m ]; ++i ) {
						std::cout << ", " << CRS.row_index[ i ];
					}
					std::cout << std::endl;
				}
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp barrier
			}
 #endif
#endif
/*#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			omp_set_num_threads( dbgt ); // DBG
#endif*/
		}

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
			typename RIT, typename CIT, typename NIT
		>
		RC mxm_generic(
			Matrix< OutputType, reference, RIT, CIT, NIT > &C,
			const Matrix< InputType1, reference, RIT, CIT, NIT > &A,
			const Matrix< InputType2, reference, RIT, CIT, NIT > &B,
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

#ifdef _DEBUG_REFERENCE_BLAS3
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
			auto &CRS_raw = internal::getCRS( C );
			auto &CCS_raw = internal::getCCS( C );

			const bool inplace = grb::nnz( C ) > 0;

			// a basic analytic model based on the number of nonzeroes
			size_t max_threads = config::OMP::threads();
			{
				size_t target_nnz = 0;
				if( phase == EXECUTE ) {
					target_nnz = grb::capacity( C );
				} else {
					assert( phase == RESIZE );
					target_nnz = std::max( grb::nnz( A ), grb::nnz( B ) );
				}
				const size_t nnz_based_nthreads =
					target_nnz / config::CACHE_LINE_SIZE::value();
				if( nnz_based_nthreads < max_threads ) {
					max_threads = nnz_based_nthreads;
				}
#ifdef _DEBUG_REFERENCE_BLAS3
				std::cout << "\t simple analytic model selects max threads of "
					<< max_threads << "\n";
#endif
			}

			MXM_BufferMetaData< NIT, OutputType > bufferMD( m, n, max_threads );

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			// derive number of threads
			size_t nthreads = bufferMD.threads();
 //#ifdef _DEBUG_REFERENCE_BLAS3
			std::cout << "\t mxm_generic will use " << nthreads << " threads\n";
 //#endif DBG
#endif

			// resize phase logic
			if( phase == RESIZE ) {
				size_t nzc = 0;
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp parallel num_threads( nthreads )
#endif
				{
					// get thread-local buffers
					char * arr = nullptr;
					char * buf = nullptr;
					OutputType * valbuf = nullptr;
					mxm_ompPar_getSPABuffers( arr, buf, valbuf, bufferMD, C );

					// do count
					size_t local_nzc;
					mxm_generic_ompPar_get_row_col_counts< descr, crs_only, RESIZE >(
						local_nzc, CRS_raw, CCS_raw,
						m, n, A, B, C,
						arr, buf,
						inplace
					);
					#pragma omp atomic update
					nzc += local_nzc;
				}
				if( !crs_only ) {
					// do final resize
					const RC ret = grb::resize( C, nzc );
					return ret;
				} else {
					// we are using an auxiliary CRS that we cannot resize
					// instead, we updated the offset array in the above and can now exit
					// TODO FIXME This assumes crs_only is only used for non-owning views,
					//            which may not always be true.
					static bool print_warning_once = false;
					if( !print_warning_once ) {
						std::cerr << "Warning: grb::mxm( reference ): current implementation has "
						       << "that force_row_major implies a non-owning view of the output "
						       << "storage. If this is not a correct assumption in your use case, "
						       << "please submit a bug report.\n";
						print_warning_once = true;
					}
					CRS_raw.col_start[ m ] = nzc;
					return SUCCESS;
				}
			}

			// execute phase logic
			assert( phase == EXECUTE );
			bool clear_at_exit = false;
			RC ret = SUCCESS;
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp parallel num_threads( nthreads )
#endif
			{
				size_t start, end, local_nzc;
				RC local_rc = SUCCESS;

				// get thread-local buffers
				char * arr = nullptr;
				char * buf = nullptr;
				OutputType * valbuf = nullptr;
				mxm_ompPar_getSPABuffers( arr, buf, valbuf, bufferMD, C );
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp critical
 #endif
				std::cout << "\t\t arr @ " << static_cast< void * >(arr) << ", "
					<< "buf @ " << static_cast< void * >(buf) << ", "
					<< "vbf @ " << static_cast< void * >(valbuf) << "\n";

 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp barrier
				#pragma omp single
 #endif
				{
					std::cout << "\t\t CRS_raw original start array = { "
						<< CRS_raw.col_start[ 0 ];
					for( size_t i = 1; i <= m; ++i ) {
						std::cout << ", " << CRS_raw.col_start[ i ];
					}
					std::cout << " }\n";
				}
#endif

				// phase 1: symbolic phase

				NIT * originalRowOffsets = nullptr;
				void * shiftBuffer = nullptr;
				size_t shiftBufferSize = 0;
				if( inplace ) {
					// first save old row offsets
					originalRowOffsets = bufferMD.getRowOffsetBuffer(
						&shiftBuffer, &shiftBufferSize );
					grb::internal::FinalBackend< reference >::memcpy(
						originalRowOffsets, CRS_raw.col_start, (m + 1) * sizeof(NIT) );
				}

				{
					config::OMP::localRange( start, end, 0, m );
	//				std::cout << "#### " << CRS_raw.col_start[ end ] << " " << CRS_raw.col_start[ end - 1 ] << "\n"; // DBG
				}

				local_rc = mxm_generic_ompPar_get_row_col_counts<
					descr, crs_only, EXECUTE
				>(
					local_nzc, CRS_raw, CCS_raw,
					m, n, A, B, C,
					arr, buf,
					inplace
				);

//				assert( CRS_raw.col_start[ 64 ] != 63 ); // DBG

#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp barrier
				#pragma omp single
 #endif
				{
					std::cout << "\t\t CRS_raw start array = { " << CRS_raw.col_start[ 0 ];
					for( size_t i = 1; i <= m; ++i ) {
						std::cout << ", " << CRS_raw.col_start[ i ];
					}
					std::cout << " }\n";
					std::cout << "\t\t CRS_raw values array = { " << CRS_raw.values[ 0 ];
					for( size_t k = 1; k < CRS_raw.col_start[ m ]; ++k ) {
						std::cout << ", " << CRS_raw.values[ k ];
					}
					std::cout << " }\n";
					std::cout << "\t\t CCS column counts = { " << CCS_raw.col_start[ 0 ];
					for( size_t i = 1; i <= n; ++i ) {
						std::cout << ", " << CCS_raw.col_start[ i ];
					}
					std::cout << " }\n";
				}
#endif

				if( local_rc != SUCCESS ) {
					std::cerr << "mxm_generic_ompPar_get_row_col_counts returned " <<
						grb::toString( local_rc );
				} else if( static_cast< size_t >(CRS_raw.col_start[ m ]) >
					grb::capacity( C )
				) {
					std::cerr << "mxm_generic: not enough capacity in output matrix while "
						<< "phase is EXECUTE\n"
						<< "\t capacity: " << grb::capacity( C ) << "\n"
						<< "\t required: " << CRS_raw.col_start[ m ] << "\n";
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
					#pragma omp atomic write
 #endif
					clear_at_exit = true;
				} else {
					// phase 2 (optional): in-place shift (make room for new nonzeroes)

					if( inplace ) {
						// shifting requires CRS.col_start be finalised, but the function that
						// populates that (get_row_col_counts) does not end with a barrier (for
						// performance in the out-of-place case).
						// This barrier can also not be moved until after the below code block
						// that operates on the CCS since that code reuses the buffer in which
						// originalRowOffsets resides. For the same reason, this code block also
						// cannot be interleaved with the below code block.
						// While this does indicate a latency - memory tradeoff opportunity, this
						// implementation presently does not attempt to take benefit.
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
						#pragma omp barrier
 #endif
						mxm_ompPar_shiftByOffset( C, originalRowOffsets,
							shiftBuffer, shiftBufferSize );
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
						#pragma omp barrier
 #endif
					}

					// phase 3: computational phase

					// get thread-shared column offset buffer
					NIT * const C_col_index = bufferMD.getColOffsetBuffer();

					// get thread-local SPA
					internal::Coordinates< reference > coors;
					coors.set_seq( arr, false, buf, n );

					// prefix sum for C_col_index, fused with setting CCS_raw.col_start to zero
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
					config::OMP::localRange( start, end, 0, n + 1 );
#else
					start = 0;
					end = n + 1;
#endif

#ifndef NDEBUG
					if( !crs_only && start < end && start == 0 ) {
						assert( CCS_raw.col_start[ 0 ] == 0 );
					}
#endif

					if( !crs_only ) {
						// this is a manually fused for-loop from start to end
						C_col_index[ start ] = 0;
						if( start < end ) {
							for( size_t j = start + 1; j < end - 1; ++j ) {
								CCS_raw.col_start[ j ] += CCS_raw.col_start[ j - 1 ];
								C_col_index[ j ] = 0;
							}
							CCS_raw.col_start[ end - 1 ] += CCS_raw.col_start[ end - 2 ];
							if( end <= n ) {
								C_col_index[ end - 1 ] = 0;
							}
						}
						// end manually fused for-loop

#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
						#pragma omp barrier
						#pragma omp single
 #endif
						{
							std::cout << "\t\t CCS start array, locally prefixed = { "
								<< CCS_raw.col_start[ 0 ];
							for( size_t i = 1; i <= n; ++i ) {
								std::cout << ", " << CCS_raw.col_start[ i ];
							}
							std::cout << " }\n";
						}
#endif

						// now finish up prefix-sum on CCS_raw
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
						NIT ps_ws;
						#pragma omp barrier
						utils::prefixSum_ompPar_phase2< false >( CCS_raw.col_start, n + 1, ps_ws );
						#pragma omp barrier
						utils::prefixSum_ompPar_phase3< false >( CCS_raw.col_start, n + 1, ps_ws );
#endif
					}
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
					#pragma omp barrier
					#pragma omp single
 #endif
					{
						std::cout << "\t\t CCS start array = { " << CCS_raw.col_start[ 0 ];
						for( size_t i = 1; i <= n; ++i ) {
							std::cout << ", " << CCS_raw.col_start[ i ];
						}
						std::cout << " }\n";
					}
#endif
#ifndef NDEBUG
					if( !crs_only && start < end && end == n + 1 ) {
						assert( CRS_raw.col_start[ m ] == CCS_raw.col_start[ n ] );
					}
#endif

					// use previously computed CCS offset array to update CCS during the
					// computational phase
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
					config::OMP::localRange( start, end, 0, m );
					/*if( omp_get_thread_num() == 0 ) {
						start = 0;
						end = m;
						std::cout << "!!!! " << CRS_raw.col_start[ end ] << " " << CRS_raw.col_start[ end - 1 ] << "\n"; // DBG
					} else {
						start = end = m;
					} // DBG with this code, the bug still persists(!!!)*/
#else
					start = 0;
					end = m;
#endif
					if( start < end && start == 0 ) {
						CRS_raw.col_start[ 0 ] = 0;
					}
					local_nzc = CRS_raw.col_start[ start ];
					for( size_t i = start; i < end; ++i ) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
 #ifdef _DEBUG_REFERENCE_BLAS3
						#pragma omp critical
 #endif
#endif
						coors.clear_seq();
						if( inplace ) {
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
							#pragma omp critical
 #endif
							std::cout << "\t\t in-place mxm, output matrix has "
								<< (CRS_raw.col_start[ i + 1 ] - CRS_raw.col_start[ i ])
								<< " nonzeroes on row " << i << " at offset "
								<< CRS_raw.col_start[ i ] << " to " << CRS_raw.col_start[ i + 1 ]
								<< "\n";
#endif
							for(
								auto k = CRS_raw.col_start[ i ];
								k < CRS_raw.col_start[ i + 1 ];
								++k
							) {
								const auto index = CRS_raw.row_index[ k ];
								if( index == static_cast< NIT >(n) ) {
									break;
								}
								if( !coors.assign( index ) ) {
									valbuf[ index ] = CRS_raw.getValue( k,
										monoid.template getIdentity< OutputType >() );
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
									#pragma omp critical
 #endif
									std::cout << "\t\t\t pre-existing nonzero at "
										<< i << ", " << index << ": value = " << valbuf[ index ] << "\n";
#endif
								} else {
#ifndef NDEBUG
									const bool repeated_column_indices_detected = false;
									assert( repeated_column_indices_detected );
#endif
								}
							}
						}
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
						#pragma omp critical
 #endif
						std::cout << "\t\t mxm processes row " << i << " which contains "
							<< (A_raw.col_start[ i + 1 ] - A_raw.col_start[ i ]) << " nonzeroes\n"
							<< "\t\t valbuf @ " << static_cast< void * >(valbuf) << ", t = "
							<< omp_get_thread_num() << "\n";
#endif
						for( auto k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
							const size_t k_col = A_raw.row_index[ k ];
							for( auto l = B_raw.col_start[ k_col ];
								l < B_raw.col_start[ k_col + 1 ];
								++l
							) {
								const size_t l_col = B_raw.row_index[ l ];
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
								#pragma omp critical
 #endif
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
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
									#pragma omp critical
 #endif
									std::cout << "\t C( " << i << ", " << l_col << " ) now reads "
										<< valbuf[ l_col ] << "\n";
#endif
								} else {
									OutputType temp = monoid.template getIdentity< OutputType >();
									(void) grb::apply( temp,
										A_raw.getValue( k,
											mulMonoid.template getIdentity< typename Operator::D1 >() ),
										B_raw.getValue( l,
											mulMonoid.template getIdentity< typename Operator::D2 >() ),
										oper );
									(void) grb::foldl( valbuf[ l_col ], temp, monoid.getOperator() );
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
									#pragma omp critical
 #endif
									{
										std::cout << "\t C( " << i << ", " << l_col << " ) += " << temp
											<< "\n";
										std::cout << "\t C( " << i << ", " << l_col << " ) now reads "
											<< valbuf[ l_col ] << "\n";
									}
#endif
								}
							}
						}
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
						#pragma omp critical
 #endif
						std::cout << "\t will write out " << coors.nonzeroes() << " nonzeroes "
							<< "to row " << i << "\n";
#endif
						for( size_t k = 0; k < coors.nonzeroes(); ++k ) {
							const size_t j = coors.index( k );
							// update CRS
							CRS_raw.row_index[ local_nzc ] = j;
							// this above index store is superfluous for some entries in the
							// in-place case, but we, at this point, have lost the information which
							// part of the row_index array did not need updating. We could prevent it
							// at the cost of m+1 memory, but we chose not to expend that and just
							// take this hit
							CRS_raw.setValue( local_nzc, valbuf[ j ] );
							// update CCS
							if( !crs_only ) {
								size_t atomic_offset;
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
								// TODO It may be better to switch to a parallel transposition
								//      if there are too many threads / conflict updates (FIXME)
								#pragma omp atomic capture
#endif
								{
									atomic_offset = C_col_index[ j ];
#ifndef _H_GRB_REFERENCE_OMP_BLAS3
									(void)
#endif
										C_col_index[ j ]++;
								}
								const size_t CCS_index = atomic_offset + CCS_raw.col_start[ j ];
								assert( CCS_index < CCS_raw.col_start[ j + 1 ] );
								CCS_raw.row_index[ CCS_index ] = i;
								CCS_raw.setValue( CCS_index, valbuf[ j ] );
							}
							// update count
							(void) ++local_nzc;
						}
#ifdef _DEBUG_REFERENCE_BLAS3
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
						#pragma omp critical
 #endif
						std::cout << "\t wrote out " << (local_nzc - CRS_raw.col_start[ i ])
							<< " nonzeroes to row " << i << "\n";
#endif
						assert( static_cast< size_t >(CRS_raw.col_start[ i + 1 ]) == local_nzc );
					}
#ifndef NDEBUG
 #ifdef _H_GRB_REFERENCE_OMP_BLAS3
					#pragma omp barrier
					#pragma omp single
 #endif
					{
						if( !crs_only ) {
							for( size_t j = 0; j < n; ++j ) {
								assert( CCS_raw.col_start[ j + 1 ] - CCS_raw.col_start[ j ] ==
									C_col_index[ j ] );
							}
						}
					}
#endif
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
					#pragma omp critical
#endif
					{
						if( local_rc != SUCCESS && ret == SUCCESS ) {
							ret = local_rc;
						}
					}
				}
			}

			if( ret == SUCCESS ) {
				assert( !clear_at_exit );
				if( !crs_only ) {
					assert( CRS_raw.col_start[ m ] == CCS_raw.col_start[ n ] );
				}
				// set final number of nonzeroes in output matrix
				internal::setCurrentNonzeroes( C, CRS_raw.col_start[ m ] );
			}

			if( clear_at_exit ) {
				const RC clear_rc = clear( C );
				if( clear_rc != SUCCESS ) { return PANIC; }
			}

			// done
			return ret;
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

#ifdef _DEBUG_REFERENCE_BLAS3
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
#ifdef _DEBUG_REFERENCE_BLAS3
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
#ifdef _DEBUG_REFERENCE_BLAS3
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
#ifdef _DEBUG_REFERENCE_BLAS3
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
#ifdef _DEBUG_REFERENCE_BLAS3
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

	namespace internal {

		template<
			Descriptor descr,
			bool left_handed, class Monoid,
			typename InputType, typename IOType,
			typename RIT, typename CIT, typename NIT
		>
		RC fold_unmasked_generic(
			IOType &x,
			const Matrix< InputType, reference, RIT, CIT, NIT > &A,
			const Monoid &monoid = Monoid()
		) {
			static_assert(
				!std::is_void< InputType >::value,
				"This implementation of folding a matrix into a scalar only applies to "
				"non-pattern matrices. Please submit a bug report."
			);
#ifdef _DEBUG_REFERENCE_BLAS3
			std::cout << "In grb::internal::fold_unmasked_generic( reference )\n";
#endif
			static_assert( !(descr & descriptors::add_identity),
				"internal::fold_unmasked_generic should not be called with add_identity descriptor" );

			constexpr bool transpose =
				descr & grb::descriptors::transpose_matrix ||
				descr & grb::descriptors::transpose_left;

			assert( !(nnz( A ) == 0 || nrows( A ) == 0 || ncols( A ) == 0) );

			if( (descr & descriptors::force_row_major) && transpose ) {
				std::cerr << "this implementation requires that force_row_major and "
					<< "transpose descriptors are mutually exclusive\n";
				return UNSUPPORTED;
			}

			const auto &A_raw = transpose ? getCCS( A ) : getCRS( A );
			const size_t A_nnz = nnz( A );

			const auto &op = monoid.getOperator();
			RC rc = SUCCESS;

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp parallel
#endif
			{
				RC local_rc = rc;
				size_t start = 0;
				size_t end = A_nnz;
				auto local_x = monoid.template getIdentity< typename Monoid::D3 >();
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				config::OMP::localRange( start, end, 0, A_nnz );
#endif
				for( size_t idx = start; idx < end; ++idx ) {
					// Get A value
					// note: default-constructed InputType is fine as default-constructible is
					//       mandatory for POD types and the callees ensure that this function
					//       is never called on void matrices.
					const InputType a_val = A_raw.getValue( idx, InputType() );
					// Compute the fold for this coordinate
					if( left_handed ) {
						local_rc = grb::foldl< descr >( local_x, a_val, op );
					} else {
						local_rc = grb::foldr< descr >( a_val, local_x, op );
					}
#ifdef NDEBUG
					(void) local_rc;
#else
					assert( local_rc == RC::SUCCESS );
#endif
				}

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp critical
#endif
				{ // Reduction with the global result (critical section if OpenMP)
					if( left_handed ) {
						local_rc = grb::foldl< descr >( x, local_x, op );
					} else {
						local_rc = grb::foldr< descr >( local_x, x, op );
					}
					assert( local_rc == RC::SUCCESS );
					rc = rc ? rc : local_rc;
				}
			}

			// done
			return rc;
		}

		template<
			Descriptor descr,
			bool left_handed, class Ring,
			typename InputType, typename IOType,
			typename RIT, typename CIT, typename NIT
		>
		RC fold_unmasked_generic__add_identity(
			IOType &x,
			const Matrix< InputType, reference, RIT, CIT, NIT > &A,
			const Ring &ring = Ring()
		) {
#ifdef _DEBUG_REFERENCE_BLAS3
			std::cout << "In grb::internal::fold_unmasked_generic__add_identity( "
				<< "reference )\n";
#endif

			const size_t min_m_n = nrows( A ) > ncols( A )
				? ncols( A )
				: nrows( A );

			const auto &zero = left_handed
				? ring.template getZero< typename Ring::D2 >()
				: ring.template getZero< typename Ring::D1 >();

			const auto &one = left_handed
				? ring.template getOne< typename Ring::D2 >()
				: ring.template getOne< typename Ring::D1 >();

			const auto &op = ring.getAdditiveOperator();
			RC rc = RC::SUCCESS;

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp parallel
#endif
			{
				size_t start = 0;
				size_t end = min_m_n;
				RC local_rc = rc;
				auto local_x = zero;
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				config::OMP::localRange( start, end, 0, min_m_n );
#endif

				for( size_t k = start; k < end; ++k ) {
					if( left_handed ) {
						local_rc = grb::foldl< descr >( local_x, one, op );
					} else {
						local_rc = grb::foldr< descr >( one, local_x, op );
					}
				}

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				#pragma omp critical
#endif
				{ // Reduction with the global result (critical section if OpenMP)
					if( local_rc == RC::SUCCESS ) {
						if( left_handed ) {
							local_rc = grb::foldl< descr >( x, local_x, op );
						} else {
							local_rc = grb::foldr< descr >( local_x, x, op );
						}
					}
					rc = rc ? rc : local_rc;
				} // end of the critical region

			} // end of the parallel region

			if( rc != RC::SUCCESS ) {
				return rc;
			}

			return internal::fold_unmasked_generic<
				descr & (~descriptors::add_identity),
				left_handed,
				typename Ring::AdditiveMonoid
			>( x, A );
		}

		template<
			Descriptor descr = descriptors::no_operation,
			bool left_handed, class Monoid,
			typename InputType, typename IOType, typename MaskType,
			typename RIT_A, typename CIT_A, typename NIT_A,
			typename RIT_M, typename CIT_M, typename NIT_M,
			typename CoordinatesRowTranslator,
			typename CoordinatesColTranslator
		>
		RC fold_masked_generic(
			IOType &x,
			const Matrix< InputType, reference, RIT_A, CIT_A, NIT_A > &A,
			const Matrix< MaskType, reference, RIT_M, CIT_M, NIT_M > &mask,
			const InputType * const one_A = nullptr,
			const CoordinatesRowTranslator &row_union_to_global =
				CoordinatesRowTranslator(),
			const CoordinatesColTranslator &col_union_to_global =
				CoordinatesColTranslator(),
			const Monoid &monoid = Monoid()
		) {
#ifndef NDEBUG
			if( descr & descriptors::add_identity ) {
				assert( one_A != nullptr );
			}
#endif
#ifdef _DEBUG_REFERENCE_BLAS3
			std::cout << "In grb::internal::fold_masked_generic( reference )\n";
#endif
			constexpr bool add_identity = (descr & descriptors::add_identity);
			constexpr bool transpose_mask = (descr & descriptors::transpose_right);
			constexpr bool transpose_input = (descr & descriptors::transpose_left);

			const auto &value_zero = left_handed
				? monoid.template getIdentity< typename Monoid::D2 >()
				: monoid.template getIdentity< typename Monoid::D1 >();
			const auto &op = monoid.getOperator();

			assert( !(nnz( mask ) == 0 || nnz( A ) == 0 ||
				nrows( A ) == 0 || ncols( A ) == 0) );

			if( (descr & descriptors::force_row_major) &&
				(transpose_mask || transpose_input)
			) {
#ifdef _DEBUG_REFERENCE_BLAS3
				std::cerr << "force_row_major and transpose descriptors are mutually "
					<< "exclusive in this implementation\n";
#endif
				return UNSUPPORTED;
			}
			const auto &A_raw = transpose_input ? getCCS( A ) : getCRS( A );
			const size_t m_A = transpose_input ? ncols( A ) : nrows( A );
			const size_t n_A = transpose_input ? nrows( A ) : ncols( A );
			const auto &mask_raw = transpose_mask ? getCCS( mask ) : getCRS( mask );
			const auto &localRow2Global = transpose_input
				? col_union_to_global
				: row_union_to_global;
			const auto &localCol2Global = transpose_input
				? row_union_to_global
				: col_union_to_global;


#ifndef NDEBUG
			const size_t m_mask = transpose_mask ? ncols( mask ) : nrows( mask );
			const size_t n_mask = transpose_mask ? nrows( mask ) : ncols( mask );
			assert( m_A == m_mask && n_A == n_mask );
#endif

			// retrieve buffers
			char * arr = nullptr, * buf = nullptr;
			InputType * vbuf = nullptr;
			internal::getMatrixBuffers( arr, buf, vbuf, 1, A );
			// end buffer retrieval

			// initialisations
			internal::Coordinates< reference > coors;
			coors.set( arr, false, buf, n_A );
			// end initialisations

			RC rc = SUCCESS;
			// Note for review: by indicating that these two values can only be incremented
			// by 1, as the col_start arrays are the results of a prefix_sum,
			// it might simplify optimisation for the compiler
			auto mask_k = mask_raw.col_start[ 0 ];
			auto A_k = A_raw.col_start[ 0 ];

			for( size_t i = 0; i < m_A; ++i ) {
				coors.clear();

				for(; mask_k < mask_raw.col_start[ i + 1 ]; ++mask_k ) {
					if( !utils::interpretMatrixMask< descr, MaskType >(
						true, mask_raw.getValues(), mask_k )
					) { continue; }

					const auto j = mask_raw.row_index[ mask_k ];
					coors.assign( j );
				}

				// If there is no value in the mask for this row
				if( coors.nonzeroes() == 0 ) { continue; }

				for(; A_k < A_raw.col_start[ i + 1 ]; ++A_k ) {
					const auto j = A_raw.row_index[ A_k ];

					// Skip if the coordinate is not in the mask
					if( !coors.assigned( j ) ) { continue; }

					const auto global_i = localRow2Global( i );
					const auto global_j = localCol2Global( j );

					const InputType identity_increment = add_identity &&
						(
							( global_i == global_j )
							? *one_A
							: value_zero
						);

					// Get A value
					const InputType a_val = A_raw.getValue( A_k, identity_increment );
					// Compute the fold for this coordinate
					if( left_handed ) {
						rc = rc ? rc : grb::foldl< descr >( x, a_val + identity_increment, op );
					} else {
						rc = rc ? rc : grb::foldr< descr >( a_val + identity_increment, x, op );
					}
				}
			}

			return rc;
		}

		template<
			Descriptor descr,
			bool left_handed, class Ring,
			typename InputType, typename IOType, typename MaskType,
			typename RIT_A, typename CIT_A, typename NIT_A,
			typename RIT_M, typename CIT_M, typename NIT_M,
			typename CoordinatesRowTranslator,
			typename CoordinatesColTranslator
		>
		RC fold_masked_generic__add_identity(
			IOType &x,
			const Matrix< InputType, reference, RIT_A, CIT_A, NIT_A > &A,
			const Matrix< MaskType, reference, RIT_M, CIT_M, NIT_M > &mask,
			const CoordinatesRowTranslator &row_union_to_global =
				CoordinatesRowTranslator(),
			const CoordinatesColTranslator &col_union_to_global =
				CoordinatesColTranslator(),
			const Ring &ring = Ring()
		) {
#ifdef _DEBUG_REFERENCE_BLAS3
			std::cout << "In grb::internal::fold_masked_generic__add_identity( "
				<< "reference )\n";
#endif
			const auto& one = left_handed
				? ring.template getOne< typename Ring::D2 >()
				: ring.template getOne< typename Ring::D1 >();

			return internal::fold_masked_generic<
				descr,
				left_handed,
				typename Ring::AdditiveMonoid
			>(
				x, A, mask,
				&one,
				row_union_to_global,
				col_union_to_global
			);
		}

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

#ifdef _DEBUG_REFERENCE_BLAS3
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

#ifdef _DEBUG_REFERENCE_BLAS3
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
#endif
			{
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
				size_t start, end;
				config::OMP::localRange( start, end, 0, n + 1 );
#else
				const size_t start = 0;
				const size_t end = n + 1;
#endif
				for( size_t j = start; j < end; ++j ) {
					CCS_raw.col_start[ j ] = 0;
				}
			}
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
#ifdef _DEBUG_REFERENCE_BLAS3
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
#endif
				{
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
					size_t start, end;
					config::OMP::localRange( start, end, 0, n );
#else
					const size_t start = 0;
					const size_t end = n;
#endif
					for( size_t j = start; j < end; ++j ) {
						C_col_index[ j ] = 0;
					}
				}

				// do computations
				size_t nzc = 0;
				C_raw.col_start[ 0 ] = 0;
				for( size_t i = 0; i < m; ++i ) {
					coors1.clear();
					coors2.clear();
#ifdef _DEBUG_REFERENCE_BLAS3
					std::cout << "\t The elements ";
#endif
					for( size_t k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
						const size_t k_col = A_raw.row_index[ k ];
						coors1.assign( k_col );
						valbuf[ k_col ] = A_raw.getValue( k,
							mulMonoid.template getIdentity< typename Operator::D1 >() );
#ifdef _DEBUG_REFERENCE_BLAS3
						std::cout << "A( " << i << ", " << k_col << " ) = " << A_raw.getValue( k,
							mulMonoid.template getIdentity< typename Operator::D1 >() ) << ", ";
#endif
					}
#ifdef _DEBUG_REFERENCE_BLAS3
					std::cout << "are multiplied pairwise with:\n";
#endif
					for( size_t l = B_raw.col_start[ i ]; l < B_raw.col_start[ i + 1 ]; ++l ) {
						const size_t l_col = B_raw.row_index[ l ];
						if( coors1.assigned( l_col ) ) {
							coors2.assign( l_col );
							(void) grb::foldl( valbuf[ l_col ], B_raw.getValue(
									l,
									mulMonoid.template getIdentity< typename Operator::D2 >()
								),
							oper );
#ifdef _DEBUG_REFERENCE_BLAS3
							std::cout << "\t\t B( " << i << ", " << l_col << " ) = "
								<< B_raw.getValue(
									l, mulMonoid.template getIdentity< typename Operator::D2 >() )
								<< " to yield C( " << i << ", " << l_col << " ) = " << valbuf[ l_col ]
								<< "\n";
#endif
						}
					}
#ifdef _DEBUG_REFERENCE_BLAS3
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
						(void) ++nzc;
					}
					C_raw.col_start[ i + 1 ] = nzc;
#ifdef _DEBUG_REFERENCE_BLAS3
					std::cout << "\n";
#endif
				}

#ifndef NDEBUG
				for( size_t j = 0; j < n; ++j ) {
					assert( CCS_raw.col_start[ j + 1 ] - CCS_raw.col_start[ j ] ==
						C_col_index[ j ] );
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
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
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

#ifdef _DEBUG_REFERENCE_BLAS3
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
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
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

	template<
		Descriptor descr = descriptors::no_operation,
		class SelectionOperator,
		typename Tin,
		typename RITin, typename CITin, typename NITin,
		typename Tout,
		typename RITout, typename CITout, typename NITout
	>
	RC select(
		Matrix< Tout, reference, RITout, CITout, NITout > &out,
		const Matrix< Tin, reference, RITin, CITin, NITin > &in,
		const SelectionOperator &op = SelectionOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!is_object< Tin >::value &&
			!is_object< Tout >::value
		>::type * const = nullptr
	) {
#ifdef _DEBUG_REFERENCE_BLAS3
		std::cout << "In grb::select( reference )\n";
#endif
		// static sanity checks
		static_assert(
			!(descr & descriptors::no_casting) && (
				(std::is_void< Tin >::value && std::is_void< Tout >::value) ||
					std::is_same< Tin, Tout >::value
			),
			"grb::select (reference): "
			"input and output matrix types must match"
		);

		// dispatch
		return internal::select_generic< descr >(
			out,
			in,
			op,
			[]( size_t i ) { return i; },
			[]( size_t j ) { return j; },
			phase
		);
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType, typename MaskType,
		typename RIT_A, typename CIT_A, typename NIT_A,
		typename RIT_M, typename CIT_M, typename NIT_M
	>
	RC foldr(
		const Matrix< InputType, reference, RIT_A, CIT_A, NIT_A > &A,
		const Matrix< MaskType, reference, RIT_M, CIT_M, NIT_M > &mask,
		IOType &x,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( !std::is_void< InputType >::value,
			"grb::foldr( reference, IOType <- [[IOType]], monoid, masked ): "
			"the provided matrix may not be a pattern matrix."
			"Possible fix: provide a semiring instead of a ring"
		);
		static_assert( grb::is_commutative< Monoid >::value,
			"grb::foldr( reference, IOType <- [[IOType]], monoid, masked ): "
			"the provided monoid must be commutative (but is not)"
		);
		static_assert( (std::is_same< typename Monoid::D1, InputType >::value),
			"grb::foldr( reference, IOType <- [[IOType]], monoid, masked ): "
			"called with a prefactor input type that does not match the first domain of "
			"the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D2, IOType >::value),
			"grb::foldr( reference, IOType <- [[IOType]], monoid, masked ): "
			"called with a postfactor input type that does not match the second domain "
			"of the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldr( reference, IOType <- [[IOType]], monoid, masked ): "
			"called with an output type that does not match the output domain of the "
			"given monoid"
		);
		static_assert( !(descr & descriptors::add_identity),
			"grb::foldr( reference, IOType <- [[IOType]], monoid, masked ): "
			"the use of the add_identity descriptor requires a semiring, but a monoid "
			"was given"
		);
		static_assert( !(
				(descr & descriptors::invert_mask) &&
				(descr & descriptors::structural)
			), "grb::foldr( reference, IOType <- [[IOType]], monoid, masked ): "
			"may not select an inverted structural mask for matrices"
		);
		static_assert( !(descr & descriptors::add_identity),
			"grb::foldr( reference, IOType <- [[IOType]], monoid ): "
			"the use of the add_identity descriptor requires a semiring, but a monoid "
			"was given"
		);
#ifdef _DEBUG_REFERENCE_BLAS3
		std::cout << "In grb::foldr( reference, mask, matrix, monoid )\n";
#endif
		// first check whether we can dispatch
		if( grb::nrows( mask ) == 0 && grb::ncols( mask ) == 0 ) {
			return foldr< descr >( A, x, monoid );
		}

		// dynamic checks
		if( grb::nrows( A ) != grb::nrows( mask ) ||
			grb::ncols( A ) != grb::ncols( mask )
		) {
			return RC::MISMATCH;
		}

		// check for trivial op
		if( grb::nnz( A ) == 0 || grb::nnz( mask ) == 0 ||
			grb::nrows( A ) == 0 || grb::ncols( A ) == 0
		) {
			return RC::SUCCESS;
		}

		// do fold
		const InputType * const nil = nullptr;
		return internal::fold_masked_generic< descr, false, Monoid >(
			x, A, mask,
			nil,
			[]( size_t i ){ return i; },
			[]( size_t i ){ return i; }
		);
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType,
		typename RIT, typename CIT, typename NIT
	>
	RC foldr(
		const Matrix< InputType, reference, RIT, CIT, NIT > &A,
		IOType &x,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( !std::is_void< InputType >::value,
			"grb::foldr( reference, IOType <- [[IOType]], monoid ): "
			"the provided matrix may not be a pattern matrix."
			"Possible fix: provide a semiring instead of a ring"
		);
		static_assert( grb::is_commutative< Monoid >::value,
			"grb::foldr( reference, IOType <- [[IOType]], monoid ): "
			"the provided monoid must be commutative (but is not)"
		);
		static_assert( (std::is_same< typename Monoid::D1, InputType >::value),
			"grb::foldr( reference, IOType <- [[IOType]], monoid ): "
			"called with a prefactor input type that does not match the first domain of "
			"the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D2, IOType >::value),
			"grb::foldr( reference, IOType <- [[IOType]], monoid ): "
			"called with a postfactor input type that does not match the second domain "
			"of the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldr( reference, IOType <- [[IOType]], monoid ): "
			"called with an output type that does not match the output domain of the "
			"given monoid"
		);
		static_assert( !(descr & descriptors::add_identity),
			"grb::foldr( reference, IOType <- [[IOType]], monoid ): "
			"the use of the add_identity descriptor requires a semiring, but a monoid "
			"was given"
		);
#ifdef _DEBUG_REFERENCE_BLAS3
		std::cout << "In grb::foldr( reference, matrix, monoid )\n";
#endif
		// check for trivial op
		if( grb::nnz( A ) == 0 || grb::nrows( A ) == 0 || grb::ncols( A ) == 0 ) {
			return RC::SUCCESS;
		}

		// otherwise, go ahead
		return internal::fold_unmasked_generic< descr, false, Monoid >(
			x, A, monoid
		);
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Semiring,
		typename InputType, typename IOType,
		typename RIT, typename CIT, typename NIT
	>
	RC foldr(
		const Matrix< InputType, reference, RIT, CIT, NIT > &A,
		IOType &x,
		const Semiring &semiring = Semiring(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_semiring< Semiring >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( (std::is_same< typename Semiring::D3, InputType >::value),
			"grb::foldr( reference, IOType <- [[IOType]], semiring ): "
			"called with a prefactor input type that does not match the third domain of "
			"the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, IOType >::value),
			"grb::foldr( reference, IOType <- [[IOType]], semiring ): "
			"called with a postfactor input type that does not match the fourth domain "
			"of the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, IOType >::value),
			"grb::foldr( reference, IOType <- [[IOType]], semiring ): "
			"called with an output type that does not match the fourth domain of the "
			"given semiring"
		);
#ifdef _DEBUG_REFERENCE_BLAS3
		std::cout << "In grb::foldr( reference, matrix, semiring )\n";
#endif
		// check for trivial op
		if( grb::nnz( A ) == 0 || grb::nrows( A ) == 0 || grb::ncols( A ) == 0 ) {
			return RC::SUCCESS;
		}

		// otherwise, go ahead
		if( descr & descriptors::add_identity ) {
			return internal::fold_unmasked_generic__add_identity<
				descr, false, Semiring
			>(
				x, A, semiring
			);
		}
		return internal::fold_unmasked_generic<
			descr & (~descriptors::add_identity),
			false,
			typename Semiring::AdditiveMonoid
		>(
			x, A, semiring.getAdditiveMonoid()
		);
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Semiring,
		typename InputType, typename IOType, typename MaskType,
		typename RIT_A, typename CIT_A, typename NIT_A,
		typename RIT_M, typename CIT_M, typename NIT_M
	>
	RC foldr(
		const Matrix< InputType, reference, RIT_A, CIT_A, NIT_A > &A,
		const Matrix< MaskType, reference, RIT_M, CIT_M, NIT_M > &mask,
		IOType &x,
		const Semiring &semiring = Semiring(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Semiring >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( (std::is_same< typename Semiring::D3, InputType >::value),
			"grb::foldr( reference, IOType <- [[IOType]], semiring, masked ): "
			"called with a prefactor input type that does not match the third domain of "
			"the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, IOType >::value),
			"grb::foldr( reference, IOType <- [[IOType]], semiring, masked ): "
			"called with a postfactor input type that does not match the fourth domain "
			"of the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, IOType >::value),
			"grb::foldr( reference, IOType <- [[IOType]], semiring, masked ): "
			"called with an output type that does not match the fourth domain of the "
			"given semiring"
		);
		static_assert( !(
				(descr & descriptors::invert_mask) &&
				(descr & descriptors::structural)
			), "grb::foldr( reference, IOType <- [[IOType]], semiring, masked ): "
			"may not select an inverted structural mask for matrices"
		);
		// first check whether we can dispatch
		if( grb::nrows( mask ) == 0 && grb::ncols( mask ) == 0 ) {
			return foldr< descr >( A, x, semiring );
		}

		// dynamic checks
		if( grb::nrows( A ) != grb::nrows( mask ) ||
			grb::ncols( A ) != grb::ncols( mask )
		) {
			return RC::MISMATCH;
		}

		// check for trivial op
		if( grb::nnz( A ) == 0 || grb::nnz( mask ) == 0 ||
			grb::nrows( A ) == 0 || grb::ncols( A ) == 0
		) {
			return RC::SUCCESS;
		}

		// do fold
		return internal::fold_masked_generic__add_identity<
			descr, false, Semiring
		>(
			x, A, mask,
			[](size_t i) { return i; },
			[](size_t i) { return i; },
			semiring
		);
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType, typename MaskType,
		typename RIT_A, typename CIT_A, typename NIT_A,
		typename RIT_M, typename CIT_M, typename NIT_M
	>
	RC foldl(
		IOType &x,
		const Matrix< InputType, reference, RIT_A, CIT_A, NIT_A > &A,
		const Matrix< MaskType, reference, RIT_M, CIT_M, NIT_M > &mask,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( !std::is_void< InputType >::value,
			"grb::foldl( reference, IOType <- [[IOType]], monoid ): "
			"the provided matrix may not be a pattern matrix."
			"Possible fix: provide a semiring instead of a ring"
		);
		static_assert( grb::is_commutative< Monoid >::value,
			"grb::foldl( reference, IOType <- [[IOType]], monoid ): "
			"the provided monoid must be commutative (but is not)"
		);
		static_assert( (std::is_same< typename Monoid::D1, IOType >::value),
			"grb::foldl( reference, IOType <- [[IOType]], monoid, masked ): "
			"called with a prefactor input type that does not match the first domain of "
			"the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D2, InputType >::value),
			"grb::foldl( reference, IOType <- [[IOType]], monoid, masked ): "
			"called with a postfactor input type that does not match the second domain "
			"of the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldl( reference, IOType <- [[IOType]], monoid, masked ): "
			"called with an output type that does not match the output domain of the "
			"given monoid"
		);
		static_assert( !(
				(descr & descriptors::invert_mask) &&
				(descr & descriptors::structural)
			), "grb::foldl( reference, IOType <- [[IOType]], monoid, masked ): "
			"may not select an inverted structural mask for matrices"
		);
		static_assert( !(descr & descriptors::add_identity),
			"grb::foldl( reference, IOType <- [[IOType]], monoid ): "
			"the use of the add_identity descriptor requires a semiring, but a monoid "
			"was given"
		);
#ifdef _DEBUG_REFERENCE_BLAS3
		std::cout << "In grb::foldl( reference, mask, matrix, monoid )\n";
#endif
		// first check whether we can dispatch
		if( grb::nrows( mask ) == 0 && grb::ncols( mask ) == 0 ) {
			return foldl< descr >( x, A, monoid );
		}

		// dynamic checks
		if( grb::nrows( A ) != grb::nrows( mask ) ||
			grb::ncols( A ) != grb::ncols( mask )
		) {
			return RC::MISMATCH;
		}

		// check for trivial op
		if( grb::nnz( A ) == 0 || grb::nnz( mask ) == 0 ||
			grb::nrows( A ) == 0 || grb::ncols( A ) == 0
		) {
			return RC::SUCCESS;
		}

		// do fold
		const InputType * const nil = nullptr;
		return internal::fold_masked_generic<
			descr, true, Monoid
		>(
			x, A, mask,
			nil,
			[]( size_t i ){ return i; },
			[]( size_t i ){ return i; }
		);
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType,
		typename RIT, typename CIT, typename NIT
	>
	RC foldl(
		IOType &x,
		const Matrix< InputType, reference, RIT, CIT, NIT > &A,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( !std::is_void< InputType >::value,
			"grb::foldl( reference, IOType <- [[IOType]], monoid ): "
			"the provided matrix may not be a pattern matrix."
			"Possible fix: provide a semiring instead of a ring"
		);
		static_assert( grb::is_commutative< Monoid >::value,
			"grb::foldl( reference, IOType <- [[IOType]], monoid ): "
			"the provided monoid must be commutative (but is not)"
		);
		static_assert( (std::is_same< typename Monoid::D1, IOType >::value),
			"grb::foldl( reference, IOType <- [[IOType]], monoid ): "
			"called with a prefactor input type that does not match the first domain of "
			"the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D2, InputType >::value),
			"grb::foldl( reference, IOType <- [[IOType]], monoid ): "
			"called with a postfactor input type that does not match the second domain "
			"of the given monoid"
		);
		static_assert( (std::is_same< typename Monoid::D3, IOType >::value),
			"grb::foldl( reference, IOType <- [[IOType]], monoid ): "
			"called with an output type that does not match the output domain of the "
			"given monoid"
		);
		static_assert( !(descr & descriptors::add_identity),
			"grb::foldl( reference, IOType <- [[IOType]], monoid ): "
			"the use of the add_identity descriptor requires a semiring, but a monoid "
			"was given"
		);
#ifdef _DEBUG_REFERENCE_BLAS3
		std::cout << "In grb::foldl( reference, matrix, monoid )\n";
#endif
		// check for trivial op
		if( grb::nnz( A ) == 0 || grb::nrows( A ) == 0 || grb::ncols( A ) == 0 ) {
			return RC::SUCCESS;
		}

		// do folding
		return internal::fold_unmasked_generic< descr, true, Monoid >(
			x, A, monoid
		);
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Semiring,
		typename InputType, typename IOType,
		typename RIT, typename CIT, typename NIT
	>
	RC foldl(
		IOType &x,
		const Matrix< InputType, reference, RIT, CIT, NIT > &A,
		const Semiring &semiring = Semiring(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_semiring< Semiring >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( (std::is_same< typename Semiring::D3, IOType >::value),
			"grb::foldl( reference, IOType <- [[IOType]], semiring ): "
			"called with a prefactor input type that does not match the third domain of "
			"the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, InputType >::value),
			"grb::foldl( reference, IOType <- [[IOType]], semiring ): "
			"called with a postfactor input type that does not match the fourth domain "
			"of the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, IOType >::value),
			"grb::foldl( reference, IOType <- [[IOType]], semiring ): "
			"called with an output type that does not match the fourth domain of the "
			"given semiring"
		);
#ifdef _DEBUG_REFERENCE_BLAS3
		std::cout << "In grb::foldl( reference, matrix, semiring )\n";
#endif
		// check for trivial op
		if( grb::nnz( A ) == 0 || grb::nrows( A ) == 0 || grb::ncols( A ) == 0 ) {
			return RC::SUCCESS;
		}

		// do folding
		if( descr & descriptors::add_identity ) {
			return internal::fold_unmasked_generic__add_identity<
				descr, true, Semiring
			>(
				x, A, semiring
			);
		} else {
			return internal::fold_unmasked_generic<
				descr & (~descriptors::add_identity),
				true,
				typename Semiring::AdditiveMonoid
			>(
				x, A, semiring.getAdditiveMonoid()
			);
		}
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Semiring,
		typename InputType, typename IOType, typename MaskType,
		typename RIT_A, typename CIT_A, typename NIT_A,
		typename RIT_M, typename CIT_M, typename NIT_M
	>
	RC foldl(
		IOType &x,
		const Matrix< InputType, reference, RIT_A, CIT_A, NIT_A > &A,
		const Matrix< MaskType, reference, RIT_M, CIT_M, NIT_M > &mask,
		const Semiring &semiring = Semiring(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Semiring >::value, void
		>::type * const = nullptr
	) {
		// static checks
		static_assert( (std::is_same< typename Semiring::D4, IOType >::value),
			"grb::foldl( reference, IOType <- [[IOType]], semiring, masked ): "
			"called with a prefactor input type that does not match the fourth domain "
			"of the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D4, InputType >::value),
			"grb::foldl( reference, IOType <- [[IOType]], semiring, masked ): "
			"called with a postfactor input type that does not match the fourth domain "
			"of the given semiring"
		);
		static_assert( (std::is_same< typename Semiring::D3, IOType >::value),
			"grb::foldl( reference, IOType <- [[IOType]], semiring, masked ): "
			"called with an output type that does not match the third domain of the "
			"given semiring"
		);
		static_assert( !(
				(descr & descriptors::invert_mask) &&
				(descr & descriptors::structural)
			), "grb::foldl( reference, IOType <- [[IOType]], semiring, masked ): "
			"may not select an inverted structural mask for matrices"
		);
#ifdef _DEBUG_REFERENCE_BLAS3
		std::cout << "In grb::foldl( reference, mask, matrix, semiring )\n";
#endif
		// first check whether we can dispatch
		if( grb::nrows( mask ) == 0 && grb::ncols( mask ) == 0 ) {
			return foldl< descr >( x, A, semiring );
		}

		// dynamic checks
		if( grb::nrows( A ) != grb::nrows( mask ) ||
			grb::ncols( A ) != grb::ncols( mask )
		) {
			return RC::MISMATCH;
		}

		// check for trivial op
		if( grb::nnz( A ) == 0 || grb::nnz( mask ) == 0 ||
			grb::nrows( A ) == 0 || grb::ncols( A ) == 0
		) {
			return RC::SUCCESS;
		}

		// do fold
		return internal::fold_masked_generic__add_identity<
			descr, true, Semiring
		>(
			x, A, mask,
			[](size_t i) { return i; },
			[](size_t i) { return i; },
			semiring
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

