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
 * @date 10th of August, 2016
 */

#if ! defined _H_GRB_REFERENCE_MATRIX || defined _H_GRB_REFERENCE_OMP_MATRIX
#define _H_GRB_REFERENCE_MATRIX

#include <numeric> //std::accumulate
#include <sstream> //std::stringstream
#include <algorithm>

#include <assert.h>

#include <graphblas/backends.hpp>
#include <graphblas/base/matrix.hpp>
#include <graphblas/config.hpp>
#include <graphblas/utils.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/reference/compressed_storage.hpp>
#include <graphblas/reference/init.hpp>
#include <graphblas/type_traits.hpp>
#include <graphblas/utils/autodeleter.hpp>
#include <graphblas/utils/DMapper.hpp>
#include <graphblas/utils/pattern.hpp> //for help with dealing with pattern matrix input
#include <graphblas/type_traits.hpp>

#include <graphblas/algorithms/hpcg/ndim_matrix_builders.hpp>
#include <graphblas/utils/MatrixVectorIterator.hpp>
#include "non_zero_wrapper.h"


#include "forward.hpp"

#ifndef __REF_BACKEND_NAME__
 #define __REF_BACKEND_NAME__ reference
#endif


namespace grb {

#ifndef _H_GRB_REFERENCE_OMP_MATRIX
	namespace internal {

		template< typename D >
		class SizeOf {
		public:
			static constexpr size_t value = sizeof( D );
		};

		template<>
		class SizeOf< void > {
		public:
			static constexpr size_t value = 0;
		};

		template<
			typename ValType, typename ColType, typename IndType,
			Backend backend = config::default_backend
		>
		const grb::Matrix< ValType, backend, ColType, ColType, IndType >
		wrapCRSMatrix(
			const ValType *__restrict__ const value_array,
			const ColType *__restrict__ const index_array,
			const IndType *__restrict__ const offst_array,
			const size_t m, const size_t n
		);

		template<
			typename ValType, typename ColType, typename IndType,
			Backend backend = config::default_backend
		>
		grb::Matrix< ValType, backend, ColType, ColType, IndType >
		wrapCRSMatrix(
			ValType *__restrict__ const value_array,
			ColType *__restrict__ const index_array,
			IndType *__restrict__ const offst_array,
			const size_t m, const size_t n, const size_t cap,
			char * const buf1 = nullptr, char * const buf2 = nullptr,
			ValType *__restrict__ const buf3 = nullptr
		);

	} // namespace internal
#endif

	namespace internal {

		template< typename D, typename RIT, typename CIT, typename NIT >
		const size_t & getNonzeroCapacity(
			const grb::Matrix< D, __REF_BACKEND_NAME__, RIT, CIT, NIT > &A
		) noexcept {
			return A.cap;
		}
		template< typename D, typename RIT, typename CIT, typename NIT >
		const size_t & getCurrentNonzeroes(
			const grb::Matrix< D, __REF_BACKEND_NAME__, RIT, CIT, NIT > &A
		) noexcept {
			return A.nz;
		}
		template< typename D, typename RIT, typename CIT, typename NIT >
		void setCurrentNonzeroes(
			grb::Matrix< D, __REF_BACKEND_NAME__, RIT, CIT, NIT > &A,
			const size_t nnz
		) noexcept {
			A.nz = nnz;
		}

		/**
		 * \internal
		 *
		 * Retrieves internal SPA buffers.
		 *
		 * @param[out] coorArr Pointer to the bitmask array
		 * @param[out] coorBuf Pointer to the stack
		 * @param[out] valBuf  Pointer to the value buffer
		 * @param[in]    k     If 0, the row-wise SPA is returned
		 *                     If 1, the column-wise SPA is returned
		 *                     Any other value is not allowed
		 * @param[in]    A     The matrix of which to return the associated SPA
		 *                     data structures.
		 *
		 * @tparam InputType The type of the value buffer.
		 *
		 * \endinternal
		 */
		template< typename InputType, typename RIT, typename CIT, typename NIT >
		void getMatrixBuffers(
			char * &coorArr, char * &coorBuf, InputType * &valbuf,
			const unsigned int k,
			const grb::Matrix< InputType, __REF_BACKEND_NAME__, RIT, CIT, NIT > &A
		) noexcept {
			assert( k < 2 );
			coorArr = const_cast< char * >( A.coorArr[ k ] );
			coorBuf = const_cast< char * >( A.coorBuf[ k ] );
			valbuf = const_cast< InputType * >( A.valbuf[ k ] );
		}

		template<
			Descriptor descr,
			bool input_dense, bool output_dense,
			bool masked,
			bool left_handed,
			template< typename > class One,
			typename IOType,
			class AdditiveMonoid, class Multiplication,
			typename InputType1, typename InputType2, typename InputType3,
			typename RowColType, typename NonzeroType,
			typename Coords
		>
		void vxm_inner_kernel_scatter( RC &rc,
			Vector< IOType, __REF_BACKEND_NAME__, Coords > &destination_vector,
			IOType * __restrict__ const &destination,
			const size_t &destination_range,
			const Vector< InputType1, __REF_BACKEND_NAME__, Coords > &source_vector,
			const InputType1 * __restrict__ const &source,
			const size_t &source_index,
			const internal::Compressed_Storage< InputType2, RowColType, NonzeroType > &matrix,
			const Vector< InputType3, __REF_BACKEND_NAME__, Coords > &mask_vector,
			const InputType3 * __restrict__ const &mask,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const std::function< size_t( size_t ) > &src_local_to_global,
			const std::function< size_t( size_t ) > &dst_global_to_local
		);

		template<
			Descriptor descr,
			bool masked, bool input_masked, bool left_handed,
			template< typename > class One,
			class AdditiveMonoid, class Multiplication,
			typename IOType, typename InputType1, typename InputType2,
			typename InputType3, typename InputType4,
			typename Coords, typename RIT, typename CIT, typename NIT
		>
		RC vxm_generic(
			Vector< IOType, __REF_BACKEND_NAME__, Coords > &u,
			const Vector< InputType3, __REF_BACKEND_NAME__, Coords > &mask,
			const Vector< InputType1, __REF_BACKEND_NAME__, Coords > &v,
			const Vector< InputType4, __REF_BACKEND_NAME__, Coords > &v_mask,
			const Matrix< InputType2, __REF_BACKEND_NAME__, RIT, CIT, NIT > &A,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const std::function< size_t( size_t ) > &row_l2g,
			const std::function< size_t( size_t ) > &row_g2l,
			const std::function< size_t( size_t ) > &col_l2g,
			const std::function< size_t( size_t ) > &col_g2l
		);

#ifdef _H_GRB_REFERENCE_OMP_MATRIX


		template< typename itertype,
			typename RowIndexType,
			typename ValType,
			typename NonzeroIndexType,
			typename RowGetterType,
			typename ColGetterType
		> RC count_sort_omp(
			itertype it,
			size_t nz,
			size_t imin,
			size_t imax,
			size_t nsize,
			size_t *prefix_sum_buffer,
			const size_t prefix_sum_buffer_size,
			RowIndexType *row_values_buffer, // MUST have size nz, or be nullptr
			Compressed_Storage< ValType, RowIndexType, NonzeroIndexType > &CXX,
			RowGetterType row_getter,
			ColGetterType col_getter,
			bool save_by_i ){

			(void)col_getter;

			if ( nz < 1) {
#ifndef NDEBUG
				std::cout << "num non-zeroes is 0" << std::endl;
#endif
				return RC::ILLEGAL;
			}

			if ( imin == imax ) {
#ifdef _DEBUG
				std::cout << "imin == imax = " << imax << ": return" << std::endl ;
#endif
				return RC::ILLEGAL;
			}

			if ( prefix_sum_buffer_size <= 1 ) {
#ifdef _DEBUG
				std::cerr << "error: buffersize < 1" << std::endl;
#endif
				return RC::ILLEGAL;
			}

			const size_t values_range = imax - imin + 1;

			//local buffer size
			const size_t per_thread_buffer_size = prefix_sum_buffer_size / nsize;
#ifdef _DEBUG
			if ( nsize == 1 ) {
				std::cout << "readjustment:  per_thread_buffer_size =" << per_thread_buffer_size
					<< ": using full buffer with one thread" << std::endl;
			}
#endif

			/*
			while( per_thread_buffer_size <= 1 && nsize >= 2 ) {
				(void)--nsize;
				per_thread_buffer_size = prefix_sum_buffer_size / nsize;
#ifdef _DEBUG
				std::cout << "readjustment: per_thread_buffer_size = " << per_thread_buffer_size
					<< ": cannot use all threads" << std::endl
					<< "\tnsize = " << nsize << std::endl ;
#endif
			}
			*/

			const size_t bucketlen =  ( values_range == per_thread_buffer_size ? 0
				: values_range / per_thread_buffer_size ) + 1;

#ifdef _DEBUG
			std::cout << "nz =" << nz << ", nsize = " << nsize << ", bufferlen = "
				<< per_thread_buffer_size << ", bucketlen = " << bucketlen << std::endl;
#endif
#ifdef _CSO_DEBUG
			std::cout << "values_range= " << values_range << std::endl;
			std::cout << "nz= " << nz << std::endl;
			std::cout << "per_thread_buffer_size= " << per_thread_buffer_size << std::endl;
			std::cout << "prefix_sum_buffer_size= " << prefix_sum_buffer_size << std::endl;
			std::cout << "nsize= " << nsize << std::endl;
			std::cout << "minimum buffer size= " << nz  + per_thread_buffer_size << std::endl;
			std::cout << "bucketlen= " << bucketlen << std::endl;
#endif
			#pragma omp parallel for
			for ( size_t i = 0; i < prefix_sum_buffer_size; i++ ) prefix_sum_buffer[ i ] = 0;


			// firstprivate(it) is MANDATORY for each thread to have a private copy
			// and NOT change the master's object (for the following loops)
			#pragma omp parallel firstprivate(it)
			{
				// count the number of elements per bucket
				const size_t irank = omp_get_thread_num();
				if( irank < nsize ) {
					size_t i = irank * nz / nsize;
					it += i;
					for ( ; i < ( ( irank + 1 ) * nz ) / nsize; i++) {
						const size_t bucket_num = ( row_getter( it ) - imin ) / bucketlen;
						const size_t offset = irank * per_thread_buffer_size + bucket_num;
						prefix_sum_buffer[ offset ]++ ;
						/*
						printf( "thread %lu: increment at %lu, val %lu\n", irank, irank * per_thread_buffer_size + bucket_num,
							prefix_sum_buffer[ irank * per_thread_buffer_size + bucket_num ] );
						*/
						(void)++it;
					}
				}
			}

#ifdef _CSO_DEBUG
			std::cout << "after first step:" << std::endl;
			for( size_t s = 0; s < prefix_sum_buffer_size; s++ ) {
				std::cout << s << ": " << prefix_sum_buffer[s] << std::endl;
			}
#endif

			// cumulative sum along threads, for each bucket
			#pragma omp for
			for ( size_t i = 0; i < per_thread_buffer_size; i++ ) {
				for ( size_t irank = 1; irank < nsize; irank++ ) {
					prefix_sum_buffer[ irank * per_thread_buffer_size + i ] +=
						prefix_sum_buffer[ ( irank - 1 ) * per_thread_buffer_size + i ];
				}
			}
#ifdef _CSO_DEBUG
			std::cout << "after second step:" << std::endl;
			for( size_t s = 0; s < prefix_sum_buffer_size; s++ ) {
				std::cout << s << ": " << prefix_sum_buffer[s] << std::endl;
			}
#endif

			// cumulative sum for each bucket on last thread, to get the final size of each bucket

			//this loop still not parallel, no significant speedup
			for ( size_t i = 1; i < per_thread_buffer_size; i++ ) {
				prefix_sum_buffer[ ( nsize - 1 ) * per_thread_buffer_size + i ] +=
					prefix_sum_buffer[ ( nsize - 1 ) * per_thread_buffer_size + i - 1 ];
			}
#ifdef _CSO_DEBUG
			std::cout << "after third step:" << std::endl;
			for( size_t s = 0; s < prefix_sum_buffer_size; s++ ) {
				std::cout << s << ": " << prefix_sum_buffer[s] << std::endl;
			}
#endif

			// propagate cumulative sums for each bucket on each thread, to get the final offset
			#pragma omp for
			for ( size_t irank = 0; irank < nsize - 1; irank++ ) {
				for ( size_t i = 1; i < per_thread_buffer_size; i++ ) {
					prefix_sum_buffer[ irank * per_thread_buffer_size + i ] +=
						prefix_sum_buffer[ ( nsize - 1 ) * per_thread_buffer_size + i - 1 ];
				}
			}
#ifdef _CSO_DEBUG
			std::cout << "after fourth step:" << std::endl;
			for( size_t s = 0; s < prefix_sum_buffer_size; s++ ) {
				std::cout << s << ": " << prefix_sum_buffer[s] << std::endl;
			}
#endif

			// record value inside CXX data structure, with inter-bucket sorting
			// but no intra-bucket sorting
			#pragma omp parallel firstprivate(it,row_values_buffer)
			{
				const size_t irank = omp_get_thread_num();
				if( irank < nsize ) {
					size_t i = irank * nz / nsize;
					it += i;
					for ( ; i < ( ( irank + 1 ) * nz ) / nsize; i++) {
						RowIndexType row = row_getter( it );
						const size_t bucket_num = ( row - imin ) / bucketlen;
						auto i1 = irank * per_thread_buffer_size + bucket_num;
						(void)--prefix_sum_buffer[ i1 ];
						//printf( "thread %lu, set %u, %d into %lu\n", irank, row_getter( it ), it.v(), prefix_sum_buffer[ i1 ] );
						CXX.recordValue( prefix_sum_buffer[ i1 ], save_by_i, it );
						if( row_values_buffer != nullptr ) {
							row_values_buffer[ prefix_sum_buffer[ i1 ] ] = row;
						}
						//printf( "thread %lu, %lu -> %u, %u, %d\n", irank, offset, row_getter( it ), col_getter( it ), it.v() );
						(void)++it;
					}
				}
			}

#ifdef _CSO_DEBUG
			std::cout << "CSR data before sorting:" << std::endl;
			for( size_t s = 0; s < nz; s++ ) {
				std::cout << s << ": ";
				if ( row_values_buffer != nullptr ) {
					std::cout << row_values_buffer[ s ] << ", ";
				}
				std::cout << CXX.row_index[s] << ", " << CXX.values[s] << std::endl;
			}
#endif

			if( bucketlen == 1UL ) {
				// if( bucketlen == 1UL ) (i.e. we had full parallelism), we are
				// almost done: we must only write the values of the prefix sum
				// into col_start
				assert( prefix_sum_buffer_size >= ccs_col_buffer_size );
				#pragma omp parallel for schedule( static ), num_threads( num_threads )
				for ( size_t i = 0; i < ccs_col_buffer_size ; i++ ) {
					storage.col_start[ i ] = prefix_sum_buffer[ i ];
				}
#ifdef _CSO_DEBUG
				std::cout << "after col_start:" << std::endl;
				for( size_t s = 0; s < imax - imin + 1; s++ ) {
					std::cout << s << ": " << CXX.col_start[ s ] << std::endl;
				}

				std::cout << " **** array already fully sorted in the bucket sort  ***** \n";
#endif
				return RC::SUCCESS;
			}

			//sort buckets using std:sort


#ifdef _CSO_DEBUG
			std::cout << " **** sorting buckets  ***** \n";
#endif

			assert( row_values_buffer != nullptr );

#ifdef _CSO_DEBUG
			std::fill( CXX.col_start, CXX.col_start + imax + 1, 0);
			std::cout << "col_start before sorting:" << std::endl;
			for( size_t s = 0; s < imax - imin + 1; s++ ) {
				std::cout << s << ": " << CXX.col_start[ s ] << std::endl;
			}
#endif


			#pragma omp parallel for
			for( size_t i = 0; i < per_thread_buffer_size; i++ ) {
				// ith bucket borders
				const size_t ipsl_min = prefix_sum_buffer[ i ];
				const size_t ipsl_max = ( i + 1 < per_thread_buffer_size ) ? prefix_sum_buffer[ i + 1 ] : nz;

				if( ipsl_max == ipsl_min ) {
					// the rows are all empty, then done here
					continue ;
				}
				//do the sort
#ifdef _CSO_DEBUG
				printf( "-- thread %d, sort %lu  - %lu\n", omp_get_thread_num(), ipsl_min, ipsl_max );
#endif
				NZIterator< ValType, RowIndexType, NonzeroIndexType > begin( CXX, row_values_buffer, ipsl_min );
				NZIterator< ValType, RowIndexType, NonzeroIndexType > end( CXX, row_values_buffer, ipsl_max );
				std::sort( begin, end );


				const size_t max_row = std::min( ( i + 1 ) * bucketlen, imax ); // use it to cap current_destination
#ifdef _CSO_DEBUG
				printf( ">> max_row=%lu\n", max_row );
#endif
				RowIndexType previous_destination = i * bucketlen;

				// INIT: populate initial value with existing count
				CXX.col_start[ previous_destination ] = ipsl_min;
#ifdef _CSO_DEBUG
				printf( "thread %d, init write %lu to pos %u\n", omp_get_thread_num(), ipsl_min, previous_destination );
#endif

				size_t count = ipsl_min;
				size_t previous_count = count;
				size_t row_buffer_index = ipsl_min; // start from next
				while ( row_buffer_index < ipsl_max ) {
					const RowIndexType current_row = row_values_buffer[ row_buffer_index ];
					const RowIndexType current_destination = current_row + 1;
					// fill previous rows [previous_destination + 1, current_destinatio) if skipped because empty
					if ( previous_destination + 1 <= current_row ) {
#ifdef _CSO_DEBUG
						printf( "thread %d, write %lu in range [%u - %u)\n",
							omp_get_thread_num(), count, previous_destination + 1, current_destination );
#endif
						std::fill( CXX.col_start + previous_destination + 1, CXX.col_start + current_destination, previous_count );
					}
					// count occurrences of 'current_row'
					for( ; row_values_buffer[ row_buffer_index ] == current_row && row_buffer_index < ipsl_max;
						row_buffer_index++, count++ );
					assert( current_destination <= max_row );
					// if current_destination < max_row, then write the count;
					// otherwise, the next thread will do it in INIT
					if ( current_destination < max_row ){
						CXX.col_start[ current_destination ] = count;
#ifdef _CSO_DEBUG
						printf( "thread %d, write %lu to pos %u\n", omp_get_thread_num(), count, current_destination );
#endif
					}
					previous_destination = current_destination;
					previous_count = count;
				}
				// if the rows in [ previous_destination + 1, max_row ) are empty,
				// write the count also there, since the loop has skipped them
				if ( previous_destination + 1 < max_row ) {
#ifdef _CSO_DEBUG
					printf( "thread %d, final write %lu in range [%u, %lu)\n",
						omp_get_thread_num(), previous_count, previous_destination + 1, max_row );
#endif
					std::fill( CXX.col_start + previous_destination + 1, CXX.col_start + max_row, previous_count );
				}
			}
			const RowIndexType last_existing_row = row_values_buffer[ nz - 1 ];
#ifdef _CSO_DEBUG
			printf( "final offset %u\n", last_existing_row );
#endif

			if ( last_existing_row + 1 <= imax ) {
#ifdef _CSO_DEBUG
				printf( "final write %lu into [%u, %lu]\n", nz, last_existing_row + 1, imax );
#endif
				std::fill( CXX.col_start + last_existing_row + 1, CXX.col_start + imax + 1, nz );
			}

#ifdef _CSO_DEBUG
			std::cout << "CSR data after sorting:" << std::endl;
			for( size_t s = 0; s < nz; s++ ) {
				std::cout << s << ": ";
				if ( row_values_buffer != nullptr ) {
					std::cout << row_values_buffer[ s ] << ", ";
				}
				std::cout << CXX.row_index[s] << ", " << CXX.values[s] << std::endl;
			}

			std::cout << "col_start after sorting:" << std::endl;
			for( size_t s = 0; s < imax - imin + 1; s++ ) {
				std::cout << s << ": " << CXX.col_start[ s ] << std::endl;
			}
#endif

			//sort buckets using std:sort

			return RC::SUCCESS;

		}

		template< typename rndacc_iterator, typename RowGetterType, typename ColGetterType,
			typename ValType, typename RowIndexType, typename NonzeroIndexType >
		RC __invoke_( size_t amin, size_t amax , size_t nz, rndacc_iterator& _start,
			RowGetterType row_getter, ColGetterType col_getter,
			Compressed_Storage< ValType, RowIndexType, NonzeroIndexType
		> &CRS, bool read_row ) {

			size_t nsize = static_cast< size_t >( omp_get_max_threads() );
			size_t range = amax - amin + 1;

			// ensure enugh parallelism while using a reasonable memory amount: the maximum parallelism
			// is achieved with nsize * range, but this might much larger than nz when there are many cores
			// and the matrix is very sparse ( nz ~= nrows ); hence the std::min()
			const size_t partial_parallel_row_values_buffer_size = (
					( nz * sizeof( RowIndexType ) + config::CACHE_LINE_SIZE::value() - 1 )
					/ config::CACHE_LINE_SIZE::value()
				) * config::CACHE_LINE_SIZE::value(); // round up to config::CACHE_LINE_SIZE::value()

			// buffer to store prefix sums: to ensure good paralelism, allow storing nz elements
			// TODO: this can be further limited by limiting the available parallelism
			const size_t partial_parallel_prefix_sums_buffer_size = nz * sizeof( size_t );

			const size_t partial_parallel_buffer_size = partial_parallel_row_values_buffer_size
				+ partial_parallel_prefix_sums_buffer_size;

			const size_t fully_parallel_buffer_size = nsize * range * sizeof( size_t );

			const bool is_fully_parallel = fully_parallel_buffer_size <= partial_parallel_buffer_size;

//#ifdef _CSO_DEBUG
			if( is_fully_parallel ) {
				std::cout << "fully parallel matrix creation: no extra sorting required" << std::endl;
			} else {
				std::cout << "partially parallel matrix creation: extra sorting required" << std::endl;
			}
//#endif

			size_t bufferlen_tot = is_fully_parallel ? fully_parallel_buffer_size : partial_parallel_buffer_size;
			if ( !internal::ensureReferenceBufsize< unsigned char >( bufferlen_tot ) ) {
//#ifndef NDEBUG
				std::cerr << "Not enough memory available for count_sort_omp buffer" << std::endl;
//#endif
				return RC::OUTOFMEM;
			}
			const size_t prefix_sum_buffer_size = is_fully_parallel ? nsize * range
				: nz;
			//const size_t row_values_buffer_size = is_fully_parallel ? 0 : nz;
			unsigned char * const __buffer = grb::internal::getReferenceBuffer< unsigned char >( bufferlen_tot );
			size_t * pref_sum_buffer = is_fully_parallel ? reinterpret_cast < size_t * >( __buffer ) :
				reinterpret_cast < size_t * >( __buffer + partial_parallel_row_values_buffer_size );
			RowIndexType * row_values_buffer = is_fully_parallel ? nullptr : reinterpret_cast < RowIndexType * >( __buffer );

			return count_sort_omp( _start, nz, amin, amax, nsize, pref_sum_buffer, prefix_sum_buffer_size,
				row_values_buffer, CRS, row_getter, col_getter, read_row );
			internal::forceDeallocBuffer();
		}
#endif


	} // namespace internal

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	size_t nrows( const Matrix< DataType, __REF_BACKEND_NAME__, RIT, CIT, NIT > & ) noexcept;

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	size_t ncols( const Matrix< DataType, __REF_BACKEND_NAME__, RIT, CIT, NIT > & ) noexcept;

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	size_t nnz( const Matrix< DataType, __REF_BACKEND_NAME__, RIT, CIT, NIT > & ) noexcept;

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	RC clear( Matrix< InputType, __REF_BACKEND_NAME__, RIT, CIT, NIT > & ) noexcept;

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	RC resize(
		Matrix< DataType, __REF_BACKEND_NAME__, RIT, CIT, NIT > &,
		const size_t
	) noexcept;

	template<
		class ActiveDistribution, typename Func, typename DataType,
		typename RIT, typename CIT, typename NIT
	>
	RC eWiseLambda(
		const Func f,
		const Matrix< DataType, __REF_BACKEND_NAME__, RIT, CIT, NIT > &A,
		const size_t s, const size_t P
	);

	/**
	 * A GraphBLAS matrix, reference implementation.
	 *
	 * Uses Compressed Column Storage (CCS) plus Compressed Row Storage (CRS).
	 *
	 * \warning This implementation prefers speed over memory efficiency.
	 *
	 * @tparam D The type of a nonzero element.
	 *
	 * \internal
	 * @tparam RowIndexType The type used for row indices
	 * @tparam ColIndexType The type used for column indices
	 * @tparam NonzeroIndexType The type used for nonzero indices
	 * \endinternal
	 */
	template<
		typename D,
		typename RowIndexType,
		typename ColIndexType,
		typename NonzeroIndexType
	>
	class Matrix< D, __REF_BACKEND_NAME__, RowIndexType, ColIndexType, NonzeroIndexType > {

		static_assert( !grb::is_object< D >::value,
			"Cannot create an ALP matrix of ALP objects!" );

		/* *********************
		        BLAS2 friends
		   ********************* */

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		friend size_t nrows(
			const Matrix< DataType, __REF_BACKEND_NAME__, RIT, CIT, NIT > &
		) noexcept;

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		friend size_t ncols(
			const Matrix< DataType, __REF_BACKEND_NAME__, RIT, CIT, NIT > &
		) noexcept;

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		friend size_t nnz(
			const Matrix< DataType, __REF_BACKEND_NAME__, RIT, CIT, NIT > &
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend RC clear(
			Matrix< InputType, __REF_BACKEND_NAME__, RIT, CIT, NIT > &
		) noexcept;

		template< typename DataType, typename RIT, typename CIT, typename NIT  >
		friend RC resize(
			Matrix< DataType, __REF_BACKEND_NAME__, RIT, CIT, NIT > &,
			const size_t
		) noexcept;

		template<
			class ActiveDistribution, typename Func, typename DataType,
			typename RIT, typename CIT, typename NIT
		>
		friend RC eWiseLambda(
			const Func,
			const Matrix< DataType, __REF_BACKEND_NAME__, RIT, CIT, NIT > &,
			const size_t, const size_t
		);

		template<
			Descriptor descr,
			bool input_dense, bool output_dense, bool masked, bool left_handed,
			template< typename > class One,
			typename IOType,
			class AdditiveMonoid, class Multiplication,
			typename InputType1, typename InputType2,
			typename InputType3,
			typename RowColType, typename NonzeroType,
			typename Coords
		>
		friend void internal::vxm_inner_kernel_scatter(
			RC &rc,
			Vector< IOType, __REF_BACKEND_NAME__, Coords > &destination_vector,
			IOType * __restrict__ const &destination,
			const size_t &destination_range,
			const Vector< InputType1, __REF_BACKEND_NAME__, Coords > &source_vector,
			const InputType1 * __restrict__ const &source,
			const size_t &source_index,
			const internal::Compressed_Storage<
				InputType2, RowColType, NonzeroType
			> &matrix,
			const Vector< InputType3, __REF_BACKEND_NAME__, Coords > &mask_vector,
			const InputType3 * __restrict__ const &mask,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const std::function< size_t( size_t ) > &src_local_to_global,
			const std::function< size_t( size_t ) > &dst_global_to_local
		);

		template<
			Descriptor descr,
			bool masked, bool input_masked, bool left_handed,
			template< typename > class One,
			class AdditiveMonoid, class Multiplication,
			typename IOType, typename InputType1, typename InputType2,
			typename InputType3, typename InputType4,
			typename Coords, typename RIT, typename CIT, typename NIT
		>
		friend RC internal::vxm_generic(
			Vector< IOType, __REF_BACKEND_NAME__, Coords > &u,
			const Vector< InputType3, __REF_BACKEND_NAME__, Coords > &mask,
			const Vector< InputType1, __REF_BACKEND_NAME__, Coords > &v,
			const Vector< InputType4, __REF_BACKEND_NAME__, Coords > &v_mask,
			const Matrix< InputType2, __REF_BACKEND_NAME__, RIT, CIT, NIT > &A,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const std::function< size_t( size_t ) > &row_l2g,
			const std::function< size_t( size_t ) > &row_g2l,
			const std::function< size_t( size_t ) > &col_l2g,
			const std::function< size_t( size_t ) > &col_g2l
		);

		/* ********************
		        IO friends
		   ******************** */

		template<
			Descriptor descr, typename InputType,
			typename RIT, typename CIT, typename NIT,
			typename fwd_iterator
		>
		friend RC buildMatrixUnique(
			Matrix< InputType, __REF_BACKEND_NAME__, RIT, CIT, NIT > &,
			fwd_iterator, const fwd_iterator,
			const IOMode
		);

		friend internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType > &
		internal::getCRS<>(
			Matrix<
				D, __REF_BACKEND_NAME__,
				RowIndexType, ColIndexType, NonzeroIndexType
			> &A
		) noexcept;

		friend const internal::Compressed_Storage<
			D,
			RowIndexType, NonzeroIndexType
		> & internal::getCRS<>(
			const Matrix<
				D, __REF_BACKEND_NAME__,
				RowIndexType, ColIndexType, NonzeroIndexType
			> &A
		) noexcept;

		friend internal::Compressed_Storage< D, ColIndexType, NonzeroIndexType > &
		internal::getCCS<>(
			Matrix<
				D, __REF_BACKEND_NAME__,
				RowIndexType, ColIndexType, NonzeroIndexType
			> &A
		) noexcept;

		friend const internal::Compressed_Storage<
			D, ColIndexType, NonzeroIndexType
		> & internal::getCCS<>(
			const Matrix<
				D, __REF_BACKEND_NAME__,
				RowIndexType, ColIndexType, NonzeroIndexType
			> &A
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend const size_t & internal::getNonzeroCapacity(
			const grb::Matrix< InputType, __REF_BACKEND_NAME__, RIT, CIT, NIT > &
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend const size_t & internal::getCurrentNonzeroes(
			const grb::Matrix< InputType, __REF_BACKEND_NAME__, RIT, CIT, NIT > &
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend void internal::setCurrentNonzeroes(
			grb::Matrix< InputType, __REF_BACKEND_NAME__, RIT, CIT, NIT > &, const size_t
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend void internal::getMatrixBuffers(
			char *&, char *&, InputType *&,
			const unsigned int,
			const grb::Matrix< InputType, __REF_BACKEND_NAME__, RIT, CIT, NIT > &
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend uintptr_t getID(
			const Matrix< InputType, __REF_BACKEND_NAME__, RIT, CIT, NIT > &
		);

		/* *************************
		   Friend internal functions
		   ************************* */

		friend const grb::Matrix<
			D, __REF_BACKEND_NAME__,
			ColIndexType, ColIndexType, NonzeroIndexType
		>
		internal::wrapCRSMatrix< D, ColIndexType, NonzeroIndexType, __REF_BACKEND_NAME__ >(
			const D *__restrict__ const,
			const ColIndexType *__restrict__ const,
			const NonzeroIndexType *__restrict__ const,
			const size_t, const size_t
		);

		friend grb::Matrix<
			D, __REF_BACKEND_NAME__,
			ColIndexType, ColIndexType, NonzeroIndexType
		>
		internal::wrapCRSMatrix< D, ColIndexType, NonzeroIndexType, __REF_BACKEND_NAME__ >(
			D *__restrict__ const,
			ColIndexType *__restrict__ const,
			NonzeroIndexType *__restrict__ const,
			const size_t, const size_t, const size_t,
			char * const, char * const,
			D *__restrict__ const
		);

		/* ***********************************
		   Friend other matrix implementations
		   *********************************** */

		template<
			typename InputType, Backend backend,
			typename RIT, typename CIT, typename NIT
		>
		friend class Matrix;


	private:

		/** Our own type. */
		typedef Matrix<
			D, __REF_BACKEND_NAME__,
			RowIndexType, ColIndexType, NonzeroIndexType
		> self_type;

		/**
		 * \internal Returns the required global buffer size for a matrix of the
		 *           given dimensions.
		 */
		static size_t reqBufSize( const size_t m, const size_t n ) {
			// static checks
			constexpr size_t globalBufferUnitSize =
				sizeof(RowIndexType) +
				sizeof(ColIndexType) +
				grb::utils::SizeOf< D >::value;
			static_assert(
				globalBufferUnitSize >= sizeof(NonzeroIndexType),
				"We hit here a configuration border case which the implementation does not "
				"handle at present. Please submit a bug report."
			);
			// compute and return
			return std::max( (std::max( m, n ) + 1) * globalBufferUnitSize,
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
				config::OMP::threads() * config::CACHE_LINE_SIZE::value() *
					utils::SizeOf< D >::value
#else
				static_cast< size_t >( 0 )
#endif
		       );
		}

		/** The Row Compressed Storage */
		class internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType > CRS;

		/** The Column Compressed Storage */
		class internal::Compressed_Storage< D, ColIndexType, NonzeroIndexType > CCS;

		/** The determinstically-obtained ID of this container. */
		uintptr_t id;

		/** Whether to remove #id on destruction. */
		bool remove_id;

		/**
		 * The number of rows.
		 *
		 * \internal Not declared const to be able to implement move in an elegant
		 *           way.
		 */
		size_t m;

		/**
		 * The number of columns.
		 *
		 * \internal Not declared const to be able to implement move in an elegant
		 *           way.
		 */
		size_t n;

		/** The nonzero capacity (in elements). */
		size_t cap;

		/** The current number of nonzeroes. */
		size_t nz;

		/** Array buffer space required for SPA used in symbolic phases. */
		char * __restrict__ coorArr[ 2 ];

		/** Stack buffer space required for SPA used in symbolic phases. */
		char * __restrict__ coorBuf[ 2 ];

		/** Value buffer space required for symbolic phases. */
		D * __restrict__ valbuf[ 2 ];

		/**
		 * Six utils::AutoDeleter objects to free matrix resources automatically
		 * once these go out of scope. We interpret each resource as a block of
		 * bytes, hence we choose \a char as datatype here. The amount of bytes
		 * is controlled by the internal::Compressed_Storage class.
		 */
		utils::AutoDeleter< char > _deleter[ 6 ];

		/**
		 * #utils::AutoDeleter objects that, different from #_deleter, are not
		 * retained e.g. when pinning a matrix.
		 */
		utils::AutoDeleter< char > _local_deleter[ 6 ];

		/**
		 * Internal constructor for manual construction of matrices.
		 *
		 * Should be followed by a manual call to #initialize.
		 */
		Matrix() : id( std::numeric_limits< uintptr_t >::max() ),
			remove_id( false ), m( 0 ), n( 0 ), cap( 0 ), nz( 0 )
		{}

		/**
		 * Internal constructor that wraps around an existing external Compressed Row
		 * Storage (CRS).
		 *
		 * The internal column-major storage will \em not be initialised after a call
		 * to this constructor. Resulting instances must be used only in combination
		 * with #grb::descriptors::force_row_major. Container IDs will not be
		 * available for resulting instances.
		 *
		 * @param[in] _values         Array of nonzero values.
		 * @param[in] _column_indices Array of nonzero column indices.
		 * @param[in] _offset_array   CRS offset array of size \a _m + 1.
		 * @param[in] _m              The number of matrix rows.
		 * @param[in] _n              The number of matrix columns.
		 *
		 * The arrays \a _values and \a _column_indices must have size equal to
		 * <tt>_offset_array[ _m ];</tt>. The entries of \a _column_indices must
		 * all be smaller than \a _n. The entries of \a _offset_array must be
		 * monotonically increasing.
		 *
		 * If the wrapped matrix is to be used as an output for grb::mxm, then the
		 * following buffers must also be provided:
		 *
		 * @param[in] buf1 A buffer of Coordinates< T >::arraySize( \a n ) bytes.
		 * @param[in] buf2 A buffer of Coordinates< T >::bufferSize( \a n ) bytes.
		 * @param[in] buf3 A buffer of <tt>sizeof( D )</tt> times \a n bytes.
		 *
		 * Failure to provide such buffers for an output matrix will lead to undefined
		 * behaviour during a call to grb::mxm.
		 */
		Matrix(
			const D *__restrict__ const _values,
			const ColIndexType *__restrict__ const _column_indices,
			const NonzeroIndexType *__restrict__ const _offset_array,
			const size_t _m, const size_t _n,
			const size_t _cap,
			char *__restrict__ const buf1 = nullptr,
			char *__restrict__ const buf2 = nullptr,
			D *__restrict__ const buf3 = nullptr
		) :
			id( std::numeric_limits< uintptr_t >::max() ), remove_id( false ),
			m( _m ), n( _n ), cap( _cap ), nz( _offset_array[ _m ] ),
			coorArr{ nullptr, buf1 }, coorBuf{ nullptr, buf2 },
			valbuf{ nullptr, buf3 }
		{
			assert( (_m > 0 && _n > 0) || _column_indices[ 0 ] == 0 );
			CRS.replace( _values, _column_indices );
			CRS.replaceStart( _offset_array );
			// CCS is not initialised (and should not be used)
			if( !internal::template ensureReferenceBufsize< char >(
				reqBufSize( m, n ) )
			) {
				throw std::runtime_error( "Could not resize global buffer" );
			}
		}

		/**
		 * Takes care of the initialisation of a new matrix.
		 */
		void initialize(
			const uintptr_t * const id_in,
			const size_t rows, const size_t columns,
			const size_t cap_in
		) {
#ifdef _DEBUG
			std::cerr << "\t in Matrix< __REF_BACKEND_NAME__ >::initialize...\n"
				<< "\t\t matrix size " << rows << " by " << columns << "\n"
				<< "\t\t requested capacity " << cap_in << "\n";
#endif

			// dynamic checks
			assert( id == std::numeric_limits< uintptr_t >::max() );
			assert( !remove_id );
			if( rows >= static_cast< size_t >(
					std::numeric_limits< RowIndexType >::max()
				)
			) {
				throw std::overflow_error( "Number of rows larger than configured "
					"RowIndexType maximum!" );
			}
			if( columns >= static_cast< size_t >(
					std::numeric_limits< ColIndexType >::max()
				)
			) {
				throw std::overflow_error( "Number of columns larger than configured "
					"ColIndexType maximum!" );
			}

			// initial setters
			if( id_in != nullptr ) {
				id = *id_in;
#ifdef _DEBUG
				std::cerr << "\t\t inherited ID " << id << "\n";
#endif
			}
			m = rows;
			n = columns;
			cap = nz = 0;

			// memory allocations
			RC alloc_ok = SUCCESS;
			char * alloc[ 8 ] = {
				nullptr, nullptr, nullptr, nullptr,
				nullptr, nullptr, nullptr, nullptr
			};
			if( !internal::template ensureReferenceBufsize< char >(
				reqBufSize( m, n ) )
			) {
				throw std::runtime_error( "Could not resize global buffer" );
			}
			if( m > 0 && n > 0 ) {
				// check whether requested capacity is sensible
				if( cap_in / m > n ||
					cap_in / n > m ||
					(cap_in / m == n && (cap_in % m > 0)) ||
					(cap_in / n == m && (cap_in % n > 0))
				) {
#ifdef _DEBUG
					std::cerr << "\t\t Illegal capacity requested\n";
#endif
					throw std::runtime_error( toString( ILLEGAL ) );
				}
				// get sizes of arrays that we need to allocate
				size_t sizes[ 12 ];
				sizes[ 0 ] = internal::Coordinates< __REF_BACKEND_NAME__ >::arraySize( m );
				sizes[ 1 ] = internal::Coordinates< __REF_BACKEND_NAME__ >::arraySize( n );
				sizes[ 2 ] = internal::Coordinates< __REF_BACKEND_NAME__ >::bufferSize( m );
				sizes[ 3 ] = internal::Coordinates< __REF_BACKEND_NAME__ >::bufferSize( n );
				sizes[ 4 ] = m * internal::SizeOf< D >::value;
				sizes[ 5 ] = n * internal::SizeOf< D >::value;
				if( cap_in > 0 ) {
					CRS.getStartAllocSize( &( sizes[ 6 ] ), m );
					CCS.getStartAllocSize( &( sizes[ 7 ] ), n );
					CRS.getAllocSize( &(sizes[ 8 ]), cap_in );
					CCS.getAllocSize( &(sizes[ 10 ]), cap_in );
				} else {
					sizes[ 8 ] = sizes[ 9 ] = sizes[ 10 ] = sizes[ 11 ] = 0;
				}
				// allocate required arrays
				alloc_ok = utils::alloc(
					"grb::Matrix< T, __REF_BACKEND_NAME__ >::Matrix()",
					"initial capacity allocation",
					coorArr[ 0 ], sizes[ 0 ], false, _local_deleter[ 0 ],
					coorArr[ 1 ], sizes[ 1 ], false, _local_deleter[ 1 ],
					coorBuf[ 0 ], sizes[ 2 ], false, _local_deleter[ 2 ],
					coorBuf[ 1 ], sizes[ 3 ], false, _local_deleter[ 3 ],
					alloc[ 6 ], sizes[ 4 ], false, _local_deleter[ 4 ],
					alloc[ 7 ], sizes[ 5 ], false, _local_deleter[ 5 ],
					alloc[ 0 ], sizes[ 6 ], true, _deleter[ 0 ],
					alloc[ 1 ], sizes[ 7 ], true, _deleter[ 1 ],
					alloc[ 2 ], sizes[ 8 ], true, _deleter[ 2 ],
					alloc[ 3 ], sizes[ 9 ], true, _deleter[ 3 ],
					alloc[ 4 ], sizes[ 10 ], true, _deleter[ 4 ],
					alloc[ 5 ], sizes[ 11 ], true, _deleter[ 5 ]
				);
			} else {
				const size_t sizes[ 2 ] = {
					m * internal::SizeOf< D >::value,
					n * internal::SizeOf< D >::value
				};
				coorArr[ 0 ] = coorArr[ 1 ] = nullptr;
				coorBuf[ 0 ] = coorBuf[ 1 ] = nullptr;
				alloc_ok = utils::alloc(
					"grb::Matrix< T, __REF_BACKEND_NAME__ >::Matrix()",
					"empty allocation",
					alloc[ 6 ], sizes[ 0 ], false, _local_deleter[ 4 ],
					alloc[ 7 ], sizes[ 1 ], false, _local_deleter[ 5 ]
				);
			}

			if( alloc_ok == OUTOFMEM ) {
				throw std::runtime_error( "Could not allocate memory during grb::Matrix construction" );
			} else if( alloc_ok != SUCCESS ) {
				throw std::runtime_error( toString( alloc_ok ) );
			}

			if( m > 0 && n > 0 ) {
				cap = cap_in;
			}
			valbuf[ 0 ] = reinterpret_cast< D * >( alloc[ 6 ] );
			valbuf[ 1 ] = reinterpret_cast< D * >( alloc[ 7 ] );

			if( m > 0 && n > 0 ) {
#ifdef _DEBUG
				std::cerr << "\t\t allocations for an " << m << " by " << n << " matrix "
					<< "have successfully completed\n";
#endif
				CRS.replaceStart( alloc[ 0 ] );
				CCS.replaceStart( alloc[ 1 ] );
				CRS.replace( alloc[ 2 ], alloc[ 3 ] );
				CCS.replace( alloc[ 4 ], alloc[ 5 ] );
				if( id_in == nullptr ) {
					id = internal::reference_mapper.insert(
						reinterpret_cast< uintptr_t >(CRS.getOffsets())
					);
					remove_id = true;
#ifdef _DEBUG
					std::cerr << "\t\t assigned new ID " << id << "\n";
#endif
				} else {
					assert( !remove_id );
				}
			}
		}

		/** Implements a move. */
		void moveFromOther( self_type &&other ) {
			// move from other
			CRS = std::move( other.CRS );
			CCS = std::move( other.CCS );
			id = other.id;
			remove_id = other.remove_id;
			m = other.m;
			n = other.n;
			cap = other.cap;
			nz = other.nz;
			for( unsigned int i = 0; i < 2; ++i ) {
				coorArr[ i ] = other.coorArr[ i ];
				coorBuf[ i ] = other.coorBuf[ i ];
				valbuf[ i ] = other.valbuf[ i ];
			}
			for( unsigned int i = 0; i < 6; ++i ) {
				_deleter[ i ] = std::move( other._deleter[ i ] );
				_local_deleter[ i ] = std::move( other._local_deleter[ i ] );
			}

			// invalidate other fields
			for( unsigned int i = 0; i < 2; ++i ) {
				other.coorArr[ i ] = other.coorBuf[ i ] = nullptr;
				other.valbuf[ i ] = nullptr;
			}
			other.id = std::numeric_limits< uintptr_t >::max();
			other.remove_id = false;
			other.m = 0;
			other.n = 0;
			other.cap = 0;
			other.nz = 0;
		}

		/** @see Matrix::clear */
		RC clear() {
			// update nonzero count
			nz = 0;

			// catch trivial case
			if( m == 0 || n == 0 ) {
				return SUCCESS;
			}

			// catch uninitialised case
			if( CRS.col_start == NULL || CCS.col_start == NULL ) {
				// sanity check
				assert( CRS.col_start == NULL && CCS.col_start == NULL );
				// nothing to do
				return SUCCESS;
			}

			// do both CRS and CCS simultaneously as far as possible
			const size_t min_dim = ( m < n ) ? m : n;
			for( size_t i = 0; i < min_dim; ++i ) {
				CRS.col_start[ i ] = CCS.col_start[ i ] = 0;
			}

			// do last remainder of either CRS or CCS
			if( min_dim == m ) {
				for( size_t i = min_dim; i < n; ++i ) {
					CCS.col_start[ i ] = 0;
				}
			} else {
				for( size_t i = min_dim; i < m; ++i ) {
					CRS.col_start[ i ] = 0;
				}
			}

			// done
			return SUCCESS;
		}

		/** @see grb::resize() */
		RC resize( const size_t nonzeroes ) {
			// check for trivial case
			if( m == 0 || n == 0 || nonzeroes == 0 ) {
				// simply do not do anything and return
				return SUCCESS;
			}

			// do not do anything if current capacity is sufficient
			if( nonzeroes <= cap ) {
				return SUCCESS;
			}

			if( nonzeroes >= static_cast< size_t >(
					std::numeric_limits< NonzeroIndexType >::max()
				)
			) {
				return OVERFLW;
			}

			// allocate and catch errors
			char * alloc[ 4 ] = { nullptr, nullptr, nullptr, nullptr };
			size_t sizes[ 4 ];
			// cache old allocation data
			size_t old_sizes[ 4 ];
			size_t freed = 0;
			if( cap > 0 ) {
				CRS.getAllocSize( &( old_sizes[ 0 ] ), cap );
				CCS.getAllocSize( &( old_sizes[ 2 ] ), cap );
			}

			// compute new required sizes
			CRS.getAllocSize( &( sizes[ 0 ] ), nonzeroes );
			CCS.getAllocSize( &( sizes[ 2 ] ), nonzeroes );

			// construct a description of the matrix we are allocating for
			std::stringstream description;
			description << ", for " << nonzeroes << " nonzeroes in an " << m << " "
				<< "times " << n << " matrix.\n";

			// do allocation
			RC ret = utils::alloc(
				"grb::Matrix< T, __REF_BACKEND_NAME__ >::resize", description.str(),
				alloc[ 0 ], sizes[ 0 ], true, _deleter[ 2 ],
				alloc[ 1 ], sizes[ 1 ], true, _deleter[ 3 ],
				alloc[ 2 ], sizes[ 2 ], true, _deleter[ 4 ],
				alloc[ 3 ], sizes[ 3 ], true, _deleter[ 5 ]
			);

			if( ret != SUCCESS ) {
				// exit function without side-effects
				return ret;
			}

			// put allocated arrays in their intended places
			CRS.replace( alloc[ 0 ], alloc[ 1 ] );
			CCS.replace( alloc[ 2 ], alloc[ 3 ] );

			// if we had old data emplaced
			if( cap > 0 ) {
				for( unsigned int i = 0; i < 4; ++i ) {
					if( old_sizes[ i ] > 0 ) {
						freed += sizes[ i ];
					}
				}
				if( config::MEMORY::report( "grb::Matrix< T, __REF_BACKEND_NAME__ >::resize",
					"freed (or will eventually free)", freed, false )
				) {
					std::cout << ", for " << cap << " nonzeroes "
						<< "that this matrix previously contained.\n";
				}
			}

			// set new capacity
			cap = nonzeroes;

			// done, return error code
			return SUCCESS;
		}


   		/** @see Matrix::buildMatrixUnique */
		template< Descriptor descr = descriptors::no_operation, typename fwd_iterator>
		RC buildMatrixUnique( const fwd_iterator & _start, const fwd_iterator & _end, const IOMode mode ) {
			// here we can safely ignore the mode and dispatch based only on the iterator type
			// since in shared memory the input data reside by definition all on the same machine
			(void)mode;
			typename iterator_tag_selector< fwd_iterator >::iterator_category category;
			return buildMatrixUniqueImpl( _start, _end, category );
		}


		//forward iterator version
		template <typename fwd_iterator>
		RC buildMatrixUniqueImpl(fwd_iterator _start, fwd_iterator _end, std::forward_iterator_tag) {
#ifdef _DEBUG
		        std::cout << " fwrd acces iterator " << '\n';
		        //compilation of the next lines should fail
		        //#pragma omp parallel for
  		        //for( fwd_iterator it = _start; it != _end; ++it ){}
			std::cout << "buildMatrixUnique called with " << cap << " nonzeroes.\n";
			std::cout << "buildMatrixUnique: input is\n";
			for( fwd_iterator it = _start; it != _end; ++it ) {
				std::cout << "\t" << it.i() << ", " << it.j() << "\n";
			}
			std::cout << "buildMatrixUnique: end input.\n";
#endif

#ifdef _GRB_BUILD_MATRIX_UNIQUE_TRACE
			::__trace_build_matrix_iomode( __REF_BACKEND_NAME__, false );
#endif
			// detect trivial case
			if( _start == _end || m == 0 || n == 0 ) {
				return SUCCESS;
			}

			// keep count of nonzeroes
			nz = 0;


			// reset col_start array to zero, fused loop
			{
				// get minimum dimension
				size_t min_dim = static_cast< size_t >( m );
				size_t max_dim = static_cast< size_t >( n );
				if( min_dim > max_dim ) {
					std::swap( min_dim, max_dim );
				}
				// fill until minimum
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
				#pragma omp parallel for schedule( \
 		dynamic, config::CACHE_LINE_SIZE::value() )
#endif
				for( size_t i = 0; i < min_dim; ++i ) {
					CRS.col_start[ i ] = 0;
					CCS.col_start[ i ] = 0;
				}
				// if the minimum dimension is the row dimension
				if( min_dim == static_cast< size_t >( m ) ) {
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
					#pragma omp parallel for schedule( \
 		dynamic, config::CACHE_LINE_SIZE::value() )
#endif
					// then continue to fill column dimension
					for( size_t i = min_dim; i < max_dim; ++i ) {
						CCS.col_start[ i ] = 0;
					}
				} else {
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
					#pragma omp parallel for schedule( \
 		dynamic, config::CACHE_LINE_SIZE::value() )
#endif
					// otherwise, continue to fill row dimension
					for( size_t i = min_dim; i < max_dim; ++i ) {
						CRS.col_start[ i ] = 0;
					}
				}
			}



			for( fwd_iterator it = _start; it != _end; ++it )
			  {
			    if( it.i() >= m ) {
			      std::cerr << "Error: " << m << " times " << n
					<< " matrix nonzero ingestion encounters row "
					<< "index at " << it.i() << "\n";
			      return MISMATCH;
			    }
			    if( it.j() >= n ) {
			      std::cerr << "Error: " << m << " times " << n
					<< " matrix nonzero ingestion encounters column "
					<< "input at " << it.j() << "\n";
			      return MISMATCH;
			    }
			    (void)++( CRS.col_start[ it.i() ] );
			    (void)++( CCS.col_start[ it.j() ] );
			    (void)++nz;
			  }


			// check if we can indeed store nz values
			if( nz >= static_cast< size_t >(
					std::numeric_limits< grb::config::NonzeroIndexType >::max()
				)
			) {
				return OVERFLW;
			}

			// put final entries
			CRS.col_start[ m ] = nz;
			CCS.col_start[ n ] = nz;

			// allocate enough space
			resize( nz );

			// make counting sort array cumulative
			for( size_t i = 1; i < m; ++i ) {
#ifdef _DEBUG
				std::cout << "There are " << CRS.col_start[ i ] << " "
					<< "nonzeroes at row " << i << "\n";
#endif
				CRS.col_start[ i ] += CRS.col_start[ i - 1 ];
			}

			// make counting sort array cumulative
			for( size_t i = 1; i < n; ++i ) {
#ifdef _DEBUG
				std::cout << "There are " << CCS.col_start[ i ] << " "
					<< "nonzeroes at column " << i << "\n";
#endif
				CCS.col_start[ i ] += CCS.col_start[ i - 1 ];
			}




			// perform counting sort
			fwd_iterator it = _start;

			for( size_t k = 0; it != _end; ++k, ++it ) {
				const size_t crs_pos = --( CRS.col_start[ it.i() ] );
				CRS.recordValue( crs_pos, false, it );
#ifdef _DEBUG
				std::cout << "Nonzero " << k << ", ( " << it.i() << ", " << it.j() << " ) "
					<< "is stored at CRS position "
					<< static_cast< size_t >( crs_pos ) << ".\n";
				// Disabled the following to support pattern matrices:
				// "Its stored value is " << CRS.values[crs_pos] << ", "
				// "while the original input was " << it.v() << ".\n";
#endif
				const size_t ccs_pos = --( CCS.col_start[ it.j() ] );
				CCS.recordValue( ccs_pos, true, it );
#ifdef _DEBUG
				std::cout << "Nonzero " << k << ", ( " << it.i() << ", " << it.j() << " ) "
					<< "is stored at CCS position "
					<< static_cast< size_t >( ccs_pos ) << ".\n";
				// Disabled the following to support pattern matrices:
				// ". Its stored value is " << CCS.values[ccs_pos] << ", while the "
				// "original input was " << it.v() << ".\n";
#endif
			}

#ifdef _DEBUG
			for( size_t i = 0; i <= m; ++i ) {
				(void)printf( "row_start[ %ld ] = %ld.\n", i, CRS.col_start[ i ] );
			}
			for( size_t i = 0; i <= n; ++i ) {
				(void)printf( "col_start[ %ld ] = %ld.\n", i, CCS.col_start[ i ] );
			}
#endif
			// done


			return SUCCESS;
		}


#ifdef _H_GRB_REFERENCE_OMP_MATRIX
		//random access version
		template <typename rndacc_iterator>
		RC buildMatrixUniqueImpl(rndacc_iterator _start, rndacc_iterator _end, std::random_access_iterator_tag) {
#ifdef _DEBUG
		        std::cout << " rnd access iterator " << '\n';
				std::cout << "buildMatrixUnique called with " << cap << " nonzeroes.\n";
				std::cout << "buildMatrixUnique: input is\n";
				for( rndacc_iterator it = _start; it != _end; ++it ) {
					std::cout << "\t" << it.i() << ", " << it.j() << "\n";
				}
				std::cout << "buildMatrixUnique: end input.\n";
#endif

#ifdef _GRB_BUILD_MATRIX_UNIQUE_TRACE
			::__trace_build_matrix_iomode( __REF_BACKEND_NAME__, true );
#endif

				// detect trivial case
				if( _start == _end || m == 0 || n == 0 ) {
					return SUCCESS;
				}

				// count of nonzeroes
				nz = _end-_start;

			// reset col_start array to zero, fused loop
			{
				// get minimum dimension
				size_t min_dim = static_cast< size_t >( m );
				size_t max_dim = static_cast< size_t >( n );
				if( min_dim > max_dim ) {
					std::swap( min_dim, max_dim );
				}
				// fill until minimum
				#pragma omp parallel for schedule( \
 		dynamic, config::CACHE_LINE_SIZE::value() )
				for( size_t i = 0; i < min_dim; ++i ) {
					CRS.col_start[ i ] = 0;
					CCS.col_start[ i ] = 0;
				}
				// if the minimum dimension is the row dimension
				if( min_dim == static_cast< size_t >( m ) ) {
					#pragma omp parallel for schedule( \
 		dynamic, config::CACHE_LINE_SIZE::value() )
					// then continue to fill column dimension
					for( size_t i = min_dim; i < max_dim; ++i ) {
						CCS.col_start[ i ] = 0;
					}
				} else {
					#pragma omp parallel for schedule( \
 		dynamic, config::CACHE_LINE_SIZE::value() )
					// otherwise, continue to fill row dimension
					for( size_t i = min_dim; i < max_dim; ++i ) {
						CRS.col_start[ i ] = 0;
					}
				}
			}

			// check if we can indeed store nz values
			if( nz >= static_cast< size_t >(
					std::numeric_limits< NonzeroIndexType >::max()
				)
			) {
				return RC::OVERFLW;
			}

			// put final entries
			CRS.col_start[ m ] = nz;
			CCS.col_start[ n ] = nz;

			// allocate enough space
			resize( nz );

			struct {
				config::RowIndexType operator()( const rndacc_iterator& itr ) {
					return itr.i();
				}
			} row_getter;


			struct {
				config::ColIndexType operator()( const rndacc_iterator& itr ) {
					return itr.j();
				}
			} col_getter;

			RC ret = internal::__invoke_( 0, m, nz, _start, row_getter, col_getter, CRS, false );

			/*
			RC ret = count_sort_omp( it, nz, amin, amax, nsize, buffer, bufferlen_tot,
				CRS, row_getter, false );
			*/

			if ( ret != SUCCESS ) {
				return ret;
			}


			/*
			return count_sort_omp( it, nz, amin, amax, nsize, buffer, bufferlen_tot,
				CCS, col_getter, true ) ;
			*/

			return internal::__invoke_( 0, n, nz, _start, col_getter, row_getter, CCS, true );

		}
#endif


	public:

		/** @see Matrix::value_type */
		typedef D value_type;

		/** The iterator type over matrices of this type. */
		typedef typename internal::Compressed_Storage<
			D, RowIndexType, NonzeroIndexType
		>::template ConstIterator<
			internal::Distribution< __REF_BACKEND_NAME__ >
		> const_iterator;

		/**
		 * \parblock
		 * \par Performance semantics
		 *
		 * This backend specifies the following performance semantics for this
		 * constructor:
		 *   -# \f$ \Theta( n ) \f$ work
		 *   -# \f$ \Theta( n ) \f$ intra-process data movement
		 *   -# \f$ \Theta( (rows + cols + 2)x + nz(y+z) ) \f$ storage requirement
		 *   -# system calls, in particular memory allocations and re-allocations up
		 *      to \f$ \Theta( n ) \f$ memory, will occur.
		 * Here,
		 *   -# n is the maximum of \a rows, \a columns, \em and \a nz;
		 *   -# x is the size of integer used to refer to nonzero indices;
		 *   -# y is the size of integer used to refer to row or column indices; and
		 *   -# z is the size of the nonzero value type.
		 *
		 * Note that this backend does not support multiple user processes, so inter-
		 * process costings are omitted.
		 *
		 * In the case of the reference_omp backend, the critical path length for
		 * work is \f$ \Theta( n / T + T ) \f$. This assumes that memory allocation is
		 * a scalable operation (while in reality the complexity of allocation is, of
		 * course, undefined).
		 * \endparblock
		 */
		Matrix( const size_t rows, const size_t columns, const size_t nz ) : Matrix()
		{
#ifdef _DEBUG
			std::cout << "In grb::Matrix constructor (reference, with requested capacity)\n";
#endif
			initialize( nullptr, rows, columns, nz );
		}

		/**
		 * \parblock
		 * \par Performance semantics
		 * This backend specifies the following performance semantics for this
		 * constructor:
		 *   -# \f$ \Theta( n ) \f$ work
		 *   -# \f$ \Theta( n ) \f$ intra-process data movement
		 *   -# \f$ \Theta( (rows + cols + 2)x + n(y+z) ) \f$ storage requirement
		 *   -# system calls, in particular memory allocations and re-allocations
		 *      are allowed.
		 * Here,
		 *   -# n is the maximum of \a rows and \a columns;
		 *   -# x is the size of integer used to refer to nonzero indices;
		 *   -# y is the size of integer used to refer to row or column indices; and
		 *   -# z is the size of the nonzero value type.
		 * Note that this backend does not support multiple user processes, so inter-
		 * process costings are omitted.
		 * \endparblock
		 */
		Matrix( const size_t rows, const size_t columns ) :
			Matrix( rows, columns, std::max( rows, columns ) )
		{
#ifdef _DEBUG
			std::cerr << "In grb::Matrix constructor (reference, default capacity)\n";
#endif
		}

		/**
		 * \parblock
		 * \par Performance semantics
		 * This backend specifies the following performance semantics for this
		 * constructor:
		 *   -# first, the performance semantics of a constructor call with arguments
		 *          nrows( other ), ncols( other ), capacity( other )
		 *      applies.
		 *   -# then, the performance semantics of a call to grb::set apply.
		 * \endparblock
		 */
		Matrix(
			const Matrix<
				D, __REF_BACKEND_NAME__,
				RowIndexType, ColIndexType, NonzeroIndexType
			> &other
		) :
			Matrix( other.m, other.n, other.cap )
		{
#ifdef _DEBUG
			std::cerr << "In grb::Matrix (reference) copy-constructor\n";
#endif
			nz = other.nz;

			// if empty, return; otherwise copy
			if( nz == 0 ) { return; }

#ifdef _H_GRB_REFERENCE_OMP_MATRIX
			#pragma omp parallel
#endif
			{
				size_t range = CRS.copyFromRange( nz, n );
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
				size_t start, end;
				config::OMP::localRange( start, end, 0, range );
#else
				const size_t start = 0;
				size_t end = range;
#endif
				CRS.copyFrom( other.CRS, nz, n, start, end );
				range = CCS.copyFromRange( nz, m );
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
				config::OMP::localRange( start, end, 0, range );
#else
				end = range;
#endif
				CCS.copyFrom( other.CCS, nz, m, start, end );
			}
		}

		/** \internal No implementation notes. */
		Matrix( self_type &&other ) noexcept {
			moveFromOther( std::forward< self_type >(other) );
		}

		/** \internal No implementation notes. */
		self_type& operator=( self_type &&other ) noexcept {
			moveFromOther( std::forward< self_type >(other) );
			return *this;
		}

		/**
		 * \parblock
		 * \par Performance semantics
		 *
		 * This backend specifies the following performance semantics for this
		 * destructor:
		 *   -# \f$ \mathcal{O}( n ) \f$ work
		 *   -# \f$ \mathcal{O}( n ) \f$ intra-process data movement
		 *   -# storage requirement is reduced to zero
		 *   -# system calls, in particular memory de-allocations, are allowed.
		 *
		 * Here,
		 *   -# n is the maximum of \a rows, \a columns, and current capacity.
		 *
		 * Note that this backend does not support multiple user processes, so inter-
		 * process costings are omitted.
		 *
		 * Note that the big-Oh bound is only achieved if the underlying system
		 * requires zeroing out memory after de-allocations, as may be required, for
		 * example, as an information security mechanism.
		 * \endparblock
		 */
		~Matrix() {
#ifdef _DEBUG
			std::cerr << "In ~Matrix (reference)\n"
				<< "\t matrix is " << m << " by " << n << "\n"
				<< "\t capacity is " << cap << "\n"
				<< "\t ID is " << id << "\n";
#endif
#ifndef NDEBUG
			if( CRS.row_index == nullptr ) {
				assert( CCS.row_index == nullptr );
				assert( m == 0 || n == 0 || nz == 0 );
				assert( cap == 0 );
			}
#endif
			if( m > 0 && n > 0 && remove_id ) {
				internal::reference_mapper.remove( id );
			}
		}

		/**
		 * \internal No implementation notes.
		 *
		 * \todo should we specify performance semantics for retrieving iterators?
		 */
		template< class ActiveDistribution = internal::Distribution< __REF_BACKEND_NAME__ > >
		typename internal::Compressed_Storage<
			D,
			RowIndexType, NonzeroIndexType
		>::template ConstIterator< ActiveDistribution > begin(
			const IOMode mode = PARALLEL,
			const size_t s = 0, const size_t P = 1
		) const {
			assert( mode == PARALLEL );
			(void)mode;
			typedef typename internal::Compressed_Storage<
				D,
				RowIndexType,
				NonzeroIndexType
			>::template ConstIterator< ActiveDistribution > IteratorType;
#ifdef _DEBUG
			std::cout << "In grb::Matrix<T,reference>::cbegin\n";
#endif
			return IteratorType( CRS, m, n, nz, false, s, P );
		}

		/**
		 * \internal No implementation notes.
		 *
		 * \todo should we specify performance semantics for retrieving iterators?
		 */
		template< class ActiveDistribution = internal::Distribution< __REF_BACKEND_NAME__ > >
		typename internal::Compressed_Storage<
			D,
			RowIndexType,
			NonzeroIndexType
		>::template ConstIterator< ActiveDistribution > end(
			const IOMode mode = PARALLEL,
			const size_t s = 0, const size_t P = 1
		) const {
			assert( mode == PARALLEL );
			(void)mode;
			typedef typename internal::Compressed_Storage<
				D,
				RowIndexType,
				NonzeroIndexType
			>::template ConstIterator< ActiveDistribution > IteratorType;
			return IteratorType( CRS, m, n, nz, true, s, P );
		}

		/**
		 * \internal No implementation notes.
		 *
		 * \todo should we specify performance semantics for retrieving iterators?
		 */
		template< class ActiveDistribution = internal::Distribution< __REF_BACKEND_NAME__ > >
		typename internal::Compressed_Storage<
			D,
			RowIndexType,
			NonzeroIndexType
		>::template ConstIterator< ActiveDistribution > cbegin(
			const IOMode mode = PARALLEL
		) const {
			return begin< ActiveDistribution >( mode );
		}

		/**
		 * \internal No implementation notes.
		 *
		 * \todo should we specify performance semantics for retrieving iterators?
		 */
		template< class ActiveDistribution = internal::Distribution< __REF_BACKEND_NAME__ > >
		typename internal::Compressed_Storage<
			D,
			RowIndexType,
			NonzeroIndexType
		>::template ConstIterator< ActiveDistribution > cend(
			const IOMode mode = PARALLEL
		) const {
			return end< ActiveDistribution >( mode );
		}

	};

	// template specialisation for GraphBLAS type traits
	template< typename D, typename RIT, typename CIT, typename NIT >
	struct is_container< Matrix< D, __REF_BACKEND_NAME__, RIT, CIT, NIT > > {
		/** A reference Matrix is a GraphBLAS object. */
		static const constexpr bool value = true;
	};

	namespace internal {

#ifndef _H_GRB_REFERENCE_OMP_MATRIX
		template<
			typename ValType, typename ColType, typename IndType,
			Backend backend
		>
		const grb::Matrix< ValType, backend, ColType, ColType, IndType >
		wrapCRSMatrix(
			const ValType *__restrict__ const value_array,
			const ColType *__restrict__ const index_array,
			const IndType *__restrict__ const offst_array,
			const size_t m, const size_t n
		) {
			grb::Matrix< ValType, backend, ColType, ColType, IndType > ret(
				value_array, index_array, offst_array, m, n, offst_array[ m ]
			);
			return ret;
		}

		template<
			typename ValType, typename ColType, typename IndType,
			Backend backend
		>
		grb::Matrix< ValType, backend, ColType, ColType, IndType >
		wrapCRSMatrix(
			ValType *__restrict__ const value_array,
			ColType *__restrict__ const index_array,
			IndType *__restrict__ const offst_array,
			const size_t m, const size_t n, const size_t cap,
			char * const buf1, char * const buf2,
			ValType *__restrict__ const buf3
		) {
			grb::Matrix< ValType, backend, ColType, ColType, IndType > ret(
				value_array, index_array, offst_array, m, n, cap,
				buf1, buf2, buf3
			);
			return ret;
		}
#endif

	} // end namespace grb::internal

} // end namespace grb

// parse again for reference_omp backend
#ifdef _GRB_WITH_OMP
 #ifndef _H_GRB_REFERENCE_OMP_MATRIX
  #define _H_GRB_REFERENCE_OMP_MATRIX
  #undef __REF_BACKEND_NAME__
  #define __REF_BACKEND_NAME__ reference_omp
  #include "matrix.hpp"
  #undef __REF_BACKEND_NAME__
  #undef _H_GRB_REFERENCE_OMP_MATRIX
 #endif
#endif

#endif // end ``_H_GRB_REFERENCE_MATRIX''

