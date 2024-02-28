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

#if !defined _H_GRB_REFERENCE_MATRIX || defined _H_GRB_REFERENCE_OMP_MATRIX
#define _H_GRB_REFERENCE_MATRIX

#include <numeric> //std::accumulate
#include <sstream> //std::stringstream
#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <iterator>
#include <cmath>

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
#include <graphblas/type_traits.hpp>
#include <graphblas/utils/iterators/utils.hpp>

#include "NonzeroWrapper.hpp"
#include "forward.hpp"


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

	} // end namespace internal
#endif

	namespace internal {

		template< typename D, typename RIT, typename CIT, typename NIT >
		const size_t & getNonzeroCapacity(
			const grb::Matrix< D, reference, RIT, CIT, NIT > &A
		) noexcept {
			return A.cap;
		}
		template< typename D, typename RIT, typename CIT, typename NIT >
		const size_t & getCurrentNonzeroes(
			const grb::Matrix< D, reference, RIT, CIT, NIT > &A
		) noexcept {
			return A.nz;
		}
		template< typename D, typename RIT, typename CIT, typename NIT >
		void setCurrentNonzeroes(
			grb::Matrix< D, reference, RIT, CIT, NIT > &A,
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
			const grb::Matrix< InputType, reference, RIT, CIT, NIT > &A
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
			Vector< IOType, reference, Coords > &destination_vector,
			IOType * __restrict__ const &destination,
			const size_t &destination_range,
			const Vector< InputType1, reference, Coords > &source_vector,
			const InputType1 * __restrict__ const &source,
			const size_t &source_index,
			const internal::Compressed_Storage< InputType2, RowColType, NonzeroType > &matrix,
			const Vector< InputType3, reference, Coords > &mask_vector,
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
			Vector< IOType, reference, Coords > &u,
			const Vector< InputType3, reference, Coords > &mask,
			const Vector< InputType1, reference, Coords > &v,
			const Vector< InputType4, reference, Coords > &v_mask,
			const Matrix< InputType2, reference, RIT, CIT, NIT > &A,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const std::function< size_t( size_t ) > &row_l2g,
			const std::function< size_t( size_t ) > &row_g2l,
			const std::function< size_t( size_t ) > &col_l2g,
			const std::function< size_t( size_t ) > &col_g2l
		);

#ifdef _H_GRB_REFERENCE_OMP_MATRIX

		/**
		 * \internal
		 * Utility to get the major coordinate from an input iterator.
		 *
		 * @tparam populate_crs Whether to assume CRS or CSC.
		 *
		 * This is the CSR implementation.
		 * \endinternal
		 */
		template<
			typename RowIndexType,
			typename IterT,
			bool populate_crs
		>
		struct ColGetter {
			RowIndexType operator()( const IterT &itr ) {
				return itr.i();
			}
		};

		/**
		 * \internal
		 * Utility to get the major coordinate from an input iterator.
		 *
		 * This is the CSC implementation.
		 * \endinternal
		 */
		template<
			typename ColIndexType,
			typename IterT
		>
		struct ColGetter< ColIndexType, IterT, true > {
			ColIndexType operator()( const IterT &itr ) {
				return itr.j();
			}
		};

		/**
		 * \internal
		 * Populates a matrix storage (CRS or CCS) in parallel, using multiple
		 * threads. It requires the input iterator \a _it to be a random access
		 * iterator.
		 *
		 * @tparam populate_ccs <tt>true</tt> if ingesting into a CCS, <tt>false</tt>
		 *                      if ingesting into a CRS. Without loss of generality,
		 *                      the below assumes CCS.
		 *
		 * @param[in] _it The input random access iterator.
		 * @param[in] nz  The number of nonzeroes \a _it iterates over.
		 * @param[in] num_cols The number of columns in the matrix.
		 * @param[in] num_rows The number of rows in the matrix.
		 *
		 * \warning When \a populate_ccs is <tt>false</tt>, \a num_cols and
		 *          \a num_rows correspond to the number of rows and columns,
		 *          respectively.
		 *
		 * @param[in] num_threads The number of threads used during ingestion.
		 * @param[in] prefix_sum_buffer Workspace for computing index prefix sums.
		 * @param[in] prefix_sum_buffer_size The size (in elements) of
		 *                                   \a prefix_sum_buffer.
		 * @param[in] col_values_buffer Must have size \a nz or be <tt>nullptr</tt>
		 * @param[out] storage Where to store the matrix.
		 *
		 * The number of threads may be restricted due to a limited work space.
		 * \endinternal
		 */
		template<
			bool populate_ccs,
			typename rndacc_iterator,
			typename ColIndexType,
			typename RowIndexType,
			typename ValType,
			typename NonzeroIndexType
		>
		RC populate_storage_parallel(
			const rndacc_iterator &_it,
			const size_t nz,
			const size_t num_cols,
			const size_t num_rows,
			const size_t num_threads,
			size_t * const prefix_sum_buffer,
			const size_t prefix_sum_buffer_size,
			ColIndexType * const col_values_buffer,
			Compressed_Storage< ValType, RowIndexType, NonzeroIndexType > &storage
		) {
			// if we are populating a CCS, we compute the bucket from the column indices;
			// if CRS, buckets are computed from row indices. From the below code's POV,
			// and without loss of generality, we assume CCS.
			ColGetter< ColIndexType, rndacc_iterator, populate_ccs > col_getter;

			if( nz < 1 ) {
#ifdef _DEBUG
				std::cerr << "Attempting to ingest an iterator in end position"
					<< std::endl;
#endif
				return RC::ILLEGAL;
			}
			if( num_cols == 0 || num_rows == 0 ) {
#ifdef _DEBUG
				std::cerr << "Attempting to ingest into an empty matrix" << std::endl;
#endif
				return RC::ILLEGAL;
			}
			if( prefix_sum_buffer_size < num_threads ) {
				std::cerr << "Error: buffersize (" << prefix_sum_buffer_size << ") "
					<< "which is smaller than num_threads (" << num_threads << "). "
					<< "This indicates an internal error as the minimum required buffer size "
					<< "should have been guaranteed available." << std::endl;
#ifndef NDEBUG
				const bool minimum_required_buffer_size_was_not_guaranteed = false;
				assert( minimum_required_buffer_size_was_not_guaranteed );
#endif
				return RC::PANIC;
			}

			// the actual matrix sizes depend on populate_ccs: flip them if it is false
			const size_t matrix_rows = populate_ccs ? num_rows : num_cols;
			const size_t matrix_cols = populate_ccs ? num_cols : num_rows;

			// compute thread-local buffer size
			const size_t ccs_col_buffer_size = num_cols + 1;
			const size_t per_thread_buffer_size = prefix_sum_buffer_size / num_threads;
			assert( per_thread_buffer_size > 0 );
			const size_t bucketlen = ( ccs_col_buffer_size == per_thread_buffer_size
					? 0
					: ccs_col_buffer_size / per_thread_buffer_size
				) + 1;

#ifdef _DEBUG
			std::cout << "In populate_storage_parallel with\n"
				<< "\tnz =" << nz << ", "
				<< "bufferlen = " << per_thread_buffer_size << ", "
				<< "bucketlen = " << bucketlen << "\n"
				<< "\tnum_threads = " << num_threads << "\n"
				<< "\tccs_col_buffer_size =  " << ccs_col_buffer_size << "\n"
				<< "\tper_thread_buffer_size = " << per_thread_buffer_size << "\n"
				<< "\tprefix_sum_buffer_size = " << prefix_sum_buffer_size << "\n"
				<< "\tminimum buffer size = " << nz + per_thread_buffer_size << std::endl;
#endif

			RC global_rc __attribute__ ((aligned)) = SUCCESS;
			#pragma omp parallel num_threads( num_threads )
			{
				const size_t irank = grb::config::OMP::current_thread_ID();
				assert( irank < num_threads );

				size_t start, end;
				config::OMP::localRange( start, end, 0, prefix_sum_buffer_size,
						grb::config::CACHE_LINE_SIZE::value(), irank, num_threads );
				for( size_t i = start; i < end; i++ ) {
					prefix_sum_buffer[ i ] = 0;
				}

				#pragma omp barrier

				// count the number of elements per bucket using a thread-private buffer
				RC local_rc = SUCCESS;
				size_t * const thread_prefix_sum_buffer = prefix_sum_buffer
					+ irank * per_thread_buffer_size;
				grb::config::OMP::localRange( start, end, 0, nz,
					config::CACHE_LINE_SIZE::value(), irank, num_threads );
				rndacc_iterator it = _it;
				it += start;
				for( size_t i = start; i < end; i++ ) {
					const ColIndexType col = col_getter( it );
					const size_t bucket_num = col / bucketlen;
					local_rc = utils::check_input_coordinates( it, matrix_rows, matrix_cols );
					if( local_rc != SUCCESS ) {
						break;
					}
					(void) thread_prefix_sum_buffer[ bucket_num ]++;
					(void) ++it;
				}
				if( local_rc != SUCCESS ) {
					// we assume enum writes are atomic in the sense that racing writes will
					// never result in an invalid enum value -- i.e., that the result of a race
					// will be some serialisation of it. If there are multiple threads with
					// different issues reported in their respective local_rc, then this
					// translates to the global_rc reflecting *a* issue, though leaving it
					// undefined *which* issue is reported.
					// The above assumption holds on x86 and ARMv8, iff global_rc is aligned.
					global_rc = local_rc;
				}
				// all threads MUST wait here for the prefix_sum_buffer to be populated
				// and for global_rc to be possibly set otherwise the results are not
				// consistent
				#pragma omp barrier

				// continue only if no thread detected an error
				if( global_rc == SUCCESS ) {
#ifdef _DEBUG
					#pragma omp single
					{
						std::cout << "after first step:" << std::endl;
						for( size_t s = 0; s < prefix_sum_buffer_size; s++ ) {
							std::cout << s << ": " << prefix_sum_buffer[s] << std::endl;
						}
					}
#endif
					// cumulative sum along threads, for each bucket
					grb::config::OMP::localRange( start, end, 0, per_thread_buffer_size,
						config::CACHE_LINE_SIZE::value(), irank, num_threads );
					for( size_t i = start; i < end; i++ ) {
						for( size_t irank = 1; irank < num_threads; irank++ ) {
							prefix_sum_buffer[ irank * per_thread_buffer_size + i ] +=
								prefix_sum_buffer[ ( irank - 1 ) * per_thread_buffer_size + i ];
						}
					}

					#pragma omp barrier

					// at this point, the following array of length per_thread_buffer_size
					// holds the number of elements in each bucket across all threads:
					//   - prefix_sum_buffer + (num_threads - 1) * per_thread_buffer_size

#ifdef _DEBUG
					#pragma omp single
					{
						std::cout << "after second step: " << std::endl;
						for( size_t s = 0; s < prefix_sum_buffer_size; s++ ) {
							std::cout << s << ": " << prefix_sum_buffer[ s ] << std::endl;
						}
					}
#endif
					#pragma omp single
					{
						// cumulative sum for each bucket on last thread, to get the final size of
						// each bucket.
						//
						// This loop is not parallel since no significant speedup measured
						//
						// TODO FIXME This is a prefix sum on prefix_sum_buffer +
						//            (num_threads-1) * per_thread_buffer_size and could
						//            recursively call a prefix sum implementation
						//
						// TODO FIXME Such a recursive call should use
						//            grb::config::OMP::minLoopSize to determine whether to
						//            parallelise.
						//
						// See also internal issue #192 and internal issue #320.
						for( size_t i = 1; i < per_thread_buffer_size; i++ ) {
							prefix_sum_buffer[ (num_threads - 1) * per_thread_buffer_size + i ] +=
								prefix_sum_buffer[ (num_threads - 1) * per_thread_buffer_size + i - 1 ];
						}
					}
#ifdef _DEBUG
					#pragma omp single
					{
						std::cout << "after third step:" << std::endl;
						for( size_t s = 0; s < prefix_sum_buffer_size; s++ ) {
							std::cout << s << ": " << prefix_sum_buffer[ s ] << std::endl;
						}
					}
#endif
					// propagate cumulative sums for each bucket on each thread, to get the
					// final, global, offset
					if( irank < num_threads - 1 ) {
						for( size_t i = 1; i < per_thread_buffer_size; i++ ) {
							prefix_sum_buffer[ irank * per_thread_buffer_size + i ] +=
								prefix_sum_buffer[ (num_threads - 1) * per_thread_buffer_size + i - 1 ];
						}
					}

					#pragma omp barrier
#ifdef _DEBUG
					#pragma omp single
					{
						std::cout << "after fourth step:" << std::endl;
						for( size_t s = 0; s < prefix_sum_buffer_size; s++ ) {
							std::cout << s << ": " << prefix_sum_buffer[ s ] << std::endl;
						}
					}
#endif
					// record value inside storage data structure, with inter-bucket sorting
					// but no intra-bucket sorting
					grb::config::OMP::localRange( start, end, 0, nz,
						grb::config::CACHE_LINE_SIZE::value(), irank, num_threads );
					// iterator in totally new field since sometimes copy-assignment
					// (overwriting an older iterator) may not be defined(??)
					rndacc_iterator rit = _it;
					rit += start;
					for( size_t i = start; i < end; ++i, ++rit ) {
						ColIndexType col = col_getter( rit );
						const size_t bucket_num = col / bucketlen;
						size_t i1 = irank * per_thread_buffer_size + bucket_num;
						(void) --prefix_sum_buffer[ i1 ];
						storage.recordValue( prefix_sum_buffer[ i1 ], populate_ccs, rit );
						if( col_values_buffer != nullptr ) {
							col_values_buffer[ prefix_sum_buffer[ i1 ] ] = col;
						}
					}
				}
			} // end OpenMP parallel section(!), implicit barrier

			if( global_rc != SUCCESS ) {
				std::cerr << "error while reading input values" << std::endl;
				return global_rc;
			}
#ifdef _DEBUG
			std::cout << "CCS/CRS before sort:" << std::endl;
			for( size_t s = 0; s < nz; s++ ) {
				std::cout << s << ": ";
				if( col_values_buffer != nullptr ) {
					std::cout << col_values_buffer[ s ] << ", ";
				}
				std::cout << storage.row_index[ s ] << ", "
					<< storage.getPrintValue( s )
					<< std::endl;
			}
#endif

			if( bucketlen == 1UL ) {
				// if( bucketlen == 1UL ) (i.e., one bucket corresponds to one column, then
				// we are almost done: we must only write the values of the prefix sum into
				// col_start
				assert( prefix_sum_buffer_size >= ccs_col_buffer_size );
				// we still limit the threads by num_threads as it is a good indicator of
				// when the problem is (way) too small to use the full number of available
				// threads
				#pragma omp parallel num_threads( num_threads )
				{
					size_t start, end;
					config::OMP::localRange( start, end, 0, ccs_col_buffer_size );
					for( size_t i = start; i < end; i++ ) {
						storage.col_start[ i ] = prefix_sum_buffer[ i ];
					}
				}
#ifdef _DEBUG
				std::cout << "\t col_start array already fully sorted after bucket sort:\n";
				for( size_t s = 0; s < ccs_col_buffer_size; s++ ) {
					std::cout << "\t\t" << s << ": " << storage.col_start[ s ] << "\n";
				}
				std::cout << "\t exiting" << std::endl;
#endif
				return RC::SUCCESS;
			}

			// In this case, a bucket stores more than one column, and we must sort each
			// bucket prior to generating a col_start.
#ifdef _DEBUG
			std::cout << "\t sorting buckets\n";
#endif
			assert( col_values_buffer != nullptr );
#ifdef _DEBUG
			// fill is not parallel, but so is (not) the printout-- and this is debug
			// mode only
			std::fill( storage.col_start, storage.col_start + ccs_col_buffer_size, 0 );
			std::cout << "\t col_start before sorting:" << std::endl;
			for( size_t s = 0; s < ccs_col_buffer_size; s++ ) {
				std::cout << "\t\t" << s << ": " << storage.col_start[ s ] << "\n";
			}
			std::cout << "\t <end col_start>" << std::endl;
#endif
			// within each bucket, sort all nonzeroes in-place into the final storage,
			// using also the col_values_buffer to store column values, and update
			// the prefix sum in the col_start buffer accordingly.
			// The number of nonzeroes in each bucket is irregular, therefore use a
			// dynamic schedule.
			#pragma omp parallel for schedule( dynamic ), num_threads( num_threads )
			for( size_t i = 0; i < per_thread_buffer_size; i++ ) {
				// ith bucket borders
				const size_t ipsl_min = prefix_sum_buffer[ i ];
				const size_t ipsl_max = (i + 1 < per_thread_buffer_size)
					? prefix_sum_buffer[ i + 1 ]
					: nz;

				ColIndexType previous_destination = std::min( i * bucketlen, num_cols );
				const size_t max_col = std::min( (i + 1) * bucketlen, num_cols );

				if( ipsl_max == ipsl_min ) {
					// the rows are all empty, then done here
#ifdef _DEBUG
					std::cout << "-- thread " << omp_get_thread_num() << ", empty cols fill ["
						<< previous_destination << ", " << max_col << ")" << std::endl;
#endif
					std::fill( storage.col_start + previous_destination,
						storage.col_start + max_col, ipsl_min );
					continue ;
				}

				//do the sort in-place, using the storage and the columns buffer
				NZIterator< ValType, RowIndexType, NonzeroIndexType, ColIndexType > begin(
					storage, col_values_buffer, ipsl_min );
				NZIterator< ValType, RowIndexType, NonzeroIndexType, ColIndexType > end(
					storage, col_values_buffer, ipsl_max );
				std::sort( begin, end );
#ifdef _DEBUG
				std::cout << "-- thread " << omp_get_thread_num() <<", sort [" << previous_destination
					<< ", " << max_col <<")\n" << ">> max_col= " << max_col << std::endl;
#endif
				// INIT: populate initial value with existing count
				storage.col_start[ previous_destination ] = ipsl_min;
#ifdef _DEBUG
				std::cout << "thread " << omp_get_thread_num() << ", init write "
					<< ipsl_min << " to pos " << previous_destination << std::endl;
#endif
				// go through the per-bucket sorted list of nonzeroes and store
				// them into the storage, also updating the prefix sum in col_start
				// starting from the existing values in prefix_sum_buffer and copying
				// the last value in empty columns
				size_t count = ipsl_min;
				size_t previous_count = count;
				size_t col_buffer_index = ipsl_min; // start from next
				while( col_buffer_index < ipsl_max ) {
					const ColIndexType current_col = col_values_buffer[ col_buffer_index ];
					const ColIndexType current_destination = current_col + 1;
					// fill previous columns [previous_destination + 1, current_destination)
					// if skipped because empty
					if( previous_destination + 1 <= current_col ) {
#ifdef _DEBUG
						std::cout << "thread " << omp_get_thread_num() << ", write "
							<< count <<" in range [" << previous_destination + 1
							<< " - " << current_destination << ")" << std::endl;
#endif
						std::fill( storage.col_start + previous_destination + 1,
							storage.col_start + current_destination, previous_count );
					}
					// count occurrences of 'current_col'
					for( ;
						col_buffer_index < ipsl_max &&
							col_values_buffer[ col_buffer_index ] == current_col;
						col_buffer_index++, count++
					);
					assert( current_destination <= max_col );

					// if current_destination < max_col, then write the count;
					// otherwise, the next thread will do it in INIT
					if( current_destination < max_col ) {
						storage.col_start[ current_destination ] = count;
#ifdef _DEBUG
						std::cout << "thread " << omp_get_thread_num() << ", write "
							<< count << " to pos " << current_destination << std::endl;
#endif
					}

					// go for next column
					previous_destination = current_destination;
					previous_count = count;
				}

				// if the columns in [ previous_destination + 1, max_col ) are empty,
				// write the count also there, since the loop has skipped them
				if( previous_destination + 1 < max_col ) {
#ifdef _DEBUG
					std::cout << "thread " << omp_get_thread_num() << ", final write "
						<< previous_count << " in range [" << previous_destination + 1
						<< ", " << max_col << ")" << std::endl;
#endif
					std::fill( storage.col_start + previous_destination + 1,
						storage.col_start + max_col, previous_count );
				}
			}
			const ColIndexType last_existing_col = col_values_buffer[ nz - 1 ];
#ifdef _DEBUG
			std::cout << "final offset " << last_existing_col << std::endl;
#endif

			if( last_existing_col + 1 <= num_cols ) {
#ifdef _DEBUG
				std::cout << "final write " << nz << " into [" << last_existing_col + 1
					<<", " << num_cols << "]" << std::endl;
#endif
				std::fill( storage.col_start + last_existing_col + 1,
					storage.col_start + ccs_col_buffer_size, nz );
			}
#ifdef _DEBUG
			std::cout << "CRS data after sorting:" << std::endl;
			for( size_t s = 0; s < nz; s++ ) {
				std::cout << s << ": ";
				if( col_values_buffer != nullptr ) {
					std::cout << col_values_buffer[ s ] << ", ";
				}
				std::cout << storage.row_index[ s ] << ", "
					<< storage.getPrintValue( s )
					<< std::endl;
			}

			std::cout << "col_start after sorting:" << std::endl;
			for( size_t s = 0; s < ccs_col_buffer_size; s++ ) {
				std::cout << s << ": " << storage.col_start[ s ] << std::endl;
			}
#endif
			return RC::SUCCESS;
		}

		/**
		 * Computes size of memory buffer and threads to use, allowing for maximised
		 * number of threads within a fixed memory budget.
		 *
		 * @param[in] nz           number of nonzeroes
		 * @param[in] sys_threads  maximum number of threads
		 * @param[out] buf_size    size of the buffer for the nonzeroes
		 * @param[out] num_threads number of threads to use
		 */
		static void compute_buffer_size_num_threads(
			const size_t nz,
			const size_t sys_threads,
			size_t &buf_size,
			size_t &num_threads
		) {
			// in case the global buffer already has more memory allocated than would be
			// sufficient for us, make use of that
			const size_t existing_buf_size = getCurrentBufferSize< size_t >();
			const size_t luxury_bucket_factor =
				existing_buf_size / static_cast< double >( nz );
			const size_t luxury_num_threads = sys_threads;
			const size_t luxury_buf_size = existing_buf_size;
			const bool luxury_enabled = luxury_num_threads * luxury_bucket_factor >=
				existing_buf_size / luxury_num_threads;

			// Ideally, we have at least one bucket per thread. However:
			// we may consider making this configurable and so require a minimum number
			// of buckets per threads -- but this may push memory-constrained deployments
			// to run out of memory. That is why we keep it at one for now. Please raise
			// an issue if you would like this functionality.
			constexpr size_t ideal_bucket_factor = 1;

			// if we do not have enough memory for either the luxury or ideal case
			// (compared to nz to ensure memory scalability), then we must reduce the
			// number of threads
			const size_t max_memory = std::min(
				nz,
				sys_threads * sys_threads * ideal_bucket_factor
			);

			// minimum between
			//  - using all threads vs.
			//  - all threads for which there is available memory
			num_threads = std::min(
				static_cast< size_t >(std::sqrt(static_cast< double >(max_memory))),
				sys_threads
			);

			// set buffer size accordingly, also rounding down
			// this is the total number, so it must be multiplied by num_threads
			buf_size = std::max(
					max_memory / num_threads,
					num_threads * ideal_bucket_factor
				) * num_threads;

			// NOTE: this buf_size may or may not be larger than existing_buf_size, in
			//       which case the callee must resize the global buffer. (Only in the
			//       `luxury' case are we guaranteed to not have to resize.)

			// If the above selects the full number of threads *and* we are in a luxury
			// situation with regards to the pre-existing buffer size, then use those
			// settings instead.
			if( num_threads == sys_threads && luxury_enabled ) {
				buf_size = luxury_buf_size;
				num_threads = luxury_num_threads;
			}
		}

		/**
		 * Populates the storage \p storage with the nonzeroes retrieved via the
		 * random access iterator \p _start.
		 *
		 * @tparam populate_ccs Whether \a storage refers to a CRS or CCS. Without
		 *                      loss of generality, the below assumes CCS.
		 *
		 * @param[in] num_cols The number of columns (in case of CCS)
		 * @param[in] num_rows The number of rows (in case of CCS)
		 *
		 * \warning When \a populate_ccs is <tt>false</tt>, \a num_cols refers to the
		 *          number of rows while \a num_rows refers to the number of columns.
		 *
		 * @param[in]  nz      The number of elements in the container to be ingested.
		 * @param[in]  _start  The random access iterator to ingest.
		 * @param[out] storage Where to store the \a nz elements from \a start.
		 */
		template<
			bool populate_ccs,
			typename ColIndexType,
			typename RowIndexType,
			typename ValType,
			typename NonzeroIndexType,
			typename rndacc_iterator
		>
		RC populate_storage(
			const size_t num_cols, const size_t num_rows, const size_t nz,
			const rndacc_iterator &_start,
			Compressed_Storage< ValType, RowIndexType, NonzeroIndexType > &storage
		) {
			// buffer to store prefix sums
			size_t partial_parallel_prefix_sums_buffer_els, partial_parallel_num_threads;

			const size_t max_num_threads =
				static_cast< size_t >( omp_get_max_threads() );

			// maximise the number of threads employed while using a reasonable memory
			// amount: we may use all threads available if we have num_threads * range
			// memory available, but this memory requirement may be (much) larger than
			// nz (when there are many cores or the matrix is very sparse as in nz ~ m).
			// First, compute how many blocks of cache lines correspond to nz nonzeroes:
			const size_t partial_parallel_col_values_buffer_size = (
					(nz * sizeof( ColIndexType ) + config::CACHE_LINE_SIZE::value() - 1)
					/ config::CACHE_LINE_SIZE::value()
				) * config::CACHE_LINE_SIZE::value();

			// Second, decide how many threads we may use
			compute_buffer_size_num_threads(
				nz, max_num_threads,
				partial_parallel_prefix_sums_buffer_els, partial_parallel_num_threads
			);

			// partial_parallel_prefix_sums_buffer_els = std::max( nz, max_num_threads );
			// num_threads = max_num_threads;

			const size_t partial_parallel_prefix_sums_buffer_size =
				partial_parallel_prefix_sums_buffer_els * sizeof( size_t );

			const size_t partial_parallel_buffer_size = partial_parallel_col_values_buffer_size
				+ partial_parallel_prefix_sums_buffer_size;

			const size_t fully_parallel_buffer_els = max_num_threads * ( num_cols + 1 ); // + 1 for prefix sum
			const size_t fully_parallel_buffer_size = fully_parallel_buffer_els * sizeof( size_t );

			const size_t existing_buf_size = getCurrentBufferSize< size_t >();
			const bool is_fully_parallel = fully_parallel_buffer_size <= partial_parallel_buffer_size
				// a buffer already exists large enough for fully parallel execution
				|| existing_buf_size >= fully_parallel_buffer_els;

#ifdef _DEBUG
			if( is_fully_parallel ) {
				std::cout << "fully parallel matrix creation: no extra sorting required" << std::endl;
			} else {
				std::cout << "partially parallel matrix creation: extra sorting required" << std::endl;
				std::cout << "partial_parallel_num_threads= " << partial_parallel_num_threads << std::endl;
				std::cout << "partial_parallel_prefix_sums_buffer_els= "
					<< partial_parallel_prefix_sums_buffer_els << std::endl;
			}
#endif

			const size_t bufferlen_tot = is_fully_parallel
				? fully_parallel_buffer_size
				: partial_parallel_buffer_size;
			if( !internal::ensureReferenceBufsize< unsigned char >( bufferlen_tot ) ) {
#ifndef _DEBUG
				std::cerr << "Not enough memory available for populate_storage_parallel buffer" << std::endl;
#endif
				return RC::OUTOFMEM;
			}

			const size_t prefix_sum_buffer_els = is_fully_parallel
				? fully_parallel_buffer_els
				: partial_parallel_prefix_sums_buffer_els;

			unsigned char * const __buffer =
				getReferenceBuffer< unsigned char >( bufferlen_tot );

			size_t * pref_sum_buffer = is_fully_parallel
				? reinterpret_cast < size_t * >( __buffer )
				: reinterpret_cast < size_t * >(
						__buffer + partial_parallel_col_values_buffer_size
					);

			ColIndexType* col_values_buffer = is_fully_parallel
				? nullptr
				: reinterpret_cast < ColIndexType * >( __buffer );

			const size_t num_threads = is_fully_parallel
				? max_num_threads
				: partial_parallel_num_threads;

			return populate_storage_parallel< populate_ccs >(
				_start, nz, num_cols, num_rows,
				num_threads, pref_sum_buffer, prefix_sum_buffer_els, col_values_buffer,
				storage
			);
		}
#endif


	} // end namespace internal

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	size_t nrows( const Matrix< DataType, reference, RIT, CIT, NIT > & ) noexcept;

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	size_t ncols( const Matrix< DataType, reference, RIT, CIT, NIT > & ) noexcept;

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	size_t nnz( const Matrix< DataType, reference, RIT, CIT, NIT > & ) noexcept;

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	RC clear( Matrix< InputType, reference, RIT, CIT, NIT > & ) noexcept;

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	RC resize(
		Matrix< DataType, reference, RIT, CIT, NIT > &,
		const size_t
	) noexcept;

	template<
		Descriptor,
		class ActiveDistribution, typename Func, typename DataType,
		typename RIT, typename CIT, typename NIT
	>
	RC eWiseLambda(
		const Func f,
		const Matrix< DataType, reference, RIT, CIT, NIT > &A,
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
	class Matrix< D, reference, RowIndexType, ColIndexType, NonzeroIndexType > {

		static_assert( !grb::is_object< D >::value,
			"Cannot create an ALP matrix of ALP objects!" );

		/* *********************
			I/O friends
		   ********************* */

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		friend size_t nrows(
			const Matrix< DataType, reference, RIT, CIT, NIT > &
		) noexcept;

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		friend size_t ncols(
			const Matrix< DataType, reference, RIT, CIT, NIT > &
		) noexcept;

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		friend size_t nnz(
			const Matrix< DataType, reference, RIT, CIT, NIT > &
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend RC clear(
			Matrix< InputType, reference, RIT, CIT, NIT > &
		) noexcept;

		template< typename DataType, typename RIT, typename CIT, typename NIT  >
		friend RC resize(
			Matrix< DataType, reference, RIT, CIT, NIT > &,
			const size_t
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend uintptr_t getID(
			const Matrix< InputType, reference, RIT, CIT, NIT > &
		);

		template<
			Descriptor descr, typename InputType,
			typename RIT, typename CIT, typename NIT,
			typename fwd_iterator
		>
		friend RC buildMatrixUnique(
			Matrix< InputType, reference, RIT, CIT, NIT > &,
			fwd_iterator, const fwd_iterator,
			const IOMode
		);

		/* *********************
			BLAS2 friends
		   ********************* */

		template<
			Descriptor,
			typename Func,
			typename DataType1, typename RIT, typename CIT, typename NIT
		>
		friend RC eWiseLambda(
			const Func,
			const Matrix< DataType1, reference, RIT, CIT, NIT > &
		);

		template<
			Descriptor,
			class ActiveDistribution, typename Func, typename DataType,
			typename RIT, typename CIT, typename NIT
		>
		friend RC internal::eWiseLambda(
			const Func,
			const Matrix< DataType, reference, RIT, CIT, NIT > &,
			const size_t, const size_t, const size_t, const size_t
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
			Vector< IOType, reference, Coords > &destination_vector,
			IOType * __restrict__ const &destination,
			const size_t &destination_range,
			const Vector< InputType1, reference, Coords > &source_vector,
			const InputType1 * __restrict__ const &source,
			const size_t &source_index,
			const internal::Compressed_Storage<
				InputType2, RowColType, NonzeroType
			> &matrix,
			const Vector< InputType3, reference, Coords > &mask_vector,
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
			Vector< IOType, reference, Coords > &u,
			const Vector< InputType3, reference, Coords > &mask,
			const Vector< InputType1, reference, Coords > &v,
			const Vector< InputType4, reference, Coords > &v_mask,
			const Matrix< InputType2, reference, RIT, CIT, NIT > &A,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const std::function< size_t( size_t ) > &row_l2g,
			const std::function< size_t( size_t ) > &row_g2l,
			const std::function< size_t( size_t ) > &col_l2g,
			const std::function< size_t( size_t ) > &col_g2l
		);

		/* ********************
		     Internal friends
		   ******************** */

		friend internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType > &
		internal::getCRS<>(
			Matrix<
				D, reference,
				RowIndexType, ColIndexType, NonzeroIndexType
			> &A
		) noexcept;

		friend const internal::Compressed_Storage<
			D,
			RowIndexType, NonzeroIndexType
		> & internal::getCRS<>(
			const Matrix<
				D, reference,
				RowIndexType, ColIndexType, NonzeroIndexType
			> &A
		) noexcept;

		friend internal::Compressed_Storage< D, ColIndexType, NonzeroIndexType > &
		internal::getCCS<>(
			Matrix<
				D, reference,
				RowIndexType, ColIndexType, NonzeroIndexType
			> &A
		) noexcept;

		friend const internal::Compressed_Storage<
			D, ColIndexType, NonzeroIndexType
		> & internal::getCCS<>(
			const Matrix<
				D, reference,
				RowIndexType, ColIndexType, NonzeroIndexType
			> &A
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend const size_t & internal::getNonzeroCapacity(
			const grb::Matrix< InputType, reference, RIT, CIT, NIT > &
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend const size_t & internal::getCurrentNonzeroes(
			const grb::Matrix< InputType, reference, RIT, CIT, NIT > &
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend void internal::setCurrentNonzeroes(
			grb::Matrix< InputType, reference, RIT, CIT, NIT > &, const size_t
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend void internal::getMatrixBuffers(
			char *&, char *&, InputType *&,
			const unsigned int,
			const grb::Matrix< InputType, reference, RIT, CIT, NIT > &
		) noexcept;

		friend const grb::Matrix<
			D, reference,
			ColIndexType, ColIndexType, NonzeroIndexType
		>
		internal::wrapCRSMatrix< D, ColIndexType, NonzeroIndexType, reference >(
			const D *__restrict__ const,
			const ColIndexType *__restrict__ const,
			const NonzeroIndexType *__restrict__ const,
			const size_t, const size_t
		);

		friend grb::Matrix<
			D, reference,
			ColIndexType, ColIndexType, NonzeroIndexType
		>
		internal::wrapCRSMatrix< D, ColIndexType, NonzeroIndexType, reference >(
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
				D, reference,
				RowIndexType, ColIndexType, NonzeroIndexType
			> SelfType;

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
				const size_t rows, const size_t cols,
				const size_t cap_in
			) {
#ifdef _DEBUG
				std::cerr << "\t in Matrix< reference >::initialize...\n"
					<< "\t\t matrix size " << rows << " by " << cols << "\n"
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
				if( cols >= static_cast< size_t >(
						std::numeric_limits< ColIndexType >::max()
					)
				) {
					throw std::overflow_error( "Number of columns larger than configured "
						"ColIndexType maximum!" );
				}

				// memory allocations
				RC alloc_ok = SUCCESS;
				char * alloc[ 8 ] = {
					nullptr, nullptr, nullptr, nullptr,
					nullptr, nullptr, nullptr, nullptr
				};
				if( !internal::template ensureReferenceBufsize< char >(
					reqBufSize( rows, cols ) )
				) {
					throw std::runtime_error( "Could not resize global buffer" );
				}
				if( rows > 0 && cols > 0 ) {
					// check whether requested capacity is sensible
					if( cap_in / rows > cols ||
						cap_in / cols > rows ||
						(cap_in / rows == cols && (cap_in % rows > 0)) ||
						(cap_in / cols == rows && (cap_in % cols > 0))
					) {
#ifdef _DEBUG
						std::cerr << "\t\t Illegal capacity requested\n";
#endif
						throw std::runtime_error( toString( ILLEGAL ) );
					}
					// get sizes of arrays that we need to allocate
					size_t sizes[ 12 ];
					sizes[ 0 ] = internal::Coordinates< reference >::arraySize( rows );
					sizes[ 1 ] = internal::Coordinates< reference >::arraySize( cols );
					sizes[ 2 ] = internal::Coordinates< reference >::bufferSize( rows );
					sizes[ 3 ] = internal::Coordinates< reference >::bufferSize( cols );
					sizes[ 4 ] = rows * internal::SizeOf< D >::value;
					sizes[ 5 ] = cols * internal::SizeOf< D >::value;
					CRS.getStartAllocSize( &( sizes[ 6 ] ), rows );
					CCS.getStartAllocSize( &( sizes[ 7 ] ), cols );
					if( cap_in > 0 ) {
						CRS.getAllocSize( &(sizes[ 8 ]), cap_in );
						CCS.getAllocSize( &(sizes[ 10 ]), cap_in );
					} else {
						sizes[ 8 ] = sizes[ 9 ] = sizes[ 10 ] = sizes[ 11 ] = 0;
					}
					// allocate required arrays
					alloc_ok = utils::alloc(
						"grb::Matrix< T, reference >::Matrix()",
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
						rows * internal::SizeOf< D >::value,
						cols * internal::SizeOf< D >::value
					};
					coorArr[ 0 ] = coorArr[ 1 ] = nullptr;
					coorBuf[ 0 ] = coorBuf[ 1 ] = nullptr;
					alloc_ok = utils::alloc(
						"grb::Matrix< T, reference >::Matrix()",
						"empty allocation",
						alloc[ 6 ], sizes[ 0 ], false, _local_deleter[ 4 ],
						alloc[ 7 ], sizes[ 1 ], false, _local_deleter[ 5 ]
					);
				}

				// check allocation status
				if( alloc_ok == OUTOFMEM ) {
					throw std::runtime_error( "Could not allocate memory during grb::Matrix construction" );
				} else if( alloc_ok != SUCCESS ) {
					throw std::runtime_error( toString( alloc_ok ) );
				}
#ifdef _DEBUG
				if( rows > 0 && cols > 0 ) {
					std::cerr << "\t\t allocations for an " << m << " by " << n << " matrix "
						<< "have successfully completed.\n";
				} else {
					std::cerr << "\t\t allocations for an empty matrix have successfully "
						<< "completed.\n";
				}
#endif
				// either set ID or retrieve one
				if( id_in != nullptr ) {
					assert( !remove_id );
					id = *id_in;
#ifdef _DEBUG
					std::cerr << "\t\t inherited ID " << id << "\n";
#endif
				} else {
					if( rows > 0 && cols > 0 && id_in == nullptr ) {
						id = internal::reference_mapper.insert(
							reinterpret_cast< uintptr_t >(alloc[ 0 ])
						);
						remove_id = true;
#ifdef _DEBUG
						std::cerr << "\t\t assigned new ID " << id << "\n";
#endif
					}
				}

				// all OK, so set and exit
				m = rows;
				n = cols;
				nz = 0;
				if( m > 0 && n > 0 ) {
					cap = cap_in;
				}
				valbuf[ 0 ] = reinterpret_cast< D * >( alloc[ 6 ] );
				valbuf[ 1 ] = reinterpret_cast< D * >( alloc[ 7 ] );
				if( m > 0 && n > 0 ) {
					CRS.replaceStart( alloc[ 0 ] );
					CCS.replaceStart( alloc[ 1 ] );
					CRS.replace( alloc[ 2 ], alloc[ 3 ] );
					CCS.replace( alloc[ 4 ], alloc[ 5 ] );
				}
			}

			/** Implements a move. */
			void moveFromOther( SelfType &&other ) {
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

			/**
			 * Sets CRS and CCS offset arrays to zero.
			 *
			 * Does not clear any other field.
			 *
			 * It relies on the values stored in the m and n fields for the sizes.
			 */
			void clear_cxs_offsets() {
				// two-phase strategy: fill until minimum, then continue filling the larger
				// array.
				size_t min_dim = static_cast< size_t >( std::min( m, n ) );
				size_t max_dim = static_cast< size_t >( std::max( m, n ) );
				NonzeroIndexType * const larger = max_dim == m
					? CRS.col_start
					: CCS.col_start;

				// fill until minimum
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
				#pragma omp parallel
				{
					size_t start, end;
					config::OMP::localRange( start, end, 0, min_dim );
#else
					size_t start = 0;
					size_t end = min_dim;
#endif
					for( size_t i = start; i < end; ++i ) {
						CRS.col_start[ i ] = 0;
						CCS.col_start[ i ] = 0;
					}
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
					config::OMP::localRange( start, end, min_dim, max_dim );
#else
					start = min_dim;
					end = max_dim;
#endif
					for( size_t i = start; i < end; ++i ) {
						larger[ i ] = 0;
					}
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
				}
#endif
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
				if( CRS.col_start == nullptr || CCS.col_start == nullptr ) {
					// sanity check
					assert( CRS.col_start == nullptr && CCS.col_start == nullptr );
				} else {
					// clear offsets
					clear_cxs_offsets();
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
				size_t old_sizes[ 4 ] = { 0, 0, 0, 0 };
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
					"grb::Matrix< T, reference >::resize", description.str(),
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
					if( config::MEMORY::report( "grb::Matrix< T, reference >::resize",
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

			/**
			 * @see Matrix::buildMatrixUnique.
			 *
			 * This dispatcher calls the sequential or the parallel implementation based
			 * on the tag of the input iterator of type \p input_iterator.
			 */
			template<
				Descriptor descr = descriptors::no_operation,
				typename InputIterator
			>
			RC buildMatrixUnique(
				const InputIterator &_start,
				const InputIterator &_end,
				const IOMode mode
			) {
				// here we can safely ignore the mode and dispatch based only on the
				// iterator type since in shared memory the input data reside by definition
				// all on the same machine
				(void) mode;
				static_assert( utils::is_alp_matrix_iterator< D, InputIterator >::value,
					"the given iterator is not a valid input iterator, "
					"see the ALP specification for input iterators"
				);
				typename std::iterator_traits< InputIterator >::iterator_category category;
				return buildMatrixUniqueImpl( _start, _end, category );
			}

			/**
			 * When given a forward iterator tag, calls the sequential implementation of
			 * buildMatrixUnique.
			 */
			template< typename fwd_iterator >
			RC buildMatrixUniqueImpl(
				const fwd_iterator &_start,
				const fwd_iterator &_end,
				std::forward_iterator_tag
			) {
				return buildMatrixUniqueImplSeq( _start, _end );
			}

			/**
			 * The sequential implementation of buildMatrixUnique.
			 */
			template< typename fwd_iterator >
			RC buildMatrixUniqueImplSeq(
				const fwd_iterator &_start,
				const fwd_iterator &_end
			) {
#ifdef _DEBUG
				std::cout << "forward access iterator detected\n";
				std::cout << "buildMatrixUnique called with " << cap << " nonzeroes.\n";
				std::cout << "buildMatrixUnique: input is\n";
				for( fwd_iterator it = _start; it != _end; ++it ) {
					std::cout << "\t" << it.i() << ", " << it.j() << "\n";
				}
				std::cout << "buildMatrixUnique: end input.\n";
#endif
				// detect trivial case
				if( _start == _end || m == 0 || n == 0 ) {
					return SUCCESS;
				}
				// keep count of nonzeroes
				nz = 0;

				// counting sort, phase 1
				clear_cxs_offsets();
				for( fwd_iterator it = _start; it != _end; ++it ) {
					if( utils::check_input_coordinates( it, m, n ) != SUCCESS ) {
						return MISMATCH;
					}
					(void) ++( CRS.col_start[ it.i() ] );
					(void) ++( CCS.col_start[ it.j() ] );
					(void) ++nz;
				}

				// check if we can indeed store nz values
				if( nz >= static_cast< size_t >(
						std::numeric_limits< grb::config::NonzeroIndexType >::max()
					)
				) {
					return OVERFLW;
				}

				// put final entries in offset arrays
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

				// counting sort, phase 2
				fwd_iterator it = _start;
				for( size_t k = 0; it != _end; ++k, ++it ) {
					const size_t crs_pos = --( CRS.col_start[ it.i() ] );
					CRS.recordValue( crs_pos, false, it );
#ifdef _DEBUG
					std::cout << "Nonzero " << k << ", ( " << it.i() << ", " << it.j() << " ) "
						<< "is stored at CRS position "
						<< static_cast< size_t >( crs_pos ) << ".\n";
#endif
					const size_t ccs_pos = --( CCS.col_start[ it.j() ] );
					CCS.recordValue( ccs_pos, true, it );
#ifdef _DEBUG
					std::cout << "Nonzero " << k << ", ( " << it.i() << ", " << it.j() << " ) "
						<< "is stored at CCS position "
						<< static_cast< size_t >( ccs_pos ) << ".\n";
#endif
				}
#ifdef _DEBUG
				for( size_t i = 0; i <= m; ++i ) {
					std::cout << "row_start[ " << i << " ] = " << CRS.col_start[ i ]
						<< "." << std::endl;
				}
				for( size_t i = 0; i <= n; ++i ) {
					std::cout << "col_start[ " << i << " ] = " << CCS.col_start[ i ]
						<< "." << std::endl;
				}
#endif
				// done
				return SUCCESS;
			}

#ifdef _H_GRB_REFERENCE_OMP_MATRIX
			/**
			 * The shared-memory parallel implementation of buildMatrixUnique.
			 */
			template< typename rndacc_iterator >
			RC buildMatrixUniqueImpl(
				const rndacc_iterator &_start,
				const rndacc_iterator &_end,
				std::random_access_iterator_tag
			) {
#ifdef _DEBUG
				std::cout << " rnd access iterator " << '\n';
				std::cout << "buildMatrixUnique called with " << cap << " nonzeroes.\n";
				std::cout << "buildMatrixUnique: input is\n";
				for( rndacc_iterator it = _start; it != _end; ++it ) {
					std::cout << "\t" << it.i() << ", " << it.j() << "\n";
				}
				std::cout << "buildMatrixUnique: end input.\n";
#endif

				// detect trivial case
				if( _start == _end || m == 0 || n == 0 ) {
					return SUCCESS;
				}

				// count of nonzeroes
				size_t _nz = _end - _start;

				// check if we can indeed store nz values
				if( _nz >= static_cast< size_t >(
						std::numeric_limits< NonzeroIndexType >::max()
					)
				) {
					// no need to clean here, since we did not allocate any additional memory
					return RC::OVERFLW;
				}
				// after checkign it's possible, store it
				nz = _nz;

				// for small sizes, delegate to sequential routine
				// since the overheads of OMP dominate for small sizes
				if(
					nz <= static_cast< size_t >( omp_get_max_threads() ) ||
					nz < config::OMP::minLoopSize()
				) {
					return buildMatrixUniqueImplSeq( _start, _end );
				}

				// reset col_start arrays to zero
				clear_cxs_offsets();

				// put final entries
				CRS.col_start[ m ] = nz;
				CCS.col_start[ n ] = nz;

				// allocate enough space
				RC ret = resize( nz );
				if( ret != SUCCESS ) {
#ifdef _DEBUG
					std::cerr << "cannot resize the matrix to store the nonzero" << std::endl;
#endif
					return ret;
				}
				ret = internal::populate_storage<
					true, ColIndexType, RowIndexType
				>( n, m, nz, _start, CCS );
				if( ret != SUCCESS ) {
#ifdef _DEBUG
					std::cerr << "cannot populate the CRS" << std::endl;
#endif
					clear(); // we resized before, so we need to clean the memory
					return ret;
				}
				ret = internal::populate_storage<
					false, RowIndexType, ColIndexType
				>( m, n, nz, _start, CRS );
				if( ret != SUCCESS ) {
#ifdef _DEBUG
					std::cerr << "cannot populate the CCS" << std::endl;
#endif
					clear();
				}
				return ret;
			}
#endif


		public:

			/** @see Matrix::value_type */
			typedef D value_type;

			/** The iterator type over matrices of this type. */
			typedef typename internal::Compressed_Storage<
				D, RowIndexType, NonzeroIndexType
			>::template ConstIterator<
				internal::Distribution< reference >
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
			Matrix( const size_t rows, const size_t columns, const size_t nz ) :
				Matrix()
			{
#ifdef _DEBUG
				std::cout << "In grb::Matrix constructor (reference, with requested "
					<< "capacity)\n";
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
				std::cout << "In grb::Matrix constructor (reference, default capacity)\n";
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
					D, reference,
					RowIndexType, ColIndexType, NonzeroIndexType
				> &other
			) :
				Matrix( other.m, other.n, other.cap )
			{
#ifdef _DEBUG
				std::cerr << "In grb::Matrix (reference) copy-constructor\n"
					<< "\t source matrix has " << other.nz << " nonzeroes\n";
#endif
				nz = other.nz;

				// if empty, return; otherwise copy
				if( nz == 0 ) { return; }

#ifdef _H_GRB_REFERENCE_OMP_MATRIX
				#pragma omp parallel
#endif
				{
					size_t range = CRS.copyFromRange( nz, m );
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
					size_t start, end;
					config::OMP::localRange( start, end, 0, range );
#else
					const size_t start = 0;
					size_t end = range;
#endif
					CRS.copyFrom( other.CRS, nz, m, start, end );
					range = CCS.copyFromRange( nz, n );
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
					config::OMP::localRange( start, end, 0, range );
#else
					end = range;
#endif
					CCS.copyFrom( other.CCS, nz, n, start, end );
				}
			}

			/** \internal No implementation notes. */
			Matrix( SelfType &&other ) noexcept {
				moveFromOther( std::forward< SelfType >(other) );
			}

			/** \internal No implementation notes. */
			SelfType& operator=( SelfType &&other ) noexcept {
				moveFromOther( std::forward< SelfType >(other) );
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
			 *       (GitHub issue 32)
			 */
			template< class ActiveDistribution = internal::Distribution< reference > >
			typename internal::Compressed_Storage<
				D,
				RowIndexType, NonzeroIndexType
			>::template ConstIterator< ActiveDistribution > begin(
				const IOMode mode = PARALLEL,
				const size_t s = 0, const size_t P = 1
			) const {
				assert( mode == PARALLEL );
				(void) mode;
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
			 *       (GitHub issue 32)
			 */
			template< class ActiveDistribution = internal::Distribution< reference > >
			typename internal::Compressed_Storage<
				D,
				RowIndexType,
				NonzeroIndexType
			>::template ConstIterator< ActiveDistribution > end(
				const IOMode mode = PARALLEL,
				const size_t s = 0, const size_t P = 1
			) const {
				assert( mode == PARALLEL );
				(void) mode;
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
			 *       (GitHub issue 32)
			 */
			template< class ActiveDistribution = internal::Distribution< reference > >
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
			 *       (GitHub issue 32)
			 */
			template< class ActiveDistribution = internal::Distribution< reference > >
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
	struct is_container< Matrix< D, reference, RIT, CIT, NIT > > {
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
  #define reference reference_omp
  #include "graphblas/reference/matrix.hpp"
  #undef reference
  #undef _H_GRB_REFERENCE_OMP_MATRIX
 #endif
#endif

#endif // end ``_H_GRB_REFERENCE_MATRIX''

