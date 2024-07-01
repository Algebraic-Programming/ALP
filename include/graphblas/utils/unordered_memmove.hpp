
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
 * Sequential and parallel algorithms for moving array elements to another,
 * potentially overlapping, array, without necessarily retaining the original
 * element order. The functions provided support batched movements and, for the
 * provided parallel variants, parallelise across batches.
 *
 * Both sequential and parallel variants are provided; see
 *  -# #grb::utils::unordered_memmove_seq,
 *  -# #grb::utils::unordered_memmove_omp, and
 *  -# #grb::utils::unordered_memmove_ompPar.
 *
 * \todo This utility relies on <tt>memmove</tt> and <tt>memcpy</tt>. We may
 *       want to check if providing our own implementations for these functions
 *       can outperform the system-provided variants -- but this should not be
 *       the case.
 *
 * @author A. N. Yzelman
 * @date 19th of June, 2024
 */

#ifndef _H_GRB_UTILS_UNORDERED_MEMMOVE
#define _H_GRB_UTILS_UNORDERED_MEMMOVE

#ifdef _GRB_WITH_OMP
 #include <graphblas/omp/config.hpp>
#endif

#include <graphblas/base/config.hpp>

#include <graphblas/utils/binsearch.hpp>

#include <cstring>


namespace grb {

	namespace utils {

		/**
		 * Rearranges an array of elements by copying batches of elements from one
		 * source location to a given destination location.
		 *
		 * The source and destination locations are given as offsets into the same
		 * \a source array. The \a source and \a destination memory regions of each
		 * batch may overlap.
		 *
		 * This function, on the granularity of a single batch, hence is similar to
		 * <tt>memmove</tt>, except that 1) this function is aware of array elements
		 * that not necessary have the size of a single byte, 2) this function
		 * may operate on multiple batches of ``memmoves'' simultaneously, while
		 * point 3) is better to put as a clear warning:
		 *
		 * \warning this function may change the order in which elements are
		 *          stored.
		 *
		 * @param[in,out] source  Pointer to the array that contains the batches to be
		 *                        moved around within the array.
		 * @param[in] src_offsets The start offset in \a source that indicates the
		 *                        start position of each batch in the array.
		 * @param[in] dst_offsets The start offset in \a source that indicates where
		 *                        each batch should move to.
		 * @param[in] batches     How many batches in \a source should be moved.
		 *
		 * The array \a src_offsets must contain \a batches + 1 elements. The last
		 * element in \a src_offsets designates where the last batch ends.
		 *
		 * For any subsequent pair of entries in \a dst_offsets at positions \f$ i \f$
		 * and \f$ i + 1 \f$, the difference must be greater than or equal to the
		 * difference between the subsequent pair at the same locations
		 * \f$ i, i + 1 \f$ of \a src_offsets. More informally, the destination
		 * offsets at position \f$ i, i + 1 \f$ must allow enough space to store the
		 * full \f$ i \f$th batch. Violation of this constraint will lead to undefined
		 * behaviour.
		 *
		 * The \a src_offsets and \a dst_offsets may be such that there may be
		 * arbitrary overlaps between different batches and where they should move.
		 *
		 * \note A more classic <tt>memmove</tt> type interface takes two pointers
		 *       and an integer, which indicate the destination location, the source
		 *       location, and the size of a memory area. This can be translated in
		 *       the below interface by taking the minimum of the source and
		 *       destination pointer, choosing the source and destination offsets
		 *       accordingly, and for \a src_offsets, to pack the source offset
		 *       together with the size argument. We presently do \em not provide
		 *       such a more canonical interface since there is no use case for it
		 *       (yet)-- anyone who has one is welcome to submit a feature request,
		 *       however, and we will look into supplying one.
		 *
		 * Optionally, a workspace may be provided. The workspace may or may not be
		 * used to speed up the requested operation.
		 *
		 * @param[in,out] workspace      A pointer to a unique memory region that may
		 *                               hold up to \a workspace_size elements.
		 * @param[in]     workspace_size The size (in number of elements) of the
		 *                               \a workspace.
		 *
		 * This variant implements a sequential unordered memmove-- this is indicated
		 * by the postfix <tt>_seq</tt>. An OpenMP parallel variant is indicated by
		 * the postfix <tt>_par</tt>. That variant creates its own parallel region.
		 * An OpenMP variant that may be called from within a pre-existing parallel
		 * region has postfix <tt>_ompPar</tt>.
		 */
		template< typename T, typename IND >
		void unordered_memmove_seq(
			T * const source,
			const IND * const src_offsets,
			const IND * const dst_offsets,
			const size_t batches,
			T * const workspace = nullptr,
			const size_t workspace_size = 0
		) {
			// in the sequential case, use of a buffer is never sensible
			(void) workspace;
			(void) workspace_size;

			// we work from the last batch to the first, thus consecutively freeing up
			// space for each successive move of a batch
			for( size_t i = batches - i; i < batches; --i ) {
				// simply cast the single-batch operation back to a call to memmove
				const size_t nelems = src_offsets[ i + 1 ] - src_offsets[ i ];
				const size_t bsize = nelems * sizeof( T );
				(void) std::memmove(
					source + dst_offsets[ i ], source + src_offsets[ i ], bsize );
			}
		}

#ifdef _GRB_WITH_OMP

		namespace internal {

			/**
			 * This implements case 1 of the #grb::utils::unordered_memmove_ompPar.
			 *
			 * Please see the code comments in that function for a description of this
			 * one.
			 */
			template< typename T, typename IND >
			void unordered_memmove_ompPar_case1(
				T * const source,
				const IND * const src_offsets,
				const IND * const dst_offsets,
				const size_t batches
			) {
				// dynamic checks
 #ifndef NDEBUG
				const size_t upper = static_cast< size_t >(src_offsets[ batches ]);
				for( size_t i = 0; i < batches; ++i ) {
					assert( static_cast< size_t >(dst_offsets[ i ]) >= upper );
				}
 #endif
				// partition the source elements across threads
				size_t start, end;
				config::OMP::localRange(
					start, end,
					src_offsets[ 0 ], src_offsets[ batches ]
				);
				if( start < end ) {
					assert( start < static_cast< size_t >(src_offsets[ batches ]) );
					assert( end <= static_cast< size_t >(src_offsets[ batches ] ));
					const size_t my_start_batch =
						grb::utils::binsearch( start, src_offsets, src_offsets + batches );
					const size_t my_end_batch =
						grb::utils::binsearch( end, src_offsets, src_offsets + batches );
					assert( my_start_batch < batches );
					assert( my_end_batch <= batches );
					assert( my_start_batch < my_end_batch );
					assert( static_cast< size_t >(src_offsets[ my_start_batch ]) <= start );
					assert( start < static_cast< size_t >(src_offsets[ my_start_batch + 1 ]) );
					assert( static_cast< size_t >(src_offsets[ my_end_batch - 1 ]) <= end );
					assert( end <= static_cast< size_t >(src_offsets[ my_end_batch ]) );
					(void) std::memcpy(
						source + dst_offsets[ my_start_batch ] +
							start - src_offsets[ my_start_batch ],
						source + start,
						(src_offsets[ my_start_batch + 1 ] - start) * sizeof( T )
					);
					for( size_t k = my_start_batch + 1; k < my_end_batch - 1; ++k ) {
						(void) std::memcpy(
							source + dst_offsets[ k ],
							source + src_offsets[ k ],
							(src_offsets[ k + 1 ] - src_offsets[ k ]) * sizeof( T )
						);
					}
					(void) std::memcpy(
						source + dst_offsets[ my_end_batch - 1 ],
						source + src_offsets[ my_end_batch - 1 ],
						(end - src_offsets[ my_end_batch ]) * sizeof( T )
					);
				}
			}

			/**
			 * This implements case 2 of the #grb::utils::unordered_memmove_ompPar.
			 *
			 * Please see the code comments in that function for a description of this
			 * one.
			 */
			template< typename T, typename IND >
			void unordered_memmove_ompPar_case2_inplace(
				T * const source,
				const IND * const src_offsets,
				const IND * const dst_offsets
			) {
				assert( *dst_offsets < src_offsets[ 1 ] );
				assert( src_offsets[ 0 ] < *dst_offsets );
				assert( src_offsets[ 0 ] < src_offsets[ 1 ] );
				const size_t copySize = *dst_offsets - *src_offsets;
				const size_t offset = src_offsets[ 1 ] - *dst_offsets;
				size_t start, end;
				config::OMP::localRange( start, end, 0, copySize );
				if( start < end ) {
					(void) std::memcpy( source + offset, source, copySize * sizeof( T ) );
				}
			}

			/** @returns The parallelism of the above function. */
			template< typename IND >
			void unordered_memmove_ompPar_case2_inplace_parallelism(
				const IND * const src_offsets,
				const IND * const dst_offsets
			) {
				return config::OMP::nranges( *dst_offsets - *src_offsets );
			}

		} // end namespace grb::utils::internal

		/**
		 * Rearranges an array of elements by copying batches of elements from one
		 * source location to a given destination location.
		 *
		 * See #unordered_memmove_seq for full documentation.
		 *
		 * This variant implements the OpenMP parallel variant that is designed to be
		 * called from a pre-existing OpenMP parallel region. This is indicated by the
		 * postfix <tt>_ompPar</tt>.
		 * Other variants are the sequential unordered memmove-- indicated by the
		 * postfix <tt>_seq</tt>. An OpenMP parallel variant (to be called from a
		 * serial context and thus spawning its own parallel region) is indicated by
		 * the postfix <tt>_par</tt>.
		 *
		 * @param[in,out] source  Pointer to the array that contains the batches to be
		 *                        moved around within the array.
		 * @param[in] src_offsets The start offset in \a source that indicates the
		 *                        start position of each batch in the array.
		 * @param[in] dst_offsets The start offset in \a source that indicates where
		 *                        each batch should move to.
		 * @param[in] batches     How many batches in \a source should be moved.
		 *
		 * Optionally, a workspace may be provided. The workspace may or may not be
		 * used to speed up the requested operation.
		 *
		 * @param[in,out] workspace      A pointer to a unique memory region that may
		 *                               hold up to \a workspace_size elements.
		 * @param[in]     workspace_size The size (in number of elements) of the
		 *                               \a workspace.
		 */
		template< typename T, typename IND >
		void unordered_memmove_ompPar(
			T * const source,
			const IND * const src_offsets,
			const IND * const dst_offsets,
			const size_t batches,
			T * const workspace = nullptr,
			const size_t workspace_size = 0
		) {
			// In the parallel case, things get more complicated. The main idea is still
			// to work from the last batch to the first batch, in order to free up space
			// in the array that is potentially / likely overwritten by subsequent batch
			// moves.
			// There are two main cases: 1) the source batches move to destinations that
			// are beyond the last element of the last batch (i.e., no overlap), and 2)
			// the last batch and the destination it moves to, overlap.
			// In case #1, potentially many batches can be moved in parallel without risk
			// of conflict. The implementation here first identifies how many such
			// batches can move, then finds how many elements those span, and then
			// distributes those elements equally across all available threads. These
			// then are copied into their respective destinations. Note that with this
			// approach, different threads may contribute to the movement of a different
			// number of batches.
			// In case #2, since we allow for changing the order of elements as they
			// appear in a single batch, we may move only the head of the batch and
			// append those elements at the tail of the batch in order to complete the
			// move. This `head-move' can be executed in parallel -- if there are
			// enough elements. If there are not, this case results in a necessary
			// sequential phase-- we cannot process any other batches, as the
			// destinations for those batches lie precisely in the array head that
			// should move first. While possible that after freeing up some of the head,
			// parts of other batches may be moved in the freed-up space immediately,
			// this algorithm does not exploit that.
			// After handling case 1 or 2, not all batches may have been processed yet.
			// After handling case 1, the next batch (if any) must necessarily fall under
			// case 2. After handling case 2, the next batch(es) may fall under either
			// case.
			// An alternative option for case #2 is to employ an auxiliary buffer. We may
			// then first copy sources into the buffer, and then from the buffer into the
			// destination locations. Parallelism is limited by the size of the buffer--
			// the larger the buffer, the more threads can operate simultaneously. The
			// drawback is of course the moving of all related data twice. Since system
			// bandwidth typically saturates at just a few cores (out of the full
			// number of cores), it is non-trivial to decide when which variant is
			// better-- however, note that
			//  a) if two threads were able to achieve double throughput, then this would
			//     offset the cost of moving data twice and break even versus a
			//     sequential variant; while
			//  b) typically, single-core memory throughput is far less than half of full
			//     system band-width.
			// Combining these two observations, the below code for case 2 uses the
			// variant that results in the highest amount of parallelism, while on a tie,
			// favouring the in-place variant.

			// first handle trivial case
			if( batches == 0 ) { return; }

			// prelims
			const IND upper = src_offsets[ batches ];
			size_t batch = batches - 1;
			size_t not_processed = batches;

			do {
				if( dst_offsets[ batch ] >= upper ) {
					// we are in case 1
					size_t nbatches;
					// first check that movement is non-trivial
					if( src_offsets[ batch ] < dst_offsets[ batch ] ) {
						// check how many batches we can process
						while(
							batch > 0 && dst_offsets[ batch - 1 ] >= upper &&
							src_offsets[ batch ] < dst_offsets[ batch ]
						) { (void) --batch; }
						nbatches = not_processed - batch;
						std::cout << "\t\t will process " << nbatches << " batches in case 1\n"; // DBG
						// process them
						internal::unordered_memmove_ompPar_case1(
							source,
							src_offsets + batch,
							dst_offsets + batch,
							nbatches
						);
					} else {
						std::cout << "\t\t trivial batch detected I -- skipping\n"; // DBG
						assert( src_offsets[ batch ] == dst_offsets[ batch ] );
						nbatches = 1;
					}

					// progress to next batch (which will be case 2, trivial, or end)
					not_processed -= nbatches;
					(void) --batch;
				} else {
					assert( dst_offsets[ batch ] < src_offsets[ batch + 1 ] );
					// first check if movement is non-trivial
					if( src_offsets[ batch ] < dst_offsets[ batch ] ) {
						// we are in case 2
						assert( dst_offsets[ batch ] < src_offsets[ batch + 1 ] );
						(void) workspace;
						(void) workspace_size;
						/*const size_t head_move_parallelism =
							internal::unordered_memmove_case2_inplace_parallelism(
								src_offsets + batch, dst_offsets + batch );
						const size_t buffered_parallelism =
							internal::unordered_memmove_case2_buffered_cost(
								source, src_offsets, dst_offsets,
								not_processed, workspace_size
							);
						if( buffered_parallelism > head_move_parallelism ) {
							batch = internal::unordered_memmove_case2_buffered(
								source, src_offsets, dst_offsets,
								not_processed, workspace, workspace_size
							);
						} else {TODO */
							std::cout << "\t\t will handle case-2 batch in-place\n"; // DBG
							internal::unordered_memmove_ompPar_case2_inplace(
								source, src_offsets + batch, dst_offsets + batch );
						// } TODO
					} else {
						std::cout << "\t\t trivial batch detected II -- skipping\n"; // DBG
						assert( src_offsets[ batch ] == dst_offsets[ batch ] );
					}
					(void) --batch;
					(void) --not_processed;
				}
			} while( not_processed > 0 );

			// done
		}

		/**
		 * Rearranges an array of elements by copying batches of elements from one
		 * source location to a given destination location.
		 *
		 * See #unordered_memmove_seq for full documentation.
		 *
		 * This variant implements the OpenMP variant that is designed to be called
		 * from a sequential context, thus spawning its own OpenMP parallel region;
		 * this is indicated by the postfix <tt>_omp</tt>.
		 *
		 * Other variants are the sequential unordered memmove-- indicated by the
		 * postfix <tt>_seq</tt> and an OpenMP parallel variant that is to be called
		 * from within a pre-existing OpenMP parallel region and is indicated by the
		 * postfix <tt>_par</tt>.
		 *
		 * @param[in,out] source  Pointer to the array that contains the batches to be
		 *                        moved around within the array.
		 * @param[in] src_offsets The start offset in \a source that indicates the
		 *                        start position of each batch in the array.
		 * @param[in] dst_offsets The start offset in \a source that indicates where
		 *                        each batch should move to.
		 * @param[in] batches     How many batches in \a source should be moved.
		 *
		 * Optionally, a workspace may be provided. The workspace may or may not be
		 * used to speed up the requested operation.
		 *
		 * @param[in,out] workspace      A pointer to a unique memory region that may
		 *                               hold up to \a workspace_size elements.
		 * @param[in]     workspace_size The size (in number of elements) of the
		 *                               \a workspace.
		 */
		template< typename T, typename IND >
		void unordered_memmove_par(
			T * const source,
			const IND * const src_offsets,
			const IND * const dst_offsets,
			const size_t batches,
			T * const workspace = nullptr,
			const size_t workspace_size = 0
		) {
			// use a simple performance model to limit the number of threads in the
			// parallel region if the workload is especially small
			const size_t n = src_offsets[ batches ];

			// if too small, do not spawn any parallel region
			if( n < config::OMP::minLoopSize() ) {
				unordered_memmove_seq( source, src_offsets, dst_offsets, batches,
					workspace, workspace_size );
				return;
			}

			// the basic analytic model
			const size_t nthreads = config::OMP::nranges( 0, n );

			// spawn the parallel region
			#pragma omp parallel num_threads( nthreads )
			{
				// call the parallel implementation
				unordered_memmove_ompPar( source, src_offsets, dst_offsets, batches,
					workspace, workspace_size );
			}
		}

#endif // end _GRB_WITH_OMP

	} // end namespace grb::utils

} // end namespace grb

#endif

