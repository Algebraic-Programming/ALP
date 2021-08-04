
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
 * @date 27th of January, 2017
 */

#ifndef _H_GRB_BSP1D_DISTRIBUTION
#define _H_GRB_BSP1D_DISTRIBUTION

#include <graphblas/base/config.hpp>
#include <graphblas/distribution.hpp>

namespace grb {

	namespace internal {

		/**
		 * This class defines the distribution for the BSP1D implementation of the
		 * GraphBLAS.
		 *
		 * Let \a b be the blocksize of the distribution. This value by default is
		 * set to config::CACHE_LINE_SIZE::value() because this ensures, if all data
		 * is perfectly aligned, that a vector of type \a char will operate on units
		 * of single cache lines.
		 *
		 * This implementation uses a one-dimensional block-cyclic distribution with
		 * block size \a b.
		 *
		 * For a vector of size \a n, this means that the vector is split into
		 * \f$ \lceil n / b \rceil \f$ blocks. These blocks are distributed cyclically
		 * over all \a P processes. The last block will have \f$ n\text{ mod }b \f$
		 * elements, instead of the full \a b elements.
		 *
		 * For an \a m times \a n matrix, this means that the matrix is split row-wise
		 * into \f$ \lceil m / b \rceil \f$ blocks. These blocks are distributed
		 * cyclically over all \a P processes. The last block will have
		 * \f$ m\text{ mod }b \f$ rows, instead of the full \a b rows. Each of the
		 * local rows stored at each of the \a P processes will store all elements that
		 * appear on that row in the original (global) input matrix.
		 *
		 * During sparse matrix--vector multiplication (or vector--matrix
		 * multiplication), the input vector has to be available as a whole on each of
		 * the \a P processes. Thus the cost of level-2 operations typically incurs an
		 * additional cost of an allgather of total size \a n, where \a n is the size
		 * of the input vector.
		 *
		 * For large \a P, this behaviour will not scale. For small \a P, however, this
		 * implementation is perfectly acceptable. The fastest possible implementation
		 * requires pre-processing by explicit matrix partitioning.
		 */
		template<>
		class Distribution< BSP1D > {

		public:
			/** @return The blocksize of this distribution. */
			static constexpr size_t blocksize() {
				return config::CACHE_LINE_SIZE::value();
			}

			/**
			 * For a given global index, to which process an element or row with that
			 * index should be distributed to.
			 *
			 * @param[in] global The global index of the parameter.
			 * @param[in]   n    The global size of the vector or matrix dimension.
			 *                   Must be larger than \a global.
			 * @param[in]   P    The total number of user processes.
			 *
			 * \note In this BSP1D distribution, \a n does not have influence on the
			 *       result of a call to this function.
			 *
			 * @returns A process ID between 0 (inclusive) and \a P (exclusive) that
			 *          signifies where to store this vector element or matrix row
			 *          with the given \a global index.
			 *
			 * This function completes in \f$ \Theta(1) \f$ time.
			 */
			static inline size_t global_index_to_process_id( const size_t global, const size_t n, const size_t P ) {
				(void)n;
				return ( global / blocksize() ) % P;
			}

			/**
			 * For a given global index, to which local index at the process \a s this
			 * element or row is stored.
			 *
			 * Here, \a s is given by #global_index_to_process_id( global, P ).
			 *
			 * @param[in] global The global index of the parameter.
			 * @param[in]   n    The global size of the vector or matrix dimension.
			 *                   Must be larger than \a global.
			 * @param[in]   P    The total number of user processes. Must be larger
			 *                   than \a s.
			 *
			 * \note In this BSP1D distribution, \a n does not have influence on the
			 *       result of a call to this function.
			 *
			 * @returns A process ID between 0 (inclusive) and \a P (exclusive) that
			 *          signifies what local index this vector element or matrix row
			 *          should be stored as.
			 *
			 * This function completes in \f$ \Theta(1) \f$ time.
			 */
			static inline size_t global_index_to_local( const size_t global, const size_t n, const size_t P ) {
				(void)n; // this distribution need not consider the global length
				return ( ( global / blocksize() ) / P ) * blocksize() + ( global % blocksize() );
			}

			/**
			 * For a given local index at a given process, calculate the corresponding
			 * global index.
			 *
			 * @param[in] local The local index of the vector or matrix row/column
			 *                  coordinate.
			 * @param[in]   n   The total length of the given vector, or the total
			 *                  number of matrix rows or columns.
			 * @param[in]   s   This process ID.
			 * @param[in]   P   The global number of user processes tied up with this
			 *                  GraphBLAS run.
			 *
			 * @return The global index of the given local \a index.
			 */
			static inline size_t local_index_to_global( const size_t local, const size_t n, const size_t s, const size_t P ) {
				(void)n; // this distribution need not consider the global length
				const size_t my_block = ( local / blocksize() ) * P + s;
				const size_t offset = local % blocksize();
				return ( my_block * blocksize() ) + offset;
			}

			/**
			 * For a given global length, how many elements or rows shall be stored at
			 * the given process \a s.
			 *
			 * @param[in] global The global size of the vector or of the matrix
			 *                   dimension.
			 * @param[in]   s    Request the local length at this process.
			 * @param[in]   P    The global number of active user processes. Must be
			 *                   larger than \a s.
			 *
			 * @returns The number of vector elements or matrix rows to store at the
			 *          given process \a s, given the \a global size of the vector or
			 *          matrix.
			 *
			 * This function completes in \f$ \Theta(1) \f$ time.
			 */
			static inline size_t global_length_to_local( const size_t global, const size_t s, const size_t P ) {
				constexpr size_t b = blocksize();                 // the number of elements in a single block
				size_t ret = ( global / b ) / P;                  // this is the number of blocks distributed to each process, rounded down
				ret *= b;                                         // translates back to the number of elements, instead of number of blocks
				const size_t block_overflow = ( global / b ) % P; // computes which processes overflow beyond ret elements
				if( block_overflow == s ) {                       // in this case, the last couple of elements flow into this process
					ret += global % b;
				} else if( block_overflow > s ) { // in this case, given that it is not equal to s-1, this process overflows
					ret += b;                     // by exactly one full block
				}
				// and otherwise this process has exactly the minimum number elemenents. In all cases, we are done:
				return ret;
			}

			/**
			 * For a given global length, how many elements or rows are stored at
			 * \em all user processes preceding a given process \a s. This function is
			 * semantically equivalent to the following implementation:
			 *
			 * \code
			 * size_t local_offset( const size_t global, const size_t s, const size_t P ) {
			 *     size_t ret = 0;
			 *     for( size_t i = 0; i < s; ++i ) {
			 *         ret += global_length_to_local( global, i, P );
			 *     }
			 *     return ret;
			 * }
			 * \endcode
			 *
			 * @param[in] global The global size of the vector or of the matrix
			 *                   dimension.
			 * @param[in]   s    Request the local length at this process.
			 * @param[in]   P    The global number of active user processes. Must be
			 *                   larger than \a s.
			 *
			 * @returns The number of vector elements or matrix rows stored at all
			 *          processes with ID less than the given \a s.
			 *
			 * This function completes in \f$ \Theta(1) \f$ time.
			 */
			static inline size_t local_offset( const size_t global, const size_t s, const size_t P ) {
				constexpr size_t b = blocksize(); // the number of elements in a single block
				size_t ret = ( global / b ) / P;  // the number of blocks distributed to each process,
				// rounded down
				ret *= b; // lower bound on the number of elements distributed
				// to each process
				const size_t block_overflow = ( global / b ) % P; // computes which processes overflow beyond
				                                                  // the minimum size
				if( s <= block_overflow ) {                       // in this case, the preceding processes to local_pid
					ret += b;                                     // all preceding processes have one additional full block
					ret *= s;                                     // multiply by the number of preceding processes
				} else {
					const size_t minsize = ret;                  // cache the minimum size
					ret += block_overflow * ( ret + b );         // add all preceding full blocks
					ret += global % b;                           // add last not-so-full block
					ret += ( s - block_overflow - 1 ) * minsize; // add all preceding minimum blocks
				}
				// done
				return ret;
			}

			/**
			 * Which process a given offset occupy-- i.e., the inverse of #local_offset.
			 *
			 * This function is semantically equivalent to the following implementation:
			 *
			 * \code
			 * size_t offset_to_pid(
			 *     const size_t offset,
			 *     const size_t size,
			 *     const size_t P
			 * ) {
			 *     size_t cur_pid = 0;
			 *     size_t cur_offset = local_offset( size, ret, P );
			 *     while( cur_pid < P && offset >= cur_offset ) {
			 *         (void) ++cur_pid;
			 *         cur_offset = local_offset( size, ret, P );
			 *     }
			 *     return cur_pid;
			 * }
			 * \endcode
			 *
			 * However, like #local_offset, the function completes in Theta(1) time.
			 *
			 * @param[in] offset The offset to translate to a PID.
			 * @param[in]  size  The total (global) length of the array.
			 * @param[in]   P    The global number of active user processes.
			 *
			 * @returns The largest s that, when passed to #local_offset, returns a
			 *          value smaller or equal to the given \a offset.
			 * @returns If no such value exists, \a P will be returned.
			 */
			static inline size_t offset_to_pid( const size_t offset, const size_t size, const size_t P ) {
				constexpr size_t b = blocksize(); // the number of elements in a single block
				const size_t nonFullBlockSize = size % b;
				const size_t minFullBlockSize = ( ( size / b ) / P ) * b;
				const size_t maxFullBlockSize = minFullBlockSize + b;

				// determine where nonFullBlockSize goes
				const size_t nonFullBlockSizePID = ( size / b ) % P;

				// we are left of nonFullBlockSizePID
				const size_t ret1 = offset / maxFullBlockSize;
				if( ret1 < nonFullBlockSizePID ) {
					return ret1;
				}

				// we are at nonFullBlockSizePID
				assert( offset >= nonFullBlockSizePID * maxFullBlockSize );
				size_t running_offset = offset - nonFullBlockSizePID * maxFullBlockSize;
				if( running_offset < minFullBlockSize + nonFullBlockSize || minFullBlockSize == 0 ) {
					return nonFullBlockSizePID;
				}

				// we are right of nonFullBlockSizePID
				assert( running_offset >= minFullBlockSize + nonFullBlockSize );
				running_offset -= minFullBlockSize + nonFullBlockSize;
				return nonFullBlockSizePID + running_offset / minFullBlockSize + 1;
			}
		};

	} // namespace internal
} // namespace grb

#endif // end `_H_GRB_BSP1D_DISTRIBUTION'
