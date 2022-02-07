
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
 * @date 16th of February, 2017
 */

#ifndef _H_GRB_BSP1D_IO
#define _H_GRB_BSP1D_IO

#include <memory>

#include "graphblas/blas1.hpp"                 //for grb::size
#include "graphblas/utils/NonzeroIterator.hpp" //for transforming an std::vector::iterator into a GraphBLAS-compatible iterator
#include "graphblas/utils/pattern.hpp"         //for handling pattern input
#include <graphblas/base/io.hpp>

#include "lpf/core.h"
#include "matrix.hpp" //for BSP1D matrix
#include "vector.hpp" //for BSP1D vector

#define NO_CAST_ASSERT( x, y, z )                                                  \
	static_assert( x,                                                              \
		"\n\n"                                                                     \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n"                                                 \
		"*     ERROR      | " y " " z ".\n"                                        \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n"                                                 \
		"* Possible fix 1 | Remove no_casting from the template parameters in "    \
		"this call to " y ".\n"                                                    \
		"* Possible fix 2 | Provide a value input iterator with element types "    \
		"that match the output vector element type.\n"                             \
		"* Possible fix 3 | If applicable, provide an index input iterator with "  \
		"element types that are integral.\n"                                       \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n" );

namespace grb {

	/**
	 * Implementation details:
	 *
	 * All user processes read in all input data but record only the data which
	 * are to be stored locally.
	 *
	 * No communication will be incurred. The cost of this function, however, is
	 *   \f$ \Theta( n ) \f$,
	 * where \a n is the global vector size.
	 *
	 * \warning If the number of user processes is larger than one, a parallel
	 *          \a IOMode is not supported.
	 *
	 * \note If the number of user processes is equal to one, a parallel \a IOMode
	 *       is equivalent to a sequential one.
	 *
	 * \warning Thus, this performance of this function does \em not scale.
	 *
	 * @see grb::buildVector for the user-level specification.
	 */
	template< Descriptor descr = descriptors::no_operation, typename InputType, typename fwd_iterator, typename Coords, class Dup = operators::right_assign< InputType > >
	RC buildVector( Vector< InputType, BSP1D, Coords > & x, fwd_iterator start, const fwd_iterator end, const IOMode mode, const Dup & dup = Dup() ) {
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, typename std::iterator_traits< fwd_iterator >::value_type >::value ),
			"grb::buildVector (BSP1D implementation)",
			"Input iterator does not match output vector type while no_casting "
			"descriptor was set" );

		// prepare
		RC ret = SUCCESS;
		const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();
		// differentiate trivial case from general case
		if( data.P == 1 ) {
			ret = buildVector< descr >( x._local, start, end, SEQUENTIAL, dup );
		} else {
			// parallel mode input is disallowed in the dense constructor.
			if( mode == PARALLEL ) {
				return ILLEGAL;
			} else {
				// sanity check
				assert( mode == SEQUENTIAL );
				// cache only elements going to this processor
				std::vector< InputType > cache;
				size_t i = 0;
				for( ; ret == SUCCESS && start != end; ++start, ++i ) {
					// sanity check
					if( i >= x._n ) {
						ret = MISMATCH;
					} else {
						// if this element is distributed to me
						if( internal::Distribution< BSP1D >::global_index_to_process_id( i, x._n, data.P ) == data.s ) {
							// cache it locally
							cache.push_back( *start );
						}
					}
				}

				// defer to local constructor
				if( ret == SUCCESS ) {
					ret = buildVector< descr >( x._local, cache.begin(), cache.end(), SEQUENTIAL, dup );
				}
			}
		}

		// check for illegal at sibling processes
		if( data.P > 1 && ( descr & descriptors::no_duplicates ) ) {
#ifdef _DEBUG
			std::cout << "\t global exit-check\n";
#endif
			if( collectives< BSP1D >::allreduce( ret, grb::operators::any_or< grb::RC >() ) != SUCCESS ) {
				return PANIC;
			}
		}

		// update nnz count
		if( ret == SUCCESS ) {
			x._nnz_is_dirty = true;
			ret = x.updateNnz();
		}

		// done
		return ret;
	}

	/**
	 * Implementation details:
	 *
	 * In sequential mode, the input from the iterators is filtered and cached in
	 * memory. Afterwards, the buildVector of the reference implementation is
	 * called.
	 *
	 * In parallel mode, the input iterators corresponding to indices that are to
	 * be stored locally, are directly read into local memory. Remote elements are
	 * sent to the process who owns the nonzero via bulk-synchronous message
	 * passing. After the iterators have been exhausted, the incoming message
	 * buffers are drained into the storage memory.
	 */
	template< Descriptor descr = descriptors::no_operation, typename InputType, typename fwd_iterator1, typename fwd_iterator2, typename Coords, class Dup = operators::right_assign< InputType > >
	RC buildVector( Vector< InputType, BSP1D, Coords > & x,
		fwd_iterator1 ind_start,
		const fwd_iterator1 ind_end,
		fwd_iterator2 val_start,
		const fwd_iterator2 val_end,
		const IOMode mode,
		const Dup & dup = Dup() ) {
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, decltype( *std::declval< fwd_iterator2 >() ) >::value ||
							std::is_integral< decltype( *std::declval< fwd_iterator1 >() ) >::value ),
			"grb::buildVector (BSP1D implementation)",
			"Input iterator does not match output vector type while no_casting "
			"descriptor was set" );

		// get access to user process data on s and P
		const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();

		// sequential case first. This one is easier as it simply discards input iterator elements
		// whenever they are not local
		if( mode == SEQUENTIAL ) {
#ifdef _DEBUG
			std::cout << "buildVector< BSP1D > called, index + value "
						 "iterators, SEQUENTIAL mode\n";
#endif

			// sequential mode is not performant anyway, so let us just rely on the reference
			// implementation of the buildVector routine for Vector< InputType, reference >.
			std::vector< InputType > value_cache;
			std::vector< typename std::iterator_traits< fwd_iterator1 >::value_type > index_cache;
			const size_t n = grb::size( x );

			// loop over all input
			for( ; ind_start != ind_end && val_start != val_end; ++ind_start, ++val_start ) {

				// sanity check on input
				if( *ind_start >= n ) {
#ifdef _DEBUG
					std::cout << "\t mismatch detected, returning\n";
#endif
					return MISMATCH;
				}

				// check if this element is distributed to me
				if( internal::Distribution< BSP1D >::global_index_to_process_id( *ind_start, n, data.P ) == data.s ) {
					// yes, so cache
					const size_t localIndex = internal::Distribution< BSP1D >::global_index_to_local( *ind_start, n, data.P );
					index_cache.push_back( localIndex );
					value_cache.push_back( static_cast< InputType >( *val_start ) );
#ifdef _DEBUG
					std::cout << "\t local nonzero will be added to " << localIndex << ", value " << ( *val_start ) << "\n";
#endif
				} else {
#ifdef _DEBUG
					std::cout << "\t remote nonzero at " << ( *ind_start ) << " will be skipped.\n";
#endif
				}
			}

			// do delegate
			auto ind_it = index_cache.cbegin();
			auto val_it = value_cache.cbegin();
			RC rc = buildVector< descr >( internal::getLocal( x ), ind_it, index_cache.cend(), val_it, value_cache.cend(), SEQUENTIAL, dup );

			if( data.P > 1 && ( descr & descriptors::no_duplicates ) ) {
#ifdef _DEBUG
				std::cout << "\t global exit check (2)\n";
#endif
				if( collectives< BSP1D >::allreduce( rc, grb::operators::any_or< grb::RC >() ) != SUCCESS ) {
					return PANIC;
				}
			}

			if( rc == SUCCESS ) {
				x._nnz_is_dirty = true;
				return x.updateNnz();
			} else {
				return rc;
			}
		}

		// now handle parallel IOMode
		assert( mode == PARALLEL );

		return PANIC;
	}

	/**
	 * TODO describe implementation details
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename fwd_iterator
	>
	RC buildMatrixUnique(
		Matrix< InputType, BSP1D > &A,
		fwd_iterator start, const fwd_iterator end,
		const IOMode mode
	) {
		typedef typename fwd_iterator::row_coordinate_type IType;
		typedef typename fwd_iterator::column_coordinate_type JType;
		typedef typename fwd_iterator::nonzero_value_type VType;

		// static checks
		NO_CAST_ASSERT( !( descr & descriptors::no_casting ) || (
				std::is_same< InputType, VType >::value &&
				std::is_integral< IType >::value &&
				std::is_integral< JType >::value
			), "grb::buildMatrixUnique (BSP1D implementation)",
			"Input iterator does not match output vector type while no_casting "
			"descriptor was set" );

		// get access to user process data on s and P
		internal::BSP1D_Data & data = internal::grb_BSP1D.load();

		// delegate for sequential case
		if( data.P == 1 ) {
			return buildMatrixUnique< descr >( A._local, start, end, SEQUENTIAL );
		}

		// function semantics require the matrix be cleared first
		RC ret = clear( A );

#ifdef _DEBUG
		std::cout << "buildMatrixUnique is called from process " << data.s << " "
			<< "out of " << data.P << " processes total.\n";
#endif

		// local cache, used to delegate to reference buildMatrixUnique
		std::vector< typename fwd_iterator::value_type > cache;
		// caches non-local nonzeroes (in case of Parallel IO)
		std::vector< std::vector< typename fwd_iterator::value_type > > outgoing;
		if( mode == PARALLEL ) {
			outgoing.resize( data.P );
		}
		// NOTE: this copies a lot of the above methodology

#ifdef _DEBUG
		const size_t my_offset =
			internal::Distribution< BSP1D >::local_offset( A._n, data.s, data.P );
		std::cout << "Local column-wise offset at PID " << data.s << " is "
			<< my_offset << "\n";
#endif
		// loop over all input
		for( size_t k = 0; ret == SUCCESS && start != end; ++k, ++start ) {

			// sanity check on input
			if( start.i() >= A._m ) {
				return MISMATCH;
			}
			if( start.j() >= A._n ) {
				return MISMATCH;
			}

			// compute process-local indices (even if remote, for code readability)
			const size_t global_row_index = start.i();
			const size_t row_pid =
				internal::Distribution< BSP1D >::global_index_to_process_id(
					global_row_index, A._m, data.P
				);
			const size_t row_local_index =
				internal::Distribution< BSP1D >::global_index_to_local(
					global_row_index, A._m, data.P
				);
			const size_t global_col_index = start.j();
			const size_t column_pid =
				internal::Distribution< BSP1D >::global_index_to_process_id(
					global_col_index, A._n, data.P
				);
			const size_t column_local_index =
				internal::Distribution< BSP1D >::global_index_to_local(
					global_col_index, A._n, data.P
				);
			const size_t column_offset =
				internal::Distribution< BSP1D >::local_offset(
					A._n, column_pid, data.P
				);

			// check if local
			if( row_pid == data.s ) {
				// push into cache
				cache.push_back( *start );
				// translate nonzero
				utils::updateNonzeroCoordinates(
					cache[ cache.size() - 1 ],
					row_local_index,
					column_offset + column_local_index
				);
#ifdef _DEBUG
				std::cout << "Translating nonzero at ( " << start.i() << ", " << start.j()
					<< " ) to one at ( " << row_local_index << ", "
					<< ( column_offset + column_local_index ) << " ) at PID "
					<< row_pid << "\n";
#endif
			} else if( mode == PARALLEL ) {
#ifdef _DEBUG
				std::cout << "Sending nonzero at ( " << start.i() << ", " << start.j()
					<< " ) to PID " << row_pid << " at ( " << row_local_index
					<< ", " << ( column_offset + column_local_index ) << " )\n";
#endif
				// send original nonzero to remote owner
				outgoing[ row_pid ].push_back( *start );
				// translate nonzero here instead of at
				// destination for brevity / code readibility
				utils::updateNonzeroCoordinates(
					outgoing[ row_pid ][ outgoing[ row_pid ].size() - 1 ],
					row_local_index, column_offset + column_local_index
				);
			} else {
#ifdef _DEBUG
				std::cout << "PID " << data.s << " ignores nonzero at ( "
					<< start.i() << ", " << start.j() << " )\n";
#endif
			}
		}

		// report on memory usage
		(void)config::MEMORY::report( "grb::buildMatrixUnique",
			"has local cache of size",
			cache.size() * sizeof( typename fwd_iterator::value_type )
		);

		if( mode == PARALLEL ) {
			// declare memory slots
			lpf_memslot_t cache_slot = LPF_INVALID_MEMSLOT;
			std::vector< lpf_memslot_t > out_slot( data.P, LPF_INVALID_MEMSLOT );

			// make sure we have enough space available to all-to-all #outgoing messages,
			// and enough space to do a prefix sum on those
			if( ret == SUCCESS ) {
				ret = data.checkBufferSize( 3 * data.P * sizeof( size_t ) );
			}

			// get handle directly into the BSP buffer, interpreted as size_t *
			size_t * const buffer_sizet = data.template getBuffer< size_t >();

			// make sure we support allgather/all-to-all patterns
			if( ret == SUCCESS ) {
				ret = data.ensureMaxMessages( 2 * data.P - 2 );
			}

			// make sure we have enough memslots available
			// cache_slot plus P-1 out_slots
			if( ret == SUCCESS ) {
				ret = data.ensureMemslotAvailable( data.P );
			}

			// send remote contribution counts
			size_t outgoing_bytes = 0;
			for( size_t k = 0; ret == SUCCESS && k < data.P; ++k ) {
				if( k == data.s ) {
					// copy process-local data directly into destination area
					buffer_sizet[ data.P + k ] = 0;
					// sanity check
					assert( outgoing[ k ].size() == 0 );
					// done
					continue;
				}
				// cache size into buffer
				buffer_sizet[ k ] = outgoing[ k ].size();
				outgoing_bytes += outgoing[ k ].size() *
					sizeof( typename fwd_iterator::value_type );
#ifdef _DEBUG
				std::cout << "Process " << data.s << ", which has " << cache.size()
					<< " local nonzeroes, sends " << buffer_sizet[ k ]
					<< " nonzeroes to process " << k << "\n";
				std::cout << data.s << ": lpf_put( ctx, " << data.slot << ", "
					<< ( k * sizeof( size_t ) ) << ", " << k << ", " << data.slot << ", "
					<< ( data.P * sizeof( size_t ) + data.s * sizeof( size_t ) ) << ", "
					<< sizeof( size_t ) << ", LPF_MSG_DEFAULT );\n";
#endif
				// request RDMA
				const lpf_err_t brc = lpf_put( data.context,
					data.slot, k * sizeof( size_t ),
					k, data.slot, data.P * sizeof( size_t ) + data.s * sizeof( size_t ),
					sizeof( size_t ), LPF_MSG_DEFAULT
				);
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}
			(void)config::MEMORY::report( "grb::buildMatrixUnique (PARALLEL mode)",
				"has an outgoing cache of size", outgoing_bytes
			);
			// wait for RDMA to finish
			if( ret == SUCCESS ) {
				const lpf_err_t brc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}
			// do local prefix
			buffer_sizet[ 0 ] = cache.size();
			for( size_t k = 1; ret == SUCCESS && k < data.P; ++k ) {
				// no need to skip k == data.s as we set buffer_sizet[ data.P + data.s ] to 0
				buffer_sizet[ k ] = buffer_sizet[ k - 1 ] + buffer_sizet[ data.P + k - 1 ];
			}
			// self-prefix is not used, update to reflect total number of local elements
			if( data.s + 1 < data.P ) {                                                  // if data.s == data.P - 1 then the current number is already correct
				buffer_sizet[ data.s ] =
					buffer_sizet[ data.P - 1 ] + buffer_sizet[ 2 * data.P - 1 ]; // otherwise overwrite with correct number
			}
			// communicate prefix
			for( size_t k = 0; ret == SUCCESS && k < data.P; ++k ) {
				// but we do need to skip here or else we violate our max messages contract
				if( k == data.s ) {
					continue;
				}
#ifdef _DEBUG
				std::cout << "Process " << data.s << ", which has " << cache.size()
					<< " local nonzeroes, sends offset " << buffer_sizet[ k ]
					<< " to process " << k << "\n";
				std::cout << data.s << ": lpf_put( ctx, " << data.slot << ", "
					<< ( k * sizeof( size_t ) ) << ", " << k << ", " << data.slot << ", "
					<< ( 2 * data.P * sizeof( size_t ) + data.s * sizeof( size_t ) )
					<< ", " << sizeof( size_t ) << ", LPF_MSG_DEFAULT );\n";
#endif
				// send remote offsets
				const lpf_err_t brc = lpf_put( data.context,
					data.slot, k * sizeof( size_t ),
					k, data.slot, 2 * data.P * sizeof( size_t ) + data.s * sizeof( size_t ),
					sizeof( size_t ), LPF_MSG_DEFAULT
				);
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}

			// register nonzero memory areas for all-to-all: global
			if( ret == SUCCESS ) {
				// enure local cache is large enough
				(void)config::MEMORY::report( "grb::buildMatrixUnique (PARALLEL mode)",
					"will increase local cache to size",
					buffer_sizet[ data.s ] * sizeof( typename fwd_iterator::value_type ) );
				cache.resize( buffer_sizet[ data.s ] ); // see self-prefix comment above
				// register memory slots for all-to-all
				const lpf_err_t brc = cache.size() > 0 ?
					lpf_register_global(
						data.context,
						&( cache[ 0 ] ), cache.size() *
							sizeof( typename fwd_iterator::value_type ),
						&cache_slot
					) :
                                                lpf_register_global(
							data.context,
							nullptr, 0,
							&cache_slot
						);
#ifdef _DEBUG
				std::cout << data.s << ": address " << &( cache[ 0 ] ) << " (size "
					<< cache.size() * sizeof( typename fwd_iterator::value_type )
					<< ") binds to slot " << cache_slot << "\n";
#endif
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}
			// register nonzero memory areas for all-to-all: local
			for( size_t k = 0; ret == SUCCESS && k < data.P; ++k ) {
				if( k == data.s ) {
					continue;
				}
				assert( out_slot.size() == data.P );
				const lpf_err_t brc = outgoing[ k ].size() > 0 ?
					lpf_register_local( data.context,
						&(outgoing[ k ][ 0 ]),
						outgoing[ k ].size() *
							sizeof( typename fwd_iterator::value_type ),
						&(out_slot[ k ])
					) :
					lpf_register_local( data.context,
						nullptr, 0,
						&(out_slot[ k ])
					);
#ifdef _DEBUG
				std::cout << data.s << ": address " << &( outgoing[ k ][ 0 ] ) << " (size "
					<< outgoing[ k ].size() * sizeof( typename fwd_iterator::value_type )
					<< ") binds to slot " << out_slot[ k ] << "\n";
#endif
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}

			// wait for RDMA to finish
			if( ret == SUCCESS ) {
				const lpf_err_t brc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}

			// schedule all-to-all
			for( size_t k = 0; ret == SUCCESS && k < data.P; ++k ) {
				if( k == data.s ) {
					continue;
				}
#ifdef _DEBUG
				for( size_t s = 0; ret == SUCCESS && s < data.P; ++s ) {
					if( s == data.s ) {
						std::cout << data.s << ": lpf_put( ctx, "
							<< out_slot[ k ] << ", 0, " << k << ", " << cache_slot << ", "
							<< buffer_sizet[ 2 * data.P + k ] *
								sizeof( typename fwd_iterator::value_type )
							<< ", " << outgoing[ k ].size() *
								sizeof( typename fwd_iterator::value_type )
							<< ", LPF_MSG_DEFAULT );\n";
					}
					lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}
#endif
				if( outgoing[ k ].size() > 0 ) {
					const lpf_err_t brc = lpf_put( data.context,
							out_slot[ k ], 0,
							k, cache_slot, buffer_sizet[ 2 * data.P + k ] *
								sizeof( typename fwd_iterator::value_type ),
							outgoing[ k ].size() *
								sizeof( typename fwd_iterator::value_type ),
							LPF_MSG_DEFAULT
					);
					if( brc != LPF_SUCCESS ) {
						ret = PANIC;
					}
				}
			}
			// wait for RDMA to finish
			if( ret == SUCCESS ) {
				const lpf_err_t brc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}
			// clean up memslots, even on error (but still cause error when cleanup fails)
			for( size_t k = 0; k < data.P; ++k ) {
				if( out_slot[ k ] != LPF_INVALID_MEMSLOT ) {
					const lpf_err_t brc = lpf_deregister( data.context, out_slot[ k ] );
					if( brc != LPF_SUCCESS && ret == SUCCESS ) {
						ret = PANIC;
					}
				}
			}
			if( cache_slot != LPF_INVALID_MEMSLOT ) {
				const lpf_err_t brc = lpf_deregister( data.context, cache_slot );
				if( brc != LPF_SUCCESS && ret == SUCCESS ) {
					ret = PANIC;
				}
			}
			// clean up outgoing slots, which goes from 2x to 1x memory store for the
			// nonzeroes here contained
			{
				std::vector< std::vector< typename fwd_iterator::value_type > > emptyVector;
				std::swap( emptyVector, outgoing );
			}
		}

#ifdef _DEBUG
		std::cout << "Dimensions at PID " << data.s << ": "
			<< "( " << A._m << ", " << A._n << " ). "
			<< "Locally cached: " << cache.size() << "\n";
#endif

		if( ret == SUCCESS ) {
			// sanity check
			assert( nnz( A._local ) == 0 );
			// delegate and done!
			ret = buildMatrixUnique< descr >( A._local,
				utils::makeNonzeroIterator< IType, JType, VType >( cache.begin() ),
				utils::makeNonzeroIterator< IType, JType, VType >( cache.end() ),
				SEQUENTIAL
			);
			// sanity checks
			assert( ret != MISMATCH );
			assert( nnz( A._local ) == cache.size() );
		}

#ifdef _DEBUG
		std::cout << "Number of nonzeroes at the local matrix at PID " << data.s
			<< " is " << nnz( A._local ) << "\n";
#endif

		return ret;
	}

	/** \internal Simply rely on backend implementation. */
	template< typename InputType, typename Coords >
	uintptr_t getID( const Vector< InputType, BSP1D, Coords > &x ) {
		return x._id;
	}

	/** \internal Simply rely on backend implementation. */
	template< typename InputType >
	uintptr_t getID( const Matrix< InputType, BSP1D > &A ) {
		return A._id;
	}

} // namespace grb

#undef NO_CAST_ASSERT

#endif // end ``_H_GRB_BSP1D_IO''

