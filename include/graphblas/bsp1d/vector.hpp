
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
 * @date 9th of January 2016
 */

#ifndef _H_GRB_BSP1D_VECTOR
#define _H_GRB_BSP1D_VECTOR

#include <map>
#include <sstream>
#include <vector>

#include <lpf/core.h>

#include <graphblas/backends.hpp>
#include <graphblas/base/pinnedvector.hpp>
#include <graphblas/collectives.hpp>
#include <graphblas/config.hpp>
#include <graphblas/reference/blas1-raw.hpp>
#include <graphblas/reference/coordinates.hpp>
#include <graphblas/reference/vector.hpp>
#include <graphblas/type_traits.hpp>
#include <graphblas/utils/alloc.hpp>
#include <graphblas/utils/autodeleter.hpp>

#include "config.hpp"
#include "distribution.hpp"
#include "init.hpp"

#ifdef _DEBUG
#include "spmd.hpp"
#endif

namespace grb {

	// forward declaration for vector's friends
	template< typename D >
	class Matrix< D, BSP1D >;

	namespace internal {
		template< typename DataType, typename Coords >
		RC synchronizeVector( const Vector< DataType, BSP1D, Coords > & x ) {
			return x.synchronize();
		}

		template< typename DataType, typename Coords >
		void setDense( Vector< DataType, BSP1D, Coords > & x );

		template< Descriptor descr,
			bool output_masked,
			bool input_masked,
			bool left_handed,
			class Ring,
			typename IOType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename InputType4,
			typename Coords >
		RC bsp1d_mxv( Vector< IOType, BSP1D, Coords > & u,
			const Vector< InputType3, BSP1D, Coords > & u_mask,
			const Matrix< InputType2, BSP1D > & A,
			const Vector< InputType1, BSP1D, Coords > & v,
			const Vector< InputType4, BSP1D, Coords > & v_mask,
			const Ring & ring );

		template< Descriptor descr, bool, bool, bool, class Ring, typename IOType, typename InputType1, typename InputType2, typename InputType3, typename InputType4, typename Coords >
		RC bsp1d_vxm( Vector< IOType, BSP1D, Coords > & u,
			const Vector< InputType3, BSP1D, Coords > & u_mask,
			const Vector< InputType1, BSP1D, Coords > & v,
			const Vector< InputType4, BSP1D, Coords > & v_mask,
			const Matrix< InputType2, BSP1D > & A,
			const Ring & ring = Ring() );

		/**
		 * Retrieves the process-local part of a distributed vector.
		 *
		 * @tparam DataType The type of a vector element.
		 *
		 * @param[in] x The distributed BSP1D vector.
		 *
		 * @returns A vector of type \a _GRB_BSP!D_BACKEND, usually grb::reference or
		 *          grb::reference_omp.
		 *
		 * Contents of the returned vector may be modified.
		 */
		template< typename DataType, typename Coords >
		Vector< DataType, _GRB_BSP1D_BACKEND, Coordinates< _GRB_BSP1D_BACKEND > > & getLocal( Vector< DataType, BSP1D, Coords > & x );

		/**
		 * Retrieves the process-local part of a distributed vector.
		 *
		 * @tparam DataType The type of a vector element.
		 *
		 * @param[in] x The distributed BSP1D vector, inmutable variant.
		 *
		 * @returns A vector of type \a _GRB_BSP1D_BACKEND, usually grb::reference or
		 *          grb::reference_omp.
		 *
		 * Contents of the returned vector may not be modified.
		 */
		template< typename DataType, typename Coords >
		const Vector< DataType, _GRB_BSP1D_BACKEND, Coordinates< _GRB_BSP1D_BACKEND > > & getLocal( const Vector< DataType, BSP1D, Coords > & x );

		/**
		 * Retrieves the global mirror of a distributed vector.
		 *
		 * @tparam DataType The data type of each vector element.
		 *
		 * @param[in] x The distributed BSP1D vector.
		 *
		 * @returns A vector of type \a _GRB_BSP1D_BACKEND.
		 */
		template< typename DataType, typename Coords >
		Vector< DataType, _GRB_BSP1D_BACKEND, Coordinates< _GRB_BSP1D_BACKEND > > & getGlobal( Vector< DataType, BSP1D, Coords > & x );

		/**
		 * Retrieves the global mirror of a distributed vector.
		 *
		 * @tparam DataType The data type of each vector element.
		 *
		 * @param[in] x The distributed BSP1D vector, inmutable variant.
		 *
		 * @returns A vector of type \a _GRB_BSP1D_BACKEND.
		 */
		template< typename DataType, typename Coords >
		const Vector< DataType, _GRB_BSP1D_BACKEND, Coordinates< _GRB_BSP1D_BACKEND > > & getGlobal( const Vector< DataType, BSP1D, Coords > & x );

		/**
		 * Signals change in the sparsity structure of the local vector.
		 *
		 * @tparam DataType The type of each element of the vector.
		 *
		 * @param[in] x The vector which underwent a local change to its sparsity.
		 */
		template< typename DataType, typename Coords >
		void signalLocalChange( Vector< DataType, BSP1D, Coords > & x );

		/**
		 * Updates the nonzero count of a given vector.
		 *
		 * This function should be called whenever an operation has
		 * completed that has or may have updated the nonzero structure
		 * of \a x.
		 *
		 * @tparam DataType The type of values stored by \a x.
		 *
		 * @param[in,out] x The vector of which to update the nonzero
		 *                  count.
		 *
		 * @returns SUCCESS When the operation has completed as
		 *                  intended.
		 * @returns PANIC   On a failure of the underlying
		 *                  communications layer.
		 */
		template< typename DataType, typename Coords >
		RC updateNnz( Vector< DataType, BSP1D, Coords > & x );

		/** TODO documentation */
		template< typename DataType, typename Coords >
		void setDense( Vector< DataType, BSP1D, Coords > & x );
	} // namespace internal

	// BLAS1 forward declaration of friends
	template< typename DataType, typename Coords >
	size_t size( const Vector< DataType, BSP1D, Coords > & x ) noexcept;

	template< typename DataType, typename Coords >
	RC clear( Vector< DataType, BSP1D, Coords > & x ) noexcept;

	template< typename DataType, typename Coords >
	size_t nnz( const Vector< DataType, BSP1D, Coords > & x );

	/**
	 * A BSP1D vector. Uses a block-cyclic distribution.
	 */
	template< typename D, typename C >
	class Vector< D, BSP1D, C > {

		/* *********************
		        BLAS1 friends
		   ********************* */

		// template< typename DataType >
		friend RC clear< D, C >( Vector< D, BSP1D, C > & ) noexcept;

		// template< typename DataType >
		friend size_t size< D, C >( const Vector< D, BSP1D, C > & ) noexcept;

		// template< typename DataType >
		friend size_t nnz< D, C >( const Vector< D, BSP1D, C > & );

		template< Descriptor descr, typename OutputType, typename Coords, typename InputType >
		friend RC set( Vector< OutputType, BSP1D, Coords > &, const Vector< InputType, BSP1D, Coords > & );

		template< Descriptor, typename InputType, typename fwd_iterator, typename Coords, class Dup >
		friend RC buildVector( Vector< InputType, BSP1D, Coords > &, fwd_iterator, const fwd_iterator, const IOMode, const Dup & );

		template< Descriptor, typename InputType, typename fwd_iterator1, typename fwd_iterator2, typename Coords, class Dup >
		friend RC buildVector( Vector< InputType, BSP1D, Coords > &, fwd_iterator1, const fwd_iterator1, fwd_iterator2, const fwd_iterator2, const IOMode, const Dup & );

		// template< typename DataType >
		friend RC internal::updateNnz< D, C >( Vector< D, BSP1D, C > & );

		// template< typename DataType >
		friend void internal::setDense< D, C >( Vector< D, BSP1D, C > & );

		template< Descriptor, bool, bool, bool, class Ring, typename IOType, typename InputType1, typename InputType2, typename InputType3, typename InputType4, typename Coords >
		friend RC internal::bsp1d_mxv( Vector< IOType, BSP1D, Coords > &,
			const Vector< InputType3, BSP1D, Coords > &,
			const Matrix< InputType2, BSP1D > &,
			const Vector< InputType1, BSP1D, Coords > &,
			const Vector< InputType4, BSP1D, Coords > &,
			const Ring & );

		template< Descriptor descr, bool, bool, bool, class Ring, typename IOType, typename InputType1, typename InputType2, typename InputType3, typename InputType4, typename Coords >
		friend RC internal::bsp1d_vxm( Vector< IOType, BSP1D, Coords > &,
			const Vector< InputType3, BSP1D, Coords > &,
			const Vector< InputType1, BSP1D, Coords > &,
			const Vector< InputType4, BSP1D, Coords > &,
			const Matrix< InputType2, BSP1D > & A,
			const Ring & ring );

		friend RC internal::synchronizeVector< D, C >( const Vector< D, BSP1D, C > & );

		friend Vector< D, _GRB_BSP1D_BACKEND, internal::Coordinates< _GRB_BSP1D_BACKEND > > & internal::getLocal< D, C >( Vector< D, BSP1D, C > & );

		friend const Vector< D, _GRB_BSP1D_BACKEND, internal::Coordinates< _GRB_BSP1D_BACKEND > > & internal::getLocal< D, C >( const Vector< D, BSP1D, C > & );

		friend Vector< D, _GRB_BSP1D_BACKEND, internal::Coordinates< _GRB_BSP1D_BACKEND > > & internal::getGlobal< D, C >( Vector< D, BSP1D, C > & );

		friend const Vector< D, _GRB_BSP1D_BACKEND, internal::Coordinates< _GRB_BSP1D_BACKEND > > & internal::getGlobal< D, C >( const Vector< D, BSP1D, C > & );

		template< typename Func, typename DataType, typename Coords >
		friend RC eWiseLambda( const Func, const Vector< DataType, BSP1D, Coords > & );

		template< typename Func, typename DataType1, typename DataType2, typename Coords, typename... Args >
		friend RC eWiseLambda( const Func, const Vector< DataType1, BSP1D, Coords > &, const Vector< DataType2, BSP1D, Coords > &, Args const &... args );

		/* *********************
		    Level-1 collectives
		          friends
		   ********************* */

		template< Descriptor, class Ring, typename OutputType, typename InputType1, typename InputType2 >
		friend RC internal::allreduce( OutputType &,
			const Vector< InputType1, BSP1D, C > &,
			const Vector< InputType2, BSP1D, C > &,
			RC( reducer )( OutputType &, const Vector< InputType1, BSP1D, C > &, const Vector< InputType2, BSP1D, C > &, const Ring & ),
			const Ring & );

		/* ********************
		      IO collectives
		         friends
		   ******************** */

		friend class PinnedVector< D, BSP1D >;

		/* ********************
		     internal friends
		   ******************** */

		friend void internal::signalLocalChange< D, C >( Vector< D, BSP1D, C > & );

	private:

		/** The local vector type. */
		typedef Vector< D, _GRB_BSP1D_BACKEND, internal::Coordinates< _GRB_BSP1D_BACKEND > > LocalVector;

		/** The blocksize of the block-cyclic distribution of this vector. */
		static constexpr size_t _b = config::CACHE_LINE_SIZE::value();

		/** Stores a map of which global vector offset starts at which process ID.*/
		std::map< size_t, size_t > PIDmap;

		/** Raw vector of size \a _n. */
		D * _raw;

		/**
		 * Raw boolean vector of size \a _n.
		 *
		 * Note that this corresponds to the first \a _n * sizeof(bool) bytes of the
		 * sparsity memory area in grb::internal::Coordinates. We need to interpret
		 * these first bytes as a raw array of booleans, which corresponds to this
		 * pointer.
		 */
		bool * _assigned;

		/**
		 * Buffer area required by the #_local and #_global coordinates.
		 */
		char * _buffer;

		/** The actual local vector. */
		LocalVector _local;

		/**
		 * The global vector. Must call the private synchronize() function prior to
		 * using this container.
		 */
		LocalVector _global;

		/** The local size of this distributed vector. */
		size_t _local_n;

		/**
		 * Cached the local offset after which element in _local the locally owned
		 * part of the vector is stored.
		 */
		size_t _offset;

		/** The global size of this distributed vector. */
		size_t _n;

		/**
		 * The global number of nonzeroes in this distributed vector.
		 *
		 * This field is declared \a mutable since it is a cached global count. The
		 * global count can change by local operations, in which case an allreduce
		 * must occur from possibly \a const contexts.
		 */
		mutable size_t _nnz;

		/** Memory slot corresponding to the \a _raw memory area. */
		lpf_memslot_t _raw_slot;

		/**
		 * Memory slot corresponding to the array part in the \a _assigned memory
		 * area.
		 */
		lpf_memslot_t _assigned_slot;

		/** Memory slot corresponding to the stack in #_buffer. */
		lpf_memslot_t _stack_slot;

		/**
		 * Whether cleared was called without a subsequent call to
		 * #synchronize_sparsity.
		 *
		 * When a vector is cleared, the sparsity information must be reset. This
		 * may happen from a \a const context, hence this field is declared
		 * \a mutable.
		 */
		mutable bool _cleared;

		/**
		 * Whether the local vector became dense without a subsequent call to
		 * #synchronize_sparsity.
		 *
		 * When a vector became dense, the spartity information must be synced
		 * accordingly. This may happen from a \a const context, hence this field is
		 * declared \a mutable.
		 */
		mutable bool _became_dense;

		/**
		 * Whether the vector has possibly changed its global nonzero count.
		 *
		 * This field is declared \a mutable not because a non \a const context can
		 * set this flag to \a true, but because a \a const context may updated the
		 * cached value for #_nnz and thus should set this flag to \a false.
		 */
		mutable bool _nnz_is_dirty;

		/**
		 * Whether #_global has entered an invalid state.
		 */
		bool _global_is_dirty;

		/**
		 * Will automatically free \a _raw, depending on how the vector was
		 * initialized and also depending on whether the underlying data was pinned
		 * by the user.
		 */
		utils::AutoDeleter< D > _raw_deleter;

		/**
		 * Will automatically free \a _assigned, depending on how the
		 * vector was initialized and also depending on whether the underlying data
		 * was pinned by the user.
		 */
		utils::AutoDeleter< char > _assigned_deleter;

		/**
		 * Will automatically free #_buffer, depending on how the vector was
		 * initialised and on whether this vector has become pinned.
		 */
		utils::AutoDeleter< char > _buffer_deleter;

		/**
		 * Function to manually initialise this vector instance. This function is
		 * to be called by constructors only.
		 *
		 * @param[in] raw_in      The raw memory area this vector should wrap
		 *                        around. If \a NULL is passed, this function will
		 *                        allocate a new memory region to house \a cap_in
		 *                        vector elements. If \a NULL is passed, \a NULL
		 *                        must also be passed to \a assigned_in.
		 * @param[in] assigned_in The raw memory area this vector should wrap
		 *                        around. If \a NULL is passed, this function will
		 *                        allocate a new memory region to house \a cap_in
		 *                        booleans. If \a NULL is passed, \a NULL must also
		 *                        be passed to \a raw_in.
		 * @param[in] buffer_in   Any additional memory that is required. If
		 *                        <tt>NULL</tt> is passed, this function will
		 *                        allocate a new memory region.
		 * @param[in] cap_in      The \em global size of the vector.
		 *
		 * \a raw_in may never equal \a assigned_in unless they are both \a NULL;
		 * both pointers must refer to completely disjoint memory areas of size
		 * \f$ \mathit{cap\_in} \mathit{sizeof}( D ) \f$ and
		 * \f$ \mathit{cap\_in} \mathit{sizeof}( \mathit{bool} ) \f$, bytes
		 * respectively. If these constraints are not met, undefined behaviour
		 * occurs.
		 *
		 * @throws Runtime error When initialisation fails due to out-of-memory
		 *                       conditions.
		 * @throws Runtime error When the POSIX call to get an aligned memory area
		 *                       fails for any other reason.
		 * @throws Runtime error When not enough memory can be reserved for the BSP
		 *                       backend, e.g., when out of memory.
		 * @throws Runtime error When a call to lpf_register_global fails (which
		 *                       should be impossible).
		 *
		 * \note The case where \a raw_in is not equal to \a NULL is currently
		 *       unused and untested.
		 */
		void initialize( void * const raw_in, void * const assigned_in, void * const buffer_in, const size_t cap_in ) {
#ifdef _DEBUG
			std::cout << "grb::Vector< T, BSP1D, C >::initialize called\n";
#endif
			// check for undefined behaviour
#ifndef NDEBUG
			if( raw_in == NULL || assigned_in == NULL || buffer_in == NULL ) {
				// Illegal arguments: raw_in and assigned_in must both be NULL or both be non-NULL.
				assert( ! ( raw_in != NULL || assigned_in != NULL || buffer_in != NULL ) );
			}
#endif

			// if no vector was provided, create a new one
			if( raw_in == NULL ) {
				// build a descriptor string of this vector
				std::stringstream sstream;
				sstream << ", for a vector of size " << cap_in;
				// declare new assigned array as char *
				char * new_assigned = NULL;

				const size_t bufferSize = internal::Coordinates< _GRB_BSP1D_BACKEND >::bufferSize( _local_n ) + internal::Coordinates< _GRB_BSP1D_BACKEND >::bufferSize( cap_in );
				const RC rc = grb::utils::alloc( "grb::Vector< T, BSP1D, C > (initialize)",
					sstream.str(), _raw, cap_in, true,
					_raw_deleter,                                                                                            // allocate raw array
					new_assigned, internal::Coordinates< _GRB_BSP1D_BACKEND >::arraySize( cap_in ), true, _assigned_deleter, // allocate assigned array
					_buffer, bufferSize, true, _buffer_deleter );
				// identify error and throw
				if( rc == OUTOFMEM ) {
					throw std::runtime_error( "Out-of-memory during BSP1D Vector memory allocation" );
				} else if( rc != SUCCESS ) {
					throw std::runtime_error( "Unhandled runtime error during BSP1D Vector memory allocation" );
				}
				// all OK, so set and exit
				_assigned = reinterpret_cast< bool * >( new_assigned );
			} else {
				// note that this does not catch overlapping cases, nor multiply-used memory areas
				// checking for all of this is way too expensive.

				// just take the provided memory areas
				_raw = static_cast< D * >( raw_in );
				_assigned = reinterpret_cast< internal::Coordinates< _GRB_BSP1D_BACKEND >::ArrayType * >( assigned_in );
				_buffer = static_cast< char * >( buffer_in );
				// note that we do not set the AutoDeleter, the callee must handle the memory we have been given
			}

			const size_t local_buffer_offset = internal::Coordinates< _GRB_BSP1D_BACKEND >::bufferSize( cap_in );

			// delegate to sequential implementation
			_global.initialize( _raw, _assigned, false, _buffer, cap_in );
			_local.initialize( _raw + _offset, _assigned + _offset, true, _buffer + local_buffer_offset, _local_n );

			// now set remaining fieldds
			_n = cap_in;
			const size_t arraySize = internal::Coordinates< _GRB_BSP1D_BACKEND >::arraySize( cap_in );
			const size_t stackSize = internal::Coordinates< _GRB_BSP1D_BACKEND >::stackSize( cap_in );
			void * stack = NULL;
			{
				size_t tmp;
				stack = internal::getCoordinates( _global ).getRawStack( tmp );
			}

			// get thread-local store
			auto &data = internal::grb_BSP1D.load();
#ifdef _DEBUG
			std::cout << data.s
					  << ": local and global coordinates are initialised. The "
						 "array size is "
					  << arraySize << " while the stack size is " << stackSize << " (in bytes). The value array size is " << _n * sizeof( D ) << " bytes.\n";
#endif

#ifndef NDEBUG
			if( _n == 0 ) {
				assert( _raw == NULL );
			}
			if( _raw == NULL ) {
				assert( _n == 0 );
			}
#endif

			if( _n > 0 ) {
				// make sure we can cache all vector data inside the GraphBLAS buffer
				// this is actually an over-estimation
#ifdef _DEBUG
				std::cout << "Ensuring buffer capacity for vector of global size " << _n << ", local size " << _local_n << ", and P = " << data.P << ". Context is " << data.context << std::endl;
#endif
				if( data.ensureBufferSize(
						// combine preamble
						4 * data.P * sizeof( size_t ) +
						std::max(
							// stack-based combine
							2 * data.P * sizeof( size_t ) +
							// buffer for alltoallv
							( _n + 1 ) * ( 2 * sizeof( D ) + sizeof( internal::Coordinates< _GRB_BSP1D_BACKEND >::StackType ) ), // +1 is for padding
							// array-based combine
							_local_n * data.P * ( sizeof( D ) + sizeof( internal::Coordinates< _GRB_BSP1D_BACKEND >::ArrayType ) )
						) ) != SUCCESS ) {
					throw std::runtime_error( "Error during resizing of global GraphBLAS buffer" );
				}

				// make sure we can take three additional memory slots
				if( data.ensureMemslotAvailable( 3 ) != SUCCESS ) {
					throw std::runtime_error( "Error during resizing of BSP buffers" );
				}
				// get a memory slot for _raw
				lpf_err_t rc = lpf_register_global( data.context, _raw, _n * sizeof( D ), &_raw_slot );
				if( rc == LPF_SUCCESS ) {
					data.signalMemslotTaken();
				}
				if( rc == LPF_SUCCESS ) {
#ifdef _DEBUG
					std::cout << data.s << ": pointer at " << _raw << " registered. Size is " << _n << ". Slot is " << _raw_slot << ".\n";
#endif
#ifndef NDEBUG
					if( arraySize == 0 ) {
						assert( _assigned == NULL );
					}
					if( _assigned == NULL ) {
						assert( arraySize == 0 );
					}
#endif
					rc = lpf_register_global( data.context, _assigned, arraySize, &_assigned_slot );
					if( rc == LPF_SUCCESS ) {
						data.signalMemslotTaken();
					}
				}
				if( rc == LPF_SUCCESS ) {
#ifdef _DEBUG
					std::cout << data.s << ": pointer at " << _assigned << " registered. Size is " << arraySize << ". Slot is " << _assigned_slot << ".\n";
#endif
#ifndef NDEBUG
					if( stackSize == 0 ) {
						assert( stack == NULL );
					}
					if( stack == NULL ) {
						assert( stackSize == 0 );
					}
#endif
					rc = lpf_register_global( data.context, stack, stackSize, &_stack_slot );
					if( rc == LPF_SUCCESS ) {
#ifdef _DEBUG
						std::cout << data.s << ": pointer at " << stack << " registered. Size is " << stackSize << ". Slot is " << _stack_slot << ".\n";
#endif
						data.signalMemslotTaken();
					}
				}
				assert( _raw_slot != LPF_INVALID_MEMSLOT );
				assert( _assigned_slot != LPF_INVALID_MEMSLOT );
				assert( _stack_slot != LPF_INVALID_MEMSLOT );

				// sanity check
				if( rc != LPF_SUCCESS ) {
					// according to the spec, this can never happen. So if it does, it's proper to panic.
					throw std::runtime_error( "Error during call to "
											  "lpf_register_global during "
											  "BSP1D Vector initialisation" );
				}

				// activate registrations
				if( lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
					throw std::runtime_error( "Could not activate new memory registrations" );
				}
			}

			//build PIDmap
			{
				size_t totalLength = 0;
				for( size_t k = 0; k < data.P; ++k ) {
					const size_t curLength = internal::Distribution< BSP1D >::global_length_to_local( _n, k, data.P );
					if( curLength > 0 ) {
						totalLength += curLength;
						PIDmap[ totalLength ] = k;
#ifdef _DEBUG
						std::cout << "\t" << data.s << ": PIDmap[ " << totalLength << " ] = " << k << "\n";
#endif
					}
				}
			}

		}

		/** Updates the number of nonzeroes if and only if the nonzero count might have changed. */
		RC updateNnz() const noexcept {
			// if nonzero count cannot have changed
			if( ! _nnz_is_dirty ) {
				return SUCCESS;
			}

			// cache old number of nonzeroes
			const size_t old_nnz = _nnz;
			// get local number of nonzeroes
			_nnz = nnz( _local );
			// call allreduce on it
			const RC rc = collectives< BSP1D >::allreduce< descriptors::no_casting, operators::add< size_t > >( _nnz );
			// check for error
			if( rc == SUCCESS ) {
				// update _became_dense flag
				if( old_nnz < _n && _nnz == _n ) {
					assert( ! _became_dense );
					_became_dense = true;
				}
				_nnz_is_dirty = false;
			}

			// done
			return rc;
		}

		/**
		 * Synchronises the nonzero values of the global view, assuming the global
		 * view is a dense vector. The run-time will be linear in the global vector
		 * size n. This code will execute one allgather.
		 */
		RC dense_synchronize( internal::Coordinates< _GRB_BSP1D_BACKEND > &global_coordinates ) const {
			const auto data = internal::grb_BSP1D.cload();
			assert( data.P > 1 );

#ifdef _DEBUG
			std::cout << "Issuing allgathers on value array from offset " << _offset * sizeof( bool ) << " length " << _local_n << ". P = " << data.P << "\n";
#endif

			// perform allgather on the vector data
			lpf_err_t rc = internal::allgather(
				_raw_slot, _offset * sizeof( D ),
				_raw_slot, _offset * sizeof( D ),
				_local_n * sizeof( D ),
				_n * sizeof( D )
			);
			if( rc != LPF_SUCCESS ) {
				std::cerr << "\t Allgather on values failed\n";
				return PANIC;
			}

			// set global coordinates to dense
			if( rc == LPF_SUCCESS ) {
				global_coordinates.assignAll();
			}

			// and done!
			return SUCCESS;
		}

		/**
		 * Synchronises the nonzero values of the global view.
		 *
		 * This variant uses the _assigned array of the local views to reconstruct the
		 * global stack. The run-time will be linear in the global vector size n. This
		 * code will execute two allgathers.
		 */
		RC array_synchronize( internal::Coordinates< _GRB_BSP1D_BACKEND > &global_coordinates ) const {
			const auto data = internal::grb_BSP1D.cload();
			assert( data.P > 1 );

#ifdef _DEBUG
			std::cout << "Issuing allgathers on value and assigned array from offset " << _offset * sizeof( bool ) << " length " << _local_n << ". P = " << data.P << "\n";
#endif

			// perform allgather on the vector data
			lpf_err_t rc = internal::allgather(
				_raw_slot, _offset * sizeof( D ),
				_raw_slot, _offset * sizeof( D ),
				_local_n * sizeof( D ),
				_n * sizeof( D )
			);

			if( rc != LPF_SUCCESS ) {
				std::cerr << "\t Allgather on values failed\n";
				return PANIC;
			}

			// perform allgather on the sparsity information
			rc = internal::allgather(
					_assigned_slot, _offset * sizeof( bool ),
					_assigned_slot, _offset * sizeof( bool ),
					_local_n * sizeof( bool ),
					_n * sizeof( bool )
				);

			// if succeeded, rebuild stack
			if( rc == LPF_SUCCESS ) {
#ifdef _DEBUG
				std::cout << "Calling rebuild...\n";
#endif
				global_coordinates.rebuild( _became_dense );
				_cleared = false;
				_became_dense = false;
			} else {
				std::cerr << "\t Allgather on sparsity array failed\n";
				return PANIC;
			}

			return SUCCESS;
		}

		/**
		 * Synchronises the nonzero values of the global view.
		 *
		 * This variant uses the _stack array of the local views to reconstruct the
		 * global stack. The run-time will be linear in the total number of nonzeroes
		 * in the global vector. This code will execute two allgathers.
		 *
		 * @param[in] nzs The number of nonzeroes in each local stack.
		 */
		RC stack_synchronize( internal::Coordinates< _GRB_BSP1D_BACKEND > &global_coordinates, size_t * const nzs, const internal::Coordinates< _GRB_BSP1D_BACKEND > &local_coordinates ) const {
			auto &data = internal::grb_BSP1D.load();
			assert( data.P > 1 );

#ifdef _DEBUG
			std::cout << "Issuing allgathers using packed value and stack arrays\n";
#endif

			// compute offsets
			for( size_t i = 1; i < data.P; ++i ) {
				nzs[ i ] += nzs[ i-1 ];
			}

			const size_t global_nz = nzs[ data.P - 1 ];
#ifdef _DEBUG
			std::cout << "\t computed prefix sum array of ( 0 ";
			for( size_t i = 0; i < data.P; ++i ) {
				std::cout << nzs[ i ] << " ";
			}
			std::cout << ")\n";
#endif

			assert(
				data.checkBufferSize(
					data.P * sizeof(size_t) + // nzs
					nzs[ data.s ] * sizeof(internal::Coordinates< _GRB_BSP1D_BACKEND >::StackType) + // output_ind_buf
					(nzs[ data.s ] + global_nz) * utils::SizeOf< D >::value + // output_val_buf and input_val_buf
					sizeof(size_t) // padding
				) == SUCCESS
			);

			RC ret = SUCCESS;
			D * input_val_buf = nullptr;
			D * output_val_buf = nullptr;
			typename internal::Coordinates< _GRB_BSP1D_BACKEND >::StackType * output_ind_buf = nullptr;
			size_t input_val_buf_o = 0;
			size_t output_val_buf_o = 0;
			size_t output_ind_buf_o = 0;

			// find offsets in shared global buffer
			// value input and output sections are only allocated in case of non-void vectors
			if( utils::SizeOf< D >::value > 0 ) {
				// compute offsets for local input and output buffers for nonzero values
				input_val_buf_o = data.P * sizeof(size_t);
				input_val_buf = static_cast< D * >( static_cast< void * >( nzs + data.P ) );
				output_val_buf_o = input_val_buf_o + global_nz * sizeof(D);
				output_val_buf = input_val_buf + global_nz;
				{
					char * unaligned_ptr = static_cast< char * >( static_cast< void * >(output_val_buf + nzs[ data.s ]) );
					const uintptr_t x = reinterpret_cast< uintptr_t >(unaligned_ptr) % sizeof(int);
					const ptrdiff_t y = sizeof(int) - x;
					unaligned_ptr += y;
					output_ind_buf_o = output_val_buf_o + nzs[ data.s ] * sizeof(D) + y;
					output_ind_buf = static_cast< internal::Coordinates< _GRB_BSP1D_BACKEND >::StackType * >( static_cast< void * >(unaligned_ptr) );
				}
			} else {
				output_ind_buf_o = data.P * sizeof(size_t);
				output_ind_buf = static_cast< internal::Coordinates< _GRB_BSP1D_BACKEND >::StackType * >( static_cast< void * >( nzs + data.P ) );
			}

			// pack values and indices
			ret = local_coordinates.packValues( output_ind_buf, _offset, output_val_buf, _raw + _offset );

			// clear global vector
			if( ret == SUCCESS ) {
				global_coordinates.clear();
			}

			// perform allgather for values
			if( input_val_buf != nullptr && ret == SUCCESS ) {
				const size_t offset = data.s == 0 ? 0 : nzs[ data.s - 1 ];
#ifdef _DEBUG
				std::cout << "Issuing allgather on shared buffer:\n"
					<< "\t source offset: " << output_val_buf_o << "\n"
					<< "\t destination offset: " << (input_val_buf_o + offset * utils::SizeOf< D >::value) << "\n"
					<< "\t source length: " << (local_coordinates.nonzeroes() * utils::SizeOf< D >::value) << "\n"
					<< "\t total incoming bytes: " << (global_nz * utils::SizeOf< D >::value) << "\n";
				std::cout << "Source values: ( ";
				for( size_t k = 0; k < local_coordinates.nonzeroes(); ++k ) {
					std::cout << output_val_buf[ k ] << " ";
				}
				std::cout << ")\n";
#endif
				ret = internal::allgather( data.slot, output_val_buf_o,
					data.slot, input_val_buf_o + offset * utils::SizeOf< D >::value,
					local_coordinates.nonzeroes() * utils::SizeOf< D >::value,
					global_nz * utils::SizeOf< D >::value,
					false
				);
#ifdef _DEBUG
				std::cout << "Received values: ( ";
				for( size_t k = 0; k < global_nz; ++k ) {
					std::cout << input_val_buf[ k ] << " ";
				}
				std::cout << ")\n";
#endif
			}

			// perform allgather for stack entries
			if( ret == SUCCESS ) {
				const size_t offset = data.s == 0 ? 0 : nzs[ data.s - 1 ];
				ret = internal::allgather( data.slot, output_ind_buf_o,
					_stack_slot, offset * sizeof( internal::Coordinates< _GRB_BSP1D_BACKEND >::StackType ),
					local_coordinates.nonzeroes() * sizeof( internal::Coordinates< _GRB_BSP1D_BACKEND >::StackType ),
					global_nz * sizeof( internal::Coordinates< _GRB_BSP1D_BACKEND >::StackType ),
					false
				);
			}

			// rebuild global sparsity structure in O(nz) time
			if( ret == SUCCESS ) {
				ret = global_coordinates.rebuildFromStack( _raw, input_val_buf, global_nz );
			}

			// done
			return ret;
		}

		/**
		 * Synchronises the vector across all \a P user processes to obtain a
		 * consistent and up-to-date global view of the vector for local processing.
		 *
		 * During normal operation, only \a _local is kept up to date. During some
		 * operations, however, like grb::mxv, the chosen 1D distribution requires
		 * a global view of an input vector. This view is obtainable via \a _global
		 * which is \em not kept up to date during normal GraphBLAS operation. This
		 * function synchronises this global view.
		 *
		 * \parblock
		 * \par Performance semantics
		 * This function incurs the BSP cost of two allgathers resulting in two
		 * arrays of size \a _n, each consisting of
		 *   -# elements of type \a D;
		 *   -# elements of type \a bool.
		 * The cost of these two allgathers are at most
		 * \f$ 2 ( ( \lceil n/P \rceil g + l ) \log P ), \f$ where
		 * \a n is the global vector length, \a P the number of user processes,
		 * \a g the BSP message gap, and \a l the BSP latency.
		 *
		 * See internal::allgather for an exact cost description.
		 * \endparblock
		 *
		 * @return SUCCESS If the synchronisation is successful.
		 * @return PANIC   If the communication layer fails in an unmitigable way.
		 *
		 * @see synchronize_values for the synchronisation of nonzero values.
		 * @see synchronize_sparsity for the synchronisation of sparsity information.
		 */
		RC synchronize() const {
			RC ret = SUCCESS;

			auto &global_coordinates =
				const_cast< typename internal::Coordinates< _GRB_BSP1D_BACKEND >& >(
					internal::getCoordinates( _global )
				);
			auto &data = internal::grb_BSP1D.load();

			// catch trivial case
			if( data.P == 1 ) {
				global_coordinates.rebuild( _became_dense );
				_cleared = false;
				_became_dense = false;
				return ret;
			}

			const auto &local_coordinates = internal::getCoordinates( _local );
			const size_t n = local_coordinates.size();
			const bool local_dense = local_coordinates.nonzeroes() == n;

#ifdef _DEBUG
			std::cout << "Synchronizing local vectors (mine has " << local_coordinates.nonzeroes()
				<< " / " << n << " nonzeroes) to one vector of size " << _n << ".\n";
			std::cout << "\t my local vector has nonzeroes at coordinates ";
			for( size_t k = 0; k < local_coordinates.nonzeroes(); ++k ) {
				std::cout << local_coordinates.index( k ) << " ";
			}
			std::cout << "\n\t my present global view has " << global_coordinates.nonzeroes()
				<< " / " << global_coordinates.size() << " nonzeroes.\n";
#endif

			bool global_dense = local_dense;
			ret = collectives<>::allreduce( global_dense, grb::operators::logical_and< bool >() );
			if( ret != SUCCESS ) {
				std::cerr << "\t failed inferring global density status\n";
				return ret;
			}

#ifdef _DEBUG
			if( global_dense ) {
				std::cout << "\t all local vectors are dense, "
					<< "therefore the global vector shall become dense also\n";
			} else {
				std::cout << "\t not all local vectors are dense, "
					<< "therefore the global vector shall become sparse\n";
			}
#endif

			// get statistics beyond binary density status
			size_t global_nz = local_coordinates.nonzeroes();
			size_t * nzs = nullptr;
			if( !global_dense ) {
				assert( data.checkBufferSize( data.P * sizeof(size_t) ) == SUCCESS );
				nzs = data.template getBuffer< size_t >();
				nzs[ data.s ] = local_coordinates.nonzeroes();
				ret = internal::allgather(
					data.slot, data.s * sizeof(size_t),
					data.slot, data.s * sizeof(size_t), sizeof(size_t),
					(data.P - 1) * sizeof(size_t)
				);
				for( size_t k = 0; ret == SUCCESS && k < data.s; ++k ) {
					global_nz += nzs[ k ];
				}
				for( size_t k = data.s + 1; ret == SUCCESS && k < data.P; ++k ) {
					global_nz += nzs[ k ];
				}
#ifdef _DEBUG
				std::cout << "\t nzs = ( ";
				for( size_t k = 0; k < data.P; ++k ) {
					std::cout << nzs[ k ] << " ";
				}
				std::cout << ")\n\t global_nz = " << global_nz << ".\n";
#endif
			}

			// dispatch to proper variant
			if( ret == SUCCESS ) {
				if( global_dense ) {
#ifdef _DEBUG
					std::cout << "\t using the dense synchronization algorithm\n";
#endif
					ret = dense_synchronize( global_coordinates );
				} else if( global_coordinates.size() * sizeof(bool) <
					global_nz * sizeof(grb::config::VectorIndexType)
				) {
#ifdef _DEBUG
					std::cout << "\t using the array-driven synchronization algorithm\n";
#endif
					ret = array_synchronize( global_coordinates );
				} else {
#ifdef _DEBUG
					std::cout << "\t using the stack-driven synchronization algorithm\n";
#endif
					ret = stack_synchronize( global_coordinates, nzs, local_coordinates );
				}
			}

#ifdef _DEBUG
			if( ret == SUCCESS ) {
				std::cout << "Sync completed. Returning a global vector with " << global_coordinates.nonzeroes() << " / " << _n << " nonzeroes at positions ";
				for( size_t k = 0; k < global_coordinates.nonzeroes(); ++k ) {
					std::cout << global_coordinates.index( k ) << " ";
				}
				std::cout << "\n";
			}
#endif
			return ret;
		}

		/**
		 * Takes P dense vectors and performs a reduce-scatter, resulting in a dense
		 * local vector.
		 *
		 * @param[in,out] data Process-local data and vectors.
		 * @param[in]     acc  The accumulator to use for reduction.
		 *
		 * @returns SUCCESS  when the function completes successfully
		 * @returns PANIC    when the communication layer failed during all-to-all
		 * @returns ILLEGAL  when reducing empty containers
		 * @returns MISMATCH when reducing mismatching containers after all-to-all
		 */
		template< Descriptor descr, class Acc >
		RC dense_combine( internal::BSP1D_Data &data, const Acc &acc ) {
			const auto &P = data.P;
			const auto &s = data.s;
#ifndef NDEBUG
			auto &local_coordinates = internal::getCoordinates( _local );
#endif
#ifdef _DEBUG
			std::cout << "\t" << s
					  << ": performing a dense combine, requesting all-to-all of "
					  << _local_n * sizeof( D ) << " bytes at local offset " << _offset * sizeof( D ) << "...\n";
#endif
			assert( local_coordinates.size() == _local_n );
			assert( data.checkBufferSize( _local_n * P ) == grb::SUCCESS );
			// NOTE: this alltoall does not perform more communication than optimal
			RC ret = internal::alltoall( _raw_slot, _offset * sizeof( D ), _local_n * sizeof( D ) );
			if( ret == SUCCESS ) {
#ifdef _DEBUG
				std::cout << "\t\t" << s << ": post all-to-all... " << std::endl;
#endif
				if( ! internal::getCoordinates( _local ).isDense() ) {
					(void)internal::getCoordinates( _local ).assignAll();
				}
				const D * __restrict__ const valbuf = data.template getBuffer< D >();
				if( s != 0 ) {
					ret = internal::foldl_from_raw_matrix_to_vector< descriptors::no_operation >(
						_local,
						valbuf,
						_local_n, s,
						s,
						acc
					);
				}
				if( ret == SUCCESS && s + 1 != P ) {
#ifdef _DEBUG
					std::cout << "\t\t\t" << s << ": shifting buffer to offset " << ( s + 1 ) * _local_n << "\n";
#endif
					ret = internal::foldl_from_raw_matrix_to_vector< descriptors::no_operation >(
						_local,
						valbuf + (s + 1) * _local_n,
						_local_n, (P - s - 1),
						(P - s - 1),
						acc
					);
				}
#ifdef _DEBUG
				std::cout << "\t\t" << s << ": local vector now contains " << nnz( _local ) << " / " << size( _local ) << " nonzeroes... ";
#endif

#ifdef _DEBUG
				std::cout << "\t\t" << s << ": complete!\n";
#endif
			}
#ifdef _DEBUG
			else {
				std::cout << "failed with return code " << ret << "!\n";
			}
#endif
			return ret;
		}

		/**
		 * Takes P space vectors and performs a reduce-scatter using the sparsity
		 * structure as given by a dense bitmask array. Both the sparsity array as
		 * well as the value array will be all-to-all'd, with the latter using the
		 * assumption that it is dense.
		 *
		 * @param[in,out] data Process-local data and vectors.
		 * @param[in]     acc  The accumulator to use for reduction.
		 *
		 * @returns SUCCESS  when the function completes successfully
		 * @returns PANIC    when the communication layer failed during all-to-all
		 * @returns ILLEGAL  when reducing empty containers
		 * @returns MISMATCH when reducing mismatching containers after all-to-all
		 */
		template< Descriptor descr, class Acc >
		RC array_combine( internal::BSP1D_Data &data, const Acc &acc ) {
			const auto &P = data.P;
			const auto &s = data.s;
#ifdef _DEBUG
			std::cout << "\t" << s << ": in array-based sparse combine\n";
#endif

			// initialise local buffers
			D * valbuf = nullptr;
			bool * agnbuf = nullptr;
			assert( data.checkBufferSize( P * _local_n *
					(sizeof(D) + sizeof( internal::Coordinates< _GRB_BSP1D_BACKEND >::ArrayType))
				) == grb::SUCCESS
			);
			valbuf = data.template getBuffer< D >();
			static_assert( ( sizeof( D ) % sizeof( bool ) ) == 0, "Bad alignment resulting in UB detected! Please submit a bug report." );
			agnbuf = reinterpret_cast< bool * >( ( valbuf + (P * _local_n) ) );
			const size_t bitmask_array_offset = P * _local_n * sizeof(D);

#ifdef _DEBUG
			std::cout << "\t" << s << ": valbuf at " << valbuf << ".\n"
				<< "\t" << s << ": agnbuf at " << agnbuf << ".\n"
				<< "\t" << s << ": offset of bitmask to value array is " << bitmask_array_offset << ".\n"
				<< "\t" << s << ": alltoall from " << _raw_slot << " @ " << _offset * sizeof( D )
				<< " of length " << _local_n * sizeof( D ) << " requested. Destination offset is 0.\n";
#endif
			// exchange value array
			RC ret = internal::alltoall( _raw_slot, _offset * sizeof( D ), _local_n * sizeof( D ) );

			// exchange SPA (bitmask) array
			if( ret == SUCCESS ) {
#ifdef _DEBUG
				std::cout << "\t" << s << ": alltoall from " << _assigned_slot
					<< " @ " << _offset * sizeof( internal::Coordinates< _GRB_BSP1D_BACKEND >::ArrayType )
					<< " of length " << _local_n * sizeof( internal::Coordinates< _GRB_BSP1D_BACKEND >::ArrayType ) << " requested."
					<< " Destination offset is " << bitmask_array_offset << ".\n";
#endif
				ret = internal::alltoall(
					_assigned_slot, _offset * sizeof( internal::Coordinates< _GRB_BSP1D_BACKEND >::ArrayType ),
					_local_n * sizeof( internal::Coordinates< _GRB_BSP1D_BACKEND >::ArrayType ),
					bitmask_array_offset
				);
			}
			if( ret == SUCCESS && s > 0 ) {
#ifdef _DEBUG
				std::cout << "\t" << s << ": foldl_from_raw_matrix_to_vector into " << &_local << " requested.\n"
					<< "\t To-be-folded matrix is of size " << _local_n << " by " << s << ".\n";
#endif
				ret = internal::foldl_from_raw_matrix_to_vector< descriptors::no_operation, true >(
					_local,
					valbuf, agnbuf,
					_local_n, s,
					s, acc
				);
			}
			if( ret == SUCCESS && s + 1 < P ) {
#ifdef _DEBUG
				std::cout << "\t" << s << ": foldl_from_raw_matrix_to_vector into " << &_local << " requested. To-be-folded matrix is of size " << _local_n << " by " << ( P - s - 1 )
						  << ", and was shifted with " << ( s + 1 ) << " columns. Agnbuf offset is " << ( s + 1 ) * _local_n << std::endl;
#endif
				ret = internal::foldl_from_raw_matrix_to_vector< descriptors::no_operation, true >(
					_local,
					valbuf + (s + 1) * _local_n, agnbuf + (s + 1) * _local_n,
					_local_n, (P - s - 1),
					(P - s - 1),
					acc
				);
			}

#ifdef _DEBUG
			std::cout << "\t" << s << ": exiting array-based sparse combine with exit code " << ret << "\n";
#endif
			// done
			return ret;
		}

		/**
		 * Takes P space vectors and performs a reduce-scatter using the sparsity
		 * structure as given by a nonzero stack.
		 *
		 * @param[in,out] data Process-local data and vectors.
		 * @param[in]     nzsk Buffer for counting outgoing nonzeroes.
		 * @param[in]     nzks Buffer for counting incoming nonzeroes.
		 * @param[in] global_nzs Buffer for counting the total number of nonzeroes.
		 * @param[in]  recv_nz Number of nonzeroes this process will reseive.
		 * @param[in]  sent_nz Number of nonzeroes this process will send.
		 * @param[in]     acc  The accumulator to use for reduction.
		 *
		 * @returns SUCCESS  when the function completes successfully
		 * @returns PANIC    when the communication layer failed during all-to-all
		 * @returns ILLEGAL  when reducing empty containers
		 * @returns MISMATCH when reducing mismatching containers after all-to-all
		 */
		template< Descriptor descr, class Acc >
		RC stack_combine( internal::BSP1D_Data &data, size_t * const nzsk, size_t * const nzks, size_t * const global_nzs, size_t &recv_nz, size_t sent_nz, const Acc &acc ) {
			const auto &P = data.P;
			const auto &s = data.s;
			const auto &global_coordinates = internal::getCoordinates( _global );
			auto &local_coordinates = internal::getCoordinates( _local );
#ifdef _DEBUG
			std::cout << "\t" << s
					  << ": in stack-based sparse combine. Retrieving "
						 "stack and initialising counting sort..."
					  << std::endl;
#endif
			// retrieve stack of global coordinates
			size_t stackSize = 0;
			internal::Coordinates< reference >::StackType * __restrict__ stack = global_coordinates.getStack( stackSize );
#ifdef _DEBUG
			std::cout << "\t" << s << ": local stack size is " << stackSize << ".\n";
#endif
			static_assert( sizeof( size_t ) % sizeof( internal::Coordinates< reference >::StackType ) == 0,
				"size_t is not a multiple of StackType's size while the "
				"code does assume this is true. Please submit a ticket to "
				"get this fixed!" );

			// compute global_nzs using nzsk
			{
#ifdef _DEBUG
				std::cout << "\t" << s << ": nzsk = ( ";
				for( size_t i = 0; i < P; ++i ) {
					std::cout << nzsk[ i ] << " ";
				}
				std::cout << ")\n";
#endif
				global_nzs[ 0 ] = 0;
#ifdef _DEBUG
				std::cout << "\t" << s << ": global_nzs reads ( 0 ";
#endif
				for( size_t i = 0; P > 1 && i < P - 1; ++i ) {
					global_nzs[ i + 1 ] = global_nzs[ i ] + nzsk[ i ];
#ifdef _DEBUG
					std::cout << global_nzs[ i + 1 ] << " ";
#endif
				}
#ifdef _DEBUG
				std::cout << "). Check is " << ( global_nzs[ P - 1 ] == stackSize - nzsk[ P - 1 ] ) << std::endl;
#endif
				assert( global_nzs[ P - 1 ] == stackSize - nzsk[ P - 1 ] );
			}

			// replace nzsk by pos array, and initialise
			size_t * const pos = nzks + data.P;
			{
				for( size_t i = 0; i < P; ++i ) {
					pos[ i ] = 0;
				}
			}

			// compute recv_nz and sent_nz
			recv_nz = nzks[ 0 ];
			sent_nz = nzsk[ 0 ];
			for( size_t i = 1; i < P; ++i ) {
				recv_nz += nzks[ i ];
				sent_nz += nzsk[ i ];
			}
			recv_nz -= nzks[ s ];
			sent_nz -= nzsk[ s ];
#ifdef _DEBUG
			std::cout << "\t" << s << ": local #elements to receive:  " << recv_nz << "\n";
			std::cout << "\t" << s << ": local #elements to send out: " << sent_nz << "\n";
#endif
			// prepare buffer
			assert( data.checkBufferSize( 6 * P * sizeof( size_t ) + ( recv_nz + 1 ) * ( sizeof( internal::Coordinates< reference >::StackType ) + sizeof( D ) ) +
								( sent_nz + nzsk[ s ] + 1 ) * sizeof( D ) ) == grb::SUCCESS );
			char * raw_buffer = data.template getBuffer< char >();

			// store outgoing values after 6P size_t values
			size_t valbuf_o = 6 * P * sizeof( size_t ) + sizeof( D ) - 1;
			valbuf_o -= ( reinterpret_cast< uintptr_t >( raw_buffer + valbuf_o ) % sizeof( D ) );
			D * __restrict__ const valbuf = reinterpret_cast< D * >( raw_buffer + valbuf_o );
			assert( reinterpret_cast< uintptr_t >( valbuf ) - reinterpret_cast< uintptr_t >( raw_buffer ) == valbuf_o );

			// store incoming offsets after that
			size_t indbuf_o = valbuf_o + ( sent_nz + nzsk[ data.s ] ) * sizeof( D ) + sizeof( internal::Coordinates< reference >::StackType ) - 1;
			indbuf_o -= ( reinterpret_cast< uintptr_t >( raw_buffer + indbuf_o ) % sizeof( internal::Coordinates< reference >::StackType ) );
			internal::Coordinates< reference >::StackType * __restrict__ const indbuf = reinterpret_cast< internal::Coordinates< reference >::StackType * >( raw_buffer + indbuf_o );
			assert( reinterpret_cast< uintptr_t >( indbuf ) - reinterpret_cast< uintptr_t >( raw_buffer ) == indbuf_o );

			// store incoming values after that
			size_t dstbuf_o = indbuf_o + recv_nz * sizeof( internal::Coordinates< reference >::StackType ) + sizeof( D ) - 1;
			dstbuf_o -= ( reinterpret_cast< uintptr_t >( raw_buffer + dstbuf_o ) % sizeof( D ) );
			D * __restrict__ dstbuf = reinterpret_cast< D * >( raw_buffer + dstbuf_o );

#ifdef _DEBUG
			std::cout << "\t" << s << ": receive buffers created at " << valbuf << ", " << indbuf << ", and " << dstbuf << ".\n";
			std::cout << "\t\t" << s
					  << ": these corresponds to the following "
						 "offsets; "
					  << valbuf_o << " (valbuf_o), " << indbuf_o << " (indbuf_o), " << dstbuf_o << " (dstbuf_o)\n";
			std::cout << "\t" << s << ": performing counting sort of " << stackSize << " stack elements..." << std::endl;
			for( size_t i = 0; i < 10 && i < stackSize; ++i ) {
				std::cout << "\t\t" << stack[ i ] << "\n";
			}
			if( stackSize > 10 ) {
				std::cout << "\t\t...\n";
				for( size_t i = stackSize - 10; i < stackSize; ++i ) {
					std::cout << "\t\t" << stack[ i ] << "\n";
				}
			}
			std::cout << "\t" << s << ": end (partial) list of stack elements.\n";
#endif

			// finalise counting sort of stack
			{
				size_t i = 0, src = 0;
				while( src < P && global_nzs[ src + 1 ] == 0 ) {
					(void)++src;
				}
				while( src < P && i < stackSize ) {
#ifdef _DEBUG
					std::cout << "\t" << s << ": stack @ " << stack << ", position " << i << " / " << stackSize;
#endif
					const size_t index = stack[ i ];
#ifdef _DEBUG
					std::cout << ", has index " << index << " which refers to value " << _raw[ index ];
#endif
					const size_t dst = PIDmap.upper_bound( index )->second;
#ifdef _DEBUG
					std::cout << ", and should map to PID " << dst << ".\n";
#endif
					if( src == dst ) {
#ifdef _DEBUG
						std::cout << "\t" << s
								  << ": source matches destination, "
									 "copying value...\n";
#endif
						valbuf[ i ] = _raw[ index ];
						(void)++i;
						if( i == global_nzs[ src + 1 ] ) {
#ifdef _DEBUG
							std::cout << "\t" << s << ": these were all " << global_nzs[ src + 1 ] - global_nzs[ src ] << " elements that were assigned to PID " << src << ".\n";
#endif
							if( src + 1 < P ) {
#ifdef _DEBUG
								std::cout << "\t" << s
										  << ": shifting to next "
											 "bucket...\n";
#endif
								(void)++src;
							}
							while( src + 1 < P && i == global_nzs[ src + 1 ] ) {
#ifdef _DEBUG
								std::cout << "\t" << s << ": bucket " << src
										  << " was also already completed. "
											 "Shifting to next one, and skipping "
										  << pos[ src ] << " elements.\n";
#endif
								i += pos[ src ];
								(void)++src;
							}
							if( src == P ) {
#ifdef _DEBUG
								std::cout << "\t" << s << ": all buckets sorted!\n";
#endif
								break;
							}
						}
					} else {
						const size_t j = global_nzs[ dst ] + pos[ dst ];
#ifdef _DEBUG
						std::cout << "\t" << s << ": swapping " << i << " with " << j
								  << " and writing value to the latter "
									 "index in valbuf...\n";
#endif
						std::swap( stack[ i ], stack[ j ] );
						valbuf[ j ] = _raw[ index ];
					}
					(void)++( pos[ dst ] );
#ifdef _DEBUG
					std::cout << "\t" << s << " shifted number of elements in bucket " << dst << " by one. New value is " << pos[ dst ] << ".\n";
#endif
				}
			}
#ifdef _DEBUG
			std::cout << "\t" << s
					  << ": counting sort on stack completed. Now "
						 "computing offsets..."
					  << std::endl;
#endif

			size_t * const local_offset = pos;
			size_t * const remote_offset = pos + P;
			size_t * const remote_val_offset = data.template getBuffer< size_t >(); // OK since no buffered collective calls (like collectives<>::allreduce) are forthcoming
			size_t recv = data.s == 0 ? 0 : nzks[ 0 ];
			local_offset[ 0 ] = remote_offset[ 0 ] = 0;
#ifdef _DEBUG
			std::cout << "\t" << s << ": local_offset[ 0 ] is 0\n";
#endif
			for( size_t i = 1; i < P; ++i ) {
				if( i != data.s ) {
					recv += nzks[ i ];
				}
				if( i - 1 == s ) {
					remote_offset[ i ] = remote_offset[ i - 1 ];
				} else {
					remote_offset[ i ] = remote_offset[ i - 1 ] + nzsk[ i - 1 ];
				}
				local_offset[ i ] = local_offset[ i - 1 ] + nzsk[ i - 1 ];
			}
			assert( data.template getBuffer< size_t >() + 4 * P == local_offset );
			assert( data.template getBuffer< size_t >() + 5 * P == remote_offset );
			assert( data.template getBuffer< size_t >() == remote_val_offset );
			RC ret = internal::alltoall( data.slot, 5 * P * sizeof( size_t ) + s * sizeof( size_t ), sizeof( size_t ), 0, false );
			if( ret == SUCCESS ) {
				ret = internal::alltoall( data.slot, 4 * P * sizeof( size_t ) + s * sizeof( size_t ), sizeof( size_t ), 5 * P * sizeof( size_t ), false );
			}

			if( ret == SUCCESS ) {
#ifdef _DEBUG
				for( size_t i = 0; i < P; ++i ) {
					std::cout << "\t" << s << ": remote_offset[ " << i << " ] is " << remote_offset[ i ] << "\n";
					std::cout << "\t" << s << ": remote_val_offset[ " << i << " ] is " << remote_val_offset[ i ] << "\n";
				}
#endif

				local_offset[ 0 ] = 0;
				for( size_t i = 1; i < P; ++i ) {
					if( s == i - 1 ) {
						local_offset[ i ] = local_offset[ i - 1 ];
					} else {
						local_offset[ i ] = local_offset[ i - 1 ] + nzks[ i - 1 ];
					}
				}

#ifdef _DEBUG
				for( size_t i = 0; i < P; ++i ) {
					std::cout << "\t" << s << ": local_offset[ " << i << " ] is " << local_offset[ i ] << "\n";
				}
#endif
#ifndef NDEBUG
				// check the stack is indeed monotonically increasingly stored
				assert( P > 1 );
#ifdef _DEBUG
				std::cout << "\t" << s << ": stack size is " << stackSize << "\n";
				std::cout << "\t\t" << s
						  << ": source indices are at offset 0 from "
							 "slot "
						  << _stack_slot << "\n";
				std::cout << "\t\t" << s << ": source values are at offset " << reinterpret_cast< uintptr_t >( valbuf ) - reinterpret_cast< uintptr_t >( raw_buffer ) << " from slot "
						  << data.slot << "\n";
#endif
#ifdef _DEBUG
				for( size_t k = 0; k < P; ++k ) {
					if( k == s ) {
						if( stackSize > 0 ) {
							std::cout << "\t" << s
									  << ": sorted stack entry 0 has "
										 "index "
									  << stack[ 0 ] << " and value " << valbuf[ 0 ] << "\n";
						}
#endif
						if( stackSize > 1 ) {
							for( size_t i = 1; i < stackSize; ++i ) {
#ifdef _DEBUG
								std::cout << "\t" << s << ": sorted stack entry " << i << " has index " << stack[ i ] << " and value " << valbuf[ i ] << " and should go to PID "
										  << PIDmap.upper_bound( stack[ i ] )->second << "\n";
#endif
								assert( PIDmap.upper_bound( stack[ i - 1 ] )->second <= PIDmap.upper_bound( stack[ i ] )->second );
							}
						}
#ifdef _DEBUG
						std::cout << "\t" << s
								  << ": sorted stack sanity check now "
									 "complete!"
								  << std::endl;
					}
					assert( spmd< BSP1D >::sync() == SUCCESS );
				}
#endif
#endif

				// nzsk and nzks should now refer to bytes, not elements
#ifdef _DEBUG
				std::cout << "\t" << s << ": Now proceeding to alltoallvs...\n";
				std::cout << "\t\t" << s
						  << ": indices will go into local buffer at "
							 "offset "
						  << indbuf_o << "\n";
#endif
				// do alltoallvs
				for( size_t k = 0; k < P; ++k ) {
					local_offset[ k ] *= sizeof( internal::Coordinates< reference >::StackType );
					remote_offset[ k ] *= sizeof( internal::Coordinates< reference >::StackType );
					nzsk[ k ] *= sizeof( internal::Coordinates< reference >::StackType );
					nzks[ k ] *= sizeof( internal::Coordinates< reference >::StackType );
#ifdef _DEBUG
					for( size_t t = 0; t < P; ++t ) {
						if( t == s ) {
							std::cout << "\t" << t << ": will get " << nzsk[ k ] << " bytes from PID " << k << " at offset " << remote_offset[ k ] << " to local offset " << indbuf_o << " + "
									  << local_offset[ k ] << " = " << ( indbuf_o + local_offset[ k ] ) << " receiving " << nzks[ k ]
									  << " bytes. It will overwrite the values "
										 "starting with "
									  << ( *reinterpret_cast< D * >( raw_buffer + indbuf_o + local_offset[ k ] ) ) << "\n";
						}
						spmd< BSP1D >::sync();
					}
#endif
				}
				ret = internal::alltoallv( _stack_slot, nzsk, 0, remote_offset, nzks, indbuf_o, local_offset, true );
				for( size_t k = 0; k < P; ++k ) {
					local_offset[ k ] /= sizeof( internal::Coordinates< reference >::StackType );
					remote_offset[ k ] /= sizeof( internal::Coordinates< reference >::StackType );
					nzsk[ k ] /= sizeof( internal::Coordinates< reference >::StackType );
					nzks[ k ] /= sizeof( internal::Coordinates< reference >::StackType );
				}
			}
#ifdef _DEBUG
			std::cout << "\t\t" << s << ": values will go into local buffer at offset " << dstbuf_o
				  << "\nReprinting local stacks after 1st all-to-all:\n";
			for( size_t k = 0; k < P; ++k ) {
				if( k == s ) {
					if( stackSize > 0 ) {
						std::cout << "\t" << s << ": sorted stack entry 0 has index " << stack[ 0 ] << " and value " << valbuf[ 0 ] << "\n";
					}
					if( stackSize > 1 ) {
						for( size_t i = 1; i < stackSize; ++i ) {
							std::cout << "\t" << s << ": sorted stack entry " << i << " has index " << stack[ i ] << " and value " << valbuf[ i ]
								<< " and should go to PID " << PIDmap.upper_bound( stack[ i ] )->second << "\n";
						}
					}
					std::cout << "\t" << s << ": sorted stack sanity check now complete!" << std::endl;
				}
				(void)spmd< BSP1D >::sync();
			}

#endif
			if( ret == SUCCESS ) {
				for( size_t k = 0; k < P; ++k ) {
					local_offset[ k ] *= sizeof( D );
					remote_val_offset[ k ] = remote_offset[ k ] * sizeof( D );
					nzsk[ k ] *= sizeof( D );
					nzks[ k ] *= sizeof( D );
#ifdef _DEBUG
					for( size_t t = 0; t < P; ++t ) {
						if( t == s ) {
							std::cout << "\t" << t << ": will get " << nzsk[ k ] << " bytes from PID " << k << " at offset " << valbuf_o << " + " << remote_val_offset[ k ] << " = "
									  << ( valbuf_o + remote_val_offset[ k ] ) << " to local offset " << dstbuf_o << " + " << local_offset[ k ] << " = " << ( dstbuf_o + local_offset[ k ] )
									  << " receiving " << nzks[ k ]
									  << " bytes. It will overwrite the values "
										 "starting with "
									  << ( *reinterpret_cast< D * >( raw_buffer + dstbuf_o + local_offset[ k ] ) ) << ".\n";
							std::cout << "\t" << t
									  << ": remote processes will retrieve "
										 "values from me starting at "
									  << valbuf_o << ". Its first value is " << ( *reinterpret_cast< D * >( raw_buffer + valbuf_o ) ) << ".\n";
						}
						spmd< BSP1D >::sync();
					}
#endif
				}
				ret = internal::alltoallv( data.slot, // source slots
					nzsk,                             // outgoing sizes
					valbuf_o, remote_val_offset,      // source offsets
					nzks,                             // incoming sizes
					dstbuf_o, local_offset,           // dest. offsets
					true                              // exclude self
				);
			}
			{
#ifdef _DEBUG
				for( size_t k = 0; k < P; ++k ) {
					if( k == s ) {
						std::cout << "\t" << s
								  << ": alltoallv on stacks and value buffers "
									 "completed. Now rewinding the "
								  << recv << " received contributions.\n";
						std::cout << "\t\t" << s << ": indices stack is at offset " << reinterpret_cast< uintptr_t >( indbuf ) - reinterpret_cast< uintptr_t >( raw_buffer ) << "\n";
						std::cout << "\t\t" << s << ": values stack is at offset " << reinterpret_cast< uintptr_t >( dstbuf ) - reinterpret_cast< uintptr_t >( raw_buffer ) << "\n";
#endif
						// TODO internal issue #197
						for( size_t i = 0; i < recv; ++i ) {
							const auto index = indbuf[ i ];
							const D value = dstbuf[ i ];
#ifdef _DEBUG
							std::cout << "\t" << s << ": processing received nonzero #" << i << ", index is " << index << " (offset is " << _offset << ") value is " << value << "..."
									  << std::endl;
#endif
							assert( index >= _offset );
							assert( index - _offset < local_coordinates.size() );
							if( local_coordinates.assign( index - _offset ) ) {
								(void)foldl( _raw[ index ], value, acc );
							} else {
								_raw[ index ] = value;
							}
						}
#ifdef _DEBUG
					}
					spmd< BSP1D >::sync();
				}
#endif
			}
#ifdef _DEBUG
			std::cout << "\t" << s
					  << ": sparse stack-based combine complete; local "
						 "vector has "
					  << local_coordinates.nonzeroes() << " / " << local_coordinates.size() << " nonzeroes.\n";
#endif
			return ret;
		}

		/**
		 * Suppose each user process updated a global view of this vector. Then this
		 * function reduces the various updates. For each element, the owner process
		 * retrieves all \f$ P-1 \f$ copies of that elements and folds them according
		 * to the operator instance \a op of operator type \a OP.
		 *
		 * @tparam Acc  The operator according which to reduce the \a P copies.
		 *
		 * @param[in] The operator instance to be used for reduction.
		 */
		template< Descriptor descr = descriptors::no_operation, class Acc >
		RC combine( const Acc & acc ) {
			// we need access to LPF context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();
			constexpr const bool is_dense = descr & descriptors::dense;
			const auto &global_coordinates = internal::getCoordinates( _global );
			auto &local_coordinates = internal::getCoordinates( _local );
			const auto &P = data.P;
			const auto &s = data.s;

#ifdef _DEBUG
			std::cout << s << ": in Vector< BSP1D >::combine...\n";
			std::cout << "\t" << s << " global coordinates hold " << global_coordinates.nonzeroes() << " / " << global_coordinates.size() << " nonzeroes:\n";
			// for( size_t i = 0; i < global_coordinates.size(); ++i ) {
			//	std::cout << "\t" << global_coordinates._assigned[ i ] << "\n";
			//}
			std::cout << "\t" << s << " local coordinates hold " << local_coordinates.nonzeroes() << " / " << local_coordinates.size() << " nonzeroes:\n";
			// for( size_t i = 0; i < local_coordinates.size(); ++i ) {
			//	std::cout << "\t" << local_coordinates._assigned[ i ] << "\n";
			//}
#endif
			// check trivial case
			if( P == 1 ) {
				const bool is_dense{ ( descr & descriptors::dense ) != descriptors::no_operation };
				_local._coordinates.rebuild( is_dense );
				return SUCCESS;
			}

#ifdef _DEBUG
			std::cout << "\t" << s << ": non-trivial vector combine requested with a " << descriptors::toString( descr ) << "\n";
#endif

			RC ret = SUCCESS;
			assert( data.checkBufferSize( 4 * data.P * sizeof( size_t ) ) == grb::SUCCESS );
			size_t * nzsk = NULL;
			size_t * nzks = NULL;
			size_t * global_nzs = NULL;
			size_t min_global_nz = 0;
			size_t max_global_nz = 0;
			if( ret == SUCCESS ) {
				nzsk = data.template getBuffer< size_t >();
				nzsk += P;
				global_nzs = nzsk + P;
				nzks = global_nzs + P;
				global_nzs[ s ] = global_coordinates.nonzeroes();
				min_global_nz = global_nzs[ s ];
				max_global_nz = global_nzs[ s ];
				for( size_t i = 0; i < P; ++i ) {
					nzsk[ i ] = 0;
				}
				// TODO internal issue #197
				for( size_t i = 0; i < global_coordinates.nonzeroes(); ++i ) {
					const size_t index = is_dense ? i : global_coordinates.index( i );
					const size_t process_id = PIDmap.upper_bound( index )->second;
#ifdef _DEBUG
					std::cout << "\t" << s << ": global stack entry " << i << " has index " << index << " which should map to process " << process_id << "\n";
#endif
					(void)++( nzsk[ process_id ] );
				}
#ifdef _DEBUG
				std::cout << "\t" << s << ": pre-alltoall, my nzsk array is ( " << nzsk[ 0 ];
				for( size_t k = 1; k < data.P; ++k ) {
					std::cout << ", " << nzsk[ k ];
				}
				std::cout << " )\n";
				std::cout << "\t" << s << ": allgather from " << data.slot << " @ " << ( P + s ) << " (" << nzsk[ P + s ] << ") to " << data.slot << " @ " << ( P + s ) << std::endl;
#endif
				ret = internal::allgather( data.slot, ( 2 * P + s ) * sizeof( size_t ), data.slot, ( 2 * P + s ) * sizeof( size_t ), sizeof( size_t ), P * sizeof( size_t ), true );
#ifdef _DEBUG
				if( ret != SUCCESS ) {
					std::cout << "\t" << s << ": allgather failed.\n";
				} else {
					std::cout << "\t" << s << ": post-allgather, global_nzs array is ( " << global_nzs[ 0 ];
					for( size_t i = 1; i < P; ++i ) {
						std::cout << ", " << global_nzs[ i ];
					}
					std::cout << " )\n";
				}
#endif
			}
			if( ret == SUCCESS ) {
				ret = internal::alltoall( data.slot, ( s + P ) * sizeof( size_t ), sizeof( size_t ), 3 * P * sizeof( size_t ), false );
#ifdef _DEBUG
				if( ret != SUCCESS ) {
					std::cout << "\t" << s << ": alltoall failed.\n";
				}
#endif
			}
			// buffer contents at this point:
			//(nz^s_0,...,nz^s_{p-1},nz_0,...,nz_{p-1},nz_s^0,...,nz_s^{p-1})
			if( ret == SUCCESS ) {
				for( size_t k = 0; k < data.P; ++k ) {
					if( k == s ) {
						continue;
					}
					if( min_global_nz > global_nzs[ k ] ) {
						min_global_nz = global_nzs[ k ];
					}
					if( max_global_nz < global_nzs[ k ] ) {
						max_global_nz = global_nzs[ k ];
					}
				}
			}

#ifdef _DEBUG
			for( size_t k = 0; ret == SUCCESS && k < P; ++k ) {
				if( k == s ) {
					std::cout << "\t" << s << ": my global nnz is " << global_nzs[ s ] << "(/" << _n << "), minimum across all user processes is " << min_global_nz
							  << ", maximum across all user processes is " << max_global_nz << std::endl;
					std::cout << "\t" << s << ": my nzsk array is ( " << nzsk[ 0 ];
					for( size_t i = 1; i < P; ++i ) {
						std::cout << ", " << nzsk[ i ];
					}
					std::cout << " )\n";
					std::cout << "\t" << s << ": my nzks array is ( " << nzks[ 0 ];
					for( size_t i = 1; i < P; ++i ) {
						std::cout << ", " << nzks[ i ];
					}
					std::cout << " )\n";
				}
				ret = spmd< BSP1D >::sync( 0, 0 );
			}
#else
			(void)max_global_nz;
#endif

			// exit on error
			if( ret != SUCCESS ) {
#ifdef _DEBUG
				std::cout << "Combine quitting early due to intermediate error code "
					<< ret << "\n";
#endif
				return ret;
			}

			// otherwise, pick one of three variants:
			if( min_global_nz == _n ) {
				ret = dense_combine< descr >( data, acc );
			} else {
#ifdef _DEBUG
				std::cout << "\t" << s << ": global vector to be reduced is sparse at at least one neighbour. Mine holds "
					<< internal::getCoordinates( _global ).nonzeroes() << " / " << internal::getCoordinates( _global ).size() << " nonzeroes.\n";
				std::cout << "\t" << s << ": local vector prior to rebuild holds "
					<< internal::getCoordinates( _local ).nonzeroes() << " / " << internal::getCoordinates( _local ).size()
					<< " nonzeroes.\n";
				// std::cout << "\t" << s << ": local assigned array is at " << local_coordinates._assigned << ", global one is at " << global_coordinates._assigned << ". Global plus offset is " <<
				// global_coordinates._assigned + _offset << "\n";
#endif
				// rebuild local stack
				local_coordinates.rebuild( false );
#ifdef _DEBUG
				std::cout << "\t" << s << ": local vector after rebuild holds "
					<< internal::getCoordinates( _local ).nonzeroes() << " / " << internal::getCoordinates( _local ).size()
					<< " nonzeroes.\n";
				std::cout << "\t" << s << ": nzsk = ( ";
				for( size_t i = 0; i < data.P; ++i ) {
					std::cout << nzsk[ i ] << " ";
				}
				std::cout << ")\n";
#endif
				size_t sent_nz, recv_nz;
				sent_nz = recv_nz = 0;
				for( size_t k = 0; k < P; ++k ) {
					if( k == s ) {
						continue;
					}
					sent_nz += nzsk[ k ];
					recv_nz += nzks[ k ];
				}
#ifdef _DEBUG
				std::cout << "\t" << s << ": calling allreduce over sent_nz = " << sent_nz << "\n";
#endif
				ret = collectives< BSP1D >::allreduce( sent_nz, operators::max< size_t >() );
				if( ret == SUCCESS ) {
#ifdef _DEBUG
					std::cout << "\t" << s << ": reduced sent_nz = " << sent_nz << ". Now calling allreduce over recv_nz = " << recv_nz << "\n";
#endif
					ret = collectives< BSP1D >::allreduce( recv_nz, operators::max< size_t >() );
				}
#ifdef _DEBUG
				if( ret == SUCCESS ) {
					std::cout << "\t" << s << ": reduced recv_nz = " << recv_nz << ".\n";
				}
#endif
				const size_t stack_h = std::max( sent_nz, recv_nz );
				const size_t cost_array = ( _n - internal::Distribution< BSP1D >::global_length_to_local( _n, P - 1, P ) ) * ( sizeof( D ) + sizeof( bool ) );
				const size_t cost_stack = stack_h * ( sizeof( D ) * sizeof( grb::config::VectorIndexType ) );
#ifdef _DEBUG
				std::cout << "\t" << s << ": array-based sparse combine costs " << cost_array << "\n";
				std::cout << "\t" << s << ": stack-based sparse combine costs " << cost_stack << "\n";
#endif
				if( cost_array < cost_stack ) {
					ret = array_combine< descr >( data, acc );
				} else {
					ret = stack_combine< descr >( data, nzsk, nzks, global_nzs, recv_nz, sent_nz, acc );
				}
			}

#ifdef _DEBUG
			std::cout << "\t" << s << ": at Vector< BSP1D >::combine coda with exit code " << ret << "." << std::endl;
#endif
			// global number of nonzeroes may have changed
			if( ret == SUCCESS ) {
#ifdef _DEBUG
				std::cout << "\t" << s << ": now synchronising global number of nonzeroes..." << std::endl;
#endif
				const size_t old_nnz = _nnz;
				operators::add< size_t > adder;
				assert( local_coordinates.nonzeroes() == nnz( _local ) );
				_nnz = local_coordinates.nonzeroes();
#ifdef _DEBUG
				std::cout << "\t" << s << ": allreducing " << _nnz << "...\n";
#endif
				ret = collectives< BSP1D >::allreduce( _nnz, adder );
#ifdef _DEBUG
				std::cout << "\t" << s << ": allreduced global number of nonzeroes: " << _nnz << "." << std::endl;
#endif
				_nnz_is_dirty = false;
				if( _nnz == _n && old_nnz != _n ) {
					_became_dense = true;
				}
			}

			// sync global_coordinates to local_coordinates
			if( ret == SUCCESS ) {
#ifdef _DEBUG
				std::cout << "\t" << s
						  << ": resetting global vector sparsity pattern to "
							 "match that of the combined local vector..."
						  << std::endl;
#endif
				internal::getCoordinates( _global ).template rebuildGlobalSparsity< ( ( descr & descriptors::dense ) > 0 ) >( local_coordinates, _offset );
			}

#ifdef _DEBUG
			std::cout << "\t" << s << ": exiting Vector< BSP1D >::combine with exit code " << ret << ". New global number of nonzeroes: " << _nnz << std::endl;
#endif
			// done
			return ret;
		}

		/**
		 * Constructs the vector given thread-local data corresponding to this user
		 * process.
		 *
		 * \note This constructor is never called explicitly.
		 *
		 * @param[in] data The thread-local data corresponding to this user process.
		 * @param[in] n    The global length of the input vector.
		 *
		 * This implementation inherits the exception throwing properties of the
		 * reference implementation of Vector<DataType,reference>::Vector(n).
		 *
		 * @see Vector for the user-level specfication.
		 *
		 * @throws Out-of-memory When initialisation fails due to out-of-memory
		 *                       conditions.
		 * @throws Runtime error When the POSIX call to get an aligned memory area
		 *                       fails for any other reason.
		 */
		Vector( const internal::BSP1D_Data & data, const size_t n ) :
			_n( 0 ), _nnz( 0 ),
			_raw_slot( LPF_INVALID_MEMSLOT ),
			_assigned_slot( LPF_INVALID_MEMSLOT ),
			_stack_slot( LPF_INVALID_MEMSLOT ),
			_cleared( false ), _became_dense( false ),
			_nnz_is_dirty( false ),	_global_is_dirty( false )
		{
			if( n > 0 ) {
				// set non-trivial fields
				_local_n = internal::Distribution< BSP1D >::global_length_to_local( n, data.s, data.P );
				_offset = internal::Distribution< BSP1D >::local_offset( n, data.s, data.P );
				// delegate
				initialize( NULL, NULL, NULL, n );
			} else {
				// set trivial fields and exit
				_raw = nullptr;
				_assigned = nullptr;
				_buffer = nullptr;
				_global.initialize( nullptr, nullptr, true, nullptr, 0 );
				_local.initialize( nullptr, nullptr, true, nullptr, 0 );
				_local_n = _offset = 0;
			}
		}


	public:
		/** @see Vector::value_type. */
		typedef D value_type;

		/**
		 * Use the same iterator as the reference implementation.
		 *
		 * \internal
		 *
		 *      for performance gain, a specialised iterator could be written that
		 *      when a global view is requested, simply caches the first pb blocks
		 *      locally. This incurs global communication every pb iterations. It
		 *      does require synchronised access to the iterators, however. This
		 *      idea requires some more thinking. Internal issue #94 (TODO).
		 */
		typedef typename LocalVector::template ConstIterator< BSP1D > const_iterator;

		/**
		 * This constructor may throw exceptions.
		 *
		 * @see Vector for the user-level specfication.
		 *
		 * This delegates to the #LocalVector constructor to create a local data
		 * cache. The size of the local cache is \f$ \lfloor n / p \rfloor + 1 \f$,
		 * which is an upper bound on the required local storage in all cases.
		 *
		 * The vector is distributed in a block-cyclic fashion. The block size is
		 * given by \f$ b= \f$ grb::config::CACHE_LINE_SIZE. The first local element
		 * corresponds to element \f$ s \f$ of the global vector.
		 *
		 * \par The global to local map.
		 * The \f$ j \f$th element of the global vector is stored locally only if
		 * the following equals \f$ s \f$:
		 * \f$ \lfloor j / b \rfloor\text{ mod }P \f$.
		 * If true, this element is stored at local index
		 * \f$ \lfloor \lfloor j / b \rfloor / p \rfloor + \cdot j\text{ mod }b \f$.
		 *
		 * \par The local to global map.
		 * The \f$ i \f$th local element of this vector corresponds to the global
		 * index
		 * \f$ \lfloor i / b \rfloor \cdot pb + i\text{ mod }b \f$.
		 *
		 * @param[in] n The global vector length.
		 */
		Vector( const size_t n ) : Vector( internal::grb_BSP1D.cload(), n ) {}

		/**
		 * Copy constructor.
		 *
		 * Incurs the same costs as the normal constructor, followed by a grb::set.
		 *
		 * @throws runtime_error If the call to grb::set fails, the error code is
		 *                       caught and thrown.
		 */
		Vector( const Vector< D, BSP1D, C > & x ) : Vector( internal::grb_BSP1D.cload(), size( x ) ) {
			const RC rc = set( *this, x );
			if( rc != SUCCESS ) {
				throw std::runtime_error( "grb::set inside copy-constructor: " + toString( rc ) );
			}
		}

		/**
		 * Move constructor. This is a \f$ \Theta(1) \f$ operation.
		 *
		 * No implementation remarks.
		 *
		 * @see grb::Vector for the user-level specfication.
		 */
		Vector( Vector< D, BSP1D, C > && x ) noexcept :
			_raw( x._raw ), _assigned( x._assigned ), _buffer( x._buffer ), _local_n( x._local_n ), _offset( x._offset ), _n( x._n ), _nnz( x._nnz ), _raw_slot( x._raw_slot ),
			_assigned_slot( x._assigned_slot ), _stack_slot( x._stack_slot ), _cleared( x._cleared ), _became_dense( x._became_dense ), _nnz_is_dirty( x._nnz_is_dirty ),
			_global_is_dirty( x._global_is_dirty ) {
			_local = std::move( x._local );
			_global = std::move( x._global );
			_raw_deleter = std::move( x._raw_deleter );
			_assigned_deleter = std::move( x._assigned_deleter );
			_buffer_deleter = std::move( x._buffer_deleter );

			// invalidate fields of x
			x._raw = NULL;
			x._assigned = NULL;
			x._buffer = NULL;
			// local and global have been invalidated by std::move
			x._local_n = x._offset = x._n = x._nnz = 0;
			x._raw_slot = x._assigned_slot = x._stack_slot = LPF_INVALID_MEMSLOT;
			x._cleared = x._became_dense = x._nnz_is_dirty = x._global_is_dirty = false;
			// deleters have been invalidated by std::move

			// done
		}

		/**
		 * Assign-from-temporary. This is a \f$ \Theta(1) \f$ operation.
		 *
		 * No implementation remarks.
		 *
		 * @see grb::Vector for the user-level specfication.
		 */
		Vector< D, BSP1D, C > & operator=( Vector< D, BSP1D, C > && x ) noexcept {
			// move all fields from x to our instance
			_raw = x._raw;
			_assigned = x._assigned;
			_buffer = x._buffer;
			_local = std::move( x._local );
			_global = std::move( x._global );
			_local_n = x._local_n;
			_offset = x._offset;
			_n = x._n;
			_nnz = x._nnz;
			_raw_slot = x._raw_slot;
			_assigned_slot = x._assigned_slot;
			_stack_slot = x._stack_slot;
			_cleared = x._cleared;
			_became_dense = x._became_dense;
			_nnz_is_dirty = x._nnz_is_dirty;
			_global_is_dirty = x._global_is_dirty;
			_raw_deleter = std::move( x._raw_deleter );
			_assigned_deleter = std::move( x._assigned_deleter );
			_buffer_deleter = std::move( x._buffer_deleter );

			// invalidate fields of x
			x._raw = NULL;
			x._assigned = NULL;
			x._buffer = NULL;
			// local and global have been invalidated by std::move
			x._local_n = x._offset = x._n = x._nnz = 0;
			x._raw_slot = x._assigned_slot = x._stack_slot = LPF_INVALID_MEMSLOT;
			x._cleared = x._became_dense = x._nnz_is_dirty = x._global_is_dirty = false;
			// deleters have been invalidated by std::move

			// done
			return *this;
		}

		/** Base destructor. */
		~Vector() {
			// get thread-local store
			auto & data = internal::grb_BSP1D.load();
#ifdef _DEBUG
			std::cout << data.s << ", Vector< BSP1D >::~Vector< BSP1D > called.\n";
#endif
			// if GraphBLAS is currently still initialised
			if( ! data.destroyed ) {
				// then do bookkeeping; deregister memslot
				lpf_err_t rc = LPF_SUCCESS;
				if( _raw_slot != LPF_INVALID_MEMSLOT ) {
#ifdef _DEBUG
					std::cout << "\t" << data.s << ", deregistering value array @ " << _raw << ", slot #" << _raw_slot << "...\n";
#endif
					rc = lpf_deregister( data.context, _raw_slot );
					assert( rc == LPF_SUCCESS );
					if( rc == LPF_SUCCESS ) {
						data.signalMemslotReleased();
					}
				}
				if( _assigned_slot != LPF_INVALID_MEMSLOT ) {
#ifdef _DEBUG
					std::cout << "\t" << data.s << ", deregistering assigned array @ " << _assigned << ", slot #" << _assigned_slot << "...\n";
#endif
					rc = lpf_deregister( data.context, _assigned_slot );
					assert( rc == LPF_SUCCESS );
					if( rc == LPF_SUCCESS ) {
						data.signalMemslotReleased();
					}
				}
				if( _stack_slot != LPF_INVALID_MEMSLOT ) {
#ifdef _DEBUG
					std::cout << "\t" << data.s << ", deregistering stack array, slot #" << _stack_slot << "...\n";
#endif
					rc = lpf_deregister( data.context, _stack_slot );
					assert( rc == LPF_SUCCESS );
					if( rc == LPF_SUCCESS ) {
						data.signalMemslotReleased();
					}
				}
			}
#ifdef _DEBUG
			std::cout << "\t" << data.s << ", GraphBLAS vector at ( " << _raw << ", " << _assigned << " ) destroyed.\n";
			std::cout << data.s << ", Vector< BSP1D >::~Vector< BSP1D > done.\n";
#endif
			// note that the free of _raw and _assigned is handled by there AutoDeleters.
		}

		/**
		 * No implementation remarks.
		 *
		 * @see Vector::cbegin
		 */
		const_iterator cbegin() const {
			const auto data = internal::grb_BSP1D.cload();
			return _local.template cbegin< BSP1D >( data.s, data.P );
		}

		/**
		 * No implementation remarks.
		 *
		 * @see Vector::begin
		 */
		const_iterator begin() const {
			return cbegin();
		}

		/**
		 * No implementation remarks.
		 *
		 * @see Vector::cend
		 */
		const_iterator cend() const {
			const auto data = internal::grb_BSP1D.cload();
			return _local.template cend< BSP1D >( data.s, data.P );
		}

		/**
		 * No implementation remarks.
		 *
		 * @see Vector::end
		 */
		const_iterator end() const {
			return cend();
		}

		/**
		 * Implementation simply defers to the reference implementation operator
		 * overload. This means this function expects local indices, which happens
		 * automatically when using eWiseLambda.
		 */
		typename LocalVector::lambda_reference operator[]( const size_t i ) {
			// return reference
			return _local[ i ];
		}

		/** No implementation notes (see above). */
		const typename LocalVector::lambda_reference operator[]( const size_t i ) const {
			// return const reference
			return _local[ i ];
		}

		/**
		 * \internal Returns a raw handle to the process-local memory.
		 * \warning for debugging purposes only!
		 */
		D * raw() {
			return _local.raw();
		}
	};

	// template specialisation for GraphBLAS type traits
	template< typename D, typename Coords >
	struct is_container< Vector< D, BSP1D, Coords > > {
		/** A BSP1D vector is a GraphBLAS object. */
		static const constexpr bool value = true;
	};

	namespace internal {

		// documentation is in forward declaration
		template< typename DataType, typename Coords >
		void signalLocalChange( Vector< DataType, BSP1D, Coords > & x ) {
			x._global_is_dirty = true;
			x._nnz_is_dirty = true;
		}

	} // namespace internal

} // namespace grb

#endif

