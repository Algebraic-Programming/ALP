
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
 * @date 5th of July, 2017
 */

#if !defined _H_GRB_REFERENCE_COORDINATES || defined _H_GRB_REFERENCE_OMP_COORDINATES
#define _H_GRB_REFERENCE_COORDINATES

#include <stddef.h> //size_t

#include <algorithm> // std::max
#include <stdexcept> // std::runtime_error

#include <assert.h>
#include <string.h> // memcpy

#include <graphblas/backends.hpp>
#include <graphblas/base/coordinates.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/utils.hpp>

#include "config.hpp"

#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
 #include <omp.h>
#endif

#if defined _DEBUG && ! defined NDEBUG
 #include <set>
#endif


namespace grb {

	namespace internal {

		/**
		 * This class encapsulates everything needed to store a sparse set of 1D
		 * coordinates. Its use is internal via, e.g., grb::Vector< T, reference, C >.
		 * All functions needed to rebuild or update sparsity information are
		 * encapsulated here.
		 */
		template<>
		class Coordinates< reference > {

			public:

				/** The type of elements #saveFromStack returns. */
				typedef typename config::VectorIndexType StackType;

				/** Local update type for use with #asyncAssign and #joinAssign. */
				typedef StackType * Update;

				/** The type of elements #saveFromArray returns. */
				typedef bool ArrayType;


			private:

				/** Pointer to the underlying indexing array. */
				bool * __restrict__ _assigned;

				/**
				 * Stack of assigned coordinates.
				 * This array will be P larger than required, so that the overflow can act
				 * as a buffer of size P required by #rebuildSparsity.
				 */
				StackType * __restrict__ _stack;

				/**
				 * A buffer required for parallel updates.
				 */
				StackType * __restrict__ _buffer;

				/** Size of the this vector (in number of elements). */
				size_t _n;

				/** Capacity of this vector (in number of elements). */
				size_t _cap;

				/** Per-thread capacity for parallel stack updates. */
				size_t _buf;

				/**
				 * Increments the number of nonzeroes in the current thread-local stack.
				 *
				 * @param[in,out] update Which stack will have a new nonzero added in.
				 *
				 * @returns The number of nonzeroes prior to the increment.
				 */
				inline StackType incrementUpdate( Update &update ) {
					return ++( update[ 0 ] );
				}

				/**
				 * Empties a thread-local stack.
				 *
				 * @param[in,out] update The stack to be reset.
				 *
				 * @returns The old number of elements in the stack.
				 */
				inline StackType resetUpdate( Update &update ) {
					const StackType ret = update[ 0 ];
					update[ 0 ] = 0;
					return ret;
				}

#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
				/**
				 * Retrieves a thread-local stack from the buffer.
				 *
				 * @param[out] update Which thread-local update to set.
				 */
				inline Update getLocalUpdate() {
					assert( _buffer != nullptr || _cap == 0 );
					const int tid = omp_get_thread_num();
					const size_t bs = _buf / omp_get_num_threads();
					Update update = _buffer;
					update += tid * bs;
					return update;
				}
#endif

				/**
				 * Shared header function for the set, set_seq, and set_ompPar functions.
				 *
				 * \warning This function does not set _buf
				 */
				void set_shared_header(
					void * const arr, void * const buf, const size_t dim
				) noexcept {
					// catch trivial case
					if( arr == nullptr || buf == nullptr ) {
						assert( arr == nullptr );
						assert( buf == nullptr );
						assert( dim == 0 );
						_assigned = nullptr;
						_stack = nullptr;
						_buffer = nullptr;
						_n = 0;
						_cap = 0;
						_buf = 0;
						return;
					}

					// _assigned has no alignment issues, take directly from input buffer
					assert( reinterpret_cast< uintptr_t >( _assigned ) % sizeof( bool ) == 0 );
					_assigned = static_cast< bool * >( arr );
					// ...but _stack does have potential alignment issues:
					char * buf_raw = static_cast< char * >( buf );
					constexpr const size_t size = sizeof( StackType );
					const size_t mod = reinterpret_cast< uintptr_t >( buf_raw ) % size;
					if( mod != 0 ) {
						buf_raw += size - mod;
					}
					_stack = reinterpret_cast< StackType * >( buf_raw );
					// no alignment issues between stack and buffer, so just shift by dim:
					_buffer = _stack + dim;
					// initialise
					_n = 0;
					_cap = dim;
				}

				/**
				 * Shared inner-most code for the set, set_seq, and set_ompPar functions.
				 *
				 * Sets the assigned array to false within the given start and end bounds.
				 */
				inline void set_kernel( const size_t start, const size_t end ) noexcept {
					// initialise _assigned only if necessary
					if( _cap > 0 && start < end ) {
						for( size_t i = start; i < end; ++i ) {
							_assigned[ i ] = false;
						}
					}
				}

				inline void clear_header() noexcept {
#ifndef NDEBUG
					if( _n == _cap && _assigned == nullptr && _cap > 0 ) {
						const bool dense_coordinates_may_not_call_clear = false;
						assert( dense_coordinates_may_not_call_clear );
					}
#endif
				}

				inline void clear_oh_n_kernel( const size_t start, const size_t end ) noexcept {
					for( size_t i = start; i < end; ++i ) {
						_assigned[ i ] = false;
					}
				}

#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
				void clear_oh_n_ompPar() noexcept {
					size_t start, end;
					config::OMP::localRange( start, end, 0, _cap );
					clear_oh_n_kernel( start, end );
				}

				void clear_oh_n_omp() noexcept {
					if( _cap < config::OMP::minLoopSize() ) {
						clear_oh_n_kernel( 0, _cap );
					} else {
						const size_t nblocks = _cap / config::CACHE_LINE_SIZE::value();
						const size_t nthreads = std::min(
								config::OMP::threads(),
								std::max(
									static_cast< size_t >( 1 ),
									(_cap % config::CACHE_LINE_SIZE::value() == 0) ? nblocks : nblocks + 1
								)
							);
						#pragma omp parallel num_threads( nthreads )
						{
							clear_oh_n_ompPar();
						}
					}
				}
#endif

				inline void clear_oh_nz_seq() noexcept {
					for( size_t k = 0; k < _n; ++k ) {
						_assigned[ _stack[ k ] ] = false;
					}
				}

#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
				inline void clear_oh_nz_ompPar() noexcept {
					// dynamic schedule since performance may differ significantly depending
					// on the un-orderedness of the _stack
					#pragma omp for schedule( dynamic, config::CACHE_LINE_SIZE::value() )
					for( size_t k = 0; k < _n; ++k ) {
						_assigned[ _stack[ k ] ] = false;
					}
				}

				void clear_oh_nz_omp() noexcept {
					if( _n < config::OMP::minLoopSize() ) {
						clear_oh_nz_seq();
					} else {
						// use a simple analytic model to determine nthreads
						const size_t bsize = _n / config::CACHE_LINE_SIZE::value();
						const size_t nthreads = std::min(
								config::OMP::threads(),
								std::max(
									static_cast< size_t >( 1 ),
									(_n % config::CACHE_LINE_SIZE::value() == 0)
										? bsize
										: bsize + 1
								)
							);
						#pragma omp parallel num_threads( nthreads )
						{
							clear_oh_nz_ompPar();
						}
					}
				}
#endif


			public:

				/**
				 * Computes the required size of an array, in bytes, to store a nonzero
				 * structure of a given size.
				 *
				 * @param[in] dim The nonzero array size.
				 *
				 * @returns The required size, in bytes.
				 *
				 * @see #set.
				 */
				static inline size_t arraySize( const size_t dim ) noexcept {
					if( dim == 0 ) {
						return 0;
					}
					return ( dim + 1 ) * sizeof( ArrayType );
				}

				/**
				 * Computes the maximum stack size, in bytes.
				 *
				 * @param[in] dim The size of the coordinate instance.
				 *
				 * @returns The requested maximum stack size.
				 */
				static inline size_t stackSize( const size_t dim ) noexcept {
					if( dim == 0 ) {
						return 0;
					}
					return ( dim + 1 ) * sizeof( StackType );
				}

				/**
				 * @returns Computes the buffer size required to perform a parallel
				 *          prefix-sum.
				 */
				static inline size_t prefixbufSize() noexcept {
					int P = 1;
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
					P = config::OMP::threads();
#endif
					return ( P + 1 ) * sizeof( StackType );
				}

				/**
				 * Computes the parallel update buffer size, in bytes.
				 *
				 * @param[in] n The size of the coordinate instance.
				 *
				 * @returns The requested buffer size.
				 */
				static inline size_t parbufSize( const size_t n ) noexcept {
					return config::IMPLEMENTATION< reference >::vectorBufferSize( n,
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						   config::OMP::threads()
#else
						   1
#endif
					) * sizeof( StackType );
				}

				/**
				 * Computes the required size of a buffer, in bytes, to store a nonzero
				 * structure of a given size.
				 *
				 * @param[in] dim The nonzero array size.
				 *
				 * @returns The required size, in bytes.
				 *
				 * @see #set.
				 */
				static inline size_t bufferSize( const size_t dim ) noexcept {
					// buffer should at least contain space for:
					//    1. the stack
					//    2. parallel updates to that stack
					//    3. parallel prefix sums over stack sizes
					size_t ret = stackSize( dim );
					ret += parbufSize( dim );
					ret += prefixbufSize();
					return ret;
				}

				/** Base constructor. Creates an empty coordinates list of dimension 0. */
				inline Coordinates() noexcept :
					_assigned( nullptr ), _stack( nullptr ), _buffer( nullptr ),
					_n( 0 ), _cap( 0 ), _buf( 0 )
				{}

				/**
				 * Move constructor.
				 * @param[in] x The Coordinates instance to move contents from.
				 */
				inline Coordinates( Coordinates &&x ) noexcept :
					_assigned( x._assigned ), _stack( x._stack ), _buffer( x._buffer ),
					_n( x._n ), _cap( x._cap ), _buf( x._buf )
				{
					x._assigned = nullptr;
					x._stack = nullptr;
					x._buffer = nullptr;
					x._n = x._cap = x._buf = 0;
				}

				/**
				 * \internal
				 * Shallow copy constructor for use with #PinnedVector.
				 *
				 * This is for internal use only.
				 * \endinternal
				 */
				inline Coordinates( const Coordinates &x ) noexcept :
					_assigned( x._assigned ), _stack( x._stack ), _buffer( x._buffer ),
					_n( x._n ), _cap( x._cap ), _buf( x._buf )
				{
					// self-assignment is a programming error
					assert( this != &x );
				}

				/**
				 * \internal
				 * Follows the above shallow copy constructor.
				 *
				 * This is for internal use only.
				 * \endinternal
				 */
				inline Coordinates & operator=( const Coordinates &other ) {
					Coordinates replace( other );
					*this = std::move( replace );
					return *this;
				}

				/**
				 * Assign from temporary.
				 */
				inline Coordinates & operator=( Coordinates &&x ) noexcept {
					assert( this != &x );
					_assigned = x._assigned;
					_stack = x._stack;
					_buffer = x._buffer;
					_n = x._n;
					_cap = x._cap;
					_buf = x._buf;
					x._assigned = NULL;
					x._stack = x._buffer = NULL;
					x._n = x._cap = x._buf = 0;
					return *this;
				}

				/**
				 * Base destructor.
				 */
				inline ~Coordinates() noexcept {
					// done (the #_assigned and #_stack memory
					// blocks are not managed by this class)
				}

				/**
				 * @returns An empty thread-local stack for new nonzeroes.
				 */
				inline Update EMPTY_UPDATE() {
					if( _assigned == nullptr && _cap > 0 && _cap == _n ) {
						return nullptr;
					}
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
					Update ret = getLocalUpdate();
					ret[ 0 ] = 0;
					return ret;
#else
					return nullptr;
#endif
				}

				/**
				 * Sets the data structure. A call to this function sets the number of
				 * coordinates to zero.
				 *
				 * @param[in] arr Pointer to an array of size #arraySize. This array is
				 *                is managed by a container outside this class (and thus
				 *                will not be free'd on destruction of this instance).
				 * @param[in] arr_initialized Whether the memory pointed to by \a arr has
				 *                            already been initialized (i.e., has each of
				 *                            its entries set to <tt>false</tt>).
				 * @param[in] buf Pointer to an array of size #bufferSize. This array is
				 *                managed by a container outside this class (and thus will
				 *                not be freed on destruction of this instance).
				 * @param[in] dim Size (dimension) of this vector, in number of elements.
				 *
				 * The memory area \a raw will be reset to reflect an empty coordinate set.
				 *
				 * \note \a raw may be larger than \a storageSize, but must not be smaller
				 *       or undefined behaviour may occur.
				 *
				 * This function may be called on instances with any state. After a correct
				 * call to this function, the state shall become valid.
				 */
				void set(
					void * const arr, bool arr_initialized,
					void * const buf, const size_t dim
				) noexcept {
					set_shared_header( arr, buf, dim );
					_buf = config::IMPLEMENTATION< reference >::vectorBufferSize(
						_cap,
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						config::OMP::threads()
#else
						1
#endif
					);
					if( arr_initialized ) { return; }
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
					#pragma omp parallel
					{
						size_t start, end;
						config::OMP::localRange( start, end, 0, dim );
#else
						const size_t start = 0;
						const size_t end = dim;
#endif
						set_kernel( start, end );
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
					}
#endif
				}

				/**
				 * Sets the data structure.
				 *
				 * This variant of #set assumes this instance will only ever be used by a
				 * single thread.
				 */
				void set_seq(
					void * const arr, bool arr_initialized,
					void * const buf, const size_t dim
				) noexcept {
					set_shared_header( arr, buf, dim );
					_buf = config::IMPLEMENTATION< reference >::vectorBufferSize( _cap, 1 );
					if( !arr_initialized ) {
						set_kernel( 0, dim );
					}
				}

#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
				/**
				 * Sets the data structure.
				 *
				 * This variant of #set assumes this instance will be called from within a
				 * parallel OMP section. It (thus) assumes the initialised instance may be
				 * referred to by multiple threads.
				 */
				void set_ompPar(
					void * const arr, bool arr_initialized,
					void * const buf, const size_t dim
				) noexcept {
					size_t start, end;
					config::OMP::localRange( start, end, 0, dim );
					#pragma omp single
					{
						set_shared_header( arr, buf, dim );
						_buf = config::IMPLEMENTATION< reference >::vectorBufferSize( _cap,
							config::OMP::threads() );
					}
					#pragma omp barrier
					if( !arr_initialized ) {
						set_kernel( start, end );
					}
				}
#endif

				/**
				 * Sets this data structure to a dummy placeholder for a dense structure.
				 *
				 * This structure will be immutable, and does not support the majority of
				 * operations this class defines; use dense coordinates with care.
				 */
				void setDense( const size_t dim ) noexcept {
					_assigned = nullptr;
					_stack = nullptr;
					_buffer = nullptr;
					_n = dim;
					_cap = dim;
					_buf = 0;
				}

				/**
				 * Rebuild nonzero data structure after this instance has become invalid.
				 *
				 * May not be called on dense instances.
				 *
				 * An instance of this type can have one out of four states: valid, invalid,
				 * assign-updated, or copy-updated. An invalid state only allows function
				 * calls to
				 *   -#clear (the instance will become valid once more)
				 *   -#clearRange (the instance will become valid once more)
				 *   -#rebuildSparsity (the instance will become valid once more)
				 *   -#set (the instance will become valid once more)
				 * An instance becomes invalid by a call to #set, #clearRange, or by any
				 * auxiliary event changing the contents of the #_assigned array.
				 *
				 * A call to this function will update the mechanisms that control
				 * sparsity, such as the nonzero count #_n and the nonzero stack #_stack
				 *     <em>according to the #_assigned array</em>;
				 * i.e., the #_assigned array is assumed to be correct, and all other
				 * fields will be updated accordingly.
				 *
				 * This function completes in \f$ \Theta( n ) \f$ time, where \a n is the
				 * dimension of this vector. It moves \a n booleans and two elements of
				 * \a size_t of data.
				 *
				 * @param[in] dense Whether the vector is dense. In this case, the net
				 *                  effect is that #_n is set to #_cap; in particular,
				 *                  \a nonzero_count_only will be ignored.
				 *
				 * @see joinUpdate To turn an instance with assign-updated state back into a
				 *                 valid state.
				 * @see joinCopy   To turn an instance with copy-update state back into a
				 *                 valid state.
				 */
				void rebuild( const bool dense ) noexcept {
#ifdef _DEBUG
					std::cout << "Coordinates::rebuild called with dense = " << dense << "\n";
#endif
					// catch most trivial case: a vector of dimension 0 (empty vector)
					if( _cap == 0 ) {
						return;
					}
#ifndef NDEBUG
					if( _assigned == nullptr && _n == _cap ) {
						const bool dense_coordinates_may_not_call_rebuild = false;
						assert( dense_coordinates_may_not_call_rebuild );
					}
					assert( _assigned != nullptr );
#endif
					// catch the other trivial-ish case (since can delegate)
					if( dense && _n != _cap ) {
#ifdef _DEBUG
						std::cout << "rebuildSparsity: dense case\n";
#endif
						assignAll();
						return;
					}

					StackType * __restrict__ counts = nullptr;

#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
					#pragma omp parallel
#endif
					{
#ifdef _DEBUG
 #ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						#pragma omp single
 #endif
						std::cout << "rebuildSparsity: sparse case\n";
#endif
						// sparse update. Re-count unassigned array.
						size_t local_count = 0;
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						const size_t P = static_cast< size_t >( omp_get_num_threads() );
						const size_t s = static_cast< size_t >( omp_get_thread_num() );
						#pragma omp single
						counts = _buffer;
#else
						const size_t P = 1;
						const size_t s = 0;
						counts = _buffer;
#endif
						size_t start, end;
						config::OMP::localRange( start, end, 0, _cap );
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						assert( start <= end );
						assert( end <= _cap );
#endif
#ifdef _DEBUG
						std::cout << "Coordinates::rebuild: thread " << s << " has range "
							<< start << "--" << end << "\n";
#endif
						// do recount interleaved with stack re-construction
						for( size_t i = start; i < end; ++i ) {
							if( _assigned[ i ] ) {
								(void) ++local_count;
							}
						}
						counts[ s ] = local_count;
#ifdef _DEBUG
 #ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						#pragma omp critical
 #endif
						std::cout << "Coordinates::rebuild: thread " << s << " found "
							<< local_count << " nonzeroes." << std::endl;
#endif

#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						#pragma omp barrier
						#pragma omp single
						{
#endif

#ifdef _DEBUG
							std::cout << "Coordinates::rebuild: thread 0 found " << counts[ 0 ]
								<< " nonzeroes.\n";
#endif
							for( size_t k = 1; k < P; ++k ) {
#ifdef _DEBUG
								std::cout << "Coordinates::rebuild: thread " << k << " found "
									<< counts[ k ] << " nonzeroes.\n";
#endif
								counts[ k ] += counts[ k - 1 ];
							}
							assert( counts[ P - 1 ] <= _cap );
							_n = counts[ P - 1 ];
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						}
						#pragma omp barrier
#endif
						local_count = ( s == 0 ) ? 0 : counts[ s - 1 ];
						for( size_t i = start; i < end; ++i ) {
							if( _assigned[ i ] ) {
								_stack[ local_count++ ] = i;
							}
						}
						assert( local_count == counts[ s ] );
#ifdef _DEBUG
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						#pragma omp single
#endif
						std::cout << "\tnew nonzero count: " << _n << "\n";
#endif
					} // end sparse version
				}

				/**
				 * Rebuilds the #_assigned array using the contents of #_stack.
				 *
				 * May not be called on dense instances.
				 *
				 * \warning Assumes that #_assigned is never set to \a true whenever the
				 * corresponding index does not appear in #_stack.
				 *
				 * This function should only be used after #_stack has been modified outside
				 * of this class.
				 *
				 * This variant performs a copy of a packed array of nonzero values into
				 * an unpacked array of nonzero values, on the fly.
				 *
				 * @tparam The nonzero type.
				 *
				 * @param[out] array_out An array of #_cap elements of nonzero values.
				 * @param[in]  packed_in An array of \a new_nz elements of nonzero values.
				 * @param[in]  new_nz    The number of nonzeroes in #_stack.
				 */
				template< typename DataType >
				RC rebuildFromStack(
					DataType * const array_out,
					const DataType * const packed_in,
					const size_t new_nz
				) {
					if( _assigned == nullptr && _cap > 0 && _n == _cap ) {
						std::cerr << "Coordinates< reference >::rebuildFromStack called from a "
							<< "dense coordinate instance!\n";
#ifndef NDEBUG
						const bool dense_coordinates_may_not_call_rebuildFromStack = false;
						assert( dense_coordinates_may_not_call_rebuildFromStack );
#endif
						return PANIC;
					}
#ifdef _DEBUG
					std::cout << "Entering Coordinates::rebuildFromStack (reference backend, non-void version). New stack count: " << new_nz << ".\n";
					std::cout << "\t stack contents: ( ";
					for( size_t k = 0; k < new_nz; ++k ) {
						std::cout << _stack[ k ] << " ";
					}
					std::cout << ")\n";
#endif
					assert( array_out != nullptr );
					assert( packed_in != nullptr );
					_n = new_nz;
#if defined _DEBUG && ! defined NDEBUG
					{
						// this is an extra and costly check only enabled with _DEBUG mode
						std::set< size_t > indices;
						for( size_t k = 0; k < _n; ++k ) {
							indices.insert( _stack[ k ] );
						}
						for( size_t i = 0; i < _cap; ++i ) {
							if( _assigned[ i ] ) {
								assert( indices.find( i ) != indices.end() );
							}
						}
					}
#endif
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
					#pragma omp parallel
					{
						size_t start, end;
						config::OMP::localRange( start, end, 0, _n );
#else
						const size_t start = 0;
						const size_t end = _n;
#endif
						for( size_t k = start; k < end; ++k ) {
							const size_t i = _stack[ k ];
#ifdef _DEBUG
 #ifdef _H_GRB_REFERENCE_OMP_COORDINATES
							#pragma omp critical
 #endif
							{
								std::cout << "\tProcessing global stack element " << k << " which has index " << i << "."
									<< " _assigned[ index ] = " << _assigned[ i ] << " and value[ index ] will be set to " << packed_in[ k ] << ".\n";
							}
#endif
							assert( i < _cap );
							_assigned[ i ] = true;
							array_out[ i ] = packed_in[ k ];
						}
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
					}
#endif
					return SUCCESS;
				}

				/**
				 * Rebuilds the #_assigned array using the contents of #_stack.
				 *
				 * May not be called on dense instances.
				 *
				 * \warning Assumes that #_assigned is never set to \a true whenever the
				 * corresponding index does not appear in #_stack.
				 *
				 * This function should only be used after #_stack has been modified outside
				 * of this class.
				 *
				 * This variant does not perform on the fly copies of packed into unpacked
				 * nonzero arrays. It does, however, employ the same interface as the version
				 * that does so as to simplify the life of callees.
				 *
				 * @param[in]  new_nz The number of nonzeroes in #_stack.
				 */
				RC rebuildFromStack( void * const, const void * const, const size_t new_nz ) {
					if( _assigned == nullptr && _cap > 0 && _n == _cap ) {
						std::cerr << "Coordinates< reference >::rebuildFromStack called from a "
							<< "dense coordinate instance!\n";
#ifndef NDEBUG
						const bool dense_coordinates_may_not_call_rebuildFromStack = false;
						assert( dense_coordinates_may_not_call_rebuildFromStack );
#endif
						return PANIC;
					}
#if defined _DEBUG && ! defined NDEBUG
					{
						// this is an extra and costly check only enabled with _DEBUG mode
						std::set< size_t > indices;
						for( size_t k = 0; k < _n; ++k ) {
							indices.insert( _stack[ k ] );
						}
						for( size_t i = 0; i < _cap; ++i ) {
							if( _assigned[ i ] ) {
								assert( indices.find( i ) != indices.end() );
							}
						}
					}
#endif
					_n = new_nz;
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
					#pragma omp parallel
					{
						size_t start, end;
						config::OMP::localRange( start, end, 0, _n );
#else
						const size_t start = 0;
						const size_t end = _n;
#endif
						for( size_t k = start; k < end; ++k ) {
							const size_t i = _stack[ k ];
							assert( i < _cap );
							_assigned[ i ] = true;
						}
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
					}
#endif
					return SUCCESS;
				}

				/**
				 * Packs nonzero indices and nonzero values into an external stack and
				 * packed array, respectively.
				 *
				 * @tparam DataType The nonzero type.
				 *
				 * @param[out] stack_out  An array of length #_n that on output will be a
				 *                        copy of #_stack.
				 * @param[in]  offset     An offset to be applied to the entries of
				 *                        \a stack_out.
				 * @param[out] packed_out An array of length #_n that on output will contain
				 *                        those nonzeroes corresponding to the indices in
				 *                        \a stack_out, in matching order.
				 * @param[in]  array_in   An array of size #_cap from which contains the
				 *                        nonzero values pointed to by #_stack.
				 */
				template< typename DataType >
				RC packValues(
					StackType * const stack_out, const size_t offset,
					DataType * const packed_out,
					const DataType * const array_in
				) const {
#ifdef _DEBUG
					std::cout << "Called Coordinates::packValues (reference backend, non-void version)\n";
#endif
					assert( stack_out != nullptr );
					assert( packed_out != nullptr );
					assert( array_in != nullptr );
					if( _n == _cap ) {
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						#pragma omp parallel
						{
							size_t start, end;
							config::OMP::localRange( start, end, 0, _cap );
#else
							const size_t start = 0;
							const size_t end = _cap;
#endif
							for( size_t i = start; i < end; ++i ) {
								stack_out[ i ] = i + offset;
								packed_out[ i ] = array_in[ i ];
#ifdef _DEBUG
 #ifdef _H_GRB_REFERENCE_OMP_COORDINATES
								#pragma omp critical
 #endif
								{
									std::cout << "\t packing local index " << i << " into global index "
										<< stack_out[ i ] << " with nonzero " << packed_out[ i ] << "\n";
								}
#endif
							}
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						}
#endif
					} else {
#ifndef NDEBUG
						if( _assigned == nullptr ) {
							const bool a_dense_coordinate_instance_should_not_reach_this_point =
								false;
							assert( a_dense_coordinate_instance_should_not_reach_this_point );
						}
						assert( _stack != nullptr );
#endif
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						#pragma omp parallel
						{
							size_t start, end;
							config::OMP::localRange( start, end, 0, _n );
#else
							const size_t start = 0;
							const size_t end = _n;
#endif
							for( size_t k = start; k < end; ++ k ) {
								const size_t i = _stack[ k ];
								assert( i < _cap );
								stack_out[ k ] = i + offset;
								packed_out[ k ] = array_in[ i ];
#ifdef _DEBUG
 #ifdef _H_GRB_REFERENCE_OMP_COORDINATES
								#pragma omp critical
 #endif
								{
									std::cout << "\t packing local index " << i << " into global index "
										<< stack_out[ k ] << " with nonzero " << packed_out[ k ] << "\n";
								}
#endif
							}
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						}
#endif
					}
					return SUCCESS;
				}

				/**
				 * Packs nonzero indices into an external stack.
				 *
				 * @param[out] stack_out An array of length #_n that on output will be a
				 *                       copy of #_stack.
				 * @param[in]  offset    An offset to be applied to the entries of
				 *                       \a stack_out.
				 *
				 * The interface is the same as for the above variant to simplify the life
				 * of callees.
				 */
				RC packValues(
					StackType * const stack_out, const size_t offset,
					void * const, const void * const
				) const {
					assert( stack_out != nullptr );
					if( _n == _cap ) {
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						#pragma omp parallel
						{
							size_t start, end;
							config::OMP::localRange( start, end, 0, _cap );
#else
							const size_t start = 0;
							const size_t end = _cap;
#endif
							for( size_t i = start; i < end; ++i ) {
								stack_out[ i ] = i + offset;
							}
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						}
#endif
					} else {
#ifndef NDEBUG
						if( _assigned == nullptr ) {
							const bool a_dense_coordinate_instance_should_not_reach_this_point =
								false;
							assert( a_dense_coordinate_instance_should_not_reach_this_point );
						}
						assert( _stack != nullptr );
#endif
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						#pragma omp parallel
						{
							size_t start, end;
							config::OMP::localRange( start, end, 0, _n );
#else
							const size_t start = 0;
							const size_t end = _n;
#endif
							for( size_t k = start; k < end; ++ k ) {
								const size_t i = _stack[ k ];
								assert( i < _cap );
								stack_out[ k ] = i + offset;
							}
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						}
#endif
					}
					return SUCCESS;
				}

				/**
				 * Sets this coordinate set to match that of a subset of itself. All entries
				 * not in the subset are removed from this instance. The subset must be
				 * contiguous.
				 *
				 * May not be called on dense instances.
				 *
				 * @tparam dense       Set to \a true if and only if it is known beforehand
				 *                     that \a localSparsity is dense. (This does \em not
				 *                     mean this coordinate set will become dense!)
				 *
				 * @param[in] localSparsity The subset coordinate set.
				 * @param[in] offset        Where the overlap between this instance and that
				 *                          of \a localSparsity starts. This means that for
				 *                          all \a i in range of \a localSparsity that
				 *                          \f$ \mathit{\_assigned} + i+ \mathit{offset} =
				 *                              \mathit{localSparsity.\_assigned} + i \f$
				 *
				 * This is useful if this instances' _assigned array's contents are only
				 * correct for the subset in \a localSparsity, for example when this user
				 * process only has ownership over that particular range.
				 */
				template< bool dense >
				void rebuildGlobalSparsity(
					const Coordinates &localSparsity,
					const size_t offset
				) noexcept {
#ifndef NDEBUG
					if( _assigned == nullptr && _cap > 0 && _n == _cap ) {
						const bool dense_coordinates_may_not_call_rebuildGlobalSparsity = false;
						assert( dense_coordinates_may_not_call_rebuildGlobalSparsity );
					}
#endif
#ifdef _DEBUG
					std::cout << "rebuildGlobalSparsity called with ";
					if( dense ) { std::cout << "a dense local coordinate structure "; }
					else { std::cout << "a possibly sparse local coordinate structure "; }
					std::cout << "at offset " << offset << "\n";
#endif
					assert( localSparsity._cap <= _cap );
					// if dense, do a direct assign of our local structures
					if( dense || localSparsity.isDense() ) {
#ifdef _DEBUG
						if( !dense ) { std::cout << "\t our possibly sparse local coordinates were found to be dense\n"; }
#endif
						assert( localSparsity._n == localSparsity._cap );
						// if we are dense ourselves, just memset everything and set our stack ourselves
						// this is a Theta(n) operation which touches exactly n data elements
						if( isDense() ) {
#ifdef _DEBUG
							std::cout << "\t We are dense ourselves\n";
#endif
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
							// is this not totally unnecessary if assuming our structure was cleared first,
							// and isn't that always the case making this branch therefore dead code?
							// internal issue #262
							#pragma omp parallel
							{
#endif
								size_t start, end;
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
								config::OMP::localRange( start, end, 0, offset );
#else
								start = 0;
								end = offset;
#endif
								for( size_t i = start; i < end; ++i ) {
									_assigned[ i ] = 0;
								}
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
								config::OMP::localRange(
									start, end,
									offset + localSparsity.size(), _cap
								);
#else
								start = offset + localSparsity.size();
								end = _cap;
#endif
								for( size_t i = start; i < end; ++i ) {
									_assigned[ i ] = 0;
								}
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
								config::OMP::localRange( start, end, 0, localSparsity._cap );
 #ifndef NDEBUG
								#pragma omp barrier
 #endif
#else
								start = 0;
								end = localSparsity._cap;
#endif
								for( size_t i = start; i < end; ++i ) {
									assert( _assigned[ i + offset ] );
									_stack[ i ] = i + offset;
								}
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
							}
#endif
							_n = localSparsity._cap;
							// done
							return;
						}
					}

#ifdef _DEBUG
					std::cout << "\t our local coordinates are sparse\n";
#endif
					// at this point we are either sparse or dense. When dense,
					// localCoordinates cannot be dense; otherwise the above code would have
					// kicked in. We handle this case first:
					if( isDense() ) {
#ifdef _DEBUG
						std::cout << "\t our own coordinates were dense\n";
#endif
						// this is an O(n) loop. It touches n+n/p data elements.
						// clear nonlocal elements
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						// is this not totally unnecessary if assuming our structure was cleared
						// first, and isn't that always the case making this branch therefore dead
						// code?
						// internal issue #262
						#pragma omp parallel
						{
#endif
							size_t start, end;
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
							config::OMP::localRange( start, end, 0, offset );
#else
							start = 0;
							end = offset;
#endif
							for( size_t i = start; i < end; ++i ) {
								_assigned[ i ] = 0;
							}
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
							config::OMP::localRange(
								start, end,
								offset + localSparsity.size(), _cap
							);
#else
							start = offset + localSparsity.size();
							end = _cap;
#endif
							for( size_t i = start; i < end; ++i ) {
								_assigned[ i ] = 0;
							}
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						}
#endif
					} else {
#ifdef _DEBUG
						std::cout << "\t our own sparsity structure was sparse\n";
#endif
						// we are sparse. Loop over our own nonzeroes and then use #_assigned
						// to determine if they're still there. This is a Theta(nnz)-sized loop
						// and touches 2nnz data elements.
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						#pragma omp parallel
#endif
						{
							size_t start = 0, end = _n;
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
							config::OMP::localRange( start, end, 0, _n );
#endif
							size_t k = start;
							while( k < end ) {
								const StackType i = _stack[ k ];
								if( _assigned[ i ] ) {
									// this nonzero is only valid if it is in the range of the
									// localCoordinates
									if( i >= offset && i < offset + localSparsity.size() ) {
										// all OK-- go check the next one
										(void)++k;
										continue;
									} else {
										// now an invalid nonzero
										_assigned[ i ] = false;
									}
								}
								// this nonzero has become invalid, ignore it
								(void)++k;
								// and continue the loop
							}
						}
					}
					// in both cases, we need to rebuild the stack. We copy it from
					// localCoordinates:
#ifdef _DEBUG
					std::cout << "\t rebuilding stack\n";
#endif
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
					#pragma omp parallel
#endif
					{
						size_t start = 0, end = localSparsity.nonzeroes();
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						config::OMP::localRange( start, end, 0, localSparsity.nonzeroes() );
#endif
						if( start < end ) {
							size_t local_n = start;
							for( ; local_n < end; ++local_n ) {
								_stack[ local_n ] = localSparsity.index( local_n ) + offset;
							}
							assert( local_n == end );
						}
					}
					_n = localSparsity.nonzeroes();
#ifdef _DEBUG
					std::cout << "\t final debug-mode sanity check on output stack before "
						<< "exit...";
					for( size_t i = 0; i < _n; ++i ) {
						assert( _stack[ i ] < _cap );
						assert( _assigned[ _stack[ i ] ] );
						assert( _stack[ i ] == localSparsity.index( i ) + offset );
					}
					std::cout << "done\n";
#endif
					// done
				}

				/**
				 * Set the given coordinate to nonzero.
				 *
				 * @returns \a true if and only if the given coordinate already held a
				 *          nonzero.
				 *
				 * \note Thus, if no previous nonzero existed at the given coordinate, this
				 *       function returns \a false.
				 *
				 * This function is \em not thread safe.
				 *
				 * This function may only be called on instances with valid state.
				 */
				inline bool assign( const size_t i ) noexcept {
					if( _n == _cap ) {
						return true;
					}
					if( !_assigned[ i ] ) {
						_assigned[ i ] = true;
						const size_t newSize = _n + 1;
						assert( _n <= _cap );
						assert( newSize <= _cap );
						_stack[ _n ] = i;
						_n = newSize;
						return false;
					} else {
						return true;
					}
				}

				/**
				 * Signals all coordinates are now taken.
				 *
				 * If the instance is valid and the coordinate set was already dense, then
				 * a call to this function has no effect.
				 * Otherwise, this is an \$f \mathcal{O}(n) \f$ operation, where \a n is the
				 * vector capacity.
				 *
				 * @tparam maybe_invalid Indicates the container may be in an invalid state
				 *                       which is to be restored via this call. This will
				 *                       then be an \f$ Theta(n) \f$ operation regardless of
				 *                       density.
				 */
				template< bool maybe_invalid = false >
				inline void assignAll() noexcept {
					if( maybe_invalid || _n != _cap ) {
						if( _assigned != nullptr ) {
							assert( _stack != nullptr );
							assert( maybe_invalid || _n < _cap );
							assert( !maybe_invalid || _n <= _cap );
							_n = _cap;
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
							#pragma omp parallel
							{
								size_t start, end;
								config::OMP::localRange( start, end, 0, _n );
#else
								const size_t start = 0;
								const size_t end = _n;
#endif
								for( size_t i = start; i < end; ++i ) {
									_assigned[ i ] = true;
									_stack[ i ] = i;
								}
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
							}
#endif
						}
					}
				}

				/**
				 * @returns How many asynchronous assignments a single thread is guaranteed
				 *          to be able to push without synchronisation.
				 */
				inline size_t maxAsyncAssigns() {
					if( _assigned == nullptr && _cap > 0 && _cap == _n ) {
						return 0;
					}
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
					const int T = omp_get_num_threads();
					assert( _buf % T == 0 );
					return _buf / T - 1;
#else
					return _cap;
#endif
				}

				/**
				 * Set the given coordinate to nonzero but keep a local count on the number
				 * of added nonzeroes. This count can be separately made effective by
				 * calling #joinAssign. This is useful for parallel updates to the
				 * underlying data structure.
				 *
				 * \note In other words, this function is thread-safe.
				 *
				 * After a call to this function, this instance enters the assign-updated
				 * state. It can only be followed by more calls to #asyncAssign, or a call
				 * to #joinAssign. A call to the latter will restore this instance's state
				 * to valid once more. Both calls have to use the same \a localUpdate.
				 *
				 * @param[in] i The index to add to the current coordinate set.
				 * @param[in] localUpdate An instance of #Update that is local to the
				 *                        calling thread.
				 *
				 * @returns \a true if and only if the given coordinate already held a
				 *          nonzero.
				 *
				 * \note #Update can initially be set via a call to #EMPTY_UPDATE() which
				 *       creates initial `empty' updates.
				 *
				 * The same instance of #Update can be passed multiple times to this
				 * function, as long as this is done by a single thread per instance only.
				 * Updates are consumed via calls to #joinAssign. Only after all instances
				 * that were passed to concurrent calls to this function have been joined
				 * via #joinAssign does it become legal to use any other function of this
				 * class.
				 *
				 * It is not allowed that multiple threads issue a call to this function
				 * with equal values for the parameter \a i.
				 *
				 * After a call to this function, calling any other member function from
				 * this class will result in undefined behaviour, except for the following
				 * two:
				 *   -# #asyncAssign (calls may occur concurrently).
				 *   -# #joinAssign (calls may \em not occur concurrently).
				 * It will become legal again to call other member functions only if the
				 * conditions described in #joinAssign are met: that is, all threads that
				 * called this function must have called #joinAssign.
				 * Only then shall it be legal to call any other member function of this
				 * instance. Calling other member functions without adhering to the above
				 * will result in undefined behaviour.
				 *
				 * This function may only be called on instances with valid or
				 * assigned-updated state.
				 *
				 * \note Calling this function on an already-dense vector will have no
				 *       effect, as can also be inferred from the above description.
				 */
				inline bool asyncAssign( const size_t i, Update &localUpdate ) noexcept {
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
					// if dense, then all assigns are trivial
					if( _n == _cap ) {
						return true;
					}
					// otherwise, check sparsity
					if( !_assigned[ i ] ) {
						_assigned[ i ] = true;
						const size_t localPos = incrementUpdate( localUpdate );
						assert( localPos - 1 < maxAsyncAssigns() );
						assert( localUpdate[ 0 ] <= maxAsyncAssigns() );
						localUpdate[ localPos ] = i;
						return false;
					} else {
						return true;
					}
#else
					(void) localUpdate;
					return assign( i );
#endif
				}

				/**
				 * Consumes an instance of #Update. After the call to this function the
				 * given instance will be as though it was newly constructed (i.e., any
				 * updates recorded via #asyncAssign are deleted)-- i.e., \a update will
				 * will be empty once more.
				 *
				 * Once all instances of #Update derived from the same #Coordinates class
				 * are empty, it becomes valid to make a call to #rebuildSparsity, next to
				 * even more calls to #asyncAssign and #joinAssign.
				 * After a next call to #rebuildSparsity, all other member functions are
				 * legal to call again. After a next call to #asyncAssign, a call to
				 * #rebuildSparsity shall again become illegal until such time all updates
				 * are again consumed by a call to this function.
				 *
				 * If one thread makes a call to this function, any other concurrent threads
				 * that an update construct derived from the same #Coordinates class must
				 * make a matching call to this function.
				 *
				 * Any one thread may only have one \a update instance.
				 *
				 * @param[in] update The update to consume.
				 *
				 * \warning The contents of the updates are not verified, even when compiled
				 *          in debug mode.
				 *
				 * This function may only be called on instances with assign-updated state.
				 *
				 * @returns Whether \a update was empty for all threads.
				 */
				inline bool joinUpdate( Update &update ) noexcept {
					if( _assigned == nullptr && _cap > 0 && _cap == _n ) {
						return true;
					}

#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
					const int t = omp_get_thread_num();
					const int T = omp_get_num_threads();
#ifdef _DEBUG
					#pragma omp critical
					std::cout << t << ": joinUpdate called. Has " << _n << " nonzeroes on "
						<< "entry. Thread-local stack has " << update[ 0 ] << " nonzeroes.\n";
#endif
					StackType * __restrict__ const pfBuf = _buffer +
						config::IMPLEMENTATION< reference >::vectorBufferSize( _cap, T );

					// reset the thread-local stack and get current number of elements
					const size_t elements = resetUpdate( update );
					pfBuf[ t ] = elements;
#ifdef _DEBUG
					#pragma omp critical
					std::cout << "\t Thread " << t << " has " << elements << " elements\n";
#endif

					#pragma omp barrier

					// compute prefix sum of stack elements
					#pragma omp single
					{
#ifdef _DEBUG
						std::cout << "\t Nonzero counts on entry of prefix-sum "
								 "computation: ";
						for( int k = 0; k <= T; ++k ) {
							std::cout << pfBuf[ k ] << " ";
						}
						std::cout << "\n";
#endif
						for( int k = 1; k < T; ++k ) {
							pfBuf[ k ] += pfBuf[ k - 1 ];
						}
						for( int k = T; k > 0; --k ) {
							pfBuf[ k ] = pfBuf[ k - 1 ];
							assert( pfBuf[ k ] <= _cap );
						}
						pfBuf[ 0 ] = 0;
#ifdef _DEBUG
						std::cout << "\t Prefix sums: ";
						for( int k = 0; k <= T; ++k ) {
							std::cout << pfBuf[ k ] << " ";
						}
						std::cout << "\n";
#endif
					} // implied barrier here, which is required

					// catch trivial case
					if( pfBuf[ T ] == 0 ) {
#ifdef _DEBUG
						std::cout << "\t " << t << ": No updates to perform. "
							<< "Exiting joinUpdate with TRUE\n";
#endif
						return true;
					}

					// otherwise perform copy -- completely in parallel
					assert( _n <= _cap );
					const size_t stack_offset = _buf / T;
					assert( _buf % T == 0 );
					const size_t global_bs = pfBuf[ T ] / T + ( pfBuf[ T ] % T > 0 ? 1 : 0 );
					const size_t global_start = t * global_bs > pfBuf[ T ] ?
						pfBuf[ T ] :
						t * global_bs;
					const size_t global_end = global_start + global_bs > pfBuf[ T ] ?
						pfBuf[ T ] :
						global_start + global_bs;
					const size_t global_length = global_end - global_start;
#ifdef _DEBUG
					#pragma omp critical
					std::cout << "\t Thread " << t << " has range "
						<< global_start << " -- " << global_end << "\n";
#endif
					if( global_length > 0 ) {
						size_t t_start = 0, t_end = 0;
						size_t local_cur = global_start;
						for( int k = 1; k < T; ++k, ++t_start ) {
							if( pfBuf[ k ] > global_start ) {
								break;
							}
						}
						assert( local_cur >= pfBuf[ t_start ] );
						local_cur -= pfBuf[ t_start ];
						for( int k = 0; k <= T; ++k, ++t_end ) {
							if( pfBuf[ k ] >= global_end ) {
								break;
							}
						}
						size_t global_count = 0;
						assert( t_start < t_end );
						assert( local_cur < pfBuf[ t_start + 1 ] );
						for( size_t t_cur = t_start; t_cur < t_end; ++t_cur ) {
							// The below is a _DEBUG statement that is extremely noisy, yet sometimes
							// useful. Hence kept disabled by default.
/*#ifdef _DEBUG
							#pragma omp critical
							{
								std::cout << "\t Thread " << t << " processing nonzero " << global_count
									<< " / " << global_length << " using the local stack of thread "
									<< t_cur << " starting from local index " << local_cur << ".\n";
#endif*/
							assert( local_cur <= pfBuf[ t_cur + 1 ] - pfBuf[ t_cur ] );
							StackType * __restrict__ const cur_stack =
								_buffer + t_cur * stack_offset + 1;
							for( ; global_count < global_length &&
								local_cur < pfBuf[ t_cur + 1 ] - pfBuf[ t_cur ];
								++global_count, ++local_cur
							) {
								const size_t global_cur = _n + global_start + global_count;
								assert( global_cur < _cap );
								_stack[ global_cur ] = cur_stack[ local_cur ];
							}
							local_cur = 0;
						}
					}

					// make sure everyone is done reading _n
					#pragma omp barrier

					// then update _n
					#pragma omp single
					{
						assert( _n < _cap );
						_n += pfBuf[ T ];
						assert( _n <= _cap );
					}

#ifdef _DEBUG
					#pragma omp critical
					std::cout << "\t Thread " << t << " exiting joinUpdate. "
						<< "New nonzero count is " << _n << "\n";
#endif

					// make sure the view of _n is synchronised on exiting the join
					#pragma omp barrier

					// done
					update[ 0 ] = 0;
					return false;
#else
					(void)update;
					return true;
#endif
				}

				/**
				 * Copies the state of the i-th coordinate. This function is thread safe.
				 * No two threads should call this functions with the same parameter \a i.
				 * When this function is called, all threads combined should make exactly
				 * #size() calls to this function. All such calls must be followed by a
				 * single call to #joinCopy by a single thread; no other function calls
				 * are valid before such time. An instance unto which one thread has called
				 * \a asyncCopy will enter the copy-updated state. The state shall only
				 * revert to a regular valid one once the above conditions have been met.
				 *
				 * This function may only be called on instances with valid or copy-updated
				 * state.
				 *
				 * May not be called from dense instances.
				 *
				 * \note Thus under the above conditions, this function is thread-safe.
				 *
				 * @param[in] x Where to copy the coordinates from.
				 * @param[in] i Which index to copy.
				 *
				 * @return The nonzero index the i-th nonzero corresponds to.
				 */
				inline StackType asyncCopy( const Coordinates &x, const size_t &i ) noexcept {
#ifndef NDEBUG
					if( _assigned == nullptr && _cap > 0 && _n == _cap ) {
						const bool dense_coordinate_may_not_call_asyncCopy = false;
						assert( dense_coordinate_may_not_call_asyncCopy );
					}
					assert( _buf == x._buf );
					assert( _cap == x._cap );
					assert( x._n <= x._cap );
					assert( i < x._n );
#endif
					const size_t index = x._stack[ i ];
					assert( index < _cap );
					assert( x._assigned[ index ] == true );
					_assigned[ index ] = true;
					_stack[ i ] = index;
					return index;
				}

				/**
				 * This function must be called after exactly #size calls to #asyncCopy.
				 * After the call to this function this instance is a full copy of \a x.
				 *
				 * May not be called from dense instances.
				 *
				 * This function should be called \em exactly once after #size calls to
				 * #asyncCopy. It will revert any copy-updated state back to a valid
				 * state. This function should only be called on instances with
				 * copy-updated state.
				 *
				 * @param[in] x Where to copy the coordinates from.
				 */
				inline void joinCopy( const Coordinates &x ) noexcept {
#ifndef NDEBUG
					if( _assigned == nullptr && _cap > 0 && _n == _cap ) {
						const bool dense_coordinates_may_not_call_joinCopy = false;
						assert( dense_coordinates_may_not_call_joinCopy );
					}
					assert( _buf == x._buf );
					assert( _cap == x._cap );
#endif
					_n = x._n;
				}

				/**
				 * Set the coordinate set to empty. This instance shall becomes valid.
				 * This function may be called on instances with any state. Any pending
				 * #Update instances immediately become invalid.
				 *
				 * May not be called from dense instances.
				 *
				 * This function may be called on instances with any (other) state.
				 */
				void clear() noexcept {
					clear_header();
					if( _n == _cap ) {
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						clear_oh_n_omp();
#else
						clear_oh_n_kernel( 0, _cap );
#endif
					} else {
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						clear_oh_nz_omp();
#else
						clear_oh_nz_seq();
#endif
					}
					_n = 0;
				}

				void clear_seq() noexcept {
					clear_header();
					if( _n == _cap ) {
						clear_oh_n_kernel( 0, _cap );
					} else {
						clear_oh_nz_seq();
					}
					_n = 0;
				}

				/**
				 * Set a range of coordinates to empty. This instance becomes invalid
				 * immediately. By exception, a call to #nonzeroes does remain functional.
				 *
				 * May not be called on dense instances.
				 *
				 * \internal The exception was added since guaranteeing it costs only
				 *           \f$ \Theta(1) \f$ time.
				 *
				 * @param[in] start The first index which shall be cleared.
				 * @param[in] end   The first index from \a start onwards which will
				 *                  \em not be cleared.
				 */
				inline void clearRange( const size_t start, const size_t end ) noexcept {
#ifndef NDEBUG
					if( _assigned == nullptr && _cap > 0 && _n == _cap ) {
						const bool dense_coordinates_cannot_call_clearRange = false;
						assert( dense_coordinates_cannot_call_clearRange );
					}
					assert( start <= end );
					assert( end <= _cap );
#endif
					size_t removed = 0;

#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
					#pragma omp parallel
					{
						size_t local_removed = 0;
						// chose a static schedule, even though there may be imbalance in the
						// number of writes to #_assigned. Assuming they are roughly balanced
						// is in-line with a similar assumption when (sometimes) choosing a
						// static schedule during sparse matrix--vector multiplication in
						// reference/blas2.hpp
						size_t loop_start, loop_end;
						config::OMP::localRange( loop_start, loop_end, start, end );
#else
						const size_t loop_start = start;
						const size_t loop_end = end;
#endif
						for( size_t i = loop_start; i < loop_end; ++i ) {
							if( _assigned[ i ] ) {
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
								(void) ++local_removed;
#else
								(void) ++removed;
#endif
								_assigned[ i ] = false;
							}
						}
#ifdef _H_GRB_REFERENCE_OMP_COORDINATES
						#pragma omp critical
						{
							removed += local_removed;
						}
					}
#endif
					_n -= removed;
				}

				/**
				 * @returns Whether the coordinate set is empty.
				 * This function may only be called on instances with valid state.
				 */
				inline bool isEmpty() const noexcept {
					if( _n == 0 ) {
						return true;
					} else {
						return false;
					}
				}

				/** Returns whether this coordinate set is dense. */
				inline bool isDense() const noexcept {
					return _n == _cap;
				}

				/**
				 * @returns The size (dimension) of the coordinate set.
				 * This function may be called on instances with any state.
				 */
				inline size_t size() const noexcept {
					return _cap;
				}

				/**
				 * Whether a specific index was assigned.
				 *
				 * @param[in] The index \a i to check for. This value must be less than
				 *            #size.
				 *
				 * @returns Whether the i-th nonzero is assigned.
				 *
				 * This function may only be called on instances with valid state. If the
				 * current coordinate instance is dense, this function always returns
				 * <tt>true</tt>.
				 */
				inline bool assigned( const size_t i ) const noexcept {
					assert( i < _cap );
					return _n == _cap || _assigned[ i ];
				}

				/**
				 * Prefetches the result of a call to #assigned with the same argument \a a.
				 *
				 * @param[in] i The index to prefetch.
				 *
				 * This is equivalent to a prefetch hint -- a call to this function may or
				 * not translate to a no-op.
				 *
				 * \warning Debug-mode assertion allows for out-of-range \a i within the
				 *          configured prefetch distance.
				 */
				inline void prefetch_assigned( const size_t i ) const noexcept {
					assert( i < _cap + config::PREFETCHING< reference >::distance() );
					__builtin_prefetch( _assigned + i );
				}

				/**
				 * Prefetches a nonzero value at a given offset \a i.
				 *
				 * @param[in] i The index to prefetch.
				 * @param[in] x The nonzero value array.
				 *
				 * This is equivalent to a prefetch hint -- a call to this function may or
				 * not translate to a no-op.
				 *
				 * For sparse vectors to be used in conjunction with #prefetch_assigned.
				 *
				 * For <tt>void</tt> nonzero types, translates into a no-op.
				 *
				 * \warning Debug-mode assertion allows for out-of-range \a i within the
				 *          configured prefetch distance.
				 */
				template< typename T >
				inline void prefetch_value(
					const size_t i,
					const T *__restrict__ const x
				) const noexcept {
					assert( i < _cap + config::PREFETCHING< reference >::distance() );
					__builtin_prefetch( x + i );
				}

				/**
				 * Specialisation for void nonzero element types.
				 *
				 * Translates to a no-op.
				 *
				 * \warning Debug-mode assertion allows for out-of-range \a i within the
				 *          configured prefetch distance.
				 */
				inline void prefetch_value(
					const size_t i,
					const void *__restrict__ const
				) const noexcept {
					assert( i < _cap + config::PREFETCHING< reference >::distance() );
#ifdef NDEBUG
					(void) i;
#endif
				}

				/**
				 * @returns The value of #assigned(i) interpreted as a mask. If \a descr
				 *          demands it, the element itself at position \a i, \a val, may
				 *          need to be inspected also.
				 * This function may only be called on instances with valid state.
				 */
				template< Descriptor descr, typename T >
				inline bool mask( const size_t i, const T * const val ) const noexcept {
					assert( i < _cap );
					return utils::interpretMask< descr >( assigned( i ), val, i );
				}

				/**
				 * @returns The number of coordinates in the current coordinate set.
				 * This function may only be called on instances with valid state.
				 */
				inline size_t nonzeroes() const noexcept {
					assert( _n <= _cap );
					return _n;
				}

				/**
				 * Function to retrieve the nonzero indices of nonzeroes in this container.
				 *
				 * @param[in] k Which nonzero to request the index from. This value is 0 or
				 *              larger and strictly smaller than #nonzeroes()-1.
				 *
				 * \note There is no guarantee on the order of the returned indices.
				 *
				 * @returns The index of the requested nonzero.
				 *
				 * This function may only be called on instances with valid state.
				 */
				inline size_t index( const size_t k ) const noexcept {
					assert( k < _n );
					return isDense() ? k : _stack[ k ];
				}

				/**
				 * May not be called on dense coordinates.
				 *
				 * @param[out] size The size of the stack, in bytes.
				 *
				 * @returns A pointer to the stack memory area.
				 */
				void * getRawStack( size_t &size ) const noexcept {
#ifndef NDEBUG
					if( _assigned == nullptr && _cap > 0 && _cap == _n ) {
						const bool dense_coordinates_cannot_call_getRawStack = false;
						assert( dense_coordinates_cannot_call_getRawStack );
					}
					assert( _stack != nullptr || _cap == 0 );
#endif
					size = _n * sizeof( StackType );
					return _stack;
				}

				/**
				 * May not be called from dense instances.
				 *
				 * @param[out] size The size of the stack, in number of elements.
				 *
				 * @returns The stack.
				 */
				StackType * getStack( size_t &size ) const noexcept {
#ifndef NDEBUG
					if( _assigned == nullptr && _cap > 0 && _cap == _n ) {
						const bool dense_coordinates_cannot_call_getRawStack = false;
						assert( dense_coordinates_cannot_call_getRawStack );
					}
					assert( _stack != nullptr || _cap == 0 );
#endif
					size = _n;
					return _stack;
				}
			};

	} // namespace internal

} // namespace grb

// parse this unit again for OpenMP support
#ifdef _GRB_WITH_OMP
 #ifndef _H_GRB_REFERENCE_OMP_COORDINATES
  #define _H_GRB_REFERENCE_OMP_COORDINATES
  #define reference reference_omp
  #include "graphblas/reference/coordinates.hpp"
  #undef reference
  #undef _H_GRB_REFERENCE_OMP_COORDINATES
 #endif
#endif

#endif // end `_H_GRB_REFERENCE_COORDINATES'

