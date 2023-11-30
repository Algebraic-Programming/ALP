
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
 * Coordinates for the nonblocking backend
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_COORDINATES
#define _H_GRB_NONBLOCKING_COORDINATES

#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif


#include <stdexcept> //std::runtime_error
#include <vector>
#if defined _DEBUG && ! defined NDEBUG
 #include <set>
#endif

#include <cstddef> //size_t
#include <cassert>
#include <algorithm>

#include <graphblas/rc.hpp>
#include <graphblas/backends.hpp>
#include <graphblas/descriptors.hpp>

#include <graphblas/utils.hpp>

#include <graphblas/base/coordinates.hpp>

#include <graphblas/reference/config.hpp>

#include <graphblas/nonblocking/init.hpp>
#include <graphblas/nonblocking/analytic_model.hpp>
#include <iomanip>

// #define _LOCAL_DEBUG


namespace grb {

	namespace internal {

		/**
		 * The Coordinates class is based on that of the reference backend.
		 * A set of new methods is added to handle local coordinates used
		 * by the nonblocking backend. The bufferSize method used by the
		 * Matrix class relies on parbufSize and prefixbufSize that have
		 * their own implementation for the nonblocking backend.
		 */
		template<>
		class Coordinates< nonblocking > {

			public:

				typedef typename config::VectorIndexType StackType;

				typedef bool ArrayType;


				// TODO: Remove me
				bool _debug_is_counting_sort_done = false;


			private:

				bool * __restrict__ _assigned;

				StackType * __restrict__ _stack;

				StackType * __restrict__ _buffer;

				size_t _n;

				size_t _cap;

				size_t _buf;

				// pointers to the data of the local coordinates mechanism
				std::vector< config::VectorIndexType * > local_buffer;
				config::VectorIndexType * __restrict__ local_new_nnzs;
				config::VectorIndexType * __restrict__ pref_sum;

				std::vector<config::VectorIndexType> counting_sum;


				// the analytic model used during the execution of a pipeline
				AnalyticModel analytic_model;


			public:

				static inline size_t arraySize( const size_t dim ) noexcept {
					if( dim == 0 ) {
						return 0;
					}
					return ( dim + 1 ) * sizeof( ArrayType );
				}

				static inline size_t stackSize( const size_t dim ) noexcept {
					if( dim == 0 ) {
						return 0;
					}
					return ( dim + 1 ) * sizeof( StackType );
				}

				static inline size_t prefixbufSize() noexcept {
					int P = 1;
					return ( P + 1 ) * sizeof( StackType );
				}

				static inline size_t parbufSize( const size_t n ) noexcept {
					return internal::NONBLOCKING::vectorBufferSize( n ) * sizeof( StackType );
				}

				static inline size_t bufferSize( const size_t dim ) noexcept {
					size_t ret = stackSize( dim );
					ret += parbufSize( dim );
					ret += prefixbufSize();
					return ret;
				}

				inline Coordinates() noexcept :
					_assigned( nullptr ), _stack( nullptr ), _buffer( nullptr ),
					_n( 0 ), _cap( 0 ), _buf( 0 )
				{}

				inline Coordinates( Coordinates< nonblocking > &&x ) noexcept :
					_assigned( x._assigned ), _stack( x._stack ), _buffer( x._buffer ),
					_n( x._n ), _cap( x._cap ), _buf( x._buf )
				{
					x._assigned = nullptr;
					x._stack = nullptr;
					x._buffer = nullptr;
					x._n = x._cap = x._buf = 0;
				}

				inline Coordinates( const Coordinates< nonblocking > &x ) noexcept :
					_assigned( x._assigned ), _stack( x._stack ), _buffer( x._buffer ),
					_n( x._n ), _cap( x._cap ), _buf( x._buf )
				{
					assert( this != &x );
				}

				inline Coordinates< nonblocking > & operator=(
					const Coordinates< nonblocking > &other
				) {
					Coordinates replace( other );
					*this = std::move( replace );
					return *this;
				}

				inline Coordinates< nonblocking > & operator=(
					Coordinates< nonblocking > &&x
				) noexcept {
					assert( this != &x );
					_assigned = x._assigned;
					_stack = x._stack;
					_buffer = x._buffer;
					_n = x._n;
					_cap = x._cap;
					_buf = x._buf;
					x._assigned = nullptr;
					x._stack = x._buffer = nullptr;
					x._n = x._cap = x._buf = 0;
					return *this;
				}

				inline ~Coordinates() noexcept {
					// done (the #_assigned and #_stack memory
					// blocks are not managed by this class)
				}

				void set(
					void * const arr, bool arr_initialized,
					void * const buf, const size_t dim, bool parallel = true
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
					_buf = internal::NONBLOCKING::vectorBufferSize( _cap );

					// and initialise _assigned (but only if necessary)
					if( dim > 0 && !arr_initialized ) {
						if( parallel ) {
							#pragma omp parallel
							{
								size_t start, end;
								config::OMP::localRange( start, end, 0, dim );
								for( size_t i = start; i < end; ++i ) {
									_assigned[ i ] = false;
								}
							}
						} else {
							for( size_t i = 0; i < dim; ++i ) {
								_assigned[ i ] = false;
							}
						}
					}
				}

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

				template< bool maybe_invalid = false >
				inline void local_assignAll( ) noexcept {
					if( maybe_invalid || _n != _cap ) {
						if( _assigned != nullptr ) {
							assert( _stack != nullptr );
							assert( maybe_invalid || _n < _cap );
							assert( !maybe_invalid || _n <= _cap );
							_n = _cap;

							for( size_t i = 0; i < _n; ++i ) {
								_assigned[ i ] = true;
								_stack[ i ] = i;
							}
						}
					}

					// the counter of initial nonzeroes in the local stack is stored in the
					// buffer immediately before the local stack
					StackType * __restrict__ local_nnzs = _stack - 1;

					// the counter for the local stack must be set to zero such that the number
					// of new nonzeroes will be set to _n by asyncJoinSubset and joinSubset
					// will update the global stack based on the local_new_nnzs counter the
					// global stack has become empty and _assigned = false so the local
					// coordinates of this tile must be added in the global stack from scratch
					// regardless whether this tile was already dense or not as it is hard to
					// know which part of the global stack contains the coordinates of this
					// tile
					*local_nnzs = 0;
				}

				template< bool maybe_invalid = false >
				inline void local_assignAllNotAlreadyAssigned( ) noexcept {
					if( maybe_invalid || _n != _cap ) {
						if( _assigned != nullptr ) {
							assert( _stack != nullptr );
							assert( maybe_invalid || _n < _cap );
							assert( !maybe_invalid || _n <= _cap );

							// searching for the not already assigned elements and add them to the
							// local stack such that joinSubset will add to the global stack only
							// those elements that are not already assigned
							for( size_t i = 0; i < _cap; ++i ) {
								if( !_assigned[ i ] ) {
									_assigned[ i ] = true;
									_stack[ _n++ ] = i;
								}
							}

							assert( _n == _cap );
						}
					}
				}

				inline void clear() noexcept {

					if( _n == _cap ) {
#ifndef NDEBUG
						if( _assigned == nullptr && _cap > 0 ) {
							const bool dense_coordinates_may_not_call_clear = false;
							assert( dense_coordinates_may_not_call_clear );
						}
#endif

						#pragma omp parallel for schedule( dynamic, config::CACHE_LINE_SIZE::value() )
						for( size_t i = 0; i < _cap; ++i ) {
							_assigned[ i ] = false;
						}
					} else {
						if( _n < config::OMP::minLoopSize() ) {
							for( size_t k = 0; k < _n; ++k ) {
								_assigned[ _stack[ k ] ] = false;
							}
						} else {
							#pragma omp parallel for schedule( dynamic, config::CACHE_LINE_SIZE::value() )
							for( size_t k = 0; k < _n; ++k ) {
								_assigned[ _stack[ k ] ] = false;
							}
						}
					}
					_n = 0;
				}

				inline void local_clear() noexcept {

					if( _n == _cap ) {
#ifndef NDEBUG
						if( _assigned == nullptr && _cap > 0 ) {
							const bool dense_coordinates_may_not_call_clear = false;
							assert( dense_coordinates_may_not_call_clear );
						}
#endif

						for( size_t i = 0; i < _cap; ++i ) {
							_assigned[ i ] = false;
						}
					} else {
						for( size_t k = 0; k < _n; ++k ) {
							_assigned[ _stack[ k ] ] = false;
						}
					}
					_n = 0;

					// the counter of initial nonzeroes in the local stack is stored in the
					// buffer immediately before the local stack
					StackType * __restrict__ local_nnzs = _stack - 1;

					// the counter for the local stack must be set to zero such that any new
					// assigned element will be written to the global stack
					*local_nnzs = 0;
				}

				inline void reset_global_nnz_counter() noexcept {
					_n = 0;
				}

				inline bool isEmpty() const noexcept {
					if( _n == 0 ) {
						return true;
					} else {
						return false;
					}
				}

				inline bool isDense() const noexcept {
					return _n == _cap;
				}

				inline size_t size() const noexcept {
					return _cap;
				}

				inline bool assigned( const size_t i ) const noexcept {
					assert( i < _cap );
					return _n == _cap || _assigned[ i ];
				}

				template< Descriptor descr, typename T >
				inline bool mask( const size_t i, const T * const val ) const noexcept {
					assert( i < _cap );
					return utils::interpretMask< descr >( assigned( i ), val, i );
				}

				inline size_t nonzeroes() const noexcept {
					assert( _n <= _cap );
					return _n;
				}

				inline size_t index( const size_t k ) const noexcept {
					assert( k < _n );
					return isDense() ? k : _stack[ k ];
				}

				void localCoordinatesInit( const AnalyticModel &am ) {

					analytic_model = am;

					const size_t nthreads = analytic_model.getNumThreads();
					const size_t tile_size = analytic_model.getTileSize();
					const size_t num_tiles = analytic_model.getNumTiles();

					assert( num_tiles > 0 );
					assert( num_tiles <= internal::NONBLOCKING::maxBufferTiles( _cap ) );
					assert( _buf >= 4 * num_tiles );

					local_buffer.resize( analytic_model.getNumTiles() );

					#pragma omp parallel for default(none) \
						firstprivate(tile_size, num_tiles) \
						shared(local_buffer, _buffer) \
						schedule(dynamic) num_threads(nthreads)
					for( size_t tile_id = 0; tile_id < num_tiles; ++tile_id ) {
						local_buffer[ tile_id ] = _buffer + tile_id * ( tile_size + 1 );
					}

					local_new_nnzs = _buffer + num_tiles * ( tile_size + 1 );
					pref_sum = _buffer + num_tiles * ( tile_size + 2 );
				}

				bool should_use_bitmask_asyncSubsetInit(
					const size_t /* num_tiles */,
					const size_t /* tile_id */,
					const size_t lower_bound,
					const size_t upper_bound
				) const noexcept {
					assert( _cap > 0 );
					assert( _n <= _cap );
					assert( lower_bound <= upper_bound );
					return nonzeroes() * (upper_bound - lower_bound) > size();
				}

				void _asyncSubsetInit_bitmask(
						const size_t lower_bound,
						const size_t upper_bound,
						const size_t /*tile_id*/,
						config::VectorIndexType *local_nnzs,
						config::VectorIndexType *local_stack
				) noexcept {
					assert( _cap > 0 );

					for( size_t i = lower_bound; i < upper_bound; ++i ) {
						if( _assigned[ i ] ) {
							local_stack[ (*local_nnzs)++ ] = i - lower_bound;
						}
					}
				}

				void _asyncSubsetInit_search(
						const size_t lower_bound,
						const size_t upper_bound,
						const size_t tile_id,
						config::VectorIndexType *local_nnzs,
						config::VectorIndexType *local_stack
				) noexcept {
					(void) tile_id;
					(void) lower_bound;
					(void) upper_bound;

					const auto lower_bound_idx = counting_sum[ tile_id ];
					const auto upper_bound_idx = counting_sum[ tile_id+1 ];
					if( lower_bound_idx == upper_bound_idx ) { return; }
#if defined(_DEBUG) || defined(_LOCAL_DEBUG)
					#pragma omp critical
						fprintf(stderr, "[T%02d] - _asyncSubsetInit_search():  tile_id=%zu, lower_bound_idx=%u, upper_bound_idx=%u\n",
							omp_get_thread_num(), tile_id, lower_bound_idx, upper_bound_idx );
#endif
					for( size_t i = lower_bound_idx; i < upper_bound_idx; ++i ) {
						const size_t k = _stack[ i ];
#if defined(_DEBUG) || defined(_LOCAL_DEBUG)
						if( not ( lower_bound <= k && k < upper_bound ) ) {
							#pragma omp critical
								fprintf(stderr, "ERROR [T%02d] - _asyncSubsetInit_search(): i=%zu, k=%zu, lower_bound=%zu, upper_bound=%zu\n",
									omp_get_thread_num(), i, k, lower_bound, upper_bound );
						}
#endif
						ASSERT( lower_bound <= k && k < upper_bound, "i=" << i << ", k=" << k << ", lower_bound=" << lower_bound << ", upper_bound=" << upper_bound );
						ASSERT( _assigned[ k ], "i=" << i << ", k=" << k << ", lower_bound=" << lower_bound << ", upper_bound=" << upper_bound );
						local_stack[ (*local_nnzs)++ ] = k - lower_bound;
					}
				}

				/**
				 * Initialises a Coordinate instance that refers to a subset of this
				 * coordinates instance. Multiple disjoint subsets may be retrieved
				 * and concurrently updated, up to a maximum of tiles given by
				 *   #internal::NONBLOCKING::maxBufferTiles().
				 *
				 * Subsets must be contiguous. If one thread calls this function, all
				 * other threads must make a matching call.
				 *
				 * @param[in] lower_bound     The start index of the contiguous subset
				 *                            (inclusive).
				 * @param[in] upper_bound     The end index of the contiguous subset
				 *                            (exclusive).
				 */
				void asyncSubsetInit(
					const size_t num_tiles,
					const size_t lower_bound,
					const size_t upper_bound
				) noexcept {
					(void) num_tiles;
					if( _cap == 0 ) { return; }

					const size_t tile_id = lower_bound / analytic_model.getTileSize();

					config::VectorIndexType *local_nnzs = local_buffer[ tile_id ];
					config::VectorIndexType *local_stack = local_buffer[ tile_id ] + 1;

					*local_nnzs = 0;
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					_asyncSubsetInit_bitmask( lower_bound, upper_bound, tile_id, local_nnzs, local_stack );
#else
					if( should_use_bitmask_asyncSubsetInit( num_tiles, tile_id, lower_bound, upper_bound ) ) {
						#ifdef _LOCAL_DEBUG
						#pragma omp critical
							std::cerr << "> Using bitmask\n";
						#endif
						_asyncSubsetInit_bitmask( lower_bound, upper_bound, tile_id, local_nnzs, local_stack );
					} else {
						assert( _debug_is_counting_sort_done );
						#ifdef _LOCAL_DEBUG
						#pragma omp critical
							std::cerr << "> Using search\n";
						#endif
						_asyncSubsetInit_search( lower_bound, upper_bound, tile_id, local_nnzs, local_stack );
					}
#endif

					// the number of new nonzeroes is initialized here
					local_new_nnzs[ tile_id ] = 0;
				}

				static size_t getTileId(
						size_t k,
						const size_t num_tiles,
						const std::vector< size_t > &lower_bounds,
						const std::vector< size_t > &upper_bounds
				) {
					ASSERT( num_tiles > 0, "num_tiles = " << num_tiles );
					(void) num_tiles;
					(void) lower_bounds;
					(void) upper_bounds;

					const auto tile_size = upper_bounds[0] - lower_bounds[0];
					ASSERT( tile_size > 0, "tile_size = " << tile_size );
					const size_t tile_id = k / tile_size;

					ASSERT(tile_id < num_tiles, "tile_id = " << tile_id << ", num_tiles = " << num_tiles);
					ASSERT(k < upper_bounds[tile_id], "k = " << k << ", tile_id = " << tile_id << ", upper_bounds[tile_id] = " << upper_bounds[tile_id]);
					ASSERT(k >= lower_bounds[tile_id], "k = " << k << ", tile_id = " << tile_id << ", lower_bounds[tile_id] = " << lower_bounds[tile_id]);
					return tile_id;
				}

				void countingSumComputation_sequential(
						const size_t num_tiles,
						const std::vector< size_t > &lower_bounds,
						const std::vector< size_t > &upper_bounds,
						const std::vector< size_t > &tiles_to_process
				) noexcept {
					(void) tiles_to_process;

					// TODO: Move me to the initialisation phase, and use _buffer instead of a vector
					counting_sum.resize( num_tiles+1 );

					// Initialise counting to zero
					#pragma omp for simd
					for( size_t i = 0; i <= num_tiles; ++i ) {
						counting_sum[ i ] = 0;
					}

					if( num_tiles == 0 ) { return; }

					// For-each element in the stack
					for (size_t i = 0; i < _n; ++i) {
						const auto k = _stack[i];

						// Find the tile id of the element
						size_t tile_id = getTileId( k, num_tiles, lower_bounds, upper_bounds );

						// Assertions
						ASSERT( _assigned[k], "i=" << i << ", k=" << k << ", tile_id=" << tile_id );
						ASSERT( k >= lower_bounds[tile_id], "i=" << i << ", k=" << k << ", tile_id=" << tile_id << ", lower_bound=" << lower_bounds[tile_id] );
						ASSERT( k < upper_bounds[tile_id], "i=" << i << ", k=" << k << ", tile_id=" << tile_id << ", upper_bound=" << upper_bounds[tile_id] );
						ASSERT( tile_id < num_tiles, "i=" << i << ", k=" << k << ", tile_id=" << tile_id << ", num_tiles=" << num_tiles );

						// Increment the counting for the element's tile
						counting_sum[tile_id + 1]++;

					} // end for-each element in the stack


#if defined(_DEBUG) || defined(_LOCAL_DEBUG)
					std::cout << "counting_sum (not-prefixed): {num_tiles=" << num_tiles << "}[ ";
					for(size_t i = 0; i <= num_tiles; ++i )
						std::cout << std::setw(3) << counting_sum[ i ] << " ";
					std::cout << "]\n";
#endif

					// TODO: Make this parallel
					// Prefix-sum computation of the counting
					for( size_t i = 0; i < num_tiles; ++i ) {
						counting_sum[i+1] += counting_sum[i];
					}

#if defined(_DEBUG) || defined(_LOCAL_DEBUG)
					std::cout << "counting_sum (prefixed):     {num_tiles=" << num_tiles << "}[ ";
					for(size_t i = 0; i <= num_tiles; ++i )
						std::cout << std::setw(3) << counting_sum[ i ] << " ";
					std::cout << "]\n";
#endif

					ASSERT( counting_sum[ num_tiles ] == _n, "counting_sum[ num_tiles ] = " << counting_sum[ num_tiles ] << ", _n = " << _n );
				}

				void countingSortComputation(
					const size_t num_tiles,
					const std::vector< size_t > &lower_bounds,
					const std::vector< size_t > &upper_bounds,
					const std::vector< size_t > &tiles_to_process
				) noexcept {
					if(num_tiles == 1) {
						#pragma omp critical
							std::cerr << "countingSortComputation(): num_tiles == 1\n";
						_debug_is_counting_sort_done = true;
						return;
					}

					countingSumComputation_sequential( num_tiles, lower_bounds, upper_bounds, tiles_to_process );

					// For-each tile
					for( size_t tile_id : tiles_to_process ) {

						// Bounds of the current tile
						const auto lower_bound = lower_bounds[tile_id];
						const auto upper_bound = upper_bounds[tile_id];

						// Allows to keep counting_sort intact and singular
						size_t assigned_in_tile = 0;

						// Allows quick exit from the loop if the tile has already been filled
						const size_t max_assigned_in_tile = upper_bound - lower_bound;

						// For-each element in the stack, from the end of the last processed tile
						for (size_t i = counting_sum[tile_id + 1]; i < _n && assigned_in_tile < max_assigned_in_tile; ++i) {
							const auto k = _stack[i];

							// If the element is not in the current tile, skip it
							if(not(lower_bound <= k && k < upper_bound)) { continue; }

							// Find the new index of the element: beginning of the current tile + assigned_in_tile
							const auto stack_new_idx = counting_sum[tile_id] + assigned_in_tile;

							// Increment the number of assigned elements in the current tile
							assigned_in_tile++;

							// Assertions
							assert(stack_new_idx < _n);
							assert(_assigned[k]);
							assert(k >= lower_bound);
							assert(k < upper_bound);

							// Swap the element with the one at the new index
							std::swap(_stack[i], _stack[stack_new_idx]);

						} // end for-each element in the stack
					} // end for-each tile

#if defined(_DEBUG) || defined(_LOCAL_DEBUG)
					std::cout << "_stack (after sort): [";
					for( size_t _tile_id = 0; _tile_id < num_tiles; ++_tile_id ) {
						std::cout << "\n\t| ";
						for( size_t i = counting_sum[_tile_id]; i < counting_sum[_tile_id+1]; ++i )
							std::cout << _stack[i] << " ";
					}
					std::cout << "\n]\n";
#endif

//					{ // Pass over the _stack and check that the coordinates are sorted
//						for (size_t i = 0; i < _n; i++) {
//							const auto k = _stack[i];
//							const auto tile_id = getTileId( k, num_tiles, lower_bounds, upper_bounds );
//							(void) tile_id;
//							ASSERT(_assigned[k], "i=" << i << ", k=" << k << ", tile_id=" << tile_id);
//							ASSERT(k < upper_bounds[tile_id], "i=" << i << ", k=" << k << ", tile_id=" << tile_id << ", upper_bounds[tile_id]=" << upper_bounds[tile_id]);
//							ASSERT(k >= lower_bounds[tile_id], "i=" << i << ", k=" << k << ", tile_id=" << tile_id << ", lower_bounds[tile_id]=" << lower_bounds[tile_id]);
//							ASSERT(tile_id < num_tiles, "i=" << i << ", k=" << k << ", tile_id=" << tile_id << ", num_tiles=" << num_tiles);
//						}
//					}

					_debug_is_counting_sort_done = true;
				}

				/**
				 * Retrieves a subset coordinate instance that was previously initialised
				 * using a call to #asyncSubsetInit.
				 *
				 * @returns A Coordinates instance that only supports sequential
				 *          (synchronous) updates as well as all queries.
				 */
				Coordinates< nonblocking > asyncSubset(
					const size_t lower_bound, const size_t upper_bound
				) const noexcept {
					assert(_cap > 0);

					const size_t tile_id = lower_bound / analytic_model.getTileSize();

					config::VectorIndexType *local_nnzs = local_buffer[ tile_id ];
					config::VectorIndexType *local_stack = local_buffer[ tile_id ] + 1;

					Coordinates< nonblocking > ret;
					assert( upper_bound - lower_bound <= analytic_model.getTileSize() );

					ret.set( _assigned + lower_bound, true, local_stack,
						upper_bound - lower_bound, false );

					// the number of new nonzeroes is used to determine the total number
					// of nonzeroes for the given local coordinates, since some of the
					// nonzeroes are already written on the local statck
					ret._n = (*local_nnzs) + local_new_nnzs[ tile_id ];
					assert( ret._n <= ret._cap );

					ret._buf = 0;

					return ret;
				}

				/**
				 * Saves the state of a subset Coordinates instance. Can be retrieved later
				 * once again via a call to #asyncSubset. New nonzeroes will be committed
				 * to the global coordinate structure via a call to #joinSubset, which will
				 * furthermore set the related tile to inactive.
				 */
				void asyncJoinSubset(
					const Coordinates< nonblocking > &subset,
					const size_t lower_bound, const size_t upper_bound
				) {
					assert( _cap > 0 );

					(void) upper_bound;

					const size_t tile_id = lower_bound / analytic_model.getTileSize();

					config::VectorIndexType *local_nnzs = local_buffer[ tile_id ];

					assert( subset._n <= subset._cap );
					assert( (*local_nnzs) <= subset._cap );

					local_new_nnzs[ tile_id ] = subset._n - (*local_nnzs);
				}

				bool newNonZeroes() const {

					if( _cap == 0 ) {
						return false;
					}

					const size_t num_tiles = analytic_model.getNumTiles();

					for( size_t i = 0; i < num_tiles; i++ ) {
						if( local_new_nnzs[ i ] > 0 ) {
							return true;
						}
					}
					return false;
				}

				void prefixSumComputation() {

					const size_t num_tiles = analytic_model.getNumTiles();

					// takes into accout the size of data for each iteration of the prefix sum
					// computation which is used to determine the number of parallel task that
					// should be used such that the data of each parallel task fit in the L1
					// cache
					constexpr size_t size_of_data = sizeof( pref_sum[0] ) +
						sizeof( local_new_nnzs[0] );

					// make use of the analytic model to estimate a proper number of threads
					// and a tile size
					AnalyticModel am( size_of_data, num_tiles, 1 );

					const size_t nthreads = am.getNumThreads();
					const size_t prefix_sum_tile_size = am.getTileSize();
					const size_t prefix_sum_num_tiles = am.getNumTiles();

					// make a run-time decision to choose between sequential and parallel
					// prefix sum implementation the sequential prefix sum implementation is
					// more efficient for a small number of tiles
					if( num_tiles < prefix_sum_tile_size ) {
						// sequential computation of the prefix sum
						pref_sum[ 0 ] = _n + local_new_nnzs[ 0 ];
						for( size_t i = 1; i < num_tiles; i++ ) {
							pref_sum[ i ] = pref_sum[ i - 1 ] + local_new_nnzs[ i ];
						}
					} else {
						// parallel computation of the prefix sum
						size_t local_prefix_sum[ prefix_sum_num_tiles ];

						#pragma omp parallel num_threads(nthreads)
						{
							#pragma omp for
							for( size_t id = 0; id < prefix_sum_num_tiles; id++ ) {

								size_t lower, upper;
								config::OMP::localRange( lower, upper, 0, num_tiles,
									prefix_sum_tile_size, id, prefix_sum_num_tiles );

								// the number of threads used for parallel computation must not exceed
								// num_tiles, otherwise the code below results in data races
								assert( id <= num_tiles );
								assert( id < prefix_sum_num_tiles - 1 || upper == num_tiles );
								assert( lower <= upper );
								assert( upper <= num_tiles );

								pref_sum[ lower ] = local_new_nnzs[ lower ];
								for( size_t i = lower + 1; i < upper; i++ ) {
									pref_sum[ i ] = pref_sum[ i - 1 ] + local_new_nnzs[ i ];
								}

								// each thread stores the prefix sum of its last element in
								// local_prefix_sum
								// the memory location is specified by the identifier of the thread to
								// avoid data races
								local_prefix_sum[ id ] = pref_sum[ upper - 1 ];
							}

							// here, there is an implicit barrier that ensures all threads have
							// already written the local prefix sum for each parallel task

							// a single threads computes the prefix sum for the last element of each
							// thread
							#pragma omp single
							{
								for( size_t i = 1; i < prefix_sum_num_tiles; i++ ) {
									local_prefix_sum[ i ] += local_prefix_sum[ i - 1 ];
								}
							}

							#pragma omp for
							for(size_t id = 0; id < prefix_sum_num_tiles; id++ ) {

								size_t lower, upper;
								config::OMP::localRange( lower, upper, 0, num_tiles,
									prefix_sum_tile_size, id, prefix_sum_num_tiles );

								// the first thread (id=0) needs to add only the number of nonzeroes(_n)
								const size_t acc = _n + ( ( id > 0 ) ? local_prefix_sum[ id - 1 ] : 0 );
								for( size_t i = lower; i < upper; i++ ) {
									pref_sum[ i ] += acc;
								}
							}
						}

#ifdef _DEBUG
						// ensures that the parallel implementation computes the same result
						// with the following sequential implementation
						size_t seq_offsets[ num_tiles ];
						seq_offsets[ 0 ] = _n + local_new_nnzs[ 0 ];
						for( size_t i = 1; i < num_tiles; i++ ) {
							seq_offsets[ i ] = seq_offsets[ i - 1 ] + local_new_nnzs[ i ];
						}

						for( size_t i = 0; i < num_tiles; i++ ) {
							assert( seq_offsets[i] == pref_sum[i] );
						}
#endif
					}

					// a single thread updates the number of nonzeroes
					// the last element of prefix_sum_ofssets alredy includes
					// the current number of nonzeroes _n which was added earlier
					_n = pref_sum[ num_tiles - 1 ];
				}

				/**
				 * Takes a currently active subset and commits it to the global storage.
				 * After completion the given active tile will be marked inactive.
				 */
				void joinSubset( const size_t lower_bound, const size_t upper_bound ) {
					if( _cap == 0 ) {
						return;
					}
#ifdef NDEBUG
					( void )upper_bound;
#endif
					const size_t tile_id = lower_bound / analytic_model.getTileSize();

					config::VectorIndexType *local_nnzs = local_buffer[ tile_id ];
					config::VectorIndexType *local_stack = local_buffer[ tile_id ] + 1;

					const size_t local_stack_start = *local_nnzs;
					const size_t local_stack_end = *local_nnzs + local_new_nnzs[ tile_id ];
					assert( local_stack_start <= local_stack_end );

					size_t pos = pref_sum[ tile_id ] - local_new_nnzs[ tile_id ];

					for( size_t k = local_stack_start; k < local_stack_end; ++k ) {
						const size_t local_index = local_stack[ k ];
						const size_t global_index = local_index + lower_bound;

						assert( global_index >= lower_bound );
						assert( global_index < upper_bound );
						assert( _assigned[ global_index ] );
						assert( pos < _cap );

						_stack[ pos++ ] = global_index;
					}

					local_new_nnzs[ tile_id ] = 0;
				}
			};

	} // namespace internal

} // namespace grb

#endif // end `_H_GRB_NONBLOCKING_COORDINATES'

