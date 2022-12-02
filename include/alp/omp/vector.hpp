
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
 * @date 14th of January 2022
 */

#ifndef _H_ALP_OMP_VECTOR
#define _H_ALP_OMP_VECTOR


#include <stdexcept>
#include <memory>

#include <assert.h>

#include <alp/backends.hpp>
#include <alp/rc.hpp>
#include <alp/imf.hpp>
#include <alp/density.hpp>
#include <alp/views.hpp>

#include <alp/base/vector.hpp>
#include <alp/amf-based/vector.hpp>

#include "config.hpp"
#include "matrix.hpp"
#include "storage.hpp"

#ifdef _ALP_OMP_WITH_REFERENCE
 #include <alp/reference/vector.hpp>
#endif


namespace alp {

	namespace internal {

		template< typename T >
		size_t getLength( const Vector< T, omp > & ) noexcept;

		template< typename T >
		const bool & getInitialized( const Vector< T, omp > & v ) noexcept;

		template< typename T >
		void setInitialized( Vector< T, omp > & v, const bool initialized ) noexcept;

		template< typename T >
		const Vector< T, config::default_sequential_backend > &getLocalContainer(
			const Vector< T, omp > &v, const size_t thread, const size_t block
		) noexcept;

		template< typename T >
		Vector< T, config::default_sequential_backend > &getLocalContainer(
			Vector< T, omp > &v, const size_t thread, const size_t block
		) noexcept;

		/**
		 * The parallel shared memory implementation of the ALP/Dense vector.
		 *
		 * @tparam T The type of an element of this vector. \a T shall not be a
		 *           GraphBLAS type.
		 *
		 * \warning Creating a alp::Vector of other GraphBLAS types is
		 *                <em>not allowed</em>.
		 *          Passing a GraphBLAS type as template parameter will lead to
		 *          undefined behaviour.
		 */
		template< typename T >
		class Vector< T, omp > {

			/* ********************
				IO friends
			   ******************** */

			friend size_t internal::getLength< T >( const Vector< T, omp > & ) noexcept;

			friend const bool & internal::getInitialized< T >( const Vector< T, omp > & ) noexcept;

			friend void internal::setInitialized< T >( Vector< T, omp > & , bool ) noexcept;

			friend const Vector< T, config::default_sequential_backend > &getLocalContainer<>(
				const Vector< T, omp > &v, const size_t thread, const size_t block
			) noexcept;

			friend Vector< T, config::default_sequential_backend > &getLocalContainer<>(
				Vector< T, omp > &v, const size_t thread, const size_t block
			) noexcept;

			private:

				/** The number of buffers. */
				size_t num_buffers;

				/** The array of buffers. */
				T **buffers;

				/** Containers constructed around portions of buffers. */
				using container_type = Vector< T, config::default_sequential_backend >;
				std::vector< std::vector< container_type > > containers;

				/** Whether the container is presently initialized. */
				bool initialized;

			public:

				/** Exposes the element type. */
				typedef T value_type;

				/**
				 * The main ALP/Dense vector constructor.
				 *
				 * The constructed object will be uninitalised after successful construction.
				 *
				 *
				 * @param length      The number of elements in the new vector.
				 *
				 * @return SUCCESS This function never fails.
				 *
				 * \parblock
				 * \par Performance semantics.
				 *        -# This constructor entails \f$ \Theta(1) \f$ amount of work.
				 *        -# This constructor may allocate \f$ \Theta( length ) \f$ bytes
				 *           of dynamic memory.
				 *        -# This constructor will use \f$ \Theta(1) \f$ extra bytes of
				 *           memory beyond that at constructor entry.
				 *        -# This constructor incurs \f$ \Theta(1) \f$ data movement.
				 *        -# This constructor \em may make system calls.
				 * \endparblock
				 *
				 * \warning Avoid the use of this constructor within performance critical
				 *          code sections.
				 */
				Vector(
					const Distribution_2_5D &d,
					const size_t cap = 0
				) : num_buffers( d.getNumberOfThreads() ),
					containers( num_buffers ),
					initialized( false ) {

					(void) cap;

#ifdef DEBUG
					std::cout << "Entered OMP internal::Vector constructor\n";
#endif

					// TODO: Implement allocation properly
					buffers = new ( std::nothrow ) value_type*[ num_buffers ];
					if( ( num_buffers > 0 ) && ( buffers == nullptr ) ){
						throw std::runtime_error( "Could not allocate memory during alp::Vector<omp> construction." );
					}

					#pragma omp parallel for
					for( size_t thread = 0; thread < config::OMP::current_threads(); ++thread ) {
						const auto t_coords = d.getThreadCoords( thread );
						const auto block_grid_dims = d.getLocalBlockGridDims( t_coords );

						// Assuming that all blocks are of the same size
						const size_t alloc_size = block_grid_dims.first * block_grid_dims.second * d.getBlockSize();

#ifdef DEBUG
						#pragma omp critical
						{
							if( thread != config::OMP::current_thread_ID() ) {
								std::cout << "Warning: thread != OMP::current_thread_id()\n";
							}
							std::cout << "Thread with global coordinates tr = " << t_coords.tr << " tc = " << t_coords.tc
								<< " on OpenMP thread " << config::OMP::current_thread_ID()
								<< " allocating buffer of " << alloc_size << " elements "
								<< " holding " << block_grid_dims.first << " x " << block_grid_dims.second << " blocks.\n";
						}
#endif

						// TODO: Implement allocation properly
						buffers[ thread ] = new ( std::nothrow ) value_type[ alloc_size ];

						if( buffers[ thread ] == nullptr ) {
							throw std::runtime_error( "Could not allocate memory during alp::Vector<omp> construction." );
						}

						// Reserve space for all internal container wrappers to avoid re-allocation
						containers[ thread ].reserve( block_grid_dims.first * block_grid_dims.second );

						// Populate the array of internal container wrappers
						for( size_t br = 0; br < block_grid_dims.first; ++br ) {
							for( size_t bc = 0; bc < block_grid_dims.second; ++bc ) {
								const size_t offset = d.getBlocksOffset( t_coords, br, bc );
								containers[ thread ].emplace_back( &( buffers[ thread ][ offset ] ), d.getBlockSize() );
							}
						}

						// Ensure that the array contains the expected number of containers
						assert( containers[ thread ].size() == block_grid_dims.first * block_grid_dims.second );
					}
				}

				/**
				 * Copy constructor.
				 *
				 * @param other The vector to copy. The initialization state of the copy
				 *              reflects the state of \a other.
				 *
				 * \parblock
				 * \par Performance semantics.
				 *      Allocates the same capacity as the \a other vector, even if the
				 *      actual number of elements contained in \a other is less.
				 *        -# This constructor entails \f$ \Theta(1) \f$ amount of work.
				 *        -# This constructor allocates \f$ \Theta(\max{mn, cap} ) \f$ bytes
				 *           of dynamic memory.
				 *        -# This constructor incurs \f$ \Theta(mn) \f$ of data
				 *           movement.
				 *        -# This constructor \em may make system calls.
				 * \endparblock
				 *
				 * \warning Avoid the use of this constructor within performance critical
				 *          code sections.
				 */
				Vector( const Vector< T, omp > &other ) : Vector( other.n, other.cap ) {
					initialized = other.initialized;
					// const RC rc = set( *this, other ); // note: initialized will be set as part of this call
					// if( rc != SUCCESS ) {
					// 	throw std::runtime_error( "alp::Vector< T, omp > (copy constructor): error during call to alp::set (" + toString( rc ) + ")" );
					// }
				}

				/**
				 * Move constructor. The new vector equal the given
				 * vector. Invalidates the use of the input vector.
				 *
				 * @param[in] other The GraphBLAS vector to move to this new instance.
				 *
				 * \parblock
				 * \par Performance semantics.
				 *        -# This constructor entails \f$ \Theta(1) \f$ amount of work.
				 *        -# This constructor will not allocate any new dynamic memory.
				 *        -# This constructor will use \f$ \Theta(1) \f$ extra bytes of
				 *           memory beyond that at constructor entry.
				 *        -# This constructor will move \f$ \Theta(1) \f$ bytes of data.
				 * \endparblock
				 */
				Vector( Vector< T, omp > &&other ) : buffers( other.buffers ) {
					other.buffers = nullptr;
					// data_deleter = std::move( other.data_deleter );
					// initialized = other.initialized; other.initialized = false;
				}

				/**
				 * Vector destructor.
				 *
				 * \parblock
				 * \par Performance semantics.
				 *        -# This destructor entails \f$ \Theta(1) \f$ amount of work.
				 *        -# This destructor will not perform any memory allocations.
				 *        -# This destructor will use \f$ \mathcal{O}(1) \f$ extra bytes of
				 *           memory beyond that at constructor entry.
				 *        -# This destructor will move \f$ \Theta(1) \f$ bytes of data.
				 *        -# This destructor makes system calls.
				 * \endparblock
				 *
				 * \warning Avoid calling destructors from within performance critical
				 *          code sections.
				 */
				~Vector() {
					if( buffers != nullptr ) {
						for( size_t i = 0; i < num_buffers; ++i ) {
							if( buffers[ i ] != nullptr ) {
								delete buffers[ i ];
							}
						}
						delete [] buffers;
					}
				}

		};

		/** Identifies any omp internal vector as an internal container. */
		template< typename T >
		struct is_container< internal::Vector< T, omp > > : std::true_type {};

	} // end namespace ``alp::internal''

	namespace internal {

		template< typename T >
		size_t getLength( const Vector< T, omp > &v ) noexcept {
			return v.n;
		}

		template< typename T >
		const bool & getInitialized( const Vector< T, omp > & v ) noexcept {
			return v.initialized;
		}

		template< typename T >
		void setInitialized( Vector< T, omp > & v, bool initialized ) noexcept {
			v.initialized = initialized;
		}

		template< typename T >
		const Vector< T, config::default_sequential_backend > &getLocalContainer(
			const Vector< T, omp > &v,
			const size_t thread, const size_t block
		) noexcept {
			assert( thread < v.num_buffers );
			assert( block < v.containers[ thread ].size() );
			return v.containers[ thread ][ block ];
		}

		template< typename T >
		Vector< T, config::default_sequential_backend > &getLocalContainer(
			Vector< T, omp > &v,
			const size_t thread, const size_t block
		) noexcept {
			assert( thread < v.num_buffers );
			assert( block < v.containers[ thread ].size() );
			return v.containers[ thread ][ block ];
		}

	} // end namespace ``alp::internal''

} // end namespace ``alp''

#endif // end ``_H_ALP_OMP_VECTOR''

