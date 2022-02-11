
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
 * Contains the BSP1D implementation for the PinnedVector class.
 *
 * @author A. N. Yzelman
 * @date 20180725
 */

#ifndef _H_GRB_BSP1D_PINNEDVECTOR
#define _H_GRB_BSP1D_PINNEDVECTOR

#include <type_traits>

#include <graphblas/base/pinnedvector.hpp>
#include <graphblas/base/vector.hpp>
#include <graphblas/reference/coordinates.hpp>
#include <graphblas/utils/autodeleter.hpp>

#include "distribution.hpp"
#include "init.hpp"

#include <assert.h>


namespace grb {

	/** No implementation notes. */
	template< typename IOType >
	class PinnedVector< IOType, BSP1D > {

		private:

			/**
			 * Tell the system to delete \a _buffered_values only when we had its last
			 * reference.
			 */
			utils::AutoDeleter< IOType > _raw_deleter;

			/**
			 * Tell the system to delete \a _buffered_coordinates only when we had its
			 * last reference.
			 */
			utils::AutoDeleter< char > _assigned_deleter;

			/** A buffer of the local vector. */
			IOType * _buffered_values;

			/** A buffer of the sparsity pattern of \a _buffered_values. */
			internal::Coordinates< _GRB_BSP1D_BACKEND > _buffered_coordinates;

			/** Whether the vector was pinned in parallel or sequential mode. */
			IOMode _mode;

			/** The global length of the vector. */
			size_t _length;

			/** The user process ID of the information in \a _buffered_values. */
			size_t _s;

			/**
			 * The total number of user process IDs in the context of
			 * \a _buffered_values.
			 */
			size_t _P;

			/**
			 * Translates a local index to a global one.
			 *
			 * @param[in] i A local index.
			 *
			 * @returns The corresponding global index.
			 */
			size_t local_index_to_global( size_t i ) const noexcept {
#ifndef NDEBUG
				size_t local_length = _buffered_coordinates.size();
#endif
				assert( i < _buffered_coordinates.size() );
				if( _mode == SEQUENTIAL ) {
					assert( _length == local_length );
					size_t s = 0;
					size_t remote_length =
						internal::Distribution< BSP1D >::global_length_to_local(
							_length, s, _P
						);
					while( s < _P && i >= remote_length ) {
						i -= remote_length;
						(void) ++s;
						remote_length =
							internal::Distribution< BSP1D >::global_length_to_local(
								_length, s, _P
							);
					}
					assert( s < _P );
					const size_t ret =
						internal::Distribution< BSP1D >::local_index_to_global(
							i, _length, s, _P
						);
					assert( ret < _length );
					return ret;
				} else {
					assert( _mode == PARALLEL );
					const size_t ret =
						internal::Distribution< BSP1D >::local_index_to_global(
							i, _length, _s, _P
						);
					assert( ret < _length );
					return ret;
				}
			}


		public:

			/** \internal No implementation notes. */
			PinnedVector() :
				_buffered_values( nullptr ), _mode( PARALLEL ),
				_length( 0 ), _s( 0 ), _P( 0 )
			{}

			/** \internal No implementation notes. */
			template< typename Coords >
			PinnedVector( const Vector< IOType, BSP1D, Coords > &x, const IOMode mode ) :
				_raw_deleter( x._raw_deleter ), _assigned_deleter( x._assigned_deleter ),
				_buffered_values( mode == PARALLEL ? x._raw + x._offset : x._raw ),
				_mode( mode ), _length( x._global._coordinates.size() )
			{
				const auto data = internal::grb_BSP1D.cload();
				_s = data.s;
				_P = data.P;
				if( mode != PARALLEL ) {
					assert( mode == SEQUENTIAL );
					x.synchronize();
					_buffered_coordinates = x._global._coordinates;
				} else {
					_buffered_coordinates = x._local._coordinates;
				}
			}

			/** \internal No implementation notes. */
			PinnedVector( const PinnedVector< IOType, BSP1D > &other ) :
				_raw_deleter( other._raw_deleter ),
				_assigned_deleter( other._assigned_deleter ),
				_buffered_values( other._buffered_values ),
				_buffered_coordinates( other._buffered_coordinates ),
				_mode( other._mode ), _length( other._length ),
				_s( other._s ), _P( other._P )
			{}

			/** \internal No implementation notes. */
			PinnedVector( PinnedVector< IOType, BSP1D > &&other ) :
				_raw_deleter( other._raw_deleter ),
				_assigned_deleter( other._assigned_deleter ),
				_buffered_values( other._buffered_values ),
				//_buffered_coordinates uses std::move, below
				_mode( other._mode ), _length( other._length ),
				_s( other._s ), _P( other._P )
			{
				_buffered_coordinates = std::move( other._buffered_coordinates );
			}

			/** \internal No implementation notes. */
			PinnedVector< IOType, BSP1D >& operator=(
				const PinnedVector< IOType, BSP1D > &other
			) {
				_raw_deleter = other._raw_deleter;
				_assigned_deleter = other._assigned_deleter;
				_buffered_values = other._buffered_values;
				_buffered_coordinates = other._buffered_coordinates;
				_mode = other._mode;
				_length = other._length;
				_s = other._s;
				_P = other._P;
				return *this;
			}

			/** \internal No implementation notes. */
			PinnedVector< IOType, BSP1D >& operator=(
				PinnedVector< IOType, BSP1D > &&other
			) {
				_raw_deleter = other._raw_deleter;
				_assigned_deleter = other._assigned_deleter;
				_buffered_values = other._buffered_values;
				_buffered_coordinates = std::move( other._buffered_coordinates );
				_mode = other._mode;
				_length = other._length;
				_s = other._s;
				_P = other._P;
				return *this;
			}

			/** \internal No implementation notes. */
			inline size_t size() const noexcept {
				return _length;
			}

			/** \internal No implementation notes. */
			inline size_t nonzeroes() const noexcept {
				if( _length == 0 ) {
					return 0;
				} else {
					return _buffered_coordinates.nonzeroes();
				}
			}

			/** \internal No implementation notes. */
			template< typename OutputType = IOType >
			inline OutputType getNonzeroValue(
				const size_t k,
				const OutputType one
			) const noexcept {
				assert( _length > 0 );
				assert( k < _buffered_coordinates.size() );
				assert( k < _buffered_coordinates.nonzeroes() );
				if( _buffered_values == nullptr ) {
					assert( (std::is_same< IOType, void >::value) );
					return one;
				} else {
					const size_t local_i = _buffered_coordinates.index( k );
					return _buffered_values[ local_i ];
				}
			}

			/** \internal No implementation notes. */
			inline IOType getNonzeroValue( const size_t k ) const noexcept {
				assert( _length > 0 );
				assert( k < _buffered_coordinates.size() );
				assert( k < _buffered_coordinates.nonzeroes() );
				const size_t local_i = _buffered_coordinates.index( k );
				return _buffered_values[ local_i ];
			}

			/** \internal No implementation notes. */
			inline size_t getNonzeroIndex( const size_t k ) const noexcept {
				assert( _length > 0 );
				assert( k < _buffered_coordinates.size() );
				assert( k < _buffered_coordinates.nonzeroes() );
				const size_t local_i = _buffered_coordinates.index( k );
				const size_t global_i = local_index_to_global( local_i );
				assert( global_i < _length );
				return global_i;
			}

	};

} // namespace grb

#endif // end ``_H_GRB_BSP1D_PINNEDVECTOR''

