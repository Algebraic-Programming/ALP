
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
 * Contains the reference and reference_omp implementations for the
 * PinnedVector class.
 *
 * @author A. N. Yzelman
 * @date 20180725
 */

#if ! defined _H_GRB_REFERENCE_PINNEDVECTOR || defined _H_GRB_REFERENCE_OMP_PINNEDVECTOR
#define _H_GRB_REFERENCE_PINNEDVECTOR

#include <graphblas/base/pinnedvector.hpp>
#include <graphblas/utils/autodeleter.hpp>

#include "coordinates.hpp"
#include "vector.hpp"


namespace grb {

	/** \internal No implementation notes. */
	template< typename IOType >
	class PinnedVector< IOType, reference > {

		private:

			/**
			 * Tell the system to delete \a _buffered_values only when we had its last
			 * reference.
			 */
			utils::AutoDeleter< IOType > _raw_deleter;

			/**
			 * Tell the system to delete the stack of the \a _buffered_coordinates only
			 * when we had its last reference.
			 */
			utils::AutoDeleter< char > _stack_deleter;

			/** A buffer of the local vector. */
			IOType * _buffered_values;

			/** A buffer of the sparsity pattern of \a _buffered_values. */
			internal::Coordinates<
				config::IMPLEMENTATION< reference >::coordinatesBackend()
			> _buffered_coordinates;


		public:

			/** \internal No implementation notes. */
			PinnedVector() : _buffered_values( nullptr ) {}

			/** \internal No implementation notes. */
			PinnedVector(
				const Vector< IOType, reference, internal::Coordinates<
					config::IMPLEMENTATION< reference >::coordinatesBackend()
				> > &x,
				const IOMode mode
			) :
				_raw_deleter( x._raw_deleter ), _stack_deleter( x._buffer_deleter ),
				_buffered_values( x._raw ), _buffered_coordinates( x._coordinates )
			{
				(void) mode; // sequential and parallel IO mode are equivalent for this
				             // implementation.
			}

			// default destructor is OK

			/** \internal No implementation notes. */
			inline size_t size() const noexcept {
#ifndef NDEBUG
				if( _buffered_coordinates.size() == 0 ) {
					assert( _buffered_values == nullptr );
				}
#endif
				return _buffered_coordinates.size();
			}

			/** \internal No implementation notes. */
			inline size_t nonzeroes() const noexcept {
#ifndef NDEBUG
				if( _buffered_coordinates.size() == 0 ) {
					assert( _buffered_values == nullptr );
				}
#endif
				return _buffered_coordinates.nonzeroes();
			}

			/** \internal No implementation notes. */
			template< typename OutputType = IOType >
			inline OutputType getNonzeroValue(
				const size_t k,
				const OutputType one
			) const noexcept {
				assert( k < nonzeroes() );
				assert( _buffered_coordinates.size() > 0 );
				if( _buffered_values == nullptr ) {
					return one;
				} else {
					const size_t index = getNonzeroIndex( k );
					return static_cast< OutputType >(
						_buffered_values[ index ]
					);
				}
			}

			/** \internal No implementation notes. */
			inline IOType getNonzeroValue(
				const size_t k
			) const noexcept {
				assert( k < nonzeroes() );
				assert( _buffered_coordinates.size() > 0 );
				assert( _buffered_values != nullptr );
				const size_t index = getNonzeroIndex( k );
				assert( index < _buffered_coordinates.size() );
				return _buffered_values[ index ];
			}

			/** \internal No implementation notes. */
			inline size_t getNonzeroIndex(
				const size_t k
			) const noexcept {
				assert( k < nonzeroes() );
				return _buffered_coordinates.index( k );
			}

	};

} // namespace grb

// parse again for reference_omp backend
#ifdef _GRB_WITH_OMP
 #ifndef _H_GRB_REFERENCE_OMP_PINNEDVECTOR
  #define _H_GRB_REFERENCE_OMP_PINNEDVECTOR
  #define reference reference_omp
  #include "graphblas/reference/pinnedvector.hpp"
  #undef reference
  #undef _H_GRB_REFERENCE_OMP_PINNEDVECTOR
 #endif
#endif

#endif // end ``_H_GRB_REFERENCE_PINNEDVECTOR''

