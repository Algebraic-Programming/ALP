
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
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_PINNEDVECTOR
#define _H_GRB_NONBLOCKING_PINNEDVECTOR

#include <graphblas/base/pinnedvector.hpp>
#include <graphblas/utils/autodeleter.hpp>

#include "coordinates.hpp"
#include "vector.hpp"
#include "lazy_evaluation.hpp"


namespace grb {

	namespace internal {

		extern LazyEvaluation le;

	}

	/** The PinnedVector class is based on that of the reference backend */
	template< typename IOType >
	class PinnedVector< IOType, nonblocking > {

		private:

			utils::AutoDeleter< IOType > _raw_deleter;

			utils::AutoDeleter< char > _stack_deleter;

			IOType * _buffered_values;

			internal::Coordinates<
				config::IMPLEMENTATION< nonblocking >::coordinatesBackend()
			> _buffered_coordinates;


		public:

			PinnedVector() : _buffered_values( nullptr ) {}

			PinnedVector(
				const Vector< IOType, nonblocking, internal::Coordinates<
					config::IMPLEMENTATION< nonblocking >::coordinatesBackend()
				> > &x,
				const IOMode mode
			) {
				// The execution of a pipeline that uses the vector is necessary.
				if( internal::getCoordinates(x).size() > 0 ) {
					internal::le.execution( &x );
				}

				_raw_deleter = x._raw_deleter;
				_stack_deleter = x._buffer_deleter;
				_buffered_values = x._raw;
				_buffered_coordinates = x._coordinates;

				(void)mode;
			}

			inline size_t size() const noexcept {
#ifndef NDEBUG
				if( _buffered_coordinates.size() == 0 ) {
					assert( _buffered_values == nullptr );
				}
#endif
				return _buffered_coordinates.size();
			}

			inline size_t nonzeroes() const noexcept {
#ifndef NDEBUG
				if( _buffered_coordinates.size() == 0 ) {
					assert( _buffered_values == nullptr );
				}
#endif
				return _buffered_coordinates.nonzeroes();
			}

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

			inline size_t getNonzeroIndex(
				const size_t k
			) const noexcept {
				assert( k < nonzeroes() );
				return _buffered_coordinates.index( k );
			}

	};

} // namespace grb

#endif // end ``_H_GRB_NONBLOCKING_PINNEDVECTOR''

