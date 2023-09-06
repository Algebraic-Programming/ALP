
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
 * Contains the hyperdags implementations for the PinnedVector class
 *
 * @author A. Karanasiou
 * @date August 17, 2022
 */

#ifndef _H_GRB_HYPERDAGS_PINNEDVECTOR
#define _H_GRB_HYPERDAGS_PINNEDVECTOR

#include <graphblas/base/pinnedvector.hpp>
#include <graphblas/utils/autodeleter.hpp>

#include "vector.hpp"


namespace grb {

	/** \internal No implementation notes. */
	template< typename IOType >
	class PinnedVector< IOType, hyperdags > {

		private:

			/** This implementation relies on the sub-backend. */
			typedef PinnedVector< IOType, grb::_GRB_WITH_HYPERDAGS_USING >
				MyPinnedVector;

			/** Instance of the underlying backend. */
			MyPinnedVector pinned_vector;


		public:

			/** \internal No implementation notes. */
			PinnedVector() : pinned_vector() {}

			/** \internal No implementation notes. */
			PinnedVector( const PinnedVector< IOType, hyperdags > & ) = default;

			/** \internal No implementation notes. */
			PinnedVector( PinnedVector< IOType, hyperdags > && ) = default;

			/** \internal No implementation notes. */
			PinnedVector(
				const Vector< IOType, hyperdags, internal::hyperdags::Coordinates > &x,
				const IOMode mode
			): pinned_vector( internal::getVector(x), mode ) {};

			/** \internal No implementation notes. */
			~PinnedVector() = default;

			/** \internal No implementation notes. */
			PinnedVector< IOType, hyperdags >& operator=(
					const PinnedVector< IOType, hyperdags > &
				) = default;

			/** \internal No implementation notes. */
			PinnedVector< IOType, hyperdags >& operator=(
					PinnedVector< IOType, hyperdags > &&
				) = default;

			/** \internal No implementation notes. */
			inline size_t size() const noexcept {
				return pinned_vector.size();
			}

			/** \internal No implementation notes. */
			inline size_t nonzeroes() const noexcept {
				return pinned_vector.nonzeroes();
			}

			/** \internal No implementation notes. */
			template< typename OutputType = IOType >
			inline OutputType getNonzeroValue(
				const size_t k,
				const OutputType one
			) const noexcept {
				return pinned_vector.getNonzeroValue( k, one );
			}

			/** \internal No implementation notes. */
			inline IOType getNonzeroValue(
				const size_t k
			) const noexcept {
				return pinned_vector.getNonzeroValue( k );
			}

			/** \internal No implementation notes. */
			inline size_t getNonzeroIndex(
				const size_t k
			) const noexcept {
				return pinned_vector.getNonzeroIndex( k );
			}

	};

} // namespace grb

#endif // end ``_H_GRB_HYPERDAGS_PINNEDVECTOR''

