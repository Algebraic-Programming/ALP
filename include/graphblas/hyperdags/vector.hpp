
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
 * @date 31st of January, 2022
 */

#ifndef _H_GRB_HYPERDAGS_VECTOR
#define _H_GRB_HYPERDAGS_VECTOR

#include <graphblas/config.hpp>


namespace grb {

	namespace internal {

		namespace hyperdags {
			typedef Coordinates<
				grb::config::IMPLEMENTATION< grb::hyperdags >::coordinatesBackend()
			> Coordinates;
		}

		template< typename T >
		Vector< T, _GRB_WITH_HYPERDAGS_USING, internal::hyperdags::Coordinates > & getVector(
			Vector< T, grb::hyperdags, internal::hyperdags::Coordinates > &
		);

		template< typename T >
		const Vector< T, _GRB_WITH_HYPERDAGS_USING, internal::hyperdags::Coordinates > & getVector(
			const Vector< T, grb::hyperdags, internal::hyperdags::Coordinates > &x
		);

	}

	template< typename T >
	class Vector< T, hyperdags, internal::hyperdags::Coordinates > {

		template< typename A >
		friend Vector< A, _GRB_WITH_HYPERDAGS_USING, internal::hyperdags::Coordinates > & internal::getVector(
			Vector< A, grb::hyperdags, internal::hyperdags::Coordinates > &
		);

		template< typename A >
		friend const Vector< A, _GRB_WITH_HYPERDAGS_USING, internal::hyperdags::Coordinates > & internal::getVector(
			const Vector< A, grb::hyperdags, internal::hyperdags::Coordinates > &
		);


		private:

			/** \internal Simply use an underlying implementation */
			typedef Vector< T, grb::_GRB_WITH_HYPERDAGS_USING, internal::hyperdags::Coordinates > MyVectorType;

			template< Backend A >
			using ConstIterator = typename MyVectorType::template ConstIterator< A >;

			MyVectorType vector;


		public:

			typedef typename MyVectorType::const_iterator const_iterator;

			Vector( const size_t n ) : vector( n ) {}

			Vector() : Vector( 0 ) {}

			Vector( const MyVectorType &x ) : vector( x.vector ) {}

			Vector( MyVectorType &&x ) noexcept {
				x = std::move( x.other );
			}

			MyVectorType & operator=( const MyVectorType &x ) noexcept {
				x = x.other;
				return *this;
			}

			MyVectorType & operator=( MyVectorType &&x ) noexcept {
				x = std::move( x.other );
				return *this;
			}

			~Vector() {}

			template< Backend spmd_backend = reference >
			ConstIterator< spmd_backend > cbegin( const size_t s = 0, const size_t P = 1 ) const {
				return vector.cbegin( s, P );
			}

			template< Backend spmd_backend = reference >
			ConstIterator< spmd_backend > cend( const size_t s = 0, const size_t P = 1 ) const {
				return vector.cend( s, P );
			}

			template< Backend spmd_backend = reference >
			ConstIterator< spmd_backend > begin( const size_t s = 0, const size_t P = 1 ) const {
				return vector.begin( s, P );
			}

			template< Backend spmd_backend = reference >
			ConstIterator< spmd_backend > end( const size_t s = 0, const size_t P = 1 ) const {
				return vector.end( s, P );
			}

	};

	namespace internal {

		template< typename T >
		Vector< T, _GRB_WITH_HYPERDAGS_USING, internal::hyperdags::Coordinates > & getVector(
			Vector< T, grb::hyperdags, internal::hyperdags::Coordinates > &x
		) {
			return x.vector;
		}

		template< typename T >
		const Vector< T, _GRB_WITH_HYPERDAGS_USING, internal::hyperdags::Coordinates > & getVector(
			const Vector< T, grb::hyperdags, internal::hyperdags::Coordinates > &x
		) {
			return x.vector;
		}

	}

}

#endif

