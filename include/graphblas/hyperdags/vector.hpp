
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
	template< typename T >
	class Matrix< T, hyperdags >;

	namespace internal {

		namespace hyperdags {
			typedef grb::internal::Coordinates<
				grb::config::IMPLEMENTATION< grb::hyperdags >::coordinatesBackend()
			> Coordinates;
		}

		template< typename T >
		Vector< T, _GRB_WITH_HYPERDAGS_USING, typename hyperdags::Coordinates > & getVector(
			Vector< T, grb::hyperdags, typename hyperdags::Coordinates > &
		);

		template< typename T >
		const Vector< T, _GRB_WITH_HYPERDAGS_USING, typename hyperdags::Coordinates > & getVector(
			const Vector< T, grb::hyperdags, typename hyperdags::Coordinates > &x
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

			typedef Vector< T, hyperdags, internal::hyperdags::Coordinates > SelfType;
			
			/** \internal Simply use an underlying implementation */
			typedef Vector< T, grb::_GRB_WITH_HYPERDAGS_USING, internal::hyperdags::Coordinates > MyVectorType;

			template< Backend A >
			using ConstIterator = typename MyVectorType::template ConstIterator< A >;

			MyVectorType vector;


		public:

			typedef typename MyVectorType::const_iterator const_iterator;

			Vector( const size_t n ) : vector( n ) {}

			Vector() : Vector( 0 ) {}
			
			Vector( const SelfType &x ) : vector( x.vector ) {}

			Vector( SelfType &&x ) noexcept {
				vector = std::move( x.vector );
			}

			Vector( const size_t n, const size_t nz ) : vector( n, nz ) {}

			SelfType & operator=( const SelfType &x ) noexcept {
				vector = x.vector;
				return *this;
			}

			SelfType & operator=( SelfType &&x ) noexcept {
				vector = std::move( x.vector );
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
			
			T & operator[]( const size_t i ) {
				return vector[ i ];
			}

			T & operator[]( const size_t i ) const {
				return vector[ i ];
			}
			/**
			 * Non-standard data accessor for debug purposes.
			 *
			 * \warning Do not use this fucntion.
			 *
			 * The user promises to never write to this data when GraphBLAS can operate
			 * on it. The user understands that data read out may be subject to incoming
			 * changes caused by preceding GraphBLAS calls.
			 *
			 * \warning This function is only defined for the reference and hyperdags backends--
			 *          thus switching backends may cause your code to not compile.
			 *
			 * @return A const reference to the raw data this vector contains.
			 *
			 * \note This function is used internally for testing purposes.
			 */
			T * raw() const {
				return vector.raw();
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

