
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
 * Provides the vector container for the HyperDAGs backend
 *
 * @author A. N. Yzelman
 * @date 31st of January, 2022
 */

#ifndef _H_GRB_HYPERDAGS_VECTOR
#define _H_GRB_HYPERDAGS_VECTOR

#include <graphblas/config.hpp>
#include <graphblas/base/pinnedvector.hpp>


namespace grb {

	template< typename T, typename RIT, typename CIT, typename NIT >
	class Matrix< T, hyperdags, RIT, CIT, NIT >;

	namespace internal {

		namespace hyperdags {
			typedef grb::internal::Coordinates<
				grb::config::IMPLEMENTATION< grb::hyperdags >::coordinatesBackend()
			> Coordinates;
		}

		template< typename T >
		Vector< T, _GRB_WITH_HYPERDAGS_USING, typename hyperdags::Coordinates > &
		getVector(
			Vector< T, grb::hyperdags, typename hyperdags::Coordinates > &
		);

		template< typename T >
		const Vector< T, _GRB_WITH_HYPERDAGS_USING, typename hyperdags::Coordinates > &
		getVector(
			const Vector< T, grb::hyperdags, typename hyperdags::Coordinates > &x
		);

		template< typename T>
		inline const T * getRaw(
			const Vector<
				T, grb::hyperdags,
				typename internal::hyperdags::Coordinates
			> &x
		);

		template< typename T>
		inline T * getRaw(
			Vector< T, grb::hyperdags, typename internal::hyperdags::Coordinates > &x
		);

	}

	template< typename T >
	class Vector< T, hyperdags, internal::hyperdags::Coordinates > {

		template< typename A >
		friend Vector<
			A, _GRB_WITH_HYPERDAGS_USING,
			internal::hyperdags::Coordinates
		> & internal::getVector(
			Vector< A, grb::hyperdags, internal::hyperdags::Coordinates > &
		);

		template< typename A >
		friend const Vector<
			A, _GRB_WITH_HYPERDAGS_USING,
			internal::hyperdags::Coordinates
		> & internal::getVector(
			const Vector< A, grb::hyperdags, internal::hyperdags::Coordinates > &
		);

		friend class PinnedVector< T, hyperdags >;


		private:

			/** \internal My own type */
			typedef Vector< T, hyperdags, internal::hyperdags::Coordinates > SelfType;

			/** \internal Simply use an underlying implementation */
			typedef Vector<
				T, grb::_GRB_WITH_HYPERDAGS_USING,
				internal::hyperdags::Coordinates
			> MyVectorType;

			/** \internal Iterator type inherited from underlying backend */
			template< Backend A >
			using ConstIterator = typename MyVectorType::template ConstIterator< A >;

			/** \internal Simply wrap around underlying backend */
			MyVectorType vector;

			/** \internal Registers this vector as a source container */
			void register_vector() {
#ifdef _DEBUG
				std::cout << "\t registering vector with pointer " << this << "\n";
#endif
				if( size( vector ) > 0 ) {
					internal::hyperdags::generator.addContainer( getID( vector ) );
				}
			}


		public:

			typedef typename MyVectorType::const_iterator const_iterator;

			Vector( const size_t n ) : vector( n ) {
#ifdef _DEBUG
				std::cout << "Vector (hyperdags) constructor\n";
#endif
				register_vector();
			}

			Vector() : Vector( 0 ) {
#ifdef _DEBUG
				std::cout << "Vector (hyperdags) default constructor\n";
#endif
			}

			Vector( const SelfType &x ) : vector( x.vector ) {
#ifdef _DEBUG
				std::cout << "Vector (hyperdags) copy constructor\n";
#endif
				register_vector();
			}

			Vector( SelfType &&x ) noexcept {
#ifdef _DEBUG
				std::cout << "Vector (hyperdags) move constructor\n";
#endif
				vector = std::move( x.vector );
				register_vector();
			}

			Vector( const size_t n, const size_t nz ) : vector( n, nz ) {
#ifdef _DEBUG
				std::cout << "Vector (hyperdags) capacity constructor\n";
#endif
				register_vector();
			}

			~Vector() {
#ifdef _DEBUG
				std::cout << "Vector (hyperdags) destructor\n";
#endif
			}

			SelfType & operator=( const SelfType &x ) {
#ifdef _DEBUG
				std::cout << "Vector (hyperdags) copy assignment\n";
#endif
				vector = x.vector;
				return *this;
			}

			SelfType & operator=( SelfType &&x ) noexcept {
#ifdef _DEBUG
				std::cout << "Vector (hyperdags) move assignment\n";
#endif
				vector = std::move( x.vector );
				return *this;
			}

			template< Backend spmd_backend = reference >
			ConstIterator< spmd_backend > cbegin(
				const size_t s = 0, const size_t P = 1
			) const {
				return vector.cbegin( s, P );
			}

			template< Backend spmd_backend = reference >
			ConstIterator< spmd_backend > cend(
				const size_t s = 0, const size_t P = 1
			) const {
				return vector.cend( s, P );
			}

			template< Backend spmd_backend = reference >
			ConstIterator< spmd_backend > begin(
				const size_t s = 0, const size_t P = 1
			) const {
				return vector.begin( s, P );
			}

			template< Backend spmd_backend = reference >
			ConstIterator< spmd_backend > end(
				const size_t s = 0, const size_t P = 1
			) const {
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
		Vector<
			T, _GRB_WITH_HYPERDAGS_USING,
			internal::hyperdags::Coordinates
		> & getVector(
			Vector< T, grb::hyperdags, internal::hyperdags::Coordinates > &x
		) {
			return x.vector;
		}

		template< typename T >
		const Vector<
			T, _GRB_WITH_HYPERDAGS_USING,
			internal::hyperdags::Coordinates
		> & getVector(
			const Vector< T, grb::hyperdags, internal::hyperdags::Coordinates > &x
		) {
			return x.vector;
		}

		template< typename T>
		inline const T * getRaw(
			const Vector< T, grb::hyperdags, internal::hyperdags::Coordinates > &x
		) {
			return getRaw(getVector<T>(x));
		};

		template< typename T>
		inline T * getRaw(
			Vector< T, grb::hyperdags, internal::hyperdags::Coordinates > &x
		) {
			return getRaw(getVector<T>(x));
		};

	}

}

#endif

