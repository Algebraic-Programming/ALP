
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
 * @date 17th of January, 2022
 */

#ifndef _H_GRB_DENSEREF_VECTOR_ITERATOR
#define _H_GRB_DENSEREF_VECTOR_ITERATOR

#include <graphblas/backends.hpp>

#include <iterator>


namespace grb {

	namespace internal {

		template< typename T, Backend spmd_backend = reference_dense >
		class ConstDenserefVectorIterator : public std::iterator<
			std::random_access_iterator_tag,
			std::pair< const size_t, const T >,
			size_t
		> {
			friend class Vector< T, reference_dense, void >;

			private:

				using ActiveDistribution = internal::Distribution< spmd_backend >;

				const T *__restrict__ data;

				size_t pos;

				size_t n;

				size_t s;

				size_t P;

				std::pair< size_t, T > currentEntry;

				ConstDenserefVectorIterator(
					const T * const data_in, const size_t n_in, const bool end,
					const size_t processID = 0, const size_t numProcesses = 1
				) noexcept :
					data( data_in ), pos( 0 ), n( n_in ),
					s( processID ), P( numProcesses )
				{
					assert( P > 0 );
					assert( s < P );
					if( !end ) {
						pos = 0;
						if( n > 0 ) {
							currentEntry.first = 0;
							currentEntry.second = data[ 0 ];
						}
					} else {
						pos = n;
					}
				}
				
				ConstDenserefVectorIterator( const T * const data_in,
					const size_t pos_in, const size_t n_in,
					const size_t s_in, const size_t P_in
				) noexcept : data( data_in ), pos( pos_in ), n( n_in ),
					s( s_in ), P( P_in )
				{
					if( pos < n && n > 0 ) {
						currentEntry.first = pos;
						currentEntry.second = data[ pos ];
					}
				}


			public:

				ConstDenserefVectorIterator() noexcept :
					ConstDenserefVectorIterator( nullptr, 0, true )
				{}

				ConstDenserefVectorIterator( const ConstDenserefVectorIterator< T, spmd_backend > &other ) noexcept :
					data( other.data ), pos( other.pos ), n( other.n ),
					s( other.s ), P( other.P ),
					currentEntry( other.currentEntry )
				{}

				ConstDenserefVectorIterator( ConstDenserefVectorIterator< T, spmd_backend > &&other ) noexcept :
					data( other.data ), pos( other.pos ), n( other.n ),
					s( other.s ), P( other.P )
				{
					other.data = nullptr; other.pos = 0; other.n = 0;
					currentEntry = std::move( other.currentEntry );
				}

				ConstDenserefVectorIterator< T, spmd_backend >& operator=( const ConstDenserefVectorIterator< T, spmd_backend > &other ) noexcept {
					data = other.data; pos = other.pos; n = other.n;
					assert( s == other.s ); assert( P == other.P );
					currentEntry = other.currentEntry;
					return *this;
				}

				ConstDenserefVectorIterator< T, spmd_backend >& operator=( ConstDenserefVectorIterator< T, spmd_backend > &&other ) noexcept {
					data = other.data; other.data = nullptr;
					pos = other.pos; other.pos = 0;
					n = other.n; other.n = 0;
					assert( s == other.s ); assert( P == other.P );
					currentEntry = std::move( other.currentEntry );
					return *this;
				}

				bool operator==( const ConstDenserefVectorIterator< T, spmd_backend > &other ) const noexcept {
					assert( data == other.data ); assert( n == other.n );
					assert( s == other.s ); assert( P == other.P );
					return pos == other.pos;
				}
				
				bool operator!=( const ConstDenserefVectorIterator< T, spmd_backend > &other ) const noexcept {
					assert( data == other.data ); assert( n == other.n );
					assert( s == other.s ); assert( P == other.P );
					return pos != other.pos;
				}

				bool operator<( const ConstDenserefVectorIterator< T, spmd_backend > &other ) const noexcept {
					assert( data == other.data ); assert( n == other.n );
					assert( s == other.s ); assert( P == other.P );
					return pos < other.pos;
				}

				std::pair< const size_t, const T > operator[]( const size_t i ) {
					assert( pos + i < n );
					assert( n > 0 );
					std::pair< size_t, T > ret;
					ret.first = pos + i;
					ret.second = data[ pos + i ];
					return ret;
				}

				std::pair< const size_t, const T >& operator*() const noexcept {
					assert( n > 0 );
					assert( pos < n );
					return currentEntry;
				}

				const std::pair< size_t, T >* operator->() const noexcept {
					assert( n > 0 );
					assert( pos < n );
					return &currentEntry;
				}

				ConstDenserefVectorIterator< T, spmd_backend >& operator+=( const size_t i ) noexcept {
					assert( pos + i <= n );
					pos = std::max( n, pos + i );
					if( n > 0 && pos < n ) {
						currentEntry.first = pos;
						currentEntry.second = data[ pos ];
					}
					return *this;
				}
				
				ConstDenserefVectorIterator< T, spmd_backend >& operator-=( const size_t i ) noexcept {
					assert( i >= pos );
					if( i > pos ) {
						pos = n;
					} else {
						pos -= i;
					}
					assert( pos <= n );
					if( n > 0 && pos < n ) {
						currentEntry.first = pos;
						currentEntry.second = data[ pos ];
					}
					return *this;
				}

				ConstDenserefVectorIterator< T, spmd_backend >& operator++() noexcept {
					return operator+=( 1 );
				}

				ConstDenserefVectorIterator< T, spmd_backend >& operator--() noexcept {
					return operator-=( 1 );
				}

				ConstDenserefVectorIterator< T, spmd_backend > operator+( const ConstDenserefVectorIterator< T, spmd_backend > &other ) noexcept {
					assert( data == other.data );
					assert( pos + other.pos < n );
					assert( n == other.n );
					assert( s == other.s ); assert( P == other.P );
					const size_t newPos = std::max( pos + other.pos, n );
					return ConstDenserefVectorIterator< T, spmd_backend >( data, pos, n, s, P );
				}

				ConstDenserefVectorIterator< T, spmd_backend > operator-( const ConstDenserefVectorIterator< T, spmd_backend > &other ) noexcept {
					assert( data == other.data );
					assert( pos >= other.pos );
					assert( n == other.n );
					assert( s == other.s ); assert( P == other.P );
					const size_t newPos = pos >= other.pos ? pos - other.pos : n;
					return ConstDenserefVectorIterator< T, spmd_backend >( data, pos, n, s, P );
				}

		};

	} // end namespace ``grb::internal''

} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_VECTOR_ITERATOR''

