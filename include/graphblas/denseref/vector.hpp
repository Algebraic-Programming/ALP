
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

#ifndef _H_GRB_DENSEREF_VECTOR
#define _H_GRB_DENSEREF_VECTOR

#include <graphblas/rc.hpp>
#include <graphblas/utils/alloc.hpp>
#include <graphblas/utils/autodeleter.hpp>

#include <graphblas/denseref/vectoriterator.hpp>

#include <stdexcept>

#include <assert.h>


namespace grb {

	namespace internal {

		template< typename T >
		T * getRaw( Vector< T, reference_dense, void > & ) noexcept;

		template< typename T >
		const T * getRaw( Vector< T, reference_dense, void > & ) noexcept;

		template< typename T >
		size_t getLength( const Vector< T, reference_dense, void > & ) noexcept;

	} // end namespace ``grb::internal''

	/** \internal TODO */
	template< typename T >
	class Vector< T, reference_dense, void > {

		friend T * internal::getRaw< T >( Vector< T, reference_dense, void > & ) noexcept;
		friend const T * internal::getRaw< T >( const Vector< T, reference_dense, void > & ) noexcept;
		friend size_t internal::getLength< T >( const Vector< T, reference_dense, void > & ) noexcept;

		private:

			/** The length of the vector. */
			size_t n;

			/** The vector data. */
			T *__restrict__ data;

			/** Deleter corresponding to #data. */
			utils::AutoDeleter< T > data_deleter;

			/** Whether the container presently is empty (uninitialized). */
			bool empty;


		public:

			/** Exposes the element type. */
			typedef T value_type;

			/** The return type of #operator[](). */
			typedef T& lambda_reference;

			/** The iterator type. */
			typedef typename internal::ConstDenserefVectorIterator< T, reference_dense > const_iterator;

			/**
			 * @param[in] length The requested vector length.
			 *
			 * \internal Allocates a single array of size \a length.
			 */
			Vector( const size_t length ) : n( length ), empty( true ) {
				const RC rc = grb::utils::alloc(
					"grb::Vector< T, reference_dense > (constructor)", "",
					data, length, true, data_deleter
				);
				if( rc == OUTOFMEM ) {
					throw std::runtime_error( "Out-of-memory during Vector< T, reference_dense > construction" );
				} else if( rc != SUCCESS ) {
					throw std::runtime_error( "Unhandled runtime error during Vector< T, reference_dense > construction " );
				}
			}

			/** \internal Simply calls Vector( 0 ). */
			Vector() : Vector( 0 ) {}

			/** \internal Makes a deep copy of \a other. */
			Vector( const Vector< T, reference_dense, void > &other ) : Vector( other.n ) {
				empty = true;
				const RC rc = set( *this, other ); // note: empty will be set to false as part of this call
				if( rc != SUCCESS ) {
					throw std::runtime_error( "grb::Vector< T, reference_dense > (copy constructor): error during call to grb::set (" + toString( rc ) + ")" );
				}
			}

			/** \internal No implementation notes. */
			Vector( Vector< T, reference_dense, void > &&other ) {
				n = other.n; other.n = 0;
				data = other.data; other.data = 0;
				data_deleter = std::move( other.data_deleter );
				empty = other.empty; other.empty = true;
			}

			/** \internal No implementation notes. */
			~Vector() {
				n = 0;
				empty = true;
				// free of data will be handled by #data_deleter
			}

			/** \internal No implementation notes. */
			lambda_reference operator[]( const size_t i ) noexcept {
				assert( i < n );
				assert( !empty );
				return data[ i ];
			}

			/** \internal No implementation notes. */
			const lambda_reference operator[]( const size_t i ) const noexcept {
				assert( i < n );
				assert( !empty );
				return data[ i ];
			}

			/** \internal Relies on #internal::ConstDenserefVectorIterator. */
			const_iterator cbegin() const noexcept {
				return const_iterator( data, n, false );
			}

			/** \internal Relies on #internal::ConstDenserefVectorIterator. */
			const_iterator begin() const noexcept {
				return cbegin();
			}

			/** \internal Relies on #internal::ConstDenserefVectorIterator. */
			const_iterator cend() const noexcept {
				return const_iterator( data, n, true );
			}

			/** \internal Relies on #internal::ConstDenserefVectorIterator. */
			const_iterator end() const noexcept {
				return cend();
			}

	};

	/** Identifies any reference_dense vector as an ALP vector. */
	template< typename T >
	struct is_container< Vector< T, reference_dense, void > > {
		/** A reference_vector is an ALP object. */
		static const constexpr bool value = true;
	};

	namespace internal {

		template< typename T >
		T * getRaw( Vector< T, reference_dense, void > &v ) noexcept {
			return v.data;
		}

		template< typename T >
		const T * getRaw( Vector< T, reference_dense, void > &v ) noexcept {
			return v.data;
		}

		template< typename T >
		size_t getLength( const Vector< T, reference_dense, void > &v ) noexcept {
			return v.n;
		}

	} // end namespace ``grb::internal''

} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_VECTOR''

