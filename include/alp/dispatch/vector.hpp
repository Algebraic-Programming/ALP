
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

#ifndef _H_ALP_DISPATCH_VECTOR
#define _H_ALP_DISPATCH_VECTOR

#include <alp/backends.hpp>
#include <alp/base/vector.hpp>
#include <alp/amf-based/vector.hpp>
#include <alp/reference/vector.hpp>
#include <alp/config.hpp>
#include <alp/storage.hpp>
#include <alp/structures.hpp>
#include <alp/imf.hpp>

namespace alp {

	namespace internal {

		template< typename T >
		T * getRaw( Vector< T, dispatch > & ) noexcept;

		template< typename T >
		const T * getRaw( const Vector< T, dispatch > & ) noexcept;

		template< typename T >
		size_t getLength( const Vector< T, dispatch > & ) noexcept;

		template< typename T >
		const bool & getInitialized( const Vector< T, dispatch > & v ) noexcept;

		template< typename T >
		void setInitialized( Vector< T, dispatch > & v, const bool initialized ) noexcept;


		/**
		 * The dispatch implementation of the ALP/Dense vector.
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
		class Vector< T, dispatch > : public Vector< T, reference > {

			friend T * internal::getRaw< T >( Vector< T, dispatch > & ) noexcept;
			friend const T * internal::getRaw< T >( const Vector< T, dispatch > & ) noexcept;
			friend size_t internal::getLength< T >( const Vector< T, dispatch > & ) noexcept;

			/* ********************
				IO friends
			   ******************** */

			friend const bool & internal::getInitialized< T >( const Vector< T, dispatch > & ) noexcept;

			friend void internal::setInitialized< T >( Vector< T, dispatch > & , bool ) noexcept;

			using Vector< T, reference >::Vector;

		};

		/** Identifies any dispatch internal vector as an internal container. */
		template< typename T >
		struct is_container< internal::Vector< T, dispatch > > : std::true_type {};

	} // end namespace ``alp::internal''

	namespace internal {

		template< typename T >
		T * getRaw( Vector< T, dispatch > &v ) noexcept {
			return v.data;
		}

		template< typename T >
		const T * getRaw( const Vector< T, dispatch > &v ) noexcept {
			return v.data;
		}

		template< typename T >
		size_t getLength( const Vector< T, dispatch > &v ) noexcept {
			return v.n;
		}

		template< typename T >
		const bool & getInitialized( const Vector< T, dispatch > & v ) noexcept {
			return v.initialized;
		}

		template< typename T >
		void setInitialized( Vector< T, dispatch > & v, bool initialized ) noexcept {
			v.initialized = initialized;
		}

		/**
		 * Returns the pointer to the element corresponding to element (0,0)
		 * of the provided vector.
		 *
		 * @tparam MatrixType  Type of the given ALP vector
		 *
		 * @param[in] A        The ALP vector
		 *
		 * @returns Pointer of type VectorType::value_type (a.k.a T)
		 *
		 */
		template<
			typename VectorType,
			std::enable_if_t< alp::is_vector< VectorType >::value > * = nullptr
		>
		typename VectorType::value_type *getRawPointerToFirstElement( VectorType &v ) {
			return &( v[ 0 ] );
		}

		/** const variant */
		template<
			typename VectorType,
			std::enable_if_t< alp::is_vector< VectorType >::value > * = nullptr
		>
		const typename VectorType::value_type *getRawPointerToFirstElement( const VectorType &v ) {
			return &( v[ 0 ] );
		}

		/**
		 * Returns the increment between two consecutive elements in the
		 * internal container of the given ALP vector.
		 *
		 * @tparam VectorType  Type of the given ALP vector
		 *
		 * @param[in] v        The ALP vector
		 *
		 * @returns The increment of type std::ptrdiff_t
		 *
		 */
		template<
			typename VectorType,
			std::enable_if_t< alp::is_vector< VectorType >::value > * = nullptr
		>
		std::ptrdiff_t getIncrement( const VectorType &v ) {
			const typename VectorType::value_type *first_elem_ptr = &( v[ 0 ] );
			const typename VectorType::value_type *second_elem_ptr = &( v[ 1 ] );
			std::ptrdiff_t inc = second_elem_ptr - first_elem_ptr;
			if( inc < 0 ) {
				std::cerr << "Warning: getIncrement: increment is negative.\n";
			}
			return inc;
		}

	} // end namespace ``alp::internal''
} // namespace alp

#endif // end ``_H_ALP_DISPATCH_VECTOR''