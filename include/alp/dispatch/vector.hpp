
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
	} // end namespace ``alp::internal''
} // namespace alp

#endif // end ``_H_ALP_DISPATCH_VECTOR''
