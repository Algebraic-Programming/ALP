
/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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
 * @file IteratorValueAdaptor.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @brief Adaptor to extract a given value out of an iterator.
 * @date 2022-10-08
 */

#ifndef H_GRB_UTILS_ITERATOR_VALUE_ADAPTOR
#define H_GRB_UTILS_ITERATOR_VALUE_ADAPTOR

#include <type_traits>
#include <iterator>
#include <utility>

namespace grb {
	namespace utils {

		/**
		 * Adaptor for an iterator, to extract the value pointed to by the * operator.
		 * It wraps an iterator under the same interface, using an object of type \a AdaptorType
		 * to adapt the returned value.
		 *
		 * @tparam InnerIterType type of the underlying iterator
		 * @tparam AdaptorType type of the adaptor, to be instantiated by default
		 */
		template<
			typename InnerIterType,
			typename AdaptorType
		> struct IteratorValueAdaptor {

			static_assert( std::is_default_constructible< AdaptorType >::value, "RefType must be default-constructible" );
			static_assert( std::is_copy_constructible< AdaptorType >::value, "RefType must be copy-constructible" );
			static_assert( std::is_copy_assignable< AdaptorType >::value, "RefType must be copy-assignable" );

			typedef decltype( std::declval< AdaptorType >()( *std::declval< InnerIterType >() ) ) reference;
			typedef typename std::decay< reference >::type value_type;
			typedef value_type * pointer;
			typedef const value_type * const_pointer;
			typedef typename std::iterator_traits< InnerIterType >::iterator_category iterator_category;
			typedef typename std::iterator_traits< InnerIterType >::difference_type difference_type;

			static constexpr bool is_random_access = std::is_base_of<
				std::random_access_iterator_tag, iterator_category >::value;

			InnerIterType iter;
			AdaptorType adaptor;

			using SelfType = IteratorValueAdaptor< InnerIterType, AdaptorType >;

			/**
			 * Construct a new Iterator Value Adaptor object fro an actual iterator.
			 * The adaptor is built via its default constructor.
			 *
			 * @param _iter the underlying iterator, to be copied
			 */
			IteratorValueAdaptor(
				const InnerIterType &_iter
			) :
				iter( _iter ),
				adaptor() {}

			/**
			 * Construct a new Iterator Value Adaptor object fro an actual iterator.
			 * The adaptor is built via its default constructor.
			 *
			 * @param _iter the underlying iterator, to be moved
			 */
			IteratorValueAdaptor(
				InnerIterType &&_iter
			) :
				iter( std::move( _iter ) ),
				adaptor() {}

			IteratorValueAdaptor() = delete;

			IteratorValueAdaptor( const SelfType & ) = default;

			IteratorValueAdaptor( SelfType && ) = default;

			SelfType& operator=( const SelfType & ) = default;

			SelfType& operator=( SelfType && ) = default;

			bool operator!=( const SelfType & o ) const { return o.iter != iter; }

			bool operator==( const SelfType & o ) const { return ! operator!=( o ); }

			reference operator*() { return adaptor( *iter ); }

			const reference operator*() const { return adaptor( *iter ); }

			pointer operator->() { return adaptor( *iter ); }

			const_pointer operator->() const { return adaptor( *iter ); }

			SelfType& operator++() { ++iter; return *this; }

			SelfType & operator+=( typename std::enable_if< is_random_access, const size_t >::type offset ) {
				iter += offset;
				return *this;
			}

			difference_type operator-( typename std::enable_if< is_random_access, const SelfType & >::type other ) {
				return iter - other.iter;
			}
		};

	} // end namespace utils
} // end namespace grb

#endif // H_GRB_UTILS_ITERATOR_VALUE_ADAPTOR
