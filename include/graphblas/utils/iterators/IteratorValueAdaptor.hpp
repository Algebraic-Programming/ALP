
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
 * @dir include/graphblas/utils/iterators
 * Various utilities to work with STL-like iterators and ALP/GraphBLAS iterators:
 * adaptors, partitioning facilities, traits and functions to check compile-time
 * and runtime properties.
 */

/**
 * @file IteratorValueAdaptor.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Definition of an adaptor to extract a given value out of an iterator.
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

			static_assert( std::is_copy_constructible< AdaptorType >::value,
				"AdaptorType must be copy-constructible" );
			static_assert( std::is_copy_assignable< AdaptorType >::value,
				"AdaptorType must be copy-assignable" );

			typedef typename std::decay<
				decltype( *std::declval< AdaptorType >()( *std::declval< InnerIterType >() ) )>::type value_type;
			typedef value_type & reference;
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
			 * Construct a new IteratorValueAdaptor object from an actual iterator.
			 * The adaptor is built via its default constructor.
			 */
			IteratorValueAdaptor( typename std::enable_if< std::is_default_constructible< AdaptorType >::value,
				const InnerIterType & >::type _iter ) :
				iter( _iter ),
				adaptor() {}

			/**
			 * Construct a new IteratorValueAdaptor object from an iterator and an existing adaptor object.
			 */
			IteratorValueAdaptor(
				const InnerIterType &_iter,
				const AdaptorType &_adaptor
			) :
				iter( _iter ),
				adaptor( _adaptor ) {}

			/**
			 * Construct a new Iterator Value Adaptor object from an actual iterator.
			 * The adaptor is built via its default constructor.
			 *
			 * @param _iter the underlying iterator, to be moved
			 */
			IteratorValueAdaptor( typename std::enable_if< std::is_default_constructible< AdaptorType >::value,
				InnerIterType && >::type _iter
			) :
				iter( std::move( _iter ) ),
				adaptor() {}

			/**
			 * Construct a new IteratorValueAdaptor object from an actual iterator
			 * and an existing adaptor object by moving their state.
			 */
			IteratorValueAdaptor(
				InnerIterType &&_iter,
				AdaptorType &&_adaptor
			) :
				iter( std::move( _iter ) ),
				adaptor( std::move( _adaptor ) ) {}

			IteratorValueAdaptor() = delete;

			// since it is an iterator, we MUST have copy and move semantics
			IteratorValueAdaptor( const SelfType & ) = default;

			IteratorValueAdaptor( SelfType && ) = default;

			SelfType& operator=( const SelfType & ) = default;

			SelfType& operator=( SelfType && ) = default;

			bool operator!=( const SelfType & o ) const { return o.iter != iter; }

			bool operator==( const SelfType & o ) const { return ! operator!=( o ); }

			reference operator*() { return *adaptor( *iter ); }

			const reference operator*() const { return *adaptor( *iter ); }

			pointer operator->() { return adaptor( *iter ); }

			const_pointer operator->() const { return adaptor( *iter ); }

			SelfType& operator++() { ++iter; return *this; }

			SelfType & operator+=(
				typename std::enable_if< is_random_access,
				const size_t >::type offset
			) {
				iter += offset;
				return *this;
			}

			difference_type operator-(
				typename std::enable_if< is_random_access,
				const SelfType & >::type other
			) {
				return iter - other.iter;
			}
		};

	} // end namespace utils
} // end namespace grb

#endif // H_GRB_UTILS_ITERATOR_VALUE_ADAPTOR
