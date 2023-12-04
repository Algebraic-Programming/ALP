
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
 * @date 20th of July, 2022
 *
 * This is a reorganisation of changes by Alberto Scolari originally made to
 * graphblas/type_traits.hpp.
 */

#ifndef _H_GRB_UTILS_ITERATORS_TYPE_TRAITS
#define _H_GRB_UTILS_ITERATORS_TYPE_TRAITS

#include <type_traits>
#include <iterator>


namespace grb {

	namespace utils {

		/**
		 * Selects the common iterator tag from multiple STL-style iterators.
		 *
		 * The given iterator types may potentially be of different kinds. The most
		 * basic tag is returned.
		 *
		 * This is the version for the base of recursion.
		 *
		 * @tparam IterT1 first iterator type
		 * @tparam IterTs second iterator type
		 */
		template< typename IterT1, typename... IterTs >
		class common_iterator_tag {

			public:

				using iterator_category =
					typename std::iterator_traits< IterT1 >::iterator_category;

		};

		/**
		 * Selects the common iterator tag from multiple STL-style iterators.
		 *
		 * The given iterator types may potentially be of different kinds. The most
		 * basic tag is returned.
		 *
		 * This is the recursive version.
		 *
		 * @tparam IterT1 first iterator type
		 * @tparam IterTs second iterator type
		 */
		template< typename IterT1, typename IterT2, typename... IterTs >
		class common_iterator_tag< IterT1, IterT2, IterTs... > {

			private:

				using cat1 = typename std::iterator_traits< IterT1 >::iterator_category;
				using cats =
					typename common_iterator_tag< IterT2, IterTs... >::iterator_category;


			public:

				// STL iterator tags are a hierarchy with std::forward_iterator_tag at the base
				typedef typename std::conditional<
						std::is_base_of< cat1, cats >::value,
						cat1, cats
					>::type iterator_category;

		};

		/**
		 * Used to gauge whether a given type is an ALP matrix iterator.
		 *
		 * @tparam MatrixValType Value type of the matrix; if void, does not check for
		 *                       the presence of a v() method that returns a nonzero
		 *                       value.
		 *
		 * @tparam IterT         The iterator type.
		 *
		 * An ALP matrix iterator has the following methods:
		 *  -# i(),
		 *  -# j(), and
		 *  -# v(), iff #MatrixValType is not void
		 */
		template< typename MatrixValType, typename IterT >
		class is_alp_matrix_iterator {

			private:

				// helper functions for determining, by return type, whether i, j, and v are
				// defined

				template< typename U >
				static typename std::decay<
					decltype(std::declval< U >().i())
				>::type match_i(
					typename std::decay< decltype(std::declval< U >().i()) >::type*
				) {
					return std::declval< U >().i();
				}

				template< typename U >
				static void match_i( ... ) {}

				template< typename U >
				static typename std::decay<
					decltype(std::declval< U >().j())
				>::type match_j(
					typename std::decay< decltype(std::declval< U >().j()) >::type*
				) {
					return std::declval< U >().j();
				}

				template< typename U >
				static void match_j( ... ) {}

				template< typename U >
				static typename std::decay<
					decltype(std::declval< U >().v())
				>::type match_v(
					typename std::decay< decltype(std::declval< U >().v()) >::type*
				) {
					return std::declval< U >().v();
				}

				template< typename U >
				static void match_v( ... ) {}


			public:

				/** Type of the row index */
				using RowIndexType = decltype( match_i< IterT >( nullptr ) );

				/** Type of the column index */
				using ColumnIndexType = decltype( match_j< IterT >( nullptr ) );

				/** Type of the nonzero value */
				using ValueType = decltype( match_v< IterT >( nullptr ) );

				/**
				 * Whether #IterT is an ALP matrix iterator
				 */
				static constexpr bool value =
					!std::is_same< RowIndexType, void >::value &&
					!std::is_same< ColumnIndexType, void >::value &&
					std::is_integral< RowIndexType >::value &&
					std::is_integral< ColumnIndexType >::value &&
					(
						std::is_same< MatrixValType, void >::value ||
						!std::is_same< ValueType, void >::value
					);

		};

		/**
		 * Checks whether a given ALP matrix iterator type has a .v() method.
		 *
		 * @tparam T the iterator type
		 *
		 * This type trait determines whether \a T can be used for ingesting into a
		 * value (non-void, non-pattern) ALP matrix.
		 */
		template< typename T >
		class has_value_method {

			private:

				template< typename U >
				static typename std::decay< decltype( std::declval< U >().v() ) >::type	match(
					typename std::decay< decltype( std::declval< U >().v() ) >::type*
				) {
					return std::declval< U >().v();
				}

				template< typename U >
				static void match( ... ) {}


			public:

				/**
				 * Whether \a T defines the .v() method and is an ALP matrix iterator.
				 */
				static constexpr bool value = !std::is_same<
						decltype( match< T >( nullptr ) ), void
					>::value &&
					is_alp_matrix_iterator< decltype( match< T >( nullptr ) ), T >::value;

		};

	} // end namespace utils

} // end namespace grb

#endif // end _H_GRB_UTILS_ITERATORS_TYPE_TRAITS

