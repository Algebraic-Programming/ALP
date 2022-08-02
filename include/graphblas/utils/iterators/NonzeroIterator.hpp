
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
 * @date 2nd of August, 2017
 */

#ifndef _H_NONZEROITERATOR
#define _H_NONZEROITERATOR

#include <type_traits>
#include <iostream>


namespace grb {

	namespace utils {

		/**
		 * A class that wraps around an iterator around a nested pair and exposes an
		 * ALP matrix iterator of matching types.
		 *
		 * The value type of a given iterator must be
		 *   - <tt>std::pair< std::pair< S1, S2 >, V ></tt>
		 *
		 * @tparam S1 An integral type for indicating row coordinates.
		 * @tparam S2 An integral type for indicating column coordinates.
		 * @tparam V  A nonzero value type.
		 *
		 * @tparam SubIterType The given base iterator.
		 *
		 * This declaration uses SFINAE in order to expose implementations for
		 * supported value types only, based on the given \a SubIterType.
		 */
		template< typename S1, typename S2, typename V, typename SubIterType, class Enable = void >
		class NonzeroIterator;

		/**
		 * \internal Specialisation for types that are direved from the required type.
		 */
		template< typename S1, typename S2, typename V, typename SubIterType >
		class NonzeroIterator<
			S1, S2, V,
			SubIterType,
			typename std::enable_if<
				std::is_base_of<
					typename std::pair< std::pair< S1, S2 >, V >,
					typename SubIterType::value_type
				>::value &&
				std::is_integral< S1 >::value &&
				std::is_integral< S2 >::value
			>::type
		> : public SubIterType {

			public:

				// ALP typedefs

				typedef S1 RowIndexType;
				typedef S2 ColumnIndexType;
				typedef V ValueType;

				// STL typedefs

				typedef typename std::iterator_traits< SubIterType >::value_type value_type;
				typedef typename std::iterator_traits< SubIterType >::pointer pointer;
				typedef typename std::iterator_traits< SubIterType >::reference reference;
				typedef typename std::iterator_traits< SubIterType >::iterator_category
					iterator_category;
				typedef typename std::iterator_traits< SubIterType >::difference_type
					difference_type;

				NonzeroIterator() = delete;

				/** The base constructor. */
				NonzeroIterator( const SubIterType &base ) : SubIterType( base ) {}

				/** Returns the row coordinate. */
				const S1 & i() const {
					return this->operator*().first.first;
				}

				/** Returns the column coordinate. */
				const S2 & j() const {
					return this->operator*().first.second;
				}

				/** Returns the nonzero value. */
				const V & v() const {
					return this->operator*().second;
				}
		};

		/**
		 * \internal Specialisation for pattern matrices.
		 */
		template< typename S1, typename S2, typename SubIterType >
		class NonzeroIterator<
			S1, S2, void,
			SubIterType,
			typename std::enable_if<
				std::is_base_of<
					typename std::pair< S1, S2 >,
					typename SubIterType::value_type
				>::value &&
				std::is_integral< S1 >::value &&
				std::is_integral< S2 >::value
			>::type
		> : public SubIterType {

			public:

				// ALP typedefs

				typedef S1 RowIndexType;
				typedef S2 ColumnIndexType;
				typedef void ValueType;

				// STL typedefs

				typedef typename std::iterator_traits< SubIterType >::value_type value_type;
				typedef typename std::iterator_traits< SubIterType >::pointer pointer;
				typedef typename std::iterator_traits< SubIterType >::reference reference;
				typedef typename std::iterator_traits< SubIterType >::iterator_category
					iterator_category;
				typedef typename std::iterator_traits< SubIterType >::difference_type
					difference_type;

				NonzeroIterator() = delete;

				/** The base constructor. */
				NonzeroIterator( const SubIterType &base ) : SubIterType( base ) {}

				/** Returns the row coordinate. */
				const S1 & i() const {
					return this->operator*().first;
				}

				/** Returns the column coordinate. */
				const S2 & j() const {
					return this->operator*().second;
				}

		};

		/** Creates a nonzero iterator from a given iterator over nested pairs. */
		template< typename S1, typename S2, typename V, typename SubIterType >
		NonzeroIterator< S1, S2, V, SubIterType > makeNonzeroIterator(
			const SubIterType &x
		) {
			return NonzeroIterator< S1, S2, V, SubIterType >( x );
		}

	} // namespace utils

} // namespace grb

#endif // end ``_H_NONZEROITERATOR''

