
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

namespace grb {
	namespace utils {

		template< typename S1, typename S2, typename V, typename SubIterType, class Enable = void >
		class NonzeroIterator; 

		template< typename S1, typename S2, typename V, typename SubIterType >
		class NonzeroIterator< S1, S2, V, SubIterType, typename std::enable_if< std::is_base_of< typename std::pair< std::pair< S1, S2 >, V >, typename SubIterType::value_type >::value >::type > :
			public SubIterType {

		private:
		public:
			typedef S1 row_coordinate_type;
			typedef S2 column_coordinate_type;
			typedef V nonzero_value_type;

			NonzeroIterator( const SubIterType & base ) : SubIterType( base ) {}

			const S1 & i() const {
				return this->operator*().first.first;
			}

			const S2 & j() const {
				return this->operator*().first.second;
			}

			const V & v() const {
				return this->operator*().second;
			}
		};

		template< typename S1, typename S2, typename SubIterType >
		class NonzeroIterator< S1, S2, void, SubIterType, typename std::enable_if< std::is_base_of< typename std::pair< S1, S2 >, typename SubIterType::value_type >::value >::type > :
			public SubIterType {

		private:
		public:
			typedef S1 row_coordinate_type;
			typedef S2 column_coordinate_type;

			NonzeroIterator( const SubIterType & base ) : SubIterType( base ) {}

			const S1 & i() const {
				return this->operator*().first;
			}

			const S2 & j() const {
				return this->operator*().second;
			}
		};

		template< typename S1, typename S2, typename V, typename SubIterType >
		NonzeroIterator< S1, S2, V, SubIterType > makeNonzeroIterator( const SubIterType & x ) {
			return NonzeroIterator< S1, S2, V, SubIterType >( x );
		}

	} // namespace utils
} // namespace grb

#endif // end ``_H_NONZEROITERATOR''
