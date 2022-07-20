
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
	
				using iterator_category = typename std::iterator_traits< IterT1 >::iterator_category;
	
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

	} // end namespace utils

} // end namespace grb

#endif // end _H_GRB_UTILS_ITERATORS_TYPE_TRAITS

