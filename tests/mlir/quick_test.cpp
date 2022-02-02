
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

#include <iostream>
#include <tuple>
#include <type_traits>

// template< typename ... S >
// struct structure_list {};

// template<>
// struct structure_list<> {};

// template< typename SA, typename SB >
// struct structure_list_cat;

// template< typename ... SA, typename ... SB >
// struct structure_list_cat< structure_list< SA ... >, structure_list< SB ... > >
//     { typedef structure_list< SA ..., SB ... > type; };

template< typename ... t >
struct structure_tuple_cat
    { typedef decltype( std::tuple_cat( std::declval< t >() ... ) ) type; };

struct Square {
  
  using inferred_structures = std::tuple<Square>;

};

struct FullRank {
  
  using inferred_structures = std::tuple<FullRank>;

};

struct NonSingular {
  
  using inferred_structures = structure_tuple_cat< Square::inferred_structures, FullRank::inferred_structures >::type;

};

template <typename Structure, typename List>
struct is_in;

template <typename Structure>
struct is_in<Structure, std::tuple<>> : std::false_type {};

template <typename Structure, typename ListHead, typename... Structures>
struct is_in< Structure, std::tuple<ListHead, Structures...> > : is_in<Structure, std::tuple<Structures...>> {};

template <typename Structure, typename... Structures>
struct is_in<Structure, std::tuple<Structure, Structures...>> : std::true_type {};

template <typename Structure, typename Test>
struct is_a {
  static constexpr bool value = is_in< Test, typename Structure::inferred_structures >::value;
};

int main(int argc, char **argv) {

  std::cout << is_a< NonSingular, Square  >::value << std::endl;
  return 0;
}
