
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

/**
 *
 * @file This file registers all matrix structures that are either
 *       implemented, under implementation, or were at any point in time
 *       conceived and noteworthy enough to be recorded for future consideration.
 */

#ifndef _H_GRB_STRUCTURES
#define _H_GRB_STRUCTURES

#include <tuple>
#include <type_traits>

namespace grb {

	namespace structures {

		template< typename... Tuples >
		struct tuple_cat {
			using type = decltype( std::tuple_cat( std::declval< Tuples >()... ) );
		};

		/**
		 * Check if a structure is part of a given tuple.
		 */
		template< typename Structure, typename Tuple >
		struct is_in;

		template< typename Structure >
		struct is_in< Structure, std::tuple<> > : std::false_type {};

		template< typename Structure, typename TupleHead, typename... Structures >
		struct is_in< Structure, std::tuple< TupleHead, Structures... > > : is_in< Structure, std::tuple< Structures... > > {};

		template< typename Structure, typename... Structures >
		struct is_in< Structure, std::tuple< Structure, Structures... > > : std::true_type {};

		enum SymmetryDirection {

			unspecified,

			north,

			south,

			east,

			west,

			/*
			 * Could specify symmetry with upper access
			 */
			north_west,

			/*
			 * Could specify symmetry with lower access
			 */
			south_east,

			/*
			 * Could specify persymmetry with upper access
			 */
			north_east,

			/*
			 * Could specify persymmetry with lower access
			 */
			south_west
		};

		/**
		 * List of ALP matrix structures.
		 */
		struct General {
			using inferred_structures = std::tuple< General >;
		};

		struct Square {
			using inferred_structures = structures::tuple_cat< std::tuple< Square >, General::inferred_structures >::type;
		};

		struct Symmetric {
			using inferred_structures = structures::tuple_cat< std::tuple< Symmetric >, Square::inferred_structures >::type;
		};

		struct Triangular {
			using inferred_structures = structures::tuple_cat< std::tuple< Triangular >, Square::inferred_structures >::type;
		};

		struct LowerTriangular {
			using inferred_structures = structures::tuple_cat< std::tuple< LowerTriangular >, Triangular::inferred_structures >::type;
		};

		struct UpperTriangular {
			using inferred_structures = structures::tuple_cat< std::tuple< UpperTriangular >, Triangular::inferred_structures >::type;
		};

		struct Diagonal {
			using inferred_structures = structures::tuple_cat< std::tuple< Diagonal >, LowerTriangular::inferred_structures, UpperTriangular::inferred_structures >::type;
		};

		struct FullRank {
			using inferred_structures = structures::tuple_cat< std::tuple< FullRank >, General::inferred_structures >::type;
		};

		struct NonSingular {
			using inferred_structures = structures::tuple_cat< std::tuple< NonSingular >, Square::inferred_structures, FullRank::inferred_structures >::type;
		};

	} // namespace structures

} // namespace grb

#endif
