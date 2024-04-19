
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

#include "imf.hpp"

namespace grb {

	template< typename T, typename Structure, typename Storage, typename View, enum Backend backend >
	class StructuredMatrix;

	/**
	 * Collects all ALP matrix structures.
	 * 
	 * A matrix structure is characterized by having a member type \a inferred_structures.
	 * \a inferred_structures is a tuple used to define a partial order over the 
	 * structures based on their logical implication. So if having structure \f$B\f$ implies
	 * also having structure \f$A\f$ than 
	 * \code
	 * is_same< B::inferred_structures, std::tuple<A, B> >::value == true
	 * \endcode
	 */
	namespace structures {

		template< typename... Tuples >
		struct tuple_cat {
			using type = decltype( std::tuple_cat( std::declval< Tuples >()... ) );
		};

		/**
		 * Check if a structure \a Structure is part of a given \a std::tuple \a Tuple.
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

		struct UpperTriangular;

		struct General {
			using inferred_structures = std::tuple< General >;

			template< typename T, typename Storage, typename View, enum Backend backend >
			static bool isInstantiableFrom( const StructuredMatrix< T, UpperTriangular, Storage, View, backend > & M, grb::imf::IMF & imf_l, grb::imf::IMF & imf_r ) {
				(void)M;
				return imf_l.map( 0 ) <= imf_r.map( imf_r.N );
			}
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

			// Maybe we can consider inheritance here to allow calling checks in base classes.
			// For example, in all cases we should check if IMFs do not overflow the original container.
			// (if it is actually necessary. Maybe we want to assume that the user knows what he is doing)
			template< typename T, typename Storage, typename View, enum Backend backend >
			static bool isInstantiableFrom( const StructuredMatrix< T, UpperTriangular, Storage, View, backend > & M, const grb::imf::IMF & imf_l, const grb::imf::IMF & imf_r ) {
				(void)M;
				return imf_l.isSame(imf_r);
			}
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
