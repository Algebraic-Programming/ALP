
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
 * @file 
 * 
 * This file registers all matrix structures that are either
 * implemented, under implementation, or were at any point in time
 * conceived and noteworthy enough to be recorded for future consideration.
 * 
 * A structure 
 */

#ifndef _H_ALP_STRUCTURES
#define _H_ALP_STRUCTURES

#include <tuple>
#include <type_traits>

#include "imf.hpp"

namespace alp {

	/**
	 * @brief Compile-time interval [ _left, _right )
	 *
	 * @tparam _left  left boundary of the interval.
	 * @tparam _right right boundary of the interval. Optional, in which case 
	 *                _left == _right
	 */
	template <int _left, int _right = _left + 1 >
	struct Interval {
		
		static_assert( _left < _right );

		static constexpr int left = _left;
		static constexpr int right = _right;

	};

	/**
	 * @brief Compile-time interval [ -inf, _right )
	 */
	template < int _right > 
	using LeftOpenInterval = Interval<std::numeric_limits<int>::min(), _right >;

	/**
	 * @brief Compile-time interval [ _left, +inf ]
	 */
	template <int _left >
	using RightOpenInterval = Interval< _left, std::numeric_limits<int>::max() >;

	/**
	 * @brief Compile-time interval [ -inf, +inf ]
	 */
	typedef Interval<std::numeric_limits<int>::min(), std::numeric_limits<int>::max() > OpenInterval;
	
	namespace internal {
		/**
		 * @internal Compile-time check if a tuple of intervals is sorted and non-overlapping.
		 * E.g., a pair ( [a,b) [c, d) ) with a < b <= c < d
		 */
		template< typename IntervalTuple >
		struct is_tuple_sorted_non_overlapping;

		template< int _left0, int _right0, int _left1, int _right1, typename... Intervals >
		struct is_tuple_sorted_non_overlapping < std::tuple< Interval< _left0, _right0 >, Interval< _left1, _right1 >, Intervals... > > {
			static constexpr bool value = ( _right0 <= _left1 ) && is_tuple_sorted_non_overlapping< std::tuple< Interval< _left1, _right1 >, Intervals... > >::value;
		};

		template< int _left, int _right >
		struct is_tuple_sorted_non_overlapping < std::tuple< Interval< _left, _right > > > : std::true_type { };

		template< >
		struct is_tuple_sorted_non_overlapping < std::tuple< > > : std::true_type { };

	}

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


		namespace internal {
			/**
			 * @internal WIP interface. Symmetry may be extended so to describe the
			 * direction of the symmetry.
			 */
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
		} // namespace internal 

		struct UpperTriangular;

		struct General {
			using inferred_structures = std::tuple< General >;

			/**
			 * @brief Static and runtime check to determine if a matrix view of structure TargetStructure
			 * 		  and index mapping functions (IMFs) \a imf_l and \a imf_r can be defined over \a SrcStructure.
			 * 
			 * @tparam SrcStructure The underlying structure of the target view.
			 * @param imf_l 		The IMF applied to the rows of the source matrix.
			 * @param imf_r 		The IMF applied to the columns of the source matrix.

			 * @return \a false if the function can determined that the new view may alter underlying assumptions 
			 * 			associated with the source structure \a SrcStructure; \a true otherwise. 
			 */

			template< typename SrcStructure >
			static bool isInstantiableFrom( alp::imf::IMF & imf_l, alp::imf::IMF & imf_r ) {
				return false;
			}
		};

		template<>
		inline bool General::isInstantiableFrom< UpperTriangular >( alp::imf::IMF & imf_l, alp::imf::IMF & imf_r ) {
			return imf_l.map( imf_l.n - 1 ) <= imf_r.map( 0 );
		}

		template<>
		inline bool General::isInstantiableFrom< General >( alp::imf::IMF & imf_l, alp::imf::IMF & imf_r ) {
			(void)imf_l;
			(void)imf_r;
			return true;
		}

		struct Square {
			using inferred_structures = structures::tuple_cat< std::tuple< Square >, General::inferred_structures >::type;
		};

		/**
		 * @brief A Band is a structure described by a compile-time tuple of 
		 *        sorted, non-overlapping integer intervals which 
		 *        list the groups of contiguous non-zero diagonals of a 
		 *        matrix with this structure.
		 *        Different intervals should be described considering the 
		 *        position of the main diagonal as 0 reference. This enables 
		 *        comparing intervals as well as sorting them.
		 *        Subdiagonals have negative positions (the farer from the 
		 *        main diagonal the smaller the position) while superdiagonals 
		 *        have positive ones (the farer from the main diagonal the 
		 *        larger the position).
		 *        E.g., <tt>Band< alp::Interval<-1, 2> ></tt> is a band 
		 *        structure that can be used to describe a tridiagonal matrix.
		 *
		 *        \note <tt>alp::Interval<a, b></tt> uses a past-the-end 
		 *        notation for the intervals [a, b). @see alp::Interval.
		 *
		 *        The first value of the left-most (second value of the right-
		 *        most) interval in the sequence is the lower (upper, resp.) 
		 *        bandwidth (referred to as \a lb and \a ub) of the matrix.
		 *        Such values may be open-ended if limited by the size of the 
		 *        matrix. If the lower bandwith is finite and negative than 
		 *        the number of rows
		 *        \f$m \f$ at runtime must ensure \f$m > |lb| \f$. 
		 *        Similarly, if the upper bandwith is finite and positive 
		 *        than the number of columns 
		 *        \f$n \f$ at runtime must ensure \f$n >= ub \f$. 
		 *        The concept of <tt> Band< OpenInterval > </tt> is a very 
		 *        general notion of Band and may be used for inference purposes 
		 *        (e.g., checking if a matrix is a Band matrix irrespective 
		 *        of specific bands in the structure).
		 *
		 * @tparam Intervals One or more \a alp::Interval types specifying the 
		 *                   bands of the structure. These intervals should be 
		 *                   non-overlapping and sorted according to the above 
		 *                   assumption that all intervals are defined assuming 
		 *                   the main diagonal has position zero.
		 *                   \a alp::LeftOpenInterval ( \a alp::RightOpenInterval) 
		 *                   can be used to indicate that the left bandwidth 
		 *                   (right bandwidth, respectively) is defined by the 
		 *                   size of the matrix at runtime.
		 *
		 */
		template < typename... Intervals >
		struct Band {

			typedef std::tuple< Intervals... > band_intervals;

			static_assert( alp::internal::is_tuple_sorted_non_overlapping< band_intervals >::value );

			typedef typename structures::tuple_cat< std::tuple< Band< Intervals... > >, General::inferred_structures >::type inferred_structures;
		};

		struct Symmetric {
			using inferred_structures = structures::tuple_cat< std::tuple< Symmetric >, Square::inferred_structures >::type;
		};

		struct Triangular {

			using inferred_structures = structures::tuple_cat< std::tuple< Triangular >, Square::inferred_structures, Band< OpenInterval >::inferred_structures >::type;
		};

		struct LowerTriangular {

			typedef std::tuple< LeftOpenInterval< 0 > > band_intervals;

			using inferred_structures = structures::tuple_cat< std::tuple< LowerTriangular >, Triangular::inferred_structures >::type;
		};

		struct UpperTriangular {

			typedef std::tuple< RightOpenInterval< 0 > > band_intervals;

			using inferred_structures = structures::tuple_cat< std::tuple< UpperTriangular >, Triangular::inferred_structures >::type;

			// Maybe we can consider inheritance here to allow calling checks in base classes.
			// For example, in all cases we should check if IMFs do not overflow the original container.
			// (if it is actually necessary. Maybe we want to assume that the user knows what he is doing)
			template< typename SrcStructure >
			static bool isInstantiableFrom( const alp::imf::IMF & imf_l, const alp::imf::IMF & imf_r ) {

				static_assert( std::is_same< SrcStructure, UpperTriangular >::value );

				return imf_l.isSame(imf_r);
			}
		};

		struct Tridiagonal {

			typedef std::tuple< Interval< -1, 1 > > band_intervals;

			using inferred_structures = structures::tuple_cat< std::tuple< Tridiagonal >, Square::inferred_structures, Band< OpenInterval >::inferred_structures >::type;
		};

		struct Bidiagonal {
			using inferred_structures = structures::tuple_cat< std::tuple< Bidiagonal >, Triangular::inferred_structures, Tridiagonal::inferred_structures >::type;
		};

		struct LowerBidiagonal {

			typedef std::tuple< Interval< -1, 0 > > band_intervals;

			using inferred_structures = structures::tuple_cat< std::tuple< LowerBidiagonal >, Bidiagonal::inferred_structures, LowerTriangular::inferred_structures >::type;
		};

		struct UpperBidiagonal {

			typedef std::tuple< Interval< 0, 1 > > band_intervals;

			using inferred_structures = structures::tuple_cat< std::tuple< UpperBidiagonal >, Bidiagonal::inferred_structures, UpperTriangular::inferred_structures >::type;
		};

		struct Diagonal {

			typedef std::tuple< Interval< 0 > > band_intervals;

			using inferred_structures = structures::tuple_cat< std::tuple< Diagonal >, LowerBidiagonal::inferred_structures, UpperBidiagonal::inferred_structures >::type;
		};

		struct FullRank {
			using inferred_structures = structures::tuple_cat< std::tuple< FullRank >, General::inferred_structures >::type;
		};

		struct NonSingular {
			using inferred_structures = structures::tuple_cat< std::tuple< NonSingular >, Square::inferred_structures, FullRank::inferred_structures >::type;
		};

		struct OrthogonalColumns {
			using inferred_structures = structures::tuple_cat< std::tuple< OrthogonalColumns >, FullRank::inferred_structures >::type;
		};

		struct OrthogonalRows {
			using inferred_structures = structures::tuple_cat< std::tuple< OrthogonalRows >, FullRank::inferred_structures >::type;
		};

		struct Orthogonal {
			using inferred_structures = structures::tuple_cat< std::tuple< Orthogonal >, NonSingular::inferred_structures, OrthogonalColumns::inferred_structures, OrthogonalRows::inferred_structures >::type;
		};

		struct Constant {
			using inferred_structures = structures::tuple_cat< std::tuple< Constant >, General::inferred_structures >::type;
		};

		struct Identity {
			using inferred_structures = structures::tuple_cat< std::tuple< Identity >, FullRank::inferred_structures, Diagonal::inferred_structures, Constant::inferred_structures >::type;
		};

		struct Zero {
			using inferred_structures = structures::tuple_cat< std::tuple< Zero >, Constant::inferred_structures >::type;
		};

	} // namespace structures

} // namespace alp

#endif
