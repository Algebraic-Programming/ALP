
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

#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>

#include "imf.hpp"
#include "views.hpp"


namespace alp {

	template< typename... Tuples >
	struct tuple_cat {
		using type = decltype( std::tuple_cat( std::declval< Tuples >()... ) );
	};


	/**
	 * @brief Compile-time interval [ _left, _right )
	 *
	 * @tparam _left  left boundary of the interval.
	 * @tparam _right right boundary of the interval. Optional, in which case 
	 *                _right = _left + 1.
	 */
	template < std::ptrdiff_t _left, std::ptrdiff_t _right = _left + 1 >
	struct Interval {
		
		static_assert( _left < _right );

		static constexpr std::ptrdiff_t left = _left;
		static constexpr std::ptrdiff_t right = _right;

	};

	template< typename IntervalT >
	struct is_interval: std::false_type { };

	template< std::ptrdiff_t _left, std::ptrdiff_t _right >
	struct is_interval< Interval< _left, _right > >: std::true_type { };

	/**
	 * @brief Compile-time interval [ -inf, _right )
	 */
	template < std::ptrdiff_t _right > 
	using LeftOpenInterval = Interval<std::numeric_limits< std::ptrdiff_t >::min(), _right >;

	/**
	 * @brief Compile-time interval [ _left, +inf ]
	 */
	template < std::ptrdiff_t _left >
	using RightOpenInterval = Interval< _left, std::numeric_limits< std::ptrdiff_t >::max() >;

	/**
	 * @brief Compile-time interval [ -inf, +inf ]
	 */
	typedef Interval<std::numeric_limits< std::ptrdiff_t >::min(), std::numeric_limits< std::ptrdiff_t >::max() > OpenInterval;


	/**
	 * @brief Compile-time transposition of interval [ left, right ).
	 * @typedef type The transposed [ -right + 1, -left + 1 ) interval.
	 */
	template< typename IntervalT, typename = std::enable_if_t< is_interval< IntervalT >::value > >
	struct transpose_interval {
		typedef Interval< -IntervalT::right + 1, -IntervalT::left + 1 > type;
	};

	template< std::ptrdiff_t _right >
	struct transpose_interval< LeftOpenInterval< _right > > {
		typedef RightOpenInterval< -_right + 1 > type;
	};

	template< std::ptrdiff_t _left >
	struct transpose_interval< RightOpenInterval< _left > > {
		typedef LeftOpenInterval< -_left + 1 > type;
	};

	template<>
	struct transpose_interval< OpenInterval > {
		typedef OpenInterval type;
	};

	/**
	 * Checks if a given diagonal belongs to the given interval.
	 */
	template< typename Interval >
	bool is_within_interval( const std::ptrdiff_t diag_offset ) {
		return ( ( diag_offset >= Interval::left ) && ( diag_offset < Interval::right ) );
	}

	namespace internal {

		/**
		 * Checks if a pair of coordinates (i, j) belong to non-zero structure
		 * of the band defined by the band_index and the union of bands.
		 * \note Does not check for matrix dimensions.
		 */

		/** Specialization for out-of-bounds band index */
		template<
			size_t band_index, typename Bands,
			std::enable_if_t<
				band_index >= std::tuple_size< Bands >::value
			> * = nullptr
		>
		bool is_non_zero( const size_t i, const size_t j ) {
			(void)i;
			(void)j;
			return false;
		}

		/** Specialization for within-the-bounds band index */
		template<
			size_t band_index, typename Bands,
			std::enable_if_t<
				band_index < std::tuple_size< Bands >::value
			> * = nullptr
		>
		bool is_non_zero( const size_t i, const size_t j ) {

			using band_interval = typename std::tuple_element< band_index, Bands >::type;

			if( is_within_interval< band_interval >( static_cast< std::ptrdiff_t >( j ) - static_cast< std::ptrdiff_t >( i ) ) ) {
				return true;
			} else {
				return is_non_zero< band_index + 1, Bands >( i, j );
			}
		}

	} // namespace internal

	/**
	 * Checks if a pair of coordinates (i, j) belong to non-zero structure
	 * of the band defined by the band_index and the union of bands.
	 * \note Does not check for matrix dimensions.
	 */
	template< typename Structure >
	bool is_non_zero( const size_t i, const size_t j ) {
		return internal::is_non_zero< 0, typename Structure::band_intervals >( i, j );
	}

	namespace internal {
		/**
		 * @internal Compile-time check if a tuple of intervals is sorted and non-overlapping.
		 * E.g., a pair ( [a, b) [c, d) ) with a < b <= c < d
		 */
		template< typename IntervalTuple >
		struct is_tuple_sorted_non_overlapping;

		template< std::ptrdiff_t _left0, std::ptrdiff_t _right0, std::ptrdiff_t _left1, std::ptrdiff_t _right1, typename... Intervals >
		struct is_tuple_sorted_non_overlapping < std::tuple< Interval< _left0, _right0 >, Interval< _left1, _right1 >, Intervals... > > {
			static constexpr bool value = ( _right0 <= _left1 ) && is_tuple_sorted_non_overlapping< std::tuple< Interval< _left1, _right1 >, Intervals... > >::value;
		};

		template< std::ptrdiff_t _left, std::ptrdiff_t _right >
		struct is_tuple_sorted_non_overlapping < std::tuple< Interval< _left, _right > > > : std::true_type { };

		template< >
		struct is_tuple_sorted_non_overlapping < std::tuple< > > : std::true_type { };

		/**
		 * @internal Compile-time transposition of an interval tuple.
		 * E.g., a pair ( [-2, 3) [4, 6) )
		 * Results in ( [-5, -3) [-2, 3) )
		 */
		template< typename IntervalTuple >
		struct transpose_interval_tuple;

		template< typename IntervalT, typename... Intervals >
		struct transpose_interval_tuple< std::tuple< IntervalT, Intervals... > > {
			typedef tuple_cat< 
				typename transpose_interval_tuple< std::tuple< Intervals... > >::type, 
				std::tuple< typename transpose_interval< IntervalT >::type > 
			> type;
		};

		template< typename IntervalT >
		struct transpose_interval_tuple< std::tuple< IntervalT > > {
			typedef std::tuple< typename transpose_interval< IntervalT >::type > type;
		};

	} // namespace internal

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

		struct BaseStructure {};

		struct UpperTriangular;

		struct General: BaseStructure {

			typedef std::tuple< OpenInterval > band_intervals;

			using inferred_structures = std::tuple< General >;
		};

		struct Square: BaseStructure {

			typedef std::tuple< OpenInterval > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< Square >, General::inferred_structures >::type;
		};

		/**
		 * @brief Static and runtime check to determine if a matrix view of structure TargetStructure
		 * 		  and index mapping functions (IMFs) \a imf_r and \a imf_c can be defined over \a SourceStructure.
		 *
		 * @tparam SourceStructure The underlying structure of the source view.
		 * @tparam TargetStructure The underlying structure of the target view.
		 * @param imf_r            The IMF applied to the rows of the source matrix.
		 * @param imf_c            The IMF applied to the columns of the source matrix.

		 * @return \a false if the function can determined that the new view may alter underlying assumptions
		 * 			associated with the source structure \a SourceStructure; \a true otherwise.
		 */
		template< typename SourceStructure, typename TargetStructure >
		struct isInstantiable {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				(void)imf_r;
				(void)imf_c;
				return false;
			};
		};

		template<>
		struct isInstantiable< General, General > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				(void)imf_r;
				(void)imf_c;
				return true;
			};
		};

		template<>
		struct isInstantiable< UpperTriangular, General > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return imf_r.map( imf_r.n - 1 ) <= imf_c.map( 0 );
			};
		};

		template<>
		struct isInstantiable< UpperTriangular, UpperTriangular > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return imf_r.isSame(imf_c);
			};
		};


		template<>
		struct isInstantiable< General, Square > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return (imf_r.n == imf_c.n);
			};
		};

		template<>
		struct isInstantiable< Square, General > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				(void) imf_r;
				(void) imf_c;
				return (true);
			};
		};

		template<>
		struct isInstantiable< Square, UpperTriangular > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return (imf_r.n == imf_c.n);
			};
		};

		template<>
		struct isInstantiable< Square, Square > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return (imf_r.n == imf_c.n);
			};
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
		struct Band: BaseStructure {

			typedef std::tuple< Intervals... > band_intervals;

			static_assert( alp::internal::is_tuple_sorted_non_overlapping< band_intervals >::value );

			typedef typename tuple_cat< std::tuple< Band< Intervals... > >, General::inferred_structures >::type inferred_structures;
		};

		template < typename IntervalTuple >
		struct tuple_to_band {
			// Can create Band only out of tuple of intervals
			static_assert( sizeof(IntervalTuple *) == 0, "Non-tuple type provided." ); 
		};

		template < typename... Intervals >
		struct tuple_to_band< std::tuple< Intervals... > > {
			typedef Band< Intervals... > type;
		};

		struct Symmetric: BaseStructure {

			typedef std::tuple< OpenInterval > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< Symmetric >, Square::inferred_structures >::type;
		};

		template<>
		struct isInstantiable< General, Symmetric > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return ( imf_r.n == imf_c.n );
			};
		};

		template<>
		struct isInstantiable< Symmetric, General > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return (
					( imf_r.map( imf_r.n - 1 ) <= imf_c.map( 0 ) ) ||
					( imf_c.map( imf_c.n - 1 ) <= imf_r.map( 0 ) )
				);
			};
		};

		template<>
		struct isInstantiable< Symmetric, Symmetric > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return imf_r.isSame(imf_c);
			};
		};

		template<>
		struct isInstantiable< Square, Symmetric > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return (imf_r.n == imf_c.n);
			};
		};

		struct SymmetricPositiveDefinite: BaseStructure {

			typedef std::tuple< OpenInterval > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< SymmetricPositiveDefinite >, Symmetric::inferred_structures >::type;
		};

		template<>
		struct isInstantiable< General, SymmetricPositiveDefinite > : public isInstantiable< General, Symmetric > {
		};

		template<>
		struct isInstantiable< SymmetricPositiveDefinite, General > : public isInstantiable< Symmetric, General > {
		};

		template<>
		struct isInstantiable< SymmetricPositiveDefinite, SymmetricPositiveDefinite > : public isInstantiable< Symmetric, Symmetric > {
		};

		template<>
		struct isInstantiable< Square, SymmetricPositiveDefinite > : public isInstantiable< Square, Symmetric > {
		};

		struct Hermitian: BaseStructure {

			typedef std::tuple< OpenInterval > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< Hermitian >, Square::inferred_structures >::type;
		};

		template<>
		struct isInstantiable< Hermitian, Hermitian > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return imf_r.isSame(imf_c);
			};
		};

		template<>
		struct isInstantiable< Square, Hermitian > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return ( imf_r.n == imf_c.n );
			};
		};

		template<>
		struct isInstantiable< Hermitian, General > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return (
					( imf_r.map( imf_r.n - 1 ) <= imf_c.map( 0 ) ) ||
					( imf_c.map( imf_c.n - 1 ) <= imf_r.map( 0 ) )
				);
			};
		};

		template<>
		struct isInstantiable< General, Hermitian > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return ( imf_r.n == imf_c.n );
			};
		};

		struct HermitianPositiveDefinite: BaseStructure {

			typedef std::tuple< OpenInterval > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< HermitianPositiveDefinite >, Hermitian::inferred_structures >::type;
		};

		template<>
		struct isInstantiable< General, HermitianPositiveDefinite > : public isInstantiable< General, Hermitian > {
		};

		template<>
		struct isInstantiable< HermitianPositiveDefinite, General > : public isInstantiable< Hermitian, General > {
		};

		template<>
		struct isInstantiable< HermitianPositiveDefinite, HermitianPositiveDefinite > : public isInstantiable< Hermitian, Hermitian > {
		};

		template<>
		struct isInstantiable< Square, HermitianPositiveDefinite > : public isInstantiable< Square, Hermitian > {
		};


		struct Trapezoidal: BaseStructure {

			using inferred_structures = tuple_cat< std::tuple< Trapezoidal >, Band< OpenInterval >::inferred_structures >::type;
		};

		struct Triangular: BaseStructure {

			using inferred_structures = tuple_cat< std::tuple< Triangular >, Square::inferred_structures, Trapezoidal::inferred_structures >::type;
		};

		struct LowerTrapezoidal: BaseStructure {

			typedef std::tuple< LeftOpenInterval< 1 > > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< LowerTrapezoidal >, Trapezoidal::inferred_structures >::type;
		};

		struct Diagonal;

		template<>
		struct isInstantiable< LowerTrapezoidal, Diagonal > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				static_assert( std::is_base_of< imf::Strided, ImfR >::value && std::is_base_of< imf::Strided, ImfC >::value );
				return ( ( imf_r.n == imf_c.n ) && ( imf_r.b == imf_c.b ) && ( imf_r.s == imf_c.s ) );
			};
		};

		template<>
		struct isInstantiable< LowerTrapezoidal, LowerTrapezoidal > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return ( imf_c.map( 0 ) <= imf_r.map( imf_r.n - 1 ) );
			};
		};

		template<>
		struct isInstantiable< General, LowerTrapezoidal > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				(void) imf_r;
				(void) imf_c;
				return true;
			};
		};

		template<>
		struct isInstantiable< LowerTrapezoidal, Square  > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return (
					( imf_r.n == imf_c.n ) &&
					( imf_c.map( imf_c.n - 1 ) <= imf_r.map( 0 ) )
				);
			};
		};

		struct LowerTriangular: BaseStructure {

			typedef std::tuple< LeftOpenInterval< 1 > > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< LowerTriangular >, Triangular::inferred_structures, LowerTrapezoidal::inferred_structures >::type;
		};

		template<>
		struct isInstantiable< Square, LowerTriangular > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return (imf_r.n == imf_c.n);
			};
		};

		template<>
		struct isInstantiable< LowerTriangular, LowerTriangular > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return imf_r.isSame(imf_c);
			};
		};


		struct UpperTrapezoidal: BaseStructure {

			typedef std::tuple< RightOpenInterval< 0 > > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< UpperTrapezoidal >, Trapezoidal::inferred_structures >::type;

		};

		template<>
		struct isInstantiable< General, UpperTrapezoidal > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				(void) imf_r;
				(void) imf_c;
				return true;
			};
		};

		struct UpperTriangular: BaseStructure {

			typedef std::tuple< RightOpenInterval< 0 > > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< UpperTriangular >, Triangular::inferred_structures, UpperTrapezoidal::inferred_structures >::type;
		};

		struct Tridiagonal: BaseStructure {

			private:

				typedef Interval< -1, 2 > I;

			public:

				typedef std::tuple< I > band_intervals;

				using inferred_structures = tuple_cat<
					std::tuple< Tridiagonal >,
					Square::inferred_structures,
					Band< I >::inferred_structures
				>::type;
		};

		struct SymmetricTridiagonal: BaseStructure {

			private:

				typedef Interval< -1, 2 > I;

			public:

				typedef std::tuple< I > band_intervals;

				using inferred_structures = tuple_cat<
					std::tuple< SymmetricTridiagonal >,
					Symmetric::inferred_structures,
					Tridiagonal::inferred_structures
				>::type;
		};

		template<>
		struct isInstantiable< SymmetricTridiagonal, SymmetricTridiagonal > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				return imf_r.isSame(imf_c);
			};
		};

		struct HermitianTridiagonal: BaseStructure {

			private:

				typedef Interval< -1, 2 > I;

			public:

				typedef std::tuple< I > band_intervals;

				using inferred_structures = tuple_cat<
					std::tuple< HermitianTridiagonal >,
					Hermitian::inferred_structures,
					Tridiagonal::inferred_structures
				>::type;
		};

		struct Bidiagonal: BaseStructure {
			using inferred_structures = tuple_cat< std::tuple< Bidiagonal >, Triangular::inferred_structures, Tridiagonal::inferred_structures >::type;
		};

		struct RectangularUpperBidiagonal: BaseStructure {

			typedef std::tuple< Interval< 0, 2 > > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< RectangularUpperBidiagonal >, UpperTrapezoidal::inferred_structures	>::type;
		};

		struct RectangularLowerBidiagonal: BaseStructure {

			typedef std::tuple< Interval< -1, 1 > > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< RectangularLowerBidiagonal >, LowerTrapezoidal::inferred_structures	>::type;
		};

		struct LowerBidiagonal: BaseStructure {

			typedef std::tuple< Interval< -1, 1 > > band_intervals;

			using inferred_structures = tuple_cat<
				std::tuple< LowerBidiagonal >,
				RectangularLowerBidiagonal::inferred_structures,
				Bidiagonal::inferred_structures,
				LowerTriangular::inferred_structures
			>::type;
		};

		struct UpperBidiagonal: BaseStructure {

			typedef std::tuple< Interval< 0, 2 > > band_intervals;

			using inferred_structures = tuple_cat<
				std::tuple< UpperBidiagonal >,
				RectangularUpperBidiagonal::inferred_structures,
				Bidiagonal::inferred_structures,
				UpperTriangular::inferred_structures
			>::type;
		};

		struct RectangularDiagonal: BaseStructure {

			typedef std::tuple< Interval< 0 > > band_intervals;

			using inferred_structures = tuple_cat<
				std::tuple< RectangularDiagonal >,
				RectangularLowerBidiagonal::inferred_structures,
				RectangularUpperBidiagonal::inferred_structures
			>::type;
		};

		struct Diagonal: BaseStructure {

			typedef std::tuple< Interval< 0 > > band_intervals;

			using inferred_structures = tuple_cat<
				std::tuple< Diagonal >,
				RectangularDiagonal::inferred_structures,
				LowerBidiagonal::inferred_structures,
				UpperBidiagonal::inferred_structures
			>::type;
		};


		struct FullRank: BaseStructure {

			typedef std::tuple< OpenInterval > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< FullRank >, General::inferred_structures >::type;
		};

		struct NonSingular: BaseStructure {

			typedef std::tuple< OpenInterval > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< NonSingular >, Square::inferred_structures, FullRank::inferred_structures >::type;
		};

		struct OrthogonalColumns: BaseStructure {

			typedef std::tuple< OpenInterval > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< OrthogonalColumns >, FullRank::inferred_structures >::type;
		};

		struct OrthogonalRows: BaseStructure {

			typedef std::tuple< OpenInterval > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< OrthogonalRows >, FullRank::inferred_structures >::type;
		};

		struct Orthogonal: BaseStructure {

			typedef std::tuple< OpenInterval > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< Orthogonal >, NonSingular::inferred_structures, OrthogonalColumns::inferred_structures, OrthogonalRows::inferred_structures >::type;
		};

		template<>
		struct isInstantiable< RectangularDiagonal, Diagonal > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				static_assert( std::is_base_of< imf::Strided, ImfR >::value && std::is_base_of< imf::Strided, ImfC >::value );
				return ( ( imf_r.n == imf_c.n ) && ( imf_r.b == imf_c.b ) && ( imf_r.s == imf_c.s ) );
			};
		};

		template<>
		struct isInstantiable< General, Diagonal > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				static_assert( std::is_base_of< imf::Strided, ImfR >::value && std::is_base_of< imf::Strided, ImfC >::value );
				return ( ( imf_r.n == imf_c.n ) && ( imf_r.b == imf_c.b ) && ( imf_r.s == imf_c.s ) );
			};
		};

		template<>
		struct isInstantiable< Orthogonal, Orthogonal > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				// This check has to be further improved.
				// Orthogonal matrix in the current implementation
				// means full-rank square orthogonal matrix.
				// Rectangular matrix orthogonal only by rows (or columns)
				// does not fit into the current Orthogonal structure.
				return imf_r.n == imf_c.n ;
			};
		};

		template<>
		struct isInstantiable< Orthogonal, General > {
			template< typename ImfR, typename ImfC >
			static bool check( const ImfR &imf_r, const ImfC &imf_c ) {
				(void) imf_r;
				(void) imf_c;
				return true;
			};
		};

		struct Constant: BaseStructure {

			typedef std::tuple< OpenInterval > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< Constant >, General::inferred_structures >::type;
		};

		struct Identity: BaseStructure {

			typedef std::tuple< Interval< 0 > > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< Identity >, FullRank::inferred_structures, Diagonal::inferred_structures, Constant::inferred_structures >::type;
		};

		struct Zero: BaseStructure {

			typedef std::tuple< > band_intervals;

			using inferred_structures = tuple_cat< std::tuple< Zero >, Constant::inferred_structures >::type;
		};

		/**
		 * @brief Checks if TestedStructure is a \a Structure according to the ALP's structure classification.
		 *
		 * @tparam TestedStructure   The structure to be tested.
		 * @tparam Structure 		 The structure that should be implied by \a TestedStructure.
		 */
		template< typename TestedStructure, typename Structure >
		struct is_a {

			static_assert( std::is_base_of< structures::BaseStructure, TestedStructure >::value );

			/**
			 * \a value is true iff \a Structure is implied by \a TestedStructure.
			 */
			static constexpr bool value = is_in< Structure, typename TestedStructure::inferred_structures >::value;
		};

		/**
		 * Exposes the structure obtained by applying a given view onto a given structure.
		 *
		 * By default, the exposed structure is equal to the input structure.
		 * Cases where this is not true shall be specialized.
		 */
		template< enum view::Views view, typename Structure >
		struct apply_view {
			typedef Structure type;
		};

		template<>
		struct apply_view< view::transpose, structures::LowerTriangular >{
			typedef structures::UpperTriangular type;
		};

		template<>
		struct apply_view< view::transpose, structures::UpperTriangular >{
			typedef structures::LowerTriangular type;
		};

		template< typename... Intervals >
		struct apply_view< view::transpose, structures::Band< Intervals... > >{
			typedef structures::tuple_to_band< typename alp::internal::transpose_interval_tuple< std::tuple< Intervals... > >::type > type;
		};

		template< size_t band, typename Structure >
		std::ptrdiff_t get_lower_limit( const size_t rows ) {

			const std::ptrdiff_t m = static_cast< std::ptrdiff_t >( rows );
			constexpr std::ptrdiff_t cl_a = std::tuple_element< band, typename Structure::band_intervals >::type::left;

			const std::ptrdiff_t l_a = ( cl_a < -m + 1 ) ? -m + 1 : cl_a ;

			return l_a;

		}

		template< size_t band, typename Structure >
		std::ptrdiff_t get_upper_limit( const size_t cols ) {

			const std::ptrdiff_t n = static_cast< std::ptrdiff_t >( cols );
			constexpr std::ptrdiff_t cu_a = std::tuple_element< band, typename Structure::band_intervals >::type::right;

			const std::ptrdiff_t u_a = ( cu_a > n ) ? n : cu_a ;

			return u_a;

		}

	} // namespace structures

} // namespace alp

#endif
