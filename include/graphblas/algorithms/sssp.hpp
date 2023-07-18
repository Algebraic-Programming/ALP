/*
 *   Copyright 2023 Huawei Technologies Co., Ltd.
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
 * @brief SSSP (Single-Source Shortest-Path) algorithm.
 *
 * @author B. Lozes
 * @date: June 05th, 2023
 */

#ifndef _H_GRB_SSSP
#define _H_GRB_SSSP

#include <climits>
#include <numeric>
#include <vector>

#include <graphblas/utils/iterators/NonzeroIterator.hpp>

#include <graphblas.hpp>

#define _DEBUG

namespace grb {

	namespace algorithms {

		namespace utils {
			template< class Iterator >
			void printSparseMatrixIterator( size_t rows, size_t cols, Iterator begin, Iterator end, const std::string & name = "", std::ostream & os = std::cout ) {
				(void)rows;
				(void)cols;
				(void)begin;
				(void)end;
				(void)name;
				(void)os;

#ifdef _DEBUG
				std::cout << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
				if( rows > 50 || cols > 50 ) {
					os << "   too large to print" << std::endl;
				} else {
					// os.precision( 3 );
					for( size_t y = 0; y < rows; y++ ) {
						os << std::string( 3, ' ' );
						for( size_t x = 0; x < cols; x++ ) {
							auto nnz_val = std::find_if( begin, end, [ y, x ]( const typename std::iterator_traits< Iterator >::value_type & a ) {
								return a.first.first == y && a.first.second == x;
							} );
							if( nnz_val != end )
								os << std::fixed << ( *nnz_val ).second;
							else
								os << '_';
							os << " ";
						}
						os << std::endl;
					}
				}
				os << "]" << std::endl;
#endif
			}

			template< class Iterator >
			void printSparsePatternMatrixIterator( size_t rows, size_t cols, Iterator begin, Iterator end, const std::string & name = "", std::ostream & os = std::cout ) {
				(void)rows;
				(void)cols;
				(void)begin;
				(void)end;
				(void)name;
				(void)os;

#ifdef _DEBUG
				std::cout << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
				if( rows > 50 || cols > 50 ) {
					os << "   too large to print" << std::endl;
				} else {
					// os.precision( 3 );
					for( size_t y = 0; y < rows; y++ ) {
						os << std::string( 3, ' ' );
						for( size_t x = 0; x < cols; x++ ) {
							auto nnz_val = std::find_if( begin, end, [ y, x ]( const typename std::iterator_traits< Iterator >::value_type & a ) {
								return a.first == y && a.second == x;
							} );
							if( nnz_val != end )
								os << "X";
							else
								os << '_';
							os << " ";
						}
						os << std::endl;
					}
				}
				os << "]" << std::endl;
#endif
			}

			template< typename D >
			void printSparseMatrix( const grb::Matrix< D > & mat, const std::string & name ) {
				grb::wait( mat );
				printSparseMatrixIterator( grb::nrows( mat ), grb::ncols( mat ), mat.cbegin(), mat.cend(), name, std::cout );
			}

			template<>
			void printSparseMatrix< void >( const grb::Matrix< void > & mat, const std::string & name ) {
				grb::wait( mat );
				printSparsePatternMatrixIterator( grb::nrows( mat ), grb::ncols( mat ), mat.cbegin(), mat.cend(), name, std::cout );
			}

			template< typename D >
			void printSparseVector( const grb::Vector< D > & v, const std::string & name ) {
				(void)v;
				(void)name;
#ifdef _DEBUG
				grb::wait( v );
				std::cout << "  [  ";
				if( grb::size( v ) > 50 ) {
					std::cout << "too large to print " << std::endl;
				} else if( grb::nnz( v ) <= 0 ) {
					for( size_t i = 0; i < grb::size( v ); i++ )
						std::cout << "_ ";
				} else {
					size_t nnz_idx = 0;
					for( size_t i = 0; i < grb::size( v ); i++ ) {
						if( nnz_idx < grb::nnz( v ) ) {
							auto found = std::find_if( v.cbegin(), v.cend(), [ i ]( const std::pair<size_t, D> & a ) {
								return a.first == i;
							} );
							if( found != v.cend() ){
								std::cout << std::showpos << found->second << " ";
								nnz_idx++;
								continue;
							}
						} 
						std::cout << "__ ";
					}
				}
				std::cout << " ]  -  "
						  << "Vector \"" << name << "\" (" << grb::size( v ) << ")" << std::endl;
#endif
			}

			template< typename T >
			void printStdVector( const std::vector< T > & vector, const std::string & name ) {
				(void)vector;
				(void)name;
#ifdef _DEBUG
				std::cout << "  [  ";
				if( vector.size() > 50 ) {
					std::cout << "too large to print " << std::endl;
				} else {
					for( const T & e : vector )
						std::cout << e << " ";
				}
				std::cout << " ]  -  "
						  << "Vector \"" << name << "\" (" << vector.size() << ")" << std::endl;
#endif
			}

			void debugPrint( const std::string & msg, std::ostream & os = std::cout ) {
#ifdef _DEBUG
				os << msg;
#endif
			}
		} // namespace utils

		/**
		 * Single-source-shortest-path (SSSP) algorithm.
		 *
		 * This version computes the minimum distance from the root to each vertex.
		 *
		 * @tparam D                Matrix values type
		 * @tparam T                Distance type
		 *
		 *
		 * @param[in]  A                  Matrix to explore
		 * @param[in]  root               Root vertex from which to start the exploration
		 * @param[out] explored_all       Whether all vertices have been explored
		 * @param[out] max_level          Maximum level reached by the BFS algorithm
		 * @param[out] distances          Vector containing the minumum distance to
		 *                                reach each vertex.
		 *                                Needs to be pre-allocated with nrows(A) values.
		 * @param[in]  x                  Buffer vector, needs to be pre-allocated
		 *                                with 1 value.
		 * @param[in]  y                  Buffer vector, no pre-allocation needed.
		 * @param[in]  max_iterations     Max number of iterations to perform
		 *                                (default: -1, no limit)
		 * @param[in]  not_find_distance  Distance to use for vertices that have
		 *                                not been reached (default: -1)
		 *
		 * \parblock
		 * \par Possible output values:
		 *  -# max_level: [0, nrows(A) - 1]
		 *  -# distances: - [0, nrows(A) - 1] for reached vertices,
		 *                - <tt>not_find_distance</tt> for unreached vertices
		 * \endparblock
		 *
		 * \warning Distance type <tt>T</tt> must be a signed integer type.
		 *
		 * \note The matrix <tt>A</tt> can be a pattern matrix, in which case
		 *       the identity of the semiring is used as the weight of each edge.
		 *
		 * \note The distance to the root is set to zero.
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename D,
			typename T = long,
			class MinAddSemiring = Semiring<
				operators::min< T >,
				operators::add< T >,
				identities::infinity,
				identities::zero
			>,
			class MaxMonoid = Monoid<
				operators::max< T >,
				identities::negative_infinity
			>,
			class MinNegativeMonoid = Monoid<
				operators::min< T >,
				identities::zero
			>
		>
		RC sssp(
			const Matrix< D > &A,
			size_t root,
			bool &explored_all,
			size_t &max_level,
			Vector< T > &distances,
			Vector< T > &x,
			Vector< T > &y,
			const long max_iterations = -1L,
			const T not_find_distance = std::numeric_limits< T >::max(),
			const MinAddSemiring semiring = MinAddSemiring(),
			const MaxMonoid max_monoid = MaxMonoid(),
			const MinNegativeMonoid min_negative_monoid = MinNegativeMonoid(),
			const std::enable_if<
				std::is_arithmetic< T >::value
				&& is_semiring< MinAddSemiring >::value
				&& is_monoid< MaxMonoid >::value
				&& is_monoid< MinNegativeMonoid >::value,
				void
			> * const = nullptr
		) {
			RC rc = SUCCESS;
			const size_t nvertices = nrows( A );
			utils::printSparseMatrix( A, "A" );

			assert( nrows( A ) == ncols( A ) );
			assert( size( x ) == nvertices );
			assert( size( y ) == nvertices );
			assert( capacity( x ) >= 1 );
			assert( capacity( y ) >= 0 );

			// Resize the output vector and fill it with -1, except for the root node which is set to 0
			rc = rc ? rc : resize( distances, nrows( A ) );
			rc = rc ? rc : set( distances, not_find_distance );
			rc = rc ? rc : setElement( distances, root, static_cast< T >( 0 ) );
			utils::printSparseVector( distances, "distances" );

			// Set x to the root node, initial distance is 0
			rc = rc ? rc : setElement( x, static_cast< T >( 0 ), root );
			utils::printSparseVector( x, "x" );
			rc = rc ? rc : set( y, static_cast< T >( 0 ) );
			utils::printSparseVector( y, "y" );

			size_t max_iter = max_iterations < 0 ? nvertices : max_iterations;
			max_level = 0;
			explored_all = false;
			for( size_t level = 1; level <= max_iter; level++ ) {
#ifdef _DEBUG
				std::cout << "** Level " << level << ":" << std::endl << std::flush;
#endif
				max_level = level;

				rc = rc ? rc : clear( y );

				utils::printSparseVector( x, "x" );
				rc = rc ? rc : vxm< descr >( y, x, A, semiring, Phase::RESIZE );
				rc = rc ? rc : vxm< descr >( y, x, A, semiring, Phase::EXECUTE );
				utils::printSparseVector( y, "y" );

				rc = rc ? rc : foldl( distances, y, y, operators::min< T >(), Phase::RESIZE );
				rc = rc ? rc : foldl( distances, distances, y, operators::min< T >(), Phase::EXECUTE );
				utils::printSparseVector( distances, "distances" );

				T max_distance = 0;
				rc = rc ? rc : foldl( max_distance, distances, max_monoid );
				if( max_distance < not_find_distance ) {
					explored_all = true;
#ifdef _DEBUG
					std::cout << "Explored " << level << " levels to discover all of the "
								<< nvertices << " vertices.\n" << std::flush;
#endif
					break;
				}

				std::swap( x, y );
			}

			return rc;
		}

	} // namespace algorithms

} // namespace grb

#endif // _H_GRB_SSSP
