
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
 * @file
 *
 * Implements Breadth-First Search (BFS) algorithms.
 *
 * @author B. Lozes
 * @date: May 26th, 2023
 */

#ifndef _H_GRB_BFS
#define _H_GRB_BFS

#include <climits>
#include <numeric>
#include <stack>
#include <vector>

#include <graphblas/utils/iterators/NonzeroIterator.hpp>

#include <graphblas.hpp>

// #define BFS_DEBUG

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

#ifdef BFS_DEBUG
				if( rows > 64 || cols > 64 ) {
					return;
				}
				std::cout << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
				// os.precision( 3 );
				for( size_t y = 0; y < rows; y++ ) {
					os << std::string( 6, ' ' );
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

#ifdef BFS_DEBUG
				if( rows > 64 || cols > 64 ) {
					return;
				}
				std::cout << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
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
				os << "]" << std::endl;
#endif
			}

			template< typename D >
			void printSparseMatrix( const Matrix< D > & mat, const std::string & name ) {
				(void)mat;
				(void)name;
#ifdef BFS_DEBUG
				wait( mat );
				printSparseMatrixIterator( nrows( mat ), ncols( mat ), mat.cbegin(), mat.cend(), name, std::cout );
#endif
			}

			template<>
			void printSparseMatrix< void >( const Matrix< void > & mat, const std::string & name ) {
				(void)mat;
				(void)name;
#ifdef BFS_DEBUG
				wait( mat );
				printSparsePatternMatrixIterator( nrows( mat ), ncols( mat ), mat.cbegin(), mat.cend(), name, std::cout );
#endif
			}

			template< typename D >
			void printSparseVector( const Vector< D > & v, const std::string & name ) {
				(void)v;
				(void)name;
#ifdef BFS_DEBUG
				if( size( v ) > 64 ) {
					return;
				}
				wait( v );
				std::cout << " [ ";
				if( nnz( v ) <= 0 ) {
					for( size_t i = 0; i < size( v ); i++ )
						std::cout << "_ ";
				} else {
					size_t nnz_idx = 0;
					auto it = v.cbegin();
					for( size_t i = 0; i < size( v ); i++ ) {
						if( nnz_idx < nnz( v ) && i == it->first ) {
							std::cout << it->second << " ";
							nnz_idx++;
							if( nnz_idx < nnz( v ) )
								++it;
						} else {
							std::cout << "_ ";
						}
					}
				}
				std::cout << "]  -  "
						  << "Vector \"" << name << "\" (" << size( v ) << ")" << std::endl;
#endif
			}

			template< typename T >
			void printStdVector( const std::vector< T > & vector, const std::string & name ) {
				(void)vector;
				(void)name;
#ifdef BFS_DEBUG
				if( vector.size() > 64 ) {
					return;
				}
				std::cout << " [ ";
				for( const T & e : vector )
					std::cout << e << " ";
				std::cout << "]  -  "
						  << "Vector \"" << name << "\" (" << vector.size() << ")" << std::endl;
#endif
			}

			void debugPrint( const std::string & msg, std::ostream & os = std::cout ) {
				(void)msg;
				(void)os;
#ifdef BFS_DEBUG
				os << msg;
#endif
			}
		} // namespace utils

		/**
		 * @brief Breadth-first search (BFS) algorithm.
		 */
		enum AlgorithmBFS { LEVELS, PARENTS };

		/**
		 * Breadth-first search (BFS) algorithm.
		 * This version computes the first level at which each vertex is reached.
		 *
		 * @tparam D Matrix values type
		 * @tparam T Level type
		 *
		 * @param[in]  A              Matrix to explore
		 * @param[in]  root           Root vertex from which to start the exploration
		 * @param[out] explored_all   Whether all vertices have been explored
		 * @param[out] max_level      Maximum level reached by the BFS algorithm
		 * @param[out] levels         Vector containing the lowest levels at which each vertex is reached.
		 * 						      Needs to be pre-allocated with nrows(A) values.
		 * @param[in]  x 		      Buffer vector, needs to be pre-allocated with 1 value.
		 * @param[in]  y 		      Buffer vector, no pre-allocation needed.
		 * @param[in]  not_visited    Buffer vector, needs to be pre-allocated with nrows(A) values.
		 * @param[in]  max_iterations Max number of iterations to perform (default: -1, no limit)
		 *
		 * \parblock
		 * \par Possible output values:
		 * 	-# max_level: [0, nrows(A) - 1]
		 * 	-# levels: [0, nrows(A) - 1] for each reached vertices, <tt>empty</tt> for unreached vertices
		 * \endparblock
		 *
		 * \note Values of the matrix <tt>A</tt> are ignored, hence it is recommended to use a pattern matrix.
		 */
		template< typename D = void, typename T = size_t >
		RC bfs_levels( const Matrix< D > & A,
			size_t root,
			bool & explored_all,
			T & max_level,
			Vector< T > & levels,
			Vector< bool > & x,
			Vector< bool > & y,
			Vector< bool > & not_visited,
			const long max_iterations = -1L,
			const std::enable_if< std::is_integral< T >::value > * const = nullptr ) {
			RC rc = RC::SUCCESS;
			const size_t nvertices = nrows( A );

			{
				// Frontier vectors
				rc = rc ? rc : setElement( x, true, root );

				utils::printSparseMatrix( A, "A" );
				utils::printSparseVector( x, "x" );

				// Output vector containing the minimum level at which each vertex is reached
				rc = rc ? rc : setElement( levels, static_cast< T >( 0 ), root );
				utils::printSparseVector( levels, "levels" );

				// Vector of unvisited vertices
				rc = rc ? rc : set( not_visited, true );
				rc = rc ? rc : setElement( not_visited, false, root );

				size_t max_iter = max_iterations < 0 ? nvertices : max_iterations;
				for( size_t level = 1; level <= max_iter; level++ ) {
					utils::debugPrint( "** Level " + std::to_string( level ) + ":\n" );
					max_level = level;

					// Multiply the current frontier by the adjacency matrix
					utils::printSparseVector( x, "x" );
					utils::printSparseVector( not_visited, "not_visited" );
					rc = rc ? rc : resize( y, 0UL );
					Semiring< operators::logical_or< bool >, operators::logical_and< bool >, identities::logical_false, identities::logical_true > bool_semiring;
					rc = rc ? rc : vxm( y, not_visited, x, A, bool_semiring, Phase::RESIZE );
					rc = rc ? rc : vxm( y, not_visited, x, A, bool_semiring, Phase::EXECUTE );
					utils::printSparseVector( y, "y" );

					// Update not_visited vector
					for( const std::pair< size_t, bool > e : y ) {
						if( e.second )
							setElement( not_visited, false, e.first );
					}

					// Assign the current level to the newly discovered vertices only
					const Monoid< operators::min< T >, identities::infinity > min_monoid;
					rc = rc ? rc : foldl( levels, y, level, min_monoid, Phase::RESIZE );
					rc = rc ? rc : foldl( levels, y, level, min_monoid, Phase::EXECUTE );
					utils::printSparseVector( levels, "levels" );

					// Check if all vertices have been discovered, equivalent of an std::all on the frontier
					explored_all = nnz( levels ) == nvertices;
					if( explored_all ) {
						// If all vertices are discovered, stop
						utils::debugPrint( "Explored " + std::to_string( level ) + " levels to discover all of the " + std::to_string( nvertices ) + " vertices.\n" );
						return rc;
					}
					bool can_continue = nnz( y ) > 0;
					if( ! can_continue ) {
						max_level = level - 1;
						// If no new vertices are discovered, stop
						utils::debugPrint( "Explored " + std::to_string( level ) + " levels to discover " + std::to_string( nnz( levels ) ) + " vertices.\n" );
						break;
					}

					// Swap the frontier, avoid a copy
					std::swap( x, y );
				}
			}

			// Maximum number of iteration passed, not every vertex has been discovered
			utils::debugPrint( "A full exploration is not possible on this graph. "
							   "Some vertices are not reachable from the given root: " +
				std::to_string( root ) + "\n" );

			return rc;
		}

		/**
		 * Breadth-first search (BFS) algorithm.
		 * This version computes the parents of each vertex.
		 *
		 * @tparam D Matrix values type
		 * @tparam T Parent type
		 *
		 * @param[in]  A              Matrix to explore
		 * @param[in]  root           Root vertex from which to start the exploration
		 * @param[out] explored_all   Whether all vertices have been explored
		 * @param[out] max_level      Maximum level reached by the BFS algorithm
		 * @param[out] parents        Vector containing the parent from which each vertex is reached.
		 * 						      Needs to be pre-allocated with nrows(A) values.
		 * @param[in]  x 		      Buffer vector, needs to be pre-allocated with 1 value.
		 * @param[in]  y 		      Buffer vector, no pre-allocation needed.
		 * @param[in]  max_iterations Max number of iterations to perform (default: -1, no limit)
		 * @param[in]  not_find_value Value to use for vertices that have not been reached (default: -1)
		 *
		 * \parblock
		 * \par Possible output values:
		 * 	-# max_level: [0, nrows(A) - 1]
		 * 	-# parents: [0, nrows(A) - 1] for reached vertices, <tt>not_find_value</tt> for unreached vertices
		 * \endparblock
		 *
		 * \warning Parent type <tt>T</tt> must be a signed integer type.
		 *
		 * \note Values of the matrix <tt>A</tt> are ignored, hence it is recommended to use a pattern matrix.
		 */
		template< 
			typename D = void, 
			typename T = long
		>
		RC bfs_parents( const Matrix< D > & A,
			size_t root,
			bool & explored_all,
			T & max_level,
			Vector< T > & parents,
			Vector< T > & x,
			Vector< T > & y,
			const long max_iterations = -1L,
			const T not_find_value = static_cast< T >( -1 ),
			const std::enable_if< std::is_arithmetic< T >::value && std::is_signed<T>::value, void > * const = nullptr 
		) {
			RC rc = RC::SUCCESS;
			const size_t nvertices = nrows( A );
			utils::printSparseMatrix( A, "A" );

			assert( size( x ) == nvertices );
			assert( size( y ) == nvertices );
			assert( capacity( x ) >= 1 );
			assert( capacity( y ) >= 0 );

			rc = rc ? rc : setElement( x, root, root );
			utils::printSparseVector( x, "x" );
			utils::printSparseVector( y, "y" );

			assert( size(parents) == nvertices );
			assert( capacity(parents) >= nvertices );
			rc = rc ? rc : set( parents, not_find_value );
			rc = rc ? rc : setElement( parents, root, root );
			utils::printSparseVector( parents, "parents" );

			const Semiring< operators::min< T >, operators::add< T >, identities::infinity, identities::zero > semiring;

			size_t max_iter = max_iterations < 0 ? nvertices : max_iterations;
			max_level = 0;
			explored_all = false;
			for( size_t level = 1; level <= max_iter; level++ ) {
				max_level = level;
				utils::debugPrint( "** Level " + std::to_string( level ) + ":\n" );
				rc = rc ? rc : clear( y );
				// utils::printSparseVector( x, "x - before indexing" );
				rc = rc ? rc :
						  eWiseLambda(
							  [ &x ]( const size_t i ) {
								  x[ i ] = i;
							  },
							  x );
				utils::printSparseVector( x, "x - after indexing" );

				rc = rc ? rc : vxm( y, x, A, semiring, Phase::RESIZE );
				rc = rc ? rc : vxm( y, x, A, semiring, Phase::EXECUTE );
				utils::printSparseVector( y, "y - after vxm" );

				const Monoid< operators::max< T >, identities::negative_infinity > max_monoid;
				rc = rc ? rc : foldl( parents, y, max_monoid, Phase::RESIZE );
				rc = rc ? rc : foldl( parents, y, max_monoid, Phase::EXECUTE );
				utils::printSparseVector( parents, "parents" );

				const Monoid< operators::min< T >, identities::zero > all_assigned_monoid;
				T min_parent = std::numeric_limits< T >::max();
				rc = rc ? rc : foldl( min_parent, parents, all_assigned_monoid );
				if( min_parent > not_find_value ) {
					explored_all = true;
					utils::debugPrint( "Explored " + std::to_string( max_level ) + " levels to discover all of the " + std::to_string( nvertices ) + " vertices.\n" );
					break;
				}			

				std::swap( x, y );
			}
			

			return rc;
		}

	} // namespace algorithms

} // namespace grb

#endif // _H_GRB_BFS
