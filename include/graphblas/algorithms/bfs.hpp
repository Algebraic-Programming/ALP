
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
					auto it = v.cbegin();
					for( size_t i = 0; i < grb::size( v ); i++ ) {
						if( nnz_idx < grb::nnz( v ) && i == it->first ) {
							std::cout << it->second << " ";
							nnz_idx++;
							if( nnz_idx < grb::nnz( v ) )
								++it;
						} else {
							std::cout << "_ ";
						}
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
		   * Breadth-first search (BFS) algorithm.
		   * This version computes the first level at which each vertex is reached.
		   *
		   * @tparam D Matrix values type
		   * @tparam T Level type
		   *
		   * @param[in] A Matrix to explore
		   * @param[in] root Root vertex from which to start the exploration
		   * @param[out] explored_all Whether all vertices have been explored
		   * @param[out] max_level Maximum level reached by the BFS algorithm
		   * @param[out] levels Vector containing the level at which each vertex is reached
		   * @param[in] not_find_value Value to use for vertices that have not been reached (default: -1)
		   * @return grb::SUCCESS A call to this function never fails.
		   *
		   * \parblock
		   * \par Possible output values:
		   * 	-# max_level: [0, grb::nrows(A) - 1]
		   * 	-# levels: [0, grb::nrows(A) - 1] for reached vertices, <tt>not_find_value</tt> for unreached vertices
		   * \endparblock
		   *
		   * \warning Level type <tt>T</tt> must be a signed integer type.
		   *
		   * \note Values of the matrix <tt>A</tt> are ignored, hence it is recommended to use a pattern matrix.
		   */
		template< typename D = void, typename T = long >
		grb::RC bfs_levels( const Matrix< D > & A,
			size_t root,
			bool & explored_all,
			T & max_level,
			grb::Vector< T > & levels,
			const T not_find_value = static_cast< T >( -1 ),
			const std::enable_if< std::is_integral< T >::value && std::is_signed< T >::value > * const = nullptr ) {
			grb::RC rc = grb::RC::SUCCESS;
			const size_t nvertices = grb::nrows( A );

			std::cout << std::endl << "==== Running BFS (levels) from root " << root << " on " << nvertices << " vertices ====" << std::endl;

			{
				// Frontier vectors
				grb::Vector< bool > x( nvertices ), y( nvertices );
				rc = rc ? rc : grb::setElement( x, true, root );

				utils::printSparseMatrix( A, "A" );
				utils::printSparseVector( x, "x" );

				// Output vector containing the minimum level at which each vertex is reached
				rc = rc ? rc : grb::resize( levels, nvertices );
				rc = rc ? rc : grb::setElement( levels, static_cast< T >( 0 ), root );
				utils::printSparseVector( levels, "levels" );

				// Vector of unvisited vertices
				grb::Vector< bool > not_visited( nvertices );
				rc = rc ? rc : grb::set( not_visited, true );
				grb::setElement( not_visited, false, root );

				for( size_t level = 1; level <= nvertices; level++ ) {
					utils::debugPrint( "** Level " + std::to_string( level ) + ":\n" );
					max_level = level;

					// Multiply the current frontier by the adjacency matrix
					utils::printSparseVector( x, "x" );
					utils::printSparseVector( not_visited, "not_visited" );
					grb::resize( y, 0UL );
					grb::Semiring< grb::operators::logical_or< bool >, grb::operators::logical_and< bool >, grb::identities::logical_false, grb::identities::logical_true > bool_semiring;
					rc = rc ? rc : grb::vxm( y, not_visited, x, A, bool_semiring, grb::Phase::RESIZE );
					rc = rc ? rc : grb::vxm( y, not_visited, x, A, bool_semiring, grb::Phase::EXECUTE );
					utils::printSparseVector( y, "y" );

					// Update not_visited vector
					for( const std::pair< size_t, bool > e : y ) {
						if( e.second )
							grb::setElement( not_visited, false, e.first );
					}

					// Assign the current level to the newly discovered vertices only
					const grb::Monoid< grb::operators::min< T >, grb::identities::infinity > min_monoid;
					rc = rc ? rc : grb::foldl( levels, y, level, min_monoid, grb::Phase::RESIZE );
					rc = rc ? rc : grb::foldl( levels, y, level, min_monoid, grb::Phase::EXECUTE );
					utils::printSparseVector( levels, "levels" );

					// Check if all vertices have been discovered, equivalent of an std::all on the frontier
					explored_all = grb::nnz( levels ) == nvertices;
					if( explored_all ) {
						// If all vertices are discovered, stop
						utils::debugPrint( "Explored " + std::to_string( level ) + " levels to discover all of the " + std::to_string( nvertices ) + " vertices.\n" );
						return rc;
					}
					bool can_continue = grb::nnz( y ) > 0;
					if( ! can_continue ) {
						max_level = level - 1;
						// If no new vertices are discovered, stop
						utils::debugPrint( "Explored " + std::to_string( level ) + " levels to discover " + std::to_string( grb::nnz( levels ) ) + " vertices.\n" );
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

			// Fill missing values with -1
			std::vector< bool > not_visited( nvertices, true );
			for( const std::pair< size_t, T > & p : levels )
				not_visited[ p.first ] = false;
			for( size_t i = 0; i < nvertices; i++ )
				if( not_visited[ i ] )
					rc = rc ? rc : grb::setElement( levels, not_find_value, i );

			return rc;
		}

		template< typename D = void, typename T = long >
		grb::RC bfs_parents( const Matrix< D > & A,
			size_t root,
			bool & explored_all,
			T & max_level,
			grb::Vector< T > & parents,
			const T not_find_value = static_cast< T >( -1 ),
			const std::enable_if< std::is_arithmetic< T >::value && std::is_signed< T >::value, void > * const = nullptr ) {
			grb::RC rc = grb::RC::SUCCESS;
			const size_t nvertices = grb::nrows( A );

			std::cout << std::endl << "==== Running BFS (parents) from root " << root << " on " << nvertices << " vertices ====" << std::endl;

			utils::printSparseMatrix( A, "A" );

			grb::Vector< bool > x( nvertices ), y( nvertices );
			utils::printSparseVector( x, "x" );
			utils::printSparseVector( y, "y" );

			rc = rc ? rc : grb::resize( parents, nvertices );
			rc = rc ? rc : grb::set( parents, not_find_value );
			rc = rc ? rc : grb::setElement( parents, root, root );
			utils::printSparseVector( parents, "parents" );

			grb::Vector< bool > not_visited( nvertices );
			rc = rc ? rc : grb::set( not_visited, true );
			std::vector< size_t > to_visit_current_level, to_visit_next_level;
			to_visit_next_level.reserve( nvertices );
			to_visit_current_level.reserve( nvertices );
			to_visit_current_level.push_back( root );
			utils::printStdVector( to_visit_current_level, "to_visit_current_level" );

			max_level = 0;
			for( size_t level = 1; level <= nvertices; level++ ) {
				utils::debugPrint( "** Level " + std::to_string( level ) + ":\n" );
				max_level++;

				for( size_t visiting : to_visit_current_level ) {
					utils::debugPrint( "  Visiting " + std::to_string( visiting ) + "\n" );
					rc = rc ? rc : grb::setElement( not_visited, false, visiting );
					utils::printSparseVector( not_visited, "not_visited" );

					// Explore from the current vertex only
					rc = rc ? rc : grb::setElement( x, true, visiting ); // Explore from the current vertex only
					utils::printSparseVector( x, "x" );
					rc = rc ? rc : grb::resize( y, 0 );                  // Necessary as vxm is in-place
					// Masking vxm to only explore non-explored vertices
					const grb::Semiring< grb::operators::logical_or< bool >, grb::operators::logical_and< bool >, grb::identities::logical_false, grb::identities::logical_true > bool_semiring;
					rc = rc ? rc : grb::vxm( y, not_visited, x, A, bool_semiring, grb::Phase::RESIZE );
					rc = rc ? rc : grb::vxm( y, not_visited, x, A, bool_semiring, grb::Phase::EXECUTE );
					rc = rc ? rc : grb::setElement( x, false, visiting ); // Reset the current vertex to false
					utils::printSparseVector( y, "y" );

					// Assign the current level to the newly discovered vertices only
					const grb::Monoid< grb::operators::max< T >, grb::identities::negative_infinity > max_monoid;
					rc = rc ? rc : grb::foldl( parents, y, visiting, max_monoid, grb::Phase::RESIZE );
					rc = rc ? rc : grb::foldl( parents, y, visiting, max_monoid, grb::Phase::EXECUTE );
					utils::printSparseVector( parents, "parents" );

					// Add the newly discovered vertices to the stack
					// Optimisation possible if an operator::index was available
					for( std::pair< size_t, bool > pair : y )
						if( pair.second )
							to_visit_next_level.push_back( pair.first );
					utils::printStdVector( to_visit_next_level, "to_visit_next_level" );
				}

				if( to_visit_next_level.empty() ) {
					// If all vertices are discovered, stop
					bool not_all_discovered = false;
					rc = rc ? rc : grb::foldl( not_all_discovered, not_visited, grb::Monoid< grb::operators::logical_or< bool >, grb::identities::logical_false >() );
					max_level--;
					if( ! not_all_discovered ) { // If all vertices are discovered, stop
						utils::debugPrint( "Explored " + std::to_string( level ) + " levels to discover all of the " + std::to_string( nvertices ) + " vertices.\n" );
						explored_all = true;
						break;
					}
					explored_all = false;
				}

				std::swap( to_visit_current_level, to_visit_next_level );
				to_visit_next_level.clear();
			}

			// Maximum number of iteration passed, not every vertex has been discovered
			utils::debugPrint( "A full exploration is not possible on this graph. "
							   "Some vertices are not reachable from the given root: " +
				std::to_string( root ) + "\n" );

			return rc;
		}

	} // namespace algorithms

} // namespace grb

#endif // _H_GRB_BFS
