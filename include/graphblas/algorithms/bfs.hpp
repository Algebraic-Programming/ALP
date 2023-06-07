
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

		template< typename D >
		grb::RC bfs_levels( const Matrix< D > & A, size_t root, size_t & max_level, grb::Vector< size_t > & levels ) {
			grb::RC rc = grb::RC::SUCCESS;
			const size_t nvertices = grb::nrows( A );

			std::cout << std::endl << "==== Running BFS (levels) from root " << root << " on " << nvertices << " vertices ====" << std::endl;

			max_level = std::numeric_limits< size_t >::max();
			grb::Vector< bool > x( nvertices ), y( nvertices );
			rc = rc ? rc : grb::set( x, false );
			rc = rc ? rc : grb::setElement( x, true, root );
			rc = rc ? rc : grb::set( y, x );

			utils::printSparseMatrix( A, "A" );
			utils::printSparseVector( x, "x" );

			rc = rc ? rc : grb::resize( levels, nvertices );
			rc = rc ? rc : grb::set( levels, std::numeric_limits< size_t >::max() );
			rc = rc ? rc : grb::setElement( levels, 0UL, root );
			utils::printSparseVector( levels, "levels" );

			for( size_t level = 0; level < nvertices; level++ ) {
				utils::debugPrint( "** Level " + std::to_string( level ) + ":\n" );

				// Multiply the current frontier by the adjacency matrix
				grb::Semiring< grb::operators::logical_or< bool >, grb::operators::logical_and< bool >, grb::identities::logical_false, grb::identities::logical_true > bool_semiring;
				rc = rc ? rc : grb::vxm( y, x, A, bool_semiring, grb::Phase::RESIZE );
				rc = rc ? rc : grb::vxm( y, x, A, bool_semiring, grb::Phase::EXECUTE );

				utils::printSparseVector( x, "x" );
				utils::printSparseVector( y, "y" );

				// Assign the current level to the newly discovered vertices only
				rc = rc ? rc :
						  grb::eWiseLambda(
							  [ &levels, &y, level ]( const size_t i ) {
								  if( y[ i ] )
									  levels[ i ] = std::min( levels[ i ], level + 1 );
							  },
							  levels, y );
				utils::printSparseVector( levels, "levels" );

				// Check if all vertices have been discovered, equivalent of an std::all on the frontier
				bool all_visited = true;
				rc = rc ? rc : grb::foldl( all_visited, y, grb::Monoid< grb::operators::logical_and< bool >, grb::identities::logical_true >() );
				if( all_visited ) {
					// If all vertices are discovered, stop
					utils::debugPrint( "Explored " + std::to_string( level + 1 ) + " levels to discover all of the " + std::to_string( nvertices ) + " vertices.\n" );
					max_level = level + 1;
					return rc;
				}

				// Swap the frontier, avoid a copy
				std::swap( x, y );
			}

			// Maximum number of iteration passed, not every vertex has been discovered
			utils::debugPrint( "A full exploration is not possible on this graph. "
							   "Some vertices are not reachable from the given root: " +
				std::to_string( root ) + "\n" );

			return rc;
		}

		template< typename D >
		grb::RC bfs_parents( const Matrix< D > & A, size_t root, size_t & max_level, grb::Vector< long > & parents ) {
			grb::RC rc = grb::RC::SUCCESS;
			const size_t nvertices = grb::nrows( A );

			std::cout << std::endl << "==== Running BFS (parents) from root " << root << " on " << nvertices << " vertices ====" << std::endl;

			utils::printSparseMatrix( A, "A" );

			max_level = std::numeric_limits< size_t >::max();
			grb::Vector< bool > x( nvertices ), y( nvertices );
			rc = rc ? rc : grb::set( x, false );
			rc = rc ? rc : grb::set( y, false );

			utils::printSparseVector( x, "x" );
			utils::printSparseVector( y, "y" );

			rc = rc ? rc : grb::resize( parents, nvertices );
			grb::set( parents, -1L );
			grb::setElement( parents, root, root );
			utils::printSparseVector( parents, "parents" );

			std::vector< bool > visited( nvertices, false );
			std::vector< size_t > to_visit_current_level, to_visit_next_level;
			to_visit_current_level.push_back( root );
			utils::printStdVector( to_visit_current_level, "to_visit_current_level" );

			for( size_t level = 0; level < nvertices; level++ ) {
				utils::debugPrint( "** Level " + std::to_string( level ) + ":\n" );

				const grb::Semiring< grb::operators::logical_or< bool >, grb::operators::logical_and< bool >, grb::identities::logical_false, grb::identities::logical_true > bool_semiring;

				for( size_t visiting : to_visit_current_level ) {
					visited[ visiting ] = true;
					utils::debugPrint( "  Visiting " + std::to_string( visiting ) + "\n" );

					grb::set( x, false );
					grb::setElement( x, true, visiting );
					utils::printSparseVector( x, "x" );
					grb::set( y, false );

					// Multiply the current frontier by the adjacency matrix
					rc = rc ? rc : grb::vxm( y, x, A, bool_semiring, grb::Phase::RESIZE );
					rc = rc ? rc : grb::vxm( y, x, A, bool_semiring, grb::Phase::EXECUTE );
					utils::printSparseVector( y, "y" );

					// Assign the current level to the newly discovered vertices only
					const grb::Semiring<grb::operators::right_assign_if<size_t>, grb::operators::max<size_t>, grb::identities::zero, grb::identities::negative_infinity> assign_if_semiring;
					rc = rc ? rc : grb::eWiseAdd( parents, y, parents, visiting, assign_if_semiring, grb::Phase::RESIZE );
					rc = rc ? rc : grb::eWiseAdd( parents, y, parents, visiting, assign_if_semiring, grb::Phase::EXECUTE );
					utils::printSparseVector( parents, "parents" );

					// Add the newly discovered vertices to the frontier
					for( std::pair< size_t, bool > pair : y )
						if( pair.second && ! visited[ pair.first ] )
							to_visit_next_level.push_back( pair.first );
					utils::printStdVector( to_visit_next_level, "to_visit_next_level" );
				}

				if( to_visit_next_level.empty() ) {
					// If all vertices are discovered, stop
					utils::debugPrint( "Explored " + std::to_string( level + 1 ) + " levels to discover all of the " + std::to_string( nvertices ) + " vertices.\n" );
					max_level = level;
					return rc;
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
