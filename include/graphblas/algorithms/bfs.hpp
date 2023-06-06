
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
				std::cout << "Vector \"" << name << "\" (" << grb::size( v ) << "):" << std::endl << "[  ";
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
				std::cout << " ]" << std::endl;
#endif
			}

			void debugPrint( const std::string & msg, std::ostream & os = std::cout ) {
#ifdef _DEBUG
				os << msg;
#endif
			}
		} // namespace utils

		template< typename D >
		grb::RC bfs( const Matrix< D > & A, size_t root, size_t & max_level, bool compute_levels, grb::Vector< size_t > & levels, bool compute_parents, grb::Vector< size_t > & parents ) {
			grb::RC rc = grb::RC::SUCCESS;
			const size_t nvertices = grb::nrows( A );

			std::cout << "Running BFS from " << root << " on " << nvertices << " vertices." << std::endl;

			max_level = std::numeric_limits< size_t >::max();
			grb::Vector< bool > x( nvertices, nvertices ), y( nvertices, nvertices );
			grb::set( x, false );
			grb::setElement( x, true, root );
			grb::set( y, x );

			utils::printSparseMatrix( A, "A" );
			utils::printSparseVector( x, "x" );

			if( compute_levels ) {
				grb::resize( levels, nvertices );
				grb::set( levels, std::numeric_limits< size_t >::max() );
				grb::setElement( levels, 0UL, root );
				utils::printSparseVector( levels, "levels" );
			}
			if( compute_parents ) {
				grb::resize( parents, nvertices );
				grb::set( parents, std::numeric_limits< size_t >::max() );
				grb::setElement( parents, root, root );
				utils::printSparseVector( parents, "parents" );
				// TODO:
			}

			for( size_t level = 0; level < nvertices; level++ ) {
				utils::debugPrint( "** Level " + std::to_string( level ) + ":\n" );

				// Multiply the current frontier by the adjacency matrix
				grb::Semiring< grb::operators::logical_or< bool >, grb::operators::logical_and< bool >, grb::identities::logical_false, grb::identities::logical_true > bool_semiring;
				rc = rc ? rc : grb::vxm( y, x, A, bool_semiring, grb::Phase::RESIZE );
				rc = rc ? rc : grb::vxm( y, x, A, bool_semiring, grb::Phase::EXECUTE );

				utils::printSparseVector( x, "x " );
				utils::printSparseVector( y, "y" );

				if( compute_levels ) { // Assign the current level to the newly discovered vertices only
					grb::eWiseLambda(
						[ &levels, &y, level ]( const size_t i ) {
							if( y[ i ] )
								levels[ i ] = std::min( levels[ i ], level + 1 );
						},
						levels, y );
					utils::printSparseVector( levels, "levels" );
				}
				if( compute_parents ) {
					// TODO:
				}

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
		grb::RC bfs( const Matrix< D > & A, size_t root, size_t & max_level ) {
			grb::Vector< size_t > unusued_vec( 0 );
			return bfs( A, root, max_level, false, unusued_vec, false, unusued_vec );
		}

	} // namespace algorithms

} // namespace grb

#endif // _H_GRB_BFS
