
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

#ifndef _H_GRB_TRIANGLE_ENUMERATION
#define _H_GRB_TRIANGLE_ENUMERATION

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
				} else {
					for( const std::pair< size_t, D > & pair : v )
						std::cout << pair.second << " ";
					std::cout << " ]" << std::endl;
				}
#endif
			}

			void debugPrint( const std::string & msg, std::ostream & os = std::cout ) {
#ifdef _DEBUG
				os << msg;
#endif
			}
		} // namespace utils

		template< typename D >
		RC bfs_steps( size_t & total_steps, const Matrix< D > & A, size_t root ) {
			grb::RC rc = grb::RC::SUCCESS;

			total_steps = ULONG_MAX;
			const size_t nvertices = grb::nrows( A );
			std::cout << "Running BFS from " << root << " on " << nvertices << " vertices." << std::endl;
			grb::Vector< bool > x( nvertices ), y( nvertices );
			grb::set( x, false );
			grb::setElement( x, true, root );
			grb::set( y, x );

			utils::printSparseMatrix( A, "A" );

			grb::Semiring< grb::operators::logical_or< bool >, grb::operators::logical_and< bool >, grb::identities::logical_false, grb::identities::logical_true > bool_semiring;
			grb::Monoid< grb::operators::logical_and< bool >, grb::identities::logical_true > bool_monoid;

			for( size_t depth = 0; depth < nvertices; depth++ ) {
				rc = rc ? rc : grb::vxm( y, x, A, bool_semiring, grb::Phase::RESIZE );
				rc = rc ? rc : grb::vxm( y, x, A, bool_semiring, grb::Phase::EXECUTE );

				utils::debugPrint( "-- Depth " + std::to_string( depth + 1 ) + ":\n" );
				utils::printSparseVector( x, "x" );
				utils::printSparseVector( y, "y" );

				bool all_visited = true;
				rc = rc ? rc : grb::foldl( all_visited, y, bool_monoid );

				if( all_visited ) {
					// If all vertices are discovered, stop
					utils::debugPrint( "Took " + std::to_string( depth + 1 ) + " steps to discover all of the " + std::to_string( nvertices ) + " vertices.\n" );
					total_steps = depth + 1;
					return rc;
				}

				std::swap( x, y );
			}

			// Maximum number of iteration passed, not every vertex has been discovered
			utils::debugPrint( "A full exploration is not possible on this graph. "
							   "Some vertices are not reachable from the given root: " +
				std::to_string( root ) + "\n" );
			return rc;
		}

		template< typename D >
		RC bfs_steps_per_vertex( size_t & total_steps, grb::Vector< size_t > & steps_per_vertex, const Matrix< D > & A, size_t root ) {
			grb::RC rc = grb::RC::SUCCESS;
			const size_t nvertices = grb::nrows( A );

			std::cout << "Running BFS from " << root << " on " << nvertices << " vertices." << std::endl;

			total_steps = ULONG_MAX;
			grb::Vector< bool > x( nvertices ), y( nvertices ), previous_x( nvertices );
			grb::set( x, false );
			grb::setElement( x, true, root );
			grb::set( y, x );
			grb::set( previous_x, false );
			utils::printSparseVector( x, "X - initial" );
			utils::printSparseVector( y, "Y - initial" );

			grb::resize( steps_per_vertex, nvertices );
			grb::set( steps_per_vertex, ULONG_MAX, grb::Phase::EXECUTE );
			grb::setElement( steps_per_vertex, 0UL, root );
			utils::printSparseVector( steps_per_vertex, "steps_per_vertex" );

			utils::printSparseMatrix( A, "A" );

			grb::Semiring< grb::operators::logical_or< bool >, grb::operators::logical_and< bool >, grb::identities::logical_false, grb::identities::logical_true > bool_semiring;
			grb::Monoid< grb::operators::logical_and< bool >, grb::identities::logical_true > bool_monoid;

			grb::Semiring< grb::operators::right_assign_if< size_t >, grb::operators::min< size_t >, grb::identities::one, grb::identities::zero > dist_semiring;

			for( size_t depth = 0; depth < nvertices; depth++ ) {
				rc = rc ? rc : grb::vxm( y, x, A, bool_semiring, grb::Phase::RESIZE );
				rc = rc ? rc : grb::vxm( y, x, A, bool_semiring, grb::Phase::EXECUTE );

				utils::debugPrint( "-- Depth " + std::to_string( depth ) + ":\n" );
				utils::printSparseVector( x, "X " );
				utils::printSparseVector( y, "Y" );

				grb::set( x, y );
				grb::set( previous_x, y );

				grb::eWiseLambda(
					[ &steps_per_vertex, &y, depth ]( const size_t i ) {
						if( y[ i ] )
							steps_per_vertex[ i ] = std::min( steps_per_vertex[ i ], depth + 1 );
					},
					steps_per_vertex, y );

				utils::printSparseVector( steps_per_vertex, "steps_per_vertex" );

				bool all_visited = true;
				rc = rc ? rc : grb::foldl( all_visited, y, bool_monoid );

				if( all_visited ) {
					// If all vertices are discovered, stop
					utils::debugPrint( "Took " + std::to_string( depth + 1 ) + " steps to discover all of the " + std::to_string( nvertices ) + " vertices.\n" );
					total_steps = depth + 1;
					return rc;
				}

				std::swap( x, y );
			}

			// Maximum number of iteration passed, not every vertex has been discovered
			utils::debugPrint( "A full exploration is not possible on this graph. "
							   "Some vertices are not reachable from the given root: " +
				std::to_string( root ) + "\n" );

			return rc;
		}

	} // namespace algorithms

} // namespace grb

#endif // _H_GRB_TRIANGLE_ENUMERATION
