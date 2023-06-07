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
		grb::RC sssp( const Matrix< D > & A, size_t root, grb::Vector< size_t > & distances ) {
			grb::RC rc = grb::RC::SUCCESS;
			return rc;
		}

	} // namespace algorithms

} // namespace grb

#endif // _H_GRB_SSSP
