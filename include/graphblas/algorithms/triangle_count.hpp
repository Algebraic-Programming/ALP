
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
 * Implements the triangle counting and triangle enumeration algorithms.
 *
 * @author B. Lozes
 * @date: May 10th, 2023
 */

#ifndef _H_GRB_TRIANGLE_ENUMERATION
#define _H_GRB_TRIANGLE_ENUMERATION

#include <numeric>
#include <vector>

#include <graphblas/utils/iterators/NonzeroIterator.hpp>

#include <graphblas.hpp>

constexpr bool DEBUG = false;

namespace grb {

	namespace algorithms {

		namespace utils {
			template< typename D >
			bool is_diagonal_null( const grb::Matrix< D > & A ) {
				return std::count_if( A.cbegin(), A.cend(), []( const std::pair< std::pair< size_t, size_t >, D > & e ) {
					return e.first.first == e.first.second && e.second != static_cast< D >( 0 );
				} ) == 0;
			}

			template< class Iterator >
			void printSparseMatrixIterator( size_t rows, size_t cols, Iterator begin, Iterator end, const std::string & name = "", std::ostream & os = std::cout ) {
				if( ! DEBUG )
					return;
				std::cout << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
				if( rows > 1000 || cols > 1000 ) {
					os << "   Matrix too large to print" << std::endl;
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
			}

			template< typename D >
			void printSparseMatrix( const grb::Matrix< D > & mat, const std::string & name = "", std::ostream & os = std::cout ) {
				grb::wait( mat );
				printSparseMatrixIterator( grb::nrows( mat ), grb::ncols( mat ), mat.cbegin(), mat.cend(), name, os );
			}

			void debugPrint( const std::string & msg, std::ostream & os = std::cout ) {
				if( ! DEBUG )
					return;
				os << msg;
			}

			template< typename D >
			bool tryGet( const grb::Matrix< D > & A, size_t i, size_t j, D & val ) {
				auto found = std::find_if( A.cbegin(), A.cend(), [ i, j ]( const std::pair< std::pair< size_t, size_t >, D > & a ) {
					return a.first.first == i && a.first.second == j;
				} );
				if( found == A.cend() )
					return false;
				val = ( *found ).second;
				return true;
			}
		} // namespace utils

		namespace {

			template< typename Iterator >
			class ConditionalIterator : public std::iterator< std::input_iterator_tag, typename std::iterator_traits< Iterator >::value_type > {

			public:
				typedef typename std::iterator_traits< Iterator >::value_type value_type;
				typedef typename std::iterator_traits< Iterator >::pointer pointer;
				typedef typename std::iterator_traits< Iterator >::reference reference;
				typedef typename std::iterator_traits< Iterator >::iterator_category iterator_category;
				typedef typename std::iterator_traits< Iterator >::difference_type difference_type;

				ConditionalIterator( std::function< bool( typename Iterator::value_type ) > func, Iterator it, Iterator endbound ) : _iterator( it ), _endbound( endbound ), _condition( func ) {
					while( _iterator != _endbound && ! _condition( *_iterator ) )
						++( *this );
				}

				ConditionalIterator( const ConditionalIterator & other ) : _iterator( other._iterator ), _endbound( other._endbound ), _condition( other._condition ) {}

				// Overload the dereference operator
				value_type operator*() const {
					return *_iterator;
				}

				// Overload the arrow operator
				value_type operator->() const {
					return *_iterator;
				}

				// Overload the increment operator
				ConditionalIterator & operator++() {
					do
						++_iterator;
					while( _iterator != _endbound && ! _condition( *_iterator ) );
					return *this;
				}

				// Overload the inequality operator
				bool operator!=( const ConditionalIterator & other ) const {
					return _iterator != other._iterator;
				}

				// Overload the equality operator
				bool operator==( const ConditionalIterator & other ) const {
					return _iterator == other._iterator;
				}

			private:
				Iterator _iterator, _endbound;
				std::function< bool( typename Iterator::value_type ) > _condition;
			};

			template< typename D >
			class MatrixConditionalAccessor {
				typedef ConditionalIterator< typename grb::Matrix< D >::const_iterator > iterator_type;

			public:
				MatrixConditionalAccessor( const std::function< bool( std::pair< std::pair< size_t, size_t >, D > ) > & f, const grb::Matrix< D > & A ) :
					_begin( f, A.cbegin(), A.cend() ), _end( f, A.cend(), A.cend() ) {}

				MatrixConditionalAccessor( const MatrixConditionalAccessor & other ) = delete;

				MatrixConditionalAccessor & operator=( const MatrixConditionalAccessor & other ) = delete;

				virtual ~MatrixConditionalAccessor() {}

				iterator_type cbegin() const {
					return _begin;
				}

				iterator_type begin() const {
					return cbegin();
				}

				iterator_type cend() const {
					return _end;
				}

				iterator_type end() const {
					return cend();
				}

			private:
				iterator_type _begin, _end;
			};

			template< typename D >
			class LUMatrixAccessor {
			public:
				LUMatrixAccessor( const grb::Matrix< D > & A ) :
					_lower(
						[]( const std::pair< std::pair< size_t, size_t >, D > & a ) {
							return a.first.first > a.first.second;
						},
						A ),
					_upper(
						[]( const std::pair< std::pair< size_t, size_t >, D > & a ) {
							return a.first.first < a.first.second;
						},
						A ) {}

				MatrixConditionalAccessor< D > & lower() {
					return _lower;
				}

				MatrixConditionalAccessor< D > & upper() {
					return _upper;
				}

			private:
				MatrixConditionalAccessor< D > _lower, _upper;
			};

			template< typename D, typename I, typename J >
			grb::RC trilu( const grb::Matrix< D, grb::config::default_backend, I, J > & A,
				grb::Matrix< D, grb::config::default_backend, I, J > & L,
				grb::Matrix< D, grb::config::default_backend, I, J > & U ) {
				//
				grb::RC rc = grb::RC::SUCCESS;

				// Create the custom accessor
				grb::wait( A );
				LUMatrixAccessor< D > luAccesor( A );

				// Create the lower and upper matrices from the accessor
				const std::vector< std::pair< std::pair< I, J >, D > > nnzs_lower( luAccesor.lower().cbegin(), luAccesor.lower().cend() );
				grb::buildMatrixUnique( L, grb::utils::makeNonzeroIterator< I, J, D >( nnzs_lower.cbegin() ), grb::utils::makeNonzeroIterator< I, J, D >( nnzs_lower.cend() ), IOMode::PARALLEL );
				const std::vector< std::pair< std::pair< I, J >, D > > nnzs_upper( luAccesor.upper().cbegin(), luAccesor.upper().cend() );
				grb::buildMatrixUnique( U, grb::utils::makeNonzeroIterator< I, J, D >( nnzs_upper.cbegin() ), grb::utils::makeNonzeroIterator< I, J, D >( nnzs_upper.cend() ), IOMode::PARALLEL );

				return rc;
			}

			template< typename InputType1, typename InputType2, typename OutputType >
			RC _eWiseMul( Matrix< OutputType > & C,
				const Matrix< InputType1 > & A,
				const Matrix< InputType2 > & B,
				const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value >::type * const = nullptr ) {
				grb::wait( A );
				grb::wait( B );
				grb::wait( C );
				RC rc = grb::eWiseApply( C, A, B, grb::operators::mul< InputType1, InputType2, OutputType >(), RESIZE );
				return rc ? rc : grb::eWiseApply( C, A, B, grb::operators::mul< InputType1, InputType2, OutputType >(), EXECUTE );
			}

			/**
			 * @brief Reduce operation over a matrix.
			 *
			 * @tparam D         The type of the matrix.
			 * @param A          The matrix to reduce.
			 * @param result     The result of the reduction. Initial value taken from here.
			 * @param op         The binary operator to use.
			 * @return grb::RC   Returns #grb::SUCCESS upon succesful completion.
			 */
			template< typename D, typename T, typename Func >
			grb::RC matrixReduce( const grb::Matrix< D > & A, T & result, const Func op ) {
				std::pair< std::pair< size_t, size_t >, T > init = std::make_pair( std::make_pair( 0ul, 0ul ), result );
				std::pair< std::pair< size_t, size_t >, T > accumulator = std::accumulate(
					A.cbegin(), A.cend(), init, [ op ]( const std::pair< std::pair< size_t, size_t >, T > & a, const std::pair< std::pair< size_t, size_t >, D > & b ) {
						return std::make_pair( a.first, op( a.second, b.second ) );
					} );
				grb::wait( A );
				result = accumulator.second;
				return grb::RC::SUCCESS;
			}

			template< typename D, typename T >
			grb::RC matrixSumReduce( const grb::Matrix< D > & A, T & result ) {
				return matrixReduce( A, result, []( T a, D b ) -> T {
					return a + b;
				} );
			}

			template< Descriptor descr = descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2, class Semiring >
			RC _mxm( Matrix< OutputType > & C, const Matrix< InputType1 > & A, const Matrix< InputType2 > & B, const Semiring & ring ) {
				grb::wait( A );
				grb::wait( B );
				grb::wait( C );
				auto rc = mxm< descr >( C, A, B, ring, RESIZE );
				return rc ? rc : mxm< descr >( C, A, B, ring, EXECUTE );
			}

		} // namespace

		enum class TriangleCountAlgorithm { Burkhardt, Cohen, Sandia_LL, Sandia_UU, Sandia_LUT, Sandia_ULT };

		template< typename D >
		RC triangle_count_burkhardt( size_t & u, const Matrix< D > & A ) {
			static_assert( std::is_integral< D >::value, "Type D must be integral" );
			RC rc = RC::SUCCESS;

			utils::printSparseMatrix( A, "A" );
			size_t rows = nrows( A ), cols = ncols( A );

			// Compute B = A^2
			const Semiring< grb::operators::add< D >, grb::operators::mul< D >, grb::identities::zero, grb::identities::one > semiring;
			Matrix< D > B( rows, cols );

			// FIXME: A-squared is not working
			_mxm< descriptors::transpose_right >( B, A, A, semiring );
			utils::printSparseMatrix( B, "A^2" );

			// Compute C = A .* B
			Matrix< D > C( rows, cols );
			_eWiseMul( C, A, B );
			utils::printSparseMatrix( C, "(A^2) .* A" );

			D tmpU = static_cast< D >( 0 );
			matrixSumReduce( C, tmpU );
			utils::debugPrint( "sum (sum ((L * U) .* A)) = " + std::to_string( tmpU ) + "\n" );

			tmpU /= 6;
			utils::debugPrint( "sum (sum ((L * U) .* A)) / 6 = " + std::to_string( tmpU ) + "\n" );

			u = (size_t)tmpU;

			// done
			return rc;
		}

		template< typename D >
		RC triangle_count_cohen( size_t & u, const Matrix< D > & A ) {
			static_assert( std::is_integral< D >::value, "Type D must be integral" );
			RC rc = RC::SUCCESS;

			utils::printSparseMatrix( A, "A" );
			size_t rows = nrows( A ), cols = ncols( A );

			// Split A into L (lower) and U (upper) triangular matrices
			Matrix< D > L( rows, cols ), U( rows, cols );
			rc = rc ? rc : trilu( A, L, U );
			utils::printSparseMatrix( L, "L" );
			utils::printSparseMatrix( U, "U" );

			// Compute B = L * U
			Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one > semiring;
			Matrix< D > B( rows, cols, nnz( A ) );
			_mxm( B, L, U, semiring );
			utils::printSparseMatrix( B, "L * U" );

			// Compute C = B .* A
			Matrix< D > C( rows, cols );
			_eWiseMul( C, B, A );
			utils::printSparseMatrix( C, "(L * U) .* A" );

			D tmpU = static_cast< D >( 0 );
			matrixSumReduce( C, tmpU );
			utils::debugPrint( "sum (sum ((L * U) .* A)) = " + std::to_string( tmpU ) + "\n" );

			tmpU /= 2;
			utils::debugPrint( "sum (sum ((L * U) .* A)) / 2 = " + std::to_string( tmpU ) + "\n" );

			u = (size_t)tmpU;

			// done
			return rc;
		}

		template< typename D >
		RC triangle_count_sandia_ll( size_t & u, const Matrix< D > & A ) {
			static_assert( std::is_integral< D >::value, "Type D must be integral" );
			RC rc = RC::SUCCESS;

			utils::printSparseMatrix( A, "A" );
			size_t rows = nrows( A ), cols = ncols( A );

			// Split A into L (lower) and U (upper) triangular matrices
			Matrix< D > L( rows, cols ), _( rows, cols );
			rc = rc ? rc : trilu( A, L, _ );
			utils::printSparseMatrix( L, "L" );

			// Compute B = L * L
			Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one > semiring;
			Matrix< D > B( rows, cols, nnz( A ) );
			_mxm( B, L, L, semiring );
			utils::printSparseMatrix( B, "L * L" );

			// Compute C = L .* B
			Matrix< D > C( rows, cols );
			_eWiseMul( C, B, L );
			utils::printSparseMatrix( C, "(L * L) .* L" );

			D tmpU = static_cast< D >( 0 );
			matrixSumReduce( C, tmpU );
			utils::debugPrint( "sum (sum ((L * L) .* L)) = " + std::to_string( tmpU ) + "\n" );

			u = (size_t)tmpU;

			// done
			return rc;
		}

		template< typename D >
		RC triangle_count_sandia_uu( size_t & u, const Matrix< D > & A ) {
			static_assert( std::is_integral< D >::value, "Type D must be integral" );
			RC rc = RC::SUCCESS;

			utils::printSparseMatrix( A, "A" );
			size_t rows = nrows( A ), cols = ncols( A );

			// Split A into L (lower) and U (upper) triangular matrices
			Matrix< D > _( rows, cols ), U( rows, cols );
			rc = rc ? rc : trilu( A, _, U );
			utils::printSparseMatrix( U, "U" );

			// Compute B = U * U
			Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one > semiring;
			Matrix< D > B( rows, cols, nnz( A ) );
			_mxm( B, U, U, semiring );
			utils::printSparseMatrix( B, "U * U" );

			// Compute C = U .* B
			Matrix< D > C( rows, cols );
			_eWiseMul( C, B, U );
			utils::printSparseMatrix( C, "(U * U) .* U" );

			D tmpU = static_cast< D >( 0 );
			matrixSumReduce( C, tmpU );
			utils::debugPrint( "sum (sum ((U * U) .* U)) = " + std::to_string( tmpU ) + "\n" );

			u = (size_t)tmpU;

			// done
			return rc;
		}

		template< typename D >
		RC triangle_count_sandia_lut( size_t & u, const Matrix< D > & A ) {
			static_assert( std::is_integral< D >::value, "Type D must be integral" );
			RC rc = RC::SUCCESS;

			utils::printSparseMatrix( A, "A" );
			size_t rows = nrows( A ), cols = ncols( A );

			// Split A into L (lower) and U (upper) triangular matrices
			Matrix< D > L( rows, cols ), U( rows, cols );
			rc = rc ? rc : trilu( A, L, U );
			utils::printSparseMatrix( L, "L" );
			utils::printSparseMatrix( U, "U" );

			// Compute B = L * U
			Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one > semiring;
			Matrix< D > B( rows, cols, nnz( A ) );
			_mxm< descriptors::transpose_right >( B, L, U, semiring );
			utils::printSparseMatrix( B, "L * U" );

			// Compute C = L .* B
			Matrix< D > C( rows, cols );
			_eWiseMul( C, B, L );
			utils::printSparseMatrix( C, "(L * U) .* L" );

			D tmpU = static_cast< D >( 0 );
			matrixSumReduce( C, tmpU );
			utils::debugPrint( "sum (sum ((L * U) .* L)) = " + std::to_string( tmpU ) + "\n" );

			u = (size_t)tmpU;

			// done
			return rc;
		}

		template< typename D >
		RC triangle_count_sandia_ult( size_t & u, const Matrix< D > & A ) {
			static_assert( std::is_integral< D >::value, "Type D must be integral" );
			RC rc = RC::SUCCESS;

			utils::printSparseMatrix( A, "A" );
			size_t rows = nrows( A ), cols = ncols( A );

			// Split A into L (lower) and U (upper) triangular matrices
			Matrix< D > L( rows, cols ), U( rows, cols );
			rc = rc ? rc : trilu( A, L, U );
			utils::printSparseMatrix( L, "L" );
			utils::printSparseMatrix( U, "U" );

			// Compute B = U * L
			Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one > semiring;
			Matrix< D > B( rows, cols, nnz( A ) );
			_mxm( B, U, L, semiring );
			utils::printSparseMatrix( B, "U * L" );

			// Compute C = U .* B
			Matrix< D > C( rows, cols );
			_eWiseMul( C, B, U );
			utils::printSparseMatrix( C, "(L * U) .* U" );

			D tmpU = static_cast< D >( 0 );
			matrixSumReduce( C, tmpU );
			utils::debugPrint( "sum (sum ((L * U) .* U)) = " + std::to_string( tmpU ) + "\n" );

			u = (size_t)tmpU;

			// done
			return rc;
		}

		/**
		 * Given a graph, indicates how many triangles are contained within.
		 *
		 * This implementation is based on the masked matrix multiplication kernel.
		 *
		 * @param[out]    n    The number of triangles. Any prior contents will be ignored.
		 * @param[in]     A    The input graph.
		 *
		 *
		 * @returns #grb::SUCCESS  When the computation completes successfully.
		 * @returns #grb::MISMATCH ?
		 * @returns #grb::ILLEGAL  ?
		 * @returns #grb::PANIC    If an unrecoverable error has been encountered. The
		 *                         output as well as the state of ALP/GraphBLAS is
		 *                         undefined.
		 *
		 * \par Performance semantics
		 *
		 *   -# This function does not allocate nor free dynamic memory, nor shall it
		 *      make any system calls.
		 *
		 * For performance semantics regarding work, inter-process data movement,
		 * intra-process data movement, synchronisations, and memory use, please see
		 * the specification of the ALP primitives this function relies on. These
		 * performance semantics, with the exception of getters such as #grb::nnz, are
		 * specific to the backend selected during compilation.
		 */
		template< typename D >
		RC triangle_count( TriangleCountAlgorithm Algo, size_t & u, const Matrix< D > & A_constant ) {
			auto A = A_constant;
			// Static assertions
			static_assert( std::is_integral< D >::value, "Type D must be integral" );
			// Dynamic assertions
			if( grb::nrows( A ) != grb::ncols( A ) ) {
				std::cerr << "A must be square" << std::endl;
				return RC::ILLEGAL;
			}
			if( ! utils::is_diagonal_null( A ) ) {
				// Create a mask with null values on the diagonal, and ones everywhere else
				grb::Matrix< D > M( grb::nrows( A ), grb::ncols( A ) );
				size_t nnz_mask = grb::nrows( A ) * grb::ncols( A ) - grb::nrows( A );
				std::vector< size_t > I(nnz_mask), J( nnz_mask );
				std::vector< D > V( nnz_mask, static_cast< D >(1) );
				for( size_t i = 0, k = 0; i < grb::nrows( A ); ++i ) {
					for( size_t j = 0; j < grb::ncols( A ); ++j ) {
						if( i == j ) continue;
						I[k] = i;
						J[k] = j;
						++k;
					}
				}

				buildMatrixUnique( M, I.data(), J.data(), V.data(), V.size(), grb::IOMode::PARALLEL );
				// Multiply A with the mask
				Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one > semiring;
				utils::printSparseMatrix( A, "A before diagonal annihilation" );
				utils::printSparseMatrix( M, "Mask" );
				_eWiseMul( A, A_constant, M );
				utils::printSparseMatrix( A, "A after diagonal annihilation" );
				assert( utils::is_diagonal_null( A ) );
			}

			switch( Algo ) {
				case TriangleCountAlgorithm::Burkhardt:
					utils::debugPrint( "-- Burkhardt\n" );
					return triangle_count_burkhardt( u, A );
				case TriangleCountAlgorithm::Cohen:
					utils::debugPrint( "-- Cohen\n" );
					return triangle_count_cohen( u, A );
				case TriangleCountAlgorithm::Sandia_LL:
					utils::debugPrint( "-- Sandia LL\n" );
					return triangle_count_sandia_ll( u, A );
				case TriangleCountAlgorithm::Sandia_UU:
					utils::debugPrint( "-- Sandia UU\n" );
					return triangle_count_sandia_uu( u, A );
				case TriangleCountAlgorithm::Sandia_LUT:
					utils::debugPrint( "-- Sandia LUT\n" );
					return triangle_count_sandia_lut( u, A );
				case TriangleCountAlgorithm::Sandia_ULT:
					utils::debugPrint( "-- Sandia ULT\n" );
					return triangle_count_sandia_ult( u, A );
				default:
					utils::debugPrint( "-- Unknown\n", std::cerr );
					return RC::FAILED;
			}
		}

	} // namespace algorithms

} // namespace grb

#endif // _H_GRB_TRIANGLE_ENUMERATION
