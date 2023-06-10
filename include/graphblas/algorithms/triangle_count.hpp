
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

#include <map>
#include <numeric>
#include <vector>

#include <graphblas/utils/iterators/NonzeroIterator.hpp>

#include <graphblas.hpp>

constexpr bool Debug = false;

namespace grb {

	namespace algorithms {

		namespace utils {

			template< class Iterator >
			void printSparseMatrixIterator( size_t rows, size_t cols, Iterator begin, Iterator end, const std::string & name = "", std::ostream & os = std::cout ) {
				std::cout << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
				if( rows > 100 || cols > 100 ) {
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

			template< bool debug, typename D >
			void printSparseMatrix( const grb::Matrix< D > & mat, const std::string & name = "", std::ostream & os = std::cout ) {
				if( ! debug )
					return;
				grb::wait( mat );
				printSparseMatrixIterator( grb::nrows( mat ), grb::ncols( mat ), mat.cbegin(), mat.cend(), name, os );
			}

			template< bool debug >
			void printf( const std::string & msg, std::ostream & os = std::cout ) {
				if( ! debug )
					return;
				os << msg;
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

		} // namespace

		enum class TriangleCountAlgorithm { Burkhardt, Cohen, Sandia_TT };

		std::map< TriangleCountAlgorithm, std::string > TriangleCountAlgorithmNames = { { TriangleCountAlgorithm::Burkhardt, "Burkhardt" }, { TriangleCountAlgorithm::Cohen, "Cohen" },
			{ TriangleCountAlgorithm::Sandia_TT, "Sandia_TT" } };

		template< Descriptor descr = descriptors::no_operation, typename D, typename I, typename J, class Semiring, class MulMonoid, class SumMonoid >
		RC triangle_count_generic( size_t & count,
			Matrix< D, grb::config::default_backend, I, J > & MXM_out,
			const Matrix< D, grb::config::default_backend, I, J > & MXM_lhs,
			const Matrix< D, grb::config::default_backend, I, J > & MXM_rhs,
			Matrix< D, grb::config::default_backend, I, J > & EWA_out,
			const Matrix< D, grb::config::default_backend, I, J > & EWA_rhs,
			const D div_factor,
			const Semiring mxm_semiring = Semiring(),
			const MulMonoid ewiseapply_monoid = MulMonoid(),
			const SumMonoid sumreduce_monoid = SumMonoid() ) {
			RC rc = RC::SUCCESS;

			rc = ( &MXM_out == &MXM_lhs ) ? RC::ILLEGAL : rc;
			rc = ( &MXM_out == &MXM_rhs ) ? RC::ILLEGAL : rc;

			// Compute MXM_out = Mlhs * Mrhs
			utils::printSparseMatrix< Debug >( MXM_lhs, "MXM_lhs" );
			utils::printSparseMatrix< Debug >( MXM_rhs, "MXM_rhs" );
			rc = rc ? rc : mxm< descr >( MXM_out, MXM_lhs, MXM_rhs, mxm_semiring, Phase::RESIZE );
			rc = rc ? rc : mxm< descr >( MXM_out, MXM_lhs, MXM_rhs, mxm_semiring, Phase::EXECUTE );
			utils::printSparseMatrix< Debug >( MXM_out, "MXM_out = mxm( MXM_lhs, MXM_rhs )" );

			// Compute MXM_out .*= EWA_rhs
			utils::printSparseMatrix< Debug >( EWA_rhs, "EWA_rhs" );

			// FIXME: Replace by a foldl( Matrix[in,out], Matrix[in], Monoid ) - not implemented yet
			// Will then become:
			// rc = rc ? rc : eWiseApply< descr >( MXM_out, MXM_out, EWA_rhs, ewiseapply_monoid, Phase::RESIZE );
			// rc = rc ? rc : eWiseApply< descr >( MXM_out, MXM_out, EWA_rhs, ewiseapply_monoid, Phase::EXECUTE );
			// Instead of:
			rc = rc ? rc : eWiseApply< descr >( EWA_out, MXM_out, EWA_rhs, ewiseapply_monoid, Phase::RESIZE );
			rc = rc ? rc : eWiseApply< descr >( EWA_out, MXM_out, EWA_rhs, ewiseapply_monoid, Phase::EXECUTE );
			utils::printSparseMatrix< Debug >( EWA_out, "EWA_out = ewiseapply( MXM_out, EWA_rhs )" );

			// Compute a sum reduction over <EWA_out> in <count>
			count = 0;
			rc = rc ? rc : foldl< descr >( count, EWA_out, sumreduce_monoid );
			utils::printf< Debug >( "count = foldl(EWA_out) = " + std::to_string( count ) + "\n" );

			// Apply the div_factor to the reduction result
			count /= div_factor;
			utils::printf< Debug >( "count = count / div_factor = " + std::to_string( count ) + "\n" );

			return rc;
		}

		/**
		 * Given a graph, indicates how many triangles are contained within.
		 *
		 * @tparam D 				The type of the matrix non-zero values.
		 *
		 * @param[out]    count     The number of triangles.
		 * 						    Any prior contents will be ignored.
		 * @param[in]     A         The input graph.
		 * @param[in,out] MXM_out    Buffer matrix with the same dimensions as the input
		 * 							graph. Any prior contents will be ignored.
		 * @param[in] L 		Lower triangular matrix of the input graph (optional)
		 * @param[in] U 		Lower triangular matrix of the input graph (optional)
		 *
		 *
		 * @returns #grb::SUCCESS  When the computation completes successfully.
		 * @returns #grb::MISMATCH If the dimensions of the input matrices/buffers
		 * 						   are incompatible.
		 * @returns #grb::ILLEGAL  If the given algorithm does not exist.
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
		template< Descriptor descr = descriptors::no_operation,
			typename D,
			typename I,
			typename J,
			class Semiring = grb::Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class MulMonoid = grb::Monoid< grb::operators::mul< D >, identities::one >,
			class SumMonoid = grb::Monoid< operators::add< size_t, D, size_t >, identities::zero > >
		RC triangle_count( const TriangleCountAlgorithm algo,
			size_t & count,
			const Matrix< D, grb::config::default_backend, I, J > & A,
			Matrix< D, grb::config::default_backend, I, J > & MXM_out,
			Matrix< D, grb::config::default_backend, I, J > & EWA_out,
			Matrix< D, grb::config::default_backend, I, J > & L = { 0, 0 },
			Matrix< D, grb::config::default_backend, I, J > & U = { 0, 0 } ) {
			// Static assertions
			static_assert( std::is_integral< D >::value, "Type D must be integral" );

			// Sanity checks
			if( nrows( A ) != ncols( A ) ) {
				std::cerr << "Matrix A must be square" << std::endl;
				return RC::MISMATCH;
			}
			if( ncols( L ) != nrows( L ) ) {
				std::cerr << "Matrix L must be square" << std::endl;
				return RC::MISMATCH;
			}
			if( nrows( A ) != ncols( L ) ) {
				std::cerr << "Matrices A and L must have the same dimensions" << std::endl;
				return RC::MISMATCH;
			}
			if( ncols( U ) != nrows( U ) ) {
				std::cerr << "Matrix U must be square" << std::endl;
				return RC::MISMATCH;
			}
			if( nrows( A ) != ncols( U ) ) {
				std::cerr << "Matrices A and U must have the same dimensions" << std::endl;
				return RC::MISMATCH;
			}
			if( ncols( MXM_out ) != nrows( MXM_out ) ) {
				std::cerr << "Matrix MXM_out must be square" << std::endl;
				return RC::MISMATCH;
			}
			if( nrows( A ) != ncols( MXM_out ) ) {
				std::cerr << "Matrices A and MXM_out must have the same dimensions" << std::endl;
				return RC::MISMATCH;
			}
			if( ncols( EWA_out ) != nrows( EWA_out ) ) {
				std::cerr << "Matrix EWA_out must be square" << std::endl;
				return RC::MISMATCH;
			}
			if( nrows( A ) != ncols( EWA_out ) ) {
				std::cerr << "Matrices A and EWA_out must have the same dimensions" << std::endl;
				return RC::MISMATCH;
			}

			// Dispatch to the appropriate algorithm
			switch( algo ) {
				case TriangleCountAlgorithm::Burkhardt: {
					return triangle_count_generic< descr | descriptors::transpose_right, D, I, J, Semiring, MulMonoid, SumMonoid >( count, MXM_out, A, A, EWA_out, A, 6 );
				}

				case TriangleCountAlgorithm::Cohen: {
					trilu( A, L, U );
					if( nrows( L ) + ncols( L ) == 0 ) {
						std::cerr << "Matrix L must be provided for the Cohen algorithm" << std::endl;
						return RC::MISMATCH;
					} else if( nrows( U ) + ncols( U ) == 0 ) {
						std::cerr << "Matrix U must be provided for the Cohen algorithm" << std::endl;
						return RC::MISMATCH;
					}
					return triangle_count_generic< descr, D, I, J, Semiring, MulMonoid, SumMonoid >( count, MXM_out, L, U, EWA_out,  A, 2 );
				}

				case TriangleCountAlgorithm::Sandia_TT: {
					trilu( A, L, U );
					if( ( nrows( U ) == 0 || ncols( U ) == 0 ) && ( nrows( L ) == 0 || ncols( L ) == 0 ) ) {
						std::cerr << "Matrix L or U must be provided for the Sandia_TT algorithm" << std::endl;
						return RC::MISMATCH;
					}
					const Matrix< D, grb::config::default_backend, I, J > & T = ( nrows( U ) == 0 || ncols( U ) == 0 ) ? L : U;
					return triangle_count_generic< descr, D, I, J, Semiring, MulMonoid, SumMonoid >( count, MXM_out, T, T, EWA_out, T, 1 );
				}

				default:
					std::cerr << "Unknown TriangleCountAlgorithm enum value" << std::endl;
					return RC::ILLEGAL;
			}

			return RC::SUCCESS;
		}

	} // namespace algorithms

} // namespace grb

#endif // _H_GRB_TRIANGLE_ENUMERATION
