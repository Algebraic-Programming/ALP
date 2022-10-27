
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

/*
 * @author A. N. Yzelman
 * @date 14th of January 2022
 */

#ifndef _H_ALP_REFERENCE_MATRIX
#define _H_ALP_REFERENCE_MATRIX

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include <alp/backends.hpp>
#include <alp/base/matrix.hpp>
#include <alp/amf-based/matrix.hpp>
#include <alp/config.hpp>
#include <alp/ops.hpp>
#include <alp/rc.hpp>
#include <alp/type_traits.hpp>
#include <alp/utils.hpp>
#include <alp/utils/autodeleter.hpp>
//#include <alp/utils/pattern.hpp> //for help with dealing with pattern matrix input
#include <alp/vector.hpp>
#include <alp/storage.hpp>
#include <alp/structures.hpp>
#include <alp/density.hpp>
#include <alp/views.hpp>
#include <alp/imf.hpp>

namespace alp {
	namespace internal {

		/**
		 * Retrieve the row dimension size of this matrix.
		 *
		 * @returns The number of rows the current matrix contains.
		 *
		 * \parblock
		 * \par Performance semantics.
		 *        -# This function consitutes \f$ \Theta(1) \f$ work.
		 *        -# This function allocates no additional dynamic memory.
		 *        -# This function uses \f$ \mathcal{O}(1) \f$ memory
		 *           beyond that which was already used at function entry.
		 *        -# This function will move
		 *             \f$ \mathit{sizeof}( size\_t ) \f$
		 *           bytes of memory.
		 * \endparblock
		 */
		template< typename D >
		size_t nrows( const Matrix< D, reference > & ) noexcept;

		/**
		 * Retrieve the column dimension size of this matrix.
		 *
		 * @returns The number of columns the current matrix contains.
		 *
		 * \parblock
		 * \par Performance semantics.
		 *        -# This function consitutes \f$ \Theta(1) \f$ work.
		 *        -# This function allocates no additional dynamic memory.
		 *        -# This function uses \f$ \mathcal{O}(1) \f$ memory
		 *           beyond that which was already used at function entry.
		 *        -# This function will move
		 *             \f$ \mathit{sizeof}( size\_t ) \f$
		 *           bytes of memory.
		 * \endparblock
		 */
		template< typename D >
		size_t ncols( const Matrix< D, reference > & ) noexcept;


		template< typename D >
		D * getRaw( Matrix< D, reference > & ) noexcept;

		template< typename D >
		const D * getRaw( const Matrix< D, reference > & ) noexcept;

		/** Forward declaration */
		template< typename T >
		const bool & getInitialized( const Vector< T, reference > & v ) noexcept;

		/** Forward declaration */
		template< typename T >
		void setInitialized( Vector< T, reference > & v, const bool initialized ) noexcept;

		template< typename D >
		const bool & getInitialized( const alp::internal::Matrix< D, reference > & A ) noexcept {
			return A.initialized;
		}

		template< typename D >
		void setInitialized( alp::internal::Matrix< D, reference > & A, const bool initialized ) noexcept {
			A.initialized = initialized;
		}

		/**
		 * ALP/Dense matrix container.
		 *
		 * A matrix is stored in full format.
		 * \a Matrix may be used by \a Matrix as a raw container.
		 *
		 * @tparam D  The element type.
		 */
		template< typename D >
		class Matrix< D, reference > {

		private:
			/* *********************
				BLAS2 friends
			   ********************* */

			template< typename DataType >
			friend size_t nrows( const Matrix< DataType, reference > & m ) noexcept;

			template< typename DataType >
			friend size_t ncols( const Matrix< DataType, reference > & m ) noexcept;

			/* *********************
			     `Getter' friends
			   ********************* */

			friend D * internal::getRaw< D >( Matrix< D, reference > & ) noexcept;

			friend const D * internal::getRaw< D >( const Matrix< D, reference > & ) noexcept;

			/* ********************
				IO friends
			   ******************** */

			template< typename InputType, typename fwd_iterator >
			friend RC buildMatrix( Matrix< InputType, reference > &, fwd_iterator, const fwd_iterator );

			template< typename DataType >
			friend const bool & internal::getInitialized( const alp::internal::Matrix< DataType, reference > & ) noexcept;

			template< typename DataType >
			friend void internal::setInitialized( alp::internal::Matrix< DataType, reference > & , const bool ) noexcept;

			typedef Matrix< D, reference > self_type;

			/**
			 * The number of rows.
			 */
			size_t m;

			/**
			 * The number of columns.
			 */
			size_t n;

			/** The container capacity (in elements). */
			size_t cap;

			/** The matrix data. */
			D * __restrict__ data;

			/**
			 * Whether the container presently is initialized or not.
			 * We differentiate the concept of empty matrix (matrix of size \f$0\times 0\f$)
			 * from the one of uninitialized (matrix of size \f$m\times n\f$ which was never set)
			 * and that of zero matrix (matrix with all zero elements).
			 * \note in sparse format a zero matrix result in an ampty data structure. Is this
			 * used to refer to uninitialized matrix in ALP/GraphBLAS?
			 **/
			bool initialized;

			/**
			 * @see alp::buildMatrixUnique
			 */
			template< typename fwd_iterator >
			RC buildMatrixUnique( const fwd_iterator & _start, const fwd_iterator & _end ) {
				// detect trivial case
				if ( _start == _end || m == 0 || n == 0) {
					return SUCCESS;
				}

				if ( (size_t)( _end - _start ) != ( m * n ) ) {
					return MISMATCH;
				}

				// TODO: Add more sanity checks (e.g. overflow)

				for( auto it = _start; it != _end; ++it ) {
					data[ it - _start ] = *it;
				}

				initialized = true;

				// done
				return RC::SUCCESS;
			}

		public:
			/** @see Matrix::value_type */
			typedef D value_type;

			/**
			 * The main ALP/Dense matrix constructor.
			 *
			 * The constructed object will be uninitalised after successful construction.
			 *
			 * Requesting a matrix with zero \a rows or \a columns will yield an empty
			 * matrix.
			 *
			 * @param rows        The number of rows in the new matrix.
			 * @param columns     The number of columns in the new matrix.
			 * @param cap         The capacity in terms of elements of the new matrix. Optional.
			 *
			 * @return SUCCESS This function never fails.
			 *
			 * \parblock
			 * \par Performance semantics.
			 *        -# This constructor entails \f$ \Theta(1) \f$ amount of work.
			 *        -# This constructor may allocate \f$ \Theta( \max{mn, cap} ) \f$ bytes
			 *           of dynamic memory.
			 *        -# This constructor will use \f$ \Theta(1) \f$ extra bytes of
			 *           memory beyond that at constructor entry.
			 *        -# This constructor incurs \f$ \Theta(1) \f$ data movement.
			 *        -# This constructor \em may make system calls.
			 * \endparblock
			 *
			 * \warning Avoid the use of this constructor within performance critical
			 *          code sections.
			 * \warning \a cap is present for compatibility with other matrix specializations.
			 *          In reference backend, the number of non-zeros (i.e. capacity)
			 *          depends on the used storage scheme. Therefore, this parameter is
			 *          ignored.
			 */
			Matrix( const size_t rows, const size_t columns, const size_t cap = 0 ): m( rows ), n( columns ), cap( std::max( m*n, cap ) ), initialized( false ) {
				// TODO Implement allocation properly
				if( m > 0 && n > 0) {
					data = new (std::nothrow) D[ m * n ];
				} else {
					data = nullptr;
				}

				if ( m > 0 && n > 0 && data == nullptr ) {
					throw std::runtime_error( "Could not allocate memory during alp::Matrix<reference> construction." );
				}

			}

			/**
			 * Copy constructor.
			 *
			 * @param other The matrix to copy. The initialization state of the copy
			 *              reflects the state of \a other.
			 *
			 * \parblock
			 * \par Performance semantics.
			 *      Allocates the same capacity as the \a other matrix, even if the
			 *      actual number of elements contained in \a other is less.
			 *        -# This constructor entails \f$ \Theta(1) \f$ amount of work.
			 *        -# This constructor allocates \f$ \Theta(\max{mn, cap} ) \f$ bytes
			 *           of dynamic memory.
			 *        -# This constructor incurs \f$ \Theta(mn) \f$ of data
			 *           movement.
			 *        -# This constructor \em may make system calls.
			 * \endparblock
			 *
			 * \warning Avoid the use of this constructor within performance critical
			 *          code sections.
			 */
			Matrix( const Matrix< D, reference > & other ) : Matrix( other.m, other.n ) {
				initialized = other.initialized;
			}

			/**
			 * Move constructor. The new matrix equal the given
			 * matrix. Invalidates the use of the input matrix.
			 *
			 * @param[in] other The GraphBLAS matrix to move to this new instance.
			 *
			 * \parblock
			 * \par Performance semantics.
			 *        -# This constructor entails \f$ \Theta(1) \f$ amount of work.
			 *        -# This constructor will not allocate any new dynamic memory.
			 *        -# This constructor will use \f$ \Theta(1) \f$ extra bytes of
			 *           memory beyond that at constructor entry.
			 *        -# This constructor will move \f$ \Theta(1) \f$ bytes of data.
			 * \endparblock
			 */
			// Matrix( self_type && other ) noexcept {
			// 	moveFromOther( std::forward< self_type >( other ) );
			// }

			/**
			 * Move assignment operator.
			 * @see Matrix::Matrix( Matrix && )
			 */
			// self_type & operator=( self_type && other ) noexcept {
			// 	moveFromOther( std::forward< self_type >( other ) );
			// 	return *this;
			// }

			/**
			 * Matrix destructor.
			 *
			 * \parblock
			 * \par Performance semantics.
			 *        -# This destructor entails \f$ \Theta(1) \f$ amount of work.
			 *        -# This destructor will not perform any memory allocations.
			 *        -# This destructor will use \f$ \mathcal{O}(1) \f$ extra bytes of
			 *           memory beyond that at constructor entry.
			 *        -# This destructor will move \f$ \Theta(1) \f$ bytes of data.
			 *        -# This destructor makes system calls.
			 * \endparblock
			 *
			 * \warning Avoid calling destructors from within performance critical
			 *          code sections.
			 */
			~Matrix() {
				if( data != nullptr ) {
					delete [] data;
				}
			}
		};

		template< typename T >
		T * getRaw( Matrix< T, reference > & m ) noexcept {
			return m.data;
		}

		template< typename T >
		const T * getRaw( const Matrix< T, reference > & m ) noexcept {
			return m.data;
		}

		template< typename T >
		size_t nrows( const Matrix< T, reference > & m ) noexcept {
			return m.m;
		}

		template< typename T >
		size_t ncols( const Matrix< T, reference > & m ) noexcept {
			return m.n;
		}

		/**
		 * Identifies any reference internal matrix is an internal container.
		 */
		template< typename T >
		struct is_container< internal::Matrix< T, reference > > : std::true_type {};

	} // namespace internal

	// Matrix-related implementation

	namespace internal {


		/** Specialization for general matrix */
		template<>
		struct determine_poly_factory< structures::General, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for square matrix */
		template<>
		struct determine_poly_factory< structures::Square, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for orthogonal matrix */
		template<>
		struct determine_poly_factory< structures::Orthogonal, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for upper-triangular matrix */
		template<>
		struct determine_poly_factory< structures::UpperTriangular, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::PackedFactory< storage::UPPER, storage::ROW_WISE > factory_type;
		};

		/** Specialization for lower-triangular matrix */
		template<>
		struct determine_poly_factory< structures::LowerTriangular, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::PackedFactory< storage::LOWER, storage::ROW_WISE > factory_type;
		};

		/** Specialization for symmetric matrix */
		template<>
		struct determine_poly_factory< structures::Symmetric, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::PackedFactory< storage::UPPER, storage::ROW_WISE > factory_type;
		};

		/** Specialization for hermitian matrix */
		template<>
		struct determine_poly_factory< structures::Hermitian, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for symmetric tridiagonal matrix */
		template<>
		struct determine_poly_factory< structures::SymmetricTridiagonal, imf::Id, imf::Id, reference > {

			private:
				using interval = std::tuple_element< 0, structures::SymmetricTridiagonal::band_intervals >::type;

			public:
				//typedef storage::polynomials::BandFactory< interval, storage::ROW_WISE > factory_type;
				typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for hermitian tridiagonal matrix */
		template<>
		struct determine_poly_factory< structures::HermitianTridiagonal, imf::Id, imf::Id, reference > {

			private:
				// This will be used in the commented line below once band storage is added.
				// Added for readability.
				using interval = std::tuple_element< 0, structures::SymmetricTridiagonal::band_intervals >::type;

			public:
				//typedef storage::polynomials::BandFactory< interval, storage::ROW_WISE > factory_type;
				typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for vectors */
		template< typename Structure >
		struct determine_poly_factory< Structure, imf::Id, imf::Zero, reference > {

			typedef storage::polynomials::ArrayFactory factory_type;
		};

	} // namespace internal


} // namespace alp

#endif // end ``_H_ALP_REFERENCE_MATRIX''
