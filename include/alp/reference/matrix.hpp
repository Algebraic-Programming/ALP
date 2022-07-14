
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

	/** Identifies any reference implementation of ALP matrix as an ALP matrix. */
	template< typename T, typename Structure, enum Density density, typename View, typename ImfR, typename ImfC >
	struct is_matrix< Matrix< T, Structure, density, View, ImfR, ImfC, reference > > : std::true_type {};

	// Matrix-related implementation

	namespace internal {
		/** Forward declaration */
		template< typename T, typename ImfR, typename ImfC, typename MappingPolynomial, bool requires_allocation >
		class StorageBasedMatrix;

		/** Forward declaration */
		template< typename T, typename ImfR, typename ImfC, typename DataLambdaType >
		class FunctorBasedMatrix;

		/** Container reference getters used by friend functions of specialized Matrix */
		template< typename T, typename ImfR, typename ImfC, typename MappingPolynomial, bool requires_allocation >
		const Vector< T, reference > & getContainer( const StorageBasedMatrix< T, ImfR, ImfC, MappingPolynomial, requires_allocation > & A );

		template< typename T, typename ImfR, typename ImfC, typename MappingPolynomial, bool requires_allocation >
		Vector< T, reference > & getContainer( StorageBasedMatrix< T, ImfR, ImfC, MappingPolynomial, requires_allocation > & A );

		/** Container reference getters. Defer the call to base class friend function */
		template< typename T, typename Structure, enum Density density, typename View, typename ImfR, typename ImfC >
		const Vector< T, reference > & getContainer( const alp::Matrix< T, Structure, density, View, ImfR, ImfC, reference > & A ) {
			return getContainer( static_cast< const StorageBasedMatrix< T, ImfR, ImfC,
				typename alp::Matrix< T, Structure, density, View, ImfR, ImfC, reference >::mapping_polynomial_type,
				alp::Matrix< T, Structure, density, View, ImfR, ImfC, reference >::requires_allocation > & >( A ) );
		}

		template< typename T, typename Structure, enum Density density, typename View, typename ImfR, typename ImfC >
		Vector< T, reference > & getContainer( alp::Matrix< T, Structure, density, View, ImfR, ImfC, reference > & A ) {
			return getContainer( static_cast< StorageBasedMatrix< T, ImfR, ImfC,
				typename alp::Matrix< T, Structure, density, View, ImfR, ImfC, reference >::mapping_polynomial_type,
				alp::Matrix< T, Structure, density, View, ImfR, ImfC, reference >::requires_allocation > & >( A ) );
		}

		/** Functor reference getter used by friend functions of specialized Matrix */
		template< typename T, typename ImfR, typename ImfC, typename DataLambdaType >
		const typename FunctorBasedMatrix< T, ImfR, ImfC, DataLambdaType >::functor_type &getFunctor( const FunctorBasedMatrix< T, ImfR, ImfC, DataLambdaType > &A );

		/**
		 * Getter for the functor of a functor-based matrix.
		 *
		 * @tparam MatrixType  The type of input matrix.
		 *
		 * @param[in] A        Input matrix.
		 *
		 * @returns A constant reference to a functor object within the
		 *          provided functor-based matrix.
		 */
		template<
			typename MatrixType,
			std::enable_if_t<
				internal::is_functor_based< MatrixType >::value
			> * = nullptr
		>
		const typename MatrixType::functor_type &getFunctor( const MatrixType &A ) {
			return static_cast< const typename MatrixType::base_type & >( A ).getFunctor();
		}

		/** Forward declaration */
		template< typename DerivedMatrix >
		class MatrixBase;

		template< typename DerivedMatrix >
		std::pair< size_t, size_t > dims( const MatrixBase< DerivedMatrix > & A ) noexcept;

		template< typename MatrixType,
			std::enable_if_t< is_matrix< MatrixType>::value > * = nullptr
		>
		bool getInitialized( const MatrixType &A ) noexcept;

		template< typename MatrixType,
			std::enable_if_t< is_matrix< MatrixType>::value > * = nullptr
		>
		void setInitialized( MatrixType &, const bool ) noexcept;

	} // namespace internal

	template< typename D, typename Structure, typename View, typename ImfR, typename ImfC >
	size_t nrows( const Matrix< D, Structure, Density::Dense, View, ImfR, ImfC, reference > & A ) noexcept;

	template< typename D, typename Structure, typename View, typename ImfR, typename ImfC >
	size_t ncols( const Matrix< D, Structure, Density::Dense, View, ImfR, ImfC, reference > & A ) noexcept;

	template< typename D, typename Structure, typename View, typename ImfR, typename ImfC >
	std::pair< size_t, size_t > dims( const Matrix< D, Structure, Density::Dense, View, ImfR, ImfC, reference > & A ) noexcept;

	namespace internal {

		// Forward declaration
		template< typename MatrixType >
		const typename MatrixType::access_type access( const MatrixType &, const typename MatrixType::storage_index_type & );

		template< typename MatrixType >
		typename MatrixType::access_type access( MatrixType &, const typename MatrixType::storage_index_type & );

		template< typename MatrixType >
		typename MatrixType::storage_index_type getStorageIndex( const MatrixType &A, const size_t i, const size_t j, const size_t s = 0, const size_t P = 1 );

		/**
		 * Base Matrix class containing attributes common to all Matrix specialization
		 */
		template< typename DerivedMatrix >
		class MatrixBase {

			friend std::pair< size_t, size_t > dims<>( const MatrixBase< DerivedMatrix > &A ) noexcept;

			template< typename MatrixType, std::enable_if_t< is_matrix< MatrixType>::value > * >
			friend bool getInitialized( const MatrixType &A ) noexcept;

			template< typename MatrixType, std::enable_if_t< is_matrix< MatrixType>::value > * >
			friend void setInitialized( MatrixType &A, const bool initialized ) noexcept;

			protected:

				std::pair< size_t, size_t > dims() const noexcept {
					return static_cast< const DerivedMatrix & >( *this ).dims();
				}

				template< typename MatrixType >
				friend const typename MatrixType::access_type access( const MatrixType &A, const typename MatrixType::storage_index_type &storageIndex );

				template< typename MatrixType >
				friend typename MatrixType::access_type access( MatrixType &A, const typename MatrixType::storage_index_type &storageIndex );

				template< typename MatrixType >
				friend typename MatrixType::storage_index_type getStorageIndex( const MatrixType &A, const size_t i, const size_t j, const size_t s, const size_t P );

				bool getInitialized() const {
					return static_cast< const DerivedMatrix & >( *this ).getInitialized();
				}

				void setInitialized( const bool initialized ) {
					static_cast< DerivedMatrix & >( *this ).setInitialized( initialized );
				}

				template< typename AccessType, typename StorageIndexType >
				const AccessType access( const StorageIndexType storageIndex ) const {
					static_assert( std::is_same< AccessType, typename DerivedMatrix::access_type >::value );
					static_assert( std::is_same< StorageIndexType, typename DerivedMatrix::storage_index_type >::value );
					return static_cast< const DerivedMatrix & >( *this ).access( storageIndex );
				}

				template< typename AccessType, typename StorageIndexType >
				AccessType access( const StorageIndexType &storageIndex ) {
					static_assert( std::is_same< AccessType, typename DerivedMatrix::access_type >::value );
					static_assert( std::is_same< StorageIndexType, typename DerivedMatrix::storage_index_type >::value );
					return static_cast< DerivedMatrix & >( *this ).access( storageIndex );
				}

				template< typename StorageIndexType >
				StorageIndexType getStorageIndex( const size_t i, const size_t j, const size_t s, const size_t P ) const {
					static_assert( std::is_same< StorageIndexType, typename DerivedMatrix::storage_index_type >::value );
					return static_cast< const DerivedMatrix & >( *this ).getStorageIndex( i, j, s, P );
				}

		};

		/**
		 * Matrix container specialization
		 * Implements both original containers and views on containers.
		 * @tparam requires_allocation True if the class is an original container
		 *                             False if the class is a view of another matrix
		 */
		template< typename T, typename ImfR, typename ImfC, typename MappingPolynomial, bool requires_allocation >
		class StorageBasedMatrix : public MatrixBase< StorageBasedMatrix< T, ImfR, ImfC, MappingPolynomial, requires_allocation > > {
			public:

				/** Expose static properties */

				typedef T value_type;
				typedef ImfR imf_r_type;
				typedef ImfC imf_c_type;
				/** Type returned by access function */
				typedef T &access_type;
				/** Type of the index used to access the physical storage */
				typedef size_t storage_index_type;

			protected:
				typedef StorageBasedMatrix< T, ImfR, ImfC, MappingPolynomial, requires_allocation > self_type;
				friend MatrixBase< self_type >;

				typedef typename std::conditional<
					requires_allocation,
					Vector< T, reference >,
					Vector< T, reference > &
				>::type container_type;

				/** A container-type view is characterized by its association with a physical container */
				container_type container;

				/**
				 * All matrix views use a pair of index mapping functions to
				 * capture the correspondence between their logical layout and the one
				 * of their underlying container. This may be another view leading to a composition
				 * of IMFs between the top matrix view and the physical container.
				 * Original matrix containers's index mapping functions are an identity mapping.
				 */
				//ImfR imf_r;
				//ImfC imf_c;

				/**
				 * The container's storage scheme. \a storage_scheme is not exposed to the user as an option
				 * but can defined by ALP at different points in the execution depending on the \a backend choice.
				 * In particular, if the structured matrix is not a temporary matrix than it is fixed at construction
				 * time when the allocation takes place.
				 * If the structured matrix is a temporary one than a storage storage scheme choice may or may not be
				 * made depending on whether a decision about instantiating the matrix is made by the framework.
				 *
				 * The specific storage scheme choice depends on the chosen backend and the structure of the matrix.
				 * \internal \todo Revisit this when introducing storage mapping functions.
				 */
				//Smf smf;

				/**
				 * Access mapping function maps a pair of logical coordinates
				 * into the concrete coordinate inside the actual container.
				 * \see AMF
				 */
			public: // TODO: Temporarily expose AMF publicly until proper getters are implemented
				storage::AMF< ImfR, ImfC, MappingPolynomial > amf;
			protected:

				/**
				 * @brief determines the size of the matrix via the domain of
				 * the index mapping functions.
				 *
				 * @return A pair of dimensions.
				 */
				std::pair< size_t, size_t > dims() const noexcept {
					return amf.getLogicalDimensions();
				}

				friend const Vector< T, reference > & getContainer( const self_type &A ) {
					return A.container;
				}

				friend Vector< T, reference > & getContainer( self_type &A ) {
					return A.container;
				}

				bool getInitialized() const noexcept {
					return internal::getInitialized( container );
				}

				void setInitialized( const bool initialized ) noexcept {
					internal::setInitialized( container , initialized );
				}

				/**
				 * Returns a constant reference to the element corresponding to
				 * the provided storage index.
				 *
				 * @param storageIndex  storage index in the physical iteration
				 *                      space.
				 *
				 * @return const reference or value of the element at given position.
				 */
				const access_type access( const storage_index_type &storageIndex ) const {
					return container[ storageIndex ];
				}

				access_type access( const storage_index_type &storageIndex ) {
					return container[ storageIndex ];
				}

				storage_index_type getStorageIndex( const size_t i, const size_t j, const size_t s, const size_t P ) const {
					return amf.getStorageIndex( i, j, s, P );
				}

				/**
				 * @brief Construct a new structured matrix Base object assigning identity
				 * mapping functions both to the row and column dimensions.
				 *
				 * @param rows The number of rows of the matrix.
				 * @param cols The number of columns of the matrix.
				 * @param smf  The storage mapping function assigned to this matrix.
				 */
				/** (Documentation of old MatrixCotiainer. TODO: merge with the above docs.
				 * @brief Construct a new structured matrix container object.
				 *
				 * \warning \a cap is present for compatibility with other matrix specializations.
				 *          In reference backend, the number of non-zeros (i.e. capacity)
				 *          depends on the used storage scheme. Therefore, this parameter is
				 *          ignored.
				 *
				 * TODO: Add the storage scheme a parameter to the constructor
				 * so that allocation can be made accordingly, generalizing the full case.
				 */
				StorageBasedMatrix( storage::AMF< ImfR, ImfC, MappingPolynomial > amf ) :
					// enable only if ImfR and ImfC are imf::Id
					container( internal::Vector< T, reference >( amf.getStorageDimensions() ) ),
					amf( amf ) {}

				/** View on another container */
				StorageBasedMatrix( Vector< T, reference > &container, storage::AMF< ImfR, ImfC, MappingPolynomial > amf ) :
					container( container ),
					amf( amf ) {}

		}; // class StorageBasedMatrix

		/**
		 * Specialization of MatrixReference with a lambda function as a target.
		 * Used as a result of low-rank operation to avoid the need for allocating a container.
		 * The data is produced lazily by invoking the lambda function stored as a part of this object.
		 *
		 * \note Views-over-lambda-functions types are used internally as results of low-rank operations and are not
		 *       directly exposed to users. From the users perspective, the use of objects of this type does not differ
		 *       from the use of other \a alp::Matrix types. The difference lies in a lazy implementation of the access
		 *       to matrix elements, which is not exposed to the user.
		 *
		 */
		template< typename T, typename ImfR, typename ImfC, typename DataLambdaType >
		class FunctorBasedMatrix : public MatrixBase< FunctorBasedMatrix< T, ImfR, ImfC, DataLambdaType > > {
			public:

				/** Expose static properties */
				typedef T value_type;
				/** Type returned by access function */
				typedef T access_type;
				/** Type of the index used to access the physical storage */
				typedef std::pair< size_t, size_t > storage_index_type;

			protected:

				typedef FunctorBasedMatrix< T, ImfR, ImfC, DataLambdaType > self_type;
				friend MatrixBase< self_type >;

				typedef std::function< bool() > initialized_functor_type;
				const initialized_functor_type initialized_lambda;

				const ImfR imf_r;
				const ImfC imf_c;

				const DataLambdaType data_lambda;

				std::pair< size_t, size_t > dims() const noexcept {
					return std::make_pair( imf_r.n, imf_c.n );
				}

				const DataLambdaType &getFunctor() const noexcept {
					return data_lambda;
				}

				bool getInitialized() const noexcept {
					return initialized_lambda();
				}

				void setInitialized( const bool ) noexcept {
					static_assert( "Calling setInitialized on a FunctorBasedMatrix is not allowed." );
				}

				access_type access( const storage_index_type &storage_index ) const {
					T result = 0;
					data_lambda( result, imf_r.map( storage_index.first ), imf_c.map( storage_index.second ) );
					return static_cast< access_type >( result );
				}

				storage_index_type getStorageIndex( const size_t i, const size_t j, const size_t s, const size_t P ) const {
					(void)s;
					(void)P;
					return std::make_pair( i, j );
				}

			public:

				FunctorBasedMatrix(
					initialized_functor_type initialized_lambda,
					ImfR imf_r,
					ImfC imf_c,
					const DataLambdaType data_lambda
				) :
					initialized_lambda( initialized_lambda ),
					imf_r( imf_r ),
					imf_c( imf_c ),
					data_lambda( data_lambda ) {}

		}; // class FunctorBasedMatrix
	} // namespace internal

	/**
	 * \brief An ALP structured matrix.
	 *
	 * This is an opaque data type for structured matrices.
	 *
	 * A structured matrix exposes a mathematical
	 * \em logical layout which allows to express implementation-oblivious concepts
	 * including the matrix structure itself and \em views on the matrix.
	 * The logical layout of a structured matrix maps to a physical counterpart via
	 * a storage scheme which typically depends on the chosen structure and the selected
	 * backend. alp::Matrix and alp::Vector may be used as interfaces to such a physical
	 * layout.
	 * To visualize this, you may think of a band matrix. Using a
	 * full dense or a banded storage schemes would require
	 * the use of a \a alp::Matrix container (see include/alp/density.hpp for
	 * more details about the supported storage schemes). However, the interpration of its
	 * content would differ in the two cases being a function of both the Structure
	 * information and the storage scheme combined.
	 *
	 * Views can be used to create logical \em perspectives on top of a container.
	 * For example, one may decide to refer to the transpose of a matrix or to treat
	 * for a limited part of my program a square matrix as symmetric.
	 * If a view can be expressed as a concept \em invariant of specific runtime features,
	 * such views can be defined statically (for example, one may always refer to the
	 * transpose or the diagonal of a matrix irrespective of features such as the matrix's
	 * size). Other may depend on features such as the size of a matrix
	 * (e.g., gathering/scattering the rows/columns of a matrix or permuting them).
	 *
	 * Structured matrices defined as views on other matrices do not instantiate a
	 * new container but refer to the one used by their targets.
	 * See the two specializations
	 * \a Matrix<T, structures::General, Density::Dense, View, reference >
	 * and \a Matrix<T, structures::General, Density::Dense, view::Indentity<void>, reference >
	 * as examples of structured matrix types without and with physical container, respectively.
	 *
	 *
	 * @tparam T         The type of the matrix elements. \a T shall not be a GraphBLAS
	 *                   type.
	 * @tparam Structure One of the matrix structures defined in \a alp::structures.
	 * @tparam density   Either \em enum \a Density::Dense or \em enum
	 *                   \a storage::Sparse.
	 * @tparam View      One of the matrix views in \a alp::view.
	 *                   All static views except for \a view::Original (via
	 *                   \a view::Original<void> cannot instantiate a new container
	 *                   and only allow to refer to a previously defined
	 *                   \a Matrix.
	 *                   The \a View parameter should not be used directly
	 *                   by the user but selected via \a get_view function.
	 *
	 * See examples of Matrix definitions within \a include/alp/reference/matrix.hpp
	 * and the \a dense_structured_matrix.cpp unit test.
	 *
	 */
	template< typename T, typename Structure, enum Density density, typename View, typename ImfR, typename ImfC >
	class Matrix< T, Structure, density, View, ImfR, ImfC, reference > { };

	/**
	 * @brief General matrix with physical container.
	 */
	template< typename T, typename View, typename ImfR, typename ImfC >
	class Matrix< T, structures::General, Density::Dense, View, ImfR, ImfC, reference > :
		public std::conditional<
			internal::is_view_over_functor< View >::value,
			internal::FunctorBasedMatrix< T, ImfR, ImfC, typename View::applied_to >,
			internal::StorageBasedMatrix< T, ImfR, ImfC,
				typename storage::polynomials::apply_view< View::type_id, storage::polynomials::Full_type >::type,
				internal::requires_allocation< View >::value >
		>::type {

		protected:
			typedef Matrix< T, structures::General, Density::Dense, View, ImfR, ImfC, reference > self_type;
			typedef typename View::applied_to target_type;

			/*********************
				Storage info friends
			******************** */

			template< typename fwd_iterator >
			friend RC buildMatrix( Matrix< T, structures::General, Density::Dense, View, ImfR, ImfC, reference > & A,
				const fwd_iterator & start, const fwd_iterator & end );

			template< typename fwd_iterator >
			RC buildMatrixUnique( const fwd_iterator & start, const fwd_iterator & end ) {
				std::cout << "Building Matrix<>; calling buildMatrix( Matrix<> )\n";
				return buildMatrix( *(this->_container), start, end );
			}

		public:
			/** Exposes the types and the static properties. */
			typedef structures::General structure;
			typedef typename storage::polynomials::apply_view< View::type_id, storage::polynomials::Full_type >::type mapping_polynomial_type;
			/**
			 * Indicates if a matrix needs to allocate data-related memory
			 * (for the internal container or functor object).
			 * False if it is a view over another matrix or a functor.
			 */
			static constexpr bool requires_allocation = internal::requires_allocation< View >::value;

			/**
			 * Expose the base type class to enable internal functions to cast
			 * the type of objects of this class to the base class type.
			 */
			typedef typename std::conditional<
				internal::is_view_over_functor< View >::value,
				internal::FunctorBasedMatrix< T, ImfR, ImfC, target_type >,
				internal::StorageBasedMatrix< T, ImfR, ImfC, mapping_polynomial_type, requires_allocation >
			>::type base_type;

			// A general Structure knows how to define a reference to itself (which is an original reference view)
			// as well as other static views.
			template < view::Views view_tag, bool d=false >
			struct view_type;

			template < bool d >
			struct view_type< view::original, d > {
				using type = Matrix< T, structures::General, Density::Dense, View, ImfR, ImfC, reference >;
			};

			template < bool d >
			struct view_type< view::transpose, d > {
				using type = Matrix< T, structures::General, Density::Dense, view::Transpose< self_type >, ImfR, ImfC, reference >;
			};

			template < bool d >
			struct view_type< view::diagonal, d > {
				using type = Vector< T, structures::General, Density::Dense, view::Diagonal< self_type >, imf::Id, reference >;
			};

			/**
			 * Constructor for an original matrix.
			 *
			 * @tparam ViewType A dummy type.
			 *                  Uses SFINAE to enable this constructor only for
			 *                  a storage-based matrix that allocates memory.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_storage< ViewType >::value &&
					internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( const size_t rows, const size_t cols, const size_t cap = 0 ) :
				internal::StorageBasedMatrix< T, ImfR, ImfC, mapping_polynomial_type, requires_allocation >(
					storage::AMF< ImfR, ImfC, mapping_polynomial_type >(
						imf::Id( rows ),
						imf::Id( cols ),
						storage::polynomials::Create< mapping_polynomial_type >( cols ),
						rows * cols
					)
				) {
				(void)cap;
			}

			/**
			 * Constructor for a view over another storage-based matrix.
			 *
			 * @tparam ViewType The dummy View type of the constructed matrix.
			 *                  Uses SFINAE to enable this constructor only for
			 *                 	a view over a storage-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_storage< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix, ImfR imf_r, ImfC imf_c ) :
				internal::StorageBasedMatrix< T, ImfR, ImfC, mapping_polynomial_type, requires_allocation >(
					getContainer( target_matrix ),
					storage::AMFFactory::Create( target_matrix.amf, imf_r, imf_c )
				) {}

			/**
			 * Constructor for a view over another matrix using default IMFs (Identity).
			 * Delegate to the general constructor.
			 *
			 * @tparam ViewType The dummy View type of the constructed matrix.
			 *                  Uses SFINAE to enable this constructor only for
			 *                 	a view over a storage-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_storage< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix ) :
				Matrix( target_matrix,
					imf::Id( nrows ( target_matrix ) ),
					imf::Id( ncols ( target_matrix ) ) ) {}

			/**
			 * Constructor for a view over another storage-based matrix.
			 *
			 * @tparam ViewType The dummy View type of the constructed matrix.
			 *                  Uses SFINAE to enable this constructor only for
			 *                 	a view over a storage-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_storage< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix, storage::AMF< ImfR, ImfC, mapping_polynomial_type > amf ) :
				internal::StorageBasedMatrix< T, ImfR, ImfC, mapping_polynomial_type, requires_allocation >(
					getContainer( target_matrix ),
					amf
				) {}

			/**
			 * Constructor for a functor-based matrix that allocates memory.
			 *
			 * @tparam ViewType A dummy type.
			 *                  Uses SFINAE to enable this constructor only for
			 *                  a functor-based matrix that allocates memory.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_functor< ViewType >::value &&
					internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( std::function< bool() > initialized, const size_t rows, const size_t cols, typename ViewType::applied_to lambda ) :
				internal::FunctorBasedMatrix< T, ImfR, ImfC, typename View::applied_to >( initialized, rows, cols, lambda ) {}

			/**
			 * Constructor for a view over another functor-based matrix.
			 *
			 * @tparam ViewType The dummy View type of the constructed matrix.
			 *                  Uses SFINAE to enable this constructor only for
			 *                  a view over a functor-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_functor< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix, ImfR imf_r, ImfC imf_c ) :
				internal::FunctorBasedMatrix< T, ImfR, ImfC, typename View::applied_to >(
					getFunctor( target_matrix ),
					imf_r, imf_c
				) {}

			/**
			 * Constructor for a view over another functor-based matrix.
			 *
			 * @tparam ViewType The dummy View type of the constructed matrix.
			 *                  Uses SFINAE to enable this constructor only for
			 *                  a view over a functor-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_functor< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix ) :
				Matrix( getFunctor( target_matrix ),
					imf::Id( nrows ( target_matrix ) ),
					imf::Id( ncols ( target_matrix ) )
				) {}
	}; // General Matrix

	/**
	 * @brief Square matrix
	 */
	template< typename T, typename View, typename ImfR, typename ImfC >
	class Matrix< T, structures::Square, Density::Dense, View, ImfR, ImfC, reference > :
		public std::conditional<
			internal::is_view_over_functor< View >::value,
			internal::FunctorBasedMatrix< T, ImfR, ImfC, typename View::applied_to >,
			internal::StorageBasedMatrix< T, ImfR, ImfC,
				typename storage::polynomials::apply_view< View::type_id, storage::polynomials::Full_type >::type,
				internal::requires_allocation< View >::value >
		>::type {

		protected:
			typedef Matrix< T, structures::Square, Density::Dense, View, ImfR, ImfC, reference > self_type;
			typedef typename View::applied_to target_type;

			/*********************
				Storage info friends
			******************** */

			template< typename fwd_iterator >
			friend RC buildMatrix( Matrix< T, structures::Square, Density::Dense, View, ImfR, ImfC, reference > & A,
				const fwd_iterator & start, const fwd_iterator & end );

			template< typename fwd_iterator >
			RC buildMatrixUnique( const fwd_iterator & start, const fwd_iterator & end ) {
				std::cout << "Building Matrix<>; calling buildMatrix( Matrix<> )\n";
				return buildMatrix( *(this->_container), start, end );
			}

		public:
			/** Exposes the types and the static properties. */
			typedef structures::Square structure;
			typedef typename storage::polynomials::apply_view< View::type_id, storage::polynomials::Full_type >::type mapping_polynomial_type;
			/**
			 * Indicates if a matrix needs to allocate data-related memory
			 * (for the internal container or functor object).
			 * False if it is a view over another matrix or a functor.
			 */
			static constexpr bool requires_allocation = internal::requires_allocation< View >::value;

			/**
			 * Expose the base type class to enable internal functions to cast
			 * the type of objects of this class to the base class type.
			 */
			typedef typename std::conditional<
				internal::is_view_over_functor< View >::value,
				internal::FunctorBasedMatrix< T, ImfR, ImfC, target_type >,
				internal::StorageBasedMatrix< T, ImfR, ImfC, mapping_polynomial_type, requires_allocation >
			>::type base_type;

			// A general Structure knows how to define a reference to itself (which is an original reference view)
			// as well as other static views.
			template < view::Views view_tag, bool d=false >
			struct view_type;

			template < bool d >
			struct view_type< view::original, d > {
				using type = Matrix< T, structures::Square, Density::Dense, View, ImfR, ImfC, reference >;
			};

			template < bool d >
			struct view_type< view::transpose, d > {
				using type = Matrix< T, structures::Square, Density::Dense, view::Transpose< self_type >, ImfR, ImfC, reference >;
			};

			template < bool d >
			struct view_type< view::diagonal, d > {
				using type = Vector< T, structures::Square, Density::Dense, view::Diagonal< self_type >, imf::Id, reference >;
			};

			/**
			 * Constructor for an original matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_storage< ViewType >::value &&
					internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( const size_t dim, const size_t cap = 0 ) :
				internal::StorageBasedMatrix< T, ImfR, ImfC, mapping_polynomial_type, requires_allocation >(
					storage::AMF< ImfR, ImfC, mapping_polynomial_type >(
						imf::Id( dim ),
						imf::Id( dim ),
						storage::polynomials::Create< mapping_polynomial_type >( dim ),
						dim * dim
					)
				) {
				(void)cap;
			}

			/**
			 * Constructor for a view over another storage-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_storage< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix, ImfR imf_r, ImfC imf_c ) :
				internal::StorageBasedMatrix< T, ImfR, ImfC, mapping_polynomial_type, requires_allocation >(
					getContainer( target_matrix ),
					storage::AMFFactory::Create( target_matrix.amf, imf_r, imf_c )
				) {}

			/**
			 * Constructor for a view over another matrix using default IMFs (Identity).
			 * Delegate to the general constructor.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_storage< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix ) :
				Matrix( target_matrix,
					imf::Id( nrows ( target_matrix ) ),
					imf::Id( ncols ( target_matrix ) ) ) {}

			/**
			 * Constructor for a view over another storage-based matrix.
			 *
			 * @tparam ViewType The dummy View type of the constructed matrix.
			 *                  Uses SFINAE to enable this constructor only for
			 *                 	a view over a storage-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_storage< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix, storage::AMF< ImfR, ImfC, mapping_polynomial_type > amf ) :
				internal::StorageBasedMatrix< T, ImfR, ImfC, mapping_polynomial_type, requires_allocation >(
					getContainer( target_matrix ),
					amf
				) {}

			/**
			 * Constructor for a functor-based matrix that allocates memory.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_functor< ViewType >::value &&
					internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( bool initialized, const size_t dim, typename ViewType::applied_to lambda ) :
				internal::FunctorBasedMatrix< T, ImfR, ImfC, typename View::applied_to >( initialized, dim, dim, lambda ) {}

			/**
			 * Constructor for a view over another functor-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_functor< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix, ImfR imf_r, ImfC imf_c ) :
				internal::FunctorBasedMatrix< T, ImfR, ImfC, typename View::applied_to >(
					getFunctor( target_matrix ),
					imf_r, imf_c
				) {}

			/**
			 * Constructor for a view over another functor-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_functor< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix ) :
				Matrix( getFunctor( target_matrix ),
					imf::Id( nrows ( target_matrix ) ),
					imf::Id( ncols ( target_matrix ) )
				) {}
	}; // Square Matrix

	/**
	 * @brief Symmetric matrix
	 */
	template< typename T, typename View, typename ImfR, typename ImfC >
	class Matrix< T, structures::Symmetric, Density::Dense, View, ImfR, ImfC, reference > :
		public std::conditional<
			internal::is_view_over_functor< View >::value,
			internal::FunctorBasedMatrix< T, ImfR, ImfC, typename View::applied_to >,
			internal::StorageBasedMatrix< T, ImfR, ImfC,
				typename storage::polynomials::apply_view< View::type_id, storage::polynomials::Full_type >::type,
				internal::requires_allocation< View >::value >
		>::type {

		protected:
			typedef Matrix< T, structures::Symmetric, Density::Dense, View, ImfR, ImfC, reference > self_type;
			typedef typename View::applied_to target_type;

			/*********************
				Storage info friends
			******************** */

			template< typename fwd_iterator >
			friend RC buildMatrix( Matrix< T, structures::Symmetric, Density::Dense, View, ImfR, ImfC, reference > & A,
				const fwd_iterator & start, const fwd_iterator & end );

			template< typename fwd_iterator >
			RC buildMatrixUnique( const fwd_iterator & start, const fwd_iterator & end ) {
				std::cout << "Building Matrix<>; calling buildMatrix( Matrix<> )\n";
				return buildMatrix( *(this->_container), start, end );
			}

		public:
			/** Exposes the types and the static properties. */
			typedef structures::Symmetric structure;
			typedef typename storage::polynomials::apply_view< View::type_id, storage::polynomials::Full_type >::type mapping_polynomial_type;
			/**
			 * Indicates if a matrix needs to allocate data-related memory
			 * (for the internal container or functor object).
			 * False if it is a view over another matrix or a functor.
			 */
			static constexpr bool requires_allocation = internal::requires_allocation< View >::value;

			/**
			 * Expose the base type class to enable internal functions to cast
			 * the type of objects of this class to the base class type.
			 */
			typedef typename std::conditional<
				internal::is_view_over_functor< View >::value,
				internal::FunctorBasedMatrix< T, ImfR, ImfC, target_type >,
				internal::StorageBasedMatrix< T, ImfR, ImfC, mapping_polynomial_type, requires_allocation >
			>::type base_type;

			// A general Structure knows how to define a reference to itself (which is an original reference view)
			// as well as other static views.
			template < view::Views view_tag, bool d=false >
			struct view_type;

			template < bool d >
			struct view_type< view::original, d > {
				using type = Matrix< T, structures::Symmetric, Density::Dense, View, ImfR, ImfC, reference >;
			};

			template < bool d >
			struct view_type< view::transpose, d > {
				using type = Matrix< T, structures::Symmetric, Density::Dense, view::Transpose< self_type >, ImfR, ImfC, reference >;
			};

			template < bool d >
			struct view_type< view::diagonal, d > {
				using type = Vector< T, structures::Symmetric, Density::Dense, view::Diagonal< self_type >, imf::Id, reference >;
			};

			/**
			 * Constructor for an original matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_storage< ViewType >::value &&
					internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( const size_t dim, const size_t cap = 0 ) :
				internal::StorageBasedMatrix< T, ImfR, ImfC, mapping_polynomial_type, requires_allocation >(
					storage::AMF< ImfR, ImfC, mapping_polynomial_type >(
						imf::Id( dim ),
						imf::Id( dim ),
						storage::polynomials::Create< mapping_polynomial_type >( dim ),
						dim * dim
					)
				) {
				(void)cap;
			}

			/**
			 * Constructor for a view over another storage-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_storage< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix, ImfR imf_r, ImfC imf_c ) :
				internal::StorageBasedMatrix< T, ImfR, ImfC, mapping_polynomial_type, requires_allocation >(
					getContainer( target_matrix ),
					storage::AMFFactory::Create( target_matrix.amf, imf_r, imf_c )
				) {}

			/**
			 * Constructor for a view over another matrix using default IMFs (Identity).
			 * Delegate to the general constructor.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_storage< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix ) :
				Matrix( target_matrix,
					imf::Id( nrows ( target_matrix ) ),
					imf::Id( ncols ( target_matrix ) ) ) {}

			/**
			 * Constructor for a functor-based matrix that allocates memory.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_functor< ViewType >::value &&
					internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( std::function< bool() > initialized, const size_t dim, typename ViewType::applied_to lambda ) :
				internal::FunctorBasedMatrix< T, ImfR, ImfC, typename View::applied_to >( initialized, dim, dim, lambda ) {}

			/**
			 * Constructor for a view over another functor-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_functor< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix, ImfR imf_r, ImfC imf_c ) :
				internal::FunctorBasedMatrix< T, ImfR, ImfC, typename View::applied_to >(
					getFunctor( target_matrix ),
					imf_r, imf_c
				) {}

			/**
			 * Constructor for a view over another functor-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_functor< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix ) :
				Matrix( getFunctor( target_matrix ),
					imf::Id( nrows ( target_matrix ) ),
					imf::Id( ncols ( target_matrix ) )
				) {}
	}; // Symmetric Matrix

	/**
	 * @brief UpperTriangular matrix with physical container.
	 */
	template< typename T, typename View, typename ImfR, typename ImfC >
	class Matrix< T, structures::UpperTriangular, Density::Dense, View, ImfR, ImfC, reference > :
		public std::conditional<
			internal::is_view_over_functor< View >::value,
			internal::FunctorBasedMatrix< T, ImfR, ImfC, typename View::applied_to >,
			internal::StorageBasedMatrix< T, ImfR, ImfC,
				typename storage::polynomials::apply_view< View::type_id, storage::polynomials::Full_type >::type,
				internal::requires_allocation< View >::value >
		>::type {

		protected:
			typedef Matrix< T, structures::UpperTriangular, Density::Dense, View, ImfR, ImfC, reference > self_type;
			typedef typename View::applied_to target_type;

			/*********************
				Storage info friends
			******************** */

			template< typename fwd_iterator >
			friend RC buildMatrix( Matrix< T, structures::UpperTriangular, Density::Dense, View, ImfR, ImfC, reference > & A,
				const fwd_iterator & start, const fwd_iterator & end );

			template< typename fwd_iterator >
			RC buildMatrixUnique( const fwd_iterator & start, const fwd_iterator & end ) {
				std::cout << "Building Matrix<>; calling buildMatrix( Matrix<> )\n";
				return buildMatrix( *(this->_container), start, end );
			}

		public:
			/** Exposes the types and the static properties. */
			typedef structures::UpperTriangular structure;
			typedef typename storage::polynomials::apply_view< View::type_id, storage::polynomials::Full_type >::type mapping_polynomial_type;
			/**
			 * Indicates if a matrix needs to allocate data-related memory
			 * (for the internal container or functor object).
			 * False if it is a view over another matrix or a functor.
			 */
			static constexpr bool requires_allocation = internal::requires_allocation< View >::value;

			/**
			 * Expose the base type class to enable internal functions to cast
			 * the type of objects of this class to the base class type.
			 */
			typedef typename std::conditional<
				internal::is_view_over_functor< View >::value,
				internal::FunctorBasedMatrix< T, ImfR, ImfC, target_type >,
				internal::StorageBasedMatrix< T, ImfR, ImfC, mapping_polynomial_type, requires_allocation >
			>::type base_type;

			// A general Structure knows how to define a reference to itself (which is an original reference view)
			// as well as other static views.
			template < view::Views view_tag, bool d=false >
			struct view_type;

			template < bool d >
			struct view_type< view::original, d > {
				using type = Matrix< T, structures::UpperTriangular, Density::Dense, View, ImfR, ImfC, reference >;
			};

			template < bool d >
			struct view_type< view::transpose, d > {
				using type = Matrix< T, structures::LowerTriangular, Density::Dense, view::Transpose< self_type >, ImfR, ImfC, reference >;
			};

			template < bool d >
			struct view_type< view::diagonal, d > {
				using type = Vector< T, structures::UpperTriangular, Density::Dense, view::Diagonal< self_type >, imf::Id, reference >;
			};

			/**
			 * Constructor for an original matrix.
			 *
			 * @tparam ViewType A dummy type.
			 *                  Uses SFINAE to enable this constructor only for
			 *                  a storage-based matrix that allocates memory.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_storage< ViewType >::value &&
					internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( const size_t dim, const size_t cap = 0 ) :
				internal::StorageBasedMatrix< T, ImfR, ImfC, mapping_polynomial_type, requires_allocation >(
					storage::AMF< ImfR, ImfC, mapping_polynomial_type >(
						imf::Id( dim ),
						imf::Id( dim ),
						storage::polynomials::Create< mapping_polynomial_type >( dim ),
						dim * dim
					)
				) {
				(void)cap;
			}

			/**
			 * Constructor for a view over another storage-based matrix.
			 *
			 * @tparam ViewType The dummy View type of the constructed matrix.
			 *                  Uses SFINAE to enable this constructor only for
			 *                 	a view over a storage-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_storage< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix, ImfR imf_r, ImfC imf_c ) :
				internal::StorageBasedMatrix< T, ImfR, ImfC, mapping_polynomial_type, requires_allocation >(
					getContainer( target_matrix ),
					storage::AMFFactory::Create( target_matrix.amf, imf_r, imf_c )
				) {}

			/**
			 * Constructor for a view over another matrix using default IMFs (Identity).
			 * Delegate to the general constructor.
			 *
			 * @tparam ViewType The dummy View type of the constructed matrix.
			 *                  Uses SFINAE to enable this constructor only for
			 *                 	a view over a storage-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_storage< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix ) :
				Matrix( target_matrix,
					imf::Id( nrows ( target_matrix ) ),
					imf::Id( ncols ( target_matrix ) ) ) {}

			/**
			 * Constructor for a functor-based matrix that allocates memory.
			 *
			 * @tparam ViewType A dummy type.
			 *                  Uses SFINAE to enable this constructor only for
			 *                  a functor-based matrix that allocates memory.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_functor< ViewType >::value &&
					internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( bool initialized, const size_t dim, typename ViewType::applied_to lambda ) :
				internal::FunctorBasedMatrix< T, ImfR, ImfC, typename View::applied_to >( initialized, dim, lambda ) {}

			/**
			 * Constructor for a view over another functor-based matrix.
			 *
			 * @tparam ViewType The dummy View type of the constructed matrix.
			 *                  Uses SFINAE to enable this constructor only for
			 *                  a view over a functor-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_functor< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix, ImfR imf_r, ImfC imf_c ) :
				internal::FunctorBasedMatrix< T, ImfR, ImfC, typename View::applied_to >(
					getFunctor( target_matrix ),
					imf_r, imf_c
				) {}

			/**
			 * Constructor for a view over another functor-based matrix.
			 *
			 * @tparam ViewType The dummy View type of the constructed matrix.
			 *                  Uses SFINAE to enable this constructor only for
			 *                  a view over a functor-based matrix.
			 */
			template<
				typename ViewType = View,
				std::enable_if_t<
					internal::is_view_over_functor< ViewType >::value &&
					!internal::requires_allocation< ViewType >::value
				> * = nullptr
			>
			Matrix( typename ViewType::applied_to &target_matrix ) :
				Matrix( getFunctor( target_matrix ),
					imf::Id( nrows ( target_matrix ) ),
					imf::Id( ncols ( target_matrix ) )
				) {}
	}; // UpperTriangular Matrix

	namespace structures {

		/**
		 * @brief Checks if TestedStructure is a \a Structure according to the ALP's structure classification.
		 *
		 * @tparam TestedStructure   The structure to be tested.
		 * @tparam Structure 		 The structure that should be implied by \a TestedStructure.
		 */
		template< typename TestedStructure, typename Structure >
		struct is_a {

			static_assert( std::is_base_of< structures::BaseStructure, TestedStructure >::value );

			/**
			 * \a value is true iff \a Structure is implied by \a TestedStructure.
			 */
			static constexpr bool value = is_in< Structure, typename TestedStructure::inferred_structures >::value;
		};

		template<size_t band, typename T, typename Structure, enum Density density, typename View, typename ImfL, typename ImfR >
		std::ptrdiff_t get_lower_bandwidth(const alp::Matrix< T, Structure, density, View, ImfL, ImfR, reference > &A) {

			const std::ptrdiff_t m = nrows( A );
			constexpr std::ptrdiff_t cl_a = std::tuple_element< band, typename Structure::band_intervals >::type::left;

			const std::ptrdiff_t l_a = ( cl_a < -m + 1 ) ? -m + 1 : cl_a ;

			return l_a;

		}

		template<size_t band, typename T, typename Structure, enum Density density, typename View, typename ImfL, typename ImfR >
		std::ptrdiff_t get_upper_bandwidth(const alp::Matrix< T, Structure, density, View, ImfL, ImfR, reference > &A) {

			const std::ptrdiff_t n = ncols( A );
			constexpr std::ptrdiff_t cu_a = std::tuple_element< band, typename Structure::band_intervals >::type::right;

			const std::ptrdiff_t u_a = ( cu_a > n ) ? n : cu_a ;

			return u_a;

		}

	} // namespace structures

	/**
     *
	 * @brief Generate an original view of \a source maintaining the same \a Structure.
	 * 		  The function guarantees the created view is non-overlapping with other
	 *        existing views only when the check can be performed in constant time.
	 *
	 * @tparam T         The matrix' elements type
	 * @tparam Structure The structure of the source and target matrix view
	 * @tparam density   The type (i.e., sparse or dense) of storage scheme
	 * @tparam View      The source's View type
	 * @tparam backend   The target backend
	 *
	 * @param source     The source matrix
	 *
	 * @return A new original view over the source structured matrix.
	 *
     * \parblock
     * \par Performance semantics.
     *        -# This function performs
     *           \f$ \Theta(nref) \f$ amount of work where \f$ nref \f$ is the number
	 * 			 of available views of \a source.
     *        -# A call to this function may use \f$ \mathcal{O}(1) \f$ bytes
     *           of memory beyond the memory in use at the function call entry.
     *        -# This function may make system calls.
     * \endparblock
	 *
	 */
	template<
		typename SourceMatrixType,
		std::enable_if_t< is_matrix< SourceMatrixType >::value, void > * = nullptr
	>
	typename SourceMatrixType::template view_type< view::original >::type
	get_view( SourceMatrixType &source ) {

		using target_strmat_t = typename SourceMatrixType::template view_type< view::original >::type;

		target_strmat_t target( source );

		return target;
	}

	/**
     *
	 * @brief Generate a view specified by \a target_view where the type is compliant with the
	 * 		  \a source matrix.
	 * 		  The function guarantees the created view is non-overlapping with other
	 *        existing views only when the check can be performed in constant time.
	 *
	 * @tparam target_view  One of the supported views listed in \a view::Views
	 * @tparam T            The matrix' elements type
	 * @tparam Structure    The structure of the source and target matrix view
	 * @tparam density      The type (i.e., sparse or dense) of storage scheme
	 * @tparam View         The source's View type
	 * @tparam backend      The target backend
	 *
	 * @param source        The source structured matrix
	 *
	 * @return A new \a target_view view over the source matrix.

     * \parblock
     * \par Performance semantics.
     *        -# This function performs
     *           \f$ \Theta(nref) \f$ amount of work where \f$ nref \f$ is the number
	 * 			 of available views of \a source.
     *        -# A call to this function may use \f$ \mathcal{O}(1) \f$ bytes
     *           of memory beyond the memory in use at the function call entry.
     *        -# This function may make system calls.
     * \endparblock
	 *
	 */
	template<
		enum view::Views target_view,
		typename SourceMatrixType,
		std::enable_if_t< is_matrix< SourceMatrixType >::value, void > * = nullptr
	>
	typename SourceMatrixType::template view_type< target_view >::type
	get_view( SourceMatrixType &source ) {

		using target_strmat_t = typename SourceMatrixType::template view_type< target_view >::type;

		target_strmat_t target( source );

		return target;
	}

	/**
     *
	 * @brief Generate an original view where the type is compliant with the source Matrix.
	 * 		  Version where a target structure is specified. It can only generate a valide type if the target
	 * 		  structure is the same as the source's
	 * 		  or a more specialized one that would preserve its static properties (e.g., symmetric reference
	 * 		  to a square matrix -- any assumption based on symmetry would not break those based on square).
	 * 		  The function guarantees the created view is non-overlapping with other existing views only when the
	 * 		  check can be performed in constant time.
	 *
	 * @tparam TargetStructure  The target structure of the new view. It should verify
	 *                          <code> alp::is_in<Structure, TargetStructure::inferred_structures> </code>.
	 * @tparam T                The matrix's elements type
	 * @tparam Structure        The structure of the source and target matrix view
	 * @tparam density          The type (i.e., \a alp::Density:Dense or \a alp::Density:Sparse) of storage scheme
	 * @tparam View             The source's View type
	 * @tparam backend          The target backend
	 *
	 * @param source            The source structured matrix
	 *
	 * @return A new original view over the source structured matrix.
	 *
     * \parblock
     * \par Performance semantics.
     *        -# This function performs
     *           \f$ \Theta(nref) \f$ amount of work where \f$ nref \f$ is the number
	 * 			 of available views of \a source.
     *        -# A call to this function may use \f$ \mathcal{O}(1) \f$ bytes
     *           of memory beyond the memory in use at the function call entry.
     *        -# This function may make system calls.
     * \endparblock
	 *
	 */
	template<
		typename TargetStructure,
		typename T, typename Structure, enum Density density, typename View, typename ImfR, typename ImfC, enum Backend backend >
	Matrix< T, TargetStructure, density, view::Original< Matrix< T, Structure, density, View, ImfR, ImfC, backend > >, ImfR, ImfC, backend >
	get_view( Matrix< T, Structure, density, View, ImfR, ImfC, backend > &source ) {

		static_assert( structures::is_in< Structure, typename TargetStructure::inferred_structures >::value,
			"Can only create a view when the target structure is compatible with the source." );

		using source_strmat_t = Matrix< T, Structure, density, View, ImfR, ImfC, backend >;
		using target_strmat_t = Matrix< T, TargetStructure, density, view::Original< source_strmat_t >, ImfR, ImfC, backend >;

		target_strmat_t target( source );

		return target;
	}

	namespace internal {
		/**
		 * Implement a gather through a View over compatible Structure using provided Index Mapping Functions.
		 * The compatibility depends on the TargetStructure, SourceStructure and IMFs, and is calculated during runtime.
		 */

		template<
			typename TargetStructure, typename TargetImfR, typename TargetImfC,
			typename T, typename Structure, enum Density density, typename View, typename ImfR, typename ImfC, enum Backend backend >
		alp::Matrix<
			T,
			TargetStructure,
			density,
			view::Original< alp::Matrix< T, Structure, density, View, ImfR, ImfC, backend > >,
			typename imf::composed_type< TargetImfR, ImfR >::type,
			typename imf::composed_type< TargetImfC, ImfC >::type,
			backend
		>
		get_view( alp::Matrix< T, Structure, density, View, ImfR, ImfC, backend > &source,
				TargetImfR imf_r, TargetImfC imf_c ) {

			//if( std::dynamic_pointer_cast< imf::Select >( imf_r ) || std::dynamic_pointer_cast< imf::Select >( imf_c ) ) {
			//	throw std::runtime_error("Cannot gather with imf::Select yet.");
			//}
			// No static check as the compatibility depends on IMF, which is a runtime level parameter
			//if( ! (TargetStructure::template isInstantiableFrom< Structure >( static_cast< TargetImfR & >( imf_r ), static_cast< TargetImfR & >( imf_c ) ) ) ) {
			if( ! (structures::isInstantiable< Structure, TargetStructure >::check( static_cast< TargetImfR & >( imf_r ), static_cast< TargetImfR & >( imf_c ) ) ) ) {
				throw std::runtime_error("Cannot gather into specified TargetStructure from provided SourceStructure and Index Mapping Functions.");
			}

			using source_strmat_t = alp::Matrix< T, Structure, density, View, ImfR, ImfC, backend >;
			using target_strmat_t = alp::Matrix< T, TargetStructure, density, view::Original< source_strmat_t >, TargetImfR, TargetImfC, backend >;

			target_strmat_t target( source, imf_r, imf_c );

			return target;
		}
	} // namespace internal

	/**
     *
	 * @brief Generate an original view where the type is compliant with the source Matrix.
	 * Version where a range of rows and columns are selected to form a new view with specified target
	 * structure. It can only generate a valide type if the target
	 * structure is guaranteed to preserve the static properties of the source's structure.
	 * A structural check of this kind as well as non-overlapping checks with existing views of \a source
	 * are guaranteed only when each one of them incurs constant time work.
	 *
	 * @tparam TargetStructure  The target structure of the new view. It should verify
	 *                          <code> alp::is_in<Structure, TargetStructure::inferred_structures> </code>.
	 * @tparam T                The matrix' elements type
	 * @tparam Structure        The structure of the source and target matrix view
	 * @tparam density          The type (i.e., sparse or dense) of storage scheme
	 * @tparam View             The source's View type
	 * @tparam backend          The target backend
	 *
	 * @param source            The source structured matrix
	 * @param rng_r             A valid range of rows
	 * @param rng_c             A valid range of columns
	 *
	 * @return A new original view over the source structured matrix.
	 *
     * \parblock
     * \par Performance semantics.
     *        -# This function performs
     *           \f$ \Theta(nref) \f$ amount of work where \f$ nref \f$ is the number
	 * 			 of available views of \a source.
     *        -# A call to this function may use \f$ \mathcal{O}(1) \f$ bytes
     *           of memory beyond the memory in use at the function call entry.
     *        -# This function may make system calls.
     * \endparblock
	 *
	 */
	template<
		typename TargetStructure,
		typename T, typename Structure, enum Density density, typename View, typename ImfR, typename ImfC, enum Backend backend >
	Matrix< T, TargetStructure, density, view::Original< Matrix< T, Structure, density, View, ImfR, ImfC, backend > >, imf::Strided, imf::Strided, backend >
	get_view( Matrix< T, Structure, density, View, ImfR, ImfC, backend > &source,
			const utils::range& rng_r, const utils::range& rng_c ) {

		return internal::get_view< TargetStructure >(
			source,
			std::move( imf::Strided( rng_r.count(), nrows(source), rng_r.start, rng_r.stride ) ),
			std::move( imf::Strided( rng_c.count(), ncols(source), rng_c.start, rng_c.stride ) )
		);
	}

	/**
     *
	 * @brief Generate an original view where the type is compliant with the source Matrix.
	 * Version where no target structure is specified (in this case the structure of the source type is assumed as target)
	 * with row and column selection.
	 * A structure preserving check as well as non-overlapping checks with existing views of \a source
	 * are guaranteed only when each one of them incurs constant time work.
	 *
	 * @tparam T          The matrix' elements type
	 * @tparam Structure  The structure of the source and target matrix view
	 * @tparam density    The type (i.e., sparse or dense) of storage scheme
	 * @tparam View       The source's View type
	 * @tparam backend    The target backend
	 *
	 * @param source      The source matrix
	 * @param rng_r       A valid range of rows
	 * @param rng_c       A valid range of columns
	 *
	 * @return A new original view over the source structured matrix.
	 *
     * \parblock
     * \par Performance semantics.
     *        -# This function performs
     *           \f$ \Theta(nref) \f$ amount of work where \f$ nref \f$ is the number
	 * 			 of available views of \a source.
     *        -# A call to this function may use \f$ \mathcal{O}(1) \f$ bytes
     *           of memory beyond the memory in use at the function call entry.
     *        -# This function may make system calls.
     * \endparblock
	 *
	 */

	template<
		typename T, typename Structure, enum Density density, typename View, typename ImfR, typename ImfC, enum Backend backend >
	Matrix<
		T,
		Structure,
		density,
		view::Original< Matrix< T, Structure, density, View, ImfR, ImfC, backend > >,
		typename imf::composed_type< imf::Strided, ImfR >::type,
		typename imf::composed_type< imf::Strided, ImfC >::type,
		backend
	>
	get_view( Matrix< T, Structure, density, View, ImfR, ImfC, backend > &source,
			const utils::range &rng_r, const utils::range &rng_c ) {

		return internal::get_view< Structure >(
			source,
			imf::Strided( rng_r.count(), nrows(source), rng_r.start, rng_r.stride ),
			imf::Strided( rng_c.count(), ncols(source), rng_c.start, rng_c.stride ) );
	}

	/**
	 *
	 * @brief Generate a vector view on a row of the source matrix.
	 *
	 * @tparam T          The matrix' elements type
	 * @tparam Structure  The structure of the source and target matrix view
	 * @tparam density    The type (i.e., sparse or dense) of storage scheme
	 * @tparam View       The source's View type
	 * @tparam backend    The target backend
	 *
	 * @param source      The source matrix
	 * @param sel_r       A valid row index
	 * @param rng_c       A valid range of columns
	 *
	 * @return A new original view over the source structured matrix.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -# This function performs
	 *           \f$ \Theta(nref) \f$ amount of work where \f$ nref \f$ is the number
	 *           of available views of \a source.
	 *        -# A call to this function may use \f$ \mathcal{O}(1) \f$ bytes
	 *           of memory beyond the memory in use at the function call entry.
	 *        -# This function may make system calls.
	 * \endparblock
	 *
	 */
	template<
		typename T, typename Structure, enum Density density, typename View, typename ImfR, typename ImfC, enum Backend backend >
	Vector< T, structures::General, density, view::Original< Matrix< T, Structure, density, View, ImfR, ImfC, backend > >, imf::Id, backend >
	get_view( Matrix< T, Structure, density, View, ImfR, ImfC, backend > &source,
		const size_t &sel_r, const utils::range &rng_c ) {

		// auto imf_c = std::make_shared< imf::Strided >( rng_c.count(), ncols(source), rng_c.start, rng_c.stride );

		// return internal::get_view<Structure, T, Structure, density, View, backend >( source, sel_r, imf_c );
		return Vector< T, structures::General, density, View, imf::Id, backend >();
	}

	/**
	 *
	 * @brief Generate a vector view on a column of the source matrix.
	 *
	 * @tparam T          The matrix' elements type
	 * @tparam Structure  The structure of the source and target matrix view
	 * @tparam density    The type (i.e., sparse or dense) of storage scheme
	 * @tparam View       The source's View type
	 * @tparam backend    The target backend
	 *
	 * @param source      The source matrix
	 * @param rng_r       A valid range of rows
	 * @param sel_c       A valid column index
	 *
	 * @return A new original view over the source structured matrix.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -# This function performs
	 *           \f$ \Theta(nref) \f$ amount of work where \f$ nref \f$ is the number
	 *           of available views of \a source.
	 *        -# A call to this function may use \f$ \mathcal{O}(1) \f$ bytes
	 *           of memory beyond the memory in use at the function call entry.
	 *        -# This function may make system calls.
	 * \endparblock
	 *
	 */
	template<
		typename T, typename Structure, enum Density density, typename View, typename ImfR, typename ImfC, enum Backend backend >
	Vector< T, structures::General, density, view::Original< Matrix< T, Structure, density, View, ImfR, ImfC, backend > >, imf::Id, backend >
	get_view( Matrix< T, Structure, density, View, ImfR, ImfC, backend > &source,
		const utils::range &rng_r, const size_t &sel_c ) {

		// auto imf_r = std::make_shared< imf::Strided >( rng_r.count(), nrows(source), rng_r.start, rng_r.stride );

		// return internal::get_view<Structure, T, Structure, density, View, backend >( source, imf_r, sel_c );
		return Vector< T, structures::General, density, View, imf::Id, backend >();
	}

	/**
	 *
		* @brief Generate an original view where the type is compliant with the source Matrix.
		* Version where a selection of rows and columns expressed as vectors of positions
		* form a new view with specified target structure.
		*
		* @tparam TargetStructure The target structure of the new view. It should verify
		*                         <code> alp::is_in<Structure, TargetStructure::inferred_structures> </code>.
		* @tparam T               The matrix' elements type
		* @tparam Structure       The structure of the source and target matrix view
		* @tparam density         The type (i.e., sparse or dense) of storage scheme
		* @tparam View            The source's View type
		* @tparam backend         The target backend
		*
		* @param source           The source structured matrix
		* @param sel_r            A valid permutation vector of row indeces
		* @param sel_c            A valid permutation vector of column indeces
		*
		* @return A new original view over the source structured matrix.
		*
		*/
	template<
		typename TargetStructure,
		typename IndexType, typename IndexStructure, typename IndexView, typename IndexImf,
		typename T, typename Structure, enum Density density, typename View, typename ImfR, typename ImfC,
		enum Backend backend
	>
	alp::Matrix<
		T,
		TargetStructure,
		density,
		view::Original< alp::Matrix< T, Structure, density, View, ImfR, ImfC, backend > >,
		typename imf::composed_type< imf::Select, ImfR >::type,
		typename imf::composed_type< imf::Select, ImfC >::type,
		backend
	>
	get_view( alp::Matrix< T, Structure, density, View, ImfR, ImfC, backend > &source,
			const Vector< IndexType, IndexStructure, density, IndexView, IndexImf, backend > & sel_r,
			const Vector< IndexType, IndexStructure, density, IndexView, IndexImf, backend > & sel_c ) {

		imf::Select imf_r( nrows(source), sel_r );
		imf::Select imf_c( ncols(source), sel_c );

		return internal::get_view<TargetStructure, T, Structure, density, View, ImfR, ImfC, backend>( source, imf_r, imf_c );
	}

	namespace structures {
		namespace constant {

			/** Returns a constant reference to an Identity matrix of the provided size */
			template< typename T >
			const Matrix< T, structures::Identity, Density::Dense, view::Functor< std::function< const T( const size_t, const size_t ) > >,
				imf::Id, imf::Id, reference >
			I( const size_t n ) {

				return Matrix< T, structures::Identity, Density::Dense, view::Functor< std::function< const T( const size_t, const size_t ) > >,
					imf::Id, imf::Id, reference >(
						[]( const size_t i, const size_t j ) {
							return ( i == j ) ? 1 : 0;
						},
						n,
						n
					);
			}

			/** Returns a constant reference to a Zero matrix of the provided size */
			template< typename T >
			const Matrix< T, structures::Zero, Density::Dense, view::Functor< std::function< const T( const size_t, const size_t ) > >,
				imf::Id, imf::Id, reference >
			Zero( const size_t rows, const size_t cols ) {
				return Matrix< T, structures::Zero, Density::Dense, view::Functor< std::function< const T( const size_t, const size_t ) > >,
					imf::Id, imf::Id, reference > (
						[]( const size_t, const size_t ) {
							return 0;
						},
						rows,
						cols
					);
			}

			namespace internal {
				/** Returns a constant reference to a matrix representing Givens rotation
				 * of the provided size n and parameters i, j, s and c, where
				 * s = sin( theta ) and c = cos( theta )
				 */
				template< typename T >
				const Matrix< T, structures::Square, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > &
				Givens( const size_t n, const size_t i, const size_t j, const T s, const T c ) {
					using return_type = const Matrix< T, structures::Square, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference >;
					return_type * ret = new return_type( n );
					// TODO: initialize matrix values according to the provided parameters
					return * ret;
				}
			} // namespace internal
		} // namespace constant
	} // namespace structures

	/** Definitions of previously declared global methods that operate on ALP Matrix */
	namespace internal {

		template< typename MatrixType,
			std::enable_if_t< is_matrix< MatrixType>::value > * = nullptr
		>
		bool getInitialized( const MatrixType &A ) noexcept {
			return static_cast< const MatrixBase< typename MatrixType::base_type > & >( A ).template getInitialized();
		}

		template< typename MatrixType,
			std::enable_if_t< is_matrix< MatrixType>::value > * = nullptr
		>
		void setInitialized( MatrixType &A, const bool initialized ) noexcept {
			return static_cast< MatrixBase< typename MatrixType::base_type > & >( A ).template setInitialized( initialized );
		}

		template< typename DerivedMatrix >
		std::pair< size_t, size_t > dims( const MatrixBase< DerivedMatrix > & A ) noexcept {
			return A.dims();
		}

		/** Access the matrix element.
		 *
		 * @tparam    MatrixType ALP Matrix type
		 *
		 * @param[in] A             matrix to be accessed
		 * @param[in] storageIndex  index in the physical iteration space
		 *
		 * @return For container matrices, returns a constant reference to the
		 *         element at the given physical position of matrix A.
		 *         For functor view matrices, returns a value corresponding to
		 *         the given physical position of matrix A.
		 *
		 * \note   This method may be used to access only elements local to the processor.
		 */
		template< typename MatrixType >
		const typename MatrixType::access_type access( const MatrixType &A, const typename MatrixType::storage_index_type &storageIndex ) {
			return static_cast< const MatrixBase< typename MatrixType::base_type > & >( A ).template access< typename MatrixType::access_type, typename MatrixType::storage_index_type >( storageIndex );
		}

		/** Non-constant variant. **/
		template< typename MatrixType >
		typename MatrixType::access_type access( MatrixType &A, const typename MatrixType::storage_index_type &storageIndex ) {
			return static_cast< MatrixBase< typename MatrixType::base_type > & >( A ).template access< typename MatrixType::access_type, typename MatrixType::storage_index_type >( storageIndex );
		}

		/** Return a storage index in the physical layout.
		 *
		 * @tparam    MatrixType ALP Matrix type
		 *
		 * @param[in] A  matrix to be accessed
		 * @param[in] i  row-index in the logical layout
		 * @param[in] j  column-index in the logical layout
		 * @param[in] s  process ID
		 * @param[in] P  total number of processors
		 *
		 * @return For container matrices, returns a constant reference to the
		 *         element at the given physical position of matrix A.
		 *         For functor view matrices, returns a value corresponding to
		 *         the given physical position of matrix A.
		 *
		 */
		template< typename MatrixType >
		typename MatrixType::storage_index_type getStorageIndex( const MatrixType &A, const size_t i, const size_t j, const size_t s, const size_t P ) {
			return static_cast< const MatrixBase< typename MatrixType::base_type > & >( A ).template getStorageIndex< typename MatrixType::storage_index_type >( i, j, s, P );
		}

		/** Return a pair of coordinates in logical layout.
		 *
		 * @tparam    MatrixType ALP Matrix type
		 *
		 * @param[in] A             matrix to be accessed
		 * @param[in] storageIndex  storage index in the physical layout.
		 * @param[in] s             process ID
		 * @param[in] P             total number of processors
		 *
		 * @return Returns a pair of coordinates in logical iteration space
		 *         that correspond to the provided storage index in the
		 *         physical iteration space.
		 *
		 */
		template< typename MatrixType >
		std::pair< size_t, size_t > getCoords( const MatrixType &A, const size_t storageIndex, const size_t s, const size_t P );

	} // namespace internal

	template< typename D, typename Structure, typename View, typename ImfR, typename ImfC >
	size_t nrows( const Matrix< D, Structure, Density::Dense, View, ImfR, ImfC, reference > & A ) noexcept {
		return dims( A ).first;
	}

	template< typename D, typename Structure, typename View, typename ImfR, typename ImfC >
	size_t ncols( const Matrix< D, Structure, Density::Dense, View, ImfR, ImfC, reference > & A ) noexcept {
		return dims( A ).second;
	}

	template< typename D, typename Structure, typename View, typename ImfR, typename ImfC >
	std::pair< size_t, size_t > dims( const Matrix< D, Structure, Density::Dense, View, ImfR, ImfC, reference > & A ) noexcept {
		return internal::dims( static_cast< const internal::MatrixBase<
			typename Matrix< D, Structure, Density::Dense, View, ImfR, ImfC, reference >::base_type > & > ( A ) );
	}

} // namespace alp

#endif // end ``_H_ALP_REFERENCE_MATRIX''
