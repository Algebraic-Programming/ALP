
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
		/** Forward declaration */
		template< typename T, typename AmfType, bool requires_allocation >
		class StorageBasedMatrix;

		/** Forward declaration */
		template< typename T, typename ImfR, typename ImfC, typename DataLambdaType >
		class FunctorBasedMatrix;

		/** Container reference getters used by friend functions of specialized Matrix */
		template< typename T, typename AmfType, bool requires_allocation >
		const Vector< T, reference > & getContainer( const StorageBasedMatrix< T, AmfType, requires_allocation > & A );

		template< typename T, typename AmfType, bool requires_allocation >
		Vector< T, reference > & getContainer( StorageBasedMatrix< T, AmfType, requires_allocation > & A );

		/** Container reference getters. Defer the call to base class friend function */
		template< typename T, typename Structure, enum Density density, typename View, typename ImfR, typename ImfC >
		const Vector< T, reference > & getContainer( const alp::Matrix< T, Structure, density, View, ImfR, ImfC, reference > & A ) {
			return getContainer( static_cast<
				const StorageBasedMatrix<
					T,
					typename alp::Matrix< T, Structure, density, View, ImfR, ImfC, reference >::amf_type,
					alp::Matrix< T, Structure, density, View, ImfR, ImfC, reference >::requires_allocation
				> &
			>( A ) );
		}

		template< typename T, typename Structure, enum Density density, typename View, typename ImfR, typename ImfC >
		Vector< T, reference > & getContainer( alp::Matrix< T, Structure, density, View, ImfR, ImfC, reference > & A ) {
			return getContainer( static_cast<
				StorageBasedMatrix<
					T,
					typename alp::Matrix< T, Structure, density, View, ImfR, ImfC, reference >::amf_type,
					alp::Matrix< T, Structure, density, View, ImfR, ImfC, reference >::requires_allocation
				> &
			>( A ) );
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

		template<
			typename MatrixType,
			std::enable_if_t< internal::is_storage_based< MatrixType >::value > * = nullptr
		>
		size_t getStorageDimensions( const MatrixType &A ) noexcept;

		template< typename MatrixType,
			std::enable_if_t< is_matrix< MatrixType>::value > * = nullptr
		>
		bool getInitialized( const MatrixType &A ) noexcept;

		template< typename MatrixType,
			std::enable_if_t< is_matrix< MatrixType>::value > * = nullptr
		>
		void setInitialized( MatrixType &, const bool ) noexcept;

	} // namespace internal

	namespace internal {

		/**
		 * Matrix container specialization
		 * Implements both original containers and views on containers.
		 * @tparam requires_allocation True if the class is an original container
		 *                             False if the class is a view of another matrix
		 */
		template< typename T, typename AmfType, bool requires_allocation >
		class StorageBasedMatrix : public MatrixBase< StorageBasedMatrix< T, AmfType, requires_allocation > > {

			template<
				typename MatrixType,
				std::enable_if_t< internal::is_storage_based< MatrixType >::value > *
			>
			friend size_t getStorageDimensions( const MatrixType &A ) noexcept;

			/** Get the reference to the AMF of a storage-based matrix */
			template<
				typename MatrixType,
				std::enable_if< internal::is_storage_based< MatrixType >::value > *
			>
			friend const typename MatrixType::amf_type &getAmf( const MatrixType &A ) noexcept;

			public:

				/** Expose static properties */

				typedef T value_type;
				typedef AmfType amf_type;
				typedef typename AmfType::imf_r_type imf_r_type;
				typedef typename AmfType::imf_c_type imf_c_type;
				/** Type returned by access function */
				typedef T &access_type;
				/** Type of the index used to access the physical storage */
				typedef size_t storage_index_type;

				typedef typename std::conditional<
					requires_allocation,
					Vector< T, reference >,
					Vector< T, reference > &
				>::type container_type;

			protected:
				typedef StorageBasedMatrix< T, AmfType, requires_allocation > self_type;
				friend MatrixBase< self_type >;

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
				AmfType amf;
				/**
				 * @brief determines the size of the matrix via the domain of
				 * the index mapping functions.
				 *
				 * @return A pair of dimensions.
				 */
				std::pair< size_t, size_t > dims() const noexcept {
					return amf.getLogicalDimensions();
				}

				size_t getStorageDimensions() const noexcept {
					return amf.getStorageDimensions();
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

				const AmfType &getAmf() const noexcept {
					return amf;
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
				StorageBasedMatrix( AmfType &&amf ) :
					// enable only if ImfR and ImfC are imf::Id
					container( internal::Vector< T, reference >( amf.getStorageDimensions() ) ),
					amf( std::move( amf ) ) {}

				/** View on another container */
				StorageBasedMatrix( Vector< T, reference > &container, AmfType &&amf ) :
					container( container ),
					amf( std::move( amf ) ) {}

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

	namespace structures {

		template<size_t band, typename T, typename Structure, enum Density density, typename View, typename ImfL, typename ImfR >
		std::ptrdiff_t get_lower_limit(const alp::Matrix< T, Structure, density, View, ImfL, ImfR, reference > &A) {

			const std::ptrdiff_t m = nrows( A );
			constexpr std::ptrdiff_t cl_a = std::tuple_element< band, typename Structure::band_intervals >::type::left;

			const std::ptrdiff_t l_a = ( cl_a < -m + 1 ) ? -m + 1 : cl_a ;

			return l_a;

		}

		template<size_t band, typename T, typename Structure, enum Density density, typename View, typename ImfL, typename ImfR >
		std::ptrdiff_t get_upper_limit(const alp::Matrix< T, Structure, density, View, ImfL, ImfR, reference > &A) {

			const std::ptrdiff_t n = ncols( A );
			constexpr std::ptrdiff_t cu_a = std::tuple_element< band, typename Structure::band_intervals >::type::right;

			const std::ptrdiff_t u_a = ( cu_a > n ) ? n : cu_a ;

			return u_a;

		}

		/**
		 * Specialization for reference backend.
		 * @see alp::structures::calculate_row_coordinate_limits
		 */
		template<
			size_t band_index, typename MatrixType,
			std::enable_if_t<
				is_matrix< MatrixType >::value
			> * = nullptr
		>
		std::pair< size_t, size_t > calculate_row_coordinate_limits( const MatrixType &A ) {

			using Structure = typename MatrixType::structure;

			static_assert(
				band_index < std::tuple_size< typename Structure::band_intervals >::value,
				"Provided band index is out of bounds."
			);

			// cast matrix dimensions to signed integer to allow for comparison with negative numbers
			const std::ptrdiff_t M = static_cast< std::ptrdiff_t >( nrows( A ) );
			const std::ptrdiff_t N = static_cast< std::ptrdiff_t >( ncols( A ) );

			// band limits are negated and inverted due to different orientation
			// of coordinate system of band and matrix dimensions.
			const std::ptrdiff_t l = -structures::get_upper_limit< band_index >( A );
			const std::ptrdiff_t u = N - structures::get_lower_limit< band_index >( A );

			// fit the limits within the matrix dimensions
			const size_t lower_limit = static_cast< size_t >( std::max( std::min( l, M ), static_cast< std::ptrdiff_t >( 0 ) ) );
			const size_t upper_limit = static_cast< size_t >( std::max( std::min( u, M ), static_cast< std::ptrdiff_t >( 0 ) ) );

			assert( lower_limit <= upper_limit );

			return std::make_pair( lower_limit, upper_limit );
		}

		/**
		 * Specialization for reference backend.
		 * @see alp::structures::calculate_column_coordinate_limits
		 */
		template<
			size_t band_index, typename MatrixType,
			std::enable_if_t<
				is_matrix< MatrixType >::value
			> * = nullptr
		>
		std::pair< size_t, size_t > calculate_column_coordinate_limits( const MatrixType &A, const size_t row ) {

			using Structure = typename MatrixType::structure;

			// Declaring this to avoid static casts to std::ptrdiff_t in std::min and std::max calls
			const std::ptrdiff_t signed_zero = 0;

			static_assert(
				band_index < std::tuple_size< typename Structure::band_intervals >::value,
				"Provided band index is out of bounds."
			);

			assert( row < nrows( A ) );

			// cast matrix dimensions to signed integer to allow for comparison with negative numbers
			const std::ptrdiff_t N = static_cast< std::ptrdiff_t >( ncols( A ) );

			constexpr bool is_sym = structures::is_a< Structure, structures::Symmetric >::value;
			// Temporary until adding multiple symmetry directions
			constexpr bool sym_up = is_sym;

			// Band limits
			const std::ptrdiff_t l = structures::get_lower_limit< band_index >( A );
			const std::ptrdiff_t u = structures::get_upper_limit< band_index >( A );

			// Band limits taking into account symmetry
			const std::ptrdiff_t sym_l = is_sym && sym_up ? std::max( signed_zero, l ) : l;
			const std::ptrdiff_t sym_u = is_sym && !sym_up ? std::min( signed_zero, u ) : u;

			// column coordinate lower and upper limits considering the provided row coordinate
			const std::ptrdiff_t sym_l_row = static_cast< std::ptrdiff_t >( row ) + sym_l;
			const std::ptrdiff_t sym_u_row = sym_l_row + ( sym_u - sym_l );

			// fit the limits within the matrix dimensions
			const size_t lower_limit = static_cast< size_t >( std::max( std::min( sym_l_row, N ), signed_zero ) );
			const size_t upper_limit = static_cast< size_t >( std::max( std::min( sym_u_row, N ), signed_zero ) );

			assert( lower_limit <= upper_limit );

			return std::make_pair( lower_limit, upper_limit );
		}

	} // namespace structures

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

} // namespace alp

#endif // end ``_H_ALP_REFERENCE_MATRIX''
