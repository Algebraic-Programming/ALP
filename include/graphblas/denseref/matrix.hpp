
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

#ifndef _H_GRB_DENSEREF_MATRIX
#define _H_GRB_DENSEREF_MATRIX

#include <stdexcept>
#include <memory>
#include <vector>
#include <algorithm>

#include <graphblas/backends.hpp>
#include <graphblas/base/matrix.hpp>
#include <graphblas/config.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/type_traits.hpp>
#include <graphblas/utils.hpp>
#include <graphblas/utils/autodeleter.hpp>
//#include <graphblas/utils/pattern.hpp> //for help with dealing with pattern matrix input
#include <graphblas/structures.hpp>
#include <graphblas/storage.hpp>
#include <graphblas/views.hpp>
#include <graphblas/imf.hpp>

namespace grb {

	template< typename D >
	size_t nrows( const Matrix< D, reference_dense > & ) noexcept;

	template< typename D >
	size_t ncols( const Matrix< D, reference_dense > & ) noexcept;

	namespace internal {

		template< typename D >
		D * getRaw( Matrix< D, reference_dense > & ) noexcept;

		template< typename D >
		const D * getRaw( const Matrix< D, reference_dense > & ) noexcept;

		template< typename D >
		const bool & getInitialized( const grb::Matrix< D, reference_dense > & A ) noexcept {
			return A.initialized;
		}

		template< typename D >
		void setInitialized( grb::Matrix< D, reference_dense > & A, bool initialized ) noexcept {
			A.initialized = initialized;
		}
	} // namespace internal

	/**
	 * ALP/Dense matrix container.
	 *
	 * A matrix is stored in full format. 
	 * \a Matrix may be used by \a StructuredMatrix as a raw container.
	 *
	 * @tparam D  The element type.
	 */
	template< typename D >
	class Matrix< D, reference_dense > {

	private:
		/* *********************
		        BLAS2 friends
		   ********************* */

		template< typename DataType >
		friend size_t nrows( const Matrix< DataType, reference_dense > & m ) noexcept;

		template< typename DataType >
		friend size_t ncols( const Matrix< DataType, reference_dense > & m ) noexcept;

		/* *********************
		     `Getter' friends
		   ********************* */

		friend D * internal::getRaw< D >( Matrix< D, reference_dense > & ) noexcept;

		friend const D * internal::getRaw< D >( const Matrix< D, reference_dense > & ) noexcept;

		/* ********************
		        IO friends
		   ******************** */

		template< typename InputType, typename fwd_iterator >
		friend RC buildMatrix( Matrix< InputType, reference_dense > &, fwd_iterator, const fwd_iterator );

		template< typename DataType >
		friend const bool & internal::getInitialized( const grb::Matrix< DataType, reference_dense > & ) noexcept;

		template< typename DataType >
		friend void internal::setInitialized( grb::Matrix< DataType, reference_dense > & , bool ) noexcept;

		typedef Matrix< D, reference_dense > self_type;

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
		 * @see grb::buildMatrixUnique
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
		 */
		Matrix( const size_t rows, const size_t columns, const size_t cap = 0 ): m( rows ), n( columns ), cap( std::max( m*n, cap ) ), initialized( false ) {
			// TODO Implement allocation properly
			if( m > 0 && n > 0) {
				data = new (std::nothrow) D[ m * n ];
			} else {
				data = nullptr;
			}

			if ( m > 0 && n > 0 && data == nullptr ) {
				throw std::runtime_error( "Could not allocate memory during grb::Matrix<reference_dense> construction." );
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
		Matrix( const Matrix< D, reference_dense > & other ) : Matrix( other.m, other.n ) {
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

	/**
	 * @brief A reference_dense Matrix is an ALP object. 
	 */
	template< typename T >
	struct is_container< Matrix< T, reference_dense > > {
		static const constexpr bool value = true;
	};

	template< typename T >
	T * internal::getRaw( Matrix< T, reference_dense > & m ) noexcept {
		return m.data;
	}

	template< typename T >
	const T * internal::getRaw( const Matrix< T, reference_dense > & m ) noexcept {
		return m.data;
	}

	template< typename T >
	size_t nrows( const Matrix< T, reference_dense > & m ) noexcept {
		return m.m;
	}

	template< typename T >
	size_t ncols( const Matrix< T, reference_dense > & m ) noexcept {
		return m.n;
	}



	// StructuredMatrix-related implementation


	namespace internal {
		/** Forward declaration */
		template< typename T >
		class MatrixContainer;

		/** Container reference getters used by friend functions of specialized StructuredMatrix */
		template< typename T >
		const Matrix< T, reference_dense > & getContainer( const MatrixContainer< T > & A );

		template< typename T >
		Matrix< T, reference_dense > & getContainer( MatrixContainer< T > & A );

		/** Container reference getters. Defer the call to base class friend function */
		template< typename T, typename Structure, typename Storage, typename View >
		const Matrix< T, reference_dense > & getContainer( const StructuredMatrix< T, Structure, Storage, View, reference_dense > & A ) {
			return getContainer( A );
		}

		template< typename T, typename Structure, typename Storage, typename View >
		Matrix< T, reference_dense > & getContainer( StructuredMatrix< T, Structure, Storage, View, reference_dense > & A ) {
			return getContainer( A );
		}

		/** Forward declaration */
		class MatrixBase;

		size_t nrows( const MatrixBase & A ) noexcept;

		size_t ncols( const MatrixBase & A ) noexcept;

		std::pair< size_t, size_t > dims( const MatrixBase & A ) noexcept;

		bool getInitialized( MatrixBase & ) noexcept;

		void getInitialized( MatrixBase &, bool ) noexcept;

		template< typename T, typename Structure, typename Storage, typename View >
		bool getInitialized( StructuredMatrix< T, Structure, Storage, View, reference_dense > & A ) noexcept {
			return getInitialized( A );
		}

		template< typename T, typename Structure, typename Storage, typename View >
		void setInitialized( StructuredMatrix< T, Structure, Storage, View, reference_dense > & A, bool initialized ) noexcept {
			setInitialized( A, initialized );
		}
	} // namespace internal

	template< typename D, typename Structure, typename View >
	size_t nrows( const StructuredMatrix< D, Structure, storage::Dense, View, reference_dense > & A ) noexcept {
		return internal::nrows( A );
	}

	template< typename D, typename Structure, typename View >
	size_t ncols( const StructuredMatrix< D, Structure, storage::Dense, View, reference_dense > & A ) noexcept {
		return internal::ncols( A );
	}

	template< typename D, typename Structure, typename View >
	std::pair< size_t, size_t > dims( const StructuredMatrix< D, Structure, storage::Dense, View, reference_dense > & A ) noexcept {
		return internal::dims( A );
	}

	namespace internal {
		/**
		 * Base StructuredMatrix class containing attributes common to all StructuredMatrix specialization
		 * \internal Maybe this class can be inherited by Container and Reference classes below
		 */

		class MatrixBase {

		protected:
			friend size_t nrows( const MatrixBase & A ) noexcept {
				return A._dims().first;
			}

			friend size_t ncols( const MatrixBase & A ) noexcept {
				return A._dims().second;
			}

			friend std::pair< size_t, size_t > dims( const MatrixBase & A ) noexcept {
				return A._dims();
			}

			/** 
			 * All matrix views use a pair of index mapping functions to 
			 * capture the correspondence between their logical layout and the one 
			 * of their underlying container. This may be another view leading to a composition
			 * of IMFs between the top matrix view and the physical container.
			 */
			std::shared_ptr<imf::IMF> imf_l, imf_r;

			/**
			 * @brief determines the size of the structured matrix via the domain of 
			 * the index mapping functions.
			 * 
			 * @return A pair of dimensions.
			 */
			virtual std::pair< size_t, size_t > _dims() const {
				return std::make_pair( imf_l->n, imf_r->n );
			}

			/**
			 * @brief Construct a new structured matrix Base object assigning identity
			 * mapping functions both to the row and column dimensions.
			 * 
			 * @param rows The number of rows of the matrix.
			 * @param cols The number of columns of the matrix.
			 */
			MatrixBase( size_t rows, size_t cols ) :
				imf_l( std::make_shared< imf::Id >( rows ) ),
				imf_r( std::make_shared< imf::Id >( cols ) ) {}

			MatrixBase( std::shared_ptr< imf::IMF > imf_l, std::shared_ptr< imf::IMF > imf_r ) :
				imf_l( imf_l ),
				imf_r( imf_r ) {}

		};

		/**
		 * Base class with container-related attributes, used in container-type StructuredMatrix specializations
		 */
		template< typename T >
		class MatrixContainer : public MatrixBase {
		protected:
			friend const Matrix< T, reference_dense > & getContainer( const MatrixContainer< T > & A ) {
				return *( A._container );
			}

			friend Matrix< T, reference_dense > & getContainer( MatrixContainer< T > & A ) {
				return *( A._container );
			}

			friend bool getInitialized( MatrixContainer & A ) noexcept {
				return getInitialized( getCointainer( A ) );
			}

			friend void setInitialized( MatrixContainer & A, bool initialized ) noexcept {
				setInitialized( getContainer( A, initialized ));
			}

			/** A container-type view is characterized by its association with a physical container */
			Matrix< T, reference_dense > * _container;

			/**
			 * The container's storage scheme. \a storage_scheme is not exposed to the user as an option
			 * but can defined by ALP at different points in the execution depending on the \a backend choice.
			 * In particular, if the structured matrix is not a temporary matrix than it is fixed at construction
			 * time when the allocation takes place.
			 * If the structured matrix is a temporary one than a storage storage scheme choice may or may not be 
			 * made depending on whether a decision about instantiating the matrix is made by the framework.
			 * 
			 * The specific storage scheme choice depends on the chosen backend and the structure of the matrix.
			 */
			storage::Dense storage_scheme;

			/**
			 * @brief Construct a new structured matrix container object.
			 * 
			 * TODO: Add the storage scheme a parameter to the constructor 
			 * so that allocation can be made accordingly, generalizing the full case.
			 */
			MatrixContainer( size_t rows, size_t cols ) :
				MatrixBase( rows, cols ),
				_container( new Matrix< T, reference_dense >(rows, cols) ),
				storage_scheme( storage::full ) {}

		};

		/**
		 * Base class with reference-related attributes, used in Views on container-type StructuredMatrix specializations
		 * 
		 */
		template< typename TargetType >
		class MatrixReference : public MatrixBase {
		protected:
			/** A reference-type view is characterized by an indirect association with a
			 * physical layout via either another \a MatrixReference or a \a
			 * MatrixContainer. Thus a \a MatrixReference never allocates
			 * memory but only establishes a logical view on top of it.
			 */
			TargetType * ref;

			MatrixReference() : MatrixBase( 0, 0 ), ref( nullptr ) {}
			MatrixReference( TargetType & struct_mat ) : MatrixBase( nrows( struct_mat ), ncols( struct_mat ) ), ref( & struct_mat ) {}
			MatrixReference( TargetType & struct_mat, std::shared_ptr< imf::IMF > imf_l, std::shared_ptr< imf::IMF > imf_r ) :
				MatrixBase( imf_l, imf_r ), ref( & struct_mat ) {}
		};
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
	 * backend. grb::Matrix and grb::Vector may be used as interfaces to such a physical
	 * layout.
	 * To visualize this, you may think of a band matrix. Using a 
	 * \a storage::Dense:full or a \a storage::Dense:band storage schemes would require
	 * the use of a \a grb::Matrix container (see include/graphblas/storage.hpp for
	 * more details about the two storage schemes). However, the interpration of its 
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
	 * \a StructuredMatrix<T, structures::General, storage::Dense, View, reference_dense >
	 * and \a StructuredMatrix<T, structures::General, storage::Dense, view::Indentity<void>, reference_dense > 
	 * as examples of structured matrix types without and with physical container, respectively.
	 *
	 * Finally, a structured matrix can be declared as temporary, in which case the ALP 
	 * framework has the freedom to decide whether a container should be allocated in practice
	 * or not. For example, a JIT backend may optimize away the use of such matrix which 
	 * would make memory allocation for such matrix unnecessary.
	 * 
	 * @tparam T				 The type of the matrix elements. \a T shall not be a GraphBLAS
	 *              			 type.
	 * @tparam Structure	     One of the matrix structures defined in \a grb::structures.
	 * @tparam StorageSchemeType Either \em enum \a storage::Dense or \em enum 
	 * 	                         \a storage::Sparse.
	 * 		   					 \a StructuredMatrix will be allowed to pick storage schemes 
	 *         					 defined within their specified \a StorageSchemeType.
	 * @tparam View  			 One of the matrix views in \a grb::view.
	 * 		   					 All static views except for \a view::Original (via 
	 *         					 \a view::Original<void> cannot instantiate a new container 
	 * 							 and only allow to refer to a previously defined 
	 * 							 \a StructuredMatrix.  
	 *         					 The \a View parameter should not be used directly 
	 * 							 by the user but can be set using specific member types 
	 * 							 appropriately defined by each StructuredMatrix and
	 * 							 accessible via functions such as \a grb::transpose or
	 * 							 \a grb::diagonal. (See examples of StructuredMatrix 
	 *         					 definitions within \a include/graphblas/denseref/matrix.hpp 
	 * 							 and the \a dense_structured_matrix.cpp unit test).
	 *
	 */
	template< typename T, typename Structure, typename StorageSchemeType, typename View >
	class StructuredMatrix<T, Structure, StorageSchemeType, View, reference_dense> { };

	/**
	 * @brief General matrix with physical container. 
	 */
	template< typename T >
	class StructuredMatrix< T, structures::General, storage::Dense, view::Original< void >, reference_dense > :
		public internal::MatrixContainer< T > {

	private:
		/*********************
		    Storage info friends
		******************** */

		using self_type = StructuredMatrix< T, structures::General, storage::Dense, view::Original< void >, reference_dense >;

		// template< typename fwd_iterator >
		// friend RC buildMatrix( StructuredMatrix< T, structures::General, storage::Dense, view::Original< void >, reference_dense > &, const fwd_iterator & start, const fwd_iterator & end );
		template< typename fwd_iterator >
		friend RC buildMatrix( StructuredMatrix< T, structures::General, storage::Dense, view::Original< void >, reference_dense > & A, const fwd_iterator & start, const fwd_iterator & end );

		template< typename fwd_iterator >
		RC buildMatrixUnique( const fwd_iterator & start, const fwd_iterator & end ) {
			std::cout << "Building StructuredMatrix<>; calling buildMatrix( Matrix<> )\n";
			return buildMatrix( *(this->_container), start, end );
		}

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::General;

		// A general Structure knows how to define a reference to itself (which is an original reference view)
		// as well as other static views.
		using original_t = StructuredMatrix< T, structures::General, storage::Dense, view::Original< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::General, storage::Dense, view::Transpose< self_type >, reference_dense >;

		StructuredMatrix( const size_t rows, const size_t cols ) :
			internal::MatrixContainer< T >( rows, cols ) {
		}

	}; // StructuredMatrix General, container

	/**
	 * View of a general Matrix.
	 */
	template< typename T, typename View >
	class StructuredMatrix< T, structures::General, storage::Dense, View, reference_dense > :
		public internal::MatrixReference< typename View::applied_to > {

	private:
		using self_type = StructuredMatrix< T, structures::General, storage::Dense, View, reference_dense >;
		using target_type = typename View::applied_to;

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::General;

		using original_t = StructuredMatrix< T, structures::General, storage::Dense, view::Original< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::General, storage::Dense, view::Transpose< self_type >, reference_dense >;

		StructuredMatrix( ) : internal::MatrixBase( 0, 0 ) {}

		StructuredMatrix( target_type & struct_mat ) : internal::MatrixReference< target_type >( struct_mat ) {}

		StructuredMatrix( target_type & struct_mat, std::shared_ptr< imf::IMF > imf_l, std::shared_ptr< imf::IMF > imf_r ) :
			internal::MatrixReference< target_type >( struct_mat, imf_l, imf_r ) {}

	}; // StructuredMatrix General reference

	template< typename T, typename Structure >
	class StructuredMatrix< T, Structure, storage::Dense, view::Original< void >, reference_dense > :
		public internal::MatrixContainer< T > {

	private:
		/*********************
		    Storage info friends
		******************** */

		using self_type = StructuredMatrix< T, Structure, storage::Dense, view::Original< void >, reference_dense >;

		template< typename fwd_iterator >
		friend RC buildMatrix( StructuredMatrix< T, Structure, storage::Dense, view::Original< void >, reference_dense > &, const fwd_iterator &, const fwd_iterator ) noexcept;

	public:
		using value_type = T;
		using structure = Structure;

		/** The type of an identify view over the present type */
		using original_t = StructuredMatrix< T, Structure, storage::Dense, view::Original< self_type >, reference_dense >;

		StructuredMatrix( const size_t rows, const size_t cols ) :
			internal::MatrixContainer< T >( rows, cols ) {}
	}; // class StructuredMatrix

	template< typename T >
	class StructuredMatrix< T, structures::Square, storage::Dense, view::Original< void >, reference_dense > :
		public internal::MatrixContainer< T > {

	private:
		using self_type = StructuredMatrix< T, structures::Square, storage::Dense, view::Original< void >, reference_dense >;

		template< typename InputType, typename Structure, typename Storage, typename View, typename fwd_iterator >
		friend RC buildMatrix( StructuredMatrix< InputType, Structure, Storage, View, reference_dense > &, const fwd_iterator & start, const fwd_iterator & end ) noexcept;

		template< typename fwd_iterator >
		RC buildMatrixUnique( const fwd_iterator & start, const fwd_iterator & end ) {
			std::cout << "Building StructuredMatrix<>; calling buildMatrix( Matrix<> )\n";
			return buildMatrix( *(this->_container), start, end );
		}

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::Square;

		using original_t = StructuredMatrix< T, structures::Square, storage::Dense, view::Original< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::Square, storage::Dense, view::Transpose< self_type >, reference_dense >;

		StructuredMatrix( const size_t rows ) :
			internal::MatrixContainer< T >( rows, rows ) {}

	}; // StructuredMatrix Square, container

	template< typename T, typename View >
	class StructuredMatrix< T, structures::Square, storage::Dense, View, reference_dense > :
		public internal::MatrixReference< typename View::applied_to > {

	private:
		using self_type = StructuredMatrix< T, structures::Square, storage::Dense, View, reference_dense >;
		using target_type = typename View::applied_to;

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::Square;

		using original_t = StructuredMatrix< T, structures::Square, storage::Dense, view::Original< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::Square, storage::Dense, view::Transpose< self_type >, reference_dense >;

		// ref to empty matrix
		StructuredMatrix( ) : internal::MatrixReference< target_type >() {}

		StructuredMatrix( target_type & struct_mat ) : internal::MatrixReference< target_type >( struct_mat ) {
			if( nrows( struct_mat ) != ncols( struct_mat ) ) {
				throw std::length_error( "Square StructuredMatrix reference to non-square target." );
			}
		}

	}; // StructuredMatrix Square reference

	// StructuredMatrix UpperTriangular, container
	template< typename T >
	class StructuredMatrix< T, structures::UpperTriangular, storage::Dense, view::Original< void >, reference_dense > :
		public internal::MatrixContainer< T > {

	private:
		/*********************
		    Storage info friends
		******************** */

		using self_type = StructuredMatrix< T, structures::UpperTriangular, storage::Dense, view::Original< void >, reference_dense >;

		template< typename InputType, typename Structure, typename Storage, typename View, typename fwd_iterator >
		friend RC buildMatrix( StructuredMatrix< InputType, Structure, Storage, View, reference_dense > &, const fwd_iterator & start, const fwd_iterator & end ) noexcept;

		template< typename fwd_iterator >
		RC buildMatrixUnique( const fwd_iterator & start, const fwd_iterator & end ) {
			std::cout << "Building StructuredMatrix<>; calling buildMatrix( Matrix<> )\n";
			return buildMatrix( *(this->_container), start, end );
		}

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::UpperTriangular;

		// A general Structure knows how to define a reference to itself (which is an original reference view)
		// as well as other static views.
		using original_t = StructuredMatrix< T, structures::UpperTriangular, storage::Dense, view::Original< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::LowerTriangular, storage::Dense, view::Transpose< self_type >, reference_dense >;

		StructuredMatrix( const size_t rows, const size_t cols ) :
			internal::MatrixContainer< T >( rows, cols ) {}

	}; // StructuredMatrix UpperTriangular, container

	// StructuredMatrix UpperTriangular, reference
	template< typename T, typename View >
	class StructuredMatrix< T, structures::UpperTriangular, storage::Dense, View, reference_dense > :
		public internal::MatrixReference< typename View::applied_to > {

	private:
		using self_type = StructuredMatrix< T, structures::UpperTriangular, storage::Dense, View, reference_dense >;
		using target_type = typename View::applied_to;

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::UpperTriangular;

		using original_t = StructuredMatrix< T, structures::UpperTriangular, storage::Dense, view::Original< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::UpperTriangular, storage::Dense, view::Transpose< self_type >, reference_dense >;

		// ref to empty matrix
		StructuredMatrix( ) : internal::MatrixReference< target_type >() {}

		StructuredMatrix( target_type & struct_mat ) : internal::MatrixReference< target_type >( struct_mat ) {
			// No matter the view it has to be a square matrix
		}

		StructuredMatrix( target_type & struct_mat, std::shared_ptr< imf::IMF > imf_l, std::shared_ptr< imf::IMF > imf_r ) :
			internal::MatrixReference< target_type >( struct_mat, imf_l, imf_r ) {}

	}; //  StructuredMatrix UpperTriangular, reference

	// StructuredMatrix Identity, container
	// Should Identity be a MatrixContainer?
	template< typename T >
	class StructuredMatrix< T, structures::Identity, storage::Dense, view::Original< void >, reference_dense > :
		public internal::MatrixContainer< T > {

	private:
		/*********************
		    Storage info friends
		******************** */

		using self_type = StructuredMatrix< T, structures::Identity, storage::Dense, view::Original< void >, reference_dense >;

		template< typename InputType, typename Structure, typename Storage, typename View, typename fwd_iterator >
		friend RC buildMatrix( StructuredMatrix< InputType, Structure, Storage, View, reference_dense > &, const fwd_iterator & start, const fwd_iterator & end ) noexcept;

		template< typename fwd_iterator >
		RC buildMatrixUnique( const fwd_iterator & start, const fwd_iterator & end ) {
			std::cout << "Building StructuredMatrix<>; calling buildMatrix( Matrix<> )\n";
			return buildMatrix( *(this->_container), start, end );
		}

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::Identity;

		// A general Structure knows how to define a reference to itself (which is an original reference view)
		// as well as other static views.
		using original_t = StructuredMatrix< T, structures::Identity, storage::Dense, view::Original< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::Identity, storage::Dense, view::Transpose< self_type >, reference_dense >;

		StructuredMatrix( const size_t rows ) :
			internal::MatrixContainer< T >( rows, rows ) {}

	}; // StructuredMatrix Identity, container

	namespace structures {

		/**
		 * @brief Checks if a structured matrix has structure \a Structure.
		 * 
		 * @tparam StructuredMatrixT The structured matrix type to be tested. 
		 * @tparam Structure 		 The structure type which should be implied by \a StructuredMatrixT::structure.
		 */
		template< typename StructuredMatrixT, typename Structure >
		struct is_a {
			/**
			 * \a value is true iff \a Structure is implied by \a StructuredMatrixT::structure.
			 */
			static constexpr bool value = is_in< Structure, typename StructuredMatrixT::structure::inferred_structures >::value;
		};

	} // namespace structures

	/**
     *
	 * @brief Generate an original view where the type is compliant with the source StructuredMatrix.
	 * Version where no target structure is specified. In this case the structure of the source type is assumed.
	 * 
	 * @tparam T 					The matrix' elements type
	 * @tparam Structure 			The structure of the source and target matrix view
	 * @tparam StorageSchemeType 	The type (i.e., sparse or dense) of storage scheme
	 * @tparam View 				The source's View type
	 * @tparam backend 				The target backend
	 * 
	 * @param source 				The source structured matrix
	 * 
	 * @return A new original view over the source structured matrix.
	 * 
	 */
	template< typename T, typename Structure, typename StorageSchemeType, typename View, enum Backend backend >
	StructuredMatrix< T, Structure, StorageSchemeType, view::Original< StructuredMatrix< T, Structure, StorageSchemeType, View, backend > >, backend > 
	get_view( StructuredMatrix< T, Structure, StorageSchemeType, View, backend > &source ) {

		using source_strmat_t = StructuredMatrix< T, Structure, StorageSchemeType, View, backend >;
		using target_strmat_t = StructuredMatrix< T, Structure, StorageSchemeType, view::Original< source_strmat_t >, backend >;

		target_strmat_t target( source );

		return target;
	}

	/**
     *
	 * @brief Generate an original view where the type is compliant with the source StructuredMatrix.
	 * Version where a target structure is specified. It can only generate a valide type if the target 
	 * structure is the same as the source's
	 * or a more specialized one that would preserve its static properties (e.g., symmetric reference
	 * to a square matrix -- any assumption based on symmetry would not break those based on square).
	 * 
	 * @tparam TargetStructure 		The target structure of the new view. It should verify 
	 * 								<code> grb::is_in<Structure, TargetStructure::inferred_structures> </code>.
	 * @tparam T 					The matrix' elements type
	 * @tparam Structure 			The structure of the source and target matrix view
	 * @tparam StorageSchemeType 	The type (i.e., sparse or dense) of storage scheme
	 * @tparam View 				The source's View type
	 * @tparam backend 				The target backend
	 * 
	 * @param source 				The source structured matrix
	 * 
	 * @return A new original view over the source structured matrix.
	 * 
	 */
	template< typename TargetStructure, typename T, typename Structure, typename StorageSchemeType, typename View, enum Backend backend >
	StructuredMatrix< T, TargetStructure, StorageSchemeType, view::Original< StructuredMatrix< T, Structure, StorageSchemeType, View, backend > >, backend > 
	get_view( StructuredMatrix< T, Structure, StorageSchemeType, View, backend > &source ) {

		static_assert( structures::is_in< Structure, typename TargetStructure::inferred_structures >::value,
			"Can only create a view when the target structure is compatible with the source." );

		using source_strmat_t = StructuredMatrix< T, Structure, StorageSchemeType, View, backend >;
		using target_strmat_t = StructuredMatrix< T, TargetStructure, StorageSchemeType, view::Original< source_strmat_t >, backend >;

		target_strmat_t target( source );

		return target;
	}

	/**
	 * @brief Construct a new transpose view.
	 * 
	 * @tparam StructuredMatrixT The type of the source structured matrix.
	 *
	 * @param smat 				 The source structure matrix.
	 * 
	 * @return A transposed view of the source matrix.
	 * 
	 */
	template< typename StructuredMatrixT >
	typename StructuredMatrixT::transpose_t
	transpose( StructuredMatrixT &smat ) {

		typename StructuredMatrixT::transpose_t smat_trans( smat );

		return smat_trans;
	}

	namespace internal {
		/**
		 * Implement a gather through a View over compatible Structure using provided Index Mapping Functions.
		 * The compatibility depends on the TargetStructure, SourceStructure and IMFs, and is calculated during runtime.
		 */

		template< typename TargetStructure, typename T, typename Structure, typename StorageSchemeType, typename View, enum Backend backend >
		StructuredMatrix< T, TargetStructure, StorageSchemeType, view::Original< StructuredMatrix< T, Structure, StorageSchemeType, View, backend > >, backend > 
		get_view( StructuredMatrix< T, Structure, StorageSchemeType, View, backend > &source,
				std::shared_ptr< imf::IMF > imf_r, std::shared_ptr< imf::IMF > imf_c ) {
			
			if( std::dynamic_pointer_cast< imf::Select >( imf_r ) || std::dynamic_pointer_cast< imf::Select >( imf_c ) ) {
				throw std::runtime_error("Cannot gather with imf::Select yet.");
			}
			// No static check as the compatibility depends on IMF, which is a runtime level parameter
			if( ! (TargetStructure::template isInstantiableFrom< Structure >( * imf_r, * imf_c ) ) ) {
				throw std::runtime_error("Cannot gather into specified TargetStructure from provided SourceStructure and Index Mapping Functions.");
			}

			using source_strmat_t = StructuredMatrix< T, Structure, StorageSchemeType, View, backend >;
			using target_strmat_t = StructuredMatrix< T, TargetStructure, StorageSchemeType, view::Original< source_strmat_t >, backend >;

			target_strmat_t target( source, imf_r, imf_c );

			return target;
		}
	} // namespace internal

	/**
     *
	 * @brief Generate an original view where the type is compliant with the source StructuredMatrix.
	 * Version where a range of rows and columns are selected to form a new view with specified target 
	 * structure. It can only generate a valide type if the target 
	 * structure is guaranteed to preserve the static properties of the source's structure.
	 * 
	 * @tparam TargetStructure 		The target structure of the new view. It should verify 
	 * 								<code> grb::is_in<Structure, TargetStructure::inferred_structures> </code>.
	 * @tparam T 					The matrix' elements type
	 * @tparam Structure 			The structure of the source and target matrix view
	 * @tparam StorageSchemeType 	The type (i.e., sparse or dense) of storage scheme
	 * @tparam View 				The source's View type
	 * @tparam backend 				The target backend
	 * 
	 * @param source 				The source structured matrix
	 * @param rng_r 				A valid range of rows 
	 * @param rng_c 				A valid range of columns
	 * 
	 * @return A new original view over the source structured matrix.
	 * 
	 */
	template< typename TargetStructure, typename T, typename Structure, typename StorageSchemeType, typename View, enum Backend backend >
	StructuredMatrix< T, TargetStructure, StorageSchemeType, view::Original< StructuredMatrix< T, Structure, StorageSchemeType, View, backend > >, backend > 
	get_view( StructuredMatrix< T, Structure, StorageSchemeType, View, backend > &source,
			const utils::range& rng_r, const utils::range& rng_c ) {
		
		auto imf_r = std::make_shared< imf::Strided >( rng_r.count(), nrows(source), rng_r.start, rng_r.stride );
		auto imf_c = std::make_shared< imf::Strided >( rng_c.count(), ncols(source), rng_c.start, rng_c.stride );

		return internal::get_view<TargetStructure, T, Structure, StorageSchemeType, View, backend>( source, imf_r, imf_c );
	}

	namespace internal {
		/**
		 *	
		 * @brief Generate an original view where the type is compliant with the source StructuredMatrix.
		 * Version where a selection of rows and columns expressed as vectors of positions 
		 * form a new view with specified target structure.
		 * 
		 * \warning WIP interface. Checking the structural correctness is a costly runtime operation for a 
		 * 			general vector selection. 
		 * 
		 * @tparam TargetStructure 		The target structure of the new view. It should verify 
		 * 								<code> grb::is_in<Structure, TargetStructure::inferred_structures> </code>.
		 * @tparam T 					The matrix' elements type
		 * @tparam Structure 			The structure of the source and target matrix view
		 * @tparam StorageSchemeType 	The type (i.e., sparse or dense) of storage scheme
		 * @tparam View 				The source's View type
		 * @tparam backend 				The target backend
		 * 
		 * @param source 				The source structured matrix
		 * @param sel_r 				A valid vector of row indeces (possibly in any permuted order and with repetition) 
		 * @param sel_c 				A valid vector of column indeces (possibly in any permuted order and with repetition)
		 * 
		 * @return A new original view over the source structured matrix.
		 * 
		 */
		template< typename TargetStructure, typename T, typename Structure, typename StorageSchemeType, typename View, enum Backend backend >
		StructuredMatrix< T, TargetStructure, StorageSchemeType, view::Original< StructuredMatrix< T, Structure, StorageSchemeType, View, backend > >, backend > 
		get_view( StructuredMatrix< T, Structure, StorageSchemeType, View, backend > &source,
				const std::vector< size_t >& sel_r, const std::vector< size_t >& sel_c ) {
			
			auto imf_r = std::make_shared< imf::Select >( nrows(source), sel_r );
			auto imf_c = std::make_shared< imf::Select >( ncols(source), sel_c );

			return internal::get_view<TargetStructure, T, Structure, StorageSchemeType, View, backend>( source, imf_r, imf_c );
		}
	} //namespace internal

	/** Returns a constant reference to an Identity matrix of the provided size */
	template< typename T >
	const StructuredMatrix< T, structures::Identity, storage::Dense, view::Original< void >, reference_dense > &
	I( const size_t n ) {
		using return_type = StructuredMatrix< T, structures::Identity, storage::Dense, view::Original< void >, reference_dense >;
		return_type * ret = new return_type( n );
		return * ret;
	}

	/** Returns a constant reference to a Zero matrix of the provided size */
	template< typename T >
	const StructuredMatrix< T, structures::Zero, storage::Dense, view::Original< void >, reference_dense > &
	Zero( const size_t rows, const size_t cols ) {
		using return_type = StructuredMatrix< T, structures::Zero, storage::Dense, view::Original< void >, reference_dense >;
		return_type * ret = new return_type( rows, cols );
		return * ret;
	}

	/** Returns a constant reference to a matrix representing Givens rotation
	 * of the provided size n and parameters i, j, s and c, where
	 * s = sin( theta ) and c = cos( theta )
	 */
	template< typename T >
	const StructuredMatrix< T, structures::Square, storage::Dense, view::Original< void >, reference_dense > &
	Givens( const size_t n, const size_t i, const size_t j, const T s, const T c ) {
		using return_type = const StructuredMatrix< T, structures::Square, storage::Dense, view::Original< void >, reference_dense >;
		return_type * ret = new return_type( n );
		// TODO: initialize matrix values according to the provided parameters
		return * ret;
	}

} // namespace grb

#endif // end ``_H_GRB_DENSEREF_MATRIX''
