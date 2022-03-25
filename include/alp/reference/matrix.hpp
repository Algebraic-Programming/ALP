
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

#include <stdexcept>
#include <memory>
#include <vector>
#include <algorithm>

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
		void setInitialized( alp::internal::Matrix< D, reference > & A, bool initialized ) noexcept {
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
			friend void internal::setInitialized( alp::internal::Matrix< DataType, reference > & , bool ) noexcept;

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
	} // namespace internal

	/**
	 * @brief A reference Matrix is an ALP object. 
	 */
	template< typename T >
	struct is_container< internal::Matrix< T, reference > > {
		static const constexpr bool value = true;
	};

	// Matrix-related implementation

	namespace internal {
		/** Forward declaration */
		template< typename T >
		class MatrixContainer;

		/** Container reference getters used by friend functions of specialized Matrix */
		template< typename T >
		const Matrix< T, reference > & getContainer( const MatrixContainer< T > & A );

		template< typename T >
		Matrix< T, reference > & getContainer( MatrixContainer< T > & A );

		/** Container reference getters. Defer the call to base class friend function */
		template< typename T, typename Structure, enum Density density, typename View >
		const Matrix< T, reference > & getContainer( const alp::Matrix< T, Structure, density, View, reference > & A ) {
			return getContainer( A );
		}

		template< typename T, typename Structure, enum Density density, typename View >
		Matrix< T, reference > & getContainer( alp::Matrix< T, Structure, density, View, reference > & A ) {
			return getContainer( A );
		}

		/** Forward declaration */
		class MatrixBase;

		size_t nrows( const MatrixBase & A ) noexcept;

		size_t ncols( const MatrixBase & A ) noexcept;

		std::pair< size_t, size_t > dims( const MatrixBase & A ) noexcept;

		bool getInitialized( MatrixBase & ) noexcept;

		void getInitialized( MatrixBase &, bool ) noexcept;
	} // namespace internal

	template< typename T, typename Structure, enum Density density, typename View >
	bool getInitialized( Matrix< T, Structure, density, View, reference > & A ) noexcept {
		return getInitialized( A );
	}

	template< typename T, typename Structure, enum Density density, typename View >
	void setInitialized( Matrix< T, Structure, density, View, reference > & A, bool initialized ) noexcept {
		setInitialized( A, initialized );
	}

	template< typename D, typename Structure, typename View >
	size_t nrows( const Matrix< D, Structure, Density::Dense, View, reference > & A ) noexcept {
		return internal::nrows( A );
	}

	template< typename D, typename Structure, typename View >
	size_t ncols( const Matrix< D, Structure, Density::Dense, View, reference > & A ) noexcept {
		return internal::ncols( A );
	}

	template< typename D, typename Structure, typename View >
	std::pair< size_t, size_t > dims( const Matrix< D, Structure, Density::Dense, View, reference > & A ) noexcept {
		return internal::dims( A );
	}

	namespace internal {
		/**
		 * Base Matrix class containing attributes common to all Matrix specialization
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

		}; // class MatrixBase

		/**
		 * Base class with container-related attributes, used in container-type Matrix specializations
		 */
		template< typename T >
		class MatrixContainer : public MatrixBase {
		protected:
			friend const Matrix< T, reference > & getContainer( const MatrixContainer< T > & A ) {
				return *( A._container );
			}

			friend Matrix< T, reference > & getContainer( MatrixContainer< T > & A ) {
				return *( A._container );
			}

			friend bool getInitialized( MatrixContainer & A ) noexcept {
				return getInitialized( getCointainer( A ) );
			}

			friend void setInitialized( MatrixContainer & A, bool initialized ) noexcept {
				setInitialized( getContainer( A, initialized ));
			}

			/** A container-type view is characterized by its association with a physical container */
			Matrix< T, reference > * _container;

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
			// Storage storage_scheme;

			/**
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
			MatrixContainer( size_t rows, size_t cols, size_t cap = 0 ) :
				MatrixBase( rows, cols ),
				_container( new Matrix< T, reference >( rows, cols, cap ) ) {}

		}; // class MatrixContainer

		/**
		 * Base class with reference-related attributes, used in Views on container-type Matrix specializations
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
			TargetType & ref;

			MatrixReference( TargetType & struct_mat ) : MatrixBase( nrows( struct_mat ), ncols( struct_mat ) ), ref( struct_mat ) {}
			MatrixReference( TargetType & struct_mat, std::shared_ptr< imf::IMF > imf_l, std::shared_ptr< imf::IMF > imf_r ) :
			MatrixBase( imf_l, imf_r ), ref( struct_mat ) {}
		}; // class MatrixReference
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
	template< typename T, typename Structure, enum Density density, typename View >
	class Matrix<T, Structure, density, View, reference> { };

	/**
	 * @brief General matrix with physical container. 
	 */
	template< typename T >
	class Matrix< T, structures::General, Density::Dense, view::Original< void >, reference > :
		public internal::MatrixContainer< T > {

	private:
		/*********************
		    Storage info friends
		******************** */

		using self_type = Matrix< T, structures::General, Density::Dense, view::Original< void >, reference >;

		// template< typename fwd_iterator >
		// friend RC buildMatrix( Matrix< T, structures::General, Density::Dense, view::Original< void >, reference > &, const fwd_iterator & start, const fwd_iterator & end );
		template< typename fwd_iterator >
		friend RC buildMatrix( Matrix< T, structures::General, Density::Dense, view::Original< void >, reference > & A, const fwd_iterator & start, const fwd_iterator & end );

		template< typename fwd_iterator >
		RC buildMatrixUnique( const fwd_iterator & start, const fwd_iterator & end ) {
			std::cout << "Building Matrix<>; calling buildMatrix( Matrix<> )\n";
			return buildMatrix( *(this->_container), start, end );
		}

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::General;

		// A general Structure knows how to define a reference to itself (which is an original reference view)
		// as well as other static views.
		template < view::Views view_tag, bool d=false >
		struct view_type;

		template < bool d >
		struct view_type< view::original, d > {
			using type = Matrix< T, structures::General, Density::Dense, view::Original< self_type >, reference >;
		};

		template < bool d >
		struct view_type< view::transpose, d > {
			using type = Matrix< T, structures::General, Density::Dense, view::Transpose< self_type >, reference >;
		};

		template < bool d >
		struct view_type< view::diagonal, d > {
			using type = Vector< T, structures::General, Density::Dense, view::Diagonal< self_type >, reference >;
		};

		Matrix( const size_t rows, const size_t cols, const size_t cap = 0 ) :
			internal::MatrixContainer< T >( rows, cols, cap ) {
		}

	}; // Matrix General, container

	/**
	 * View of a general Matrix.
	 */
	template< typename T, typename View >
	class Matrix< T, structures::General, Density::Dense, View, reference > :
		public internal::MatrixReference< typename View::applied_to > {

	private:
		using self_type = Matrix< T, structures::General, Density::Dense, View, reference >;
		using target_type = typename View::applied_to;

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::General;

		template < view::Views view_tag, bool d=false >
		struct view_type;

		template < bool d >
		struct view_type< view::original, d > {
			using type = Matrix< T, structures::General, Density::Dense, view::Original< self_type >, reference >;
		};

		template < bool d >
		struct view_type< view::transpose, d > {
			using type = Matrix< T, structures::General, Density::Dense, view::Transpose< self_type >, reference >;
		};

		Matrix( ) : internal::MatrixBase( 0, 0 ) {}

		Matrix( target_type & struct_mat ) : internal::MatrixReference< target_type >( struct_mat ) {}

		Matrix( target_type & struct_mat, std::shared_ptr< imf::IMF > imf_l, std::shared_ptr< imf::IMF > imf_r ) :
			internal::MatrixReference< target_type >( struct_mat, imf_l, imf_r ) {}

	}; // Matrix General reference

	template< typename T, typename Structure >
	class Matrix< T, Structure, Density::Dense, view::Original< void >, reference > :
		public internal::MatrixContainer< T > {

	private:
		/*********************
		    Storage info friends
		******************** */

		using self_type = Matrix< T, Structure, Density::Dense, view::Original< void >, reference >;

		template< typename fwd_iterator >
		friend RC buildMatrix( Matrix< T, Structure, Density::Dense, view::Original< void >, reference > &, const fwd_iterator &, const fwd_iterator ) noexcept;

	public:
		using value_type = T;
		using structure = Structure;

		/** The type of an identify view over the present type */
		template < view::Views view_tag, bool d=false >
		struct view_type;

		template < bool d >
		struct view_type< view::original, d > {
			using type = Matrix< T, Structure, Density::Dense, view::Original< self_type >, reference >;
		};

		Matrix( const size_t rows, const size_t cols, const size_t cap = 0 ) :
			internal::MatrixContainer< T >( rows, cols, cap ) {}
	}; // class Matrix

	template< typename T >
	class Matrix< T, structures::Square, Density::Dense, view::Original< void >, reference > :
		public internal::MatrixContainer< T > {

	private:
		using self_type = Matrix< T, structures::Square, Density::Dense, view::Original< void >, reference >;

		template< typename InputType, typename Structure, typename View, typename fwd_iterator >
		friend RC buildMatrix( Matrix< InputType, Structure, Density::Dense, View, reference > &, const fwd_iterator & start, const fwd_iterator & end ) noexcept;

		template< typename fwd_iterator >
		RC buildMatrixUnique( const fwd_iterator & start, const fwd_iterator & end ) {
			std::cout << "Building Matrix<>; calling buildMatrix( Matrix<> )\n";
			return buildMatrix( *(this->_container), start, end );
		}

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::Square;

		template < view::Views view_tag, bool d=false >
		struct view_type;

		template < bool d >
		struct view_type< view::original, d > {
			using type = Matrix< T, structures::Square, Density::Dense, view::Original< self_type >, reference >;
		};

		template < bool d >
		struct view_type< view::transpose, d > {
			using type = Matrix< T, structures::Square, Density::Dense, view::Transpose< self_type >, reference >;
		};

		Matrix( const size_t rows, const size_t cap = 0 ) :
			internal::MatrixContainer< T >( rows, rows, cap ) {}

	}; // Matrix Square, container

	template< typename T, typename View >
	class Matrix< T, structures::Square, Density::Dense, View, reference > :
		public internal::MatrixReference< typename View::applied_to > {

	private:
		using self_type = Matrix< T, structures::Square, Density::Dense, View, reference >;
		using target_type = typename View::applied_to;

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::Square;

		template < view::Views view_tag, bool d=false >
		struct view_type;

		template < bool d >
		struct view_type< view::original, d > {
			using type = Matrix< T, structures::Square, Density::Dense, view::Original< self_type >, reference >;
		};

		template < bool d >
		struct view_type< view::transpose, d > {
			using type = Matrix< T, structures::Square, Density::Dense, view::Transpose< self_type >, reference >;
		};

		// ref to empty matrix
		Matrix( ) : internal::MatrixReference< target_type >() {}

		Matrix( target_type & struct_mat ) : internal::MatrixReference< target_type >( struct_mat ) {
			if( nrows( struct_mat ) != ncols( struct_mat ) ) {
				throw std::length_error( "Square Matrix reference to non-square target." );
			}
		}

	}; // Matrix Square reference

	// Matrix UpperTriangular, container
	template< typename T >
	class Matrix< T, structures::UpperTriangular, Density::Dense, view::Original< void >, reference > :
		public internal::MatrixContainer< T > {

	private:
		/*********************
		    Storage info friends
		******************** */

		using self_type = Matrix< T, structures::UpperTriangular, Density::Dense, view::Original< void >, reference >;

		template< typename InputType, typename Structure, typename View, typename fwd_iterator >
		friend RC buildMatrix( Matrix< InputType, Structure, Density::Dense, View, reference > &, const fwd_iterator & start, const fwd_iterator & end ) noexcept;

		template< typename fwd_iterator >
		RC buildMatrixUnique( const fwd_iterator & start, const fwd_iterator & end ) {
			std::cout << "Building Matrix<>; calling buildMatrix( Matrix<> )\n";
			return buildMatrix( *(this->_container), start, end );
		}

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::UpperTriangular;

		template < view::Views view_tag, bool d=false >
		struct view_type;

		template < bool d >
		struct view_type< view::original, d > {
			using type = Matrix< T, structures::UpperTriangular, Density::Dense, view::Original< self_type >, reference >;
		};

		template < bool d >
		struct view_type< view::transpose, d > {
			using type = Matrix< T, structures::LowerTriangular, Density::Dense, view::Transpose< self_type >, reference >;
		};

		Matrix( const size_t rows, const size_t cols, const size_t cap = 0 ) :
			internal::MatrixContainer< T >( rows, cols, cap ) {}

	}; // Matrix UpperTriangular, container

	// Matrix UpperTriangular, reference
	template< typename T, typename View >
	class Matrix< T, structures::UpperTriangular, Density::Dense, View, reference > :
		public internal::MatrixReference< typename View::applied_to > {

	private:
		using self_type = Matrix< T, structures::UpperTriangular, Density::Dense, View, reference >;
		using target_type = typename View::applied_to;

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::UpperTriangular;

		template < view::Views view_tag, bool d=false >
		struct view_type;

		template < bool d >
		struct view_type< view::original, d > {
			using type = Matrix< T, structures::UpperTriangular, Density::Dense, view::Original< self_type >, reference >;
		};

		template < bool d >
		struct view_type< view::transpose, d > {
			using type = Matrix< T, structures::LowerTriangular, Density::Dense, view::Transpose< self_type >, reference >;
		};

		// ref to empty matrix
		Matrix( ) : internal::MatrixReference< target_type >() {}

		Matrix( target_type & struct_mat ) : internal::MatrixReference< target_type >( struct_mat ) {
			// No matter the view it has to be a square matrix
		}

		Matrix( target_type & struct_mat, std::shared_ptr< imf::IMF > imf_l, std::shared_ptr< imf::IMF > imf_r ) :
			internal::MatrixReference< target_type >( struct_mat, imf_l, imf_r ) {}

	}; //  Matrix UpperTriangular, reference

	// Matrix Identity, container
	// Should Identity be a MatrixContainer?
	template< typename T >
	class Matrix< T, structures::Identity, Density::Dense, view::Original< void >, reference > :
		public internal::MatrixContainer< T > {

	private:
		/*********************
		    Storage info friends
		******************** */

		using self_type = Matrix< T, structures::Identity, Density::Dense, view::Original< void >, reference >;

		template< typename InputType, typename Structure, typename View, typename fwd_iterator >
		friend RC buildMatrix( Matrix< InputType, Structure, Density::Dense, View, reference > &, const fwd_iterator & start, const fwd_iterator & end ) noexcept;

		template< typename fwd_iterator >
		RC buildMatrixUnique( const fwd_iterator & start, const fwd_iterator & end ) {
			std::cout << "Building Matrix<>; calling buildMatrix( Matrix<> )\n";
			return buildMatrix( *(this->_container), start, end );
		}

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::Identity;

		template < view::Views view_tag, bool d=false >
		struct view_type;

		template < bool d >
		struct view_type< view::original, d > {
			using type = Matrix< T, structures::Identity, Density::Dense, view::Original< self_type >, reference >;
		};

		template < bool d >
		struct view_type< view::transpose, d > {
			using type = Matrix< T, structures::Identity, Density::Dense, view::Transpose< self_type >, reference >;
		};

		Matrix( const size_t rows, const size_t cap = 0 ) :
			internal::MatrixContainer< T >( rows, rows, cap ) {}

	}; // Matrix Identity, container

	namespace structures {

		/**
		 * @brief Checks if a structured matrix has structure \a Structure.
		 * 
		 * @tparam MatrixT The structured matrix type to be tested. 
		 * @tparam Structure 		 The structure type which should be implied by \a MatrixT::structure.
		 */
		template< typename MatrixT, typename Structure >
		struct is_a {
			/**
			 * \a value is true iff \a Structure is implied by \a MatrixT::structure.
			 */
			static constexpr bool value = is_in< Structure, typename MatrixT::structure::inferred_structures >::value;
		};

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
		typename T, typename Structure, enum Density density, typename View, enum Backend backend >
	typename Matrix< T, Structure, density, View, backend >::template view_type< view::original >::type
	get_view( Matrix< T, Structure, density, View, backend > & source ) {

		using source_strmat_t = Matrix< T, Structure, density, View, backend >;
		using target_strmat_t = typename source_strmat_t::template view_type< view::original >::type;

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
		typename T, typename Structure, enum Density density, typename View, enum Backend backend >
	typename Matrix< T, Structure, density, View, backend >::template view_type< target_view >::type
	get_view( Matrix< T, Structure, density, View, backend > &source ) {

		using source_strmat_t = Matrix< T, Structure, density, View, backend >;
		using target_strmat_t = typename source_strmat_t::template view_type< target_view >::type;

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
		typename T, typename Structure, enum Density density, typename View, enum Backend backend >
	Matrix< T, TargetStructure, density, view::Original< Matrix< T, Structure, density, View, backend > >, backend > 
	get_view( Matrix< T, Structure, density, View, backend > &source ) {

		static_assert( structures::is_in< Structure, typename TargetStructure::inferred_structures >::value,
			"Can only create a view when the target structure is compatible with the source." );

		using source_strmat_t = Matrix< T, Structure, density, View, backend >;
		using target_strmat_t = Matrix< T, TargetStructure, density, view::Original< source_strmat_t >, backend >;

		target_strmat_t target( source );

		return target;
	}

	namespace internal {
		/**
		 * Implement a gather through a View over compatible Structure using provided Index Mapping Functions.
		 * The compatibility depends on the TargetStructure, SourceStructure and IMFs, and is calculated during runtime.
		 */

		template< 
			typename TargetStructure, 
			typename T, typename Structure, enum Density density, typename View, enum Backend backend >
		alp::Matrix< T, TargetStructure, density, view::Original< alp::Matrix< T, Structure, density, View, backend > >, backend > 
		get_view( alp::Matrix< T, Structure, density, View, backend > &source,
				std::shared_ptr< imf::IMF > imf_r, std::shared_ptr< imf::IMF > imf_c ) {
			
			if( std::dynamic_pointer_cast< imf::Select >( imf_r ) || std::dynamic_pointer_cast< imf::Select >( imf_c ) ) {
				throw std::runtime_error("Cannot gather with imf::Select yet.");
			}
			// No static check as the compatibility depends on IMF, which is a runtime level parameter
			if( ! (TargetStructure::template isInstantiableFrom< Structure >( * imf_r, * imf_c ) ) ) {
				throw std::runtime_error("Cannot gather into specified TargetStructure from provided SourceStructure and Index Mapping Functions.");
			}

			using source_strmat_t = alp::Matrix< T, Structure, density, View, backend >;
			using target_strmat_t = alp::Matrix< T, TargetStructure, density, view::Original< source_strmat_t >, backend >;

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
		typename T, typename Structure, enum Density density, typename View, enum Backend backend >
	Matrix< T, TargetStructure, density, view::Original< Matrix< T, Structure, density, View, backend > >, backend > 
	get_view( Matrix< T, Structure, density, View, backend > &source,
			const utils::range& rng_r, const utils::range& rng_c ) {
		
		auto imf_r = std::make_shared< imf::Strided >( rng_r.count(), nrows(source), rng_r.start, rng_r.stride );
		auto imf_c = std::make_shared< imf::Strided >( rng_c.count(), ncols(source), rng_c.start, rng_c.stride );

		return internal::get_view<TargetStructure, T, Structure, density, View, backend >( source, imf_r, imf_c );
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
		typename T, typename Structure, enum Density density, typename View, enum Backend backend >
	Matrix< T, Structure, density, view::Original< Matrix< T, Structure, density, View, backend > >, backend > 
	get_view( Matrix< T, Structure, density, View, backend > &source,
			const utils::range& rng_r, const utils::range& rng_c ) {
		
		auto imf_r = std::make_shared< imf::Strided >( rng_r.count(), nrows(source), rng_r.start, rng_r.stride );
		auto imf_c = std::make_shared< imf::Strided >( rng_c.count(), ncols(source), rng_c.start, rng_c.stride );

		return internal::get_view<Structure, T, Structure, density, View, backend >( source, imf_r, imf_c );
	}

	namespace internal {
		/**
		 *	
		 * @brief Generate an original view where the type is compliant with the source Matrix.
		 * Version where a selection of rows and columns expressed as vectors of positions 
		 * form a new view with specified target structure.
		 * 
		 * \warning WIP interface. Checking the structural correctness is a costly runtime operation for a 
		 * 			general vector selection. 
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
		 * @param sel_r            A valid vector of row indeces (possibly in any permuted order and with repetition) 
		 * @param sel_c            A valid vector of column indeces (possibly in any permuted order and with repetition)
		 * 
		 * @return A new original view over the source structured matrix.
		 * 
		 */
		template< 
			typename TargetStructure, 
			typename T, typename Structure, enum Density density, typename View, enum Backend backend >
		alp::Matrix< T, TargetStructure, density, view::Original< alp::Matrix< T, Structure, density, View, backend > >, backend > 
		get_view( alp::Matrix< T, Structure, density, View, backend > &source,
				const std::vector< size_t >& sel_r, const std::vector< size_t >& sel_c ) {
			
			auto imf_r = std::make_shared< imf::Select >( nrows(source), sel_r );
			auto imf_c = std::make_shared< imf::Select >( ncols(source), sel_c );

			return internal::get_view<TargetStructure, T, Structure, density, View, backend>( source, imf_r, imf_c );
		}
	} //namespace internal

	/** Returns a constant reference to an Identity matrix of the provided size */
	template< typename T >
	const Matrix< T, structures::Identity, Density::Dense, view::Original< void >, reference > &
	I( const size_t n ) {
		using return_type = Matrix< T, structures::Identity, Density::Dense, view::Original< void >, reference >;
		return_type * ret = new return_type( n );
		return * ret;
	}

	/** Returns a constant reference to a Zero matrix of the provided size */
	template< typename T >
	const Matrix< T, structures::Zero, Density::Dense, view::Original< void >, reference > &
	Zero( const size_t rows, const size_t cols ) {
		using return_type = Matrix< T, structures::Zero, Density::Dense, view::Original< void >, reference >;
		return_type * ret = new return_type( rows, cols );
		return * ret;
	}

	/** Returns a constant reference to a matrix representing Givens rotation
	 * of the provided size n and parameters i, j, s and c, where
	 * s = sin( theta ) and c = cos( theta )
	 */
	template< typename T >
	const Matrix< T, structures::Square, Density::Dense, view::Original< void >, reference > &
	Givens( const size_t n, const size_t i, const size_t j, const T s, const T c ) {
		using return_type = const Matrix< T, structures::Square, Density::Dense, view::Original< void >, reference >;
		return_type * ret = new return_type( n );
		// TODO: initialize matrix values according to the provided parameters
		return * ret;
	}

} // namespace alp

#endif // end ``_H_ALP_REFERENCE_MATRIX''