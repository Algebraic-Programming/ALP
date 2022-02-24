
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

	namespace internal {
		/** Helper class to calculate number of elements for raw matrix storage,
		 * depending on the storage scheme and the structure.
		 * 
		 * \internal Thoughts:
		 * Currently, this class has to depend on structure (to allow for "optimal" element count).
		 * E.g. Matrix< Dense:full, Structure::Triangular> can use less storage than
		 * Matrix< Dense::full, Structure::General>.
		 * If we add Storage::packed (which could store Triangular or Symetric matrices),
		 * then it seems that Structure can be removed from template list of this class.
		 * 
		 * TODO: Expand the class to StructuredMatrix
		*/
		template< typename T, typename grb::storage::Dense, typename Structure = grb::structures::General >
		class DataElementsCalculator {
		public:
			static size_t calculate( const Matrix< T, reference_dense > & A ) {
				(void)A;
				std::cout << "Cannot determine storage size due to unspecified storage scheme. \n";
				assert( false );
				return (size_t)-1;
			}
		};

		template< typename T >
		class DataElementsCalculator< T, grb::storage::Dense::full > {
		public:
			static size_t calculate( const Matrix< T, reference_dense > & A ) {
				return nrows( A ) * ncols( A );
			}
		};

		template< typename T >
		class DataElementsCalculator< T, grb::storage::Dense::full, grb::structures::Triangular > {
		public:
			static size_t calculate( const Matrix< T, reference_dense > & A ) {
				// structures::Triangular assumes that the matrix is structures::Square
				std::size_t m = nrows( A );
				return m * ( m + 1 ) / 2;
			}
		};

		template< typename T >
		class DataElementsCalculator< T, grb::storage::Dense::band > {
		public:
			static size_t calculate( const Matrix< T, reference_dense > & A ) {
				size_t ku = 1; // nsuperdiagonals( A ); only exists in banded matrix
				size_t kl = 1; // nsubdiagonals( A ); only exists in banded matrix
				return ncols( A ) * ( ku + kl + 1 );
			}
		};

		template< typename T >
		class DataElementsCalculator< T, grb::storage::Dense::array1d > {
		public:
			static size_t calculate( const Matrix< T, reference_dense > & A ) {
				size_t min_dimension = min( nrows( A ), ncols( A ) );
				// Temporary: Assume main diagonal + one subdiagonal + one superdiagonal
				return min_dimension + 2 * ( min_dimension - 1 );
			}
		};
	} // namespace internal

	template< typename T >
	T * getRaw( Matrix< T, reference_dense > & ) noexcept;

	template< typename T >
	const T * getRaw( const Matrix< T, reference_dense > & ) noexcept;

	template< typename T >
	size_t nrows( const Matrix< T, reference_dense > & ) noexcept;

	template< typename T >
	size_t ncols( const Matrix< T, reference_dense > & ) noexcept;

	namespace internal {

		template< typename T >
		const bool & getInitialized( const grb::Matrix< T, reference_dense > & A ) noexcept {
			return A.initialized;
		}

		template< typename T >
		void setInitialized( grb::Matrix< T, reference_dense > & A, bool initialized ) noexcept {
			A.initialized = initialized;
		}
	} // namespace internal

	/** \internal TODO */
	template< typename T >
	class Matrix< T, reference_dense > {

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

		friend T * getRaw< T >( Matrix< T, reference_dense > & ) noexcept;

		friend const T * getRaw< T >( const Matrix< T, reference_dense > & ) noexcept;

		/* ********************
		        IO friends
		   ******************** */

		template< typename InputType, typename fwd_iterator >
		friend RC buildMatrix( Matrix< InputType, reference_dense > &, fwd_iterator, const fwd_iterator );

		template< typename DataType >
		friend const bool & internal::getInitialized( const grb::Matrix< DataType, reference_dense > & ) noexcept;

		template< typename DataType >
		friend void internal::setInitialized( grb::Matrix< DataType, reference_dense > & , bool ) noexcept;

	private:
		/** Our own type. */
		typedef Matrix< T, reference_dense > self_type;

		/**
		 * The number of rows.
		 *
		 * \internal Not declared const to be able to implement move in an elegant way.
		 */
		size_t m;

		/**
		 * The number of columns.
		 *
		 * \internal Not declared const to be able to implement move in an elegant way.
		 */
		size_t n;

		/** The matrix data. */
		T * __restrict__ data;

		/** Whether the container presently is uninitialized. */
		bool initialized;

		/** @see Matrix::buildMatrixUnique
		 * Returns RC::MISMATCH if the input size does not match the matrix storage size.
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
		typedef T value_type;

		/** @see Matrix::Matrix() */
		Matrix( const size_t rows, const size_t columns ): m( rows ), n( columns ), initialized( false ) {
			// TODO Implement allocation properly
			if( m > 0 && n > 0) {
				data = new (std::nothrow) T[ m * n ];
			} else {
				data = nullptr;
			}

			if ( m > 0 && n > 0 && data == nullptr ) {
				throw std::runtime_error( "Could not allocate memory during grb::Matrix<reference_dense> construction." );
			}

		}

		/** @see Matrix::Matrix( const Matrix & ) */
		Matrix( const Matrix< T, reference_dense > & other ) : Matrix( other.m, other.n ) {
			initialized = other.initialized;
		}

		/** @see Matrix::Matrix( Matrix&& ). */
		Matrix( self_type && other ) noexcept {
			moveFromOther( std::forward< self_type >( other ) );
		}

		/** * Move from temporary. */
		self_type & operator=( self_type && other ) noexcept {
			moveFromOther( std::forward< self_type >( other ) );
			return *this;
		}

		/** @see Matrix::~Matrix(). */
		~Matrix() {
			if( data != nullptr ) {
				delete [] data;
			}
		}
	};

	// template specialisation for GraphBLAS type traits
	template< typename T >
	struct is_container< Matrix< T, reference_dense > > {
		/** A reference_dense Matrix is a GraphBLAS object. */
		static const constexpr bool value = true;
	};

	template< typename T >
	T * getRaw( Matrix< T, reference_dense > & m ) noexcept {
		return m.data;
	}

	template< typename T >
	const T * getRaw( const Matrix< T, reference_dense > & m ) noexcept {
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

	namespace internal {
		/** Forward declaration */
		template< typename T >
		class StructuredMatrixContainer;

		/** Container reference getters used by friend functions of specialized StructuredMatrix */
		template< typename T >
		const Matrix< T, reference_dense > & getContainer( const StructuredMatrixContainer< T > & A );

		template< typename T >
		Matrix< T, reference_dense > & getContainer( StructuredMatrixContainer< T > & A );

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
		class StructuredMatrixBase;

		size_t nrows( const StructuredMatrixBase & A ) noexcept;

		size_t ncols( const StructuredMatrixBase & A ) noexcept;

		std::pair< size_t, size_t > dims( const StructuredMatrixBase & A ) noexcept;

		bool getInitialized( StructuredMatrixBase & ) noexcept;

		void getInitialized( StructuredMatrixBase &, bool ) noexcept;

		template< typename T, typename Structure, typename Storage, typename View >
		bool getInitialized( StructuredMatrix< T, Structure, Storage, View, reference_dense > & A ) noexcept {
			return getInitialized( A );
		}

		template< typename T, typename Structure, typename Storage, typename View >
		void setInitialized( StructuredMatrix< T, Structure, Storage, View, reference_dense > & A, bool initialized ) noexcept {
			setInitialized( A, initialized );
		}
	} // namespace internal

	/**
	 * Here starts spec draft for StructuredMatrix
	 */

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

		class StructuredMatrixBase {

		protected:
			friend size_t nrows( const StructuredMatrixBase & A ) noexcept {
				return A._dims().first;
			}

			friend size_t ncols( const StructuredMatrixBase & A ) noexcept {
				return A._dims().second;
			}

			friend std::pair< size_t, size_t > dims( const StructuredMatrixBase & A ) noexcept {
				return A._dims();
			}

			friend bool getInitialized( StructuredMatrixBase & A ) noexcept {
				return A.initialized;
			}

			friend void setInitialized( StructuredMatrixBase & A, bool initialized ) noexcept {
				A.initialized = initialized;
			}

			std::shared_ptr<imf::IMF> imf_l, imf_r;

			/** Whether the container presently is initialized or not. */
			bool initialized;

			virtual std::pair< size_t, size_t > _dims() const {
				return std::make_pair( imf_l->n, imf_r->n );
			}

			StructuredMatrixBase( size_t rows, size_t cols ) :
				imf_l( std::make_shared< imf::Id >( rows ) ),
				imf_r( std::make_shared< imf::Id >( cols ) ),
				initialized( false ) {}

			StructuredMatrixBase( std::shared_ptr< imf::IMF > imf_l, std::shared_ptr< imf::IMF > imf_r ) :
				imf_l( imf_l ),
				imf_r( imf_r ),
				initialized( false ) {}

		};

		/**
		 * Base class with container-related attributes, used in container-type StructuredMatrix specializations
		 */
		template< typename T >
		class StructuredMatrixContainer : public StructuredMatrixBase {
		protected:
			friend const Matrix< T, reference_dense > & getContainer( const StructuredMatrixContainer< T > & A ) {
				return *( A._container );
			}

			friend Matrix< T, reference_dense > & getContainer( StructuredMatrixContainer< T > & A ) {
				return *( A._container );
			}

			Matrix< T, reference_dense > * _container;

			/**
			 * A container's storage scheme. \a storage_scheme is not exposed to the user as an option
			 * but can defined by ALP at different points in the execution depending on the \a backend choice.
			 * For example, if the container is associated to an I/O matrix, with a reference backend
			 * it might be set to reflect the storage scheme of the user data as specified at buildMatrix.
			 * If \a backend is set to \a mlir then the scheme could be fixed by the JIT compiler to effectively
			 * support its optimization strategy.
			 * At construction time and until the moment the scheme decision is made it may be set to
			 * an appropriate default choice, e.g. if \a StorageSchemeType is \a storage::Dense then
			 * \a storage::Dense::full could be used.
			 */
			storage::Dense storage_scheme;

			StructuredMatrixContainer( size_t rows, size_t cols ) :
				StructuredMatrixBase( rows, cols ),
				_container( new Matrix< T, reference_dense >(rows, cols) ),
				storage_scheme( storage::full ) {}

		};

		/**
		 * Base class with reference-related attributes, used in Views on container-type StructuredMatrix specializations
		 * 
		 */
		template< typename TargetType >
		class StructuredMatrixReference : public StructuredMatrixBase {
		protected:
			TargetType * ref;

			StructuredMatrixReference() : StructuredMatrixBase( 0, 0 ), ref( nullptr ) {}
			StructuredMatrixReference( TargetType & struct_mat ) : StructuredMatrixBase( nrows( struct_mat ), ncols( struct_mat ) ), ref( & struct_mat ) {}
			StructuredMatrixReference( TargetType & struct_mat, std::shared_ptr< imf::IMF > imf_l, std::shared_ptr< imf::IMF > imf_r ) :
				StructuredMatrixBase( imf_l, imf_r ), ref( & struct_mat ) {}
		};
	} // namespace internal

	template< typename T, typename Structure >
	class StructuredMatrix< T, Structure, storage::Dense, view::Identity< void >, reference_dense > :
		public internal::StructuredMatrixContainer< T > {

	private:
		/*********************
		    Storage info friends
		******************** */

		using self_type = StructuredMatrix< T, Structure, storage::Dense, view::Identity< void >, reference_dense >;

		template< typename fwd_iterator >
		friend RC buildMatrix( const StructuredMatrix< T, Structure, storage::Dense, view::Identity< void >, reference_dense > &, const fwd_iterator &, const fwd_iterator ) noexcept;

	public:
		using value_type = T;
		using structure = Structure;

		// A general Structure knows how to define a reference to itself (which is an identity reference view).
		using identity_t = StructuredMatrix< T, Structure, storage::Dense, view::Identity< self_type >, reference_dense >;

		StructuredMatrix( const size_t rows, const size_t cols ) :
			internal::StructuredMatrixContainer< T >( rows, cols ) {}
	}; // class StructuredMatrix

	template< typename T >
	class StructuredMatrix< T, structures::General, storage::Dense, view::Identity< void >, reference_dense > :
		public internal::StructuredMatrixContainer< T > {

	private:
		/*********************
		    Storage info friends
		******************** */

		using self_type = StructuredMatrix< T, structures::General, storage::Dense, view::Identity< void >, reference_dense >;

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
		using structure = structures::General;

		// A general Structure knows how to define a reference to itself (which is an identity reference view)
		// as well as other static views.
		using identity_t = StructuredMatrix< T, structures::General, storage::Dense, view::Identity< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::General, storage::Dense, view::Transpose< self_type >, reference_dense >;

		StructuredMatrix( const size_t rows, const size_t cols ) :
			internal::StructuredMatrixContainer< T >( rows, cols ) {
		}

	}; // StructuredMatrix General, container

	template< typename T >
	class StructuredMatrix< T, structures::Square, storage::Dense, view::Identity< void >, reference_dense > :
		public internal::StructuredMatrixContainer< T > {

	private:
		using self_type = StructuredMatrix< T, structures::Square, storage::Dense, view::Identity< void >, reference_dense >;

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::Square;

		using identity_t = StructuredMatrix< T, structures::Square, storage::Dense, view::Identity< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::Square, storage::Dense, view::Transpose< self_type >, reference_dense >;

		StructuredMatrix( const size_t rows ) :
			internal::StructuredMatrixContainer< T >( rows, rows ) {}

	}; // StructuredMatrix Square, container

	/**
	 * Reference to a general Matrix generalized over views.
	 */
	template< typename T, typename View >
	class StructuredMatrix< T, structures::General, storage::Dense, View, reference_dense > :
		public internal::StructuredMatrixReference< typename View::applied_to > {

	private:
		using self_type = StructuredMatrix< T, structures::General, storage::Dense, View, reference_dense >;
		using target_type = typename View::applied_to;

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::General;

		using identity_t = StructuredMatrix< T, structures::General, storage::Dense, view::Identity< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::General, storage::Dense, view::Transpose< self_type >, reference_dense >;

		StructuredMatrix( ) : internal::StructuredMatrixBase( 0, 0 ) {}

		StructuredMatrix( target_type & struct_mat ) : internal::StructuredMatrixReference< target_type >( struct_mat ) {}

		StructuredMatrix( target_type & struct_mat, std::shared_ptr< imf::IMF > imf_l, std::shared_ptr< imf::IMF > imf_r ) :
			internal::StructuredMatrixReference< target_type >( struct_mat, imf_l, imf_r ) {}

	}; // StructuredMatrix General reference

	template< typename T, typename View >
	class StructuredMatrix< T, structures::Square, storage::Dense, View, reference_dense > :
		public internal::StructuredMatrixReference< typename View::applied_to > {

	private:
		using self_type = StructuredMatrix< T, structures::Square, storage::Dense, View, reference_dense >;
		using target_type = typename View::applied_to;

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::Square;

		using identity_t = StructuredMatrix< T, structures::Square, storage::Dense, view::Identity< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::Square, storage::Dense, view::Transpose< self_type >, reference_dense >;

		// ref to empty matrix
		StructuredMatrix( ) : internal::StructuredMatrixReference< target_type >() {}

		StructuredMatrix( target_type & struct_mat ) : internal::StructuredMatrixReference< target_type >( struct_mat ) {
			if( nrows( struct_mat ) != ncols( struct_mat ) ) {
				throw std::length_error( "Square StructuredMatrix reference to non-square target." );
			}
		}

	}; // StructuredMatrix Square reference

	// StructuredMatrix UpperTriangular, container
	template< typename T >
	class StructuredMatrix< T, structures::UpperTriangular, storage::Dense, view::Identity< void >, reference_dense > :
		public internal::StructuredMatrixContainer< T > {

	private:
		/*********************
		    Storage info friends
		******************** */

		using self_type = StructuredMatrix< T, structures::UpperTriangular, storage::Dense, view::Identity< void >, reference_dense >;

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

		// A general Structure knows how to define a reference to itself (which is an identity reference view)
		// as well as other static views.
		using identity_t = StructuredMatrix< T, structures::UpperTriangular, storage::Dense, view::Identity< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::LowerTriangular, storage::Dense, view::Transpose< self_type >, reference_dense >;

		StructuredMatrix( const size_t rows, const size_t cols ) :
			internal::StructuredMatrixContainer< T >( rows, cols ) {}

	}; // StructuredMatrix UpperTriangular, container

	// StructuredMatrix UpperTriangular, reference
	template< typename T, typename View >
	class StructuredMatrix< T, structures::UpperTriangular, storage::Dense, View, reference_dense > :
		public internal::StructuredMatrixReference< typename View::applied_to > {

	private:
		using self_type = StructuredMatrix< T, structures::UpperTriangular, storage::Dense, View, reference_dense >;
		using target_type = typename View::applied_to;

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::UpperTriangular;

		using identity_t = StructuredMatrix< T, structures::UpperTriangular, storage::Dense, view::Identity< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::UpperTriangular, storage::Dense, view::Transpose< self_type >, reference_dense >;

		// ref to empty matrix
		StructuredMatrix( ) : internal::StructuredMatrixReference< target_type >() {}

		StructuredMatrix( target_type & struct_mat ) : internal::StructuredMatrixReference< target_type >( struct_mat ) {
			// No matter the view it has to be a square matrix
		}

		StructuredMatrix( target_type & struct_mat, std::shared_ptr< imf::IMF > imf_l, std::shared_ptr< imf::IMF > imf_r ) :
			internal::StructuredMatrixReference< target_type >( struct_mat, imf_l, imf_r ) {}

	}; //  StructuredMatrix UpperTriangular, reference

	namespace structures {

		// GraphBLAS type traits for structure
		template< typename StructuredMatrix, typename Structure >
		struct is_a {
			static constexpr bool value = is_in< Structure, typename StructuredMatrix::structure::inferred_structures >::value;
		};

	} // namespace structures

	/**
	 * Generate an identity view where the type is compliant with the source StructuredMatrix.
	 * If no target structure is specified the one of the source type is assumed.
	 * Otherwise it can only generate a type if the target structure is the same as the source type
	 * or a more specialized version that would preserve its static properties (e.g., symmetric reference
	 * to a square matrix -- any assumption based on symmetry would not break those based on square).
	 */

	template< typename T, typename Structure, typename StorageSchemeType, typename View, enum Backend backend >
	StructuredMatrix< T, Structure, StorageSchemeType, view::Identity< StructuredMatrix< T, Structure, StorageSchemeType, View, backend > >, backend > 
	get_view( StructuredMatrix< T, Structure, StorageSchemeType, View, backend > &source ) {

		using source_strmat_t = StructuredMatrix< T, Structure, StorageSchemeType, View, backend >;
		using target_strmat_t = StructuredMatrix< T, Structure, StorageSchemeType, view::Identity< source_strmat_t >, backend >;

		target_strmat_t target( source );

		return target;
	}

	template< typename TargetStructure, typename T, typename Structure, typename StorageSchemeType, typename View, enum Backend backend >
	StructuredMatrix< T, TargetStructure, StorageSchemeType, view::Identity< StructuredMatrix< T, Structure, StorageSchemeType, View, backend > >, backend > 
	get_view( StructuredMatrix< T, Structure, StorageSchemeType, View, backend > &source ) {

		static_assert( structures::is_in< Structure, typename TargetStructure::inferred_structures >::value,
			"Can only create a view when the target structure is compatible with the source." );

		using source_strmat_t = StructuredMatrix< T, Structure, StorageSchemeType, View, backend >;
		using target_strmat_t = StructuredMatrix< T, TargetStructure, StorageSchemeType, view::Identity< source_strmat_t >, backend >;

		target_strmat_t target( source );

		return target;
	}

	template< typename StructuredMatrixT >
	typename StructuredMatrixT::transpose_t
	transpose( StructuredMatrixT &smat ) {

		typename StructuredMatrixT::transpose_t smat_trans( smat );

		return smat_trans;
	}

	/**
	 * Implement a gather through a View over compatible Structure using provided Index Mapping Functions.
	 * The compatibility depends on the TargetStructure, SourceStructure and IMFs, and is calculated during runtime.
	 */

	template< typename TargetStructure, typename T, typename Structure, typename StorageSchemeType, typename View, enum Backend backend >
	StructuredMatrix< T, TargetStructure, StorageSchemeType, view::Identity< StructuredMatrix< T, Structure, StorageSchemeType, View, backend > >, backend > 
	get_view( StructuredMatrix< T, Structure, StorageSchemeType, View, backend > &source,
	          std::shared_ptr< imf::IMF > imf_r, std::shared_ptr< imf::IMF > imf_c ) {

		// No static check as the compatibility depends on IMF, which is a runtime level parameter

		if( ! TargetStructure::isInstantiableFrom( source, * imf_r, * imf_c ) ) {
			throw std::runtime_error("Cannot gather into specified TargetStructure from provided SourceStructure and Index Mapping Functions.");
		}

		using source_strmat_t = StructuredMatrix< T, Structure, StorageSchemeType, View, backend >;
		using target_strmat_t = StructuredMatrix< T, TargetStructure, StorageSchemeType, view::Identity< source_strmat_t >, backend >;

		target_strmat_t target( source, imf_r, imf_c );

		return target;
	}

} // namespace grb

#endif // end ``_H_GRB_DENSEREF_MATRIX''
