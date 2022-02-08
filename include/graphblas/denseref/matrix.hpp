
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

		/** @see Matrix::buildMatrixUnique */
		template< typename fwd_iterator >
		RC buildMatrixUnique( const fwd_iterator & _start, const fwd_iterator & _end ) {
			// detect trivial case
			if ( _start == _end || m == 0 || n == 0) {
				return SUCCESS;
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

	/**
	 * Here starts spec draft for StructuredMatrix
	 */

	template< typename D, typename Structure, typename View >
	size_t nrows( const StructuredMatrix< D, Structure, storage::Dense, View, reference_dense > & A ) noexcept {
		return A._dims().first;
	}

	template< typename D, typename Structure, typename View >
	size_t ncols( const StructuredMatrix< D, Structure, storage::Dense, View, reference_dense > & A ) noexcept {
		return A._dims().second;
	}

	template< typename D, typename Structure, typename View >
	std::pair< size_t, size_t > dims( const StructuredMatrix< D, Structure, storage::Dense, View, reference_dense > & A ) noexcept {
		return A._dims();
	}

	template< typename T, typename Structure >
	class StructuredMatrix< T, Structure, storage::Dense, view::Identity< void >, reference_dense > {

	private:
		/*********************
		    Storage info friends
		******************** */

		friend size_t nrows<>( const StructuredMatrix< T, Structure, storage::Dense, view::Identity< void >, reference_dense > & ) noexcept;

		friend size_t ncols<>( const StructuredMatrix< T, Structure, storage::Dense, view::Identity< void >, reference_dense > & ) noexcept;

		friend std::pair< size_t, size_t > dims<>( const StructuredMatrix< T, Structure, storage::Dense, view::Identity< void >, reference_dense > & ) noexcept;

		using self_type = StructuredMatrix< T, Structure, storage::Dense, view::Identity< void >, reference_dense >;

		// Physical layout - TBD
		// Matrix< T, reference_dense > A;

		std::shared_ptr<imf::IMF> imf_l, imf_r;

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

		/** Whether the container presently is initialized or not. */
		bool initialized;

		std::pair< size_t, size_t > _dims() const {
			return std::make_pair( imf_l->n, imf_r->n );
		}

	public:
		using value_type = T;
		using structure = Structure;

		// A general Structure knows how to define a reference to itself (which is an identity reference view).
		using identity_t = StructuredMatrix< T, Structure, storage::Dense, view::Identity< self_type >, reference_dense >;
		using reference_t = identity_t;

		StructuredMatrix( const size_t rows, const size_t cols ) : imf_l( std::make_shared< imf::Id >( rows ) ), imf_r( std::make_shared< imf::Id >( cols ) ), storage_scheme( storage::full ), initialized( false ) {}
	}; // class StructuredMatrix

	template< typename T >
	class StructuredMatrix< T, structures::General, storage::Dense, view::Identity< void >, reference_dense > {

	private:
		/*********************
		    Storage info friends
		******************** */

		friend size_t nrows<>( const StructuredMatrix< T, structures::General, storage::Dense, view::Identity< void >, reference_dense > & ) noexcept;

		friend size_t ncols<>( const StructuredMatrix< T, structures::General, storage::Dense, view::Identity< void >, reference_dense > & ) noexcept;

		friend std::pair< size_t, size_t > dims<>( const StructuredMatrix< T, structures::General, storage::Dense, view::Identity< void >, reference_dense > & ) noexcept;

		using self_type = StructuredMatrix< T, structures::General, storage::Dense, view::Identity< void >, reference_dense >;

		// Physical layout - TBD
		// Matrix< T, reference_dense > A;

		std::shared_ptr<imf::IMF> imf_l, imf_r;

		/**
		 * The container's storage scheme.
		 */
		storage::Dense storage_scheme;

		/** Whether the container presently is initialized or not. */
		bool initialized;

		std::pair< size_t, size_t > _dims() const {
			return std::make_pair( imf_l->n, imf_r->n );
		}

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::General;

		// A general Structure knows how to define a reference to itself (which is an identity reference view)
		// as well as other static views.
		using identity_t = StructuredMatrix< T, structures::General, storage::Dense, view::Identity< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::General, storage::Dense, view::Transpose< self_type >, reference_dense >;

		using reference_t = identity_t;

		StructuredMatrix( const size_t rows, const size_t cols ) : imf_l( std::make_shared< imf::Id >( rows ) ), imf_r( std::make_shared< imf::Id >( cols ) ), storage_scheme( storage::full ), initialized( false ) {}

	}; // StructuredMatrix General, container

	template< typename T >
	class StructuredMatrix< T, structures::Square, storage::Dense, view::Identity< void >, reference_dense > {

	private:
		/*********************
		    Storage info friends
		******************** */

		friend size_t nrows<>( const StructuredMatrix< T, structures::Square, storage::Dense, view::Identity< void >, reference_dense > & ) noexcept;

		friend size_t ncols<>( const StructuredMatrix< T, structures::Square, storage::Dense, view::Identity< void >, reference_dense > & ) noexcept;

		friend std::pair< size_t, size_t > dims<>( const StructuredMatrix< T, structures::Square, storage::Dense, view::Identity< void >, reference_dense > & ) noexcept;

		using self_type = StructuredMatrix< T, structures::Square, storage::Dense, view::Identity< void >, reference_dense >;

		// Physical layout - TBD
		// Matrix< T, reference_dense > A;

		std::shared_ptr<imf::IMF> imf_l, imf_r;

		/**
		 * The container's storage scheme.
		 */
		storage::Dense storage_scheme;

		/** Whether the container presently is initialized or not. */
		bool initialized;

		std::pair< size_t, size_t > _dims() const {
			return std::make_pair( imf_l->n, imf_r->n );
		}

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::Square;

		using identity_t = StructuredMatrix< T, structures::Square, storage::Dense, view::Identity< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::Square, storage::Dense, view::Transpose< self_type >, reference_dense >;

		using reference_t = identity_t;

		StructuredMatrix( const size_t rows ) : imf_l( std::make_shared< imf::Id >( rows ) ), imf_r( std::make_shared< imf::Id >( rows ) ), storage_scheme( storage::full ), initialized( false ) {}

	}; // StructuredMatrix Square, container

	/**
	 * Reference to a Square Matrix generalized over views.
	 */
	template< typename T, typename View >
	class StructuredMatrix< T, structures::General, storage::Dense, View, reference_dense > {

	private:
		using self_type = StructuredMatrix< T, structures::General, storage::Dense, View, reference_dense >;
		using target_type = typename View::applied_to;

		/*********************
		    Storage info friends
		******************** */

		friend size_t nrows<>( const self_type & ) noexcept;

		friend size_t ncols<>( const self_type & ) noexcept;

		friend std::pair< size_t, size_t > dims<>( const self_type & ) noexcept;

		target_type * ref;

		std::shared_ptr<imf::IMF> imf_l, imf_r;

		std::pair< size_t, size_t > _dims() const {
			return std::make_pair( imf_l->n, imf_r->n );
		}

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::General;

		using identity_t = StructuredMatrix< T, structures::General, storage::Dense, view::Identity< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::General, storage::Dense, view::Transpose< self_type >, reference_dense >;

		StructuredMatrix( ) : ref( nullptr ), imf_l( std::make_shared< imf::Id >( 0 ) ), imf_r( std::make_shared< imf::Id >( 0 ) ) {}

		StructuredMatrix( target_type & struct_mat ) : ref( &struct_mat ), imf_l( nullptr ), imf_r( nullptr ) {
			
			std::pair< size_t, size_t > _dims = View::dims( dims( *ref ) );
			imf_l = std::make_shared< imf::Id >( _dims.first  );
			imf_r = std::make_shared< imf::Id >( _dims.second );

		}

	}; // StructuredMatrix General reference

	template< typename T, typename View >
	class StructuredMatrix< T, structures::Square, storage::Dense, View, reference_dense > {

	private:
		using self_type = StructuredMatrix< T, structures::Square, storage::Dense, View, reference_dense >;
		using target_type = typename View::applied_to;
		/*********************
		    Storage info friends
		******************** */

		friend size_t nrows<>( const self_type & ) noexcept;

		friend size_t ncols<>( const self_type & ) noexcept;

		friend std::pair< size_t, size_t > dims<>( const self_type & ) noexcept;

		target_type * ref;

		std::shared_ptr<imf::IMF> imf_l, imf_r;

		std::pair< size_t, size_t > _dims() const {
			return std::make_pair( imf_l->n, imf_r->n );
		}

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;
		using structure = structures::Square;

		using identity_t = StructuredMatrix< T, structures::Square, storage::Dense, view::Identity< self_type >, reference_dense >;
		using transpose_t = StructuredMatrix< T, structures::Square, storage::Dense, view::Transpose< self_type >, reference_dense >;

		// ref to empty matrix
		StructuredMatrix( ) : ref( nullptr ), imf_l( std::make_shared< imf::Id >( 0 ) ), imf_r( std::make_shared< imf::Id >( 0 ) ) {}

		StructuredMatrix( target_type & struct_mat ) : ref( & struct_mat ), imf_l( nullptr ), imf_r( nullptr ) {
			if( nrows( struct_mat ) != ncols( struct_mat ) ) {
				throw std::length_error( "Square StructuredMatrix reference to non-square target." );
			}
			// No matter the view it has to be a square matrix
			imf_l = std::make_shared< imf::Id >( nrows( struct_mat ) );
			imf_r = std::make_shared< imf::Id >( nrows( struct_mat ) );
		}

	}; // StructuredMatrix Square reference

	namespace structures {

		// GraphBLAS type traits for structure
		template< typename StructuredMatrix, typename Structure >
		struct is_a {
			static constexpr bool value = is_in< Structure, typename StructuredMatrix::structure::inferred_structures >::value;
		};

	} // namespace structures

	/**
	 * Peel off any possible view from an input StructuredMatrix type.
	 * Returns a compatible container-type StructuredMatrix.
	 */
	template< typename StructuredMatrixT >
	struct remove_ref;

	template< typename T, typename Structure, typename StorageSchemeType, typename View, enum Backend backend >
	struct remove_ref< StructuredMatrix< T, Structure, StorageSchemeType, View, backend > > {
		using type = StructuredMatrix< T, Structure, StorageSchemeType, view::Identity< void >, backend >;
	};

	/**
	 * Generate a ref StructuredMatrix type.
	 * If no target structure is specified the one of the source type is assumed.
	 * Otherwise it can only generate a type if the target structure is the same as the source type
	 * or a more specialized version that would preserve its static properties (e.g., symmetric reference
	 * to a square matrix -- any assumption based on symmetry would not break those based on square).
	 */
	template< typename StructuredMatrixT, typename TargetStructure = void >
	struct get_ref;

	template< typename T, typename Structure, typename StorageSchemeType, typename View, enum Backend backend >
	struct get_ref< StructuredMatrix< T, Structure, StorageSchemeType, View, backend >, void > {
		using source_strmat_t = StructuredMatrix< T, Structure, StorageSchemeType, View, backend >;
		using type = StructuredMatrix< T, Structure, StorageSchemeType, view::Identity< source_strmat_t >, backend >;
	};

	template< typename T, typename Structure, typename StorageSchemeType, typename View, enum Backend backend, typename TargetStructure >
	struct get_ref< StructuredMatrix< T, Structure, StorageSchemeType, View, backend >, TargetStructure > {

		static_assert( structures::is_in< Structure, typename TargetStructure::inferred_structures >::value,
			"Can only create a ref StructuredMatrix with a target structure that implies the source." );

		using source_strmat_t = StructuredMatrix< T, Structure, StorageSchemeType, View, backend >;
		using type = StructuredMatrix< T, TargetStructure, StorageSchemeType, view::Identity< source_strmat_t >, backend >;
	};

} // namespace grb

#endif // end ``_H_GRB_DENSEREF_MATRIX''
