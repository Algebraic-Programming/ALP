
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

#ifndef _H_GRB_DENSEREF_VECTOR
#define _H_GRB_DENSEREF_VECTOR


#include <stdexcept>
#include <memory>

#include <assert.h>

#include <graphblas/rc.hpp>
#include <graphblas/backends.hpp>

// #include <graphblas/utils/alloc.hpp>
// #include <graphblas/utils/autodeleter.hpp>
// #include <graphblas/denseref/vectoriterator.hpp>

#include <graphblas/imf.hpp>
#include <graphblas/matrix.hpp>
#include <graphblas/storage.hpp>
#include <graphblas/views.hpp>

#include <graphblas/base/vector.hpp>


namespace grb {

	namespace internal {

		template< typename T >
		T * getRaw( Vector< T, reference_dense, void > & ) noexcept;

		template< typename T >
		const T * getRaw( const Vector< T, reference_dense, void > & ) noexcept;

		template< typename T >
		size_t getLength( const Vector< T, reference_dense, void > & ) noexcept;

	} // end namespace ``grb::internal''

	/** \internal TODO */
	template< typename T >
	class Vector< T, reference_dense, void > {

		friend T * internal::getRaw< T >( Vector< T, reference_dense, void > & ) noexcept;
		friend const T * internal::getRaw< T >( const Vector< T, reference_dense, void > & ) noexcept;
		friend size_t internal::getLength< T >( const Vector< T, reference_dense, void > & ) noexcept;

		private:

			/** The length of the vector. */
			size_t n;

			/** The vector data. */
			T *__restrict__ data;

			/** Whether the container presently is uninitialized. */
			bool initialized;


		public:

			/** Exposes the element type. */
			typedef T value_type;

			/** The return type of #operator[](). */
			typedef T& lambda_reference;

			/**
			 * @param[in] length The requested vector length.
			 *
			 * \internal Allocates a single array of size \a length.
			 */
			Vector( const size_t length ) : n( length ), initialized( false ) {
				// TODO: Implement allocation properly
				if( n > 0) {
					data = new (std::nothrow) T[ n ];
				} else {
					data = nullptr;
				}

				if ( n > 0 && data == nullptr ) {
					throw std::runtime_error( "Could not allocate memory during grb::Vector<reference_dense> construction." );
				}
			}

			// /** \internal Makes a deep copy of \a other. */
			// Vector( const Vector< T, reference_dense, void > &other ) : Vector( other.n ) {
			// 	initialized = false;
			// 	const RC rc = set( *this, other ); // note: initialized will be set as part of this call
			// 	if( rc != SUCCESS ) {
			// 		throw std::runtime_error( "grb::Vector< T, reference_dense > (copy constructor): error during call to grb::set (" + toString( rc ) + ")" );
			// 	}
			// }

			// /** \internal No implementation notes. */
			// Vector( Vector< T, reference_dense, void > &&other ) {
			// 	n = other.n; other.n = 0;
			// 	data = other.data; other.data = 0;
			// 	data_deleter = std::move( other.data_deleter );
			// 	initialized = other.initialized; other.initialized = false;
			// }

			/** \internal No implementation notes. */
			~Vector() {
				delete [] data;
			}

			/** \internal No implementation notes. */
			lambda_reference operator[]( const size_t i ) noexcept {
				assert( i < n );
				assert( initialized );
				return data[ i ];
			}

			/** \internal No implementation notes. */
			const lambda_reference operator[]( const size_t i ) const noexcept {
				assert( i < n );
				assert( initialized );
				return data[ i ];
			}

			// /** \internal Relies on #internal::ConstDenserefVectorIterator. */
			// const_iterator cbegin() const noexcept {
			// 	return initialized ?
			// 		const_iterator( data, n, false ) :
			// 		const_iterator( nullptr, 0, false );
			// }

			// /** \internal Relies on #internal::ConstDenserefVectorIterator. */
			// const_iterator begin() const noexcept {
			// 	return cbegin();
			// }

			// /** \internal Relies on #internal::ConstDenserefVectorIterator. */
			// const_iterator cend() const noexcept {
			// 	return initialized ?
			// 		const_iterator( data, n, true ) :
			// 		const_iterator( nullptr, 0, true );
			// }

			// /** \internal Relies on #internal::ConstDenserefVectorIterator. */
			// const_iterator end() const noexcept {
			// 	return cend();
			// }

	};

	/** Identifies any reference_dense vector as an ALP vector. */
	template< typename T >
	struct is_container< Vector< T, reference_dense, void > > {
		/** A reference_vector is an ALP object. */
		static const constexpr bool value = true;
	};

	namespace internal {

		template< typename T >
		T * getRaw( Vector< T, reference_dense, void > &v ) noexcept {
			return v.data;
		}

		template< typename T >
		const T * getRaw( Vector< T, reference_dense, void > &v ) noexcept {
			return v.data;
		}

		template< typename T >
		size_t getLength( const Vector< T, reference_dense, void > &v ) noexcept {
			return v.n;
		}

	} // end namespace ``grb::internal''



	/**
	 * Here starts spec draft for vectorView
	 */


	/**
	 * Identity View over a vector container.
	 */
	template< typename T, typename View >
	size_t getLength( const VectorView< T, View, storage::Dense, reference_dense, void > &v ) noexcept {
		return v._length();
	}

	template< typename T>
	class VectorView< T, view::Identity< void >, storage::Dense, reference_dense, void > {

	private:
		/*********************
		    Storage info friends
		******************** */

		using self_type = VectorView< T, view::Identity< void >, storage::Dense, reference_dense, void >;

		friend size_t getLength<>( const self_type & ) noexcept;

		// Physical layout
		std::unique_ptr< Vector< T, reference_dense, void > > v;

		std::shared_ptr<imf::IMF> imf;

		/** Whether the container presently is initialized or not. */
		bool initialized;

		size_t _length() const {
			return imf->n;
		}

	public:
		using value_type = T;

		VectorView( const size_t length ) : v( std::make_unique< Vector< T, reference_dense, void > >( length ) ), imf( std::make_shared< imf::Id >( length ) ), initialized( false ) {}

	}; // class VectorView with physical container

	/** Identifies any reference_dense vector as an ALP vector. */
	template< typename T, typename View, typename Storage >
	struct is_container< VectorView< T, View, Storage, reference_dense, void > > {
		/** A reference_vector is an ALP object. */
		static const constexpr bool value = true;
	};

	/**
	 * Vector view of a vector only via \a view::Identity of another VectorView.
	 */
	template< typename T, typename VectorViewT >
	class VectorView< T, view::Identity< VectorViewT >, storage::Dense, reference_dense, void > {

	private:
		using self_type = VectorView< T, view::Identity< VectorViewT >, storage::Dense, reference_dense, void >;
		using target_type = VectorViewT;

		/*********************
		    Storage info friends
		******************** */

		friend size_t getLength<>( const self_type & ) noexcept;

		std::shared_ptr< target_type > ref;

		std::shared_ptr<imf::IMF> imf;

		size_t _length() const {
			return imf->n;
		}

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;

		VectorView( target_type & vec_view ) : ref( &vec_view ), imf( nullptr ) {
			
			imf = std::make_shared< imf::Id >( getLength( *ref ) );

		}

		VectorView( target_type & vec_view, std::shared_ptr< imf::IMF > imf ) : ref( & vec_view ), imf( imf ) {
			if( getLength( vec_view ) != imf->N ) {
				throw std::length_error( "VectorView(vec_view, * imf): IMF range differs from target's vector length." );
			}
		}

	}; // Identity VectorView


	/**
	 * Diagonal Vector View of a structured matrix.
	 */
	template< typename T, typename StructuredMatrixT >
	class VectorView< T, view::Diagonal< StructuredMatrixT >, storage::Dense, reference_dense, void > {

	private:
		using self_type = VectorView< T, view::Diagonal< StructuredMatrixT >, storage::Dense, reference_dense, void >;
		using target_type = StructuredMatrixT;

		/*********************
		    Storage info friends
		******************** */

		friend size_t getLength<>( const self_type & ) noexcept;

		std::shared_ptr< target_type > ref;

		std::shared_ptr<imf::IMF> imf;

		size_t _length() const {
			return imf->n;
		}

	public:
		/** Exposes the element type and the structure. */
		using value_type = T;

		VectorView( target_type & struct_mat ) : ref( &struct_mat ), imf( nullptr ) {
			
			size_t _length = view::Diagonal< target_type >::getLength( dims( *ref ) );
			imf = std::make_shared< imf::Id >( _length  );

		}

	}; // Diagonal Vector view

	template< typename StructuredMatrixT >
	VectorView< typename StructuredMatrixT::value_type, view::Diagonal< StructuredMatrixT >, storage::Dense, reference_dense, void >
	diagonal( StructuredMatrixT &smat ) {

		VectorView< typename StructuredMatrixT::value_type, view::Diagonal< StructuredMatrixT >, storage::Dense, reference_dense, void > smat_diag( smat );

		return smat_diag;
	}

	/**
	 * Generate an identity view of a VectorView.
	 */
	template< typename T, typename View, typename StorageSchemeType, enum Backend backend >
	VectorView< T, view::Identity< VectorView< T, View, StorageSchemeType, backend, void > >, StorageSchemeType, backend, void > 
	get_view( VectorView< T, View, StorageSchemeType, backend, void > &source ) {

		VectorView< T, view::Identity< VectorView< T, View, StorageSchemeType, backend, void > >, StorageSchemeType, backend, void > vec_view( source );

		return vec_view;
	}

	/**
	 * Implement a gather through a View over compatible Structure using provided Index Mapping Functions.
	 */

	template< typename T, typename View, typename StorageSchemeType, enum Backend backend >
	VectorView< T, view::Identity< VectorView< T, View, StorageSchemeType, backend, void > >, StorageSchemeType, backend, void > 
	get_view( VectorView< T, View, StorageSchemeType, backend, void > &source, std::shared_ptr< imf::IMF > imf ) {

		VectorView< T, view::Identity< VectorView< T, View, StorageSchemeType, backend, void > >, StorageSchemeType, backend, void > vec_view( source, imf );

		return vec_view;
	}



} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_VECTOR''

