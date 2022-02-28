
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

		template< typename T, typename C >
		T * getRaw( Vector< T, reference_dense, C > & ) noexcept;

		template< typename T, typename C >
		const T * getRaw( const Vector< T, reference_dense, C > & ) noexcept;

		template< typename T, typename C >
		size_t getLength( const Vector< T, reference_dense, C > & ) noexcept;

	} // end namespace ``grb::internal''

	/**
	 * The reference implementation of the ALP/Dense vector.
	 *
	 * @tparam T The type of an element of this vector. \a T shall not be a
	 *           GraphBLAS type.
	 *
	 * \warning Creating a grb::Vector of other GraphBLAS types is
	 *                <em>not allowed</em>.
	 *          Passing a GraphBLAS type as template parameter will lead to
	 *          undefined behaviour.
	 */
	template< typename T, typename C >
	class Vector< T, reference_dense, C > {

		friend T * internal::getRaw< T >( Vector< T, reference_dense, C > & ) noexcept;
		friend const T * internal::getRaw< T >( const Vector< T, reference_dense, C > & ) noexcept;
		friend size_t internal::getLength< T >( const Vector< T, reference_dense, C > & ) noexcept;

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
	template< typename T, typename C >
	struct is_container< Vector< T, reference_dense, C > > {
		/** A reference_vector is an ALP object. */
		static const constexpr bool value = true;
	};

	namespace internal {

		template< typename T, typename C >
		T * getRaw( Vector< T, reference_dense, C > &v ) noexcept {
			return v.data;
		}

		template< typename T, typename C >
		const T * getRaw( Vector< T, reference_dense, C > &v ) noexcept {
			return v.data;
		}

		template< typename T, typename C >
		size_t getLength( const Vector< T, reference_dense, C > &v ) noexcept {
			return v.n;
		}

	} // end namespace ``grb::internal''



	/**
	 * Here starts spec draft for vectorView
	 */

	template< typename T, typename View, typename C, bool tmp >
	size_t getLength( const VectorView< T, View, storage::Dense, reference_dense, C, tmp > &v ) noexcept {
		return v._length();
	}

	/**
	 * \brief An ALP vector view.
	 *
	 * This is an opaque data type for vector views.
	 *
	 * A vector exposes a mathematical
	 * \em logical layout which allows to express implementation-oblivious concepts
	 * including the matrix structure itself and \em views on the matrix.
	 * The logical layout of a vector view maps to a physical counterpart via
	 * a storage scheme which typically depends on the selected backend.
	 * grb::Vector may be used as an interface to such a physical layout.
	 *
	 * Views can be used to create logical \em perspectives on top of a container.
	 * For example, one may decide to refer to the part of the vector or
	 * to reference a diagonal of a matrix as a vector.
	 * See specialization \a VectorView< T, view::Diagonal< StructuredMatrixT >, storage::Dense, reference_dense, C, tmp >
	 * as an example of such usage.
	 *
	 * Vector View defined as views on other vectors do not instantiate a
	 * new container but refer to the one used by their targets.
	 *
	 * Finally, a vector view can be declared as temporary, in which case the ALP
	 * framework has the freedom to decide whether a container should be allocated in practice
	 * or not. For example, a JIT backend may optimize away the use of such vector which
	 * would make memory allocation for such vector unnecessary.
	 *
	 * @tparam T				 The type of the vector elements. \a T shall not be a GraphBLAS
	 *              			 type.
	 * @tparam View  			 One of the vector views.
	 * 		   					 All static views except for \a view::Identity (via
	 *         					 \a view::identity<void> cannot instantiate a new container
	 * 							 and only allow to refer to a previously defined
	 * 							 \a VectorView.
	 *         					 The \a View parameter should not be used directly
	 * 							 by the user but can be set using specific member types
	 * 							 appropriately defined by each VectorView and
	 * 							 accessible via functions.
	 * @tparam StorageSchemeType Either \em enum \a storage::Dense or \em enum
	 * 	                         \a storage::Sparse.
	 * 		   					 \a VectorView will be allowed to pick storage schemes
	 *         					 defined within their specified \a StorageSchemeType.
	 * @tparam tmp  			 Whether the vector view is temporary. If \a true and the
	 * 							 and the vector view can instantiate a physical container
	 * 							 (i.e., it is defined with \a View = \a view::identity<void>)
	 * 							 then the framework may or may not decide to actually allocate
	 * 							 memory for such vector.
	 *
	 */
	template< typename T, typename View, typename C, bool tmp >
	class VectorView< T, View, storage::Dense, reference_dense, C, tmp > { };

	/**
	 * Identity View over a vector container.
	 */
	template< typename T, typename C, bool tmp >
	class VectorView< T, view::Identity< void >, storage::Dense, reference_dense, C, tmp > {

	private:
		using self_type = VectorView< T, view::Identity< void >, storage::Dense, reference_dense, C, tmp >;

		/*********************
		    Storage info friends
		******************** */

		friend size_t getLength<>( const self_type & ) noexcept;

		// Physical layout
		std::unique_ptr< Vector< T, reference_dense, C > > v;

		std::shared_ptr<imf::IMF> imf;

		/** Whether the container presently is initialized or not. Currently,
		 * container is created as uninitialized and set to initialized in a
		 * buildMatrix call.
		 */
		bool initialized;

		/** Returns the length of the vector */
		size_t _length() const {
			return imf->n;
		}

	public:
		/** @see Vector::value_type. */
		using value_type = T;

		/**
		 * @brief Any vector view type which allocates a physical container provides a version of 
		 * itself for defining a temporary vector of the same type via the \a tmp_t member type.
		 */
		using tmp_t = VectorView< T, view::Identity< void >, storage::Dense, reference_dense, C, true >;

		VectorView( const size_t length ) : v( std::make_unique< Vector< T, reference_dense, C > >( length ) ), imf( std::make_shared< imf::Id >( length ) ), initialized( false ) {}

	}; // class VectorView with physical container

	/** Identifies any reference_dense vector as an ALP vector. */
	template< typename T, typename View, typename Storage, typename C, bool tmp >
	struct is_container< VectorView< T, View, Storage, reference_dense, C, tmp > > {
		/** A reference_vector is an ALP object. */
		static const constexpr bool value = true;
	};

	/**
	 * Vector view of a vector only via \a view::Identity of another VectorView.
	 */
	template< typename T, typename VectorViewT, typename C, bool tmp >
	class VectorView< T, view::Identity< VectorViewT >, storage::Dense, reference_dense, C, tmp > {

	private:
		using self_type = VectorView< T, view::Identity< VectorViewT >, storage::Dense, reference_dense, C, tmp >;
		using target_type = VectorViewT;

		/*********************
		    Storage info friends
		******************** */

		friend size_t getLength<>( const self_type & ) noexcept;

		/** Reference to a target vector to which this vector view points to */
		std::shared_ptr< target_type > ref;

		/** Index-mapping function. @see IMF */
		std::shared_ptr<imf::IMF> imf;

		size_t _length() const {
			return imf->n;
		}

	public:
		/** Exposes the element type. */
		using value_type = T;

		/** Constructor for creating a view over a given target vector */
		VectorView( target_type & vec_view ) : ref( &vec_view ), imf( nullptr ) {
			
			imf = std::make_shared< imf::Id >( getLength( *ref ) );

		}

		/** Constructor for creating a view over a given target vector and
		 * applying the given index mapping function */
		VectorView( target_type & vec_view, std::shared_ptr< imf::IMF > imf ) : ref( & vec_view ), imf( imf ) {
			if( getLength( vec_view ) != imf->N ) {
				throw std::length_error( "VectorView(vec_view, * imf): IMF range differs from target's vector length." );
			}
		}

	}; // Identity VectorView


	/**
	 * Diagonal Vector View of a structured matrix.
	 */
	template< typename T, typename StructuredMatrixT, typename C, bool tmp >
	class VectorView< T, view::Diagonal< StructuredMatrixT >, storage::Dense, reference_dense, C, tmp > {

	private:
		/** Exposes the own type and the type of the VectorView object over
		 * which this view is created. */
		using self_type = VectorView< T, view::Diagonal< StructuredMatrixT >, storage::Dense, reference_dense, C, tmp >;
		using target_type = StructuredMatrixT;

		/*********************
		    Storage info friends
		******************** */

		friend size_t getLength<>( const self_type & ) noexcept;

		/** Pointer to a VectorView object upon which this view is created */
		std::shared_ptr< target_type > ref;

		/** @see IMF */
		std::shared_ptr<imf::IMF> imf;

		size_t _length() const {
			return imf->n;
		}

	public:
		/** Exposes the element type. */
		using value_type = T;

		VectorView( target_type & struct_mat ) : ref( &struct_mat ), imf( nullptr ) {
			
			size_t _length = view::Diagonal< target_type >::getLength( dims( *ref ) );
			imf = std::make_shared< imf::Id >( _length  );

		}

	}; // Diagonal Vector view

	/** Creates a Diagonal Vector View over a given \a StructuredMatrix
	 * 
	 * @paramt StructuredMatrixT    Type of the StructuredMatrix over which the
	 *                              view is created.
	 * @paramt C                    @see Coordinates
	 * @param[in] smat              StructuredMatrix object over which the view
	 *                              is created.
	 * 
	 * @returns                     A VectorView object.
	 * */
	template< typename StructuredMatrixT, typename C = internal::DefaultCoordinates, bool tmp = false >
	VectorView< typename StructuredMatrixT::value_type, view::Diagonal< StructuredMatrixT >, storage::Dense, reference_dense, C, tmp >
	diagonal( StructuredMatrixT &smat ) {

		VectorView< typename StructuredMatrixT::value_type, view::Diagonal< StructuredMatrixT >, storage::Dense, reference_dense, C, tmp > smat_diag( smat );

		return smat_diag;
	}

	/**
	 * Generate an identity view of a VectorView.
	 * 
	 * @param[in] source The VectorView object over which the view is created.
	 * 
	 * @returns          A VectorView object.
	 */
	template< typename T, typename View, typename StorageSchemeType, enum Backend backend, typename C = internal::DefaultCoordinates, bool tmp = false >
	VectorView< T, view::Identity< VectorView< T, View, StorageSchemeType, backend, C, tmp > >, StorageSchemeType, backend, C, tmp >
	get_view( VectorView< T, View, StorageSchemeType, backend, C, tmp > &source ) {

		VectorView< T, view::Identity< VectorView< T, View, StorageSchemeType, backend, C, tmp > >, StorageSchemeType, backend, C, tmp > vec_view( source );

		return vec_view;
	}

	/**
	 * Implement a gather through a View over compatible Structure using
	 * provided Index Mapping Functions.
	 * 
	 * @param[in] source The VectorView object over which the view is created.
	 * @param[in] imf Index-mapping function applied to the created view.
	 * 
	 * @returns          A VectorView object.
	 */

	template< typename T, typename View, typename StorageSchemeType, enum Backend backend, typename C, bool tmp = false >
	VectorView< T, view::Identity< VectorView< T, View, StorageSchemeType, backend, C, tmp > >, StorageSchemeType, backend, C, tmp >
	get_view( VectorView< T, View, StorageSchemeType, backend, C, tmp > &source, std::shared_ptr< imf::IMF > imf ) {

		VectorView< T, view::Identity< VectorView< T, View, StorageSchemeType, backend, C, tmp > >, StorageSchemeType, backend, C, tmp > vec_view( source, imf );

		return vec_view;
	}



} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_VECTOR''

