
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
#include <graphblas/coordinates.hpp>

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

		template< typename T, typename C >
		const bool & getInitialized( const Vector< T, reference_dense, C > & v ) noexcept;

		template< typename T, typename C >
		void setInitialized( Vector< T, reference_dense, C > & v, bool initialized ) noexcept;

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

		/* ********************
		        IO friends
		   ******************** */

		friend const bool & internal::getInitialized< T >( const Vector< T, reference_dense, C > & ) noexcept;

		friend void internal::setInitialized< T >( Vector< T, reference_dense, C > & , bool ) noexcept;

		private:

			/** The length of the vector. */
			size_t n;

			/** The container capacity (in elements).
			 *
			 * \warning \a cap is present for compatibility with other vector specializations.
			 *          In reference_dense backend, the number of non-zeros (i.e. capacity)
			 *          depends on the used storage scheme. Therefore, this parameter is
			 *          ignored when provided by user.
			*/
			size_t cap;

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
			 * The main ALP/Dense vector constructor.
			 *
			 * The constructed object will be uninitalised after successful construction.
			 *
			 *
			 * @param length      The number of elements in the new vector.
			 *
			 * @return SUCCESS This function never fails.
			 *
			 * \parblock
			 * \par Performance semantics.
			 *        -# This constructor entails \f$ \Theta(1) \f$ amount of work.
			 *        -# This constructor may allocate \f$ \Theta( length ) \f$ bytes
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
			Vector( const size_t length, const size_t cap = 0 ) : n( length ), cap( std::max( length, cap ) ), initialized( false ) {
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

			/**
			 * Copy constructor.
			 *
			 * @param other The vector to copy. The initialization state of the copy
			 *              reflects the state of \a other.
			 *
			 * \parblock
			 * \par Performance semantics.
			 *      Allocates the same capacity as the \a other vector, even if the
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
			Vector( const Vector< T, reference_dense, void > &other ) : Vector( other.n, other.cap ) {
				initialized = other.initialized;
				// const RC rc = set( *this, other ); // note: initialized will be set as part of this call
				// if( rc != SUCCESS ) {
				// 	throw std::runtime_error( "grb::Vector< T, reference_dense > (copy constructor): error during call to grb::set (" + toString( rc ) + ")" );
				// }
			}

			/**
			 * Move constructor. The new vector equal the given
			 * vector. Invalidates the use of the input vector.
			 *
			 * @param[in] other The GraphBLAS vector to move to this new instance.
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
			Vector( Vector< T, reference_dense, void > &&other ) : n( other.n ), cap( other.cap ), data( other.data ) {
				other.n = 0;
				other.cap = 0;
				other.data = 0;
				// data_deleter = std::move( other.data_deleter );
				// initialized = other.initialized; other.initialized = false;
			}

			/**
			 * Vector destructor.
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
			~Vector() {
				if( data != nullptr ) {
					delete [] data;
				}
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

		template< typename T, typename C >
		const bool & getInitialized( const Vector< T, reference_dense, C > & v ) noexcept {
			return v.initialized;
		}

		template< typename T, typename C >
		void setInitialized( Vector< T, reference_dense, C > & v, bool initialized ) noexcept {
			v.initialized = initialized;
		}
	} // end namespace ``grb::internal''



	/**
	 * Here starts spec draft for vectorView
	 */

	template< typename T, typename Structure, typename View >
	size_t getLength( const VectorView< T, Structure, storage::Dense, View, reference_dense > &v ) noexcept {
		return v._length();
	}

	namespace internal {
		template< typename T, typename Structure, typename View >
		bool getInitialized( VectorView< T, Structure, storage::Dense, View, reference_dense > & v ) noexcept {
			return getInitialized( v );
		}

		template< typename T, typename Structure, typename View >
		void setInitialized( VectorView< T, Structure, storage::Dense, View, reference_dense > & v, bool initialized ) noexcept {
			setInitialized( v, initialized );
		}
	} // end namespace ``grb::internal''

	/**
	 * \brief An ALP vector view.
	 *
	 * This is an opaque data type for vector views.
	 *
	 * A vector exposes a mathematical \em logical layout which allows to
	 * express implementation-oblivious concepts such as \em views on the vector.
	 * The logical layout of a vector view maps to a physical counterpart via
	 * a storage scheme which typically depends on the selected backend.
	 * grb::Vector may be used as an interface to such a physical layout.
	 *
	 * Views can be used to create logical \em perspectives on top of a container.
	 * For example, one may decide to refer to the part of the vector or
	 * to reference a diagonal of a matrix as a vector.
	 * See specialization \a VectorView< T, Structure, storage::Dense, view::Diagonal< StructuredMatrixT >, reference_dense >
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
	 * @tparam Structure		 Structure introduced to match the template
	 * 							 parameter list of \a StructuredMatrix
	 * @tparam View  			 One of the vector views.
	 * 		   					 All static views except for \a view::Original (via
	 *         					 \a view::Original<void> cannot instantiate a new container
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
	 *
	 */
	template< typename T, typename Structure, typename View >
	class VectorView< T, Structure, storage::Dense, View, reference_dense > { };

	/**
	 * Original View over a vector container.
	 */
	template< typename T, typename Structure >
	class VectorView< T, Structure, storage::Dense, view::Original< void >, reference_dense > {

	private:
		using self_type = VectorView< T, Structure, storage::Dense, view::Original< void >, reference_dense >;

		/*********************
		    Storage info friends
		******************** */

		friend size_t getLength<>( const self_type & ) noexcept;

		// Physical layout
		std::unique_ptr< Vector< T, reference_dense, internal::DefaultCoordinates > > v;

		std::shared_ptr<imf::IMF> imf;

		/** Returns the length of the vector */
		size_t _length() const {
			return imf->n;
		}

	public:
		/** @see Vector::value_type. */
		using value_type = T;

		VectorView( const size_t length, const size_t cap = 0 ) :
			v( std::make_unique< Vector< T, reference_dense, internal::DefaultCoordinates > >( length, cap ) ),
			imf( std::make_shared< imf::Id >( length ) ) {}

	}; // class VectorView with physical container

	/** Identifies any reference_dense vector as an ALP vector. */
	template< typename T, typename Structure, typename Storage, typename View >
	struct is_container< VectorView< T, Structure, Storage, View, reference_dense > > {
		/** A reference_vector is an ALP object. */
		static const constexpr bool value = true;
	};

	/**
	 * Vector view of a vector only via \a view::Original of another VectorView.
	 */
	template< typename T, typename Structure, typename VectorViewT >
	class VectorView< T, Structure, storage::Dense, view::Original< VectorViewT >, reference_dense > {

	private:
		using self_type = VectorView< T, Structure, storage::Dense, view::Original< VectorViewT >, reference_dense >;
		using target_type = VectorViewT;

		/*********************
		    Storage info friends
		******************** */

		friend size_t getLength<>( const self_type & ) noexcept;

		/** Reference to a target vector to which this vector view points to */
		// std::shared_ptr< target_type > ref;
		target_type & ref;

		/** Index-mapping function. @see IMF */
		std::shared_ptr<imf::IMF> imf;

		size_t _length() const {
			return imf->n;
		}

	public:
		/** Exposes the element type. */
		using value_type = T;

		/** Constructor for creating a view over a given target vector */
		VectorView( target_type & vec_view ) : ref( vec_view ), imf( nullptr ) {
			
			imf = std::make_shared< imf::Id >( getLength( ref ) );

		}

		/** Constructor for creating a view over a given target vector and
		 * applying the given index mapping function */
		VectorView( target_type & vec_view, std::shared_ptr< imf::IMF > imf ) : ref( vec_view ), imf( imf ) {
			if( getLength( vec_view ) != imf->N ) {
				throw std::length_error( "VectorView(vec_view, * imf): IMF range differs from target's vector length." );
			}
		}

	}; // Original VectorView


	/**
	 * Diagonal Vector View of a structured matrix.
	 */
	template< typename T, typename Structure, typename StructuredMatrixT >
	class VectorView< T, Structure, storage::Dense, view::Diagonal< StructuredMatrixT >, reference_dense > {

	private:
		/** Exposes the own type and the type of the VectorView object over
		 * which this view is created. */
		using self_type = VectorView< T, Structure, storage::Dense, view::Diagonal< StructuredMatrixT >, reference_dense >;
		using target_type = StructuredMatrixT;

		/*********************
		    Storage info friends
		******************** */

		friend size_t getLength<>( const self_type & ) noexcept;

		/** Pointer to a VectorView object upon which this view is created */
		target_type & ref;

		/** @see IMF */
		std::shared_ptr<imf::IMF> imf;

		size_t _length() const {
			return imf->n;
		}

	public:
		/** Exposes the element type. */
		using value_type = T;

		VectorView( target_type & struct_mat ) : ref( struct_mat ), imf( nullptr ) {
			
			size_t _length = view::Diagonal< target_type >::getLength( dims( ref ) );
			imf = std::make_shared< imf::Id >( _length  );

		}

	}; // Diagonal Vector view

	/** Creates a Diagonal Vector View over a given \a StructuredMatrix
	 * 
	 * @paramt StructuredMatrixT    Type of the StructuredMatrix over which the
	 *                              view is created.
	 * @param[in] smat              StructuredMatrix object over which the view
	 *                              is created.
	 * 
	 * @returns                     A VectorView object.
	 * */
	template< typename StructuredMatrixT >
	VectorView< typename StructuredMatrixT::value_type, typename StructuredMatrixT::structure, storage::Dense, view::Diagonal< StructuredMatrixT >, reference_dense >
	diagonal( StructuredMatrixT &smat ) {

		VectorView< typename StructuredMatrixT::value_type, typename StructuredMatrixT::structure, storage::Dense, view::Diagonal< StructuredMatrixT >, reference_dense > smat_diag( smat );

		return smat_diag;
	}

	/**
	 * Generate an original view of a VectorView.
	 * 
	 * @param[in] source The VectorView object over which the view is created.
	 * 
	 * @returns          A VectorView object.
	 */
	template< typename T, typename Structure, typename View, typename StorageSchemeType, enum Backend backend >
	VectorView< T, Structure, StorageSchemeType, view::Original< VectorView< T, Structure, StorageSchemeType, View, backend > >, backend > 
	get_view( VectorView< T, Structure, View, StorageSchemeType, backend > &source ) {

		VectorView< T, Structure, StorageSchemeType, view::Original< VectorView< T, Structure, View, StorageSchemeType, backend > >, backend > vec_view( source );

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

	template< typename T, typename Structure, typename View, typename StorageSchemeType, enum Backend backend >
	VectorView< T, Structure, StorageSchemeType, view::Original< VectorView< T, Structure, StorageSchemeType, View, backend > >, backend > 
	get_view( VectorView< T, Structure, StorageSchemeType, View, backend > &source, std::shared_ptr< imf::IMF > imf ) {

		VectorView< T, Structure, StorageSchemeType, view::Original< VectorView< T, Structure, StorageSchemeType, View, backend > >, backend > vec_view( source, imf );

		return vec_view;
	}



} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_VECTOR''

