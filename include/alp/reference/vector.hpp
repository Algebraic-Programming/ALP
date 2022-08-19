
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

#ifndef _H_ALP_REFERENCE_VECTOR
#define _H_ALP_REFERENCE_VECTOR


#include <stdexcept>
#include <memory>

#include <assert.h>

#include <alp/rc.hpp>
#include <alp/backends.hpp>

// #include <alp/utils/alloc.hpp>
// #include <alp/utils/autodeleter.hpp>
// #include <alp/reference/vectoriterator.hpp>

#include <alp/imf.hpp>
#include <alp/matrix.hpp>
#include <alp/density.hpp>
#include <alp/storage.hpp>
#include <alp/views.hpp>

#include <alp/base/vector.hpp>


namespace alp {

	namespace internal {

		template< typename T >
		T * getRaw( Vector< T, reference > & ) noexcept;

		template< typename T >
		const T * getRaw( const Vector< T, reference > & ) noexcept;

		template< typename T >
		size_t getLength( const Vector< T, reference > & ) noexcept;

		template< typename T >
		const bool & getInitialized( const Vector< T, reference > & v ) noexcept;

		template< typename T >
		void setInitialized( Vector< T, reference > & v, const bool initialized ) noexcept;


		/**
		 * The reference implementation of the ALP/Dense vector.
		 *
		 * @tparam T The type of an element of this vector. \a T shall not be a
		 *           GraphBLAS type.
		 *
		 * \warning Creating a alp::Vector of other GraphBLAS types is
		 *                <em>not allowed</em>.
		 *          Passing a GraphBLAS type as template parameter will lead to
		 *          undefined behaviour.
		 */
		template< typename T >
		class Vector< T, reference > {

			friend T * internal::getRaw< T >( Vector< T, reference > & ) noexcept;
			friend const T * internal::getRaw< T >( const Vector< T, reference > & ) noexcept;
			friend size_t internal::getLength< T >( const Vector< T, reference > & ) noexcept;

			/* ********************
				IO friends
			   ******************** */

			friend const bool & internal::getInitialized< T >( const Vector< T, reference > & ) noexcept;

			friend void internal::setInitialized< T >( Vector< T, reference > & , bool ) noexcept;

			private:

				/** The length of the vector. */
				size_t n;

				/** The container capacity (in elements).
				 *
				 * \warning \a cap is present for compatibility with other vector specializations.
				 *          In reference backend, the number of non-zeros (i.e. capacity)
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
						throw std::runtime_error( "Could not allocate memory during alp::Vector<reference> construction." );
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
				Vector( const Vector< T, reference > &other ) : Vector( other.n, other.cap ) {
					initialized = other.initialized;
					// const RC rc = set( *this, other ); // note: initialized will be set as part of this call
					// if( rc != SUCCESS ) {
					// 	throw std::runtime_error( "alp::Vector< T, reference > (copy constructor): error during call to alp::set (" + toString( rc ) + ")" );
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
				Vector( Vector< T, reference > &&other ) : n( other.n ), cap( other.cap ), data( other.data ) {
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

		/** Identifies any reference internal vector as an internal container. */
		template< typename T >
		struct is_container< internal::Vector< T, reference > > : std::true_type {};

	} // end namespace ``alp::internal''

	namespace internal {

		template< typename T >
		T * getRaw( Vector< T, reference > &v ) noexcept {
			return v.data;
		}

		template< typename T >
		const T * getRaw( const Vector< T, reference > &v ) noexcept {
			return v.data;
		}

		template< typename T >
		size_t getLength( const Vector< T, reference > &v ) noexcept {
			return v.n;
		}

		template< typename T >
		const bool & getInitialized( const Vector< T, reference > & v ) noexcept {
			return v.initialized;
		}

		template< typename T >
		void setInitialized( Vector< T, reference > & v, bool initialized ) noexcept {
			v.initialized = initialized;
		}
	} // end namespace ``alp::internal''



	/**
	 * Here starts spec draft for vectorView
	 */

	template< typename T, typename Structure, typename View, typename ImfR, typename ImfC >
	size_t getLength( const Vector< T, Structure, Density::Dense, View, ImfR, ImfC, reference > &v ) noexcept {
		return v._length();
	}

	namespace internal {
		template< typename T, typename Structure, typename View, typename ImfR, typename ImfC >
		bool getInitialized( const alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, reference > &v ) noexcept {
			return getInitialized( static_cast< const typename alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, reference >::base_type &>( v ) );
		}

		template< typename T, typename Structure, typename View, typename ImfR, typename ImfC >
		void setInitialized( alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, reference > & v, bool initialized ) noexcept {
			setInitialized( static_cast< typename alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, reference >::base_type &>( v ), initialized );
		}
	} // end namespace ``alp::internal''

	/**
	 * \brief An ALP vector view.
	 *
	 * This is an opaque data type for vector views.
	 *
	 * A vector exposes a mathematical \em logical layout which allows to
	 * express implementation-oblivious concepts such as \em views on the vector.
	 * The logical layout of a vector view maps to a physical counterpart via
	 * a storage scheme which typically depends on the selected backend.
	 * alp::Vector may be used as an interface to such a physical layout.
	 *
	 * Views can be used to create logical \em perspectives on top of a container.
	 * For example, one may decide to refer to the part of the vector or
	 * to reference a diagonal of a matrix as a vector.
	 * See specialization \a Vector< T, Structure, Density::Dense, view::Diagonal< MatrixT >, reference >
	 * as an example of such usage.
	 *
	 * Vector View defined as views on other vectors do not instantiate a
	 * new container but refer to the one used by their targets.
	 *
	 * @tparam T         type.
	 * @tparam Structure Structure introduced to match the template
	 *                   parameter list of \a Matrix
	 * @tparam View      One of the vector views.
	 *                   All static views except for \a view::Original (via
	 *                   \a view::Original<void> cannot instantiate a new container
	 *                   and only allow to refer to a previously defined
	 *                   \a Vector.
	 *                   The \a View parameter should not be used directly
	 *                   by the user but can be set using specific member types
	 *                   appropriately defined by each Vector and
	 *                   accessible via functions.
	 *
	 */
	template< typename T, typename Structure, typename View, typename ImfR, typename ImfC >
	class Vector< T, Structure, Density::Dense, View, ImfR, ImfC, reference > { };

	/**
	 * Original View over a vector container.
	 */
	template< typename T, typename View, typename ImfR, typename ImfC >
	class Vector< T, structures::General, Density::Dense, View, ImfR, ImfC, reference > :
		public Matrix< T, structures::General, Density::Dense, View, ImfR, ImfC, reference > {

		public:

			typedef Vector< T, structures::General, Density::Dense, View, ImfR, ImfC, reference > self_type;
			typedef Matrix< T, structures::General, Density::Dense, View, ImfR, ImfC, reference > base_type;

		private:

			/*********************
				Storage info friends
			******************** */

			friend size_t getLength<>( const self_type & ) noexcept;

			/** Returns the length of the vector */
			size_t _length() const {
				return std::max( this->amf.getLogicalDimensions().first, this->amf.getLogicalDimensions().second );
			}


		public:

			/** @see Vector::value_type. */
			using value_type = T;

			typedef structures::General structure;

			/** @see Vector::lambda_reference */
			typedef T& lambda_reference;

			template < view::Views view_tag, bool d=false >
			struct view_type;

			template < bool d >
			struct view_type< view::original, d > {
				typedef Vector< T, structures::General, Density::Dense, view::Original< self_type >, imf::Id, imf::Id, reference > type;
			};

			/**
			 * Constructor for a storage-based vector that allocates storage.
			 */
			Vector( const size_t length, const size_t cap = 0 ) :
				base_type( length, 1, cap ) {
				static_assert(
					internal::is_view_over_storage< View >::value &&
					internal::requires_allocation< View >::value,
					"This constructor can only be used in storage-based allocation-requiring Vector specializations."
				);
			}

			/**
			 * Constructor for a view over another storage-based vector.
			 *
			 * @tparam SourceType  The type of the target vector.
			 *
			 */
			template<
				typename SourceType,
				std::enable_if_t<
					std::is_same< SourceType, typename View::applied_to >::value &&
					internal::is_view_over_storage< View >::value &&
					!internal::requires_allocation< View >::value
				> * = nullptr
			>
			Vector( SourceType &source_vector, ImfR imf_r, ImfC imf_c ) :
				base_type( source_vector, imf_r, imf_c ) { }

			/**
			 * Constructor for a view over another vector applying a view defined
			 * by View template parameter of the constructed vector.
			 *
			 * @tparam SourceType  The type of the target vector.
			 *
			 */
			template<
				typename SourceType,
				std::enable_if_t<
					std::is_same< SourceType, typename View::applied_to >::value &&
					internal::is_view_over_storage< View >::value &&
					!internal::requires_allocation< View >::value
				> * = nullptr
			>
			Vector( SourceType &source_vector ) :
				base_type( source_vector ) {}

			/**
			 * @deprecated
			 * Constructor for a view over another storage-based vector.
			 *
			 * @tparam SourceType  The type of the target vector.
			 *
			 */
			template<
				typename SourceType,
				typename AmfType,
				std::enable_if_t<
					std::is_same< SourceType, typename View::applied_to >::value &&
					internal::is_view_over_storage< View >::value &&
					!internal::requires_allocation< View >::value
				> * = nullptr
			>
			Vector( SourceType &source_vector, AmfType &&amf ) :
				base_type( source_vector, std::forward< AmfType >( amf ) ) {}

			/**
			 * Constructor for a functor-based vector that allocates memory.
			 *
			 * @tparam LambdaType  The type of the lambda function associated to the data.
			 *
			 */
			template<
				typename LambdaType,
				std::enable_if_t<
					std::is_same< LambdaType, typename View::applied_to >::value &&
					internal::is_view_over_functor< View >::value &&
					internal::requires_allocation< View >::value
				> * = nullptr
			>
			Vector( bool initialized, const size_t length, LambdaType lambda ) :
				base_type( initialized, length, 1, lambda ) {}

			/**
			 * Constructor for a view over another functor-based vector.
			 *
			 * @tparam SourceType  The type of the target vector.
			 *
			 */
			template<
				typename SourceType,
				std::enable_if_t<
					std::is_same< SourceType, typename View::applied_to >::value &&
					internal::is_view_over_functor< View >::value &&
					!internal::requires_allocation< View >::value
				> * = nullptr
			>
			Vector( SourceType &target_vector, ImfR imf_r, ImfC imf_c ) :
				base_type( getFunctor( target_vector ), imf_r, imf_c ) {}

			/**
			 * Constructor for a view over another functor-based vector.
			 *
			 * @tparam SourceType  The type of the target vector.
			 *
			 */
			template<
				typename SourceType,
				std::enable_if_t<
					std::is_same< SourceType, typename View::applied_to >::value &&
					internal::is_view_over_functor< View >::value &&
					!internal::requires_allocation< View >::value
				> * = nullptr
			>
			Vector( SourceType &target_vector ) :
				base_type( getFunctor( target_vector ),
					imf::Id( nrows ( target_vector ) ),
					imf::Id( 1 )
				) {

				static_assert(
					std::is_same< ImfR, imf::Id >::value &&
					std::is_same< ImfC, imf::Id >::value,
					"This constructor can only be used with Id IMFs."
				);

			}

			/** \internal No implementation notes. */
			lambda_reference operator[]( const size_t i ) noexcept {
				assert( i < _length() );
				//assert( getInitialized( *v ) );
				/** \internal \todo revise the third and fourth parameter for parallel backends */
				return this->access( this->amf.getStorageIndex( i, i, 0, 1 ) );
			}

			/** \internal No implementation notes. */
			const lambda_reference operator[]( const size_t i ) const noexcept {
				assert( i < _length() );
				//assert( getInitialized( *v ) );
				/** \internal \todo revise the third and fourth parameter for parallel backends */
				return this->access( this->amf.getStorageIndex( i, i, 0, 1 ) );
			}

	}; // class Vector with physical container

	/** Identifies any reference ALP vector as an ALP vector. */
	template< typename T, typename Structure, typename View, typename ImfR, typename ImfC >
	struct is_vector< Vector< T, Structure, Density::Dense, View, ImfR, ImfC, reference > > : std::true_type {};

	/**
	 * @brief  Generate an original view of the input Vector. The function guarantees 
	 *         the created view is non-overlapping with other existing views only when the
	 *         check can be performed in constant time. 
	 *
	 * @tparam SourceVector  The type of the source ALP vector
	 *
	 * @param[in] source     The ALP Vector object over which the view is created.
	 *
	 * @returns A new ALP Vector object.
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
		typename SourceVector,
		std::enable_if_t< is_vector< SourceVector >::value > * = nullptr
	>
	typename SourceVector::template view_type< view::original >::type
	get_view( SourceVector &source ) {

		using target_t = typename SourceVector::template view_type< view::original >::type;

		return target_t( source );
	}

	namespace internal {

		/**
		 * Implement a gather through a View over compatible Structure using provided Index Mapping Functions.
		 * The compatibility depends on the TargetStructure, SourceStructure and IMFs, and is calculated during runtime.
		 */
		template<
			typename TargetStructure, typename TargetImfR, typename TargetImfC,
			typename SourceVector,
			std::enable_if_t< is_vector< SourceVector >::value > * = nullptr
		>
		typename internal::new_container_type_from<
			typename SourceVector::template view_type< view::original >::type
		>::template change_structure< TargetStructure >::_and_::
		template change_imfr< TargetImfR >::_and_::
		template change_imfc< TargetImfC >::type
		get_view(
			SourceVector &source,
			TargetImfR imf_r,
			TargetImfC imf_c
		) {

			//if( std::dynamic_pointer_cast< imf::Select >( imf_r ) || std::dynamic_pointer_cast< imf::Select >( imf_c ) ) {
			//	throw std::runtime_error("Cannot gather with imf::Select yet.");
			//}
			// No static check as the compatibility depends on IMF, which is a runtime level parameter
			//if( ! (TargetStructure::template isInstantiableFrom< Structure >( static_cast< TargetImfR & >( imf_r ), static_cast< TargetImfR & >( imf_c ) ) ) ) {
			if( ! (structures::isInstantiable< typename SourceVector::structure, TargetStructure >::check( static_cast< TargetImfR & >( imf_r ), static_cast< TargetImfR & >( imf_c ) ) ) ) {
				throw std::runtime_error("Cannot gather into specified TargetStructure from provided SourceStructure and Index Mapping Functions.");
			}

			using target_vec_t = typename internal::new_container_type_from<
				typename SourceVector::template view_type< view::original >::type
			>::template change_structure< TargetStructure >::_and_::
			template change_imfr< TargetImfR >::_and_::
			template change_imfc< TargetImfC >::type;

			return target_vec_t( source, imf_r, imf_c );
		}
	} // namespace internal

	/**
	 * @brief Version of get_view over vectors where a range of elements are selected to form a new view. 
	 * 		  The function guarantees the created view is non-overlapping with other existing views only when the
	 * 		  check can be performed in constant time. 
	 * 
	 * @tparam SourceVector  The type of the source ALP vector
	 *
	 * @param[in] source     The ALP Vector object over which the view is created.
	 * @param[in] rng        A valid range of elements
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
		typename SourceVector,
		std::enable_if_t< is_vector< SourceVector >::value > * = nullptr
	>
	typename internal::new_container_type_from<
		typename SourceVector::template view_type< view::original >::type
	>::template change_imfr< imf::Strided >::_and_::
	template change_imfc< imf::Strided >::type
	get_view( SourceVector &source, const utils::range& rng ) {

		return internal::get_view< typename SourceVector::structure >(
			source,
			std::move( imf::Strided( rng.count(), nrows(source), rng.start, rng.stride ) ),
			std::move( imf::Strided( rng.count(), ncols(source), rng.start, rng.stride ) )
		);
	}

} // end namespace ``alp''

#endif // end ``_H_ALP_REFERENCE_VECTOR''

