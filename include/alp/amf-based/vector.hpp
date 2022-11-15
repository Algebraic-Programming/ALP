
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

#ifndef _H_ALP_AMF_BASED_VECTOR
#define _H_ALP_AMF_BASED_VECTOR


#include <stdexcept>
#include <memory>

#include <assert.h>

#include <alp/rc.hpp>
#include <alp/backends.hpp>

#include <alp/imf.hpp>
#include <alp/matrix.hpp>
#include <alp/density.hpp>
#include <alp/storage.hpp>
#include <alp/views.hpp>

#include <alp/base/vector.hpp>


namespace alp {

	namespace internal {

		template< typename T, typename Structure, typename View, typename ImfR, typename ImfC, enum Backend backend >
		size_t getLength( const alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend > &v ) noexcept {
			return v._length();
		}

		template< typename T, typename Structure, typename View, typename ImfR, typename ImfC, enum Backend backend >
		bool getInitialized( const alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend > &v ) noexcept {
			return getInitialized( static_cast< const typename alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend >::base_type & >( v ) );
		}

		template< typename T, typename Structure, typename View, typename ImfR, typename ImfC, enum Backend backend >
		void setInitialized( alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend > & v, bool initialized ) noexcept {
			setInitialized( static_cast< typename alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend >::base_type &>( v ), initialized );
		}

		template< typename T, typename Structure, typename View, typename ImfR, typename ImfC, enum Backend backend >
		typename alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend >::iterator
		begin( alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend > & v ) noexcept;

		template< typename T, typename Structure, typename View, typename ImfR, typename ImfC, enum Backend backend >
		typename alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend >::iterator
		end( alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend > & v ) noexcept;

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
	 * See specialization \a Vector< T, Structure, Density::Dense, view::Diagonal< MatrixT >, backend >
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
	template< typename T, typename Structure, typename View, typename ImfR, typename ImfC, enum Backend backend >
	class Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend > { };

	/*
	 * ALP vector with a general structure
	 */
	template< typename T, typename View, typename ImfR, typename ImfC, enum Backend backend >
	class Vector< T, structures::General, Density::Dense, View, ImfR, ImfC, backend > :
		public Matrix< T, structures::General, Density::Dense, View, ImfR, ImfC, backend > {

		public:

			typedef Vector< T, structures::General, Density::Dense, View, ImfR, ImfC, backend > self_type;
			typedef Matrix< T, structures::General, Density::Dense, View, ImfR, ImfC, backend > base_type;

		private:
			class VectorIterator: 
				public std::iterator< std::random_access_iterator_tag, T > {
				
				friend VectorIterator internal::begin<>( self_type & v ) noexcept;
				friend VectorIterator internal::end<>( self_type & v ) noexcept;

				private:
					
					typedef typename self_type::storage_index_type index_type;

					self_type * vec;
					index_type position;

					VectorIterator( self_type * v ) noexcept : 
						vec( v ), position( 0 ) 
					{}

					VectorIterator( self_type * v, index_type pos ) noexcept : 
						vec( v ), position( pos ) 
					{}

					bool equal( const VectorIterator & other ) const noexcept {
						return ( vec == other.vec ) && ( position == other.position );
					}

					bool lessThen( const VectorIterator & other ) const noexcept {
						return ( vec == other.vec ) && ( position < other.position );
					}

				public:
					typedef typename std::iterator<std::random_access_iterator_tag, T>::pointer pointer;
					typedef typename std::iterator<std::random_access_iterator_tag, T>::reference reference;
					typedef typename std::iterator<std::random_access_iterator_tag, T>::difference_type difference_type;

					/** Default constructor. */
					VectorIterator() noexcept :
						vec( nullptr ), position( 0 )
					{}

					/** Copy constructor. */
					VectorIterator( const VectorIterator &other ) noexcept :
						vec( other.vec ),
						position( other.position )
					{}

					/** Move constructor. */
					VectorIterator( VectorIterator &&other ) :
						vec( nullptr ), position( 0 )
					{
						std::swap( vec, other.vec );
						std::swap( position, other.position );
					}

					/** Copy assignment. */
					VectorIterator& operator=( const VectorIterator &other ) noexcept {
						vec = other.vec;
						position = other.position;
						return *this;
					}

					/** Move assignment. */
					VectorIterator& operator=( VectorIterator &&other ) {
						vec = nullptr;
						position = 0;
						std::swap( vec, other.vec );
						std::swap( position, other.position );
						return *this;
					}

					reference operator*() const {
						return ( *vec )[ position ];
					}

					VectorIterator& operator++() {
						++position;
						return *this;
					}

					VectorIterator& operator--() {
						--position;
						return *this;
					}

					VectorIterator operator++(int) {
						return VectorIterator( vec, position++ );
					}

					VectorIterator operator--(int) {
						return VectorIterator( vec, position-- );
					}

					VectorIterator operator+(const difference_type& n) const {
						return VectorIterator( vec, ( position + n ) );
					}

					VectorIterator& operator+=(const difference_type& n) {
						position += n;
						return *this;
					}

					VectorIterator operator-(const difference_type& n) const {
						return VectorIterator( vec, ( position - n ) );
					}

					VectorIterator& operator-=(const difference_type& n) {
						position -= n;
						return *this;
					}

					reference operator[](const difference_type& n) const {
						return ( *vec )[ position + n ];
					}

					bool operator==(const VectorIterator& other) const {
						return equal( other );
					}

					bool operator!=(const VectorIterator& other) const {
						return !equal( other );
					}

					bool operator<(const VectorIterator& other) const {
						return lessThen( other );
					}

					bool operator>(const VectorIterator& other) const {
						return !( lessThen( other ) || equal( other ) );
					}

					bool operator<=(const VectorIterator& other) const {
						return lessThen( other ) || equal( other );
					}

					bool operator>=(const VectorIterator& other) const {
						return !lessThen( other );
					}

					difference_type operator+(const VectorIterator& other) const {
						assert( other.vec == vec );
						return position + other.position;
					}

					difference_type operator-(const VectorIterator& other) const {
						assert( other.vec == vec );
						return position - other.position;
					}
			};

			/*********************
				Storage info friends
			******************** */

			friend size_t internal::getLength<>( const Vector< T, structures::General, Density::Dense, View, ImfR, ImfC, backend > &v ) noexcept;

			/** Returns the length of the vector */
			size_t _length() const {
				return nrows( static_cast< const base_type & >( *this ) );
			}

		public:
			
			typedef VectorIterator iterator;

			/** @see Vector::value_type. */
			using value_type = T;

			typedef structures::General structure;

			/** @see Vector::lambda_reference */
			typedef typename std::conditional<
				internal::is_storage_based< self_type >::value,
				T &,
				T
			>::type lambda_reference;
			typedef typename std::conditional<
				internal::is_storage_based< self_type >::value,
				const T &,
				const T
			>::type const_lambda_reference;

			template < view::Views view_tag, bool d=false >
			struct view_type;

			template < bool d >
			struct view_type< view::original, d > {
				typedef Vector< T, structures::General, Density::Dense, view::Original< self_type >, imf::Id, imf::Id, backend > type;
			};

			template < bool d >
			struct view_type< view::gather, d > {
				typedef Vector< T, structures::General, Density::Dense, view::Gather< self_type >, imf::Strided, imf::Id, backend > type;
			};

			template < bool d >
			struct view_type< view::matrix, d > {
				typedef Matrix< T, structures::General, Density::Dense, view::Matrix< self_type >, imf::Id, imf::Id, backend > type;
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
			Vector( std::function< bool() > initialized, const size_t length, LambdaType lambda ) :
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
				return this->access( this->getStorageIndex( i, 0, 0, 1 ) );
			}

			/** \internal No implementation notes. */
			const_lambda_reference operator[]( const size_t i ) const noexcept {
				assert( i < _length() );
				//assert( getInitialized( *v ) );
				/** \internal \todo revise the third and fourth parameter for parallel backends */
				return this->access( this->getStorageIndex( i, 0, 0, 1 ) );
			}

	}; // class Vector with physical container

	namespace internal {

		template< typename T, typename Structure, typename View, typename ImfR, typename ImfC, enum Backend backend >
		typename alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend >::iterator
		begin( alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend > & v ) noexcept {
			return typename alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend >::iterator( &v );
		}

		template< typename T, typename Structure, typename View, typename ImfR, typename ImfC, enum Backend backend >
		typename alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend >::iterator
		end( alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend > & v ) noexcept {
			return typename alp::Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend >::iterator( &v, getLength( v ) );
		}

	} // end namespace ``alp::internal''

	/** Identifies any backend's implementation of ALP vector as an ALP vector. */
	template< typename T, typename Structure, typename View, typename ImfR, typename ImfC, enum Backend backend >
	struct is_vector< Vector< T, Structure, Density::Dense, View, ImfR, ImfC, backend > > : std::true_type {};

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

	/**
	 * Create a matrix view over a vector.
	 * The resulting matrix is a column matrix of size M x 1, where M is vector length.
	 * The function guarantees the created view is non-overlapping with other
	 * existing views only when the check can be performed in constant time.
	 *
	 * @tparam target_view   The type of the view to apply to the vector.
	 *                       Only supports value view::matrix.
	 * @tparam SourceVector  The type of the source ALP vector
	 *
	 * @param[in] source     The ALP Vector object over which the view is created.
	 *
	 */
	template<
		enum view::Views target_view,
		typename SourceVector,
		std::enable_if_t<
			is_vector< SourceVector >::value &&
			target_view == view::matrix
		> * = nullptr
	>
	typename SourceVector::template view_type< target_view >::type
	get_view( SourceVector &source ) {
		using target_t = typename SourceVector::template view_type< target_view >::type;
		return target_t( source );
	}

	namespace internal {

		/**
		 * Implement a gather through a View over compatible Structure using provided Index Mapping Functions.
		 * The compatibility depends on the TargetStructure, SourceStructure and IMFs, and is calculated during runtime.
		 */
		template<
			typename TargetStructure, typename TargetImfR,
			typename SourceVector,
			std::enable_if_t< is_vector< SourceVector >::value > * = nullptr
		>
		typename internal::new_container_type_from<
			typename SourceVector::template view_type< view::gather >::type
		>::template change_structure< TargetStructure >::_and_::
		template change_imfr< TargetImfR >::type
		get_view(
			SourceVector &source,
			TargetImfR imf_r,
			imf::Id imf_c
		) {

			//if( std::dynamic_pointer_cast< imf::Select >( imf_r ) || std::dynamic_pointer_cast< imf::Select >( imf_c ) ) {
			//	throw std::runtime_error("Cannot gather with imf::Select yet.");
			//}
			// No static check as the compatibility depends on IMF, which is a runtime level parameter
			//if( ! (TargetStructure::template isInstantiableFrom< Structure >( static_cast< TargetImfR & >( imf_r ), static_cast< TargetImfR & >( imf_c ) ) ) ) {
			if( ! (structures::isInstantiable< typename SourceVector::structure, TargetStructure >::check( imf_r, imf_c ) ) ) {
				throw std::runtime_error("Cannot gather into specified TargetStructure from provided SourceStructure and Index Mapping Functions.");
			}

			using target_vec_t = typename internal::new_container_type_from<
				typename SourceVector::template view_type< view::gather >::type
			>::template change_structure< TargetStructure >::_and_::
			template change_imfr< TargetImfR >::type;

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
	typename SourceVector::template view_type< view::gather >::type
	get_view( SourceVector &source, const utils::range& rng ) {

		return internal::get_view< typename SourceVector::structure >(
			source,
			std::move( imf::Strided( rng.count(), nrows(source), rng.start, rng.stride ) ),
			std::move( imf::Id( 1 ) )
		);
	}

	/**
	 *
	 * Generate a dynamic gather view where the type is compliant with the source Vector.
	 * Version where a selection of indices, expressed as a vector of indices,
	 * form a new view with specified target structure.
	 *
	 * @tparam TargetStructure The target structure of the new view. It should verify
	 *                         <code> alp::is_in<Structure, TargetStructure::inferred_structures> </code>.
	 * @tparam SourceVector    The type of the source ALP vector
	 * @tparam SelectVector    The type of the ALP vector defining permutation for rows
	 *
	 * @param source           The source ALP matrix
	 * @param sel              A valid permutation vector of a subset of indices
	 *
	 * @return A new gather view over the source ALP matrix.
	 *
	 */
	template<
		typename TargetStructure,
		typename SourceVector,
		typename SelectVector,
		std::enable_if_t<
			is_vector< SourceVector >::value &&
			is_vector< SelectVector >::value
		> * = nullptr
	>
	typename internal::new_container_type_from<
		typename SourceVector::template view_type< view::gather >::type
	>::template change_structure< TargetStructure >::_and_::
	template change_imfr< imf::Select >::type
	get_view(
		SourceVector &source,
		const SelectVector &sel
	) {
		return internal::get_view< TargetStructure >(
			source,
			imf::Select( size( source ), sel ),
			imf::Id( 1 )
		);
	}

} // end namespace ``alp''

#endif // end ``_H_ALP_AMF_BASED_VECTOR''

