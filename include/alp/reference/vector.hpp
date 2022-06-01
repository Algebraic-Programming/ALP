
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
		void setInitialized( Vector< T, reference > & v, bool initialized ) noexcept;


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
	} // end namespace ``alp::internal''

	/** Identifies any reference vector as an ALP vector. */
	template< typename T >
	struct is_container< internal::Vector< T, reference > > {
		/** A reference_vector is an ALP object. */
		static const constexpr bool value = true;
	};

	namespace internal {

		template< typename T >
		T * getRaw( Vector< T, reference > &v ) noexcept {
			return v.data;
		}

		template< typename T >
		const T * getRaw( Vector< T, reference > &v ) noexcept {
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

	template< typename T, typename Structure, typename View, typename Imf >
	size_t getLength( const Vector< T, Structure, Density::Dense, View, Imf, reference > &v ) noexcept {
		return v._length();
	}

	namespace internal {
		template< typename T, typename Structure, typename View, typename Imf >
		bool getInitialized( alp::Vector< T, Structure, Density::Dense, View, Imf, reference > & v ) noexcept {
			return getInitialized( v );
		}

		template< typename T, typename Structure, typename View, typename Imf >
		void setInitialized( alp::Vector< T, Structure, Density::Dense, View, Imf, reference > & v, bool initialized ) noexcept {
			setInitialized( v, initialized );
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
	template< typename T, typename Structure, typename View, typename Imf >
	class Vector< T, Structure, Density::Dense, View, Imf, reference > { };

	/**
	 * Original View over a vector container.
	 */
	template< typename T, typename View, typename Imf >
	class Vector< T, structures::General, Density::Dense, View, Imf, reference > :
		public Matrix< T, structures::General, Density::Dense, View, Imf, imf::Id, reference > {

		private:

			typedef Vector< T, structures::General, Density::Dense, View, Imf, reference > self_type;
			typedef Matrix< T, structures::General, Density::Dense, View, Imf, imf::Id, reference > base_type;
			typedef typename base_type::target_type target_type;

			/*********************
				Storage info friends
			******************** */

			friend size_t getLength<>( const self_type & ) noexcept;

			/** Returns the length of the vector */
			size_t _length() const {
				return this->amf.getLogicalDimensions().first;
			}


		public:

			/** @see Vector::value_type. */
			using value_type = T;

			typedef structures::General structure;

			/** @see Vector::lambda_reference */
			typedef T& lambda_reference;

			template < view::Views view_tag, typename TargetImf, bool d=false >
			struct view_type;

			template < typename TargetImf, bool d >
			struct view_type< view::original, TargetImf, d > {
				typedef Vector< T, structures::General, Density::Dense, view::Original< self_type >, TargetImf, reference > type;
			};

			Vector( const size_t length, const size_t cap = 0 ) : base_type( length, 1, cap ) {}

			/** Constructor for creating a view over a given target vector */
			template<
				typename TargetVectorType = target_type,
				typename = typename std::enable_if<
					!std::is_same< TargetVectorType, void >::value &&
					std::is_same< TargetVectorType, target_type >::value >::type
			>
			Vector( TargetVectorType &vec_view ) : base_type( vec_view ) {}

			/** Constructor for creating a view over a given target vector and
			 * applying the given index mapping function */
			template<
				typename TargetVectorType = target_type,
				typename TargetImf,
				typename = typename std::enable_if<
					!std::is_same< TargetVectorType, void >::value &&
					std::is_same< TargetVectorType, target_type >::value >::type
			>
			Vector( TargetVectorType &vec_view, TargetImf imf ) : base_type( vec_view, imf, imf::Id( 1 ) ) {
				if( getLength( vec_view ) != imf.N ) {
					throw std::length_error( "Vector(vec_view, * imf): IMF range differs from target's vector length." );
				}
			}

			/** \internal No implementation notes. */
			lambda_reference operator[]( const size_t i ) noexcept {
				assert( i < _length() );
				//assert( getInitialized( *v ) );
				return this->access( this->amf.getStorageIndex( i, 0 ) );
			}

			/** \internal No implementation notes. */
			const lambda_reference operator[]( const size_t i ) const noexcept {
				assert( i < _length() );
				//assert( getInitialized( *v ) );
				return this->access( this->amf.getStorageIndex( i, 0 ) );
			}

	}; // class Vector with physical container

	/** Identifies any reference vector as an ALP vector. */
	template< typename T, typename Structure, typename View, typename Imf >
	struct is_container< Vector< T, Structure, Density::Dense, View, Imf, reference > > {
		/** A reference_vector is an ALP object. */
		static const constexpr bool value = true;
	};

	/**
	 * @brief  Generate an original view of the input Vector. The function guarantees 
	 *         the created view is non-overlapping with other existing views only when the
	 *         check can be performed in constant time. 
	 *
	 * @tparam T         The vector's elements type
	 * @tparam Structure The structure of the source and target vector view
	 * @tparam View      The source's View type
	 * @tparam backend   The target backend
	 *
	 * @param[in] source The Vector object over which the view is created.
	 *
	 * @returns A new vector Vector object.
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
		typename T, typename Structure, enum Density density, typename View, typename Imf,
		enum Backend backend
	>
	typename Vector< T, Structure, density, View, Imf, backend >::template view_type< view::original, imf::Id >::type
	get_view( Vector< T, Structure, density, View, Imf, backend > & source ) {

		using source_vec_t = Vector< T, Structure, density, View, Imf, backend >;
		using target_vec_t = typename source_vec_t::template view_type< view::original, imf::Id >::type;

		target_vec_t vec_view( source );

		return vec_view;
	}

	/**
	 * @brief Version of get_view over vectors where a range of elements are selected to form a new view. 
	 * 		  The function guarantees the created view is non-overlapping with other existing views only when the
	 * 		  check can be performed in constant time. 
	 * 
	 * @param[in] source The Vector object over which the view is created.
	 * @param[in] rng 	 A valid range of elements
	 * 
	 * @returns          A Vector object.
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
		typename T, typename Structure, enum Density density, typename View, typename Imf,
		enum Backend backend
	>
	typename Vector< T, Structure, density, View, Imf, backend >::template view_type< view::original, imf::Strided >::type
	get_view( Vector< T, Structure, density, View, Imf, backend > &source, const utils::range& rng ) {

		auto imf_v = imf::Strided( rng.count(), getLength( source ), rng.start, rng.stride );
		typename Vector< T, Structure, density, View, Imf, backend >::template view_type< view::original, imf::Strided >::type vec_view( source, imf_v );

		return vec_view;
	}

} // end namespace ``alp''

#endif // end ``_H_ALP_REFERENCE_VECTOR''

