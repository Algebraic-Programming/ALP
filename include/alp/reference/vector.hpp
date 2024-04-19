
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

#include <memory>
#include <stdexcept>

#include <assert.h>

#include <alp/backends.hpp>
#include <alp/density.hpp>
#include <alp/imf.hpp>
#include <alp/rc.hpp>
#include <alp/views.hpp>

#include <alp/base/vector.hpp>
#include <alp/amf-based/vector.hpp>

#include "matrix.hpp"
#include "storage.hpp"


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
					/** \internal \todo See if the assert below makes sense in some scenarios. */
					//assert( initialized );
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

} // end namespace ``alp''

#endif // end ``_H_ALP_REFERENCE_VECTOR''

