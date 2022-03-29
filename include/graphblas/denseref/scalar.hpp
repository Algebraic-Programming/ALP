
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

#ifndef _H_GRB_DENSEREF_SCALAR
#define _H_GRB_DENSEREF_SCALAR


#include <stdexcept>
#include <memory>

#include <assert.h>

#include <graphblas/rc.hpp>
#include <graphblas/backends.hpp>

#include <graphblas/density.hpp>
#include <graphblas/structures.hpp>
#include <graphblas/views.hpp>

#include <graphblas/base/scalar.hpp>

namespace grb {

	namespace internal {
		template< typename T, typename Structure >
		bool getInitialized( Scalar< T, Structure, reference_dense > & ) noexcept;

		template< typename T, typename Structure >
		void setInitialized( Scalar< T, Structure, reference_dense > &, bool ) noexcept;
	} // end namespace ``grb::internal''

	/**
	 * \brief An ALP scalar.
	 *
	 * This is an opaque data type for scalars.
	 *
	 * @tparam T                 The type of the vector elements. \a T shall not
	 *                           be a ALP type.
	 * @tparam Structure         One of the structures.
	 *
	 * \warning Creating a grb::Scalar of other ALP types is
	 *                <em>not allowed</em>.
	 *          Passing a ALP type as template parameter will lead to
	 *          undefined behaviour.
	 *
	 */
	template< typename T, typename Structure >
	class Scalar< T, Structure, reference_dense > {
		private:
			typedef Scalar< T, Structure, reference_dense > self_type;

			friend bool internal::getInitialized<>( self_type & ) noexcept;

			friend void internal::setInitialized<>( self_type &, bool ) noexcept;

			// Scalar value
			T &value;

			/** Whether the scalar value is currently initialized */
			bool initialized;

		public:
			/** @see Vector::value_type. */
			typedef T value_type;

			/** @see Vector::lambda_reference */
			typedef T& lambda_reference;

			/**
			 * The main ALP scalar constructor.
			 *
			 * The constructed object will be uninitalised after successful construction.
			 *
			 *
			 * @return SUCCESS This function never fails.
			 *
			 * \parblock
			 * \par Performance semantics.
			 *        -# This constructor entails \f$ \Theta(1) \f$ amount of work.
			 *        -# This constructor may allocate \f$ \Theta(1) \f$ bytes
			 *           of dynamic memory.
			 *        -# This constructor will use \f$ \Theta(1) \f$ extra bytes of
			 *           memory beyond that at constructor entry.
			 *        -# This constructor incurs \f$ \Theta(1) \f$ data movement.
			 *        -# This constructor \em may make system calls.
			 * \endparblock
			 *
			 */
			Scalar() : initialized( false ) {}

			/**
			 * The ALP scalar constructor for converting C/C++ scalar to ALP scalar.
			 *
			 * The constructed object will be initialized after successful construction.
			 *
			 *
			 * \parblock
			 * \par Performance semantics.
			 *        -# This constructor entails \f$ \Theta(1) \f$ amount of work.
			 *        -# This constructor may allocate \f$ \Theta(1) \f$ bytes
			 *           of dynamic memory.
			 *        -# This constructor will use \f$ \Theta(1) \f$ extra bytes of
			 *           memory beyond that at constructor entry.
			 *        -# This constructor incurs \f$ \Theta(1) \f$ data movement.
			 *        -# This constructor \em may make system calls.
			 * \endparblock
			 *
			 */
			explicit Scalar( T &value ) : value( value ), initialized( true ) {}

			/**
			 * The ALP scalar constructor for converting a C/C++ scalar to ALP scalar.
			 *
			 * The constructed object will be initialized after successful construction.
			 *
			 *
			 * \parblock
			 * \par Performance semantics.
			 *        -# This constructor entails \f$ \Theta(1) \f$ amount of work.
			 *        -# This constructor may allocate \f$ \Theta(1) \f$ bytes
			 *           of dynamic memory.
			 *        -# This constructor will use \f$ \Theta(1) \f$ extra bytes of
			 *           memory beyond that at constructor entry.
			 *        -# This constructor incurs \f$ \Theta(1) \f$ data movement.
			 *        -# This constructor \em may make system calls.
			 * \endparblock
			 *
			 */
			explicit Scalar( T value ) : value( value ), initialized( true ) {}

			/**
			 * Copy constructor.
			 *
			 * @param other The scalar to copy. The initialization state of the copy
			 *              reflects the state of \a other.
			 *
			 * \parblock
			 * \par Performance semantics.
			 *        -# This constructor entails \f$ \Theta(1) \f$ amount of work.
			 *        -# This constructor allocates \f$ \Theta(1) \f$ bytes
			 *           of dynamic memory.
			 *        -# This constructor incurs \f$ \Theta(1) \f$ of data
			 *           movement.
			 *        -# This constructor \em may make system calls.
			 * \endparblock
			 *
			 */
			Scalar( const Scalar &other ) {
				// const RC rc = set( *this, other ); // note: initialized will be set as part of this call
				// if( rc != SUCCESS ) {
				// 	throw std::runtime_error( "grb::Scalar< T, Structure, Density::Dense, View::Original< void >, reference_dense > (copy constructor): error during call to grb::set (" + toString( rc ) + ")" );
				// }
			}

			/**
			 * Move constructor. The new scalar equals the given
			 * scalar. Invalidates the use of the input scalar.
			 *
			 * @param[in] other The ALP scalar to move to this new instance.
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
			Scalar( Scalar &&other ) : value( other.value ), initialized( other.initialized ) {
				other.initialized = false;
			}

			/** \internal No implementation notes. */
			lambda_reference operator*() noexcept {
				assert( getInitialized( *this ) );
				return value;
			}

			/** \internal No implementation notes. */
			const lambda_reference operator*() const noexcept {
				assert( getInitialized( *this ) );
				return value;
			}

	}; // class Scalar with physical container

	/** Identifies any reference_dense scalar as an ALP scalar. */
	template< typename T, typename Structure >
	struct is_container< Scalar< T, Structure, reference_dense > > {
		/** A scalar is an ALP object. */
		static const constexpr bool value = true;
	};

	namespace internal {
		template< typename T, typename Structure >
		bool getInitialized( Scalar< T, Structure, reference_dense > &s ) noexcept {
			return s.initialized;
		}

		template< typename T, typename Structure >
		void setInitialized( Scalar< T, Structure, reference_dense > &s, bool initialized ) noexcept {
			s.initialized = s;
		}
	} // end namespace ``grb::internal''

} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_SCALAR''

