
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

#ifndef _H_ALP_SCALAR_BASE
#define _H_ALP_SCALAR_BASE

#include <cstdlib>  //size_t
#include <stdexcept>

#include <alp/backends.hpp>
#include <alp/descriptors.hpp>
#include <alp/ops.hpp>
#include <alp/rc.hpp>

namespace alp {

	/**
	 * \brief An ALP scalar.
	 *
	 * This is an opaque data type for scalars.
	 *
	 * @tparam T         The type of the vector elements. \a T shall not
	 *                   be an ALP type.
	 * @tparam Structure One of the structures. One of possible use cases
	 *                   for a structured scalar is a random structure.
	 *                   Depending on the backend implementation, this may mean,
	 *                   for example, randomizing the scalar value on each
	 *                   interaction with the scalar.
	 *
	 * \warning Creating a alp::Scalar of other ALP types is
	 *                <em>not allowed</em>.
	 *          Passing a ALP type as template parameter will lead to
	 *          undefined behaviour.
	 *
	 */
	template< typename T, typename Structure, enum Backend backend >
	class Scalar {

		public:
			/** @see Vector::value_type. */
			typedef T value_type;

			/** @see Vector::lambda_reference */
			typedef T& lambda_reference;

			/**
			 * The default ALP scalar constructor.
			 *
			 * The constructed object will be uninitalised after successful construction.
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
			Scalar() {}

			/**
			 * The ALP scalar constructor for converting a reference to C/C++ scalar
			 * to ALP scalar.
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
			 * \warning This constructor saves the reference to the provied value.
			 *          Therefore, the changes to the container or the value will
			 *          be mirrored into each-other. For preserving the separation,
			 *          use Scalar( const T ) version.
			 *
			 */
			explicit Scalar( T &value ) {
				(void)value;
			}

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
			explicit Scalar( T value ) {
				(void)value;
			}

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
			Scalar( const Scalar &other ) noexcept {
				(void)other;
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
			Scalar( Scalar &&other ) noexcept {
				(void)other;
			}

			/**
			 * Returns a lambda reference to the value of this Scalar. The user
			 * ensures that the requested reference only corresponds to a pre-existing
			 * nonzero in this scalar, <em>or undefined behaviour will occur</em>.
			 * This addresses the sparse specialization of scalars. In the dense
			 * context, scalar is considered to have a nonzero value \em iff initialized.
			 *
			 * A lambda reference to the value of this scalar is only valid when used
			 * inside a lambda function evaluated via alp::eWiseLambda. Outside this
			 * scope the returned reference incurs undefined behaviour.
			 *
			 *
			 * \warning In parallel contexts the use of a returned lambda reference
			 *          outside the context of an eWiseLambda will incur at least one of
			 *          the following ill effects: it may
			 *            -# fail outright,
			 *            -# work on stale data,
			 *            -# work on incorrect data, or
			 *            -# incur high communication costs to guarantee correctness.
			 *          In short, such usage causes undefined behaviour. Implementers are
			 *          \em not advised to provide GAS-like functionality through this
			 *          interface, as it invites bad programming practices and bad
			 *          algorithm design decisions. This operator is instead intended to
			 *          provide for generic BLAS0-type operations only.
			 *
			 * \note    For I/O, use the iterator retrieved via cbegin() instead of
			 *          relying on a lambda_reference.
			 *
			 * @return      A lambda reference to the value of this scalar
			 *
			 * \par Example.
			 * See alp::eWiseLambda() for a practical and useful example.
			 *
			 * \warning There is no similar concept in the official GraphBLAS specs.
			 *
			 * @see lambda_reference For more details on the returned reference type.
			 * @see alp::eWiseLambda For one legal way in which to use the returned
			 *      #lambda_reference.
			 */
			lambda_reference operator*() noexcept {
#ifndef _ALP_NO_EXCEPTIONS
				assert( false ); // Requesting lambda reference of unimplemented Scalar backend.
#endif
			}

			/** Returns a constant reference to the scalar value.
			 * See the non-constant variant for further details.
			 */
			const lambda_reference operator*() const noexcept {
#ifndef _ALP_NO_EXCEPTIONS
				assert( false ); // Requesting lambda reference of unimplemented Scalar backend.
#endif
			}

	}; // class Scalar

} // namespace alp

#endif // _H_ALP_SCALAR_BASE
