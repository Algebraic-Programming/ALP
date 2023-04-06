
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

/**
 * @file
 *
 * Provides a wrapper to a scalar or a vector, for those primitives that could
 * take either.
 *
 * @author Aristeidis Mastoras
 * @date 24th of October, 2022
 */

#ifndef _H_GRB_NONBLOCKING_VECTOR_WRAPPER
#define _H_GRB_NONBLOCKING_VECTOR_WRAPPER

#include <graphblas/backends.hpp>
#include <graphblas/blas0.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/internalops.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/semiring.hpp>

#include "coordinates.hpp"
#include "vector.hpp"
#include "lazy_evaluation.hpp"
#include "blas1.hpp"


namespace grb {

	namespace internal {

		/**
		 * A wrapper class used to store a scalar value, which is passed by value to
		 * an internal function used by an ALP/GraphBLAS operation. The wrapper
		 * classes are used by operations that may have a formal parameter that is
		 * either a scalar or a vector, because the implementation is generic and
		 * handles all possible cases.
		 */
		template< bool scalar, typename InputType,  typename CoordinatesType >
		class Wrapper {

			private:

				/**
				 * \warning This is not a reference, since the semantics are that the
				 *          \em current scalar value is used.
				 */
				InputType val;


			public:

				/** Base constructor that copies the input scalar. */
				Wrapper(const InputType &value) : val( value ) {}

				/** Default copy constructor. */
				Wrapper( const Wrapper< scalar, InputType, CoordinatesType > & ) = default;

				/**
				 * @returns <tt>nullptr</tt>
				 *
				 * This function returns a raw array for vectors only).
				 */
				constexpr InputType * getRaw() const {
					return nullptr;
				}

				/**
				 * @returns <tt>nullptr</tt>
				 *
				 * This function returns coordinates only for vectors.
				 */
				constexpr CoordinatesType * getCoordinates() const {
					return nullptr;
				}

				/**
				 * @returns <tt>nullptr</tt>
				 *
				 * This function returns a vector pointer only when wrapping a vector.
				 */
				constexpr Vector< InputType, nonblocking, CoordinatesType > * getPointer()
					const
				{
					return nullptr;
				}

				/**
				 * @returns The scalar value it wraps.
				 */
				const InputType & getValue() const {
					return val;
				}

				/**
				 * @returns Whether the underlying container is dense.
				 */
				bool isDense() const {
					return true;
				}

		};

		/**
		 * A wrapper class used to store a vector, which is passed by reference to an
		 * internal function used by an ALP/GraphBLAS operation. The wrapper classes
		 * are used by by operations that may have a formal parameter that is either a
		 * scalar or a vector, because the implementation is generic and handles all
		 * possible cases.
		 */
		template< typename InputType,  typename CoordinatesType >
		class Wrapper< false, InputType, CoordinatesType > {

			private:

				/** A reference to the vector this class wraps. */
				const Vector< InputType, nonblocking, CoordinatesType > &vec;


			public:

				/** Base constructor wrapping arund a given \a vector. */
				Wrapper( const Vector< InputType, nonblocking, CoordinatesType > &vector ) :
					vec( vector )
				{}

				/** Copy constructor. */
				Wrapper( const Wrapper< false, InputType, CoordinatesType > &w ) :
					vec( w.vec )
				{}

				/** @returns The underlying raw value array. */
				const InputType * getRaw() const {
					return internal::getRaw( vec );
				}

				/** @returns The underlying coordinates instance. */
				const CoordinatesType * getCoordinates() const {
					return &internal::getCoordinates( vec );
				}

				/** @returns The underlying vector (a pointer to it). */
				const Vector< InputType, nonblocking, CoordinatesType > * getPointer()
					const
				{
					return &vec;
				}

				/**
				 * @returns a possibly unitialised value that is not intended to be
				 *          consumed.
				 *
				 * \warning This function should only be called on wrappers of scalars.
				 */
				const InputType & getValue() const {
					// this is a trick to avoid compilation errors, this value will never be
					// used in practice
					return *( getRaw( ) );
				}

				/**
				 * @returns Whether the underlying vector is dense.
				 */
				bool isDense() const {
					return internal::getCoordinates( vec ).isDense();
				}
		};

	} // end namespace ``internal''

} // end namespace ``grb''

#endif

