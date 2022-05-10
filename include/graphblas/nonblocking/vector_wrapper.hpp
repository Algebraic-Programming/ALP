
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
		 * A wrapper class used to store a scalar value, which is passed by value to an
		 * internal function used by an ALP/GraphBLAS operation. The wrapper classes are used
		 * by operations that may have a formal parameter that is either a scalar or a vector,
		 * because the implementation is generic and handles all possible cases.
		 */
		template< bool scalar, typename InputType,  typename CoordinatesType >
		class Wrapper {

			private:

				InputType val;


			public:

				Wrapper(const InputType &value) : val( value ) {

				}

				Wrapper( const Wrapper< scalar, InputType, CoordinatesType > & ) =default;

				constexpr InputType * getRaw() const {
					return nullptr;
				}

				constexpr CoordinatesType * getCoordinates() const {
					return nullptr;
				}

				constexpr Vector< InputType, nonblocking, CoordinatesType > * getPointer() const {
					return nullptr;
				}

				const InputType & getValue() const {
					return val;
				}

				bool isDense() const {
					return true;
				}
		};

		/**
		 * A wrapper class used to store a vector, which is passed by reference to an
		 * internal function used by an ALP/GraphBLAS operation. The wrapper classes are used by
		 * by operations that may have a formal parameter that is either a scalar or a vector,
		 * because the implementation is generic and handles all possible cases.
		 */
		template< typename InputType,  typename CoordinatesType >
		class Wrapper< false, InputType, CoordinatesType > {

			private:

				const Vector< InputType, nonblocking, CoordinatesType > &vec;


			public:

				Wrapper(const Vector< InputType, nonblocking, CoordinatesType > &vector) : vec( vector ) {

				}

				Wrapper( const Wrapper< false, InputType, CoordinatesType > &w ) : vec( w.vec ) {

				}

				const InputType * getRaw() const {
					return internal::getRaw( vec );
				}

				const CoordinatesType * getCoordinates() const {
					return &internal::getCoordinates( vec );
				}

				const Vector< InputType, nonblocking, CoordinatesType > * getPointer() const {
					return &vec;
				}

				const InputType & getValue() const {
					// this is a trick to avoid compilation errors, this value will never be used in practice
					return *( getRaw( ) );
				}

				bool isDense() const {
					return internal::getCoordinates( vec ).isDense();
				}
		};

	} // end namespace ``internal''

} // end namespace ``grb''

#endif

