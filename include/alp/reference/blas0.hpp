
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

#ifndef _H_GRB_DENSEREF_BLAS0
#define _H_GRB_DENSEREF_BLAS0

#include <graphblas/backends.hpp>
#include <graphblas/config.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/storage.hpp>

#ifndef NO_CAST_ASSERT
#define NO_CAST_ASSERT( x, y, z )                                              \
	static_assert( x,                                                          \
		"\n\n"                                                                 \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"*     ERROR      | " y " " z ".\n"                                    \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"* Possible fix 1 | Remove no_casting from the template parameters "   \
		"in this call to " y ".\n"                                             \
		"* Possible fix 2 | Provide a value that matches the expected type.\n" \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );
#endif

namespace grb {

	/**
	 * \defgroup BLAS0 The Level-0 Basic Linear Algebra Subroutines (BLAS)
	 *
	 * A collection of functions that let GraphBLAS operators work on
	 * zero-dimensional containers, i.e., on scalars.
	 *
	 * The GraphBLAS uses opaque data types and defines several standard functions
	 * to operate on these data types. Examples types are grb::Vector and
	 * grb::Matrix, example functions are grb::dot and grb::vxm.
	 *
	 * To input data into an opaque GraphBLAS type, each opaque type defines a
	 * member function \a build: grb::Vector::build() and grb::Matrix::build().
	 *
	 * To extract data from opaque GraphBLAS types, each opaque type provides
	 * \em iterators that may be obtained via the STL standard \a begin and \a end
	 * functions:
	 *   - grb::Vector::begin or grb::Vector::cbegin
	 *   - grb::Vector::end or grb::Vector::cend
	 *   - grb::Matrix::begin or grb::Matrix::cbegin
	 *   - grb::Matrix::end or grb::Matrix::cend
	 *
	 * Some GraphBLAS functions, however, reduce all elements in a GraphBLAS
	 * container into a single element of a given type. So for instance, grb::dot
	 * on two vectors of type grb::Vector<double> using the regular real semiring
	 * grb::Semiring<double> will store its output in a variable of type \a double.
	 *
	 * When parametrising GraphBLAS functions in terms of arbitrary Semirings,
	 * Monoids, Operators, and object types, it is useful to have a way to apply
	 * the same operators on whatever type they make functions like grb::dot
	 * produce-- that is, we require functions that enable the application of
	 * GraphBLAS operators on single elements.
	 *
	 * This group of BLAS level 0 functions provides this functionality.
	 *
	 * @{
	 */

	/** Resizes the Scalar to have at least the given number of nonzeroes.
	 * The contents of the scalar are not retained.
	 *
	 * Resizing of dense containers is not allowed as the capacity is determined
	 * by the container dimensions and the storage scheme. Therefore, this
	 * function will not change the capacity of the container.
	 * 
	 * The resize function for Scalars exist to maintain compatibility with
	 * other containers (i.e., vector and matrix).
	 *
	 * Even though the capacity remains unchanged, the contents of the scalar
	 * are not retained to maintain compatibility with the general specification.
	 * However, the actual memory will not be reallocated. Rather, the scalar
	 * will be marked as uninitialized.
	 * 
	 * @param[in] x      The Scalar to be resized.
	 * @param[in] new_nz The number of nonzeroes this vector is to contain.
	 *
	 * @return SUCCESS   If \a new_nz is not larger than 1.
	 *         ILLEGAL   If \a new_nz is larger than 1.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -$ This function consitutes \f$ \Theta(1) \f$ work.
	 *        -# This function allocates \f$ \Theta(0) \f$
	 *           bytes of dynamic memory.
	 *        -# This function does not make system calls.
	 * \endparblock
	 * \todo add documentation. In particular, think about the meaning with \a P > 1.
	 */
	template< typename InputType, typename InputStructure, typename length_type >
	RC resize( Scalar< InputType, InputStructure, reference_dense > & s, const length_type new_nz ) {
		if( new_nz <= 1 ) {
			setInitialized( s, false );
			return SUCCESS;
		} else {
			return ILLEGAL;
		}
	}

	/** @} */

} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_BLAS0''

