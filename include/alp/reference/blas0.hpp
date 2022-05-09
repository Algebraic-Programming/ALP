
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

#ifndef _H_ALP_REFERENCE_BLAS0
#define _H_ALP_REFERENCE_BLAS0

#include <type_traits> // std::enable_if, std::is_same

#include <alp/base/blas0.hpp>
#include <alp/backends.hpp>
#include <alp/rc.hpp>
#include <alp/descriptors.hpp>
#include <alp/type_traits.hpp>
#include <alp/scalar.hpp>

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

namespace alp {

	namespace internal {

		/**
		 * @internal apply \a op to internal scalar container.
		 */
		template< 
			Descriptor descr = descriptors::no_operation,
			class OP,
			typename InputType1, typename InputType2, typename OutputType
		>
		RC apply( OutputType &out,
			const InputType1 &x,
			const InputType2 &y,
			const OP &op = OP(),
			const typename std::enable_if<
				is_operator< OP >::value &&
				!is_object< InputType1 >::value &&
				!is_object< InputType2 >::value &&
				!is_object< OutputType >::value,
			void >::type * = NULL
		) {
			// static sanity check
			NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || (
					std::is_same< InputType1, typename OP::D1 >::value &&
					std::is_same< InputType2, typename OP::D2 >::value &&
					std::is_same< OutputType, typename OP::D3 >::value
				) ),
				"alp::internal::apply (level 0)",
				"Argument value types do not match operator domains while no_casting "
				"descriptor was set"
			);

			// call apply
			const typename OP::D1 left = static_cast< typename OP::D1 >( x );
			const typename OP::D2 right = static_cast< typename OP::D2 >( y );
			typename OP::D3 output = static_cast< typename OP::D3 >( out );
			op.apply( left, right, output );
			out = static_cast< OutputType >( output );

			// done
			return SUCCESS;
		}

		/**
		 * @internal \a foldr reference implementation on internal scalar container.
		 */
		template< 
			Descriptor descr = descriptors::no_operation, 
			class OP, typename InputType, typename IOType >
		RC foldr( const InputType & x,
			IOType & y,
			const OP & op = OP(),
			const typename std::enable_if< is_operator< OP >::value && ! is_object< InputType >::value && ! is_object< IOType >::value, void >::type * = NULL ) {
			// static sanity check
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) ||
								( std::is_same< InputType, typename OP::D1 >::value && std::is_same< IOType, typename OP::D2 >::value && std::is_same< IOType, typename OP::D3 >::value ) ),
				"alp::internal::foldr (level 0)",
				"Argument value types do not match operator domains while no_casting "
				"descriptor was set" );

			// call foldr
			const typename OP::D1 left = static_cast< typename OP::D1 >( x );
			typename OP::D3 right = static_cast< typename OP::D3 >( y );
			op.foldr( left, right );
			y = static_cast< IOType >( right );

			// done
			return SUCCESS;
		}

		/**
		 * @internal \a foldl reference implementation on internal scalar container.
		 */
		template< Descriptor descr = descriptors::no_operation, class OP, typename InputType, typename IOType >
		RC foldl( IOType & x,
			const InputType & y,
			const OP & op = OP(),
			const typename std::enable_if< is_operator< OP >::value && ! is_object< InputType >::value && ! is_object< IOType >::value, void >::type * = NULL ) {
			// static sanity check
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) ||
								( std::is_same< IOType, typename OP::D1 >::value && std::is_same< InputType, typename OP::D2 >::value && std::is_same< IOType, typename OP::D3 >::value ) ),
				"alp::internal::foldl (level 0)",
				"Argument value types do not match operator domains while no_casting "
				"descriptor was set" );

			// call foldl
			typename OP::D1 left = static_cast< typename OP::D1 >( x );
			const typename OP::D3 right = static_cast< typename OP::D3 >( y );
			op.foldl( left, right );
			x = static_cast< IOType >( left );

			// done
			return SUCCESS;
		}

	} // end namespace ``internal''

	/**
	 * \defgroup BLAS0 The Level-0 Basic Linear Algebra Subroutines (BLAS)
	 *
	 * A collection of functions that let GraphBLAS operators work on
	 * zero-dimensional containers, i.e., on scalars.
	 *
	 * The GraphBLAS uses opaque data types and defines several standard functions
	 * to operate on these data types. Examples types are alp::Vector and
	 * alp::Matrix, example functions are alp::dot and alp::vxm.
	 *
	 * To input data into an opaque GraphBLAS type, each opaque type defines a
	 * member function \a build: alp::Vector::build() and alp::Matrix::build().
	 *
	 * To extract data from opaque GraphBLAS types, each opaque type provides
	 * \em iterators that may be obtained via the STL standard \a begin and \a end
	 * functions:
	 *   - alp::Vector::begin or alp::Vector::cbegin
	 *   - alp::Vector::end or alp::Vector::cend
	 *   - alp::Matrix::begin or alp::Matrix::cbegin
	 *   - alp::Matrix::end or alp::Matrix::cend
	 *
	 * Some GraphBLAS functions, however, reduce all elements in a GraphBLAS
	 * container into a single element of a given type. So for instance, alp::dot
	 * on two vectors of type alp::Vector<double> using the regular real semiring
	 * alp::Semiring<double> will store its output in a variable of type \a double.
	 *
	 * When parametrising GraphBLAS functions in terms of arbitrary Semirings,
	 * Monoids, Operators, and object types, it is useful to have a way to apply
	 * the same operators on whatever type they make functions like alp::dot
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
	RC resize( Scalar< InputType, InputStructure, reference > &s, const length_type new_nz ) {
		if( new_nz <= 1 ) {
			setInitialized( s, false );
			return SUCCESS;
		} else {
			return ILLEGAL;
		}
	}

	/**
	 * @brief Reference implementation of \a apply.
	 */
	template< 
		class OP,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2,
		typename OutputType, typename OutputStructure
	>
	RC apply( 
		Scalar< OutputType, OutputStructure, reference > &out,
		const Scalar< InputType1, InputStructure1, reference > &x,
		const Scalar< InputType2, InputStructure2, reference > &y,
		const OP &op = OP(),
		const typename std::enable_if<
			is_operator< OP >::value &&
			!is_object< InputType1 >::value &&
			!is_object< InputType2 >::value &&
			!is_object< OutputType >::value,
		void >::type * = NULL
	) {

		RC rc = internal::apply( *out, *x, *y, op );
		
		return rc;
	}

	/**
	 * @brief Reference implementation of \a foldr.
	 */
	template< 
		class OP, 
		typename InputType, typename InputStructure, 
		typename IOType, typename IOStructure >
	RC foldr( const Scalar< InputType, InputStructure, reference > &x,
		Scalar< IOType, IOStructure, reference > &y,
		const OP & op = OP(),
		const typename std::enable_if< is_operator< OP >::value && ! is_object< InputType >::value && ! is_object< IOType >::value, void >::type * = NULL ) {
		
		RC rc = internal::foldr( *x, *y, op);

		return rc;
	}

	/**
	 * @brief Reference implementation of \a foldl.
	 */
	template< 
		class OP, 
		typename InputType, typename InputStructure, 
		typename IOType, typename IOStructure >
	RC foldl( Scalar< IOType, IOStructure, reference > &x,
		const Scalar< InputType, InputStructure, reference > &y,
		const OP & op = OP(),
		const typename std::enable_if< is_operator< OP >::value && ! is_object< InputType >::value && ! is_object< IOType >::value, void >::type * = NULL ) {

		RC rc = internal::foldl( *x, *y, op );

		return rc;
	}

	/** @} */
	
} // end namespace ``alp''

#endif // end ``_H_ALP_REFERENCE_BLAS0''

