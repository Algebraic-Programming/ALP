
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

#ifndef _H_GRB_DENSEREF_BLAS1
#define _H_GRB_DENSEREF_BLAS1

#include <graphblas/backends.hpp>
#include <graphblas/config.hpp>
#include <graphblas/rc.hpp>

namespace grb {

    /**
     * Calculates the dot product, \f$ \alpha = (x,y) \f$, under a given additive
     * monoid and multiplicative operator.
     *
     * @tparam descr      The descriptor to be used (descriptors::no_operation
     *                    if left unspecified).
     * @tparam Ring       The semiring type to use.
     * @tparam OutputType The output type.
     * @tparam InputType1 The input element type of the left-hand input vector.
     * @tparam InputType2 The input element type of the right-hand input vector.
     *
     * @param[in,out]  z    The output element \f$ z + \alpha \f$.
     * @param[in]      x    The left-hand input vector.
     * @param[in]      y    The right-hand input vector.
     * @param[in] addMonoid The additive monoid under which the reduction of the
     *                      results of element-wise multiplications of \a x and
     *                      \a y are performed.
     * @param[in]   anyop   The multiplicative operator under which element-wise
     *                      multiplications of \a x and \a y are performed. This can
     *                      be any binary operator.
     *
     * By the definition that a dot-product operates under any additive monoid and
     * any binary operator, it follows that a dot-product under any semiring can be
     * trivially reduced to a call to this version instead.
     *
     * @return grb::MISMATCH When the dimensions of \a x and \a y do not match. All
     *                       input data containers are left untouched if this exit
     *                       code is returned; it will be as though this call was
     *                       never made.
     * @return grb::SUCCESS  On successful completion of this call.
     *
     * \parblock
     * \par Performance semantics
     *      -# This call takes \f$ \Theta(n/p) \f$ work at each user process, where
     *         \f$ n \f$ equals the size of the vectors \a x and \a y, and
     *         \f$ p \f$ is the number of user processes. The constant factor
     *         depends on the cost of evaluating the addition and multiplication
     *         operators. A good implementation uses vectorised instructions
     *         whenever the input domains, output domain, and the operators used
     *         allow for this.
     *
     *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory used
     *         by the application at the point of a call to this function.
     *
     *      -# This call incurs at most
     *         \f$ n( \mathit{sizeof}(\mathit{D1}) + \mathit{sizeof}(\mathit{D2}) ) + \mathcal{O}(p) \f$
     *         bytes of data movement.
     *
     *      -# This call incurs at most \f$ \Theta(\log p) \f$ synchronisations
     *         between two or more user processes.
     *
     *      -# A call to this function does result in any system calls.
     * \endparblock
     *
     * \note This requires an implementation to pre-allocate \f$ \Theta(p) \f$
     *       memory for inter-process reduction, if the underlying communication
     *       layer indeed requires such a buffer. This buffer may not be allocated
     *       (nor freed) during a call to this function.
     *
     * \parblock
     * \par Valid descriptors
     *   -# grb::descriptors::no_operation
     *   -# grb::descriptors::no_casting
     *   -# grb::descriptors::dense
     * \endparblock
     *
     * If the dense descriptor is set, this implementation returns grb::ILLEGAL if
     * it was detected that either \a x or \a y was sparse. In this case, it shall
     * otherwise be as though the call to this function had not occurred (no side
     * effects).
     *
     * \note The standard, in contrast, only specifies undefined behaviour would
     *       occur. This implementation goes beyond the standard by actually
     *       specifying what will happen.
     */
	template<
		Descriptor descr = descriptors::no_operation,
		class AddMonoid, class AnyOp,
		typename OutputType, typename InputType1, typename InputType2,
		typename InputView1, typename InputView2,
		typename InputStorage1, typename InputStorage2,
		typename InputCoords1, typename InputCoords2
	>
	RC dot( OutputType &z,
		const VectorView< InputType1, InputView1, InputStorage1, reference_dense, InputCoords1 > &x,
		const VectorView< InputType2, InputView2, InputStorage2, reference_dense, InputCoords2 > &y,
		const AddMonoid &addMonoid = AddMonoid(),
		const AnyOp &anyOp = AnyOp(),
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< AddMonoid >::value &&
			grb::is_operator< AnyOp >::value,
		void >::type * const = NULL
	) {
		(void)z;
		(void)x;
		(void)y;
		(void)addMonoid;
		(void)anyOp;
		// static sanity checks
		// NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< InputType1, typename AnyOp::D1 >::value ), "grb::dot",
		// 	"called with a left-hand vector value type that does not match the first "
		// 	"domain of the given multiplicative operator" );
		// NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< InputType2, typename AnyOp::D2 >::value ), "grb::dot",
		// 	"called with a right-hand vector value type that does not match the second "
		// 	"domain of the given multiplicative operator" );
		// NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< typename AddMonoid::D3, typename AnyOp::D1 >::value ), "grb::dot",
		// 	"called with a multiplicative operator output domain that does not match "
		// 	"the first domain of the given additive operator" );
		// NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< OutputType, typename AddMonoid::D2 >::value ), "grb::dot",
		// 	"called with an output vector value type that does not match the second "
		// 	"domain of the given additive operator" );
		// NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< typename AddMonoid::D3, typename AddMonoid::D2 >::value ), "grb::dot",
		// 	"called with an additive operator whose output domain does not match its "
		// 	"second input domain" );
		// NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< OutputType, typename AddMonoid::D3 >::value ), "grb::dot",
		// 	"called with an output vector value type that does not match the third "
		// 	"domain of the given additive operator" );

		// dynamic sanity check
		// const size_t n = internal::getCoordinates( y ).size();
		// if( internal::getCoordinates( x ).size() != n ) {
		// 	return MISMATCH;
		// }

		// // cache nnzs
		// const size_t nnzx = internal::getCoordinates( x ).nonzeroes();
		// const size_t nnzy = internal::getCoordinates( y ).nonzeroes();

		// // catch trivial case
		// if( nnzx == 0 && nnzy == 0 ) {
		// 	return SUCCESS;
		// }

		// // dot will be computed out-of-place here. A separate field is needed because
		// // of possible multi-threaded computation of the dot.
		// OutputType oop = addMonoid.template getIdentity< OutputType >();

		// if descriptor says nothing about being dense...
		RC ret = SUCCESS;
		// if( !( descr & descriptors::dense ) ) {
		// 	// check if inputs are actually dense...
		// 	if( nnzx == n && nnzy == n ) {
		// 		// call dense implementation
		// 		ret = internal::dot_generic< descr | descriptors::dense >( oop, x, y, addMonoid, anyOp );
		// 	} else {
		// 		// pass to sparse implementation
		// 		ret = internal::dot_generic< descr >( oop, x, y, addMonoid, anyOp );
		// 	}
		// } else {
		// 	// descriptor says dense, but if any of the vectors are actually sparse...
		// 	if( nnzx < n || nnzy < n ) {
		// 		return ILLEGAL;
		// 	} else {
		// 		// all OK, pass to dense implementation
		// 		ret = internal::dot_generic< descr >( oop, x, y, addMonoid, anyOp );
		// 	}
		// }

		// fold out-of-place dot product into existing input, and exit
		// ret = ret ? ret : foldl( z, oop, addMonoid.getOperator() );
		return ret;
	}

    /**
	 * Provides a generic implementation of the dot computation on semirings by
	 * translating it into a dot computation on an additive commutative monoid
	 * with any multiplicative operator.
	 *
	 * For return codes, exception behaviour, performance semantics, template
	 * and non-template arguments, @see grb::dot.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename IOType, typename InputType1, typename InputType2,
		typename InputView1, typename InputView2,
		typename InputStorage1, typename InputStorage2,
		Backend backend, typename Coords1, typename Coords2
	>
	RC dot( IOType &x,
		const VectorView< InputType1, InputView1, InputStorage1, backend, Coords1 > &left,
		const VectorView< InputType2, InputView2, InputStorage2, backend, Coords2 > &right,
		const Ring &ring = Ring(),
		const typename std::enable_if<
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< IOType >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = NULL
	) {
		// return grb::dot< descr >( x,
		return grb::dot( x,
			left, right,
			ring.getAdditiveMonoid(),
			ring.getMultiplicativeOperator()
		);
	}

    /**
	 * Provides a generic implementation of the 2-norm computation.
	 *
	 * Proceeds by computing a dot-product on itself and then taking the square
	 * root of the result.
	 *
	 * This function is only available when the output type is floating point.
	 *
	 * For return codes, exception behaviour, performance semantics, template
	 * and non-template arguments, @see grb::dot.
	 *
	 * @param[out] x The 2-norm of \a y. The input value of \a x will be ignored.
	 * @param[in]  y The vector to compute the norm of.
	 * @param[in] ring The Semiring under which the 2-norm is to be computed.
	 *
	 * \warning This function computes \a x out-of-place. This is contrary to
	 *          standard ALP/GraphBLAS functions that are always in-place.
	 *
	 * \warning A \a ring is not sufficient for computing a two-norm. This
	 *          implementation assumes the standard <tt>sqrt</tt> function
	 *          must be applied on the result of a dot-product of \a y with
	 *          itself under the supplied semiring.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType, typename OutputType,
        typename InputView,
		typename InputStorage,
		Backend backend, typename Coords
	>
	RC norm2( OutputType &x,
		const VectorView< InputType, InputView, InputStorage, backend, Coords > &y,
		const Ring &ring = Ring(),
		const typename std::enable_if<
			std::is_floating_point< OutputType >::value,
		void >::type * const = NULL
	) {
		RC ret = grb::dot< descr >( x, y, y, ring );
		if( ret == SUCCESS ) {
			x = sqrt( x );
		}
		return ret;
	}

} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_BLAS1''

