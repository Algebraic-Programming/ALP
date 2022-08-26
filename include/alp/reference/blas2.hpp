
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

#ifndef _H_ALP_REFERENCE_BLAS2
#define _H_ALP_REFERENCE_BLAS2

#include <cstddef>

#include <alp/base/blas2.hpp>
#include <alp/backends.hpp>
#include <alp/blas3.hpp>
#include <alp/config.hpp>
#include <alp/matrix.hpp>
#include <alp/vector.hpp>
#include <alp/rc.hpp>

namespace alp {

	/**
	 * \addtogroup reference
	 * @{
	 */

	/**
	 * Retrieve the number of nonzeroes contained in this matrix.
	 *
	 * @returns The number of nonzeroes the current matrix contains.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -# This function performs \f$ \Theta(1) \f$ work.
	 *        -# This function allocates no additional dynamic memory.
	 *        -# This function uses \f$ \mathcal{O}(1) \f$ memory
	 *           beyond that which was already used at function entry.
	 *        -# This function will move
	 *             \f$ \mathit{sizeof}( size\_t ) \f$
	 *           bytes of memory.
	 * \endparblock
	 */
	template< typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC >
	size_t nnz( const Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > & A ) noexcept {
		(void)A;
		return UNSUPPORTED;
	}

	/**
	 * Resizes the matrix to have at least the given number of nonzeroes.
	 * The contents of the matrix are not retained.
	 *
	 * Resizing of dense containers is not allowed as the capacity is determined
	 * by the container dimensions and the storage scheme. Therefore, this
	 * function will not change the capacity of the matrix.
	 *
	 * Even though the capacity remains unchanged, the contents of the matrix
	 * are not retained to maintain compatibility with the general specification.
	 * However, the actual memory will not be reallocated. Rather, the matrix
	 * will be marked as uninitialized.
	 *
	 * @param[in] A         The matrix to be resized.
	 * @param[in] nonzeroes The number of nonzeroes this matrix is to contain.
	 *
	 * @return SUCCESS   If \a new_nz is not larger than the current capacity
	 *                   of the matrix.
	 *         ILLEGAL   If \a new_nz is larger than the current capacity of
	 *                   the matrix.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -$ This function consitutes \f$ \Theta(1) \f$ work.
	 *        -# This function allocates \f$ \Theta(0) \f$
	 *           bytes of dynamic memory.
	 *        -# This function does not make system calls.
	 * \endparblock
	 */
	template< typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC >
	RC resize( Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &A, const size_t new_nz ) noexcept {
		(void)A;
		(void)new_nz;
		// TODO implement
		// setInitialized( A, false );
		return PANIC;
	}

	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType1 = typename Ring::D1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2 = typename Ring::D2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2
	>
	RC vxm( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & u,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & v,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< alp::is_semiring< Ring >::value, void >::type * const = NULL ) {

		return mxm( 
			get_view< view::transpose > ( get_view< view::matrix > ( u ) ),
			get_view< view::transpose > ( get_view< view::matrix > ( v ) ), 
			A, 
			ring );

	}

	template< Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class AdditiveMonoid, class MultiplicativeOperator
	>
	RC vxm( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & u,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & v,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< alp::is_monoid< AdditiveMonoid >::value && alp::is_operator< MultiplicativeOperator >::value && ! alp::is_object< IOType >::value &&
				! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {

		return mxm( 
			get_view< view::transpose > ( get_view< view::matrix > ( u ) ),
			get_view< view::transpose > ( get_view< view::matrix > ( v ) ), 
			A, 
			mul, add );
	}


	/**
	 * \internal Delegates to fully masked variant.
	 */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType2 = typename Ring::D2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		typename InputType1 = typename Ring::D1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1
	>
	RC mxv( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & u,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & v,
		const Ring & ring,
		const typename std::enable_if< alp::is_semiring< Ring >::value, void >::type * const = NULL ) {

		return mxm( 
			get_view< view::matrix > ( u ),
			A, 
			get_view< view::matrix > ( v ), 
			ring );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		class AdditiveMonoid, class MultiplicativeOperator
	>
	RC mxv( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & u,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & v,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< alp::is_monoid< AdditiveMonoid >::value && alp::is_operator< MultiplicativeOperator >::value && ! alp::is_object< IOType >::value &&
				! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {

		return mxm( 
			get_view< view::matrix > ( u ),
			A, 
			get_view< view::matrix > ( v ), 
			mul, add );
	}

	/**
	 * Straightforward implementation using the column-major layout.
	 *
	 * @see alp::eWiseLambda for the user-level specification.
	 */
	template< class ActiveDistribution, typename Func,
		typename DataType, typename Structure, typename View, typename ImfR, typename ImfC
	>
	RC eWiseLambda( const Func f,
		const Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, reference > & A,
		const size_t s,
		const size_t P ) {
#ifdef _DEBUG
		std::cout << "entering alp::eWiseLambda (matrices, reference ). A is " << alp::nrows( A ) << " by " << alp::ncols( A ) << " and holds " << alp::nnz( A ) << " nonzeroes.\n";
#endif
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * This function provides dimension checking and will defer to the below
	 * function for the actual implementation.
	 *
	 * @see alp::eWiseLambda for the user-level specification.
	 */
	template< typename Func,
		typename DataType1, typename DataStructure1, typename DataView1, typename DataImfR1, typename DataImfC1,
		typename DataType2, typename DataStructure2, typename DataView2, typename DataImfR2, typename DataImfC2,
		typename... Args
	>
	RC eWiseLambda( const Func f,
		const Matrix< DataType1, DataStructure1, Density::Dense, DataView1, DataImfR1, DataImfC1, reference > & A,
		const Vector< DataType2, DataStructure2, Density::Dense, DataView2, DataImfR2, DataImfC2, reference > x,
		Args... args ) {
		// do size checking
		if( ! ( size( x ) == nrows( A ) || size( x ) == ncols( A ) ) ) {
			std::cerr << "Mismatching dimensions: given vector of size " << size( x ) << " has nothing to do with either matrix dimension (" << nrows( A ) << " nor " << ncols( A ) << ").\n";
			return MISMATCH;
		}
		// no need for synchronisation, everything is local in reference implementation
		return eWiseLambda( f, A, args... );
	}

	/**
	 * For all elements in a ALP Matrix \a A, fold the value \f$ \alpha \f$
	 * into each element.
	 *
	 * The original value of \f$ \alpha \f$ is used as the left-hand side input
	 * of the operator \a op. The right-hand side inputs for \a op are retrieved
	 * from the input Matrix \a A. The result of the operation is stored in \a A,
	 * thus overwriting its previous values.
	 *
	 * The value of \f$ A_i,j \f$ after a call to thus function thus equals
	 * \f$ \alpha \odot A_i,j \f$, for all \f$ i, j \in \{ 0, 1, \dots, n - 1 \} \f$.
	 *
	 * @tparam descr         The descriptor used for evaluating this function.
	 *                       By default, this is alp::descriptors::no_operation.
	 * @tparam OP            The type of the operator to be applied.
	 * @tparam InputType     The type of \a alpha.
	 * @tparam IOType        The type of the elements in \a A.
	 * @tparam IOStructure   The structure of the matrix \a A.
	 * @tparam IOView        The view applied to the matrix \a A.
	 *
	 * @param[in]     alpha The input value to apply as the left-hand side input
	 *                      to \a op.
	 * @param[in,out] A     On function entry: the initial values to be applied as
	 *                      the right-hand side input to \a op.
	 *                      On function exit: the output data.
	 * @param[in]     op    The monoid under which to perform this left-folding.
	 *
	 * @returns alp::SUCCESS This function always succeeds.
	 *
	 * \note We only define fold under monoids, not under plain operators.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * alp::descriptors::no_operation, alp::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If alp::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a IOType, 2) the second domain of \a op must match
	 * \a InputType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# (no requirements).
	 * \endparblock
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	//  *         the size of the vector \a x. The constant factor depends on the
	//  *         cost of evaluating the underlying binary operator. A good
	//  *         implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the operator used allow for this.
	//  *
	//  *      -# This call will not result in additional dynamic memory allocations.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ 2n \cdot \mathit{sizeof}(\mathit{IOType}) + \mathcal{O}(1) \f$
	//  *         bytes of data movement.
	//  * \endparblock
	 *
	 * @see alp::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Monoid
	>
	RC foldr( const Scalar< InputType, InputStructure, reference > & alpha,
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & A,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! alp::is_object< InputType >::value && ! alp::is_object< IOType >::value && alp::is_monoid< Monoid >::value, void >::type * const = NULL ) {
		// static sanity checks
		//NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, IOType >::value ), "alp::foldl",
		//	"called with a vector x of a type that does not match the first domain "
		//	"of the given operator" );
		//NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, InputType >::value ), "alp::foldl",
		//	"called on a vector y of a type that does not match the second domain "
		//	"of the given operator" );
		//NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, IOType >::value ), "alp::foldl",
		//	"called on a vector x of a type that does not match the third domain "
		//	"of the given operator" );
		(void)alpha;
		(void)A;
		(void)monoid;

		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/** @} */

} // end namespace ``alp''

#endif // end ``_H_ALP_REFERENCE_BLAS2''

