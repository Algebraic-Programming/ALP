
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

#include <alp/backends.hpp>
#include <alp/config.hpp>
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
	 *        -# This function consitutes \f$ \Theta(1) \f$ work.
	 *        -# This function allocates no additional dynamic memory.
	 *        -# This function uses \f$ \mathcal{O}(1) \f$ memory
	 *           beyond that which was already used at function entry.
	 *        -# This function will move
	 *             \f$ \mathit{sizeof}( size\_t ) \f$
	 *           bytes of memory.
	 * \endparblock
	 */
	template< typename InputType, typename InputStructure, typename InputView >
	size_t nnz( const Matrix< InputType, InputStructure, Density::Dense, InputView, reference > & A ) noexcept {
		return A.nz;
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
	template< typename InputType, typename InputStructure, typename InputView >
	RC resize( Matrix< InputType, InputStructure, Density::Dense, InputView, reference > &A, const size_t new_nz ) noexcept {
		(void)A;
		(void)new_nz;
		// TODO implement
		// setInitialized( A, false );
		return PANIC;
	}

	/** \internal Delegates to fully masked variant */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType, typename IOStructure, typename IOView,
		typename InputType1, typename InputStructure1, typename InputView1,
		typename InputType2, typename InputStructure2,  typename InputView2,
		typename InputType3, typename InputStructure3, typename InputView3 >
	RC vxm( Vector< IOType, IOStructure, Density::Dense, IOView, reference > & u,
		const Vector< InputType3, InputStructure3, Density::Dense, InputView3, reference > & mask,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, reference > & v,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, reference > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< alp::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const Vector< bool, structures::General, Density::Dense, view::Original< void >, reference > empty_mask( 0 );
		return vxm< descr, true, false >( u, mask, v, empty_mask, A, ring );
	}

	/** \internal Delegates to fully masked variant */
	template< Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType, typename IOStructure, typename IOView,
		typename InputType1, typename InputStructure1, typename InputView1,
		typename InputType2, typename InputStructure2, typename InputView2,
		typename InputType3, typename InputStructure3, typename InputView3 >
	RC vxm( Vector< IOType, IOStructure, Density::Dense, IOView, reference > & u,
		const Vector< InputType3, InputStructure3, Density::Dense, InputView3, reference > & mask,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, reference > & v,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, reference > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< alp::is_monoid< AdditiveMonoid >::value && alp::is_operator< MultiplicativeOperator >::value && ! alp::is_object< IOType >::value &&
				! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && ! alp::is_object< InputType3 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		const alp::Vector< bool, structures::General, Density::Dense, view::Original< void >, reference > empty_mask( 0 );
		return vxm< descr, true, false >( u, mask, v, empty_mask, A, add, mul );
	}

	/** \internal Delegates to vxm_generic. */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class Ring,
		typename IOType, typename IOStructure, typename IOView,
		typename InputType1, typename InputStructure1, typename InputView1,
		typename InputType2, typename InputStructure2, typename InputView2,
		typename InputType3, typename InputStructure3, typename InputView3,
		typename InputType4, typename InputStructure4, typename InputView4 >
	RC vxm( Vector< IOType, IOStructure, Density::Dense, IOView, reference > & u,
		const Vector< InputType3, InputStructure3, Density::Dense, InputView3, reference > & mask,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, reference > & v,
		const Vector< InputType4, InputStructure4, Density::Dense, InputView4, reference > & v_mask,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, reference > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< alp::is_semiring< Ring >::value, void >::type * const = NULL ) {
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4, typename IOStructure, typename IOView, 
		typename InputType1 = typename Ring::D1, typename InputStructure1, typename InputView1, 
		typename InputType2 = typename Ring::D2, typename InputStructure2, typename InputView2 >
	RC vxm( Vector< IOType, IOStructure, Density::Dense, IOView, reference > & u,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, reference > & v,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, reference > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< alp::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const Vector< bool, structures::General, Density::Dense, view::Original< void >, reference > empty_mask( 0 );
		return vxm< descr, false, false >( u, empty_mask, v, empty_mask, A, ring );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation, class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType, typename IOStructure, typename IOView,
		typename InputType1, typename InputStructure1, typename InputView1,
		typename InputType2, typename InputStructure2, typename InputView2 >
	RC vxm( Vector< IOType, IOStructure, Density::Dense, IOView, reference > & u,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, reference > & v,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, reference > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< alp::is_monoid< AdditiveMonoid >::value && alp::is_operator< MultiplicativeOperator >::value && ! alp::is_object< IOType >::value &&
				! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		const alp::Vector< bool, structures::General, Density::Dense, view::Original< void >, reference > empty_mask( 0 );
		return vxm< descr, false, false >( u, empty_mask, v, empty_mask, A, add, mul );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType, typename IOStructure, typename IOView,
		typename InputType1, typename InputStructure1, typename InputView1,
		typename InputType2, typename InputStructure2, typename InputView2,
		typename InputType3 = bool, typename InputStructure3, typename InputView3 >
	RC mxv( Vector< IOType, IOStructure, Density::Dense, IOView, reference > & u,
		const Vector< InputType3, InputStructure3, Density::Dense, InputView3, reference > & mask,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, reference > & A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, reference > & v,
		const Ring & ring,
		const typename std::enable_if< alp::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const Vector< bool, structures::General, Density::Dense, view::Original< void >, reference > empty_mask( 0 );
		return mxv< descr, true, false >( u, mask, A, v, empty_mask, ring );
	}

	/** \internal Delegates to vxm_generic */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class Ring,
		typename IOType, typename IOStructure, typename IOView,
		typename InputType1, typename InputStructure1, typename InputView1,
		typename InputType2, typename InputStructure2, typename InputView2,
		typename InputType3, typename InputStructure3, typename InputView3,
		typename InputType4, typename InputStructure4, typename InputView4 >
	RC mxv( Vector< IOType, IOStructure, Density::Dense, IOView, reference > & u,
		const Vector< InputType3, InputStructure3, Density::Dense, InputView3, reference > & mask,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, reference > & A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, reference > & v,
		const Vector< InputType4, InputStructure4, Density::Dense, InputView4, reference > & v_mask,
		const Ring & ring,
		const typename std::enable_if< alp::is_semiring< Ring >::value, void >::type * const = NULL ) {

		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * \internal Delegates to fully masked variant.
	 */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4, typename IOStructure, typename IOView, 
		typename InputType1 = typename Ring::D1, typename InputStructure1, typename InputView1,
		typename InputType2 = typename Ring::D2, typename InputStructure2, typename InputView2 >
	RC mxv( Vector< IOType, IOStructure, Density::Dense, IOView, reference > & u,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, reference > & A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, reference > & v,
		const Ring & ring,
		const typename std::enable_if< alp::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const Vector< bool, structures::General, Density::Dense, view::Original< void >, reference > empty_mask( 0 );
		return mxv< descr, false, false >( u, empty_mask, A, v, empty_mask, ring );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation, class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType, typename IOStructure, typename IOView,
		typename InputType1, typename InputStructure1, typename InputView1,
		typename InputType2, typename InputStructure2, typename InputView2 >
	RC mxv( Vector< IOType, IOStructure, Density::Dense, IOView, reference > & u,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, reference > & A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, reference > & v,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< alp::is_monoid< AdditiveMonoid >::value && alp::is_operator< MultiplicativeOperator >::value && ! alp::is_object< IOType >::value &&
				! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		const alp::Vector< bool, structures::General, Density::Dense, view::Original< void >, reference > empty_mask( 0 );
		return mxv< descr, false, false >( u, empty_mask, A, v, empty_mask, add, mul );
	}

	/**
	 * \internal Delegates to vxm_generic
	 */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType, typename IOStructure, typename IOView,
		typename InputType1, typename InputStructure1, typename InputView1,
		typename InputType2, typename InputStructure2, typename InputView2,
		typename InputType3, typename InputStructure3, typename InputView3,
		typename InputType4, typename InputStructure4, typename InputView4 >
	RC vxm( Vector< IOType, IOStructure, Density::Dense, IOView, reference > & u,
		const Vector< InputType3, InputStructure3, Density::Dense, InputView3, reference > & mask,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, reference > & v,
		const Vector< InputType4, InputStructure4, Density::Dense, InputView4, reference > & v_mask,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, reference > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< alp::is_monoid< AdditiveMonoid >::value && alp::is_operator< MultiplicativeOperator >::value && ! alp::is_object< IOType >::value &&
				! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && ! alp::is_object< InputType3 >::value && ! alp::is_object< InputType4 >::value &&
				! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {

		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * \internal Delegates to vxm_generic.
	 */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType, typename IOStructure, typename IOView,
		typename InputType1, typename InputStructure1, typename InputView1,
		typename InputType2, typename InputStructure2, typename InputView2,
		typename InputType3, typename InputStructure3, typename InputView3,
		typename InputType4, typename InputStructure4, typename InputView4 >
	RC mxv( Vector< IOType, IOStructure, Density::Dense, IOView, reference > & u,
		const Vector< InputType3, InputStructure3, Density::Dense, InputView3, reference > & mask,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, reference > & A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, reference > & v,
		const Vector< InputType4, InputStructure4, Density::Dense, InputView4, reference > & v_mask,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< alp::is_monoid< AdditiveMonoid >::value && alp::is_operator< MultiplicativeOperator >::value && ! alp::is_object< IOType >::value &&
				! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && ! alp::is_object< InputType3 >::value && ! alp::is_object< InputType4 >::value &&
				! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {

		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Straightforward implementation using the column-major layout.
	 *
	 * @see alp::eWiseLambda for the user-level specification.
	 */
	template< class ActiveDistribution, typename Func, typename DataType, typename Structure, typename View>
	RC eWiseLambda( const Func f,
		const Matrix< DataType, Structure, Density::Dense, View, reference > & A,
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
		typename DataType1, typename DataStructure1, typename DataView1,
		typename DataType2, typename DataStructure2, typename DataView2,  typename... Args >
	RC eWiseLambda( const Func f,
		const Matrix< DataType1, DataStructure1, Density::Dense, DataView1, reference > & A,
		const Vector< DataType2, DataStructure2, Density::Dense, DataView2, reference > x, Args... args ) {
		// do size checking
		if( ! ( size( x ) == nrows( A ) || size( x ) == ncols( A ) ) ) {
			std::cerr << "Mismatching dimensions: given vector of size " << size( x ) << " has nothing to do with either matrix dimension (" << nrows( A ) << " nor " << ncols( A ) << ").\n";
			return MISMATCH;
		}
		// no need for synchronisation, everything is local in reference implementation
		return eWiseLambda( f, A, args... );
	}

	/** @} */

} // end namespace ``alp''

#endif // end ``_H_ALP_REFERENCE_BLAS2''

