
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

#ifndef _H_GRB_DENSEREF_BLAS2
#define _H_GRB_DENSEREF_BLAS2

#include <cstddef>

#include <graphblas/backends.hpp>
#include <graphblas/config.hpp>
#include <graphblas/rc.hpp>

namespace grb {

	/**
	 * \addtogroup reference_dense
	 * @{
	 */

	/**
	 * Retrieve the row dimension size of this matrix.
	 *
	 * @returns The number of rows the current matrix contains.
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
	size_t nrows( const StructuredMatrix< InputType, InputStructure, Density::Dense, InputView, reference_dense > & A ) noexcept {
		return A.m;
	}

	/**
	 * Retrieve the column dimension size of this matrix.
	 *
	 * @returns The number of columns the current matrix contains.
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
	size_t ncols( const StructuredMatrix< InputType, InputStructure, Density::Dense, InputView, reference_dense > & A ) noexcept {
		return A.n;
	}

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
	template< typename InputType >
	size_t nnz( const StructuredMatrix< InputType, InputStructure, Density::Dense, InputView, reference_dense > & A ) noexcept {
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
	RC resize( StructuredMatrix< InputType, InputStructure, Density::Dense, InputView, reference_dense > &A, const size_t new_nz ) noexcept {
		(void)A;
		(void)new_nz;
		// TODO implement
		// setInitialized( A, false );
		return PANIC;
	}

	/** \internal Delegates to fully masked variant */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType, typename IOView, 
		typename InputType1, typename InputType1, typename InputView1, 
		typename InputType2, typename InputStructure2,  typename InputView2,
		typename InputType3, typename InputType3, typename InputView3, typename InputStorage3 >
	RC vxm( VectorView< IOType, IOStructure, Density::Dense, IOView, reference_dense > & u,
		const VectorView< InputType3, InputStructure3, Density::Dense, InputView3, reference_dense > & mask,
		const VectorView< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > & v,
		const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const VectorView< bool, reference, internal::DefaultCoordinates > empty_mask( 0 );
		return vxm< descr, true, false >( u, mask, v, empty_mask, A, ring );
	}

	/** \internal Delegates to fully masked variant */
	template< Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType, typename IOView, 
		typename InputType1, typename InputView1, 
		typename InputType2, typename InputStructure2,  typename InputView2,
		typename InputType3, typename InputView3, typename InputStorage3 >
	RC vxm( VectorView< IOType, IOStructure, Density::Dense, IOView, reference_dense > & u,
		const VectorView< InputType3, InputStructure3, Density::Dense, InputView3, reference_dense > & mask,
		const VectorView< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > & v,
		const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		const grb::VectorView< bool, reference, internal::DefaultCoordinates > empty_mask( 0 );
		return vxm< descr, true, false >( u, mask, v, empty_mask, A, add, mul );
	}

	/** \internal Delegates to vxm_generic. */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class Ring,
		typename IOType, typename IOView, 
		typename InputType1, typename InputView1, 
		typename InputType2, typename InputStructure2,  typename InputView2,
		typename InputType3, typename InputView3, 
		typename InputType4, typename InputView4, typename InputStorage4 >
	RC vxm( VectorView< IOType, IOStructure, Density::Dense, IOView, reference_dense > & u,
		const VectorView< InputType3, InputStructure3, Density::Dense, InputView3, reference_dense > & mask,
		const VectorView< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > & v,
		const VectorView< InputType4, InputStructure4, Density::Dense, InputView4, reference_dense > & v_mask,
		const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4, typename IOView, 
		typename InputType1 = typename Ring::D1, typename InputView1, 
		typename InputType2 = typename Ring::D2, typename InputStructure2,  typename InputView2 >
	RC vxm( VectorView< IOType, IOStructure, Density::Dense, IOView, reference_dense > & u,
		const VectorView< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > & v,
		const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const VectorView< bool, reference_dense, internal::DefaultCoordinates > empty_mask( 0 );
		return vxm< descr, false, false >( u, empty_mask, v, empty_mask, A, ring );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation, class AdditiveMonoid, class MultiplicativeOperator, typename IOType, typename InputType1, typename InputType2 >
	RC vxm( VectorView< IOType, IOStructure, Density::Dense, IOView, reference_dense > & u,
		const VectorView< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > & v,
		const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		const grb::VectorView< bool, reference_dense, internal::DefaultCoordinates > empty_mask( 0 );
		return vxm< descr, false, false >( u, empty_mask, v, empty_mask, A, add, mul );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType, typename IOView, 
		typename InputType1, typename InputView1, 
		typename InputType2, typename InputStructure2,  typename InputView2,
		typename InputType3 = bool, typename InputView3, typename InputStorage3 >
	RC mxv( VectorView< IOType, IOStructure, Density::Dense, IOView, reference_dense > & u,
		const VectorView< InputType3, InputStructure3, Density::Dense, InputView3, reference_dense > & mask,
		const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > & A,
		const VectorView< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > & v,
		const Ring & ring,
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const VectorView< bool, reference_dense, internal::DefaultCoordinates > empty_mask( 0 );
		return mxv< descr, true, false >( u, mask, A, v, empty_mask, ring );
	}

	/** \internal Delegates to vxm_generic */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class Ring,
		typename IOType, typename IOView, 
		typename InputType1, typename InputView1, 
		typename InputType2, typename InputStructure2,  typename InputView2,
		typename InputType3, typename InputView3, 
		typename InputType4, typename InputView4, typename InputStorage4 >
	RC mxv( VectorView< IOType, IOStructure, Density::Dense, IOView, reference_dense > & u,
		const VectorView< InputType3, InputStructure3, Density::Dense, InputView3, reference_dense > & mask,
		const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > & A,
		const VectorView< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > & v,
		const VectorView< InputType4, InputStructure4, Density::Dense, InputView4, reference_dense > & v_mask,
		const Ring & ring,
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {

		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * \internal Delegates to fully masked variant.
	 */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4, typename IOView, 
		typename InputType1 = typename Ring::D1, typename InputStructure2,  typename InputView2,
		typename InputType2 = typename Ring::D2, typename InputView1, typename InputStorage1 >
	RC mxv( VectorView< IOType, IOStructure, Density::Dense, IOView, reference_dense > & u,
		const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > & A,
		const VectorView< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > & v,
		const Ring & ring,
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const VectorView< bool, view::Original< void >, structure::full, reference_dense, internal::DefaultCoordinates > empty_mask( 0 );
		return mxv< descr, false, false >( u, empty_mask, A, v, empty_mask, ring );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation, class AdditiveMonoid, class MultiplicativeOperator, typename IOType, typename InputType1, typename InputType2 >
	RC mxv( VectorView< IOType, IOStructure, Density::Dense, IOView, reference_dense > & u,
		const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > & A,
		const VectorView< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > & v,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		const grb::VectorView< bool, view::Original< void >, structure::full, reference_dense, internal::DefaultCoordinates > empty_mask( 0 );
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
		typename IOType, typename IOView, 
		typename InputType1, typename InputView1, 
		typename InputType2, typename InputStructure2,  typename InputView2,
		typename InputType3, typename InputView3, 
		typename InputType4, typename InputView4, typename InputStorage4 >
	RC vxm( VectorView< IOType, IOStructure, Density::Dense, IOView, reference_dense > & u,
		const VectorView< InputType3, InputStructure3, Density::Dense, InputView3, reference_dense > & mask,
		const VectorView< InputType1, InputStructure1, Density::Dense, InputView1, reference_dense > & v,
		const VectorView< InputType4, InputStructure4, Density::Dense, InputView4, reference_dense > & v_mask,
		const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && ! grb::is_object< InputType4 >::value &&
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
		typename IOType, typename IOView, 
		typename InputType1, typename InputView1, 
		typename InputType2, typename InputStructure2,  typename InputView2,
		typename InputType3, typename InputView3, 
		typename InputType4, typename InputView4, typename InputStorage4 >
	RC mxv( VectorView< IOType, IOStructure, reference_dense > & u,
		const VectorView< InputType3, InputStructure3, reference_dense > & mask,
		const StructuredMatrix< InputType2, InputStructure2, Density::Dense, InputView2, reference_dense > & A,
		const VectorView< InputType1, InputStructure1, reference_dense > & v,
		const VectorView< InputType4, InputStructure4, reference_dense > & v_mask,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && ! grb::is_object< InputType4 >::value &&
				! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {

		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Straightforward implementation using the column-major layout.
	 *
	 * @see grb::eWiseLambda for the user-level specification.
	 */
	template< class ActiveDistribution, typename Func, typename DataType, typename Structure, typename View>
	RC eWiseLambda( const Func f,
		const StructuredMatrix< DataType, Structure, Density::Dense, View, reference_dense > & A,
		const size_t s,
		const size_t P ) {
#ifdef _DEBUG
		std::cout << "entering grb::eWiseLambda (matrices, reference ). A is " << grb::nrows( A ) << " by " << grb::ncols( A ) << " and holds " << grb::nnz( A ) << " nonzeroes.\n";
#endif
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * This function provides dimension checking and will defer to the below
	 * function for the actual implementation.
	 *
	 * @see grb::eWiseLambda for the user-level specification.
	 */
	template< typename Func,
		typename DataType1, typename Structure1,  typename View1,
		typename DataType2, typename View2,  typename... Args >
	RC eWiseLambda( const Func f,
		const StructuredMatrix< DataType1, Structure1, Density::Dense, View1, reference_dense > & A,
		const VectorView< DataType2, DataStructure2, Density::Dense, View2, reference_dense > x, Args... args ) {
		// do size checking
		if( ! ( size( x ) == nrows( A ) || size( x ) == ncols( A ) ) ) {
			std::cerr << "Mismatching dimensions: given vector of size " << size( x ) << " has nothing to do with either matrix dimension (" << nrows( A ) << " nor " << ncols( A ) << ").\n";
			return MISMATCH;
		}
		// no need for synchronisation, everything is local in reference implementation
		return eWiseLambda( f, A, args... );
	}

	/** @} */

} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_BLAS2''

