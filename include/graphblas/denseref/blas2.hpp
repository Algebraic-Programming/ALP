
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
	template< typename InputType, typename InputStructure, typename InputStorage, typename InputView >
	size_t nrows( const StructuredMatrix< InputType, InputStructure, InputStorage, InputView, reference_dense > & A ) noexcept {
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
	template< typename InputType, typename InputStructure, typename InputStorage, typename InputView >
	size_t ncols( const StructuredMatrix< InputType, InputStructure, InputStorage, InputView, reference_dense > & A ) noexcept {
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
	size_t nnz( const StructuredMatrix< InputType, InputStructure, InputStorage, InputView, reference_dense > & A ) noexcept {
		return A.nz;
	}

	/**
	 * Resizes the nonzero capacity of this matrix. Any current contents of the
	 * matrix are \em not retained.
	 *
	 * The dimension of this matrix is fixed. Only the number of nonzeroes that
	 * may be stored can change. If the matrix has row or column dimension size
	 * zero, all calls to this function are ignored. A request for less capacity
	 * than currently already may be allocated, may be ignored by the
	 * implementation.
	 *
	 * @param[in] nonzeroes The number of nonzeroes this matrix is to contain.
	 *
	 * @return OUTOFMEM When no memory could be allocated to store this matrix.
	 * @return PANIC    When allocation fails for any other reason.
	 * @return SUCCESS  When a valid GraphBLAS matrix has been constructed.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -$ This function consitutes \f$ \mathcal{O}(\mathit{nz} \f$ work.
	 *        -# This function allocates \f$ \mathcal{O}(\mathit{nz}+m+n+1) \f$
	 *           bytes of dynamic memory.
	 *        -# This function will likely make system calls.
	 * \endparblock
	 *
	 * \warning This is an expensive function. Use sparingly and only when
	 *          absolutely necessary
	 */
	template< typename InputType >
	RC resize( StructuredMatrix< InputType, InputStructure, InputStorage, InputView, reference_dense > & A, const size_t new_nz ) noexcept {
		// delegate
		return A.resize( new_nz );
	}

	/** \internal Delegates to fully masked variant */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType, typename IOView, typename IOStorage,
		typename InputType1, typename InputType1, typename InputView1, typename InputStorage1,
		typename InputType2, typename InputStructure2, typename InputStorage2, typename InputView2,
		typename InputType3, typename InputType3, typename InputView3, typename InputStorage3,
		typename Coords >
	RC vxm( VectorView< IOType, IOStructure, IOStorage, IOView, reference_dense, Coords > & u,
		const VectorView< InputType3, InputStructure3, InputStorage3, InputView3, reference_dense, Coords > & mask,
		const VectorView< InputType1, InputStructure1, InputStorage1, InputView1, reference_dense, Coords > & v,
		const StructuredMatrix< InputType2, InputStructure2, InputStorage2, InputView2, reference_dense > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const VectorView< bool, reference, Coords > empty_mask( 0 );
		return vxm< descr, true, false >( u, mask, v, empty_mask, A, ring );
	}

	/** \internal Delegates to fully masked variant */
	template< Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType, typename IOView, typename IOStorage,
		typename InputType1, typename InputView1, typename InputStorage1,
		typename InputType2, typename InputStructure2, typename InputStorage2, typename InputView2,
		typename InputType3, typename InputView3, typename InputStorage3,
		typename Coords >
	RC vxm( VectorView< IOType, IOStructure, IOStorage, IOView, reference_dense, Coords > & u,
		const VectorView< InputType3, InputStructure3, InputStorage3, InputView3, reference_dense, Coords > & mask,
		const VectorView< InputType1, InputStructure1, InputStorage1, InputView1, reference_dense, Coords > & v,
		const StructuredMatrix< InputType2, InputStructure2, InputStorage2, InputView2, reference_dense > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		const grb::VectorView< bool, reference, Coords > empty_mask( 0 );
		return vxm< descr, true, false >( u, mask, v, empty_mask, A, add, mul );
	}

	/** \internal Delegates to vxm_generic. */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class Ring,
		typename IOType, typename IOView, typename IOStorage,
		typename InputType1, typename InputView1, typename InputStorage1,
		typename InputType2, typename InputStructure2, typename InputStorage2, typename InputView2,
		typename InputType3, typename InputView3, typename InputStorage3,
		typename InputType4, typename InputView4, typename InputStorage4,
		typename Coords >
	RC vxm( VectorView< IOType, IOStructure, IOStorage, IOView, reference_dense, Coords > & u,
		const VectorView< InputType3, InputStructure3, InputStorage3, InputView3, reference_dense, Coords > & mask,
		const VectorView< InputType1, InputStructure1, InputStorage1, InputView1, reference_dense, Coords > & v,
		const VectorView< InputType4, InputStructure4, InputStorage4, InputView4, reference_dense, Coords > & v_mask,
		const StructuredMatrix< InputType2, InputStructure2, InputStorage2, InputView2, reference_dense > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4, typename IOView, typename IOStorage,
		typename InputType1 = typename Ring::D1, typename InputView1, typename InputStorage1,
		typename InputType2 = typename Ring::D2, typename InputStructure2, typename InputStorage2, typename InputView2,
		typename Coords >
	RC vxm( VectorView< IOType, IOStructure, IOStorage, IOView, reference_dense, Coords > & u,
		const VectorView< InputType1, InputStructure1, InputStorage1, InputView1, reference_dense, Coords > & v,
		const StructuredMatrix< InputType2, InputStructure2, InputStorage2, InputView2, reference_dense > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const VectorView< bool, reference_dense, Coords > empty_mask( 0 );
		return vxm< descr, false, false >( u, empty_mask, v, empty_mask, A, ring );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation, class AdditiveMonoid, class MultiplicativeOperator, typename IOType, typename InputType1, typename InputType2, typename Coords >
	RC vxm( VectorView< IOType, IOStructure, IOStorage, IOView, reference_dense, Coords > & u,
		const VectorView< InputType1, InputStructure1, InputStorage1, InputView1, reference_dense, Coords > & v,
		const StructuredMatrix< InputType2, InputStructure2, InputStorage2, InputView2, reference_dense > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		const grb::VectorView< bool, reference_dense, Coords > empty_mask( 0 );
		return vxm< descr, false, false >( u, empty_mask, v, empty_mask, A, add, mul );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType, typename IOView, typename IOStorage,
		typename InputType1, typename InputView1, typename InputStorage1,
		typename InputType2, typename InputStructure2, typename InputStorage2, typename InputView2,
		typename InputType3 = bool, typename InputView3, typename InputStorage3,
		typename Coords >
	RC mxv( VectorView< IOType, IOStructure, IOStorage, IOView, reference_dense, Coords > & u,
		const VectorView< InputType3, InputStructure3, InputStorage3, InputView3, reference_dense, Coords > & mask,
		const StructuredMatrix< InputType2, InputStructure2, InputStorage2, InputView2, reference_dense > & A,
		const VectorView< InputType1, InputStructure1, InputStorage1, InputView1, reference_dense, Coords > & v,
		const Ring & ring,
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const VectorView< bool, reference_dense, Coords > empty_mask( 0 );
		return mxv< descr, true, false >( u, mask, A, v, empty_mask, ring );
	}

	/** \internal Delegates to vxm_generic */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class Ring,
		typename IOType, typename IOView, typename IOStorage,
		typename InputType1, typename InputView1, typename InputStorage1,
		typename InputType2, typename InputStructure2, typename InputStorage2, typename InputView2,
		typename InputType3, typename InputView3, typename InputStorage3,
		typename InputType4, typename InputView4, typename InputStorage4,
		typename Coords >
	RC mxv( VectorView< IOType, IOStructure, IOStorage, IOView, reference_dense, Coords > & u,
		const VectorView< InputType3, InputStructure3, InputStorage3, InputView3, reference_dense, Coords > & mask,
		const StructuredMatrix< InputType2, InputStructure2, InputStorage2, InputView2, reference_dense > & A,
		const VectorView< InputType1, InputStructure1, InputStorage1, InputView1, reference_dense, Coords > & v,
		const VectorView< InputType4, InputStructure4, InputStorage4, InputView4, reference_dense, Coords > & v_mask,
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
		typename IOType = typename Ring::D4, typename IOView, typename IOStorage,
		typename InputType1 = typename Ring::D1, typename InputStructure2, typename InputStorage2, typename InputView2,
		typename InputType2 = typename Ring::D2, typename InputView1, typename InputStorage1,
		typename Coords >
	RC mxv( VectorView< IOType, IOStructure, IOStorage, IOView, reference_dense, Coords > & u,
		const StructuredMatrix< InputType2, InputStructure2, InputStorage2, InputView2, reference_dense > & A,
		const VectorView< InputType1, InputStructure1, InputStorage1, InputView1, reference_dense, Coords > & v,
		const Ring & ring,
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const VectorView< bool, view::Original< void >, structure::full, reference_dense, Coords > empty_mask( 0 );
		return mxv< descr, false, false >( u, empty_mask, A, v, empty_mask, ring );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation, class AdditiveMonoid, class MultiplicativeOperator, typename IOType, typename InputType1, typename InputType2, typename Coords >
	RC mxv( VectorView< IOType, IOStructure, IOStorage, IOView, reference_dense, Coords > & u,
		const StructuredMatrix< InputType2, InputStructure2, InputStorage2, InputView2, reference_dense > & A,
		const VectorView< InputType1, InputStructure1, InputStorage1, InputView1, reference_dense, Coords > & v,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		const grb::VectorView< bool, view::Original< void >, structure::full, reference_dense, Coords > empty_mask( 0 );
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
		typename IOType, typename IOView, typename IOStorage,
		typename InputType1, typename InputView1, typename InputStorage1,
		typename InputType2, typename InputStructure2, typename InputStorage2, typename InputView2,
		typename InputType3, typename InputView3, typename InputStorage3,
		typename InputType4, typename InputView4, typename InputStorage4,
		typename Coords >
	RC vxm( VectorView< IOType, IOStructure, IOStorage, IOView, reference_dense, Coords > & u,
		const VectorView< InputType3, InputStructure3, InputStorage3, InputView3, reference_dense, Coords > & mask,
		const VectorView< InputType1, InputStructure1, InputStorage1, InputView1, reference_dense, Coords > & v,
		const VectorView< InputType4, InputStructure4, InputStorage4, InputView4, reference_dense, Coords > & v_mask,
		const StructuredMatrix< InputType2, InputStructure2, InputStorage2, InputView2, reference_dense > & A,
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
		typename IOType, typename IOView, typename IOStorage,
		typename InputType1, typename InputView1, typename InputStorage1,
		typename InputType2, typename InputStructure2, typename InputStorage2, typename InputView2,
		typename InputType3, typename InputView3, typename InputStorage3,
		typename InputType4, typename InputView4, typename InputStorage4,
		typename Coords >
	RC mxv( VectorView< IOType, IOStructure, reference_dense, Coords > & u,
		const VectorView< InputType3, InputStructure3, reference_dense, Coords > & mask,
		const StructuredMatrix< InputType2, InputStructure2, InputStorage2, InputView2, reference_dense > & A,
		const VectorView< InputType1, InputStructure1, reference_dense, Coords > & v,
		const VectorView< InputType4, InputStructure4, reference_dense, Coords > & v_mask,
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
	template< class ActiveDistribution, typename Func, typename DataType, typename Structure, typename Storage, typename View>
	RC eWiseLambda( const Func f,
		const StructuredMatrix< DataType, Structure, Storage, View, reference_dense > & A,
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
		typename DataType1, typename Structure1, typename Storage1, typename View1,
		typename DataType2, typename View2, typename Storage2, typename Coords, typename... Args >
	RC eWiseLambda( const Func f,
		const StructuredMatrix< DataType1, Structure1, Storage1, View1, reference_dense > & A,
		const VectorView< DataType2, DataStructure2, Storage2, View2, reference_dense, Coords > x, Args... args ) {
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

