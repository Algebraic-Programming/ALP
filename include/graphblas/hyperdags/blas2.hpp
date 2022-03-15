
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
 * @file
 *
 * Implements the BLAS-2 API for the hypergraphs backend.
 *
 * @author A. Karanasiou
 * @date 3rd of March, 2022
 */

#ifndef _H_GRB_HYPERDAGS_BLAS2
#define _H_GRB_HYPERDAGS_BLAS2

#include <graphblas/matrix.hpp>

#include <graphblas/hyperdags/init.hpp>


namespace grb {
		
	template< typename InputType >
	size_t nrows( const Matrix< InputType, hyperdags > & A ) noexcept {
		return nrows(internal::getMatrix(A));
	}
	
	template< typename InputType >
	size_t ncols( const Matrix< InputType, hyperdags > & A ) noexcept {
		return ncols(internal::getMatrix(A));
	}
	
	template< typename InputType >
	size_t nnz( const Matrix< InputType, hyperdags > & A ) noexcept {
		return nnz(internal::getMatrix(A));
	}
	
	template<
		 typename InputType 
	>
	RC resize( Matrix< InputType, hyperdags > & A, const size_t new_nz ) noexcept {
		return resize(internal::getMatrix(A), new_nz);
	}
	
	template< 
		Descriptor descr = descriptors::no_operation, class Ring, typename IOType,
		typename InputType1, typename InputType2, typename InputType3, typename Coords 
	>
	RC vxm( Vector< IOType, hyperdags, Coords > & u,
		const Vector< InputType3, hyperdags, Coords > & mask,
		const Vector< InputType1, hyperdags, Coords > & v,
		const Matrix< InputType2, hyperdags > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		
		std::array< const void *, 3 > sources{ &mask, &v, &A};
		std::array< const void *, 1 > destinations{ &u };
		internal::hyperdags::generator.addOperation(
				internal::hyperdags::VXM_VECTOR_VECTOR_VECTOR_MATRIX,
				sources.begin(), sources.end(),
				destinations.begin(), destinations.end()
		);
		return vxm< descr>( internal::getVector(u), internal::getVector(mask), internal::getVector(v), internal::getMatrix(A), ring );
	}
	
	
	template< 
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename Coords 
	>
	RC vxm( Vector< IOType, hyperdags, Coords > & u,
		const Vector< InputType3, hyperdags, Coords > & mask,
		const Vector< InputType1, hyperdags, Coords > & v,
		const Matrix< InputType2, hyperdags > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
			
		std::array< const void *, 3 > sources{ &mask, &v, &A};
		std::array< const void *, 1 > destinations{ &u };
		internal::hyperdags::generator.addOperation(
				internal::hyperdags::VXM_VECTOR_VECTOR_VECTOR_MATRIX_ADD_MUL,
				sources.begin(), sources.end(),
				destinations.begin(), destinations.end()
		);
		
		return vxm< descr>( internal::getVector(u), internal::getVector(mask), internal::getVector(v), internal::getMatrix(A), add, mul );
	}

	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2,
		typename Coords 	
	>
	RC vxm( Vector< IOType, hyperdags, Coords > & u,
		const Vector< InputType1, hyperdags, Coords > & v,
		const Matrix< InputType2, hyperdags > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		
		std::array< const void *, 2 > sources{ &v, &A};
		std::array< const void *, 1 > destinations{ &u };
		internal::hyperdags::generator.addOperation(
				internal::hyperdags::VXM_VECTOR_VECTOR_MATRIX_RING,
				sources.begin(), sources.end(),
				destinations.begin(), destinations.end()
		);
		return vxm< descr >( internal::getVector(u), internal::getVector(v), internal::getMatrix(A), ring );
	}

} // end namespace grb

#endif

