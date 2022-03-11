
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
 * Implements the BLAS-3 API for the hypergraphs backend.
 *
 * @author A. Karanasiou
 * @date 3rd of March, 2022
 */

#ifndef _H_GRB_HYPERDAGS_BLAS3
#define _H_GRB_HYPERDAGS_BLAS3

#include <graphblas/matrix.hpp>

#include <graphblas/hyperdags/init.hpp>


namespace grb {
	template< typename InputType >
	RC clear( Matrix< InputType, hyperdags > & A ) noexcept {
		// delegate
		return clear(internal::getMatrix(A));
	}
	
	
	
	
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		class MulMonoid
	>
	RC eWiseApply( Matrix< OutputType, hyperdags > &C,
		const Matrix< InputType1, hyperdags > &A,
		const Matrix< InputType2, hyperdags > &B,
		const MulMonoid &mulmono,
		const PHASE phase = NUMERICAL,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< MulMonoid >::value,
		void >::type * const = NULL
	){ 
		std::array< const void *, 2 > sources{ &A, &B };
		std::array< const void *, 1 > destinations{&C};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_MATRIX_MATRIX_MATRIX_MULMONOID_PHASE,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return eWiseApply<descr>( internal::getMatrix( C ), internal::getMatrix( A ),
		internal::getMatrix( B ), mulmono, phase );
	}



	template<
		Descriptor descr = grb::descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		class Operator
	>
	RC eWiseApply( Matrix< OutputType, hyperdags > &C,
		const Matrix< InputType1, hyperdags > &A,
		const Matrix< InputType2, hyperdags > &B,
		const Operator &mulOp,
		const PHASE phase = NUMERICAL,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value,
		void >::type * const = NULL
	) {
		std::array< const void *, 2 > sources{ &A, &B };
		std::array< const void *, 1 > destinations{&C};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_MATRIX_MATRIX_MATRIX_OPERATOR_PHASE,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return eWiseApply<descr>( internal::getMatrix( C ), internal::getMatrix( A ),
		internal::getMatrix( B ), mulOp, phase );
	}


	template< 
		Descriptor descr = descriptors::no_operation,
		 typename OutputType, typename InputType
	>
	RC set( Matrix< OutputType, hyperdags > & C,
		const Matrix< InputType, hyperdags > & A ) 
		{
		std::array< const void *, 1 > sources{ &A };
		std::array< const void *, 1 > destinations{ &C };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_MATRIX_MATRIX,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return set<descr>( internal::getMatrix( C ), internal::getMatrix( A ) );
	}
	
	template< 
		Descriptor descr = descriptors::no_operation, 
		typename OutputType, typename InputType1, typename InputType2 
	>
	RC set( Matrix< OutputType, hyperdags > & C, const Matrix< InputType1, hyperdags > & A, 
		const InputType2 & val )
		{
		std::array< const void *, 1 > sources{ &A };
		std::array< const void *, 1 > destinations{ &C };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_MATRIX_MATRIX_INPUT2,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return set<descr>( internal::getMatrix( C ), internal::getMatrix( A ), val );
		}


	template<
		bool A_is_mask,
		Descriptor descr,
		typename OutputType, typename InputType1, typename InputType2 = const OutputType
	>
	RC set( Matrix< OutputType, hyperdags > &C,
			const Matrix< InputType1, hyperdags > &A,
			const InputType2 * __restrict__ id = NULL)
	{
		std::array< const void *, 1 > sources{ &A };
		std::array< const void *, 1 > destinations{ &C };
		internal::hyperdags::generator.addOperation(
				internal::hyperdags::SET_MATRIX_MATRIX_DOUBLE,
				sources.begin(), sources.end(),
				destinations.begin(), destinations.end()
		);
	return set<descr>( internal::getMatrix( C ), internal::getMatrix( A ) );	
	}
	
	
	template< 
		Descriptor descr = descriptors::no_operation, typename OutputType,
		typename InputType1, typename InputType2, class Semiring
	>
	RC mxm( Matrix< OutputType, hyperdags > & C,
		const Matrix< InputType1, hyperdags > & A,
		const Matrix< InputType2, hyperdags > & B,
		const Semiring & ring = Semiring(),
		const PHASE & phase = NUMERICAL,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value &&
		! grb::is_object< InputType2 >::value && grb::is_semiring< Semiring >::value, void >::type * const = NULL )
	{
	
		std::array< const void *, 2 > sources{ &A, &B};
		std::array< const void *, 1 > destinations{ &C };
		internal::hyperdags::generator.addOperation(
				internal::hyperdags::MXM_MATRIX_MATRIX_MATRIX_SEMIRING,
				sources.begin(), sources.end(),
				destinations.begin(), destinations.end()
		);
	return mxm<descr>( internal::getMatrix( C ), internal::getMatrix( A ),  internal::getMatrix( B ), ring, phase );
	}
		
	
	
	template< 
		Descriptor descr = grb::descriptors::no_operation, typename OutputType, 
		typename InputType1, typename InputType2, class Operator, class Monoid 
	>
	RC mxm( Matrix< OutputType, hyperdags > & C,
		const Matrix< InputType1, hyperdags > & A,
		const Matrix< InputType2, hyperdags > & B,
		const Operator & mulOp,
		const Monoid & addM,
		const PHASE & phase = NUMERICAL,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! 
		grb::is_object< InputType2 >::value && grb::is_operator< Operator >::value && grb::is_monoid< Monoid >::value, void >::type * const = NULL )
	{
		std::array< const void *, 2 > sources{ &A, &B};
		std::array< const void *, 1 > destinations{ &C };
		internal::hyperdags::generator.addOperation(
				internal::hyperdags::MXM_MATRIX_MATRIX_MATRIX_MONOID,
				sources.begin(), sources.end(),
				destinations.begin(), destinations.end()
		);
	
	
	return mxm<descr>( internal::getMatrix( C ), internal::getMatrix( A ),  internal::getMatrix( B ), mulOp, addM, phase );

	}
	
	
	template< Descriptor descr = descriptors::no_operation, typename InputType1,
		 typename InputType2, typename OutputType, typename Coords, class Operator 
	>
	RC outer( Matrix< OutputType, hyperdags > & A,
		const Vector< InputType1, hyperdags, Coords > & u,
		const Vector< InputType2, hyperdags, Coords > & v,
		const Operator & mul = Operator(),
		const PHASE & phase = NUMERICAL,
		const typename std::enable_if< grb::is_operator< Operator >::value && ! grb::is_object< InputType1 >::value && 
		! grb::is_object< InputType2 >::value && ! grb::is_object< OutputType >::value, void >::type * const = NULL ) {
		
		std::array< const void *, 2 > sources{ &u, &v};
		std::array< const void *, 1 > destinations{ &A };
		internal::hyperdags::generator.addOperation(
				internal::hyperdags::OUTER,
				sources.begin(), sources.end(),
				destinations.begin(), destinations.end()
		);
		return outer<descr>( internal::getMatrix( A ), internal::getVector( u ),  internal::getVector( v ), mul, phase);
	}
	
	
} // end n1amespace grb

#endif

