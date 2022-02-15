
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
 * @date 31st of January, 2022
 */

#ifndef _H_GRB_HYPERDAGS_BLAS1
#define _H_GRB_HYPERDAGS_BLAS1

#include <graphblas/vector.hpp>

#include <graphblas/hyperdags/init.hpp>


namespace grb {

	template< typename DataType, typename Coords >
	size_t nnz( const Vector< DataType, hyperdags, Coords > &x ) noexcept {
		std::array< const void *, 1 > sources{ &x };
		std::array< const void *, 0 > destinations;
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::NNZ_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return nnz( internal::getVector( x ) );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename Coords,
		typename T
	>
	RC set( Vector< DataType, hyperdags, Coords > &x, const T val,
		const typename std::enable_if< !grb::is_object< DataType >::value &&
			!grb::is_object< T >::value,
		void >::type * const = nullptr
	) {
		std::cout << "\t Called Set(vectir, scalar) \n";
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SET,
			&x
		);
		return set<descr>( internal::getVector( x ), val );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType,
		typename T, typename Coords
	>
	RC setElement( Vector< DataType, hyperdags, Coords > &x,
		const T val,
		const size_t i,
		const typename std::enable_if<
			!grb::is_object< DataType >::value &&
			!grb::is_object< T >::value,
		void >::type * const = nullptr
	) {
		// always force input scalar to be a new source
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&val
		);
		std::array< const void *, 2 > sources{ &x, &val };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_VECTOR_ELEMENT,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return setElement<descr>( internal::getVector( x ),
			val, i);
	}

	template< typename DataType, typename Coords >
	RC clear( Vector< DataType, hyperdags, Coords > &x ) {
		std::array< const void *, 1 > sources{ &x };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::CLEAR_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return clear( internal::getVector( x ) );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class AddMonoid, class AnyOp,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC dot( OutputType &z,
		const Vector< InputType1, hyperdags, Coords > &x,
		const Vector< InputType2, hyperdags, Coords > &y,
		const AddMonoid &addMonoid = AddMonoid(),
		const AnyOp &anyOp = AnyOp(),
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< AddMonoid >::value &&
			grb::is_operator< AnyOp >::value,
		void >::type * const = nullptr
	) {
		// always force input scalar to be a new source
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&z
		);
		std::array< const void *, 3 > sources{ &z, &x, &y };
		std::array< const void *, 1 > destinations{ &z };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::DOT,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return dot<descr>( z,
			internal::getVector(x), internal::getVector(y),
			addMonoid, anyOp
		);
	}
	
	// myadd
	template< 
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename MaskType, typename InputType,
		typename Coords
		>
	RC set( Vector< OutputType, hyperdags, Coords > & x,
		const Vector< MaskType, hyperdags, Coords > & mask,
		const Vector< InputType, hyperdags, Coords > & y,
		const typename std::enable_if< ! grb::is_object< OutputType >::value &&
			! grb::is_object< MaskType >::value &&
			! grb::is_object< InputType >::value,
		void >::type * const = NULL) {
		
		std::cout << "\t Called Set(vectir, vector, vector) \n";
		std::array< const void *, 2 > sources{ &mask, &y };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_USING_MASK_AND_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return set<descr>(internal::getVector(x),
			internal::getVector(mask), internal::getVector(y)
		);	
	}	

	template<
		Descriptor descr = descriptors::no_operation, 
		typename DataType, typename MaskType, typename T, 
		typename Coords
		>
	RC set( Vector< DataType, hyperdags, Coords > & x,
		const Vector< MaskType, hyperdags, Coords > & m,
		const T val,
		const typename std::enable_if< ! grb::is_object< DataType >::value && 
		! grb::is_object< T >::value, void >::type * const = NULL ) 	{

		std::cout << "\t Called Set(vector, vector, scalar) \n";
		std::array< const void *, 2 > sources{ &m };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_USING_MASK_AND_SCALAR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return set<descr>(internal::getVector(x), internal::getVector(m), val);	
		
	}
	
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType, typename Coords >
	RC set( Vector< OutputType, hyperdags, Coords > & x,
		const Vector< InputType, hyperdags, Coords > & y ){
		
		std::cout << "\t Called Set(vector, vector) \n";
		
		std::array< const void *, 1 > sources{ &y };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_FROM_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return set<descr>(internal::getVector(x), internal::getVector(y));	
	}

} // end namespace grb

#endif

