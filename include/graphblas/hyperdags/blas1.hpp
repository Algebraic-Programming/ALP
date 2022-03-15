
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
		typename OutputType, typename InputType, typename Coords 
	>
	RC set( Vector< OutputType, hyperdags, Coords > & x,
		const Vector< InputType, hyperdags, Coords > & y ){
		
		std::array< const void *, 1 > sources{ &y };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_FROM_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return set<descr>(internal::getVector(x), internal::getVector(y));	
	}
	
	
	
	template< 
		Descriptor descr = descriptors::no_operation, 
		typename T, typename U, typename Coords 
	>
	RC zip( Vector< std::pair< T, U >, hyperdags, Coords > & z,
		const Vector< T, hyperdags, Coords > & x,
		const Vector< U, hyperdags, Coords > & y,
		const typename std::enable_if< ! grb::is_object< T >::value &&
		! grb::is_object< U >::value, void >::type * const = NULL 
		) {

		std::array< const void *, 2 > sources{ &x, &y };
		std::array< const void *, 1 > destinations{ &z };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::ZIP,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()	
		);
		return zip<descr>(internal::getVector(z), internal::getVector(x),
			internal::getVector(y));		
	}

	template< 
		Descriptor descr = descriptors::no_operation, class OP, 
		typename OutputType, typename InputType1, typename InputType2, 
		typename Coords 
	>
	RC eWiseApply( Vector< OutputType, hyperdags, Coords > & z,
		const Vector< InputType1, hyperdags, Coords > & x,
		const Vector< InputType2, hyperdags, Coords > & y,
		const OP & op = OP(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && 
		! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
		grb::is_operator< OP >::value, void >::type * const = NULL 
		) {

		std::array< const void *, 2 > sources{ &x, &y };
		std::array< const void *, 1 > destinations{ &z };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::E_WISE_APPLY_VECTOR_VECTOR_VECTOR_OP,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return eWiseApply<descr>(internal::getVector(z), internal::getVector(x),
			internal::getVector(y), op);
	}
	
	template< 
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename InputType, typename IOType, typename Coords 
	>
	RC foldr( const Vector< InputType, hyperdags, Coords > & x,
		IOType & beta, // beta might be a pair 
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< InputType >::value && 
		! grb::is_object< IOType >::value && grb::is_monoid< Monoid >::value, void >::type * const = NULL 
		) {

		std::array< const void *, 2 > sources{ &x, &beta };
		std::array< const void *, 1 > destinations{ &beta };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDR_VECTOR_SCALAR_MONOID,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()	
		);
		return foldr<descr>(internal::getVector(x), beta, monoid);
	}
	
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename InputType, typename IOType, typename MaskType, typename Coords
	>
	RC foldl( IOType & x,
		const Vector< InputType, hyperdags, Coords > & y,
		const Vector< MaskType, hyperdags, Coords > & mask,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< IOType >::value &&
		! grb::is_object< InputType >::value && ! grb::is_object< MaskType >::value &&
		grb::is_monoid< Monoid >::value, void >::type * const = NULL 
		) {

		std::array< const void *, 3 > sources{&x, &y, &mask };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDL_SCALAR_VECTOR_MASK_MONOID,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return foldl<descr>(x, internal::getVector(y), internal::getVector(mask), monoid);
	}

	template< 
		typename Func, typename DataType, typename Coords 
		>
	RC eWiseLambda( const Func f, const Vector< DataType, hyperdags, Coords > & x ) {
		std::array< const void *, 1 > sources{ &x };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISELAMBDA,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return eWiseLambda( f, internal::getVector(x) );
	}

	
	template< 
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename fwd_iterator, typename Coords,
		class Dup = operators::right_assign< InputType >
		>
	RC buildVector( Vector< InputType, hyperdags, Coords > & x,
		fwd_iterator start, const fwd_iterator end,
		const IOMode mode, const Dup & dup = Dup()
	) { 
		std::array< const void *, 0 > sources;
		std::array< const void *, 1 > destinations{ &x};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::BUILD_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return buildVector<descr>( internal::getVector(x), start, end, mode, dup );
	}
	
	
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename fwd_iterator1, typename fwd_iterator2,
		typename Coords, class Dup = operators::right_assign< InputType >
		>
	RC buildVector( Vector< InputType, hyperdags, Coords > &x,
		fwd_iterator1 ind_start, const fwd_iterator1 ind_end,
		fwd_iterator2 val_start, const fwd_iterator2 val_end,
		const IOMode mode,
		const Dup &dup = Dup()
	) {
		std::array< const void *, 0 > sources;
		std::array< const void *, 1 > destinations{ &x};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::BUILD_VECTOR_WITH_VALUES,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return buildVector<descr>( internal::getVector(x), ind_start, ind_end, val_start, val_end, mode, dup  );
	}
	
	
	template< 
		typename DataType, typename Coords
		>
	size_t size( const Vector< DataType, hyperdags, Coords > & x ) {
	
		std::array< const void *, 1 > sources{ &x };
		std::array< const void *, 0 > destinations;
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SIZE,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return size (internal::getVector(x));
	}
	
		
	template< 
		Descriptor descr = descriptors::no_operation, class OP, typename OutputType,
		typename InputType1, typename InputType2, typename Coords 
		>
	RC eWiseApply( Vector< OutputType, hyperdags, Coords > & z,
		const Vector< InputType1, hyperdags, Coords > & x,
		const InputType2 beta,
		const OP & op = OP(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value 
		&& ! grb::is_object< InputType1 >::value 
		&& ! grb::is_object< InputType2 >::value 
		&& grb::is_operator< OP >::value,
		void >::type * const = NULL){
		
		std::array< const void *, 1 > sources{ &x };
		std::array< const void *, 1 > destinations{ &z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseApply <descr> (internal::getVector(z), internal::getVector(x), beta, op); 	
	}	
		
	template
		< Descriptor descr = descriptors::no_operation, class OP, typename OutputType,
		 typename InputType1, typename InputType2, typename Coords 
		>
	RC eWiseApply( Vector< OutputType, hyperdags, Coords > & z,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > & y,
		const OP & op = OP(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && 
		! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value 
		&& grb::is_operator< OP >::value, void >::type * const = NULL )	{
			
		std::array< const void *, 1 > sources{ &y };
		std::array< const void *, 1 > destinations{ &z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_BETA,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseApply <descr> (internal::getVector(z), alpha, internal::getVector(y), op); 	
	}	
	
	template
		< Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType,
		 typename MaskType, typename InputType1, typename InputType2, typename Coords
		>
	RC eWiseApply( Vector< OutputType, hyperdags, Coords > & z,
		const Vector< MaskType, hyperdags, Coords > & mask,
		const Vector< InputType1, hyperdags, Coords > & x,
		const InputType2 beta,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value &&
		! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_monoid< Monoid >::value,
		void >::type * const = NULL )	{
			
		std::array< const void *, 2 > sources{ &x, & mask};
		std::array< const void *, 1 > destinations{ &z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_VECTOR_BETA,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseApply <descr> (internal::getVector(z), internal::getVector(mask), internal::getVector(x), beta, monoid); 	
	}	
	
	template
		< Descriptor descr = descriptors::no_operation, class OP, typename OutputType,
		 typename MaskType, typename InputType1, typename InputType2, typename Coords 
		>
	RC eWiseApply( Vector< OutputType, hyperdags, Coords > & z,
		const Vector< MaskType, hyperdags, Coords > & mask,
		const Vector< InputType1, hyperdags, Coords > & x,
		const InputType2 beta,
		const OP & op = OP(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && 
		! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_operator< OP >::value,
		void >::type * const = NULL ) {
			
		std::array< const void *, 2 > sources{ &x, & mask};
		std::array< const void *, 1 > destinations{ &z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_VECTOR_VECTOR_BETA,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseApply <descr> (internal::getVector(z), internal::getVector(mask), internal::getVector(x), beta, op); 	
	}	
	
	template
		< Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType,
		 typename MaskType, typename InputType1, typename InputType2, typename Coords 
		>
	RC eWiseApply( Vector< OutputType, hyperdags, Coords > & z,
		const Vector< MaskType, hyperdags, Coords > & mask,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > & y,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value &&
		! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_monoid< Monoid >::value, void >::type * const = NULL ) {
		
		std::array< const void *, 2 > sources{& mask, &y,};
		std::array< const void *, 1 > destinations{ &z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_VECTOR_ALPHA_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseApply <descr> (internal::getVector(z), internal::getVector(mask), alpha, internal::getVector(y), monoid); 	
	}	
		
	
		
	template
		< Descriptor descr = descriptors::no_operation, class OP, typename OutputType,
		 typename MaskType, typename InputType1, typename InputType2, typename Coords 
		>
	RC eWiseApply( Vector< OutputType, hyperdags, Coords > & z,
		const Vector< MaskType, hyperdags, Coords > & mask,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > & y,
		const OP & op = OP(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value &&
		! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_operator< OP >::value, void >::type * const = NULL ){
		
		std::array< const void *, 2 > sources{& mask, &y,};
		std::array< const void *, 1 > destinations{ &z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_VECTOR_ALPHA_VECTOR_OP,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseApply <descr> (internal::getVector(z), internal::getVector(mask), alpha, internal::getVector(y), op); 	
	}
	
	template
		< Descriptor descr = descriptors::no_operation, class OP, typename OutputType,
		 typename MaskType, typename InputType1, typename InputType2, typename Coords
		  >
	RC eWiseApply( Vector< OutputType, hyperdags, Coords > & z,
		const Vector< MaskType, hyperdags, Coords > & mask,
		const Vector< InputType1, hyperdags, Coords > & x,
		const Vector< InputType2, hyperdags, Coords > & y,
		const OP & op = OP(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && 
		! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && 
		! grb::is_object< InputType2 >::value &&
		grb::is_operator< OP >::value, void >::type * const = NULL ){
		
		std::array< const void *, 3 > sources{& mask, &x, &y};
		std::array< const void *, 1 > destinations{ &z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_MASK_VECTOR_VECTOR_OP,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseApply <descr> (internal::getVector(z), internal::getVector(mask), internal::getVector(x), internal::getVector(y), op); 	
	}
	
	template< 
		Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType,
		typename InputType1, typename InputType2, typename Coords
		 >
	RC eWiseApply( Vector< OutputType, hyperdags, Coords > & z,
		const Vector< InputType1, hyperdags, Coords > & x,
		const InputType2 beta,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && 
		! grb::is_object< InputType1 >::value && 
		! grb::is_object< InputType2 >::value && 
		grb::is_monoid< Monoid >::value,
		void >::type * const = NULL ) {
		
		std::array< const void *, 1 > sources{& x};
		std::array< const void *, 1 > destinations{ &z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_SCALAR_MONOID,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());

		return eWiseApply <descr> (internal::getVector(z), internal::getVector(x), beta, monoid); 
	}
	
	template<
		 Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType,
		 typename InputType1, typename InputType2, typename Coords 
		 >
	RC eWiseApply( Vector< OutputType, hyperdags, Coords > & z,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > & y,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value &&
		! grb::is_object< InputType2 >::value && grb::is_monoid< Monoid >::value, void >::type * const = NULL ) {
		
		std::array< const void *, 1 > sources{&y};
		std::array< const void *, 1 > destinations{ &z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_SCALAR_VECTOR_MONOID,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseApply <descr> (internal::getVector(z), alpha, internal::getVector(y), monoid); 
		
		}
		
	template< 
		Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType,
		typename MaskType, typename InputType1, typename InputType2, typename Coords
		>
	RC eWiseApply( Vector< OutputType, hyperdags, Coords > & z,
		const Vector< MaskType, hyperdags, Coords > & mask,
		const Vector< InputType1, hyperdags, Coords > & x,
		const Vector< InputType2, hyperdags, Coords > & y,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && 
		! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value 
		&& grb::is_monoid< Monoid >::value, void >::type * const = NULL ) {
		
		std::array< const void *, 3 > sources{& mask, &x, &y};
		std::array< const void *, 1 > destinations{ &z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_MASK_VECTOR_VECTOR_MONOID,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseApply <descr> (internal::getVector(z), internal::getVector(mask), internal::getVector(x), internal::getVector(y), monoid); 
	}
	
	template
		< 
		Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType,
		typename InputType1, typename InputType2, typename Coords 
		>
	RC eWiseApply( Vector< OutputType, hyperdags, Coords > & z,
		const Vector< InputType1, hyperdags, Coords > & x,
		const Vector< InputType2, hyperdags, Coords > & y,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value &&
		! grb::is_object< InputType2 >::value && grb::is_monoid< Monoid >::value, void >::type * const = NULL ) {
		
		std::array< const void *, 2 > sources{&x, &y};
		std::array< const void *, 1 > destinations{ &z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_VECTOR_VECTOR_MONOID,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseApply <descr> (internal::getVector(z), internal::getVector(x), internal::getVector(y), monoid); 
		}
				
	template
		< 
		Descriptor descr = descriptors::no_operation, class Ring, typename InputType1,
		typename InputType2, typename InputType3, typename OutputType, typename MaskType, typename Coords 
		>
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & _z,
		const Vector< MaskType, hyperdags, Coords > & _m,
		const Vector< InputType1, hyperdags, Coords > & _a,
		const Vector< InputType2, hyperdags, Coords > & _x,
		const Vector< InputType3, hyperdags, Coords > & _y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && 
		! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value &&
		! grb::is_object< MaskType >::value, void >::type * const = NULL ){
		
		std::array< const void *, 4 > sources{&_m, &_a, &_x, &_y};
		std::array< const void *, 1 > destinations{ &_z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_MUL_ADD,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(_z), internal::getVector(_m), internal::getVector(_a),
		internal::getVector(_x), internal::getVector(_y), ring); 
		}
		
		
	template< 
		Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, 
		typename InputType2, typename InputType3, typename OutputType, typename MaskType, typename Coords
		>
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & _z,
		const Vector< MaskType, hyperdags, Coords > & _m,
		const Vector< InputType1, hyperdags, Coords > & _a,
		const Vector< InputType2, hyperdags, Coords > & _x,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && 
		! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value &&
		! grb::is_object< MaskType >::value, void >::type * const = NULL ){
		
		std::array< const void *, 3 > sources{&_m, &_a, &_x};
		std::array< const void *, 1 > destinations{ &_z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_MUL_ADD_FOUR_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(_z), internal::getVector(_m), internal::getVector(_a),
		internal::getVector(_x), gamma, ring); 
		}
		
		
		
	template< 
		Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, 
		typename InputType3, typename OutputType, typename Coords 
		>
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & _z,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > & _x,
		const Vector< InputType3, hyperdags, Coords > & _y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value &&
		! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value, void >::type * const = NULL ){
			
		std::array< const void *, 2 > sources{&_x, &_y};
		std::array< const void *, 1 > destinations{ &_z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_MUL_ADD_THREE_VECTOR_ALPHA,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(_z), alpha, internal::getVector(_x), internal::getVector(_y), ring); 
		
		}
		
		
	template< 
		Descriptor descr = descriptors::no_operation, class Ring, typename InputType1,
		 typename InputType2, typename InputType3, typename OutputType, typename Coords 
		>
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & _z,
		const Vector< InputType1, hyperdags, Coords > & _a,
		const InputType2 chi,
		const Vector< InputType3, hyperdags, Coords > & _y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value &&
		! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value, void >::type * const = NULL ){
			
		std::array< const void *, 2 > sources{&_a, &_y};
		std::array< const void *, 1 > destinations{ &_z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_MUL_ADD_THREE_VECTOR_CHI,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(_z), internal::getVector(_a), chi, internal::getVector(_y), ring); 
		}
		
	template< 
		Descriptor descr = descriptors::no_operation, class Ring, typename InputType1,
		typename InputType2, typename InputType3, typename OutputType, typename MaskType, typename Coords
		 >
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & _z,
		const Vector< MaskType, hyperdags, Coords > & _m,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > & _x,
		const Vector< InputType3, hyperdags, Coords > & _y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && 
		! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value && 
		! grb::is_object< MaskType >::value, void >::type * const = NULL ) {
			
		std::array< const void *, 3 > sources{&_m, &_x, &_y};
		std::array< const void *, 1 > destinations{ &_z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_MUL_ADD_FOUR_VECTOR_CHI,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(_z), internal::getVector(_m), alpha, internal::getVector(_x), internal::getVector(_y), ring); 
		}
		
		
		
	template< 
		Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, 
		typename InputType2, typename InputType3, typename OutputType, typename MaskType, typename Coords 
		>
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & _z,
		const Vector< MaskType, hyperdags, Coords > & _m,
		const Vector< InputType1, hyperdags, Coords > & _a,
		const InputType2 chi,
		const Vector< InputType3, hyperdags, Coords > & _y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && 
		! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value && 
		! grb::is_object< MaskType >::value, void >::type * const = NULL ) {
			
		std::array< const void *, 3 > sources{&_m, &_a, &_y};
		std::array< const void *, 1 > destinations{ &_z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_MUL_ADD_FOUR_VECTOR_CHI_RING,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(_z), internal::getVector(_m), internal::getVector(_a), chi,  internal::getVector(_y), ring); 
		}
		
		
	template<
		 Descriptor descr = descriptors::no_operation, class Ring, typename InputType1,
		 typename InputType2, typename InputType3, typename OutputType, typename MaskType, typename Coords
		>
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & _z,
		const Vector< MaskType, hyperdags, Coords > & _m,
		const Vector< InputType1, hyperdags, Coords > & _a,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && 
		! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value && 
		! grb::is_object< MaskType >::value, void >::type * const = NULL ) {
		
		std::array< const void *, 2 > sources{&_m, &_a};
		std::array< const void *, 1 > destinations{ &_z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_MUL_ADD_THREE_VECTOR_BETA,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(_z), internal::getVector(_m), internal::getVector(_a), beta,  gamma, ring); 
		
		}
		
		
	template< 
		Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2,
		typename InputType3, typename OutputType, typename MaskType, typename Coords 
		>
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & _z,
		const Vector< MaskType, hyperdags, Coords > & _m,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > & _x,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && 
		! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value && 
		! grb::is_object< MaskType >::value, void >::type * const = NULL ) {
		
		std::array< const void *, 2 > sources{&_m, &_x};
		std::array< const void *, 1 > destinations{ &_z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_MUL_ADD_THREE_VECTOR_ALPHA_GAMMA,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(_z), internal::getVector(_m), alpha, internal::getVector(_x), gamma, ring); 
		
		}
		
		
	template< 
		Descriptor descr = descriptors::no_operation, class Ring, typename OutputType, typename MaskType,
		typename InputType1, typename InputType2, typename InputType3, typename Coords 
		>
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & z,
		const Vector< MaskType, hyperdags, Coords > & m,
		const InputType1 alpha,
		const InputType2 beta,
		const Vector< InputType3, hyperdags, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && 
		! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value && 
		! grb::is_object< MaskType >::value, void >::type * const = NULL ) {
		
		std::array< const void *, 2 > sources{&m, &y};
		std::array< const void *, 1 > destinations{ &z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_MUL_ADD_TWO_VECTOR_ALPHA_BETA,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(z), internal::getVector(m), alpha, beta, internal::getVector(y), ring); 
		
		}
		
	template< 
		Descriptor descr = descriptors::no_operation, class Ring, typename OutputType, 
		typename MaskType, typename InputType1, typename InputType2, typename InputType3, typename Coords 
		>
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & z,
		const Vector< MaskType, hyperdags, Coords > & m,
		const InputType1 alpha,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && 
		! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && 
		! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		
		std::array< const void *, 1 > sources{&m};
		std::array< const void *, 1 > destinations{ &z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_MUL_ADD_TWO_VECTOR_ALPHA_BETA_GAMMA,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(z), internal::getVector(m), alpha, beta, gamma, ring); 
		
		
		}
		
	template< 
		Descriptor descr = descriptors::no_operation, typename T, typename U, typename Coords 
	>
	RC unzip( Vector< T, hyperdags, Coords > & x,
		Vector< U, hyperdags, Coords > & y,
		const Vector< std::pair< T, U >, hyperdags, Coords > & in,
		const typename std::enable_if< ! grb::is_object< T >::value && 
		! grb::is_object< U >::value, void >::type * const = NULL ) {
		
		std::array< const void *, 2 > sources{&x,&y};
		std::array< const void *, 1 > destinations{ &in};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::UNZIP_VECTOR_VECTOR_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return unzip <descr> (internal::getVector(x), internal::getVector(y), internal::getVector(in)); 	
		}
		
		
	template< 
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename Coords 
	>
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & _z,
		const Vector< InputType1, hyperdags, Coords > & _a,
		const Vector< InputType2, hyperdags, Coords > & _x,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! 
		grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && 
		! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value, void >::type * const = NULL )
		{
		std::array< const void *, 2 > sources{& _a,& _x};
		std::array< const void *, 1 > destinations{ & _z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMULADD_VECTOR_VECTOR_VECTOR_GAMMA_RING,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(_z), internal::getVector(_a), internal::getVector(_x), gamma, ring); 	
		}
	
	
	
	template< 
		Descriptor descr = descriptors::no_operation, class Ring, typename InputType1,
		typename InputType2, typename InputType3, typename OutputType, typename Coords 
	>
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & _z,
		const Vector< InputType1, hyperdags, Coords > & _a,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && 
		! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && 
		grb::is_semiring< Ring >::value, void >::type * const = NULL ) 
		{
		std::array< const void *, 1 > sources{& _a};
		std::array< const void *, 1 > destinations{ & _z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMULADD_VECTOR_VECTOR_BETA_GAMMA_RING,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(_z), internal::getVector(_a), beta, gamma, ring); 
		
		}
		
	template< 
		Descriptor descr = descriptors::no_operation, class Ring, typename InputType1,
		typename InputType2, typename InputType3, typename OutputType, typename Coords 
	>
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & _z,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > & _x,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && 
		! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
		 void >::type * const = NULL ) {
		 
		std::array< const void *, 1 > sources{& _x};
		std::array< const void *, 1 > destinations{ & _z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMULADD_VECTOR_ALPHA_VECTOR_GAMMA_RING,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(_z), alpha, internal::getVector(_x), gamma, ring); 
		} 
		
		
		
	template< 
		Descriptor descr = descriptors::no_operation, class Ring, typename OutputType,
	 	typename InputType1, typename InputType2, typename InputType3, typename Coords 
	 >
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & z,
		const InputType1 alpha,
		const InputType2 beta,
		const Vector< InputType3, hyperdags, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! 
		grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value 
		&& grb::is_semiring< Ring >::value, void >::type * const = NULL )
		{
		
		std::array< const void *, 1 > sources{& y};
		std::array< const void *, 1 > destinations{ & z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMULADD_VECTOR_ALPHA_BETA_VECTOR_RING,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(z), alpha, beta, internal::getVector(y), ring); 

		}
		
		
	template< 
		Descriptor descr = descriptors::no_operation, class Ring, typename OutputType,
		typename InputType1, typename InputType2, typename InputType3, typename Coords 
	>
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & z,
		const InputType1 alpha,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value &&
		! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value, void >::type * const = NULL )
		{
		
		std::array< const void *, 0 > sources{};
		std::array< const void *, 1 > destinations{ & z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMULADD_VECTOR_ALPHA_BETA_GAMMA_RING,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(z), alpha, beta, gamma, ring); 
		}
		
	template< 
		Descriptor descr = descriptors::no_operation, class Ring, typename InputType1,
		typename InputType2, typename InputType3, typename OutputType, typename Coords 
	>
	RC eWiseMulAdd( Vector< OutputType, hyperdags, Coords > & _z,
		const Vector< InputType1, hyperdags, Coords > & _a,
		const Vector< InputType2, hyperdags, Coords > & _x,
		const Vector< InputType3, hyperdags, Coords > & _y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value &&
		! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value, void >::type * const = NULL )	
		{
		std::array< const void *, 3 > sources{ &_a, &_x, &_y};
		std::array< const void *, 1 > destinations{ & _z};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMULADD_VECTOR_VECTOR_VECTOR_VECTOR_RING,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end());
		return eWiseMulAdd <descr> (internal::getVector(_z), internal::getVector(_a), internal::getVector(_x), internal::getVector(_y), ring); 
		}
		
		
		
		
		
		
		
		
} // end namespace grb

#endif

