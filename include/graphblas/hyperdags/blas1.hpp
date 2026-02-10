
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

/**
 * @file
 *
 * Provides the "level-1" primitives for the HyperDAGs backend
 *
 * @author A. N. Yzelman
 * @date 31st of January, 2022
 */

#ifndef _H_GRB_HYPERDAGS_BLAS1
#define _H_GRB_HYPERDAGS_BLAS1

#include <graphblas/vector.hpp>

#include <graphblas/hyperdags/init.hpp>

#include <array>


namespace grb {

	template<
		Descriptor descr = descriptors::no_operation,
		class AddMonoid, class AnyOp,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC dot(
		OutputType &z,
		const Vector< InputType1, hyperdags, Coords > &x,
		const Vector< InputType2, hyperdags, Coords > &y,
		const AddMonoid &addMonoid = AddMonoid(),
		const AnyOp &anyOp = AnyOp(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< AddMonoid >::value &&
			grb::is_operator< AnyOp >::value,
		void >::type * const = nullptr
	) {
		const RC ret = dot< descr >(
			z, internal::getVector(x), internal::getVector(y),
			addMonoid, anyOp, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&z
		);
		std::array< const void *, 1 > sourcesP{ &z };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) )
		};
		std::array< uintptr_t, 0 > destinations{};
		// NOTE scalar output is ignored
		//std::array< const void *, 1 > destinationsP{ &z };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::DOT,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		class Semiring, typename Coords
	>
	RC dot(
		OutputType &z,
		const Vector< InputType1, hyperdags, Coords > &x,
		const Vector< InputType2, hyperdags, Coords > &y,
		const Semiring &ring = Semiring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Semiring >::value,
		void >::type * const = nullptr
	) {
		// note: dispatches to the above dot-variant, which will handle the HyperDAG
		// generation.
		return dot< descr >(
			z, x, y,
			ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
			phase
		);
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename T, typename U, typename Coords
	>
	RC zip(
		Vector< std::pair< T, U >, hyperdags, Coords > &z,
		const Vector< T, hyperdags, Coords > &x,
		const Vector< U, hyperdags, Coords > &y,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< T >::value &&
			!grb::is_object< U >::value,
		void >::type * const = nullptr
	) {
		const RC ret = zip< descr >(
			internal::getVector(z),
			internal::getVector(x), internal::getVector(y),
			phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::ZIP,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename T, typename U, typename Coords
	>
	RC unzip(
		Vector< T, hyperdags, Coords > &x,
		Vector< U, hyperdags, Coords > &y,
		const Vector< std::pair< T, U >, hyperdags, Coords > &in,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< T >::value &&
			!grb::is_object< U >::value,
		void >::type * const = nullptr
	) {
		const RC ret = unzip< descr >(
			internal::getVector(x), internal::getVector(y), internal::getVector(in),
			phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(in) ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(in) )
		};
		std::array< uintptr_t, 2 > destinations{
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) )
		};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::UNZIP_VECTOR_VECTOR_VECTOR,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< InputType1, hyperdags, Coords > &x,
		const Vector< InputType2, hyperdags, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< OP >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseApply< descr >(
			internal::getVector(z),
			internal::getVector(x), internal::getVector(y),
			op, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::E_WISE_APPLY_VECTOR_VECTOR_VECTOR_OP,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename InputType, typename IOType, typename Coords
	>
	RC foldr(
		const Vector< InputType, hyperdags, Coords > &x,
		IOType &beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< InputType >::value &&
			!grb::is_object< IOType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		const RC ret = foldr< descr >( internal::getVector(x), beta, monoid, phase );
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 1 > sourcesP{ &beta };
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getVector(x) ) };
		std::array< uintptr_t, 0 > destinations{};
		// NOTE scalar output is ignored
		//std::array< const void *, 1 > destinationsP{ &beta };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDR_VECTOR_SCALAR_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename InputType, typename MaskType, typename IOType, typename Coords
	>
	RC foldr(
		const Vector< InputType, hyperdags, Coords > &x,
		const Vector< MaskType, hyperdags, Coords > &m,
		IOType &beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< InputType >::value &&
			!grb::is_object< IOType >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return foldr< descr >( x, beta, monoid, phase );
		}
		const RC ret = foldr< descr >(
			internal::getVector(x), internal::getVector(m),
			beta, monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 1 > sourcesP{ &beta };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(m) )
		};
		std::array< uintptr_t, 0 > destinations{};
		// NOTE scalar output is ignored
		// std::array< const void *, 1 > destinationsP{ &beta };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDR_VECTOR_MASK_SCALAR_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename IOType, typename InputType, typename Coords
	>
	RC foldr(
		const InputType &alpha,
		Vector< IOType, hyperdags, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< InputType >::value &&
			!grb::is_object< IOType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		const RC ret = foldr< descr >( alpha, internal::getVector(y), monoid, phase );
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(y) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		std::array< const void *, 1 > sourcesP{ &alpha };
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getVector(y) ) };
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(y) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDR_APLHA_VECTOR_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		 class OP, typename IOType, typename InputType, typename Coords
	>
	RC foldr(
		const InputType &alpha,
		Vector< IOType, hyperdags, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< InputType >::value &&
			!grb::is_object< IOType >::value &&
			grb::is_operator< OP >::value,
		void >::type * const = nullptr
	) {
		const RC ret = foldr< descr >( alpha, internal::getVector(y), op, phase );
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(y) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		std::array< const void *, 1 > sourcesP{ &alpha };
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getVector(y) ) };
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(y) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDR_APLHA_VECTOR_OPERATOR,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class OP,
		typename IOType, typename InputType, typename Coords
	>
	RC foldr(
		const Vector< InputType, hyperdags, Coords > &x,
		Vector< IOType, hyperdags, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_operator< OP >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< IOType >::value,
		void >::type * = nullptr
	) {
		const RC ret = foldr< descr >(
			internal::getVector(x),
			internal::getVector(y),
			op, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(y) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDR_VECTOR_VECTOR_OPERATOR,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		 Descriptor descr = descriptors::no_operation, class OP,
		 typename IOType, typename MaskType, typename InputType, typename Coords
	>
	RC foldr(
		const Vector< InputType, hyperdags, Coords > &x,
		const Vector< MaskType, hyperdags, Coords > &m,
		Vector< IOType, hyperdags, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_operator< OP >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< IOType >::value,
		void >::type * = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return foldr< descr >( x, y, op, phase );
		}
		const RC ret = foldr< descr >(
			internal::getVector(x),
			internal::getVector(m),
			internal::getVector(y),
			op, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(m) ),
			getID( internal::getVector(y) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(y) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDR_VECTOR_VECTOR_VECTOR_OPERATOR,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid, typename IOType, typename InputType, typename Coords
	>
	RC foldr(
		const Vector< InputType, hyperdags, Coords > &x,
		Vector< IOType, hyperdags, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< Monoid >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< IOType >::value,
		void >::type * = nullptr
	) {
		const RC ret = foldr< descr >(
			internal::getVector(x), internal::getVector(y),
			monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(y) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDR_VECTOR_VECTOR_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename IOType, typename MaskType, typename InputType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, hyperdags, Coords > &x,
		const Vector< MaskType, hyperdags, Coords > &m,
		Vector< IOType, hyperdags, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< Monoid >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< IOType >::value,
		void >::type * = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return foldr< descr >( x, y, monoid, phase );
		}
		const RC ret = foldr< descr >(
			internal::getVector(x), internal::getVector(m),
			internal::getVector(y), monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(m) ),
			getID( internal::getVector(y) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(y) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDR_VECTOR_VECTOR_VECTOR_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename InputType, typename IOType, typename Coords
	>
	RC foldl(
		IOType &x,
		const Vector< InputType, hyperdags, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
		const RC ret = foldl< descr >(
			x, internal::getVector(y), monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(y) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&x
		);
		std::array< const void *, 1 > sourcesP{ &x };
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getVector(y) ) };
		std::array< uintptr_t, 0 > destinations{};
		// NOTE scalar outputs are ignored
		//std::array< const void *, 1 > destinationsP{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDL_SCALAR_VECTOR_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename InputType, typename IOType, typename MaskType,
		typename Coords
	>
	RC foldl(
		IOType &x,
		const Vector< InputType, hyperdags, Coords > &y,
		const Vector< MaskType, hyperdags, Coords > &mask,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(mask) ) == 0 ) {
			return foldl< descr >( x, y, monoid, phase );
		}
		const RC ret = foldl< descr >(
			x, internal::getVector(y), internal::getVector(mask),
			monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&x
		);
		std::array< const void *, 1 > sourcesP{ &x };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(y) ),
			getID( internal::getVector(mask) )
		};
		std::array< uintptr_t, 0 > destinations{};
		// NOTE scalar outputs are ignored
		// std::array< const void * const, 1 > destinationsP{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDL_SCALAR_VECTOR_MASK_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Op, typename IOType, typename InputType, typename Coords
	>
	RC foldl(
		Vector< IOType, hyperdags, Coords > &x,
		const InputType beta,
		const Op &op = Op(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_operator< Op >::value,
		void >::type * = nullptr
	) {
		const RC ret = foldl< descr >( internal::getVector(x), beta, op, phase );
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 1 > sourcesP{ &beta };
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getVector(x) ) };
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDL_VECTOR_BETA_OP,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Op,
		typename IOType, typename MaskType, typename InputType, typename Coords
	>
	RC foldl(
		Vector< IOType, hyperdags, Coords > &x,
		const Vector< MaskType, hyperdags, Coords > &m,
		const InputType beta,
		const Op &op = Op(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_operator< Op >::value,
		void >::type * = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return foldl< descr >( x, beta, op, phase );
		}
		const RC ret = foldl< descr >(
			internal::getVector(x), internal::getVector(m),
			beta, op, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 1 > sourcesP{ &beta };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(m) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDL_VECTOR_VECTOR_BETA_OP,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename IOType, typename InputType, typename Coords
	>
	RC foldl(
		Vector< IOType, hyperdags, Coords > &x,
		const InputType beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * = nullptr
	) {
		const RC ret = foldl< descr >( internal::getVector(x), beta, monoid, phase );
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 1 > sourcesP{ &beta };
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getVector(x) ) };
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDL_VECTOR_BETA_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		 Descriptor descr = descriptors::no_operation, class Monoid,
		 typename IOType, typename MaskType, typename InputType,
		 typename Coords
	>
	RC foldl(
		Vector< IOType, hyperdags, Coords > &x,
		const Vector< MaskType, hyperdags, Coords > &m,
		const InputType &beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return foldl< descr >( x, beta, monoid, phase );
		}
		const RC ret = foldl< descr >(
			internal::getVector(x), internal::getVector(m),
			beta, monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 1 > sourcesP{ &beta };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(m) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDL_VECTOR_VECTOR_BETA_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template <
		Descriptor descr = descriptors::no_operation,
		class Monoid, typename IOType, typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, hyperdags, Coords > &x,
		const Vector< InputType, hyperdags, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< Monoid >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value,
		void >::type * = nullptr
	) {
		const RC ret = foldl< descr >(
			internal::getVector(x), internal::getVector(y),
			monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(y) ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDL_VECTOR_VECTOR_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template <
		Descriptor descr = descriptors::no_operation, class OP,
		typename IOType, typename MaskType, typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, hyperdags, Coords > &x,
		const Vector< MaskType, hyperdags, Coords > &m,
		const Vector< InputType, hyperdags, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_operator< OP >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType >::value, void
		>::type * = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return foldl< descr >( x, y, op, phase );
		}
		const RC ret = foldl< descr >(
			internal::getVector(x), internal::getVector(m),
			internal::getVector(y), op, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(m) ),
			getID( internal::getVector(y) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDL_VECTOR_VECTOR_VECTOR_OP,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename IOType, typename MaskType, typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, hyperdags, Coords > &x,
		const Vector< MaskType, hyperdags, Coords > &m,
		const Vector< InputType, hyperdags, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< Monoid >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType >::value,
		void >::type * = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return foldl< descr >( x, y, monoid, phase );
		}
		const RC ret = foldl< descr >(
			internal::getVector(x),internal::getVector(m),
			internal::getVector(y), monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(m) ),
			getID( internal::getVector(y) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDL_VECTOR_VECTOR_VECTOR_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP, typename IOType, typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, hyperdags, Coords > &x,
		const Vector< InputType, hyperdags, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_operator< OP >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value,
		void >::type * = nullptr
	) {
		const RC ret = foldl< descr >(
			internal::getVector(x), internal::getVector(y),
			op, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDL_VECTOR_VECTOR_OP,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename Func,
		typename DataType,
		typename Coords
	>
	RC eWiseLambda(
		const Func f,
		const Vector< DataType, hyperdags, Coords > &x
	) {
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getVector(x) ) };
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISELAMBDA,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return eWiseLambda< descr >( f, internal::getVector(x) );
	}

	namespace internal {

		/** \internal This is the end recursion */
		template<
			Descriptor descr,
			typename Func, typename DataType,
			typename Coords
		>
		RC hyperdag_ewisevector(
			const Func f,
			const Vector< DataType, grb::hyperdags, Coords > &x,
			std::vector< uintptr_t > &sources,
			std::vector< uintptr_t > &destinations
		) {
			const RC ret = grb::eWiseLambda< descr >( f, internal::getVector(x) );
			if( ret != grb::SUCCESS ) { return ret; }
			if( size( internal::getVector(x) ) == 0 ) { return ret; }
			std::array< const void *, 0 > sourcesP{};
			sources.push_back( getID( internal::getVector(x) ) );
			internal::hyperdags::generator.addOperation(
				internal::hyperdags::EWISELAMBDA_FUNC_VECTOR,
				sourcesP.cbegin(), sourcesP.cend(),
				sources.cbegin(), sources.cend(),
				destinations.cbegin(), destinations.cend()
			);
			return ret;
		}

		/** \internal This is the base recursion */
		template<
			Descriptor descr = descriptors::no_operation,
			typename Func, typename DataType1, typename DataType2,
			typename Coords, typename... Args
		>
		RC hyperdag_ewisevector(
			const Func f,
			const Vector< DataType1, grb::hyperdags, Coords > &x,
			std::vector< uintptr_t > &sources,
			std::vector< uintptr_t > &destinations,
			const Vector< DataType2, grb::hyperdags, Coords > &y,
			Args... args
		) {
			sources.push_back( getID( internal::getVector(y) ) );
			destinations.push_back( getID( internal::getVector(y) ) );
			return hyperdag_ewisevector< descr >(
				f, x, sources, destinations, args...
			);
		}

	} // end namespace grb::internal

	template<
		Descriptor descr = descriptors::no_operation,
		typename Func,
		typename DataType1, typename DataType2, typename Coords,
		typename... Args
	>
	RC eWiseLambda(
		const Func f,
		const Vector< DataType1, hyperdags, Coords > &x,
		const Vector< DataType2, hyperdags, Coords > &y,
		Args const &... args
	) {
		std::vector< uintptr_t > sources, destinations;
		return internal::hyperdag_ewisevector< descr >(
			f, x, sources, destinations, y, args...
		);
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< OP >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseApply< descr >(
			internal::getVector(z), alpha, beta,
			op, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(z) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 2 > sourcesP{ &alpha, &beta };
		std::array< uintptr_t, 1 > sourcesC{
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_ALPHA_BETA_OP,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseApply< descr >(
			internal::getVector(z), alpha, beta,
			monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 2 > sourcesP{ &alpha, &beta };
		std::array< uintptr_t, 1 > sourcesC{
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_ALPHA_BETA_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &mask,
		const InputType1 alpha,
		const InputType2 beta,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< OP >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(mask) ) == 0 ) {
			return eWiseApply< descr >( z, alpha, beta, op, phase );
		}
		const RC ret = eWiseApply< descr >(
			internal::getVector(z), internal::getVector(mask),
			alpha, beta,
			op, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(z) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 2 > sourcesP{ &alpha, &beta };
		std::array< uintptr_t, 1 > sourcesC{
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{
			getID( internal::getVector(z) )
		};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_MASK_ALPHA_BETA_OP,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &mask,
		const InputType1 alpha,
		const InputType2 beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(mask) ) == 0 ) {
			return eWiseApply< descr >( z, alpha, beta, monoid, phase );
		}
		const RC ret = eWiseApply< descr >(
			internal::getVector(z), internal::getVector(mask),
			alpha, beta,
			monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 2 > sourcesP{ &alpha, &beta };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(mask) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_MASK_ALPHA_BETA_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< InputType1, hyperdags, Coords > &x,
		const InputType2 beta,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value
			&& !grb::is_object< InputType1 >::value
			&& !grb::is_object< InputType2 >::value
			&& grb::is_operator< OP >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseApply< descr >(
			internal::getVector(z), internal::getVector(x), beta,
			op, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 1 > sourcesP{ &beta };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_VECTOR_BETA_OP,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType,
		typename InputType1, typename InputType2, typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value
			&& grb::is_operator< OP >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseApply< descr >(
			internal::getVector(z), alpha, internal::getVector(y),
			op, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(z) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		std::array< const void *, 1 > sourcesP{ &alpha };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_ALPHA_VECTOR_OP,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &mask,
		const Vector< InputType1, hyperdags, Coords > &x,
		const InputType2 beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(mask) ) == 0 ) {
			return eWiseApply< descr >( z, x, beta, monoid, phase );
		}
		const RC ret = eWiseApply< descr >(
			internal::getVector(z), internal::getVector(mask),
			internal::getVector(x), beta,
			monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 1 > sourcesP{ &beta };
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(mask) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_MASK_VECTOR_BETA_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType, typename MaskType, typename InputType1,
		typename InputType2, typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &mask,
		const Vector< InputType1, hyperdags, Coords > &x,
		const InputType2 beta,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< OP >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(mask) ) == 0 ) {
			return eWiseApply< descr >( z, x, beta, op, phase );
		}
		const RC ret = eWiseApply< descr >(
			internal::getVector(z), internal::getVector(mask),
			internal::getVector(x), beta,
			op, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 1 > sourcesP{ &beta };
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(mask) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_MASK_VECTOR_BETA_OP,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &mask,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(mask) ) == 0 ) {
			return eWiseApply< descr >( z, alpha, y, monoid, phase );
		}
		const RC ret = eWiseApply< descr >(
			internal::getVector(z), internal::getVector(mask),
			alpha, internal::getVector(y),
			monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		std::array< const void *, 1 > sourcesP{ &alpha };
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(mask) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_MASK_ALPHA_VECTOR_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType, typename MaskType, typename InputType1,
		typename InputType2, typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &mask,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< OP >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(mask) ) == 0 ) {
			return eWiseApply< descr >( z, alpha, y, op, phase );
		}
		const RC ret = eWiseApply< descr >(
			internal::getVector(z), internal::getVector(mask),
			alpha, internal::getVector(y),
			op, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		std::array< const void *, 1 > sourcesP{ &alpha };
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(mask) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_MASK_ALPHA_VECTOR_OP,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &mask,
		const Vector< InputType1, hyperdags, Coords > &x,
		const Vector< InputType2, hyperdags, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< OP >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(mask) ) == 0 ) {
			return eWiseApply< descr >( z, x, y, op, phase );
		}
		const RC ret = eWiseApply< descr >(
			internal::getVector(z), internal::getVector(mask),
			internal::getVector(x), internal::getVector(y),
			op, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 4 > sourcesC{
			getID( internal::getVector(mask) ),
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_MASK_VECTOR_VECTOR_OP,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< InputType1, hyperdags, Coords > &x,
		const InputType2 beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseApply< descr >(
			internal::getVector(z),
			internal::getVector(x), beta,
			monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 1 > sourcesP{ &beta };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_VECTOR_BETA_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseApply< descr >(
			internal::getVector(z),
			alpha, internal::getVector(y),
			monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(y) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		std::array< const void *, 1 > sourcesP{ &alpha };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_ALPHA_VECTOR_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2, typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &mask,
		const Vector< InputType1, hyperdags, Coords > &x,
		const Vector< InputType2, hyperdags, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(mask) ) == 0 ) {
			return eWiseApply< descr >( z, x, y, monoid, phase );
		}
		const RC ret = eWiseApply< descr >(
			internal::getVector(z), internal::getVector(mask),
			internal::getVector(x), internal::getVector(y),
			monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 4 > sourcesC{
			getID( internal::getVector(mask) ),
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_MASK_VECTOR_VECTOR_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< InputType1, hyperdags, Coords > &x,
		const Vector< InputType2, hyperdags, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseApply< descr >(
			internal::getVector(z),
			internal::getVector(x), internal::getVector(y),
			monoid, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_VECTOR_VECTOR_VECTOR_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &m,
		const Vector< InputType1, hyperdags, Coords > &a,
		const Vector< InputType2, hyperdags, Coords > &x,
		const Vector< InputType3, hyperdags, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value &&
			!grb::is_object< MaskType >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return eWiseMulAdd< descr >( z, a, x, y, ring, phase );
		}
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z), internal::getVector(m),
			internal::getVector(a), internal::getVector(x), internal::getVector(y),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 5 > sourcesC{
			getID( internal::getVector(m) ),
			getID( internal::getVector(a) ),
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISE_MUL_ADD,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &m,
		const Vector< InputType1, hyperdags, Coords > &a,
		const Vector< InputType2, hyperdags, Coords > &x,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value &&
			!grb::is_object< MaskType >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return eWiseMulAdd< descr >( z, a, x, gamma, ring, phase );
		}
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z), internal::getVector(m),
			internal::getVector(a), internal::getVector(x), gamma,
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&gamma
		);
		std::array< const void *, 1 > sourcesP{ &gamma };
		std::array< uintptr_t, 4 > sourcesC{
			getID( internal::getVector(m) ),
			getID( internal::getVector(a) ),
			getID( internal::getVector(x) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISE_MUL_ADD_FOUR_VECTOR,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > &x,
		const Vector< InputType3, hyperdags, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z), alpha,
			internal::getVector(x), internal::getVector(y),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		std::array< const void *, 1 > sourcesP{ &alpha };
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISE_MUL_ADD_THREE_VECTOR_ALPHA,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring, typename InputType1,
		typename InputType2, typename InputType3, typename OutputType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< InputType1, hyperdags, Coords > &a,
		const InputType2 chi,
		const Vector< InputType3, hyperdags, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z),
			internal::getVector(a), chi, internal::getVector(y),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(y) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&chi
		);
		std::array< const void *, 1 > sourcesP{ &chi };
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(a) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISE_MUL_ADD_THREE_VECTOR_CHI,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &m,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > &x,
		const Vector< InputType3, hyperdags, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value &&
			!grb::is_object< MaskType >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return eWiseMulAdd< descr >( z, alpha, x, y, ring, phase );
		}
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z), internal::getVector(m),
			alpha, internal::getVector(x), internal::getVector(y),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		std::array< const void *, 1 > sourcesP{ &alpha };
		std::array< uintptr_t, 4 > sourcesC{
			getID( internal::getVector(m) ),
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISE_MUL_ADD_FOUR_VECTOR_CHI,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &m,
		const Vector< InputType1, hyperdags, Coords > &a,
		const InputType2 chi,
		const Vector< InputType3, hyperdags, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value &&
			!grb::is_object< MaskType >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return eWiseMulAdd< descr >( z, a, chi, y, ring, phase );
		}
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z), internal::getVector(m),
			internal::getVector(a), chi, internal::getVector(y),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&chi
		);
		std::array< const void *, 1 > sourcesP{ &chi };
		std::array< uintptr_t, 4 > sourcesC{
			getID( internal::getVector(m) ),
			getID( internal::getVector(a) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISE_MUL_ADD_FOUR_VECTOR_CHI_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &m,
		const Vector< InputType1, hyperdags, Coords > &a,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value &&
			!grb::is_object< MaskType >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return eWiseMulAdd< descr >( z, a, beta, gamma, ring, phase );
		}
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z), internal::getVector(m),
			internal::getVector(a), beta,  gamma,
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&gamma
		);
		std::array< const void *, 2 > sourcesP{ &beta, &gamma };
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(m) ),
			getID( internal::getVector(a) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISE_MUL_ADD_THREE_VECTOR_BETA,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &m,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > &x,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value &&
			!grb::is_object< MaskType >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return eWiseMulAdd< descr >( z, alpha, x, gamma, ring, phase );
		}
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z), internal::getVector(m),
			alpha, internal::getVector(x), gamma,
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&gamma
		);
		std::array< const void *, 2 > sourcesP{ &alpha, &gamma };
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(m) ),
			getID( internal::getVector(x) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISE_MUL_ADD_THREE_VECTOR_ALPHA_GAMMA,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2, typename InputType3,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &m,
		const InputType1 alpha,
		const InputType2 beta,
		const Vector< InputType3, hyperdags, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value &&
			!grb::is_object< MaskType >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return eWiseMulAdd< descr >( z, alpha, beta, y, ring, phase );
		}
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z), internal::getVector(m),
			alpha, beta, internal::getVector(y),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 2 > sourcesP{ &alpha, &beta };
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(m) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISE_MUL_ADD_TWO_VECTOR_ALPHA_BETA,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename MaskType, typename InputType1,
		typename InputType2, typename InputType3, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &m,
		const InputType1 alpha,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return eWiseMulAdd< descr >( z, alpha, beta, gamma, ring, phase );
		}
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z), internal::getVector(m),
			alpha, beta, gamma,
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&gamma
		);
		std::array< const void *, 3 > sourcesP{ &alpha, &beta, &gamma };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(m) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISE_MUL_ADD_TWO_VECTOR_ALPHA_BETA_GAMMA,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< InputType1, hyperdags, Coords > &a,
		const Vector< InputType2, hyperdags, Coords > &x,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z),
			internal::getVector(a), internal::getVector(x), gamma,
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&gamma
		);
		std::array< const void *, 1 > sourcesP{ &gamma };
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(a) ),
			getID( internal::getVector(x) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMULADD_VECTOR_VECTOR_VECTOR_GAMMA_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< InputType1, hyperdags, Coords > &a,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z),
			internal::getVector(a), beta, gamma,
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(z) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&gamma
		);
		std::array< const void *, 2 > sourcesP{ &beta, &gamma };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(a) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMULADD_VECTOR_VECTOR_BETA_GAMMA_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > &x,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		 void >::type * const = nullptr
	) {
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z),
			alpha, internal::getVector(x), gamma,
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&gamma
		);
		std::array< const void *, 2 > sourcesP{ &alpha, &gamma };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMULADD_VECTOR_ALPHA_VECTOR_GAMMA_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename InputType1, typename InputType2,
		typename InputType3, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const Vector< InputType3, hyperdags, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z),
			alpha, beta, internal::getVector(y),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(y) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 2 > sourcesP{ &alpha, &beta };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMULADD_VECTOR_ALPHA_BETA_VECTOR_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename InputType1, typename InputType2,
		typename InputType3, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z),
			alpha, beta, gamma,
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(z) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&gamma
		);
		std::array< const void *, 3 > sourcesP{ &alpha, &beta, &gamma };
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getVector(z) ) };
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMULADD_VECTOR_ALPHA_BETA_GAMMA_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \warning This function is deprecated */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< InputType1, hyperdags, Coords > &a,
		const Vector< InputType2, hyperdags, Coords > &x,
		const Vector< InputType3, hyperdags, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseMulAdd< descr >(
			internal::getVector(z),
			internal::getVector(a), internal::getVector(x), internal::getVector(y),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 4 > sourcesC{
			getID( internal::getVector(a) ),
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMULADD_VECTOR_VECTOR_VECTOR_VECTOR_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< InputType1, hyperdags, Coords > &x,
		const Vector< InputType2, hyperdags, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseMul< descr >(
			internal::getVector(z), internal::getVector(x), internal::getVector(y),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMUL_VECTOR_VECTOR_VECTOR_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}


	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, hyperdags, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseMul< descr >(
			internal::getVector(z),
			alpha, internal::getVector(y),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(y) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		std::array< const void *, 1 > sourcesP{ &alpha };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMUL_VECTOR_ALPHA_VECTOR_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< InputType1, hyperdags, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseMul< descr >(
			internal::getVector(z),
			internal::getVector(x), beta,
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( internal::getVector(x) ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 1 > sourcesP{ &beta };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMUL_VECTOR_VECTOR_BETA_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, hyperdags, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseMul< descr >(
			internal::getVector(z),
			alpha, beta,
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 2 > sourcesP{ &alpha, &beta };
		std::array< uintptr_t, 1 > sourcesC{
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMUL_VECTOR_ALPHA_BETA_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename MaskType, typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &m,
		const Vector< InputType1, hyperdags, Coords > &x,
		const Vector< InputType2, hyperdags, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return eWiseMul< descr >( z, x, y, ring, phase );
		}
		const RC ret = eWiseMul< descr >(
			internal::getVector(z),
			internal::getVector(m), internal::getVector(x), internal::getVector(y),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 4 > sourcesC{
			getID( internal::getVector(m) ),
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
			internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMUL_VECTOR_VECTOR_VECTOR_VECTOR_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename MaskType, typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &m,
		const InputType1 alpha,
		const Vector< InputType2, hyperdags, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return eWiseMul< descr >( z, alpha, y, ring, phase );
		}
		const RC ret = eWiseMul< descr >(
			internal::getVector(z), internal::getVector(m),
			alpha, internal::getVector(y),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		std::array< const void *, 1 > sourcesP{ &alpha };
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(m) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
			internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMUL_VECTOR_VECTOR_ALPHA_VECTOR_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename MaskType, typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &m,
		const Vector< InputType1, hyperdags, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return eWiseMul< descr >( z, x, beta, ring, phase );
		}
		const RC ret = eWiseMul< descr >(
			internal::getVector(z), internal::getVector(m),
			internal::getVector(x), beta,
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 1 > sourcesP{ &beta };
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(m) ),
			getID( internal::getVector(x) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
			internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMUL_VECTOR_VECTOR_VECTOR_BETA_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename MaskType, typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, hyperdags, Coords > &z,
		const Vector< MaskType, hyperdags, Coords > &m,
		const InputType1 alpha,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(m) ) == 0 ) {
			return eWiseMul< descr >( z, alpha, beta, ring, phase );
		}
		const RC ret = eWiseMul< descr >(
			internal::getVector(z), internal::getVector(m),
			alpha, beta,
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&alpha
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&beta
		);
		std::array< const void *, 2 > sourcesP{ &alpha, &beta };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(m) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(z) ) };
			internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEMUL_VECTOR_VECTOR_ALPHA_BETA_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

} // end namespace grb

#endif

