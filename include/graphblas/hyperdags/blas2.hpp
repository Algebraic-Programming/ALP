
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

	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename IOType, typename InputType1, typename InputType2,
		typename InputType3, typename Coords
	>
	RC vxm(
		Vector< IOType, hyperdags, Coords > &u,
		const Vector< InputType3, hyperdags, Coords > &mask,
		const Vector< InputType1, hyperdags, Coords > &v,
		const Matrix< InputType2, hyperdags > &A,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(mask) ) == 0 ) {
			return vxm< descr >( u, v, A, ring, phase );
		}
		const RC ret = vxm< descr >(
			internal::getVector(u), internal::getVector(mask),
			internal::getVector(v), internal::getMatrix(A),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 4 > sourcesC{
			getID( internal::getVector(mask) ),
			getID( internal::getVector(v) ),
			getID( internal::getMatrix(A) ),
			getID( internal::getVector(u) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(u) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::VXM_VECTOR_VECTOR_VECTOR_MATRIX,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType, typename InputType1, typename InputType2,
		typename InputType3, typename Coords
	>
	RC vxm(
		Vector< IOType, hyperdags, Coords > &u,
		const Vector< InputType3, hyperdags, Coords > &mask,
		const Vector< InputType1, hyperdags, Coords > &v,
		const Matrix< InputType2, hyperdags > &A,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			!std::is_same< InputType2, void >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(mask) ) == 0 ) {
			return vxm< descr >( u, v, A, add, mul, phase );
		}
		const RC ret = vxm< descr >(
			internal::getVector(u), internal::getVector(mask),
			internal::getVector(v), internal::getMatrix(A),
			add, mul, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 4 > sourcesC{
			getID( internal::getVector(mask) ),
			getID( internal::getVector(v) ),
			getID( internal::getMatrix(A) ),
			getID( internal::getVector(u) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(u) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::VXM_VECTOR_VECTOR_VECTOR_MATRIX_ADD_MUL,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2,
		typename Coords
	>
	RC vxm(
		Vector< IOType, hyperdags, Coords > &u,
		const Vector< InputType1, hyperdags, Coords > &v,
		const Matrix< InputType2, hyperdags > &A,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const RC ret = vxm< descr >(
			internal::getVector(u),
			internal::getVector(v), internal::getMatrix(A),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(v) ),
			getID( internal::getMatrix(A) ),
			getID( internal::getVector(u) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(u) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::VXM_VECTOR_VECTOR_MATRIX_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2,
		typename InputType3 = bool,
		typename Coords
	>
	RC mxv(
		Vector< IOType, hyperdags, Coords > &u,
		const Vector< InputType3, hyperdags, Coords > &mask,
		const Matrix< InputType2, hyperdags > &A,
		const Vector< InputType1, hyperdags, Coords > &v,
		const Ring &ring,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(mask) ) == 0 ) {
			return mxv< descr >( u, A, v, ring, phase );
		}
		const RC ret = mxv< descr >(
			internal::getVector(u), internal::getVector(mask),
			internal::getMatrix(A), internal::getVector(v),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 4 > sourcesC{
			getID( internal::getVector(mask) ),
			getID( internal::getMatrix(A) ),
			getID( internal::getVector(v) ),
			getID( internal::getVector(u) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(u) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::MXV_VECTOR_VECTOR_MATRIX_VECTOR_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class Ring,
		typename IOType, typename InputType1, typename InputType2,
		typename InputType3, typename InputType4, typename Coords
	>
	RC mxv(
		Vector< IOType, hyperdags, Coords > &u,
		const Vector< InputType3, hyperdags, Coords > &mask,
		const Matrix< InputType2, hyperdags > &A,
		const Vector< InputType1, hyperdags, Coords > &v,
		const Vector< InputType4, hyperdags, Coords > &v_mask,
		const Ring &ring,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			!grb::is_object< InputType4 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(v_mask) ) == 0 ) {
			return mxv< descr >( u, mask, A, v, ring, phase );
		}
		const RC ret = mxv< descr >(
			internal::getVector(u), internal::getVector(mask),
			internal::getMatrix(A), internal::getVector(v), internal::getVector(v_mask),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		std::vector< uintptr_t > sourcesC{
			getID( internal::getVector(v_mask) ),
			getID( internal::getMatrix(A) ),
			getID( internal::getVector(v) ),
			getID( internal::getVector(u) )
		};
		if( size( internal::getVector(mask) ) > 0 ) {
			sourcesC.push_back( getID( internal::getVector(mask) ) );
		}
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(u) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::MXV_VECTOR_VECTOR_MATRIX_VECTOR_VECTOR_R,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType, typename InputType1, typename InputType2,
		typename InputType3, typename InputType4, typename Coords
	>
	RC mxv(
		Vector< IOType, hyperdags, Coords > &u,
		const Vector< InputType3, hyperdags, Coords > &mask,
		const Matrix< InputType2, hyperdags > &A,
		const Vector< InputType1, hyperdags, Coords > &v,
		const Vector< InputType4, hyperdags, Coords > &v_mask,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			!grb::is_object< InputType4 >::value &&
			!std::is_same< InputType2, void >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(v_mask) ) == 0 ) {
			return mxv< descr >( u, mask, A, v, add, mul, phase );
		}
		const RC ret = mxv< descr >(
			internal::getVector(u), internal::getVector(mask),
			internal::getMatrix(A), internal::getVector(v), internal::getVector(v_mask),
			add, mul, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::vector< uintptr_t > sourcesC{
			getID( internal::getMatrix(A) ),
			getID( internal::getVector(v) ),
			getID( internal::getVector(v_mask) ),
			getID( internal::getVector(u) )
		};
		if( size( internal::getVector(mask) ) > 0 ) {
			sourcesC.push_back( getID( internal::getVector(mask) ) );
		}
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(u) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::MXV_VECTOR_VECTOR_MATRIX_VECTOR_VECTOR_A,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2,
		typename Coords
	>
	RC mxv(
		Vector< IOType, hyperdags, Coords > &u,
		const Matrix< InputType2, hyperdags > &A,
		const Vector< InputType1, hyperdags, Coords > &v,
		const Ring &ring,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const RC ret = mxv< descr >(
			internal::getVector(u),
			internal::getMatrix(A), internal::getVector(v),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getMatrix(A) ),
			getID( internal::getVector(v) ),
			getID( internal::getVector(u) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(u) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::MXV_VECTOR_MATRIX_VECTOR_RING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType, typename InputType1, typename InputType2, typename Coords
	>
	RC mxv(
		Vector< IOType, hyperdags, Coords > &u,
		const Matrix< InputType2, hyperdags > &A,
		const Vector< InputType1, hyperdags, Coords > &v,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!std::is_same< InputType2, void >::value,
		void >::type * const = nullptr
	) {
		const RC ret = mxv< descr >(
			internal::getVector(u),
			internal::getMatrix(A), internal::getVector(v),
			add, mul, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getMatrix(A) ),
			getID( internal::getVector(v) ),
			getID( internal::getVector(u) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(u) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::MXV_VECTOR_MATRIX_VECTOR_ADD_MUL,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	/** \internal Uses a direct implementation. */
	template<
		typename Func, typename DataType
	>
	RC eWiseLambda(
		const Func f,
		const Matrix< DataType, hyperdags > &A
	) {
		const RC ret = eWiseLambda( f, internal::getMatrix(A) );
		if( ret != SUCCESS ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getMatrix(A) ) };
		std::array< uintptr_t, 0 > destinations{};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISELAMBDA_FUNC_MATRIX,
			sourcesP.cbegin(), sourcesP.cend(),
			sourcesC.cbegin(), sourcesC.cend(),
			destinations.cbegin(), destinations.cend()
		);
		return ret;
	}

	namespace internal {

		/** \internal This is the end recursion */
		template<
			typename Func, typename DataType
		>
		RC hyperdag_ewisematrix(
			const Func f,
			const Matrix< DataType, grb::hyperdags > &A,
			std::vector< uintptr_t > &sources,
			std::vector< uintptr_t > &destinations
		) {
			const RC ret = grb::eWiseLambda( f, internal::getMatrix(A) );
			if( ret != SUCCESS ) { return ret; }
			if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
			std::array< const void *, 0 > sourcesP{};
			sources.push_back( getID( internal::getMatrix(A) ) );
			internal::hyperdags::generator.addOperation(
				internal::hyperdags::EWISELAMBDA_FUNC_MATRIX,
				sourcesP.cbegin(), sourcesP.cend(),
				sources.cbegin(), sources.cend(),
				destinations.cbegin(), destinations.cend()
			);
			return ret;
		}

		/** \internal This is the base recursion */
		template<
			typename Func, typename DataType1, typename DataType2,
			typename Coords, typename... Args
		>
		RC hyperdag_ewisematrix(
			const Func f,
			const Matrix< DataType1, grb::hyperdags > &A,
			std::vector< uintptr_t > &sources,
			std::vector< uintptr_t > &destinations,
			const Vector< DataType2, grb::hyperdags, Coords > &x,
			Args... args
		) {
			sources.push_back( getID( internal::getVector(x) ) );
			destinations.push_back( getID( internal::getVector(x) ) );
			return hyperdag_ewisematrix( f, A, sources, destinations, args... );
		}

	} // end namespace grb::internal

	/** \internal Implements the recursive case */
	template<
		typename Func,
		typename DataType1, typename DataType2,
		typename Coords, typename... Args
	>
	RC eWiseLambda(
		const Func f,
		const Matrix< DataType1, hyperdags > &A,
		const Vector< DataType2, hyperdags, Coords > &x,
		Args... args
	) {
		std::vector< uintptr_t > sources, destinations;
		return internal::hyperdag_ewisematrix(
			f, A, sources, destinations, x, args...
		);
	}

	template<
		Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class Ring,
		typename IOType, typename InputType1, typename InputType2,
		typename InputType3, typename InputType4, typename Coords
	>
	RC vxm(
		Vector< IOType, hyperdags, Coords > &u,
		const Vector< InputType3, hyperdags, Coords > &mask,
		const Vector< InputType1, hyperdags, Coords > &v,
		const Vector< InputType4, hyperdags, Coords > &v_mask,
		const Matrix< InputType2, hyperdags > &A,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			!grb::is_object< InputType4 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(v_mask) ) == 0 ) {
			return vxm< descr >( u, mask, v, A, ring, phase );
		}
		const RC ret = vxm< descr >(
			internal::getVector(u), internal::getVector(mask),
			internal::getVector(v), internal::getVector(v_mask), internal::getMatrix(A),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::vector< uintptr_t > sourcesC{
			getID( internal::getVector(v) ),
			getID( internal::getMatrix(A) ),
			getID( internal::getVector(v_mask) ),
			getID( internal::getVector(u) )
		};
		if( size( internal::getVector(mask) ) > 0 ) {
			sourcesC.push_back( getID( internal::getVector(mask) ) );
		}
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(u) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::VXM_GENERIC_VECTOR_VECTOR_VECTOR_VECTOR_MATRIX_ADD_MUL,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType, typename InputType1, typename InputType2,
		typename InputType3, typename InputType4, typename Coords
	>
	RC vxm(
		Vector< IOType, hyperdags, Coords > &u,
		const Vector< InputType3, hyperdags, Coords > &mask,
		const Vector< InputType1, hyperdags, Coords > &v,
		const Vector< InputType4, hyperdags, Coords > &v_mask,
		const Matrix< InputType2, hyperdags > &A,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			!grb::is_object< InputType4 >::value &&
			!std::is_same< InputType2, void >::value,
		void >::type * const = nullptr
	) {
		if( size( internal::getVector(v_mask) ) == 0 ) {
			return vxm< descr >( u, mask, v, A, add, mul, phase );
		}
		const RC ret = vxm< descr >(
			internal::getVector(u), internal::getVector(mask),
			internal::getVector(v), internal::getVector(v_mask), internal::getMatrix(A),
			add, mul, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::vector< uintptr_t > sourcesC{
			getID( internal::getVector(v) ),
			getID( internal::getMatrix(A) ),
			getID( internal::getVector(v_mask) ),
			getID( internal::getVector(u) )
		};
		if( size( internal::getVector(mask) ) == 0 ) {
			sourcesC.push_back( getID( internal::getVector(mask) ) );
		}
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(u) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::VXM_VECTOR_VECTOR_VECTOR_VECTOR_MATRIX_ADD_MUL,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType, typename InputType1, typename InputType2, typename Coords
	>
	RC vxm(
		Vector< IOType, hyperdags, Coords > &u,
		const Vector< InputType1, hyperdags, Coords > &v,
		const Matrix< InputType2, hyperdags > &A,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!std::is_same< InputType2, void >::value,
		void >::type * const = nullptr
	) {
		const RC ret = vxm< descr >(
			internal::getVector(u),
			internal::getVector(v), internal::getMatrix(A),
			add, mul, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(v) ),
			getID( internal::getMatrix(A) ),
			getID( internal::getVector(u) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(u) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::VXM_VECTOR_VECTOR_MATRIX_ADD_MUL,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

} // end namespace grb

#endif

