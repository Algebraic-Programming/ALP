
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
 * Implements the BLAS-3 API for the hypergraphs backend
 *
 * @author A. Karanasiou
 * @date 3rd of March, 2022
 */

#ifndef _H_GRB_HYPERDAGS_BLAS3
#define _H_GRB_HYPERDAGS_BLAS3

#include <graphblas/phase.hpp>
#include <graphblas/matrix.hpp>

#include <graphblas/hyperdags/init.hpp>

#include <array>


namespace grb {

	template<
		Descriptor descr = descriptors::no_operation,
		class MulMonoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT, typename CIT, typename NIT,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2
	>
	RC eWiseApply(
		Matrix< OutputType, hyperdags, RIT, CIT, NIT > &C,
		const Matrix< InputType1, hyperdags, RIT1, CIT1, NIT1 > &A,
		const Matrix< InputType2, hyperdags, RIT2, CIT2, NIT2 > &B,
		const MulMonoid &mulmono,
		const Phase phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< MulMonoid >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseApply< descr >(
			internal::getMatrix( C ),
			internal::getMatrix( A ), internal::getMatrix( B ),
			mulmono, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getMatrix(A) ),
			getID( internal::getMatrix(B) ),
			getID( internal::getMatrix(C) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getMatrix(C) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_MATRIX_MATRIX_MATRIX_MULMONOID_PHASE,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = grb::descriptors::no_operation,
		class Operator,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT, typename CIT, typename NIT,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2
	>
	RC eWiseApply(
		Matrix< OutputType, hyperdags, RIT, CIT, NIT > &C,
		const Matrix< InputType1, hyperdags, RIT1, CIT1, NIT1 > &A,
		const Matrix< InputType2, hyperdags, RIT2, CIT2, NIT2 > &B,
		const Operator &mulOp,
		const Phase phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value,
		void >::type * const = nullptr
	) {
		const RC ret = eWiseApply< descr >(
			internal::getMatrix( C ),
			internal::getMatrix( A ), internal::getMatrix( B ),
			mulOp, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getMatrix(A) ),
			getID( internal::getMatrix(B) ),
			getID( internal::getMatrix(C) ),
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getMatrix(C) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::EWISEAPPLY_MATRIX_MATRIX_MATRIX_OPERATOR_PHASE,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation, typename OutputType,
		typename InputType1, typename InputType2,
		typename RIT, typename CIT, typename NIT,
		class Semiring
	>
	RC mxm(
		Matrix< OutputType, hyperdags, RIT, CIT, NIT > &C,
		const Matrix< InputType1, hyperdags, RIT, CIT, NIT > &A,
		const Matrix< InputType2, hyperdags, RIT, CIT, NIT > &B,
		const Semiring &ring = Semiring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Semiring >::value, void
		>::type * const = nullptr
	) {
		const RC ret = mxm< descr >( internal::getMatrix( C ),
			internal::getMatrix( A ), internal::getMatrix( B ),
			ring, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getMatrix(A) ),
			getID( internal::getMatrix(B) ),
			getID( internal::getMatrix(C) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getMatrix(C) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::MXM_MATRIX_MATRIX_MATRIX_SEMIRING,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = grb::descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT, typename CIT, typename NIT,
		class Operator, class Monoid
	>
	RC mxm(
		Matrix< OutputType, hyperdags, RIT, CIT, NIT > &C,
		const Matrix< InputType1, hyperdags, RIT, CIT, NIT > &A,
		const Matrix< InputType2, hyperdags, RIT, CIT, NIT > &B,
		const Monoid &addM,
		const Operator &mulOp,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		const RC ret = mxm< descr >(
			internal::getMatrix( C ),
			internal::getMatrix( A ), internal::getMatrix( B ),
			addM, mulOp, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getMatrix(A) ),
			getID( internal::getMatrix(B) ),
			getID( internal::getMatrix(C) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getMatrix(C) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::MXM_MATRIX_MATRIX_MATRIX_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType1, typename InputType2, typename OutputType,
		typename RIT, typename CIT, typename NIT,
		typename Coords, class Operator
	>
	RC outer(
		Matrix< OutputType, hyperdags, RIT, CIT, NIT > &A,
		const Vector< InputType1, hyperdags, Coords > &u,
		const Vector< InputType2, hyperdags, Coords > &v,
		const Operator &mul = Operator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_operator< Operator >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< OutputType >::value,
		void >::type * const = nullptr
	) {
		const RC ret = outer< descr >(
			internal::getMatrix( A ),
			internal::getVector( u ), internal::getVector( v ),
			mul, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(u) ),
			getID( internal::getVector(v) ),
			getID( internal::getMatrix(A) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getMatrix(A) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::OUTER,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		typename InputType3, typename RIT, typename CIT, typename NIT,
		typename Coords
	>
	RC zip(
		Matrix< OutputType, hyperdags, RIT, CIT, NIT > &A,
		const Vector< InputType1, hyperdags, Coords > &x,
		const Vector< InputType2, hyperdags, Coords > &y,
		const Vector< InputType3, hyperdags, Coords > &z,
		const Phase &phase = EXECUTE
	) {
		const RC ret = zip< descr >(
			internal::getMatrix( A ),
			internal::getVector( x ), internal::getVector( y ),
			internal::getVector( z ),
			phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 4 > sourcesC{
			getID( internal::getMatrix(A) ),
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(z) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getMatrix(A) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::ZIP_MATRIX_VECTOR_VECTOR_VECTOR,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType1, typename InputType2,
		typename RIT, typename CIT, typename NIT,
		typename Coords
	>
	RC zip(
		Matrix< void, hyperdags, RIT, CIT, NIT > &A,
		const Vector< InputType1, hyperdags, Coords > &x,
		const Vector< InputType2, hyperdags, Coords > &y,
		const Phase &phase = EXECUTE
	) {
		const RC ret = zip< descr >(
			internal::getMatrix( A ),
			internal::getVector( x ), internal::getVector( y ),
			phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getMatrix(A) ),
			getID( internal::getVector(x) ),
			getID( internal::getVector(y) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getMatrix(A) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::ZIP_MATRIX_VECTOR_VECTOR,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType, typename MaskType,
		typename RIT_A, typename CIT_A, typename NIT_A,
		typename RIT_M, typename CIT_M, typename NIT_M
	>
	RC foldr(
		IOType &x,
		const Matrix< InputType, hyperdags, RIT_A, CIT_A, NIT_A > &A,
		const Matrix< MaskType, hyperdags, RIT_M, CIT_M, NIT_M > &mask,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In grb::foldr (hyperdags, mask, matrix, monoid)\n";
#endif

		const RC ret = foldr< descr, Monoid >(
			x, A, mask, monoid
		);
		if( ret != SUCCESS ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getMatrix(A) ),
			getID( internal::getMatrix(mask) )
		};
		std::array< uintptr_t, 0 > destinations{};
		// NOTE scalar output is ignored
		// std::array< uintptr_t, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDR_SCALAR_MATRIX_MASK_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType,
		typename RIT, typename CIT, typename NIT
	>
	RC foldr(
		IOType &x,
		const Matrix< InputType, hyperdags, RIT, CIT, NIT > &A,
		const Monoid &monoid,
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In grb::foldr (hyperdags, matrix, monoid)\n";
#endif

		const RC ret = foldr< descr, Monoid >(
			x, A, monoid
		);
		if( ret != SUCCESS ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getMatrix(A) ) };
		std::array< uintptr_t, 0 > destinations{};
		// NOTE scalar output is ignored
		// std::array< uintptr_t, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDR_SCALAR_MATRIX_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType, typename MaskType,
		typename RIT_A, typename CIT_A, typename NIT_A,
		typename RIT_M, typename CIT_M, typename NIT_M
	>
	RC foldl(
		IOType &x,
		const Matrix< InputType, hyperdags, RIT_A, CIT_A, NIT_A > &A,
		const Matrix< MaskType, hyperdags, RIT_M, CIT_M, NIT_M > &mask,
		const Monoid &monoid,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		#ifdef _DEBUG
		std::cout << "In grb::foldl (hyperdags, mask, matrix, monoid)\n";
#endif

		const RC ret = foldl< descr, Monoid >(
			x, A, mask, monoid
		);
		if( ret != SUCCESS ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getMatrix(A) ),
			getID( internal::getMatrix(mask) )
		};
		std::array< uintptr_t, 0 > destinations{};
		// NOTE scalar output is ignored
		// std::array< uintptr_t, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDL_SCALAR_MATRIX_MASK_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType,
		typename RIT, typename CIT, typename NIT
	>
	RC foldl(
		IOType &x,
		const Matrix< InputType, hyperdags, RIT, CIT, NIT > &A,
		const Monoid &monoid,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In grb::foldl (hyperdags, matrix, monoid)\n";
#endif

		const RC ret = foldl< descr, Monoid >(
			x, A, monoid
		);
		if( ret != SUCCESS ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getMatrix(A) ) };
		std::array< uintptr_t, 0 > destinations{};
		// NOTE scalar output is ignored
		// std::array< uintptr_t, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::FOLDL_SCALAR_MATRIX_MONOID,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

} // end namespace grb

#endif

