
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
 * Defines the GraphBLAS level 2 API.
 *
 * @author A. N. Yzelman
 * @date 30th of March 2017
 */

#ifndef _H_ALP_BLAS2_BASE
#define _H_ALP_BLAS2_BASE

#include <assert.h>

#include <alp/backends.hpp>
#include <alp/blas1.hpp>
#include <alp/descriptors.hpp>
#include <alp/rc.hpp>
#include <alp/semiring.hpp>

#include "config.hpp"
#include "matrix.hpp"
#include "vector.hpp"

namespace alp {

	/**
	 * \defgroup BLAS2 The Level-2 Basic Linear Algebra Subroutines (BLAS)
	 *
	 * A collection of functions that allow GraphBLAS operators, monoids, and
	 * semirings work on a mix of zero-dimensional, one-dimensional, and
	 * two-dimensional containers.
	 *
	 * That is, these functions allow various linear algebra operations on
	 * scalars, objects of type alp::Vector, and objects of type alp::Matrix.
	 *
	 * \note The backends of each opaque data type should match.
	 *
	 * @{
	 */

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4, typename IOStructure,
		typename IOView, typename IOImfR, typename IOImfC,
		typename InputType1 = typename Ring::D1, typename InputStructure1,
		typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2 = typename Ring::D2, typename InputStructure2,
		typename InputView2, typename InputImfR2, typename InputImfC2,
		Backend backend
	>
	RC vxm(
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &u,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &v,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &A,
		const Ring &ring = Ring(),
		const std::enable_if_t< alp::is_semiring< Ring >::value > * const = nullptr
	) {
		(void) u;
		(void) v;
		(void) A;
		return UNSUPPORTED;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView,
		typename IOImfR, typename IOImfC,
		typename InputType1, typename InputStructure1, typename InputView1,
		typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2,
		typename InputImfR2, typename InputImfC2,
		class AdditiveMonoid, class MultiplicativeOperator,
		Backend backend
	>
	RC vxm(
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &u,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &v,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &A,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const std::enable_if_t<
			alp::is_monoid< AdditiveMonoid >::value &&
			alp::is_operator< MultiplicativeOperator >::value &&
			!alp::is_object< IOType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			!std::is_same< InputType2, void >::value
		> * const = nullptr
	) {
		(void) u;
		(void) v;
		(void) A;
		(void) add;
		(void) mul;
		return UNSUPPORTED;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4, typename IOStructure,
		typename IOView, typename IOImfR, typename IOImfC,
		typename InputType2 = typename Ring::D2, typename InputStructure2,
		typename InputView2, typename InputImfR2, typename InputImfC2,
		typename InputType1 = typename Ring::D1, typename InputStructure1,
		typename InputView1, typename InputImfR1, typename InputImfC1,
		Backend backend
	>
	RC mxv(
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &u,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &v,
		const Ring &ring,
		const std::enable_if_t< alp::is_semiring< Ring >::value > * const = nullptr
	) {
		(void) u;
		(void) A;
		(void) v;
		(void) ring;
		return UNSUPPORTED;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView,
		typename IOImfR, typename IOImfC,
		typename InputType2, typename InputStructure2, typename InputView2,
		typename InputImfR2, typename InputImfC2,
		typename InputType1, typename InputStructure1, typename InputView1,
		typename InputImfR1, typename InputImfC1,
		class AdditiveMonoid, class MultiplicativeOperator,
		Backend backend
	>
	RC mxv(
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &u,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &v,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const std::enable_if_t<
			alp::is_monoid< AdditiveMonoid >::value &&
			alp::is_operator< MultiplicativeOperator >::value &&
			!alp::is_object< IOType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			!std::is_same< InputType2, void >::value
		> * const = nullptr
	) {
		(void) u;
		(void) A;
		(void) v;
		(void) add;
		(void) mul;
		return UNSUPPORTED;
	}

	/**
	 * @see alp::eWiseLambda for the user-level specification.
	 */
	template<
		typename Func,
		typename DataType, typename Structure, typename View, typename ImfR, typename ImfC,
		Backend backend
	>
	RC eWiseLambda(
		const Func f,
		Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, backend > &A
	) {
		(void) f;
		(void) A;
		return UNSUPPORTED;
	}

	/**
	 * This function provides dimension checking and will defer to the below
	 * function for the actual implementation.
	 *
	 * @see alp::eWiseLambda for the user-level specification.
	 */
	template<
		typename Func,
		typename DataType1, typename Structure1, typename View1, typename ImfR1, typename ImfC1,
		typename DataType2, typename Structure2, typename View2, typename ImfR2, typename ImfC2,
		Backend backend,
		typename... Args
	>
	RC eWiseLambda(
		const Func f,
		Matrix< DataType1, Structure1, Density::Dense, View1, ImfR1, ImfC1, backend > &A,
		const Vector< DataType2, Structure2, Density::Dense, View2, ImfR2, ImfC2, backend > &x,
		Args const &... args
	) {
		// do size checking
		if( !( getLength( x ) == nrows( A ) || getLength( x ) == ncols( A ) ) ) {
			std::cerr << "Mismatching dimensions: given vector of size " << size( x )
				<< " has nothing to do with either matrix dimension (" << nrows( A ) << " nor " << ncols( A ) << ").\n";
			return MISMATCH;
		}

		return eWiseLambda( f, A, args... );
	}

	/**
	 * For all elements in a ALP Matrix \a B, fold the value \f$ \alpha \f$
	 * into each element.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Monoid,
		Backend backend
	>
	RC foldr(
		const Scalar< InputType, InputStructure, backend > &alpha,
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &B,
		const Monoid &monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value &&
			!alp::is_object< IOType >::value &&
			alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		(void) alpha;
		(void) B;
		(void) monoid;
		return UNSUPPORTED;
	}

	/** Folds element-wise alpha into B, operator variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Operator,
		Backend backend
	>
	RC foldr(
		const Scalar< InputType, InputStructure, backend > &alpha,
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &B,
		const Operator &op = Operator(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value &&
			!alp::is_object< IOType >::value &&
			alp::is_operator< Operator >::value
		> * const = nullptr
	) {
		(void) alpha;
		(void) B;
		(void) op;
		return UNSUPPORTED;
	}

	/** Folds element-wise A into B, monoid variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Monoid,
		Backend backend
	>
	RC foldr(
		const Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &A,
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &B,
		const Monoid &monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value &&
			!alp::is_object< IOType >::value &&
			alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		(void) A;
		(void) B;
		(void) monoid;
		return UNSUPPORTED;
	}

	/** Folds element-wise A into B, operator variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Operator,
		Backend backend
	>
	RC foldr(
		const Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &A,
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &B,
		const Operator &op = Operator(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value &&
			!alp::is_object< IOType >::value &&
			alp::is_operator< Operator >::value
		> * const = nullptr
	) {
		(void) A;
		(void) B;
		(void) op;
		return UNSUPPORTED;
	}

	/** Folds element-wise B into A, monoid variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Monoid,
		Backend backend
	>
	RC foldl(
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &A,
		const Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &B,
		const Monoid &monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< IOType >::value &&
			!alp::is_object< InputType >::value &&
			alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		(void) A;
		(void) B;
		(void) monoid;
		return UNSUPPORTED;
	}

	/** Folds element-wise B into A, operator variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Operator,
		Backend backend
	>
	RC foldl(
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &A,
		const Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &B,
		const Operator &op = Operator(),
		const std::enable_if_t<
			!alp::is_object< IOType >::value &&
			!alp::is_object< InputType >::value &&
			alp::is_operator< Operator >::value
		> * const = nullptr
	) {
		(void) A;
		(void) B;
		(void) op;
		return UNSUPPORTED;
	}

	/** Folds element-wise beta into A, monoid variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Monoid,
		Backend backend
	>
	RC foldl(
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &A,
		const Scalar< InputType, InputStructure, backend > &beta,
		const Monoid &monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< IOType >::value &&
			!alp::is_object< InputType >::value &&
			alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		(void) A;
		(void) beta;
		(void) monoid;
		return UNSUPPORTED;
	}

	/** Folds element-wise beta into A, operator variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Operator,
		Backend backend
	>
	RC foldl(
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &A,
		const Scalar< InputType, InputStructure, backend > &beta,
		const Operator &op = Operator(),
		const std::enable_if_t<
			!alp::is_object< IOType >::value &&
			!alp::is_object< InputType >::value &&
			alp::is_operator< Operator >::value
		> * const = nullptr
	) {
		(void) A;
		(void) beta;
		(void) op;
		return UNSUPPORTED;
	}

	/**
	 * Returns a view over the input matrix returning conjugate of the accessed element.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename Structure, typename View, typename ImfR, typename ImfC,
		Backend backend,
		std::enable_if_t<
			!structures::is_a< Structure, structures::Square >::value
		> * = nullptr
	>
	Matrix<
		DataType, Structure, Density::Dense,
		view::Functor< std::function< void( DataType &, const size_t, const size_t ) > >,
		imf::Id, imf::Id,
		backend
	>
	conjugate(
		const Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, backend > &A,
		const std::enable_if_t<
			!alp::is_object< DataType >::value
		> * const = nullptr
	) {
		(void) A;
		return UNSUPPORTED;
	}

	/** Specialization for square matrices */
	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename Structure, typename View, typename ImfR, typename ImfC,
		Backend backend,
		std::enable_if_t<
			structures::is_a< Structure, structures::Square >::value
		> * = nullptr
	>
	Matrix<
		DataType, Structure, Density::Dense,
		view::Functor< std::function< void( DataType &, const size_t, const size_t ) > >,
		imf::Id, imf::Id,
		backend
	>
	conjugate(
		const Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, backend > &A,
		const std::enable_if_t<
			!alp::is_object< DataType >::value
		> * const = nullptr
	) {
		(void) A;
		return UNSUPPORTED;
	}
	/** @} */

} // namespace alp

#endif // end _H_ALP_BLAS2_BASE
