
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
 */

#ifndef _H_ALP_BLAS3_BASE
#define _H_ALP_BLAS3_BASE

#include <alp/backends.hpp>
#include <alp/phase.hpp>
#include <alp/identities.hpp>
#include <alp/monoid.hpp>
#include <graphblas/utils/iscomplex.hpp> // use from grb

#include "matrix.hpp"
#include "vector.hpp"
#include "io.hpp"

namespace alp {

	/**
	 * \defgroup BLAS3 The Level-3 Basic Linear Algebra Subroutines (BLAS)
	 *
	 * A collection of functions that allow GraphBLAS semirings to work on
	 * one or more two-dimensional sparse containers (i.e, sparse matrices).
	 *
	 * @{
	 */

	/**
	 * @brief Computes \f$ C = A . B \f$ for a given monoid.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class MulMonoid,
		Backend backend
	>
	RC eWiseApply(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &C,
		const Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &A,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &B,
		const MulMonoid &mulmono,
		const std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_monoid< MulMonoid >::value
		> * const = nullptr
	) {
		(void) C;
		(void) A;
		(void) B;
		(void) mulmono;
		return UNSUPPORTED;
	}


	/**
	 * Computes \f$ C = alpha . B \f$ for a given monoid.
	 *
	 * Case where \a A is a scalar.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class MulMonoid,
		Backend backend
	>
	RC eWiseApply(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &C,
		const Scalar< InputType1, InputStructure1, backend > &alpha,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &B,
		const MulMonoid &mulmono,
		const std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_monoid< MulMonoid >::value
		> * const = nullptr
	) {
		(void) C;
		(void) alpha;
		(void) B;
		(void) mulmono;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ C = A . beta \f$ for a given monoid.
	 *
	 * Case where \a B is a scalar.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2,
		class MulMonoid,
		Backend backend
	>
	RC eWiseApply(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &C,
		const Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &A,
		const Scalar< InputType2, InputStructure2, backend > &beta,
		const MulMonoid &mulmono,
		const std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_monoid< MulMonoid >::value
		> * const = nullptr
	) {
		(void) C;
		(void) A;
		(void) beta;
		(void) mulmono;
		return UNSUPPORTED;
	}

	/**
	 * Calculates the element-wise multiplication of two matrices,
	 *     \f$ C = C + A .* B \f$,
	 * under a given semiring.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		Backend backend
	>
	RC eWiseMul(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &C,
		const Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &A,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &B,
		const Ring &ring = Ring(),
		const std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_semiring< Ring >::value
		> * const = nullptr
	) {
		(void) C;
		(void) A;
		(void) B;
		(void) ring;
		return UNSUPPORTED;
	}

	/**
	 * eWiseMul, version where A is a scalar.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		Backend backend
	>
	RC eWiseMul(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &C,
		const Scalar< InputType1, InputStructure1, backend > &alpha,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &B,
		const Ring &ring = Ring(),
		const std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_semiring< Ring >::value
		> * const = nullptr
	) {
		(void) C;
		(void) alpha;
		(void) B;
		(void) ring;
		return UNSUPPORTED;
	}

	/**
	 * eWiseMul, version where B is a scalar.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2,
		Backend backend
	>
	RC eWiseMul(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &C,
		const Matrix< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &A,
		const Scalar< InputType2, InputStructure2, backend > &beta,
		const Ring &ring = Ring(),
		const std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_semiring< Ring >::value
		> * const = nullptr
	) {
		(void) C;
		(void) A;
		(void) beta;
		(void) ring;
		return UNSUPPORTED;
	}

	/**
	 * @brief  Outer product of two vectors. The result matrix \a A will contain \f$ uv^T \f$.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Operator,
		Backend backend
	>
	RC outer(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &u,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &v,
		const Operator &mul = Operator(),
		const std::enable_if_t<
			alp::is_operator< Operator >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			!alp::is_object< OutputType >::value
		> * const = nullptr
	) {
		(void) A;
		(void) u;
		(void) v;
		(void) mul;
		return UNSUPPORTED;
	}

	/**
	 * Returns a view over the general rank-1 matrix computed with the outer product.
	 * This avoids creating the resulting container. The elements are calculated lazily on access.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Operator,
		Backend backend
	>
	Matrix<
		typename Operator::D3, structures::General, Density::Dense,
		view::Functor< std::function< void( InputType1 &, const size_t, const size_t ) > >,
		imf::Id, imf::Id,
		backend
	>
	outer(
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &y,
		const Operator &mul = Operator(),
		const typename std::enable_if<
			alp::is_operator< Operator >::value &&
			! alp::is_object< InputType1 >::value &&
			! alp::is_object< InputType2 >::value
		> * const = nullptr
	) {
		(void) x;
		(void) y;
		(void) mul;
		return UNSUPPORTED;
	}

	/**
	 * Returns a view over the general rank-1 matrix computed with the outer product.
	 * Version for the case when input vectors are the same vector,
	 * which results in a symmetric matrix.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		class Operator,
		Backend backend
	>
	Matrix<
		typename Operator::D3,
		typename std::conditional<
			grb::utils::is_complex< typename Operator::D3 >::value,
			alp::structures::Hermitian,
			alp::structures::Symmetric
		>::type,
		Density::Dense,
		view::Functor< std::function< void( typename Operator::D3 &, const size_t, const size_t ) > >,
		imf::Id, imf::Id,
		backend
	>
	outer(
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &x,
		const Operator &mul = Operator(),
		const std::enable_if_t<
			alp::is_operator< Operator >::value &&
			!alp::is_object< InputType >::value
		> * const = nullptr
	) {
		(void) x;
		(void) mul;
		return UNSUPPORTED;
	}
	/**
	 * @}
	 */

} // namespace alp

#endif // end _H_ALP_BLAS3_BASE
