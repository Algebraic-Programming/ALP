
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
 * @date 5th of December 2016
 */

#ifndef _H_ALP_BASE_BLAS1
#define _H_ALP_BASE_BLAS1

#include <alp/rc.hpp>
#include <alp/ops.hpp>
#include <alp/phase.hpp>
#include <alp/monoid.hpp>
#include <alp/backends.hpp>
#include <alp/semiring.hpp>
#include <alp/descriptors.hpp>
#include <alp/internalops.hpp>

#include <assert.h>


namespace alp {

	/**
	 * \defgroup BLAS1 The Level-1 ALP/GraphBLAS routines
	 * @{
	 */

	/**
	 * Folds all elements in a ALP Vector \a x into a single value \a beta.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType, typename IOStructure,
		class Monoid,
		Backend backend
	>
	RC foldr(
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &x,
		Scalar< IOType, IOStructure, backend > &beta,
		const Monoid &monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value && !alp::is_object< IOType >::value && alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		(void) x;
		(void) beta;
		(void) monoid;
		return UNSUPPORTED;
	}

	/** C++ scalar variant */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType,
		class Monoid,
		Backend backend
	>
	RC foldr(
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &x,
		IOType &beta,
		const Monoid &monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value && !alp::is_object< IOType >::value && alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		return foldr( x, Scalar< IOType >( beta ), monoid );
	}

	/**
	 * For all elements in a ALP Vector \a y, fold the value \f$ \alpha \f$
	 * into each element.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Monoid,
		Backend backend
	>
	RC foldr(
		const Scalar< InputType, InputStructure, backend > &alpha,
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &y,
		const Monoid & monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value && !alp::is_object< IOType >::value && alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		(void) alpha;
		(void) y;
		(void) monoid;
		return UNSUPPORTED;
	}

	/**
	 * Computes y = x + y, operator variant.
	 *
	 * Specialisation for scalar \a x.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class OP, Backend backend
	>
	RC foldr(
		const Scalar< InputType, InputStructure, backend > &alpha,
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &y,
		const OP & op = OP(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value && ! alp::is_object< IOType >::value && alp::is_operator< OP >::value
		> * const = nullptr
	) {
		(void) alhpa;
		(void) y;
		(void) op;
		return UNSUPPORTED;
	}

	/**
	 * Folds all elements in a ALP Vector \a x into the corresponding
	 * elements from an input/output vector \a y.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class OP, Backend backend
	>
	RC foldr(
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &x,
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &y,
		const OP & op = OP(),
		const std::enable_if_t<
			alp::is_operator< OP >::value && ! alp::is_object< InputType >::value && ! alp::is_object< IOType >::value
		> * = nullptr
	) {
		(void) x;
		(void) y;
		(void) op;
		return UNSUPPORTED;
	}

	/**
	 * Folds all elements in a ALP Vector \a x into the corresponding
	 * elements from an input/output vector \a y.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Monoid,
		Backend backend
	>
	RC foldr(
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &x,
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &y,
		const Monoid & monoid = Monoid(),
		const std::enable_if_t<
			alp::is_monoid< Monoid >::value && ! alp::is_object< InputType >::value && ! alp::is_object< IOType >::value
		> * = nullptr
	) {
		(void) x;
		(void) y;
		(void) monoid;
		return UNSUPPORTED;
	}

	/**
	 * For all elements in a ALP Vector \a x, fold the value \f$ \beta \f$
	 * into each element.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType, typename InputStructure,
		class Op,
		Backend backend
	>
	RC foldl(
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &x,
		const Scalar< InputType, InputStructure, backend > beta,
		const Op &op = Op(),
		const std::enable_if_t<
			! alp::is_object< IOType >::value && ! alp::is_object< InputType >::value && alp::is_operator< Op >::value
		> * = nullptr
	) {
		(void) x;
		(void) beta;
		(void) op;
		return UNSUPPORTED;
	}

	/**
	 * Folds all elements in a ALP Vector \a y into the corresponding
	 * elements from an input/output vector \a x.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		class OP,
		Backend backend
	>
	RC foldl(
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &x,
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &y,
		const OP &op = OP(),
		const std::enable_if_t<
			alp::is_operator< OP >::value && !alp::is_object< IOType >::value && !alp::is_object< InputType >::value
		> * = nullptr
	) {
		(void) x;
		(void) y;
		(void) op;
		return UNSUPPORTED;
	}

	/**
	 * Folds all elements in a ALP Vector \a y into the corresponding
	 * elements from an input/output vector \a x.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		class Monoid,
		Backend backend
	>
	RC foldl(
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, backend > &x,
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &y,
		const Monoid &monoid = Monoid(),
		const std::enable_if_t<
			alp::is_monoid< Monoid >::value && ! alp::is_object< IOType >::value && ! alp::is_object< InputType >::value
		  > * = nullptr
		) {
		(void) x;
		(void) y;
		(void) monoid;
		return UNSUPPORTED;
	}

	/**
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = x .* \beta \f$, using the given operator.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR, typename InputImfC,
		typename InputType2, typename InputStructure2,
		class OP,
		Backend backend
	>
	RC eWiseApply(
		Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR, InputImfC, backend > &x,
		const Scalar< InputType2, InputStructure2, backend > &beta,
		const OP &op = OP(),
		const typename std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_operator< OP >::value
		> * const = nullptr
	) {
		(void) z;
		(void) x;
		(void) beta;
		(void) op;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for \a x and \a y scalar, operator version.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2,
		class OP,
		Backend backend
	>
	RC eWiseApply(
		Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &z,
		const Scalar< InputType1, InputStructure1, backend> &alpha,
		const Scalar< InputType2, InputStructure2, backend> &beta,
		const OP &op = OP(),
		const typename std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_operator< OP >::value
		> * const = nullptr
	) {
		(void) z;
		(void) alpha;
		(void) beta;
		(void) op;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for \a x and \a y scalar, monoid version.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2,
		class Monoid,
		Backend backend
	>
	RC eWiseApply(
		Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &z,
		const Scalar< InputType1, InputStructure1, backend> &alpha,
		const Scalar< InputType2, InputStructure2, backend> &beta,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		(void) z;
		(void) alhpa;
		(void) beta;
		(void) monoid;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Monoid version.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Monoid,
		Backend backend
	>
	RC eWiseApply(
		Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &y,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if_t<
			! alp::is_object< OutputType >::value &&
			! alp::is_object< InputType1 >::value &&
			! alp::is_object< InputType2 >::value &&
			alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		(void) z;
		(void) x;
		(void) y;
		(void) monoid;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for scalar \a x. Monoid version.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Monoid,
		Backend backend
	>
	RC eWiseApply(
		Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &z,
		const Scalar< InputType1, InputStructure1, backend> &alpha,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &y,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		(void) z;
		(void) alhpa;
		(void) y;
		(void) monoid;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for scalar \a y. Monoid version.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2,
		class Monoid,
		Backend backend
	>
	RC eWiseApply(
		Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &x,
		const Scalar< InputType2, InputStructure2, backend > &beta,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		(void) z;
		(void) x;
		(void) beta;
		(void) monoid;
		return UNSUPPORTED;
	}

	/**
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = \alpha .* y \f$, using the given operator.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class OP,
		Backend backend
	>
	RC eWiseApply(
		Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &z,
		const Scalar< InputType1, InputStructure1, backend > &alpha,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &y,
		const OP &op = OP(),
		const typename std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_operator< OP >::value
		> * const = nullptr
	) {
		(void) z;
		(void) alpha;
		(void) y;
		(void) op;
		return UNSUPPORTED;
	}

	/**
	 * Calculates the element-wise operation on elements of two vectors,
	 * \f$ z = x .* y \f$, using the given operator.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class OP,
		Backend backend
	>
	RC eWiseApply(
		Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &y,
		const OP &op = OP(),
		const typename std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_operator< OP >::value
		> * const = nullptr
	) {
		(void) z;
		(void) x;
		(void) y;
		(void) op;
		return UNSUPPORTED;
	}

	/**
	 * Calculates the element-wise multiplication of two vectors,
	 *     \f$ z = z + x .* y \f$,
	 * under a given semiring.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Ring,
		Backend backend
	>
	RC eWiseMul(
		Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &y,
		const Ring &ring = Ring(),
		const typename std::enable_if_t<
			! alp::is_object< OutputType >::value &&
			! alp::is_object< InputType1 >::value &&
			! alp::is_object< InputType2 >::value &&
			alp::is_semiring< Ring >::value
		> * const = nullptr
	) {
		(void) z;
		(void) x;
		(void) y;
		(void) ring;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = z + x * y \f$.
	 *
	 * Specialisation for scalar \a x.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Ring,
		Backend backend
	>
	RC eWiseMul(
		Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &z,
		const Scalar< InputType1, InputStructure1, backend > &alpha,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &y,
		const Ring &ring = Ring(),
		const typename std::enable_if_t<
			! alp::is_object< OutputType >::value &&
			! alp::is_object< InputType1 >::value &&
			! alp::is_object< InputType2 >::value &&
			alp::is_semiring< Ring >::value
		> * const = nullptr
	) {
		(void) z;
		(void) alpha;
		(void) y;
		(void) ring;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = z + x * y \f$.
	 *
	 * Specialisation for scalar \a y.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2,
		class Ring
	>
	RC eWiseMul(
		Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, backend > &z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &x,
		const Scalar< InputType2, InputStructure2, backend > &beta,
		const Ring &ring = Ring(),
		const typename std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_semiring< Ring >::value
		> * const = nullptr
	) {
		(void) z;
		(void) x;
		(void) beta;
		(void) ring;
		return UNSUPPORTED;
	}

	/**
	 * Calculates the dot product, \f$ \alpha = (x,y) \f$, under a given additive
	 * monoid and multiplicative operator.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class AddMonoid, class AnyOp
	>
	RC dot(
		Scalar< OutputType, OutputStructure, backend > &z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &y,
		const AddMonoid &addMonoid = AddMonoid(),
		const AnyOp &anyOp = AnyOp(),
		const typename std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_monoid< AddMonoid >::value &&
			alp::is_operator< AnyOp >::value
		> * const = nullptr
	) {
		(void) x;
		(void) y;
		(void) addMonoid;
		(void) anyOp;
		return UNSUPPORTED;
	}

	/** C++ scalar specialization */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class AddMonoid, class AnyOp,
		Backend backend
	>
	RC dot(
		OutputType &z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &y,
		const AddMonoid &addMonoid = AddMonoid(),
		const AnyOp &anyOp = AnyOp(),
		const typename std::enable_if_t<
			!alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_monoid< AddMonoid >::value &&
			alp::is_operator< AnyOp >::value
		>::type * const = nullptr
	) {
		return UNSUPPORTED;
	}

	/**
	 * Provides a generic implementation of the dot computation on semirings by
	 * translating it into a dot computation on an additive commutative monoid
	 * with any multiplicative operator.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Ring,
		Backend backend
	>
	RC dot(
		Scalar< IOType, IOStructure, backend > &x,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &left,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &right,
		const Ring &ring = Ring(),
		const typename std::enable_if_t<
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			!alp::is_object< IOType >::value &&
			alp::is_semiring< Ring >::value,
		> * const = nullptr
	) {
		return alp::dot< descr >( x,
			left, right,
			ring.getAdditiveMonoid(),
			ring.getMultiplicativeOperator()
		);
	}

	/** C++ scalar specialization. */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename IOType,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		Backend backend
	>
	RC dot(
		IOType &x,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &left,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &right,
		const Ring &ring = Ring(),
		const typename std::enable_if_t<
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			!alp::is_object< IOType >::value &&
			alp::is_semiring< Ring >::value,
		> * const = nullptr
	) {
		(void) x;
		(void) left;
		(void) right;
		(void) ring;
		return UNSUPPORTED;
	}

	/**
	 * This is the eWiseLambda that performs length checking by recursion.
	 *
	 * in the backend implementation all vectors are distributed equally, so no
	 * need to synchronise any data structures. We do need to do error checking
	 * though, to see when to return alp::MISMATCH. That's this function.
	 *
	 * @see Vector::operator[]()
	 * @see Vector::lambda_backend
	 */
	template<
		typename Func,
		typename DataType1, typename DataStructure1, typename DataView1, typename InputImfR1, typename InputImfC1,
		typename DataType2, typename DataStructure2, typename DataView2, typename InputImfR2, typename InputImfC2,
		Backend backend,
		typename... Args
	>
	RC eWiseLambda(
		const Func f,
		Vector< DataType1, DataStructure1, Density::Dense, DataView1, InputImfR1, InputImfC1, backend > &x,
		const Vector< DataType2, DataStructure2, Density::Dense, DataView2, InputImfR2, InputImfC2, backend > &y,
		Args const &... args
	) {
		(void) f;
		(void) x;
		(void) y;
		(void) args;
		return UNSUPPORTED;
	}

	/**
	 * No implementation notes. This is the `real' implementation on backend
	 * vectors.
	 *
	 * @see Vector::operator[]()
	 * @see Vector::lambda_backend
	 */
	template<
		typename Func,
		typename DataType, typename DataStructure, typename DataView, typename DataImfR, typename DataImfC,
		Backend backend
	>
	RC eWiseLambda(
		const Func f,
		Vector< DataType, DataStructure, Density::Dense, DataView, DataImfR, DataImfC, backend > &x
	) {
		(void) f;
		(void) x;
		return UNSUPPORTED;
	}

	/**
	 * Reduces a vector into a scalar. Reduction takes place according a monoid
	 * \f$ (\oplus,1) \f$, where \f$ \oplus:\ D_1 \times D_2 \to D_3 \f$ with an
	 * associated identity \f$ 1 \in \{D_1,D_2,D_3\} \f$. Elements from the given
	 * vector \f$ y \in \{D_1,D_2\} \f$ will be applied at the left-hand or right-
	 * hand side of \f$ \oplus \f$; which, exactly, is implementation-dependent
	 * but should not matter since \f$ \oplus \f$ should be associative.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		class Monoid,
		Backend backend
	>
	RC foldl(
		Scalar< IOType, IOStructure, backend > &alpha,
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &y,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if_t<
			! alp::is_object< IOType >::value && ! alp::is_object< InputType >::value && alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		(void) alpha;
		(void) y;
		(void) monoid;
		return UNSUPPORTED;
	}

	/**
	 * Sort vectors, function available to user, e.g. to sort eigenvectors
	 */
	template<
		typename IndexType, typename IndexStructure, typename IndexView, typename IndexImfR, typename IndexImfC,
		typename ValueType, typename ValueStructure, typename ValueView, typename ValueImfR, typename ValueImfC,
		typename Compare,
		Backend backend
	>
	RC sort(
		Vector< IndexType, IndexStructure, Density::Dense, IndexView, IndexImfR, IndexImfC, backend > &permutation,
		const Vector< ValueType, ValueStructure, Density::Dense, ValueView, ValueImfR, ValueImfC, backend > &toSort,
		Compare cmp
	) noexcept {
		(void) permutation;
		(void) toSort;
		(void) cmp
		return UNSUPPORTED;
	}

    /**
	 * Provides a generic implementation of the 2-norm computation.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		class Ring,
		Backend backend
	>
	RC norm2(
		Scalar< OutputType, OutputStructure, backend > &x,
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &y,
		const Ring &ring = Ring(),
		const typename std::enable_if_t<
			std::is_floating_point< OutputType >::value || grb::utils::is_complex< OutputType >::value
		> * const = nullptr
	) {
		(void) x;
		(void) y;
		(void) ring;
		return UNSUPPORTED;
	}

	/** C++ scalar version */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		class Ring,
		Backend backend
	>
	RC norm2(
		OutputType &x,
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &y,
		const Ring &ring = Ring(),
		const typename std::enable_if_t<
			std::is_floating_point< OutputType >::value || grb::utils::is_complex< OutputType >::value
		> * const = nullptr
	) {
		(void) x;
		(void) y;
		(void) ring;
		return UNSUPPORTED;
	}

	/** @} */

} // end namespace alp

#endif // end _H_ALP_BASE_BLAS1

