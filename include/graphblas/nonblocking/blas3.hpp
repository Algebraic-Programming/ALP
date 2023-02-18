
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
 * Implements the level-3 primitives for the nonblocking backend
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_BLAS3
#define _H_GRB_NONBLOCKING_BLAS3

#include <type_traits> //for std::enable_if

#include <graphblas/base/blas3.hpp>
#include <graphblas/utils/iterators/MatrixVectorIterator.hpp>

#include "io.hpp"
#include "matrix.hpp"

#include <omp.h>

#define NO_CAST_ASSERT( x, y, z )                                              \
	static_assert( x,                                                          \
		"\n\n"                                                                 \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"*     ERROR      | " y " " z ".\n"                                    \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"* Possible fix 1 | Remove no_casting from the template parameters "   \
		"in this call to " y ".\n"                                             \
		"* Possible fix 2 | For all mismatches in the domains of input "       \
		"parameters and the semiring domains, as specified in the "            \
		"documentation of the function " y ", supply a container argument of " \
		"the expected type instead.\n"                                         \
		"* Possible fix 3 | Provide a compatible semiring where all domains "  \
		"match those of the container arguments, as specified in the "         \
		"documentation of the function " y ".\n"                               \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );


namespace grb {

	namespace internal {

		extern LazyEvaluation le;

	}

}

namespace grb {

	namespace internal {

		template<
			bool allow_void,
			Descriptor descr,
			class MulMonoid,
			typename OutputType,
			typename InputType1,
			typename InputType2,
			typename RIT,
			typename CIT,
			typename NIT,
			class Operator,
			class Monoid
		>
		RC mxm_generic(
			Matrix< OutputType, nonblocking, RIT, CIT, NIT > &C,
			const Matrix< InputType1, nonblocking, RIT, CIT, NIT > &A,
			const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &B,
			const Operator &oper,
			const Monoid &monoid,
			const MulMonoid &mulMonoid,
			const Phase &phase,
			const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value && !
				grb::is_object< InputType2 >::value &&
				grb::is_operator< Operator >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = nullptr
		) {
			(void) C;
			(void) A;
			(void) B;
			(void) oper;
			(void) monoid;
			(void) mulMonoid;
			(void) phase;
			return UNSUPPORTED;
		}

	} // end namespace grb::internal

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		typename RIT,
		typename CIT,
		typename NIT,
		class Semiring
	>
	RC mxm(
		Matrix< OutputType, nonblocking, RIT, CIT, NIT > &C,
		const Matrix< InputType1, nonblocking, RIT, CIT, NIT > &A,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &B,
		const Semiring &ring = Semiring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Semiring >::value,
		void >::type * const = nullptr
	) {
		(void) C;
		(void) A;
		(void) B;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	template<
		Descriptor descr = grb::descriptors::no_operation,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		typename RIT,
		typename CIT,
		typename NIT,
		class Operator,
		class Monoid
	>
	RC mxm(
		Matrix< OutputType, nonblocking, RIT, CIT, NIT > &C,
		const Matrix< InputType1, nonblocking, RIT, CIT, NIT > &A,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &B,
		const Monoid &addM,
		const Operator &mulOp,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
		(void) C;
		(void) A;
		(void) B;
		(void) addM;
		(void) mulOp;
		(void) phase;
		return UNSUPPORTED;
	}

	namespace internal {

		template<
			Descriptor descr = descriptors::no_operation,
			bool matrix_is_void,
			typename OutputType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename Coords
		>
		RC matrix_zip_generic(
			Matrix< OutputType, nonblocking > &A,
			const Vector< InputType1, nonblocking, Coords > &x,
			const Vector< InputType2, nonblocking, Coords > &y,
			const Vector< InputType3, nonblocking, Coords > &z,
			const Phase &phase
		) {
			(void) A;
			(void) x;
			(void) y;
			(void) z;
			(void) phase;
			return UNSUPPORTED;
		}

	} // namespace internal

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename Coords
	>
	RC zip(
		Matrix< OutputType, nonblocking > &A,
		const Vector< InputType1, nonblocking, Coords > &x,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Vector< InputType3, nonblocking, Coords > &z,
		const Phase &phase = EXECUTE
	) {
		(void) A;
		(void) x;
		(void) y;
		(void) z;
		(void) phase;
		return UNSUPPORTED;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC zip(
		Matrix< void, nonblocking > &A,
		const Vector< InputType1, nonblocking, Coords > &x,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Phase &phase = EXECUTE
	) {
		(void) A;
		(void) x;
		(void) y;
		(void) phase;
		return UNSUPPORTED;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType1,
		typename InputType2,
		typename OutputType,
		typename Coords,
		class Operator
	>
	RC outer(
		Matrix< OutputType, nonblocking > &A,
		const Vector< InputType1, nonblocking, Coords > &u,
		const Vector< InputType2, nonblocking, Coords > &v,
		const Operator &mul = Operator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_operator< Operator >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				!grb::is_object< OutputType >::value,
			void
		>::type * const = nullptr
	) {
		(void) A;
		(void) u;
		(void) v;
		(void) mul;
		(void) phase;
		return UNSUPPORTED;
	}

	namespace internal {

		template<
			bool allow_void,
			Descriptor descr,
			class MulMonoid,
			typename OutputType,
			typename InputType1,
			typename InputType2,
			class Operator
		>
		RC eWiseApply_matrix_generic(
			Matrix< OutputType, nonblocking > &C,
			const Matrix< InputType1, nonblocking > &A,
			const Matrix< InputType2, nonblocking > &B,
			const Operator &oper,
			const MulMonoid &mulMonoid,
			const Phase &phase,
			const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_operator< Operator >::value,
			void >::type * const = nullptr
		) {
			(void) C;
			(void) A;
			(void) B;
			(void) oper;
			(void) mulMonoid;
			(void) phase;
			return UNSUPPORTED;
		}

	} // namespace internal

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		class MulMonoid
	>
	RC eWiseApply(
		Matrix< OutputType, nonblocking > &C,
		const Matrix< InputType1, nonblocking > &A,
		const Matrix< InputType2, nonblocking > &B,
		const MulMonoid &mulmono,
		const Phase phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< MulMonoid >::value,
		void >::type * const = nullptr
	) {
		(void) C;
		(void) A;
		(void) B;
		(void) mulmono;
		(void) phase;
		return UNSUPPORTED;
	}

	template<
		Descriptor descr = grb::descriptors::no_operation,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		class Operator
	>
	RC eWiseApply(
		Matrix< OutputType, nonblocking > &C,
		const Matrix< InputType1, nonblocking > &A,
		const Matrix< InputType2, nonblocking > &B,
		const Operator &mulOp,
		const Phase phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value,
		void >::type * const = nullptr
	) {
		(void) C;
		(void) A;
		(void) B;
		(void) mulOp;
		(void) phase;
		return UNSUPPORTED;
	}

} // namespace grb

#undef NO_CAST_ASSERT

#endif // ``_H_GRB_REFERENCE_BLAS3''

