
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
 * Forward declarations required for the reference and reference_omp backend
 * implementation.
 *
 * @author A. N. Yzelman
 *
 * Note: this file is a newer version of the older forward.hpp and exists for
 *       similar reasons. It does, however, not at all contain the same
 *       declarations-- must instead have moved to the include/graphblas/base
 *       directory.
 */

#ifndef _H_GRB_REFERENCE_FORWARD
#define _H_GRB_REFERENCE_FORWARD

namespace grb {

	// The eWiseLambda is a friend of matrix but defined in blas2. Therefore it is
	// forward-declared and this forward definition file included from both
	// matrix.hpp and blas2.hpp
	template<
		class ActiveDistribution = internal::Distribution< reference >,
		typename Func, typename DataType,
		typename RIT, typename CIT, typename NIT
	>
	RC eWiseLambda(
		const Func f,
		const Matrix< DataType, reference, RIT, CIT, NIT > &A,
		const size_t s = 0, const size_t P = 1
	);

	template<
		class ActiveDistribution = internal::Distribution< reference_omp >,
		typename Func, typename DataType,
		typename RIT, typename CIT, typename NIT
	>
	RC eWiseLambda(
		const Func f,
		const Matrix< DataType, reference_omp, RIT, CIT, NIT > &A,
		const size_t s = 0, const size_t P = 1
	);
	// end eWiseLambda declarations

} // namespace grb

#endif // end ``_H_GRB_REFERENCE_FORWARD''

