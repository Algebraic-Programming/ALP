
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
 * Forward declarations required by the Ascend backend.
 *
 * @author A. N. Yzelman
 * @date 12th of September, 2023
 */

#ifndef _H_GRB_ASCEND_FORWARD
#define _H_GRB_ASCEND_FORWARD


namespace grb {

	// The eWiseLambda is a friend of matrix but defined in blas2. Therefore it is
	// forward-declared and this forward definition file included from both
	// matrix.hpp and blas2.hpp
	template<
		class ActiveDistribution = internal::Distribution< ascend >,
		typename Func, typename DataType,
		typename RIT, typename CIT, typename NIT
	>
	RC eWiseLambda(
		const Func f,
		const Matrix< DataType, ascend, RIT, CIT, NIT > &A,
		const size_t s = 0, const size_t P = 1
	);
	// end eWiseLambda declarations

} // namespace grb

#endif // end ``_H_GRB_ASCEND_FORWARD''

