
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
 * Forward declarations required by the nonblocking backend.
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_FORWARD
#define _H_GRB_NONBLOCKING_FORWARD


namespace grb {

	// The eWiseLambda is a friend of matrix but defined in blas2. Therefore it is
	// forward-declared and this forward definition file included from both
	// matrix.hpp and blas2.hpp
	template<
		class ActiveDistribution,
		typename Func, typename DataType,
		typename RIT, typename CIT, typename NIT
	>
	RC eWiseLambda(
		const Func,
		const Matrix< DataType, nonblocking, RIT, CIT, NIT > &,
		const size_t, const size_t, const size_t, const size_t
	);
	// end eWiseLambda declarations

} // namespace grb

#endif // end ``_H_GRB_NONBLOCKING_FORWARD''

