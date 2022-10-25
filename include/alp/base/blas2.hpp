
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

	/** @} */

} // namespace alp

#endif // end _H_ALP_BLAS2_BASE
