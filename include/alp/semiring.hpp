
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
 * @date 15th of March, 2016
 */

#ifndef _H_ALP_SEMIRING
#define _H_ALP_SEMIRING

#include <alp/identities.hpp>
#include <alp/monoid.hpp>
#include <alp/ops.hpp>
#include <graphblas/semiring.hpp>

/**
 * The main GraphBLAS namespace.
 */
namespace alp {

	/**
	 * @see grb::Semiring
	 */
	template< class _OP1, class _OP2, template< typename > class _ID1, template< typename > class _ID2 >
	using Semiring = grb::Semiring< _OP1, _OP2, _ID1, _ID2 >;

} // namespace alp

#endif
