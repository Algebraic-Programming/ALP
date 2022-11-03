
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
 * @date 15 March, 2016
 */

#ifndef _H_ALP_MONOID
#define _H_ALP_MONOID

#ifdef _DEBUG
#include <cstdio>
#endif

#include <cstddef> //size_t
#include <cstdlib> //posix_memalign, rand
#include <type_traits>

#include <assert.h>

#include <alp/identities.hpp>
#include <alp/ops.hpp>
#include <alp/type_traits.hpp>
#include <graphblas/monoid.hpp>

/**
 * The main Sparse Library namespace.
 *
 * All classes, enums, constants, and functions are declared in this namespace.
 * This source file only contains testing code outside this namespace.
 */
namespace alp {

	/**
	 * @see grb::Monoid
	 */
	template< class _OP, template< typename > class _ID >
	using Monoid = grb::Monoid< _OP, _ID >;
	// type traits

} // namespace alp

#endif
