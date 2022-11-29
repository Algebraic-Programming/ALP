
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
 * @date 14th of January 2022
 */

#ifndef _H_ALP_DISPATCH_BLAS0
#define _H_ALP_DISPATCH_BLAS0

#include <type_traits> // std::enable_if, std::is_same

#include <alp/base/blas0.hpp>
#include <alp/backends.hpp>
#include <alp/rc.hpp>
#include <alp/descriptors.hpp>
#include <alp/type_traits.hpp>

#include "scalar.hpp"

#include <alp/reference/blas0.hpp> // for internal apply and fold

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
		"* Possible fix 2 | Provide a value that matches the expected type.\n" \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );

namespace alp {

	/**
	 * @brief Reference implementation of \a foldl.
	 */
	template<
		class OP,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure >
	RC foldl( Scalar< IOType, IOStructure, dispatch > &x,
		const Scalar< InputType, InputStructure, dispatch > &y,
		const OP & op = OP(),
		const typename std::enable_if< is_operator< OP >::value && ! is_object< InputType >::value && ! is_object< IOType >::value, void >::type * = NULL ) {

		RC rc = internal::foldl( *x, *y, op );

		return rc;
	}

} // end namespace ``alp''

#undef NO_CAST_ASSERT

#endif // end ``_H_ALP_DISPATCH_BLAS0''

