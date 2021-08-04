
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
 * @author A. N. Yzelman & J. M. Nash
 * @date 20th of February, 2017
 */

#ifndef _H_GRB_BSP_COLL_BLAS1_VEC
#define _H_GRB_BSP_COLL_BLAS1_VEC

#include <graphblas/reference/vector.hpp>

#include "internal-collectives.hpp"

#define NO_CAST_ASSERT_BLAS1( x, y, z )                                            \
	static_assert( x,                                                              \
		"\n\n"                                                                     \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n"                                                 \
		"*     ERROR      | " y " " z ".\n"                                        \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n"                                                 \
		"* Possible fix 1 | Remove no_casting from the template parameters in "    \
		"this call to " y ".\n"                                                    \
		"* Possible fix 2 | Provide a value of the same type as the first domain " \
		"of the given operator.\n"                                                 \
		"* Possible fix 3 | Ensure the operator given to this call to " y " has "  \
		"all "                                                                     \
		"of "                                                                      \
		"its "                                                                     \
		"domain"                                                                   \
		"s "                                                                       \
		"equal "                                                                   \
		"to "                                                                      \
		"each "                                                                    \
		"other."                                                                   \
		"\n"                                                                       \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n" );

// test: vector has num elements
#define TEST_VEC_SIZE( vec, num )    \
	{                                \
		if( size( vec ) != ( num ) ) \
			return MISMATCH;         \
	}

// test: one vector is a factor of num larger than another vector
#define TEST_VEC_MULTIPLE( vec1, vec2, num )         \
	{                                                \
		if( size( vec1 ) * ( num ) != size( vec2 ) ) \
			return MISMATCH;                         \
	}

#include "collectives_blas1.hpp"

#undef NO_CAST_ASSERT_BLAS1
#undef TEST_VEC_SIZE
#undef TEST_VEC_MULTIPLE

#endif // end ``_H_GRB_BSP_COLL_BLAS1_VEC''
