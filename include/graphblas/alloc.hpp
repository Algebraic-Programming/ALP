
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
 * @date 13th of September, 2017
 */

#ifndef _H_GRB_ALLOC
#define _H_GRB_ALLOC

#include <stdlib.h> //posix_memalign

#include <utility> //std_forward

#include <assert.h>

#include <graphblas/base/alloc.hpp>
#include <graphblas/config.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/utils/autodeleter.hpp>

// include available allocator implementations:
#ifdef _GRB_WITH_REFERENCE
 #include "graphblas/reference/alloc.hpp"
#endif
#ifdef _GRB_WITH_HYPERDAGS
 #include "graphblas/hyperdags/alloc.hpp"
#endif
#ifdef _GRB_WITH_NONBLOCKING
 #include "graphblas/nonblocking/alloc.hpp"
#endif
#ifdef _GRB_WITH_LPF
 #include "graphblas/bsp1d/alloc.hpp"
#endif
#ifdef _GRB_WITH_BANSHEE
 #include "graphblas/banshee/alloc.hpp"
#endif

// specify default only if requested during compilation
#ifdef _GRB_BACKEND
namespace grb {
	template< enum Backend implementation = config::default_backend >
	class AllocatorFunctions ;

	template< enum Backend implementation = config::default_backend >
	class Allocator;
}
#endif

#endif // _H_GRB_ALLOC

