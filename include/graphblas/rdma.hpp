/*
 *   Copyright 2025 Huawei Technologies Co., Ltd.
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
 * @author G. Gaio
 * @date December, 2025
 */

#ifndef _H_GRB_RDMA
#define _H_GRB_RDMA

#include "base/config.hpp"
#include "base/rdma.hpp"

#ifdef _GRB_WITH_REFERENCE
 #include "graphblas/reference/rdma.hpp"
#endif
#ifdef _GRB_WITH_HYPERDAGS
 #include <graphblas/hyperdags/rdma.hpp>
#endif
#ifdef _GRB_WITH_NONBLOCKING
 #include "graphblas/nonblocking/rdma.hpp"
#endif
#ifdef _GRB_WITH_LPF
 #include "graphblas/bsp1d/rdma.hpp"
#endif

// specify default only if requested during compilation
#ifdef _GRB_BACKEND
namespace grb {
	template< Backend implementation = config::default_backend >
	class rdma;
}
#endif

#endif // end _H_GRB_RDMA
