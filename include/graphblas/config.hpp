
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
 * @date 8th of August, 2016
 */

#ifndef _H_GRB_CONFIG
#define _H_GRB_CONFIG

#include "base/config.hpp"

// include all active configurations
#ifdef _GRB_WITH_REFERENCE
 #include "graphblas/reference/config.hpp"
#endif
#ifdef _GRB_WITH_HYPERDAGS
 #include "graphblas/hyperdags/config.hpp"
#endif
#ifdef _GRB_WITH_NONBLOCKING
 #include "graphblas/nonblocking/config.hpp"
#endif
#ifdef _GRB_WITH_OMP
 #include "graphblas/omp/config.hpp"
#endif
#ifdef _GRB_WITH_LPF
 #include "graphblas/bsp1d/config.hpp"
#endif
#ifdef _GRB_WITH_BANSHEE
 #include "graphblas/banshee/config.hpp"
#endif
#ifdef _GRB_WITH_TUTORIAL
 #include "graphblas/tutorial/config.hpp"
#endif

#endif // end ``_H_GRB_CONFIG''

