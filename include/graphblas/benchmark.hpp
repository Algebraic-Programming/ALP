
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
 * @author J. W. Nash & A. N. Yzelman
 * @date 17th of April, 2017
 */

#ifndef _H_GRB_BENCH
#define _H_GRB_BENCH

#include "base/benchmark.hpp"
#include "base/config.hpp"

// include specialisations
#ifdef _GRB_WITH_REFERENCE
 #include "graphblas/reference/benchmark.hpp"
#endif
#ifdef _GRB_WITH_BANSHEE
 #include "graphblas/banshee/benchmark.hpp"
#endif
#ifdef _GRB_WITH_LPF
 #include "graphblas/bsp1d/benchmark.hpp"
#endif

#ifdef _GRB_BACKEND
namespace grb {
	template< enum EXEC_MODE mode, enum Backend implementation = config::default_backend >
	class Benchmarker;
}
#endif

#endif // end ``_H_GRB_BENCH''
