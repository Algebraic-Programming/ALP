
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
 * @date 29th of March 2017
 */

#ifndef _H_GRB_BLAS1
#define _H_GRB_BLAS1

#include <graphblas/base/blas1.hpp>

#ifdef _GRB_WITH_REFERENCE
 #include <graphblas/reference/blas1.hpp>
#endif
#ifdef _GRB_WITH_BANSHEE
 #include <graphblas/banshee/blas1.hpp>
#endif
#ifdef _GRB_WITH_LPF
 #include <graphblas/bsp1d/blas1.hpp>
#endif

#endif // end ``_H_GRB_BLAS1''

