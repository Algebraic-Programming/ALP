
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
 * @date: 30th of March 2017.
 */

#ifndef _H_GRB_BLAS2
#define _H_GRB_BLAS2

#include "base/blas2.hpp"

// now include all specialisations contained in the backend directories:
#ifdef _GRB_WITH_REFERENCE
 #include <graphblas/reference/blas2.hpp>
#endif
#ifdef _GRB_WITH_DENSE
 #include <graphblas/denseref/blas2.hpp>
#endif
#ifdef _GRB_WITH_HYPERDAGS
 #include <graphblas/hyperdags/blas2.hpp>
#endif
#ifdef _GRB_WITH_NONBLOCKING
 #include "graphblas/nonblocking/blas2.hpp"
#endif
#ifdef _GRB_WITH_BANSHEE
 #include <graphblas/banshee/blas2.hpp>
#endif
#ifdef _GRB_WITH_LPF
 #include <graphblas/bsp1d/blas2.hpp>
#endif

#endif // end ``_H_GRB_BLAS2''

