
/*
 *   Copyright 2024 Huawei Technologies Co., Ltd.
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
 * \ingroup TRANS_FUSELETS
 *
 * Implements a set of fused level-1 and level-2 ALP kernels.
 *
 * The fused kernels are designed to be easily callable from existing code
 * bases, using standard data structures such as raw pointers to vectors and the
 * Compressed Row Storage for sparse matrices.
 *
 * As a secondary goal, this standard set of fuselets demos the so-called ALP
 * <em>native interface</em>, show-casing the ease by which additional fuselets
 * that satisfy arbitrary needs can be added.
 *
 * @author A. N. Yzelman
 * @date 27/09/2024
 */

#include <graphblas.hpp>

#include <assert.h>

#include "fuselets.h"



