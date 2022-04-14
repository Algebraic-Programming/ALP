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
 *
 * @file 
 * 
 * This file registers available index mapping functions (IMFs).
 * IMFs are maps between integer intervals and can be used to define
 * affine \em access transformations in the form of access matrices.
 * For example, an access matrix \f$G_f\in R^{N\times N}\f$ 
 * parametrized by the IMF \f$f\f$ such that
 * \f[G_f = \sum_{i=0}^{n-1} e_i^n\left(e_{f(i)}^N\right)^T\f]
 * could be used to access a group of $n\eN$ rows of matrix 
 * \f$A\in R^{N\times N}\f$
 * according to \f$f\f$ by multiplying \f$A\f$ by \f$G_f\f$ from the left:
 * \f[\tilde{A} = G_f\cdot A,\quad \tilde{A}\in R^{n\times N}\f]
 *      
 * \note The idea of parametrized matrices to express matrix accesses at 
 *       a higher level of mathematical abstractions is inspired by the 
 *       SPIRAL literature (Franchetti et al. SPIRAL: Extreme Performance Portability. 
 *       http://spiral.net/doc/papers/08510983_Spiral_IEEE_Final.pdf). 
 *       Similar affine formulations are also used in the polyhedral 
 *       compilation literature to express concepts such as access
 *       relations.
 *       In this draft we use integer maps. A symbolic version of them could be 
 *       defined using external libraries such as the Integer Set Library (isl 
 *       \link https://libisl.sourceforge.io/).
 *       
 */

#ifndef _H_ALP_IMF
#define _H_ALP_IMF

#include <memory>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include <alp/imf_static.hpp>

#endif // _H_ALP_IMF
