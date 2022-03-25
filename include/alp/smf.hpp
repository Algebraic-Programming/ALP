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
 * This file registers available storage mapping functions (SMFs).
 * SMFs are maps between logical and physical storage space.
 *
 */

#ifndef _H_GRB_SMF
#define _H_GRB_SMF

#include <memory>

namespace grb {

	namespace smf {

        class SMF {
            public:
                size_t n, N;

                SMF(size_t n, size_t N): n(n), N(N) {}

                /** Maps logical to physical coordinate
                 * */
                virtual size_t map(size_t i) = 0;

                /** Returns the physical dimension of the container needed to
                 * store all elements
                 * */
                virtual size_t allocSize() const = 0;

        };

	}; // namespace smf

} // namespace grb

#endif // _H_GRB_SMF
