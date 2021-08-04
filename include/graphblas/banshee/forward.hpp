
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
 * Contains forward declarations for all the data types GraphBLAS exposes for
 * the #banshee and #banshee_omp backends.
 *
 * Forward declarations are useful to be able to declare friend functions of
 * data types, since friend declarations do not require implementation details.
 *
 * @author A. N. Yzelman
 */

#if ! defined _H_GRB_BANSHEE_FORWARD
#define _H_GRB_BANSHEE_FORWARD

/*
#include "graphblas/backends.hpp"

namespace grb {

    namespace internal {

        template<>
        class Coordinates< banshee >;

        template< typename D, typename IND, typename SIZE >
        class Compressed_Storage;

    }

    template< typename D, typename C >
    class Vector< D, banshee, C >;

    template< typename D >
    class Matrix< D, banshee >;

    template< typename D >
    class PinnedVector< D, banshee >;

    template<>
    class collectives< banshee >;

    template<>
    class spmd< banshee >;

    template<>
    class Properties< banshee >;

    namespace internal {
        template< typename InputType >
        size_t& getNonzeroCapacity( grb::Matrix< InputType, banshee >& ) noexcept;
        template< typename InputType >
        size_t& getCurrentNonzeroes( grb::Matrix< InputType, banshee >& ) noexcept;
        template< typename InputType >
        void setCurrentNonzeroes( grb::Matrix< InputType, banshee > &, const size_t ) noexcept;
    }
}
*/

#endif
