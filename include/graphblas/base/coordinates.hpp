
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
 * @author Alberto Scolari
 */

#ifndef _H_GRB_COORDINATES_BASE
#define _H_GRB_COORDINATES_BASE

#include <graphblas/backends.hpp>

#include "config.hpp"

namespace grb {

	namespace internal {

		template< enum Backend implementation >
		class Coordinates;

#ifndef _GRB_COORDINATES_BACKEND
		typedef Coordinates< config::default_backend > DefaultCoordinates;
#else
		typedef Coordinates< _GRB_COORDINATES_BACKEND > DefaultCoordinates;
#endif
	} // namespace internal
} // namespace grb

#endif // _H_GRB_COORDINATES_BASE
