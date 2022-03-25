
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
 * @date 5th of May 2017
 */

#ifndef _H_GRB_PROPERTIES
#define _H_GRB_PROPERTIES

#include "base/config.hpp"
#include "base/properties.hpp"

// now include all specialisations contained in the backend directories:
#ifdef _GRB_WITH_REFERENCE
 #include <graphblas/reference/properties.hpp>
#endif
#ifdef _GRB_WITH_LPF
 #include <graphblas/bsp1d/properties.hpp>
#endif
#ifdef _GRB_WITH_BANSHEE
 #include <graphblas/banshee/properties.hpp>
#endif

// specify default only if requested during compilation
#ifdef _GRB_BACKEND
namespace grb {
	template< enum Backend implementation = config::default_backend >
	class Properties;
}
#endif

#endif // end ``_H_GRB_PROPERTIES''

