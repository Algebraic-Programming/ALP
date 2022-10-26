
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

#ifndef _H_ALP_SCALAR
#define _H_ALP_SCALAR

#include "base/config.hpp"
#include "base/scalar.hpp"

#include <alp/structures.hpp>

// now include all specialisations contained in the backend directories:
#ifdef _ALP_WITH_REFERENCE
 #include <alp/reference/scalar.hpp>
#endif
#ifdef _ALP_WITH_DISPATCH
 #include <alp/dispatch/scalar.hpp>
#endif
#ifdef _ALP_WITH_OMP
 #include <alp/omp/scalar.hpp>
#endif

// specify default only if requested during compilation
#ifdef _ALP_BACKEND
namespace alp {

	template< typename T, typename Structure = structures::General, enum Backend backend = config::default_backend >
	class Scalar;

	/** Specializations of ALP backend-agnostic type traits */
	template< typename T, typename Structure, enum Backend backend >
	struct inspect_structure< Scalar< T, Structure, backend > > {
		typedef Structure type;
	};

}
#endif

#endif // end ``_H_ALP_SCALAR''

