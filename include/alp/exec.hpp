
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
 * @author: A. N. Yzelman
 * @date 17th of April, 2017
 */

#ifndef _H_ALP_EXEC
#define _H_ALP_EXEC

#include "base/config.hpp"
#include "base/exec.hpp"

// include template specialisations
#ifdef _ALP_WITH_REFERENCE
 #include "alp/reference/exec.hpp"
#endif
#ifdef _ALP_WITH_OMP
 #include "alp/omp/exec.hpp"
#endif

#ifdef _ALP_BACKEND
namespace alp {
	template< enum EXEC_MODE mode, enum Backend implementation = config::default_backend >
	class Launcher;
}
#endif

#endif // end ``_H_ALP_EXEC''

