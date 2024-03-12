
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
 * Class that defines the state of the code generation.
 *
 * @author A. N. Yzelman.
 * @date 12th of September, 2023.
 */


#ifndef _H_ALP_ASCEND_OPGEN
#define _H_ALP_ASCEND_OPGEN

#include <map>
#include <utility>
#include <vector>
#include <string>
#include <sstream>

#include <cxxabi.h>

#include <graphblas.hpp>

namespace alp {

	namespace internal {

		class OpGen {

			public:

				OpGen() = default;
				virtual ~OpGen() = default;

				/** Returns a string representation of a given type. */
/*				template<typename T>
				static std::string type_name(){
					int info = 0;
					return abi::__cxa_demangle( typeid(T).name(), NULL, NULL, &info );
				}
*/
				/**
				 * Maintains a mapping from chunks to their sizes.
				 *
				 * \warning The map does not guarantee that chunks who have since been
				 *          destructed will no longer appear in the map.
				 */
//TODO how is this supposed to be used?
//				static std::map< std::string, std::string > chunkSize;

				static std::string kernel_id;

				/** Indicates if the executed code is within the lambda function of a forEach */
				static size_t forEachLevel;

				static std::vector< std::vector< int > > forEachAxes;
				static std::vector< int > lastAxes;

				static std::stringstream aux_func;
				static std::stringstream analyticModelFormalParams;
				static std::stringstream hostFormalParam;
				static std::stringstream hostBody;
				static std::stringstream hostArg;
				static std::stringstream constrBody;
				static std::stringstream classMembers;
				static std::stringstream initBody;
				static std::stringstream genericProcessBody;
				static std::stringstream declarations;

				static std::vector< std::stringstream > processFunc;
				static std::vector< std::stringstream > computeFunc;
				static std::vector< std::stringstream > copyinFunc;
				static std::vector< std::stringstream > copyoutFunc;

				static void compileClear();
				static void generate( std::ostream &os );
		};
	}

}

#endif // end _H_ALP_ASCEND_OPGEN

