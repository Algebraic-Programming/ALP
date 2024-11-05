
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

#ifndef _H_GRB_ASCEND_SYMBOLTABLE
#define _H_GRB_ASCEND_SYMBOLTABLE

#include <string>
#include <map>
#include <utility>
#include <vector>
#include <unordered_set>

#include <graphblas.hpp>
#include <graphblas/ascend/utils.hpp>

namespace alp {

	class Tensor;

	namespace internal {

		class SymbolTable {

			private:

				bool TBuf_decl;

				/** Maintains a counter for unique temporary scalar names. */
				size_t temp_scalar_id;

				// pointers to Tensors do not work because any local declaration
				// inside the forEach will be invalid the moment the code is generated

				/** Maintains all the global declarations of the compiled function. */
				std::map< std::string, alp::Tensor > global_tensor_declarations;

				/** Maintains all the local declarations of the current forEach. */
				std::map< std::string, alp::Tensor > local_tensor_declarations;

				/** Maintains all the temporary declarations of the current forEach. */
				std::map< std::string, alp::Tensor > temp_tensor_declarations;

				/** Maintains all the buffers that are reused for local
				 *  and temporary declarations of the current forEach.
				 */
				std::map< std::string, std::string > temp_local_buffer_declarations;

				/** Maintains the order of all the global tensors and only the output tensors, respectively */
				std::vector< alp::Tensor > all_global_tensors;
				std::vector< alp::Tensor > outputs_global_tensors;

				/**
				 * Maintains a mapping from chunks to vectors.
				 *
				 * \warning The map does not guarantee that chunks who have since been
				 *          destructed will no longer appear in the map.
				 */
				std::map< std::string, std::string > viewToTensor;


			public:

				SymbolTable();
				bool existsTBufTensorDecl() const;
				void clearAll();

				void addGlobalTensor( const alp::Tensor &t );
				void addLocalTensor( const alp::Tensor &t );
				void addTempTensor( const alp::Tensor &t );
				void addTensorView( const std::string &view_name, const std::string &parent_name );
//				std::string newTempScalar();
				void addOutputTensor( const alp::Tensor &t );
				void printHostLogFile( std::stringstream &listOfGlobalTensors );
				std::string getLocalTempTensorBuffer( Datatype type, const std::string &size = "" );
				void generateGlobalSymbols( std::stringstream &initFormalParam,
											std::stringstream &customFormalParam,
											std::stringstream &allAccessedArg,
											std::stringstream &allTempLocalDecl ) const;
				void generateTempLocalInit( std::stringstream &allTempLocalInit ) const;
				const alp::Tensor &getTensorFromView( const alp::Tensor &tensor ) const;

				void debug_print() const;

			private:


				void reuseLocalTempTensorBuffer( const alp::Tensor &t );
		};

	}

}

#endif //end `_H_GRB_ASCEND_SYMBOLTABLE'

