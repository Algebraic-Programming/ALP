
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

#ifndef _H_ALP_ASCEND_LAZY_EVALUATION
#define _H_ALP_ASCEND_LAZY_EVALUATION

/**
 * To enable debugging information only for the ascend backend, the code
 * should be combiled with the _ASCEND_DEBUG definition, without defining
 * _DEBUG. If the code is compiled with _DEBUG, the debugging information for
 * the ascend backend is enabled as well.
 */
#if !defined(_ASCEND_DEBUG) && defined(_DEBUG)
 #define _ASCEND_DEBUG
#endif

#include <vector>

#include <graphblas/ascend/pipeline.hpp>
#include <graphblas/ascend/tensor.hpp>
#include <graphblas/ascend/utils.hpp>

namespace alp {

	namespace internal {

		class Stage;

		/**
		 * Encodes a single pipeline that may be expanded, merged, or executed.
		 */
		class AscendLazyEvaluation {

			private:

				size_t num_pipelines;
				std::vector< alp::internal::AscendPipeline > pipelines;


			public:

				AscendLazyEvaluation();
				void addPipeline();
				void insertFreeInputTensorStages( const std::vector< int > &forEachAxes );
				const Tensor &store( const Tensor &output_tensor );
				void clear();
				void addStage( alp::internal::Stagetype op_type, alp::internal::Rule rule, const Tensor &tensor1, const double alpha, const std::vector< int > &forEachAxes );
				void addStage( alp::internal::Stagetype op_type, alp::internal::Rule rule, const Tensor &tensor1, const std::vector< int > &activeAxes, const std::vector< int > &forEachAxes );
				void addStage( alp::internal::Stagetype op_type, alp::internal::Rule rule, const Tensor &tensor1, const Tensor &tensor2, const std::vector< int > &activeAxes, const std::vector< int > &forEachAxes );
				void addStage( alp::internal::Stagetype op_type, alp::internal::Rule rule, const Tensor &tensor1, const Tensor &tensor2, const Tensor &tensor3, const std::vector< int > &activeAxes, const std::vector< int > &forEachAxes );
//				void addStage( alp::internal::Stagetype op_type, alp::internal::Rule rule, const Tensor &tensor1, const Tensor &tensor2, const Tensor &tensor3, const Tensor &tensor4, const std::vector< int > &activeAxes, const std::vector< int > &forEachAxes );
				void generateDeclarations( std::stringstream &declarations );
//				void generateConstructor( std::stringstream &constructor );
				void generateHostBody( std::stringstream &os, std::stringstream &analyticModelArgs, std::stringstream &analyticModelFormalParams, std::stringstream &analyticModelDecls, std::stringstream &analyticModelConstrBody );
				void generateInit( std::stringstream &init );
				void generateProcess( std::stringstream &process, std::stringstream &processCall );

				void debug_print() const;
		};

	}

}

#endif //end `_H_ALP_ASCEND_LAZY_EVALUATION'

