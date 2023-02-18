
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
 * Configurations for the nonblocking backend
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_UTILS_ANALYTIC_MODEL
#define _H_GRB_UTILS_ANALYTIC_MODEL

#include "config.hpp"


namespace grb {

	namespace internal {

		/**
		 * The analytic model used for automatic chunk size selection and for
		 * automatic number of threads selection.
		 */
		class AnalyticModel {

			private:

				/**
				 * The size of the data type of the containers (may vary between different
				 * containers). The current design uses the maximum size of all used data
				 * types.
				 */
				size_t size_of_data_type;

				/**
				 * The size of the containers accessed in the pipeline.
				 */
				size_t size_of_vector;

				/**
				 * The number of vectors accessed in the pipeline.
				 */
				size_t num_accessed_vectors;

				/**
				 * The number of threads selected by the analytic model.
				 */
				size_t num_threads;

				/**
				 * The tile size selected by the analytic model.
				 */
				size_t tile_size;

				/**
				 * The number of total tiles that result from the selected tile size.
				 */
				size_t num_tiles;


			public:

				AnalyticModel();
				AnalyticModel(
					const size_t data_type_size,
					const size_t vector_size,
					const size_t accessed_vectors
				);
				size_t getVectorsSize() const;
				size_t getNumThreads() const;
				size_t getTileSize() const;
				size_t getNumTiles() const;
				void computePerformanceParameters();

		};

	}
}

#endif

