
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


#ifndef _H_ALP_OMP_CONFIG
#define _H_ALP_OMP_CONFIG

#include <graphblas/omp/config.hpp>

#include <alp/base/config.hpp>


namespace alp {

	namespace config {

		/** The default sequential backend to be selected for this parallel backend. */
		static constexpr alp::Backend default_sequential_backend = _ALP_SECONDARY_BACKEND;

		class OMP : public grb::config::OMP {};

		// Dimensions of blocks counted in number of elements per dimension
		constexpr size_t BLOCK_ROW_DIM = 16;
		constexpr size_t BLOCK_COL_DIM = 16;

		// Temporary solution to accomodate for 2.5D mxm
		constexpr size_t THREAD_ROW_DIM = 4;
		constexpr size_t THREAD_COL_DIM = 4;

		constexpr size_t REPLICATION_FACTOR_THREADS = 2;


	} // namespace config

} // namespace alp

#endif

