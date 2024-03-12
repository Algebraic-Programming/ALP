
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
 * Provides initialisation for the Ascend backend.
 *
 * @author A. N. Yzelman
 * @date 12th of September, 2023
 */

#include <graphblas/reference/init.hpp>

#include <graphblas/nonblocking/config.hpp>

#include <graphblas/ascend/init.hpp>
#include <graphblas/ascend/opgen.hpp>

#include <graphblas/utils/alloc.hpp>

#include <sstream>


template<>
grb::RC grb::init< grb::ascend >(
	const size_t s, const size_t P, void * const data
) {
	// If the environment variable GRB_ASCEND_TILE_SIZE is set, a fixed
	// tile size is used for all pipelines built during the ascend execution.
	// Therefore, the choice is manual. Otherwise, the choice is automatically
	// made at run-time by the analytic model and may differ for different
	// pipelines.
	std::cerr << "Info: grb::init (ascend) called.\n";
	return grb::init< grb::reference >( s, P, data );
}

template<>
grb::RC grb::finalize< grb::ascend >() {
	std::cerr << "Info: grb::finalize (ascend) called.\n";
	std::cerr << "Info: codegen will go to std::cout (TODO)\n";
//	alp::internal::OpGen::generate( std::cout );
	return grb::finalize< grb::reference >();
}

