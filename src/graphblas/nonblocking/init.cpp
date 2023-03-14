
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
 * Provides initialisation for the nonblocking backend.
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#include <graphblas/reference/init.hpp>
#include <graphblas/nonblocking/init.hpp>

#include <graphblas/utils/alloc.hpp>

#include <graphblas/nonblocking/config.hpp>

#include <sstream>


bool grb::config::ANALYTIC_MODEL::manual_tile_size = false;
size_t grb::config::ANALYTIC_MODEL::manual_fixed_tile_size =
	grb::config::IMPLEMENTATION< nonblocking >::analyticModelMinimumTileSize();
size_t grb::config::ANALYTIC_MODEL::num_threads = grb::config::OMP::threads();

template<>
grb::RC grb::init< grb::nonblocking >(
	const size_t s, const size_t P, void * const data
) {
	// It initializes the maximum number of threads used by the analytic model.
	config::ANALYTIC_MODEL::num_threads = config::OMP::threads();

	// If the environment variable GRB_NONBLOCKING_TILE_SIZE is set, a fixed
	// tile size is used for all pipelines built during the nonblocking execution.
	// Therefore, the choice is manual. Otherwise, the choice is automatically
	// made at run-time by the analytic model and may differ for different
	// pipelines.
	const char *t = getenv( "GRB_NONBLOCKING_TILE_SIZE" );
	std::stringstream cppstr( t );
	if( t != nullptr ) {
		config::ANALYTIC_MODEL::manual_tile_size = true;
		try {
			cppstr >> config::ANALYTIC_MODEL::manual_fixed_tile_size;
		} catch( ... ) {
			std::cerr << "Warning: could not parse contents of the "
				<< "GRB_NONBLOCKING_TILE_SIZE environment variable; ignoring it instead.\n";
			config::ANALYTIC_MODEL::manual_tile_size = false;
		}
	} else {
		config::ANALYTIC_MODEL::manual_tile_size = false;
	}

	std::cerr << "Info: grb::init (nonblocking) called. OpenMP is set to utilise "
		<< config::ANALYTIC_MODEL::num_threads << " threads and the tile size for "
		<< "nonblocking execution is chosen " << (
				config::ANALYTIC_MODEL::manual_tile_size
					? "manually"
					: "automatically.\n"
			);
	if( config::ANALYTIC_MODEL::manual_tile_size ) {
		std::cerr << " and is equal to "
			<< config::ANALYTIC_MODEL::manual_fixed_tile_size << "." << std::endl;
	}

	return grb::init< grb::reference >( s, P, data );
}

template<>
grb::RC grb::finalize< grb::nonblocking >() {
	std::cerr << "Info: grb::finalize (nonblocking) called.\n";
	return grb::finalize< grb::reference >();
}

