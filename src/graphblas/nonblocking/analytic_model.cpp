
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
 * Implements the analytic model used during nonblocking execution.
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#include <graphblas/nonblocking/init.hpp>
#include <graphblas/nonblocking/analytic_model.hpp>


using namespace grb::internal;

AnalyticModel::AnalyticModel() noexcept {}

AnalyticModel::AnalyticModel(
	const size_t data_type_size,
	const size_t vector_size,
	const size_t accessed_vectors
) noexcept :
	size_of_data_type( data_type_size ),
	size_of_vector( vector_size ),
	num_accessed_vectors( accessed_vectors )
{
	size_t tile_size_estimation;

	num_threads = grb::internal::NONBLOCKING::numThreads();

	const bool manual_choice =
		grb::internal::NONBLOCKING::isManualTileSize();

	if ( !manual_choice ) {
		// It automatically determines the tile size and the number of threads.
		// A different tile size and number of threads may be used for the
		// execution of different pipelines.
		constexpr size_t l1_cache_size = grb::config::MEMORY::l1_cache_size();
		constexpr size_t min_tile_size =
			grb::config::ANALYTIC_MODEL::MIN_TILE_SIZE;
		constexpr double l1_cache_usage_percentage =
			grb::config::ANALYTIC_MODEL::L1_CACHE_USAGE_PERCENTAGE;

		// A tile size estimation based on the data that fit in L1 cache.
		const size_t cache_based_tile_size =
			( l1_cache_size / size_of_data_type ) / ( num_accessed_vectors );
		// A tile size estimation based on the number of cores that can be utilized.
		const size_t cores_based_tile_size = size_of_vector / num_threads;

		// It selects the minimum tile size between the two estimations.
		tile_size_estimation = ( cache_based_tile_size < cores_based_tile_size )
			? cache_based_tile_size
			: cores_based_tile_size;

		// It assumes that the L1 cache may contain other data and takes the size of
		// these data into account.
		tile_size_estimation *= l1_cache_usage_percentage;
		// It ensures that the tile size is sufficiently large to successfully
		// amortize the runtime overhead.
		tile_size_estimation = ( min_tile_size > tile_size_estimation )
			? min_tile_size
			: tile_size_estimation;
	} else {
		// It determines the tile size and the number of threads manually.
		// A fixed tile size and number of threads is used for the execution of all
		// pipelines.
		tile_size_estimation =
			grb::internal::NONBLOCKING::manualFixedTileSize();
		std::cout << "tile prestimation: " << tile_size_estimation << std::endl;
	}

	// It ensures that the tile size does not exceed the size of vectors.
	if( tile_size_estimation > size_of_vector ) {
		tile_size = size_of_vector;
	} else {
		tile_size = tile_size_estimation;
	}

	// It computes the total number of tiles.
	num_tiles = ( size_of_vector + tile_size - 1 ) / tile_size;

	if ( !manual_choice ) {
		// It adjusts the number of threads when there are not enough tiles to
		// utilize all cores.
		num_threads = ( num_threads < num_tiles ) ? num_threads : num_tiles;
	}
}

size_t AnalyticModel::getVectorsSize() const noexcept {
	return size_of_vector;
}

size_t AnalyticModel::getNumThreads() const noexcept {
	return num_threads;
}

size_t AnalyticModel::getTileSize() const noexcept {
	return tile_size;
}

size_t AnalyticModel::getNumTiles() const noexcept {
	return num_tiles;
}

