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
 * @author A. N. Yzelman
 * @date 12th of September, 2023
 */

#ifndef _H_ALP_ASCEND_GRID
#define _H_ALP_ASCEND_GRID

#include <functional>
#include <vector>

#include <graphblas.hpp>
#include <graphblas/ascend/opgen.hpp>
#include <graphblas/ascend/semantics.hpp>
#include <graphblas/ascend/lazy_evaluation.hpp>

namespace alp
{
	namespace internal
	{
		extern AscendLazyEvaluation ale;
	}
}

namespace alp {

	namespace internal {

		class iGrid {

			private:
				size_t process_order;
				size_t problem_order;

			public:
				iGrid( size_t proc, size_t prob );
				size_t getProcessOrder() const noexcept;
				size_t getProblemOrder() const noexcept;
				std::string processSize( const size_t k ) const noexcept;
				std::string processMode( const size_t k ) const noexcept;
				std::string problemSize( const size_t k ) const noexcept;
				std::string problemMode( const size_t k ) const noexcept;
				std::string problemMainMode( const size_t k ) const noexcept;
				std::string problemTileMode( const size_t k ) const noexcept;
				std::string tileSize( const size_t k ) const noexcept;
		};

	}

	/**
	 * Specific to the ALP/Ascend backend, this class maps problem spaces on
	 * process grids in a symbolic fashion.
	 */
	template< size_t process_order, size_t problem_order >
	class Grid {

		private:
			// problem mesh related state:
//			std::vector< std::string > problem_sizes, problem_space, chunk_sizes;

		public:
			Grid() noexcept;
			std::string processSize( const size_t k ) const noexcept;
			std::string processMode( const size_t k ) const noexcept;
			std::string problemSize( const size_t k ) const noexcept;
			std::string problemMode( const size_t k ) const noexcept;
			std::string problemMainMode( const size_t k ) const noexcept;
			std::string problemTileMode( const size_t k ) const noexcept;
			std::string tileSize( const size_t k ) const noexcept;
//			std::string chunkSize( const size_t k ) const noexcept;
			grb::RC forEach( const std::vector< int > axes, const std::function < void( void ) > func ) const;
	};

	template< size_t process_order, size_t problem_order >
	Grid< process_order, problem_order >::Grid() noexcept
	{
//		for( size_t k = 0; k < problem_order; ++k ) {
//			   chunk_sizes.push_back( "BLOCK_LENGTH" + k );
//		}
	}

	template< size_t process_order, size_t problem_order >
	std::string Grid< process_order, problem_order >::processSize( const size_t k ) const noexcept {
		return "p" + std::to_string( k );
	}

	template< size_t process_order, size_t problem_order >
	std::string Grid< process_order, problem_order >::processMode( const size_t k ) const noexcept {
		return "a" + std::to_string( k );
	}

	template< size_t process_order, size_t problem_order >
	std::string Grid< process_order, problem_order >::problemSize( const size_t k ) const noexcept {
		return "n" + std::to_string( k );
	}

	template< size_t process_order, size_t problem_order >
	std::string Grid< process_order, problem_order >::problemMode( const size_t k ) const noexcept {
		return "i" + std::to_string( k );
	}

	template< size_t process_order, size_t problem_order >
	std::string Grid< process_order, problem_order >::problemMainMode( const size_t k ) const noexcept {
		return "z" + std::to_string( k );
	}

	template< size_t process_order, size_t problem_order >
	std::string Grid< process_order, problem_order >::problemTileMode( const size_t k ) const noexcept {
		return "t" + std::to_string( k );
	}

	template< size_t process_order, size_t problem_order >
	std::string Grid< process_order, problem_order >::tileSize( const size_t k ) const noexcept {
		return "tile_size" + std::to_string( k );
	}

/*	template< size_t process_order, size_t problem_order >
	std::string Grid< process_order, problem_order >::chunkSize( const size_t k ) const noexcept {
		return chunk_sizes[ k ];
	}
*/
	template< size_t process_order, size_t problem_order >
	grb::RC Grid< process_order, problem_order >::forEach( const std::vector< int > axes, const std::function < void( void ) > func ) const {

		alp::internal::OpGen gen();

		if( internal::OpGen::lastAxes.size() > 0 && internal::OpGen::lastAxes != axes ) {
			alp::internal::ale.addPipeline();
		}

		if( internal::invalidForEachAxes( axes ) == true ) {
			std::cerr << "The axes of a nested forEach cannot overlap with the axes of another forEach." << std::endl;
			std::abort();
		}

		internal::OpGen::forEachAxes.push_back( axes );

		// indicate the beginning of the forEach
		internal::OpGen::forEachLevel++;

		// TODO: this is currently used only by the Tensor class in the getView method
		//		 perhaps the getView should be a method of the Grid class
//		internal::OpGen::parallelAxes = axes;

		// the current design assumes that each forEach is a new pipeline
		// which is explicitly added here, later we need to figure out
		// how we determine the creation of a pipeline
//		alp::internal::ale.addPipeline( axes );

		// TODO: emit for-loop intro
		func();
		// TODO: emit for-loop outro

		// before leaving a forEach loop, any getView of an input Tensor
		// should match with an implicit Stage for freeing any allocated memory
		alp::internal::ale.insertFreeInputTensorStages( internal::vectorOfVectorsToVector( internal::OpGen::forEachAxes ) );

		// indicate the end of the forEach
		internal::OpGen::forEachLevel--;

		internal::OpGen::forEachAxes.pop_back();

		internal::OpGen::lastAxes = axes;

		return grb::SUCCESS;
	}
}

#endif

