
/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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

#include <vector>
#include <memory>
#include <cstddef>

#ifndef _H_GRB_ALGORITHMS_MULTIGRID_BUILDING_UTILS
#define _H_GRB_ALGORITHMS_MULTIGRID_BUILDING_UTILS

namespace grb {
	namespace algorithms {

		template<
			typename MGInfoType,
			typename CoarsenerInfoType,
			typename SmootherInfoType
		> void multigrid_allocate_data(
			const std::vector< size_t > &mg_sizes,
			std::vector< std::unique_ptr< MGInfoType > > &system_levels,
			std::vector< std::unique_ptr< CoarsenerInfoType > > &coarsener_levels,
			std::vector< std::unique_ptr< SmootherInfoType > > &smoother_levels
		) {
			if( mg_sizes.size() == 0 ) {
				throw std::invalid_argument( "at least one size should be available" );
			}
			size_t finer_size = mg_sizes[ 0 ];
			system_levels.emplace_back( new MGInfoType( 0, finer_size ) ); // create main system
			smoother_levels.emplace_back( new SmootherInfoType( finer_size ) ); // create smoother for main
			for( size_t i = 1; i < mg_sizes.size(); i++ ) {
				size_t coarser_size = mg_sizes[ i ];
				coarsener_levels.emplace_back( new CoarsenerInfoType( finer_size, coarser_size ) );
				system_levels.emplace_back( new MGInfoType( i, coarser_size ) );
				smoother_levels.emplace_back( new SmootherInfoType( coarser_size ) );
				finer_size = coarser_size;
			}
		}

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_BUILDING_UTILS
