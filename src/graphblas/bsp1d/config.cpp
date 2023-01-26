
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
 * @date 13th of September 2017
 */

#include <graphblas/bsp1d/config.hpp>


using namespace grb;

// whether mode is initialised-- false, on program start
bool config::IMPLEMENTATION< Backend::BSP1D >::set = false;

// keep uninitialised, the below function performs initialisation
config::ALLOC_MODE config::IMPLEMENTATION< Backend::BSP1D >::mode;

// sets the mode
void config::IMPLEMENTATION< BSP1D >::deduce() noexcept {
	assert( !set );
	// by default, select the one our backend is using
	mode = config::IMPLEMENTATION< _GRB_BSP1D_BACKEND >::sharedAllocMode();
	// get number of hardware threads
	const int maxthreads = sysconf( _SC_NPROCESSORS_ONLN );
	// try to get process mask
	cpu_set_t cpuset;
	if( sched_getaffinity( 0, sizeof( cpu_set_t ), &cpuset ) != 0 ) {
		std::cerr << "Info: could not get process mask. Will fall back to "
			<< toString( mode ) << " mode for shared memory segment allocations.";
	} else {
		// check if we have all-1 mask
		bool allOne = true;
		for( int i = 0; i < maxthreads; ++i ) {
			if( CPU_ISSET( i, &cpuset ) == 0 ) {
				// we have a zero, therefore assume we are with multiple processes on a
				// single node
				allOne = false;
				// we shall now assume a default allocation is more appropriate
				mode = config::IMPLEMENTATION< _GRB_BSP1D_BACKEND >::defaultAllocMode();
				std::cerr << "Info: process mask is zero at HW thread " << i << ", "
					<< ", therefore ALP assumes multiple user processes are present on this "
					<< "node and thus fall back to " << toString( mode ) << " mode for "
					<< "memory allocations that are potentially touched by multiple "
					<< "threads.\n";
				break;
			}
		}
		if( allOne ) {
			// we are running one process on this node. We shall therefore use the
			// default allocation scheme.
			std::cerr << "Info: process mask is all-one, we therefore assume a single "
				<< "user process is present on this node and thus shall use "
				<< toString( mode ) << " mode for memory allocations that are potentially "
				<< "touched by multiple threads.\n";
		}
	}
	// done
	set = true;
}

config::ALLOC_MODE config::IMPLEMENTATION< BSP1D >::sharedAllocMode() noexcept {
	if( !set ) {
		deduce();
	}
	assert( set );
	return mode;
}

