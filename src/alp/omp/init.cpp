
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
 * @date 2nd of February, 2017
 */

#include <alp/reference/init.hpp>
#include <alp/omp/config.hpp>
#include <alp/rc.hpp>

#ifndef _GRB_NO_LIBNUMA
 #include <numa.h> //numa_set_localalloc
#endif
#include <omp.h> //omp_get_num_threads


template<>
alp::RC alp::init< alp::omp >( const size_t s, const size_t P, void * const data ) {
	(void) data;
	RC rc = alp::SUCCESS;
	// print output
	const auto T = config::OMP::threads();
	std::cout << "Info: alp::init (omp) called. OpenMP is set to utilise " << T << " threads.\n";

	// sanity checks
	if( P > 1 ) {
		return alp::UNSUPPORTED;
	}
	if( s > 0 ) {
		return alp::PANIC;
	}
#ifndef _GRB_NO_LIBNUMA
	// set memory policy
	numa_set_localalloc();
#endif
	return rc;
}

template<>
alp::RC alp::finalize< alp::omp >() {
	std::cout << "Info: alp::finalize (omp) called.\n";
	return alp::SUCCESS;
}

