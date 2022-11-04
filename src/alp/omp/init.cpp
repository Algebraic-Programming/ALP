
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

#ifndef _GRB_NO_LIBNUMA
 #include <numa.h> //numa_set_localalloc
#endif
#include <omp.h> //omp_get_num_threads


template<>
alp::RC alp::init< alp::omp >( const size_t s, const size_t P, void * const data ) {
	RC rc = alp::SUCCESS;
	// print output
	const auto T = config::OMP::threads();
	std::cerr << "Info: alp::init (alp_omp) called. OpenMP is set to utilise " << T << " threads.\n";

	// use same initialisation procedure as sequential implementation
	rc = alp::init< alp::reference >( s, P, data );
	return rc;
}

template<>
alp::RC alp::finalize< alp::omp >() {
	std::cerr << "Info: alp::finalize (alp_omp) called.\n";
	// use same finalization procedure as sequential implementation
	return alp::finalize< alp::reference >();
}

