
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

#include <graphblas/reference/init.hpp>

#include <graphblas/utils/alloc.hpp>

#ifndef _GRB_NO_LIBNUMA
 #include <numa.h> //numa_set_localalloc
#endif
#ifdef _GRB_WITH_OMP
 #include <omp.h> //omp_get_num_threads
#endif


size_t * __restrict__ grb::internal::privateSizetOMP = nullptr;

grb::utils::DMapper< uintptr_t > grb::internal::reference_mapper;

char * grb::internal::reference_buffer = nullptr;

size_t grb::internal::reference_bufsize = 0;

static grb::utils::AutoDeleter< size_t > privateSizetOMP_deleter;

template<>
grb::RC grb::init< grb::reference >(
	const size_t s, const size_t P,
	void * const data
) {
	// we don't use any implementation-specific init data
	(void)data;
	// print output
	std::cerr << "Info: grb::init (reference) called.\n";
	// sanity checks
	if( P > 1 ) {
		return grb::UNSUPPORTED;
	}
	if( s > 0 ) {
		return grb::PANIC;
	}
#ifndef _GRB_NO_LIBNUMA
	// set memory policy
	numa_set_localalloc();
#endif
	// done
	return grb::SUCCESS;
}

template<>
grb::RC grb::finalize< grb::reference >() {
	std::cerr << "Info: grb::finalize (reference) called.\n";
	if( internal::reference_bufsize > 0 ) {
		delete[] internal::reference_buffer;
		internal::reference_bufsize = 0;
	}
	return grb::SUCCESS;
}

#ifdef _GRB_WITH_OMP
template<>
grb::RC grb::init< grb::reference_omp >( const size_t s, const size_t P, void * const data ) {
	RC rc = grb::SUCCESS;
	// print output
	const auto T = config::OMP::threads();
	std::cerr << "Info: grb::init (reference_omp) called. OpenMP is set to "
		<< "utilise " << T << " threads.\n";
	rc = grb::utils::alloc(
		"",
		"",
		grb::internal::privateSizetOMP,
		T * sizeof( grb::config::CACHE_LINE_SIZE::value() ) * sizeof( size_t ),
		true,
		privateSizetOMP_deleter
	);
	// use same initialisation procedure as sequential implementation
	if( rc == grb::SUCCESS ) {
		rc = grb::init< grb::reference >( s, P, data );
	}
	// pre-reserve a minimum buffer equal to the combined L1 cache size
	if( rc == grb::SUCCESS ) {
		if( !internal::template ensureReferenceBufsize< char >(
			T * config::MEMORY::l1_cache_size() )
		) {
			rc = grb::OUTOFMEM;
		}
	}
	return rc;
}
#endif

template<>
grb::RC grb::finalize< grb::reference_omp >() {
	std::cerr << "Info: grb::finalize (reference_omp) called.\n";
	// use same finalization procedure as sequential implementation
	return grb::finalize< grb::reference >();
}

