
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

#include <stdio.h>

#include "graphblas/utils/Timer.hpp"

#include "graphblas.hpp"


using namespace grb;

constexpr size_t n = 100000;
constexpr size_t rep = 100;

void grbProgram( const size_t &P, int &exit_status ) {
	(void)P;
	assert( exit_status == 0 );

#ifdef _DEBUG
	std::cerr << "Info: grbProgram (reduce) entered with parameters: "
		<< P << ", " << exit_status << "\n";
#endif

	grb::utils::Timer benchtimer;
	benchtimer.reset();

	grb::Monoid< grb::operators::add< double >, grb::identities::zero > realm;

	grb::Vector< double > xv( n );
	double check = 0.0;
	double * __restrict__ xr = nullptr;

	int rc = posix_memalign( (void **)&xr,
		grb::config::CACHE_LINE_SIZE::value(),
		n * sizeof( double )
	);
	if( rc != 0 ) {
#ifndef NDEBUG
		const bool posix_memalign_failed = false;
		assert( posix_memalign_failed );
#endif
		exit_status = 10;
		return;
	}

	if( grb::set< grb::descriptors::use_index >( xv, 0 ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool setToIndex_failed = false;
		assert( setToIndex_failed );
#endif
		exit_status = 20;
		return;
	}
	if( grb::nnz( xv ) != n ) {
#ifndef NDEBUG
		const bool setToIndex_verify_nnz_failed = false;
		assert( setToIndex_verify_nnz_failed );
#endif
		exit_status = 25;
		return;
	}
	for( const auto &pair : xv ) {
		const size_t i = pair.first;
		if( pair.second != i ) {
#ifndef NDEBUG
			const bool setToIndex_verification_failed = false;
			assert( setToIndex_verification_failed );
#endif
			exit_status = 30;
			return;
		}
	}
	for( size_t i = 0; i < n; ++ i ) {
		xr[ i ] = (double)i;
		check += (double)i;
	}

#ifdef _DEBUG
	std::cerr << "Info: grbProgram (reduce) vector allocs of size " << n << ". "
		<< "Initialisation complete." << std::endl;
#endif

	double alpha = 0.0;
	if( grb::foldl( alpha, xv, NO_MASK, realm ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool foldl_into_scalar_failed = false;
		assert( foldl_into_scalar_failed );
#endif
		exit_status = 40;
		return;
	}

	double alpha_unmasked = 0.0;
	if( grb::foldl( alpha_unmasked, xv, realm ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool foldl_into_scalar_unmasked_failed = false;
		assert( foldl_into_scalar_unmasked_failed );
#endif
		exit_status = 45;
		return;
	}

#ifdef _DEBUG
	std::cerr << "Info: grbProgram (reduce) post foldl" << std::endl;
#endif

	/*double alpha_right = 0.0;
	if( grb::foldr( xv, NO_MASK, alpha_right, realm ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool foldr_into_scalar_failed = false;
		assert( foldr_into_scalar_failed );
#endif
		exit_status = 50;
		return;
	} TODO: internal issue #311 */

	double alpha_right_unmasked = 0.0;
	if( grb::foldr( xv, alpha_right_unmasked, realm ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool foldr_into_scalar_unmasked_failed = false;
		assert( foldr_into_scalar_unmasked_failed );
#endif
		exit_status = 60;
		return;
	}

#ifdef _DEBUG
	std::cerr << "Info: grbProgram (reduce) post foldr" << std::endl;
#endif

	double beta = 0.0;
	for( size_t i = 0; i < n; ++i ) {
		beta += xr[ i ];
	}

	// verify computations
	bool error = false;
	if( !grb::utils::equals( alpha, alpha_unmasked, n-1 ) ) {
		std::cerr << "Error: " << alpha_unmasked << " (foldl, unmasked) "
			<< " does not equal " << alpha << " (foldl, masked).\n";
		error = true;
	}
	/*if( !grb::utils::equals( alpha, alpha_right, n-1 ) ) {
		std::cerr << "Error: " << alpha_right << " (foldr, masked) "
			<< " does not equal " << alpha << " (foldl, masked).\n";
		error = true;
	} TODO internal issue #311 */
	if( !grb::utils::equals( alpha, alpha_right_unmasked, n-1 ) ) {
		std::cerr << "Error: " << alpha_right_unmasked << " (foldr, unmasked) "
			<< " does not equal " << alpha << " (foldl, masked).\n";
		error = true;
	}
	if( !grb::utils::equals( check, alpha, n-1 ) ) {
		std::cerr << "Error: " << alpha << " (ALP) does not equal "
			<< check << ".\n";
		error = true;
	}
	if( !grb::utils::equals( check, beta, n ) ) {
		std::cerr << "Error: " << beta << " (compiler) does not equal "
			<< check << ".\n";
		error = true;
	}
	if( !grb::utils::equals( alpha, beta, n ) ) {
		std::cerr << "Error: " << alpha << " (ALP) does not equal "
			<< beta << " (compiler).\n";
		error = true;
	}

	if( !error ) {
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "Functional tests complete.\n\n"
				<< "Now starting benchmark run 1 (ALP foldl):" << std::endl;
		}
	} else {
		std::cerr << std::flush;
		exit_status = 70;
		return;
	}

	// first do a cold run
	if( grb::foldl( alpha, xv, realm ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool cold_foldl_into_scalar_failed = false;
		assert( cold_foldl_into_scalar_failed );
#endif
		exit_status = 80;
		return;
	}

	double ttime = 0.0;
	// now benchmark hot runs
	grb::utils::Timer timer;
	for( size_t i = 0; i < rep; ++i ) {
		alpha = realm.template getIdentity< double >();
		timer.reset();
		grb::RC looped_rc = grb::foldl( alpha, xv, realm );
		ttime += timer.time() / static_cast< double >( rep );
		if( looped_rc != grb::SUCCESS ) {
			std::cerr << "Error: foldl into scalar during hot loop failed.\n";
			error = true;
		}
		if( !grb::utils::equals( check, alpha, n-1 ) ) {
			std::cerr << "Error: " << alpha << " (ALP foldl, re-entrant) "
				<< "does not equal " << check << " (sequential).\n",
			error = true;
		}
	}
	if( grb::collectives<>::reduce( ttime, 0, grb::operators::max< double >() ) != grb::SUCCESS ) {
		std::cerr << "Error: reduction of ALP reduction time failed.\n";
		exit_status = 85;
		return;
	}
	if( grb::spmd<>::pid() == 0 ) {
		std::cout << "\t average time taken for ALP reduce by foldl: " << ttime << "." << std::endl;
	}

	if( !error ) {
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "\t benchmark run 1 complete and verified.\n\n"
				<< "Now starting benchmark run 2 (compiler):" << std::endl;
		}
	} else {
		std::cerr << std::flush;
		exit_status = 90;
		return;
	}

	// first do a cold run
	alpha = xr[ 0 ];
	for( size_t i = 1; i < n; ++i ) {
		alpha += xr[ i ];
	}
	// now benchmark hot runs
	double ctime = 0.0;
	for( size_t k = 0; k < rep; ++k ) {
		timer.reset();
		alpha = xr[ 0 ];
		for( size_t i = 1; i < n; ++i ) {
			alpha += xr[ i ];
		}
		ctime += timer.time() / static_cast< double >( rep );
		if( !grb::utils::equals( check, alpha, n-1 ) ) {
			std::cerr << "Error: " << alpha << " (compiler, re-entrant) "
				<< "does not equal " << check << " (sequential).\n";
			error = true;
		}
	}
	free( xr );
	if( grb::collectives<>::reduce( ctime, 0, grb::operators::add< double >() ) != grb::SUCCESS ) {
		std::cerr << "Error: reduction of compiler timings failed.\n";
		exit_status = 95;
		return;
	}
	if( grb::spmd<>::pid() == 0 ) {
		ctime /= static_cast< double >( grb::spmd<>::nprocs() );
		std::cout << "\t average time taken for compiler-optimised reduce: "
			<< ctime << "." << std::endl;
	}

	if( !error ) {
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "\t benchmark run 2 complete and verified.\n\n"
				<< "Now starting benchmark run 3 (ALP foldr):" << std::endl;
		}
	} else {
		std::cerr << std::flush;
		exit_status = 100;
		return;
	}

	// first do a cold run
	if( grb::foldr( xv, alpha_right_unmasked, realm ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool cold_foldr_into_scalar_failed = false;
		assert( cold_foldr_into_scalar_failed );
#endif
		exit_status = 110;
		return;
	}

	// now benchmark hot runs
	ttime = 0.0;
	for( size_t i = 0; i < rep; ++i ) {
		alpha_right_unmasked = realm.template getIdentity< double >();
		timer.reset();
		grb::RC looped_rc = grb::foldr( xv, alpha_right_unmasked, realm );
		ttime += timer.time() / static_cast< double >( rep );
		if( looped_rc != grb::SUCCESS ) {
			std::cerr << "Error: foldl into scalar during hot loop failed.\n";
			error = true;
		}
		if( !grb::utils::equals( check, alpha_right_unmasked, n-1 ) ) {
			std::cerr << "Error: " << alpha_right_unmasked << " (ALP foldr, re-entrant) "
				<< "does not equal " << check << ".\n",
			error = true;
		}
	}
	if( grb::collectives<>::reduce( ttime, 0, grb::operators::max< double >() ) != grb::SUCCESS ) {
		std::cerr << "Error: reduction of ALP foldr timing failed.\n";
		exit_status = 115;
		return;
	}
	if( grb::spmd<>::pid() == 0 ) {
		std::cout << "\t average time taken for ALP reduce by foldr: " << ttime << ".\n";
		std::cout << "\t average time taken for compiler-optimised reduce: "
			<< ctime << "." << std::endl;
	}

	if( !error ) {
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "\t benchmark run 3 complete and verified.\n" << std::endl;
		}
	} else {
		std::cerr << std::flush;
		exit_status = 120;
		return;
	}

	// done
	if( !error ) {
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "Please check the above performance figures manually-- "
				<< "the first two and last two timings should approximately match "
				<< "whenever one user process and one thread is used.\n";
#ifndef NDEBUG
			std::cout << "Since compilation did NOT define the NDEBUG macro, "
				<< "timings may differ more than usual.\n";
#endif
		}
		assert( exit_status == 0 );
	}

}

