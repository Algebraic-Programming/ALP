
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

template< Descriptor descr, typename MonT >
int expect_mismatch(
	const grb::Vector< double > &v0,
	const grb::Vector< double > &v1,
	const MonT &mon
) {
	static_assert( grb::is_monoid< MonT >::value, "must be called with a monoid" );
	assert( grb::size( v0 ) + 1 == grb::size( v1 ) );

	double alpha = -1.0;
	bool error = false;
	std::cout << "\nStarting tests for MISMATCH.\n";

	grb::RC rc = foldl< descr >( alpha, v0, v1, mon );
	if( rc != grb::MISMATCH) {
		std::cerr << "\t mismatched call to foldl (T<-[T], masked) "
			<< "returns " << grb::toString( rc ) << " instead of MISMATCH\n";
		error = true;
	}


	rc = foldr< descr >( v1, v0, alpha, mon );
	if( rc != grb::MISMATCH ) {
		std::cerr << "\t mismatched call to foldr ([T]->T, masked) "
			<< "returns " << grb::toString( rc ) << " instead of MISMATCH\n";
		error = true;
	}

	if( alpha != -1.0 ) {
		std::cerr << "One or more calls to foldl/foldr had a side effect on scalar\n";
		error = true;
	}

	if( error ) {
		std::cout << "One or more tests for MISMATCH failed\n";
		return 79;
	} else {
		std::cout << "Tests for MISMATCH complete\n";
		return 0;
	}
}

template< Descriptor descr, typename MonT >
int expect_illegal(
	const grb::Vector< double > &dense_v,
	const grb::Vector< double > &sparse_v,
	const grb::Vector< double > &sparse_m,
	const MonT &mon
) {
	static_assert( grb::is_monoid< MonT >::value, "must be called with a monoid" );
	assert( grb::nnz( dense_v ) == grb::size( dense_v ) );
	assert( grb::nnz( sparse_v ) < grb::size( sparse_v ) );
	assert( grb::nnz( sparse_m ) < grb::size( sparse_m ) );
	assert( grb::size( dense_v ) == grb::size( sparse_v ) );
	assert( grb::size( dense_v ) == grb::size( sparse_m ) );

	double alpha = -1.0;
	bool error = false;
	std::cout << "\nStarting tests for ILLEGAL.\n";

	grb::RC rc = foldl< descr | grb::descriptors::dense >( alpha, sparse_v, mon );
	if( rc != grb::ILLEGAL ) {
		std::cerr << "\t illegal call to foldl (T<-[T], sparse [T], unmasked) "
			<< "returns " << grb::toString( rc ) << " instead of ILLEGAL\n";
		error = true;
	}

	rc = foldl< descr | grb::descriptors::dense >( alpha, dense_v, sparse_m, mon );
	if( rc != grb::ILLEGAL ) {
		std::cerr << "\t illegal call to foldl (T<-[T], dense [T], sparse mask) "
			<< "returns " << grb::toString( rc ) << " instead of ILLEGAL\n";
		error = true;
	}

	rc = foldl< descr | grb::descriptors::dense >( alpha, sparse_v, dense_v,
		mon );
	if( rc != grb::ILLEGAL ) {
		std::cerr << "\t illegal call to foldl (T<-[T], sparse [T], dense mask) "
			<< "returns " << grb::toString( rc ) << " instead of ILLEGAL\n";
		error = true;
	}

	rc = foldl< descr | grb::descriptors::dense >( alpha, sparse_v, sparse_m,
		mon );
	if( rc != grb::ILLEGAL ) {
		std::cerr << "\t illegal call to foldl (T<-[T], sparse [T], sparse mask) "
			<< "returns " << grb::toString( rc ) << " instead of ILLEGAL\n";
		error = true;
	}

	rc = foldr< descr | grb::descriptors::dense >( sparse_v, alpha, mon );
	if( rc != grb::ILLEGAL ) {
		std::cerr << "\t illegal call to foldr ([T]->T, sparse [T], unmasked) "
			<< "returns " << grb::toString( rc ) << " instead of ILLEGAL\n";
		error = true;
	}

	rc = foldr< descr | grb::descriptors::dense >( dense_v, sparse_m, alpha, mon );
	if( rc != grb::ILLEGAL ) {
		std::cerr << "\t illegal call to foldr ([T]->T, dense [T], sparse mask) "
			<< "returns " << grb::toString( rc ) << " instead of ILLEGAL\n";
		error = true;
	}

	rc = foldr< descr | grb::descriptors::dense >( sparse_v, dense_v, alpha,
		mon );
	if( rc != grb::ILLEGAL ) {
		std::cerr << "\t illegal call to foldr ([T]->T, sparse [T], dense mask) "
			<< "returns " << grb::toString( rc ) << " instead of ILLEGAL\n";
		error = true;
	}

	rc = foldr< descr | grb::descriptors::dense >( sparse_v, sparse_m, alpha,
		mon );
	if( rc != grb::ILLEGAL ) {
		std::cerr << "\t illegal call to foldr ([T]->T, sparse [T], sparse mask) "
			<< "returns " << grb::toString( rc ) << " instead of ILLEGAL\n";
		error = true;
	}

	if( alpha != -1.0 ) {
		std::cerr << "One or more calls to foldl/foldr had a side effect on scalar\n";
		error = true;
	}

	if( error ) {
		std::cout << "One or more tests for ILLEGAL failed\n";
		return 77;
	} else {
		std::cout << "Tests for ILLEGAL complete\n";
		return 0;
	}
}

template< Descriptor descr, typename MonT >
int expect_success(
	grb::Vector< double > &xv,
	MonT &mon,
	const size_t n,
	const double check,
	const grb::Vector< bool > mask = NO_MASK
) {
	std::cout << "\nStarting functional tests ";
	if( grb::size( mask ) > 0 ) {
		std::cout << "with a mask holding " << grb::nnz( mask ) << " elements.\n";
	} else {
		std::cout << "without a mask.\n";
	}
	double alpha = 0.0;
	if( grb::foldl< descr >( alpha, xv, mask, mon ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool foldl_into_scalar_failed = false;
		assert( foldl_into_scalar_failed );
#endif
		return 40;
	}

	double alpha_unmasked = 0.0;
	if( grb::foldl< descr >( alpha_unmasked, xv, mon ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool foldl_into_scalar_unmasked_failed = false;
		assert( foldl_into_scalar_unmasked_failed );
#endif
		return 45;
	}

#ifdef _DEBUG
	std::cerr << "Info: grbProgram (reduce) post foldl" << std::endl;
#endif

	double alpha_right = 0.0;
	if( grb::foldr< descr >( xv, mask, alpha_right, mon ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool foldr_into_scalar_failed = false;
		assert( foldr_into_scalar_failed );
#endif
		return 50;
	}

	double alpha_right_unmasked = 0.0;
	if( grb::foldr< descr >( xv, alpha_right_unmasked, mon ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool foldr_into_scalar_unmasked_failed = false;
		assert( foldr_into_scalar_unmasked_failed );
#endif
		return 60;
	}

#ifdef _DEBUG
	std::cerr << "Info: grbProgram (reduce) post foldr" << std::endl;
#endif

	// verify computations
	bool error = false;
	if( !grb::utils::equals( alpha, alpha_right, n-1 ) ) {
		std::cerr << "Error: " << alpha_right << " (foldr, masked) "
			<< " does not equal " << alpha << " (foldl, masked).\n";
		error = true;
	}
	if( !grb::utils::equals( alpha_unmasked, alpha_right_unmasked, n-1 ) ) {
		std::cerr << "Error: " << alpha_unmasked << " (foldl, unmasked) "
			<< "does not equal " << alpha_right_unmasked << " (foldr, unmasked).\n";
		error = true;
	}
	if( size( mask ) == 0 ) {
		if( !grb::utils::equals( alpha, alpha_right_unmasked, n-1 ) ) {
			std::cerr << "Error: " << alpha_right_unmasked << " (foldr, unmasked) "
				<< " does not equal " << alpha << " (foldl, masked).\n";
			error = true;
		}
		if( !grb::utils::equals( alpha, alpha_unmasked, n-1 ) ) {
			std::cerr << "Error: " << alpha_unmasked << " (foldl, unmasked) "
				<< " does not equal " << alpha << " (foldl, masked).\n";
			error = true;
		}
	}
	if( !grb::utils::equals( check, alpha, n-1 ) ) {
		std::cerr << "Error: " << alpha << " does not equal given checksum " << check
			<< ".\n";
		error = true;
	}
	if( !error ) {
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "Functional tests complete.\n";
		}
	} else {
		std::cerr << std::flush;
		return 70;
	}
	return 0;
}

void grbProgram( const size_t &P, int &exit_status ) {
	(void) P;
	assert( exit_status == 0 );

#ifdef _DEBUG
	std::cerr << "Info: grbProgram (reduce) entered with parameters: "
		<< P << ", " << exit_status << "\n";
#endif

	grb::utils::Timer benchtimer;
	benchtimer.reset();

	grb::Monoid< grb::operators::add< double >, grb::identities::zero > realm;
	grb::operators::add< double > realop;

	grb::Vector< double > xv( n );
	double check = 0.0;
	double * __restrict__ xr = nullptr;

	int rc = posix_memalign( (void **) &xr,
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
		xr[ i ] = (double) i;
		check += (double) i;
	}

#ifdef _DEBUG
	std::cerr << "Info: grbProgram (reduce) vector allocs of size " << n << ". "
		<< "Initialisation complete." << std::endl;
#endif

	exit_status = expect_success< grb::descriptors::no_operation >( xv, realm,
		n, check );
	if( exit_status != 0 ) { return; }
	exit_status = expect_success< grb::descriptors::dense >( xv, realm, n, check );
	if( exit_status != 0 ) {
		exit_status += 200;
		return;
	}

	grb::Vector< bool > even_mask( n );
	check = 0.0;
	for( size_t i = 0; i < n; i += 2 ) {
		check += xr[ i ];
		const grb::RC setrc = grb::setElement( even_mask, true, i );
#ifndef NDEBUG
		assert( setrc == grb::SUCCESS );
#else
		(void) setrc;
#endif
	}
	exit_status = expect_success< grb::descriptors::no_operation >( xv, realm, n,
		check, even_mask );
	if( exit_status != 0 ) {
		exit_status += 300;
		return;
	}

	check = 0.0;
	for( size_t i = 1; i < n; i += 2 ) {
		check += xr[ i ];
	}

	exit_status = expect_success< grb::descriptors::invert_mask >( xv, realm, n,
		check, even_mask );
	if( exit_status != 0 ) {
		exit_status += 400;
		return;
	}

	// check whether ILLEGAL is returned when appropriate
	{
		grb::Vector< double > half_sparse( n );
		grb::Vector< double > very_sparse( n );
		if( grb::set( half_sparse, even_mask, 1.0 ) != grb::SUCCESS ) {
			std::cerr << "Could not initialise for illegal tests\n";
			exit_status = 75;
			return;
		}

		exit_status = expect_illegal< grb::descriptors::no_operation >( xv,
			very_sparse, half_sparse, realm );
		if( exit_status != 0 ) {
			return;
		}

		exit_status = expect_illegal< grb::descriptors::invert_mask >( xv,
			half_sparse, very_sparse, realm );
		if( exit_status != 0 ) {
			exit_status += 100;
			return;
		}
	}

	// check whether MISMATCH is returned when appropriate
	{
		grb::Vector< double > xp1( n + 1 );
		exit_status = expect_mismatch< grb::descriptors::no_operation >( xv, xp1,
			realm );
		if( exit_status != 0 ) {
			return;
		}
		exit_status = expect_mismatch< grb::descriptors::dense >( xv, xp1,
			realm );
		if( exit_status != 0 ) {
			return;
		}
		exit_status = expect_mismatch< grb::descriptors::invert_mask >( xv, xp1,
			realm );
		if( exit_status != 0 ) {
			return;
		}
	}

	// then do a benchmark
	std::cout << "\nNow starting benchmark run 1 (ALP foldl):" << std::endl;
	check = 0.0;
	for( size_t i = 0; i < n; ++i ) {
		check += xr[ i ];
	}

	// first do a cold run
	double alpha = 0.0;
	if( grb::foldl( alpha, xv, realm ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool cold_foldl_into_scalar_failed = false;
		assert( cold_foldl_into_scalar_failed );
#endif
		exit_status = 80;
		return;
	}

	double ttime = 0.0;
	bool error = false;
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
	double alpha_right_unmasked = 0.0;
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

