
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
int expect_sparse_success(
	grb::Vector< double > &xv,
	MonT &mon,
	const double check,
	const grb::Vector< bool > &mask,
	const double check_unmasked
) {
	const size_t nz = grb::nnz( xv );
	std::cout << "\nStarting functional tests for sparse inputs\n"
		<< "\t descriptor: " << descr << "\n"
		<< "\t nonzeroes:  " << nz << "\n"
		<< "\t checksum 1: " << check << "\n"
		<< "\t checksum 2: " << check_unmasked << "\n"
		<< "\t mask:       ";
	if( grb::size( mask ) > 0 ) {
		std::cout << grb::nnz( mask ) << " elements.\n";
	} else {
		std::cout << "none.\n";
	}

	double alpha = 3.14;
	if( grb::foldl< descr >( alpha, xv, mask, mon ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool sparse_foldl_into_scalar_failed = false;
		assert( sparse_foldl_into_scalar_failed );
#endif
		return 41;
	}

	double alpha_unmasked = 2.17;
	if( grb::foldl< descr >( alpha_unmasked, xv, mon ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool sparse_foldl_into_scalar_unmasked_failed = false;
		assert( sparse_foldl_into_scalar_unmasked_failed );
#endif
		return 46;
	}

	double alpha_right = -2.22;
	if( grb::foldr< descr >( xv, mask, alpha_right, mon ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool sparse_foldr_into_scalar_failed = false;
		assert( sparse_foldr_into_scalar_failed );
#endif
		return 51;
	}

	double alpha_right_unmasked = -check;
	if( grb::foldr< descr >( xv, alpha_right_unmasked, mon ) != grb::SUCCESS ) {
#ifndef NDEBUG
		const bool sparse_foldr_into_scalar_unmasked_failed = false;
		assert( sparse_foldr_into_scalar_unmasked_failed );
#endif
		return 61;
	}

	// verify computations
	alpha -= 3.14;
	alpha_unmasked -= 2.17;
	alpha_right += 2.22;
	alpha_right_unmasked += check;
	bool error = false;
	if( !grb::utils::equals( alpha, alpha_right, nz+1 ) ) {
		std::cerr << "Error: " << alpha_right << " (sparse foldr, masked) "
			<< " does not equal " << alpha << " (sparse foldl, masked).\n";
		error = true;
	}
	if( !grb::utils::equals( alpha_unmasked, alpha_right_unmasked, nz+1 ) ) {
		std::cerr << "Error: " << alpha_unmasked << " (sparse foldl, unmasked) "
			<< "does not equal " << alpha_right_unmasked << " "
			<< "(sparse foldr, unmasked).\n";
		error = true;
	}
	if( size( mask ) == 0 ) {
		if( !grb::utils::equals( alpha, alpha_right_unmasked, nz+1 ) ) {
			std::cerr << "Error: " << alpha_right_unmasked << " "
				<< "(sparse foldr, unmasked) does not equal " << alpha << " "
				<< "(sparse foldl, masked).\n";
			error = true;
		}
		if( !grb::utils::equals( alpha, alpha_unmasked, nz+1 ) ) {
			std::cerr << "Error: " << alpha_unmasked << " (sparse foldl, unmasked) "
				<< " does not equal " << alpha << " (sparse foldl, masked).\n";
			error = true;
		}
	}
	if( !grb::utils::equals( check, alpha, nz == 0 ? 1 : nz ) ) {
		std::cerr << "Error: " << alpha << " does not equal given checksum " << check
			<< ".\n";
		error = true;
	}
	if( size( mask ) > 0 ) {
		if( !grb::utils::equals( alpha_unmasked, check_unmasked, nz+1 ) ) {
			std::cerr << "Error: " << alpha_unmasked << " does not equal given unmasked "
				<< "checksum " << check_unmasked << ".\n";
			error = true;
		}
		if( !grb::utils::equals( alpha_right_unmasked, check_unmasked, nz+1 ) ) {
			std::cerr << "Error: " << alpha_right_unmasked << " does not equal given "
				<< "unmasked checksum " << check_unmasked << ".\n";
			error = true;
		}
	}
	if( !error ) {
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "Sparse functional tests complete.\n";
		}
	} else {
		std::cerr << std::flush;
		return 71;
	}
	return 0;
}

template< Descriptor descr, typename MonT >
int expect_sparse_success(
	grb::Vector< double > &xv,
	MonT &mon,
	const double check,
	const grb::Vector< bool > mask = NO_MASK
) {
	return expect_sparse_success< descr, MonT >( xv, mon, check, mask, check );
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

	// check happy paths, all fold-to-scalar variants
	exit_status = expect_success< grb::descriptors::no_operation >( xv, realm,
		n, check );
	if( exit_status != 0 ) { return; }

	// check happy paths, with dense descriptor
	exit_status = expect_success< grb::descriptors::dense >( xv, realm, n, check );
	if( exit_status != 0 ) {
		exit_status += 200;
		return;
	}

	// check happy paths, with masking
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

	// check happy paths, with inverted masking
	exit_status = expect_success< grb::descriptors::invert_mask >( xv, realm, n,
		check, even_mask );
	if( exit_status != 0 ) {
		exit_status += 400;
		return;
	}

	// do similar happy path testing, but now for sparse inputs
	{
		grb::Vector< double > sparse( n ), empty( n ), single( n ), singleFirst( n );
		grb::Vector< bool > empty_mask( n ), odd_mask( n ), half_mask( n ), full( n );
		grb::RC rc = grb::set( sparse, even_mask, 1.0 );
		assert( rc == grb::SUCCESS );
		rc = rc ? rc : grb::set( full, true );
		assert( rc == grb::SUCCESS );
		rc = rc ? rc : grb::setElement( single, 3.141, n/2 );
		assert( rc == grb::SUCCESS );
		rc = rc ? rc : grb::setElement( singleFirst, -1.7, 0 );
		assert( rc == grb::SUCCESS );
		rc = rc ? rc : grb::setElement( half_mask, true, n/2 );
		assert( rc == grb::SUCCESS );
		for( size_t i = 1; rc == grb::SUCCESS && i < n; i += 2 ) {
			rc = grb::setElement( odd_mask, true, i );
		}
		assert( rc == grb::SUCCESS );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Could not initialise for sparse tests\n";
			exit_status = 31;
			return;
		}

		exit_status = expect_sparse_success< grb::descriptors::no_operation >(
			empty, realm, 0.0 );
		if( exit_status != 0 ) {
			return;
		}

		exit_status = expect_sparse_success< grb::descriptors::no_operation >(
			empty, realm, 0.0, even_mask );
		if( exit_status != 0 ) {
			exit_status += 100;
			return;
		}

		exit_status = expect_sparse_success< grb::descriptors::no_operation >(
			sparse, realm, grb::nnz(sparse) );
		if( exit_status != 0 ) {
			exit_status += 200;
			return;
		}

		exit_status = expect_sparse_success< grb::descriptors::no_operation >(
			sparse, realm, 0.0, empty_mask, grb::nnz(sparse) );
		if( exit_status != 0 ) {
			exit_status += 300;
			return;
		}

		exit_status = expect_sparse_success< grb::descriptors::structural >(
			sparse, realm, 0.0, empty_mask, grb::nnz(sparse) );
		if( exit_status != 0 ) {
			exit_status += 400;
			return;
		}

		exit_status = expect_sparse_success< grb::descriptors::invert_mask >(
			sparse, realm, grb::nnz(sparse), empty_mask );
		if( exit_status != 0 ) {
			exit_status += 500;
			return;
		}

		exit_status = expect_sparse_success<
			grb::descriptors::invert_mask | grb::descriptors::structural
		>(
			sparse, realm, grb::nnz(sparse), empty_mask
		);
		if( exit_status != 0 ) {
			exit_status += 600;
			return;
		}

		exit_status = expect_sparse_success< grb::descriptors::no_operation >(
			sparse, realm, grb::nnz(sparse), even_mask );
		if( exit_status != 0 ) {
			exit_status += 700;
			return;
		}

		exit_status = expect_sparse_success< grb::descriptors::no_operation >(
			sparse, realm, 0.0, odd_mask, grb::nnz(sparse) );
		if( exit_status != 0 ) {
			exit_status += 800;
			return;
		}

		exit_status = expect_sparse_success<
			grb::descriptors::no_operation | grb::descriptors::structural
		>(
			sparse, realm, grb::nnz(sparse), even_mask
		);
		if( exit_status != 0 ) {
			exit_status += 900;
			return;
		}

		exit_status = expect_sparse_success<
			grb::descriptors::no_operation | grb::descriptors::structural
		>(
			sparse, realm, 0.0, odd_mask, grb::nnz(sparse)
		);
		if( exit_status != 0 ) {
			exit_status += 1000;
			return;
		}

		exit_status = expect_sparse_success< grb::descriptors::invert_mask >(
			sparse, realm, grb::nnz(sparse), odd_mask );
		if( exit_status != 0 ) {
			exit_status += 1100;
			return;
		}

		exit_status = expect_sparse_success<
			grb::descriptors::invert_mask | grb::descriptors::structural
		>(
			sparse, realm, grb::nnz(sparse), odd_mask
		);
		if( exit_status != 0 ) {
			exit_status += 1200;
			return;
		}

		exit_status = expect_sparse_success< grb::descriptors::structural >(
			single, realm, 3.141, full );
		if( exit_status != 0 ) {
			exit_status += 1300;
			return;
		}

		{
			grb::Vector< bool > tmpMask( n );
			rc = grb::setElement( tmpMask, true, 0 );
			rc = rc ? rc : grb::setElement( tmpMask, false, n / 2 );
			if( rc != grb::SUCCESS ) {
				exit_status = 1401;
				return;
			}
			exit_status = expect_sparse_success<
				grb::descriptors::structural | grb::descriptors::invert_mask
			>(
				singleFirst, realm, 0, tmpMask, -1.7
			);
			if( exit_status != 0 ) {
				exit_status += 1400;
				return;
			}
		}

		// warning: below set of two tests alter half_mask
		{
			double expect = (n/2) % 2 == 0
				? 1.0
				: 0.0;
			exit_status = expect_sparse_success< grb::descriptors::structural >(
				sparse, realm, expect, half_mask, grb::nnz(sparse) );
			if( exit_status != 0 ) {
				exit_status += 1500;
				return;
			}

			static_assert( n > 1, "These tests require n=2 or larger" );
			grb::RC rc = grb::setElement( half_mask, false, n/2 );
			rc = rc ? rc : grb::setElement( half_mask, true, n/2 + 1 );
			if( rc == grb::SUCCESS ) {
				expect = (n/2+1) % 2 == 0
					? 1.0
					: 0.0;
				exit_status = expect_sparse_success< grb::descriptors::no_operation >(
					sparse, realm, expect, half_mask, grb::nnz(sparse) );
			} else {
				exit_status = 1632;
				return;
			}

			if( exit_status != 0 ) {
				exit_status += 1600;
				return;
			}
		}
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

	// done
	std::cout << "\n";
	assert( exit_status == 0 );
}

