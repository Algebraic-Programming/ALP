
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

#include <iostream>
#include <sstream>
#include <vector>

#include <alp.hpp>

void alp_program( const size_t & n, alp::RC & rc ) {

	(void) n;

	/**
	 * Basic checks on the less-then relation (lt).
	 * lt is a strict total order and therefore also a strict partial order.
	 */
	typedef alp::relations::lt< double > dbl_lt;

	static_assert( alp::is_relation< dbl_lt >::value );

	static_assert( ! alp::is_partial_order< dbl_lt >::value );

	static_assert( alp::is_strict_partial_order< dbl_lt >::value );

	static_assert( ! alp::is_total_order< dbl_lt >::value );

	static_assert( alp::is_strict_total_order< dbl_lt >::value );

	static_assert( ! alp::is_equivalence_relation< dbl_lt >::value );

	if( ! dbl_lt::check(2.4, 5) ) {
#ifndef NDEBUG
		std::cerr << "dbl_lt::test(2.4, 5) failed." << std::endl;
#endif
		rc = alp::FAILED;
		return;
	}

	if( dbl_lt::check(5, 2.4) ) {
#ifndef NDEBUG
		std::cerr << "dbl_lt::check(5, 2.4) failed." << std::endl;
#endif
		rc = alp::FAILED;
		return;
	}

	if( dbl_lt::check(5, 5) ) {
#ifndef NDEBUG
		std::cerr << "dbl_lt::test(5, 5) failed." << std::endl;
#endif
		rc = alp::FAILED;
		return;
	}

	/**
	 * Basic checks on the equality relation (eq).
	 * eq is both an equivalence relation and a partial order.
	 */
	typedef alp::relations::eq< int > int_eq;

	static_assert( alp::is_relation< int_eq >::value );

	static_assert( alp::is_partial_order< int_eq >::value );

	static_assert( ! alp::is_strict_partial_order< int_eq >::value );

	static_assert( ! alp::is_total_order< int_eq >::value );

	static_assert( ! alp::is_strict_total_order< int_eq >::value );

	static_assert( alp::is_equivalence_relation< int_eq >::value );

	if( int_eq::check(2.4, 5) ) {
#ifndef NDEBUG
		std::cerr << "int_eq::test(2.4, 5) failed." << std::endl;
#endif
		rc = alp::FAILED;
		return;
	}

	if( int_eq::check(5, 2.4) ) {
#ifndef NDEBUG
		std::cerr << "int_eq::check(5, 2.4) failed." << std::endl;
#endif
		rc = alp::FAILED;
		return;
	}

	if( ! int_eq::check(5, 5) ) {
#ifndef NDEBUG
		std::cerr << "int_eq::test(5, 5) failed." << std::endl;
#endif
		rc = alp::FAILED;
		return;
	}

	if( ! int_eq::check(5.5, 5) ) {
#ifndef NDEBUG
		std::cerr << "int_eq::test(5.5, 5) failed." << std::endl;
#endif
		rc = alp::FAILED;
		return;
	}

	/**
	 * Basic checks on the less-then-or-equal relation (le).
	 * le is both a total order and therefore also a partial order.
	 */
	typedef alp::relations::le< int > int_le;

	static_assert( alp::is_relation< int_le >::value );

	static_assert( alp::is_partial_order< int_le >::value );

	static_assert( ! alp::is_strict_partial_order< int_le >::value );

	static_assert( alp::is_total_order< int_le >::value );

	static_assert( ! alp::is_strict_total_order< int_le >::value );

	static_assert( ! alp::is_equivalence_relation< int_le >::value );

	if( ! int_le::check(2.4, 5) ) {
#ifndef NDEBUG
		std::cerr << "int_le::test(2.4, 5) failed." << std::endl;
#endif
		rc = alp::FAILED;
		return;
	}

	if( int_le::check(5, 2.4) ) {
#ifndef NDEBUG
		std::cerr << "int_le::check(5, 2.4) failed." << std::endl;
#endif
		rc = alp::FAILED;
		return;
	}

	if( ! int_le::check(5, 5) ) {
#ifndef NDEBUG
		std::cerr << "int_le::test(5, 5) failed." << std::endl;
#endif
		rc = alp::FAILED;
		return;
	}

	if( ! int_le::check(5.5, 5) ) {
#ifndef NDEBUG
		std::cerr << "int_le::test(5.5, 5) failed." << std::endl;
#endif
		rc = alp::FAILED;
		return;
	}

	rc = alp::SUCCESS;

}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in;

	// error checking
	if( argc > 1 ) {
		printUsage = true;
	}

	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << "\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	alp::Launcher< alp::AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alp_program, in, out, true ) != alp::SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != alp::SUCCESS ) {
		std::cerr << "Test FAILED (" << alp::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}

