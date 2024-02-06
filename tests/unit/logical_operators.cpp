
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
#include <unordered_map>
#include <vector>
#include <string>

#include <graphblas.hpp>
#include <graphblas/ops.hpp>

using namespace grb;
using namespace grb::operators;

constexpr std::array< std::pair< bool, bool >, 4 > test_values = {
	std::make_pair( false, false ),
	std::make_pair( false, true ),
	std::make_pair( true, false ),
	std::make_pair( true, true )
};

template<
	Descriptor descr = descriptors::no_operation,
	typename OP
>
void test_apply(
	RC &rc,
	std::array< bool, 4 > &values,
	const std::array< bool, 4 > &expected
) {
	rc = apply< descr, OP >( values[0], false, false );
	rc = rc ? rc : apply< descr, OP >( values[1], false, true );
	rc = rc ? rc : apply< descr, OP >( values[2], true, false );
	rc = rc ? rc : apply< descr, OP >( values[3], true, true );
	if( ! std::equal( values.cbegin(), values.cend(), expected.cbegin() ) ) {
		rc = FAILED;
	}
}

template<
	Descriptor descr = descriptors::no_operation,
	typename OP
>
void test_foldl(
	RC &rc,
	std::array< bool, 4 > &values,
	const std::array< bool, 4 > &expected
) {
	values[0] = false;
	rc = foldl< descr, OP >( values[0], false );
	values[1] = false;
	rc = rc ? rc : foldl< descr, OP >( values[1], true );
	values[2] = true;
	rc = rc ? rc : foldl< descr, OP >( values[2], false );
	values[3] = true;
	rc = rc ? rc : foldl< descr, OP >( values[3], true );
	if( ! std::equal( values.cbegin(), values.cend(), expected.cbegin() ) ) {
		rc = FAILED;
	}
}

template<
	Descriptor descr = descriptors::no_operation,
	typename OP
>
void test_foldr(
	RC &rc,
	std::array< bool, 4 > &values,
	const std::array< bool, 4 > &expected
) {
	values[0] = false;
	rc = foldr< descr, OP >( false, values[0] );
	values[1] = false;
	rc = rc ? rc : foldr< descr, OP >( true, values[1] );
	values[2] = true;
	rc = rc ? rc : foldr< descr, OP >( false, values[2] );
	values[3] = true;
	rc = rc ? rc : foldr< descr, OP >( true, values[3] );
	if( ! std::equal( values.cbegin(), values.cend(), expected.cbegin() ) ) {
		rc = FAILED;
	}
}

template<
	Descriptor descr = descriptors::no_operation,
	typename OP
>
void test_operator(
	RC &rc,
	const std::array< bool, 4 > &expected,
	const bool expected_associative,
	const typename std::enable_if<
		grb::is_operator< OP >::value,
		void
	>::type* = nullptr
) {

	if( grb::is_associative< OP >::value != expected_associative ) {
		std::cerr << "Operator associtativity property is "
			<< grb::is_associative< OP >::value << ", should be "
			<< expected_associative << "\n";
		rc = FAILED;
		return;
	}

	std::array< bool, 4 > values;

	test_apply< descr, OP >( rc, values, expected );
	if( rc != SUCCESS ) {
		std::cerr << "Test_apply FAILED\n";
		std::cerr << "values ?= expected\n";
		for( size_t i = 0; i < 4; i++ ) {
			std::cerr << "OP( " << test_values[i].first << ";"
				<< test_values[i].second << " ): "<< values[i] << " ?= "
				<< expected[i] << "\n";
		}
		return;
	}
	test_foldl< descr, OP >( rc, values, expected );
	if( rc != SUCCESS ) {
		std::cerr << "Test_foldl FAILED\n";
		std::cerr << "values ?= expected\n";
		for( size_t i = 0; i < 4; i++ ) {
			std::cerr << "OP( " << test_values[i].first << ";"
				<< test_values[i].second << " ): "<< values[i] << " ?= "
				<< expected[i] << "\n";
		}
		return;
	}
	test_foldr< descr, OP >( rc, values, expected );
	if( rc != SUCCESS ) {
		std::cerr << "Test_foldr FAILED\n";
		std::cerr << "values ?= expected\n";
		for( size_t i = 0; i < 4; i++ ) {
			std::cerr << "OP( " << test_values[i].first << ";"
				<< test_values[i].second << " ): "<< values[i] << " ?= "
				<< expected[i] << "\n";
		}
		return;
	}
}

void grb_program( const size_t&, grb::RC &rc ) {
	rc = SUCCESS;

	// Logical operators
	{ // logical_and< bool >
		std::cout << "Testing operator: logical_and<bool>" << std::endl;
		const std::array<bool, 4> expected = { false, false, false, true };
		bool expected_associative = true;
		test_operator<
			descriptors::no_operation,
			logical_and< bool >
		>( rc, expected, expected_associative );
	}
	{ // logical_or< bool >
		std::cout << "Testing operator: logical_or<bool>" << std::endl;
		const std::array<bool, 4> expected = { false, true, true, true };
		bool expected_associative = true;
		test_operator<
			descriptors::no_operation,
			logical_or< bool >
		>( rc, expected, expected_associative );
	}
	{ // logical_xor< bool >
		std::cout << "Testing operator: logical_xor<bool>" << std::endl;
		const std::array<bool, 4> expected = { false, true, true, false };
		bool expected_associative = true;
		test_operator<
			descriptors::no_operation,
			logical_xor< bool >
		>( rc, expected, expected_associative );
	}

	// Negated operators
	{ // logical_not< logical_and< bool > >
		std::cout << "Testing operator: logical_not< logical_and< bool > >" << std::endl;
		const std::array<bool, 4> expected = { true, true, true, false };
		bool expected_associative = false;
		test_operator<
			descriptors::no_operation,
			logical_not< logical_and< bool > >
		>( rc, expected, expected_associative );
	}
	{ // logical_not< logical_or< bool > >
		std::cout << "Testing operator: logical_not< logical_or< bool > >" << std::endl;
		const std::array<bool, 4> expected = { true, false, false, false };
		bool expected_associative = false;
		test_operator<
			descriptors::no_operation,
			logical_not< logical_or< bool > >
		>( rc, expected, expected_associative );
	}
	{ // logical_not< logical_xor< bool > >
		std::cout << "Testing operator: logical_not< logical_xor< bool > >" << std::endl;
		const std::array<bool, 4> expected = { true, false, false, true };
		bool expected_associative = false;
		test_operator<
			descriptors::no_operation,
			logical_not< logical_xor< bool > >
		>( rc, expected, expected_associative );
	}

	// Double-negated operators
	{ // logical_not< logical_not < logical_and< bool > > >
		std::cout << "Testing operator: logical_not< logical_not < logical_and< bool > > >" << std::endl;
		const std::array<bool, 4> expected = { false, false, false, true };
		bool expected_associative = true;
		test_operator<
			descriptors::no_operation,
			logical_not< logical_not< logical_and< bool > > >
		>( rc, expected, expected_associative );
	}
	{ // logical_not< logical_not < logical_or< bool > > >
		std::cout << "Testing operator: logical_not< logical_not < logical_or< bool > > >" << std::endl;
		const std::array<bool, 4> expected = { false, true, true, true };
		bool expected_associative = true;
		test_operator<
			descriptors::no_operation,
			logical_not< logical_not< logical_or< bool > > >
		>( rc, expected, expected_associative );
	}
	{ // logical_not< logical_not < logical_xor< bool > > >
		std::cout << "Testing operator: logical_not< logical_not < logical_xor< bool > > >" << std::endl;
		const std::array<bool, 4> expected = { false, true, true, false };
		bool expected_associative = true;
		test_operator<
			descriptors::no_operation,
			logical_not< logical_not< logical_xor< bool > > >
		>( rc, expected, expected_associative );
	}
}

int main( int argc, char ** argv ) {
	// error checking
	if( argc > 2 ) {
		std::cerr << "Usage: " << argv[ 0 ] << "\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";

	grb::Launcher< AUTOMATIC > launcher;
	RC out = SUCCESS;
	size_t unused = 0;
	if( launcher.exec( &grb_program, unused, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED (" << grb::toString( out ) << ")\n" << std::flush;
	} else {
		std::cout << "Test OK\n" << std::flush;
	}
	return 0;
}
