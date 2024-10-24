
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

/**
 * @file
 *
 * Tests the default semiring definitions.
 *
 * @author A. N. Yzelman
 * @date 11th of October, 2024
 */

#include <graphblas/utils/suppressions.h>
GRB_UTIL_IGNORE_INT_IN_BOOL_CONTEXT

#include <graphblas.hpp>

#include <iostream>


template< typename Semiring >
bool runTests() {

	Semiring ring;

	// check zero annihilates one under multiplication, zero left:
	{
		typename Semiring::D3 tmp;
		if(
			grb::apply(
				tmp,
				ring.template getZero< typename Semiring::D1 >(),
				ring.template getOne< typename Semiring::D2 >(),
				ring.getMultiplicativeOperator()
			) != grb::SUCCESS
		) {
			std::cerr << "Unexpected error in test I\n";
			return false;
		}
		if( tmp != ring.template getZero< typename Semiring::D3 >() ) {
			std::cerr << "Zero in D1 does not annihilate one in D2\n";
			return false;
		}
	}

	// check zero annihilates one under multiplication, zero right:
	{
		typename Semiring::D3 tmp;
		if(
			grb::apply(
				tmp,
				ring.template getOne< typename Semiring::D1 >(),
				ring.template getZero< typename Semiring::D2 >(),
				ring.getMultiplicativeOperator()
			) != grb::SUCCESS
		) {
			std::cerr << "Unexpected error in test II\n";
			return false;
		}
		if( tmp != ring.template getZero< typename Semiring::D3 >() ) {
			std::cerr << "Zero in D2 does not annihilate one in D1\n";
			return false;
		}
	}

	// check zero is an identity under addition, zero left:
	{
		typename Semiring::D4 tmp;
		if(
			grb::apply(
				tmp,
				ring.template getZero< typename Semiring::D3 >(),
				ring.template getOne< typename Semiring::D4 >(),
				ring.getAdditiveOperator()
			) != grb::SUCCESS
		) {
			std::cerr << "Unexpected error in test III\n";
			return false;
		}
		if( tmp != ring.template getOne< typename Semiring::D4 >() ) {
			std::cerr << "Zero in D3 does not act as an identity under addition\n";
			return false;
		}
	}

	// check zero is an identity under addition, zero right:
	{
		typename Semiring::D4 tmp;
		if(
			grb::apply(
				tmp,
				ring.template getOne< typename Semiring::D3 >(),
				ring.template getZero< typename Semiring::D4 >(),
				ring.getAdditiveOperator()
			) != grb::SUCCESS
		) {
			std::cerr << "Unexpected error in test IV\n";
			return false;
		}
		if( tmp != ring.template getOne< typename Semiring::D4 >() ) {
			std::cerr << "Zero in D4 does not act as an identity under addition\n";
			return false;
		}
	}

	// check one is an identity under multiplication:
	{
		typename Semiring::D3 tmp;
		if(
			grb::apply(
				tmp,
				ring.template getOne< typename Semiring::D1 >(),
				ring.template getOne< typename Semiring::D2 >(),
				ring.getMultiplicativeOperator()
			) != grb::SUCCESS
		) {
			std::cerr << "Unexpected error in test V\n";
			return false;
		}
		if( tmp != ring.template getOne< typename Semiring::D3 >() ) {
			std::cerr << "One does not act as identity under multiplication\n";
			return false;
		}
	}

	// check distributive property:
	{
		typename Semiring::D4 tmp1;
		grb::RC rc = grb::apply(
			tmp1,
			ring.template getOne< typename Semiring::D3 >(),
			ring.template getOne< typename Semiring::D4 >(),
			ring.getAdditiveOperator()
		);
		typename Semiring::D3 chk1;
		rc = rc ? rc : grb::apply(
				chk1,
				ring.template getOne< typename Semiring::D1 >(),
				static_cast< typename Semiring::D2 >(tmp1),
				ring.getMultiplicativeOperator()
			);
		typename Semiring::D3 tmp2, tmp3;
		rc = rc ? rc : grb::apply(
				tmp2,
				ring.template getOne< typename Semiring::D1 >(),
				ring.template getOne< typename Semiring::D2 >(),
				ring.getMultiplicativeOperator()
			);
		rc = rc ? rc : grb::apply(
				tmp3,
				ring.template getOne< typename Semiring::D1 >(),
				ring.template getOne< typename Semiring::D2 >(),
				ring.getMultiplicativeOperator()
			);
		typename Semiring::D3 chk2;
		rc = rc ? rc : grb::apply(
			chk2,
			tmp2,
			static_cast< typename Semiring::D4 >(tmp3),
			ring.getAdditiveOperator()
		);
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected error in test VI\n";
			return false;
		}
		if( chk1 != chk2 ) {
			std::cerr << "The distributative property does not hold\n";
			return false;
		}
	}

	// check commutativity of additive monoid
	{
		typename Semiring::D4 left, right;
		if(
			grb::apply(
				left,
				ring.template getZero< typename Semiring::D3 >(),
				ring.template getOne< typename Semiring::D4 >(),
				ring.getAdditiveOperator()
			) != grb::SUCCESS
		) {
			std::cerr << "Unexpected error in test VII (1)\n";
			return false;
		}
		if(
			grb::apply(
				right,
				ring.template getOne< typename Semiring::D3 >(),
				ring.template getZero< typename Semiring::D4 >(),
				ring.getAdditiveOperator()
			) != grb::SUCCESS
		) {
			std::cerr << "Unexpected error in test VII (2)\n";
			return false;
		}
		if( left != right ) {
			std::cerr << "Non-commutative behaviour of the additive monoid detected\n";
			return false;
		}
	}

	// all OK
	return true;
}

template<
	template< typename D1, typename D2 = D1, typename D3 = D2, typename D4 = D3 >
	class Semiring
>
bool runTestsAllDomains() {
	std::cout << "\t\t testing over doubles:\n";
	if( runTests< Semiring< double > >() ) {
		std::cout << "\t\t OK\n";
	} else {
		std::cout << "\t\t ERR\n";
		return false;
	}

	std::cout << "\t\t testing over floats:\n";
	if( runTests< Semiring< float > >() ) {
		std::cout << "\t\t OK\n";
	} else {
		std::cout << "\t\t ERR\n";
		return false;
	}

	std::cout << "\t\t testing over short ints:\n";
	if( runTests< Semiring< short int > >() ) {
		std::cout << "\t\t OK\n";
	} else {
		std::cout << "\t\t ERR\n";
		return false;
	}

	std::cout << "\t\t testing over integers:\n";
	if( runTests< Semiring< int > >() ) {
		std::cout << "\t\t OK\n";
	} else {
		std::cout << "\t\t ERR\n";
		return false;
	}

	std::cout << "\t\t testing over 64-bit integers:\n";
	if( runTests< Semiring< int64_t > >() ) {
		std::cout << "\t\t OK\n";
	} else {
		std::cout << "\t\t ERR\n";
		return false;
	}

	std::cout << "\t\t testing over short unsigned integers:\n";
	if( runTests< Semiring< short unsigned int > >() ) {
		std::cout << "\t\t OK\n";
	} else {
		std::cout << "\t\t ERR\n";
		return false;
	}

	std::cout << "\t\t testing over unsigned integers:\n";
	if( runTests< Semiring< unsigned int > >() ) {
		std::cout << "\t\t OK\n";
	} else {
		std::cout << "\t\t ERR\n";
		return false;
	}

	std::cout << "\t\t testing over size_ts:\n";
	if( runTests< Semiring< size_t > >() ) {
		std::cout << "\t\t OK\n";
	} else {
		std::cout << "\t\t ERR\n";
		return false;
	}

	return true;
}

int main( int argc, char ** argv ) {
	if( argc > 1 ) {
		std::cerr << "This test does not expect any arguments\n"
			<< "\t Example usage: ./" << argv[ 0 ] << "\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	bool ok = true;

	std::cout << "\t testing grb::semirings::plusTimes...\n";
	if( runTestsAllDomains< grb::semirings::plusTimes >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	std::cout << "\t testing grb::semirings::minPlus...\n";
	if( runTestsAllDomains< grb::semirings::minPlus >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	std::cout << "\t testing grb::semirings::maxPlus over integers:\n";
	if( runTests< grb::semirings::maxPlus< int > >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	std::cout << "\t testing grb::semirings::maxPlus over doubles:\n";
	if( runTests< grb::semirings::maxPlus< double > >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	std::cout << "\t testing grb::semirings::minTimes over unsigned integers:\n";
	if( runTests< grb::semirings::minTimes< unsigned int > >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	std::cout << "\t testing grb::semirings::minMax...\n";
	if( runTestsAllDomains< grb::semirings::minMax >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	std::cout << "\t testing grb::semirings::maxMin...\n";
	if( runTestsAllDomains< grb::semirings::maxMin >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	std::cout << "\t testing grb::semirings::maxTimes over size_ts:\n";
	if( runTests< grb::semirings::maxTimes< size_t > >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	std::cout << "\t testing grb::semirings::plusMin over unsigned integers:\n";
	if( runTests< grb::semirings::plusMin< unsigned int > >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	std::cout << "\t testing grb::semirings::lorLand over Booleans:\n";
	if( runTests< grb::semirings::lorLand< bool > >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	std::cout << "\t testing grb::semirings::boolean:\n";
	if( runTests< grb::semirings::boolean >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	std::cout << "\t testing grb::semirings::landLor over Booleans:\n";
	if( runTests< grb::semirings::landLor< bool > >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	std::cout << "\t testing grb::semirings::lxorLand over Booleans:\n";
	if( runTests< grb::semirings::lxorLand< bool > >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	std::cout << "\t testing grb::semirings::lneqLand over Booleans:\n";
	if( runTests< grb::semirings::lneqLand< bool > >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	std::cout << "\t testing grb::semirings::lxnorLor over Booleans:\n";
	if( runTests< grb::semirings::lxnorLor< bool > >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	std::cout << "\t testing grb::semirings::leqLor over Booleans:\n";
	if( runTests< grb::semirings::leqLor< bool > >() ) {
		std::cout << "\t OK\n";
	} else {
		ok = false;
	}

	// done
	if( ok ) {
		std::cout << "Test OK\n" << std::endl;
	} else {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n" << std::endl;
	}
}

