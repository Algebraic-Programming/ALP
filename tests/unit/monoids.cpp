
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
 * Tests the default monoid definitions.
 *
 * @author A. N. Yzelman
 * @date 24th of October, 2024
 */

/*#include <graphblas/utils/suppressions.h>
GRB_UTIL_IGNORE_INT_IN_BOOL_CONTEXT
TODO FIXME CHECK if still needed for monoids*/

#include <graphblas.hpp>

#include <iostream>


template< typename Monoid >
bool runTests() {

	Monoid monoid;

	// get identities (the zeroes) in each input domain
	const typename Monoid::D1 d1_zero =
		monoid.template getIdentity< typename Monoid::D1 >();
	const typename Monoid::D2 d2_zero =
		monoid.template getIdentity< typename Monoid::D2 >();

	// get nonzeroes in each input domain
	// without a semiring structure, we cannot just construct one. What we do here
	// instead is use negation of the earlier-retrieved zeroes.
	const typename Monoid::D1 d1_nonzero = !(d1_zero);
	const typename Monoid::D2 d2_nonzero = !(d2_zero);

	// check zero is an identity under addition, zero left:
	{
		typename Monoid::D3 tmp;
		if(
			grb::apply(
				tmp, d1_zero, d2_nonzero, monoid.getOperator()
			) != grb::SUCCESS
		) {
			std::cerr << "Unexpected error in test I\n";
			return false;
		}
		if( tmp != static_cast< typename Monoid::D3 >( d2_nonzero ) ) {
			std::cerr << "Zero in D1 does not act as an identity\n";
			return false;
		}
	}

	// check zero is an identity under addition, zero right:
	{
		typename Monoid::D3 tmp;
		if(
			grb::apply(
				tmp, d1_nonzero, d2_zero, monoid.getOperator()
			) != grb::SUCCESS
		) {
			std::cerr << "Unexpected error in test II\n";
			return false;
		}
		if( tmp != static_cast< typename Monoid::D3 >( d1_nonzero ) ) {
			std::cerr << "Zero in D2 does not act as an identity\n";
			return false;
		}
	}

	// check commutativity (if applicable)
	if( grb::is_commutative< Monoid >::value ) {
		typename Monoid::D3 left, right;
		if(
			grb::apply(
				left, d1_zero, d2_nonzero, monoid.getOperator()
			) != grb::SUCCESS
		) {
			std::cerr << "Unexpected error in test III (1)\n";
			return false;
		}
		if(
			grb::apply(
				right, d1_nonzero, d2_zero, monoid.getOperator()
			) != grb::SUCCESS
		) {
			std::cerr << "Unexpected error in test III (2)\n";
			return false;
		}
		if( left != right ) {
			std::cerr << "Non-commutative behaviour detected "
				<< "while the commutative type trait was true\n";
			return false;
		}
	}

	// all OK
	return true;
}

template<
	template< typename D1, typename D2 = D1, typename D3 = D2 >
	class Monoid
>
bool runTestsAllDomains() {
	std::cout << "\t\t testing over doubles:\n";
	if( runTests< Monoid< double > >() ) {
		std::cout << "\t\t OK\n";
	} else {
		std::cout << "\t\t ERR\n";
		return false;
	}

	std::cout << "\t\t testing over floats:\n";
	if( runTests< Monoid< float > >() ) {
		std::cout << "\t\t OK\n";
	} else {
		std::cout << "\t\t ERR\n";
		return false;
	}

	std::cout << "\t\t testing over short ints:\n";
	if( runTests< Monoid< short int > >() ) {
		std::cout << "\t\t OK\n";
	} else {
		std::cout << "\t\t ERR\n";
		return false;
	}

	std::cout << "\t\t testing over integers:\n";
	if( runTests< Monoid< int > >() ) {
		std::cout << "\t\t OK\n";
	} else {
		std::cout << "\t\t ERR\n";
		return false;
	}

	std::cout << "\t\t testing over 64-bit integers:\n";
	if( runTests< Monoid< int64_t > >() ) {
		std::cout << "\t\t OK\n";
	} else {
		std::cout << "\t\t ERR\n";
		return false;
	}

	std::cout << "\t\t testing over short unsigned integers:\n";
	if( runTests< Monoid< short unsigned int > >() ) {
		std::cout << "\t\t OK\n";
	} else {
		std::cout << "\t\t ERR\n";
		return false;
	}

	std::cout << "\t\t testing over unsigned integers:\n";
	if( runTests< Monoid< unsigned int > >() ) {
		std::cout << "\t\t OK\n";
	} else {
		std::cout << "\t\t ERR\n";
		return false;
	}

	std::cout << "\t\t testing over size_ts:\n";
	if( runTests< Monoid< size_t > >() ) {
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
	
	std::cout << "\t testing grb::monoids::plus...\n";
	if( runTestsAllDomains< grb::monoids::plus >() ) {
		std::cout << "\t OK\n";
	} else {
		std::cout << "\t ERR\n";
		ok = false;
	}

	std::cout << "\t testing grb::monoids::add...\n";
	if( runTestsAllDomains< grb::monoids::add >() ) {
		std::cout << "\t OK\n";
	} else {
		std::cout << "\t ERR\n";
		ok = false;
	}

	std::cout << "\t testing grb::monoids::times...\n";
	if( runTestsAllDomains< grb::monoids::times >() ) {
		std::cout << "\t OK\n";
	} else {
		std::cout << "\t ERR\n";
		ok = false;
	}

	std::cout << "\t testing grb::monoids::mul...\n";
	if( runTestsAllDomains< grb::monoids::mul >() ) {
		std::cout << "\t OK\n";
	} else {
		std::cout << "\t ERR\n";
		ok = false;
	}

	std::cout << "\t testing grb::monoids::min...\n";
	if( runTestsAllDomains< grb::monoids::min >() ) {
		std::cout << "\t OK\n";
	} else {
		std::cout << "\t ERR\n";
		ok = false;
	}

	std::cout << "\t testing grb::monoids::max...\n";
	if( runTestsAllDomains< grb::monoids::max >() ) {
		std::cout << "\t OK\n";
	} else {
		std::cout << "\t ERR\n";
		ok = false;
	}

	std::cout << "\t testing grb::monoids::lor over Booleans...\n";
	if( runTests< grb::monoids::lor< bool > >() ) {
		std::cout << "\t OK\n";
	} else {
		std::cout << "\t ERR\n";
		ok = false;
	}

	std::cout << "\t testing grb::monoids::land over Booleans...\n";
	if( runTests< grb::monoids::land< bool > >() ) {
		std::cout << "\t OK\n";
	} else {
		std::cout << "\t ERR\n";
		ok = false;
	}

	std::cout << "\t testing grb::monoids::lxor over Booleans...\n";
	if( runTests< grb::monoids::lxor< bool > >() ) {
		std::cout << "\t OK\n";
	} else {
		std::cout << "\t ERR\n";
		ok = false;
	}

	std::cout << "\t testing grb::monoids::lneq over Booleans...\n";
	if( runTests< grb::monoids::lneq< bool > >() ) {
		std::cout << "\t OK\n";
	} else {
		std::cout << "\t ERR\n";
		ok = false;
	}

	std::cout << "\t testing grb::monoids::lxnor over Booleans...\n";
	if( runTests< grb::monoids::lxnor< bool > >() ) {
		std::cout << "\t OK\n";
	} else {
		std::cout << "\t ERR\n";
		ok = false;
	}

	std::cout << "\t testing grb::monoids::leq over Booleans...\n";
	if( runTests< grb::monoids::leq< bool > >() ) {
		std::cout << "\t OK\n";
	} else {
		std::cout << "\t ERR\n";
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

