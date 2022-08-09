
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
#include <cmath>

#include <graphblas/utils.hpp>


/**
 * Templated tests for floating point types.
 *
 * @tparam FloatType A floating point type such as <tt>float</tt> or
 *                   <tt>double</tt>.
 *
 * This function tests for equality, in turn:
 *   - bit-wise equality (two times);
 *   - that the two different floats used for bit-wise equality do not equal
 *     one another;
 *   - various combinations of equal and non-equal numbers in the subnormal
 *     range;
 *   - various combinations of equal and non-equal numbers that when summed
 *     would overflow;
 *   - various combinations of equal and non-equal numbers that lie close one
 *     another, then scaled three orders of magnitude and repeated.
 *
 * @returns 0 if all tests pass, and a unique nonzero identifier corresponding
 *            to the failed test otherwise.
 */
template< typename FloatType >
int floatTest() {
	int out = 0;

	// bit-wise equality
	{
		const FloatType pi = 3.1415926535;
		const FloatType zero = 0.0;
		if( !grb::utils::equals( zero, zero, 1 ) ) {
			std::cerr << "Error during bit-wise equal comparison (I)\n";
			out = 100;
		}
		if( !out && !grb::utils::equals( pi, pi, 1 ) ) {
			std::cerr << "Error during bit-wise equal comparison (II)\n";
			out = 110;
		}
		if( !out && grb::utils::equals( zero, pi, 1 ) ) {
			std::cerr << "Error during bit-wise equal comparison (III)\n";
			out = 120;
		}
		if( !out && grb::utils::equals( pi, zero, 1 ) ) {
			std::cerr << "Error during bit-wise equal comparison (IV)\n";
			out = 130;
		}
	}

	// subnormal comparisons
	if( out == 0 ) {
		const FloatType subnormal_two = std::numeric_limits< FloatType >::min() /
			static_cast< FloatType >( 2 );
		const FloatType subnormal_four = std::numeric_limits< FloatType >::min() /
			static_cast< FloatType >( 4 );
		const FloatType subnormal_five = std::numeric_limits< FloatType >::min() /
			static_cast< FloatType >( 5 );
		const FloatType subnormal_ten = std::numeric_limits< FloatType >::min() /
			static_cast< FloatType >( 10 );
		const FloatType two = static_cast< FloatType >( 2 );
		if( grb::utils::equals( subnormal_ten, subnormal_four, 2 ) ) {
			std::cerr << subnormal_ten << " should not be equal to " << subnormal_four
				<< " (subnormal I)\n";
			out = 200;
		}
		if( !out && grb::utils::equals( subnormal_four, subnormal_ten, 2 ) ) {
			std::cerr << subnormal_four << " should not be equal to " << subnormal_ten
				<< " (subnormal II)\n";
			out = 210;
		}
		const FloatType subnormal_five_too = subnormal_ten * two;
		if( !out && !grb::utils::equals( subnormal_five_too, subnormal_five, 3 ) ) {
			std::cerr << subnormal_five_too << " should be equal to " << subnormal_ten
				<< " (subnormal III)\n";
			out = 220;
		}
		if( !out && !grb::utils::equals( subnormal_five, subnormal_five_too, 3 ) ) {
			std::cerr << subnormal_ten << " should be equal to " << subnormal_five_too
				<< " (subnormal IV)\n";
			out = 230;
		}
		const FloatType subnormal1p5 = 1.5 * std::numeric_limits< FloatType >::min();
		const FloatType subnormal_two_too = subnormal1p5 -
			std::numeric_limits< FloatType >::min();
		if( !out && !grb::utils::equals( subnormal_two, subnormal_two_too, 3 ) ) {
			std::cerr << subnormal_two << " should be equal to " << subnormal_two_too
				<< " (subnormal V)\n";
			out = 240;
		}
		if( !out && !grb::utils::equals( subnormal_two_too, subnormal_two, 3 ) ) {
			std::cerr << subnormal_two_too << " should be equal to " << subnormal_two
				<< " (subnormal VI)\n";
			out = 250;
		}
	}

	// tests with one zero operand
	if( out == 0 ) {
		const FloatType zero = 0;
		const FloatType eps = std::numeric_limits< FloatType >::epsilon();
		if( grb::utils::equals( zero, eps, 1 ) ) {
			std::cerr << "Absolute tolerance in epsilons failed (I)\n";
			out = 300;
		}
		if( !out && grb::utils::equals( eps, zero, 1 ) ) {
			std::cerr << "Absolute tolerance in epsilons failed (II)\n";
			out = 310;
		}
		if( !out && !grb::utils::equals( zero, eps, 2 ) ) {
			std::cerr << "Absolute tolerance in epsilons failed (III)\n";
			out = 320;
		}
		if( !out && !grb::utils::equals( eps, zero, 2 ) ) {
			std::cerr << "Absolute tolerance in epsilons failed (IV)\n";
			out = 330;
		}
	}

	// test equality under potential overflow conditions
	if( out == 0 ) {
		const FloatType max = std::numeric_limits< FloatType >::max();
		const FloatType max_three = max / static_cast< FloatType >(3);
		const FloatType two_max_three = static_cast< FloatType >(2) * max_three;
		const FloatType max_six = max / static_cast< FloatType >(6);
		const FloatType four_max_six = static_cast< FloatType >(4) * max_six;
		const FloatType four_max_six_or_next = four_max_six == two_max_three
			? nextafter( four_max_six, max )
			: four_max_six;

		// at this point we computed two-thirds of the max floating point number in
		// two different ways. If equality normalises by an average of the two
		// operands then this will overflow-- which of course should not happen.
		// We use nextafter in case the two computations could take place in exact
		// arithmetic.
		if( !grb::utils::equals( two_max_three, four_max_six_or_next, 4 ) ) {
			std::cerr << "Overflow comparison failed (I)\n";
			out = 400;
		}
		if( !out && !grb::utils::equals( four_max_six_or_next, two_max_three, 4 ) ) {
			std::cerr << "Overflow comparison failed (II)\n";
			out = 410;
		}
		if( !out && grb::utils::equals( max_three, four_max_six, 3 ) ) {
			std::cerr << "Overflow comparison failed (III)\n";
			out = 420;
		}
		if( !out && grb::utils::equals( four_max_six, max_three, 3 ) ) {
			std::cerr << "Overflow comparison failed (IV)\n";
			out = 430;
		}
	}

	// scaling
	if( out == 0 ) {
		const FloatType eps = std::numeric_limits< FloatType >::epsilon();
		const FloatType one = static_cast< FloatType >(1);
		for( size_t factor = 1; !out && factor < 17; factor += 2 ) {
			const FloatType onepfac = one + eps * factor;
			const unsigned int id = (factor - 1) / 2;
			assert( id < 10 );
			if( grb::utils::equals( one, onepfac + eps, factor ) ) {
				std::cerr << "Scaling failed (I, step " << factor << ")\n";
				out = 500 + id;
			}
			if( !out && grb::utils::equals( onepfac + eps, one, factor ) ) {
				std::cerr << "Scaling failed (II, step " << factor << ")\n";
				out = 510 + id;
			}
			if( !grb::utils::equals( one, onepfac - eps, factor ) ) {
				std::cerr << "Scaling failed (III, step " << factor << ")\n";
				out = 520 + id;
			}
			if( !out && !grb::utils::equals( onepfac - eps, one, factor ) ) {
				std::cerr << "Scaling failed (IV, step " << factor << ")\n";
				out = 530 + id;
			}
			constexpr FloatType scale = 1337;
			if( !out && grb::utils::equals( scale, scale * (onepfac + eps), factor ) ) {
				std::cerr << "Scaling failed (V, step " << factor << ")\n";
				out = 540 + id;
			}
			if( !out && grb::utils::equals( scale * (onepfac + eps), scale, factor ) ) {
				std::cerr << "Scaling failed (VI, step " << factor << ")\n";
				out = 550 + id;
			}
			if( !grb::utils::equals( scale, scale * (onepfac - eps), factor ) ) {
				std::cerr << "Scaling failed (VII, step " << factor << ")\n";
				out = 560 + id;
			}
			if( !out && !grb::utils::equals( scale * (onepfac - eps), scale, factor ) ) {
				std::cerr << "Scaling failed (VIII, step " << factor << ")\n";
				out = 570 + id;
			}
		}
	}

	// done
	return out;
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	int out = 0;

	// error checking
	if( argc != 1 ) {
		printUsage = true;
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << "\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";

	// do some basic integer tests
	const size_t one = 1;
	const size_t three = 3;
	if( !grb::utils::equals( three, three ) ) {
		std::cerr << "Error during equal integer comparison (I)\n";
		out = 10;
	}
	if( out || grb::utils::equals( one, three ) ) {
		std::cerr << "Error during unequal integer comparison (I)\n";
		out = 20;
	}
	if( out || !grb::utils::equals( one, one ) ) {
		std::cerr << "Error during equal integer comparison (II)\n";
		out = 30;
	}
	if( out || grb::utils::equals( three, one ) ) {
		std::cerr << "Error during unequal integer comparison (II)\n";
		out = 40;
	}

	// do more involved floating point tests, single precision:
	out = out ? out : floatTest< float >();

	// double precision (with error code offset):
	if( out == 0 ) {
		const int dbl_test = floatTest< double >();
		if( dbl_test != 0 ) {
			out = 1000 + dbl_test;
		}
	}

	// done
	if( out != 0 ) {
		std::cout << "Test FAILED" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return out;
}

