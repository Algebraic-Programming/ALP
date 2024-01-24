
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

#include <vector>
#include <numeric>
#include <iostream>

#include <graphblas.hpp>
#include <graphblas/utils/iterators/filter.hpp>


template< typename FwdIt >
void testOneOut(
	const size_t &n, const size_t filtered, const size_t expected,
	FwdIt &it, const FwdIt &end,
	grb::RC &rc
) {
	rc = grb::SUCCESS;
	size_t count = 0;
	for( ; it != end; ++it, ++count ) {
		if( *it != *(it.operator->()) ) {
			std::cerr << "Error: dereference operator returns " << *it << ", "
				<< "but dereferencing the pointer operator returns " << (*(it.operator->()))
				<< "\n";
			rc = grb::FAILED;
			return;
		}
		if( *it == filtered ) {
			std::cerr << "Error: found " << filtered << ", "
				<< "which should have been filtered out\n";
			rc = grb::FAILED;
			return;
		}
	}
	if( count != expected ) {
		std::cerr << "Error: count should be " << n << ", "
			<< "but the filtered iterator returned " << count << " elements instead\n";
		rc = grb::FAILED;
	}
}

template< typename FwdIt >
void testOnlyOne(
	const size_t &n, const size_t retained, const size_t expected,
	FwdIt &it, const FwdIt &end,
	grb::RC &rc
) {
	rc = grb::SUCCESS;
	size_t count = 0;
	for( ; it != end; ++it, ++count ) {
		if( *it != *(it.operator->()) ) {
			std::cerr << "Error: dereference operator returns " << *it << ", "
				<< "but dereferencing the pointer operator returns " << (*(it.operator->()))
				<< "\n";
			rc = grb::FAILED;
			return;
		}
		if( *it != retained ) {
			std::cerr << "Error: found " << *it << ", "
				<< "but only value(s) " << retained << " were expected\n";
			rc = grb::FAILED;
			return;
		}
	}
	if( count != expected ) {
		std::cerr << "Error: count should be " << n << ", "
			<< "but the filtered iterator returned " << count << " elements instead\n";
		rc = grb::FAILED;
	}
}


void grb_program( const size_t &n, grb::RC &rc ) {
	size_t expected;
	grb::RC local_rc;
	rc = grb::SUCCESS;

	// first fill some vector v with a range of numbers from 0 to n-1 (inclusive)
	std::vector< size_t > v( n );
	std::iota( v.begin(), v.end(), 0 );

	// test for filtering out the number 7
	{
		auto first = grb::utils::iterators::make_filtered_iterator(
			v.begin(), v.end(),
			[] (const size_t val) { return val == 7; }
		);
		const auto second = grb::utils::iterators::make_filtered_iterator(
			v.end(), v.end(),
			[] (const size_t val) { return val == 7; }
		);
		if( n < 8 ) {
			expected = n;
		} else {
			expected = n - 1;
		}
		testOneOut( n, 7, expected, first, second, local_rc );
		if( local_rc != grb::SUCCESS ) {
			std::cerr << "Test 1 FAILED\n";
			rc = grb::FAILED;
		}
	}

	// similar test but with one or both iterators copied
	{
		auto first = grb::utils::iterators::make_filtered_iterator(
			v.begin(), v.end(),
			[] (const size_t val) { return val == 17; }
		);
		const auto second = grb::utils::iterators::make_filtered_iterator(
			v.begin(), v.end(),
			[] (const size_t val) { return val == 17; }
		);
		auto beginCopy = first;
		auto beginCopy2 = first;
		auto endCopy = second;
		if( n < 18 ) {
			expected = n;
		} else {
			expected = n - 1;
		}
		testOneOut( n, 17, expected, beginCopy, second, local_rc );
		if( local_rc != grb::SUCCESS ) {
			std::cerr << "Test 2 FAILED: " << grb::toString( local_rc ) << "\n";
			rc = grb::FAILED;
		}
		testOneOut( n, 17, expected, beginCopy2, endCopy, local_rc );
		if( local_rc != grb::SUCCESS ) {
			std::cerr << "Test 3 FAILED" << grb::toString( local_rc ) << "\n";
			rc = grb::FAILED;
		}
		testOneOut( n, 17, expected, first, second, local_rc );
		if( local_rc != grb::SUCCESS ) {
			std::cerr << "Test 4 FAILED" << grb::toString( local_rc ) << "\n";
			rc = grb::FAILED;
		}
	}

	// test filtering out anything other than 7
	{
		auto first = grb::utils::iterators::make_filtered_iterator(
			v.begin(), v.end(),
			[] (const size_t val) { return val != 7; }
		);
		const auto second = grb::utils::iterators::make_filtered_iterator(
			v.end(), v.end(),
			[] (const size_t val) { return val != 7; }
		);
		if( n > 7 ) {
			expected = 1;
		} else {
			expected = 0;
		}
		testOnlyOne( n, 7, expected, first, second, local_rc );
		if( local_rc != grb::SUCCESS ) {
			std::cerr << "Test 5 FAILED" << grb::toString( local_rc ) << "\n";
			rc = grb::FAILED;
		}
	}

	// similar test to number 5, but using explicit copy constructed iterators
	// also uses the std::vector const_iterators
	{
		auto first = grb::utils::iterators::make_filtered_iterator(
			v.cbegin(), v.cend(),
			[] (const size_t val) { return val != 7; }
		);
		const auto second = grb::utils::iterators::make_filtered_iterator(
			v.cend(), v.cend(),
			[] (const size_t val) { return val != 7; }
		);
		grb::utils::iterators::IteratorFilter<
			std::vector< size_t >::const_iterator
		> begin( first );
		grb::utils::iterators::IteratorFilter<
			std::vector< size_t >::const_iterator
		> end( second);
		local_rc = grb::SUCCESS;
		if( begin != first ) {
			std::cerr << "Copy of iterator in start position does not equal source\n";
			local_rc = grb::FAILED;
		} else {
			if( !(begin == first) ) {
				std::cerr << "Equality operator behaviour mismatches that of the inequality "
					<< "operator (I)\n";
				local_rc = grb::FAILED;
			}
		}
		if( end != second ) {
			std::cerr << "Copy of iterator in end position does not equal source\n";
			local_rc = grb::FAILED;
		} else {
			if( !(end == second) ) {
				std::cerr << "Equality operator behaviour mismatches that of the inequality "
					<< "operator (II)\n";
				local_rc = grb::FAILED;
			}
		}
		if( local_rc == grb::SUCCESS ) {
			if( n > 7 ) {
				expected = 1;
			} else {
				expected = 0;
			}
			testOnlyOne( n, 7, expected, begin, end, local_rc );
		}
		if( local_rc != grb::SUCCESS ) {
			std::cerr << "Test 6 FAILED" << grb::toString( local_rc ) << "\n";
			rc = grb::FAILED;
		}
	}

	// test filtering with duplicates
	assert( n > 2 );
	v[ n - 1 ] = 0;
	v[ n / 2 ] = 0;

	// filter everything except duplicate entry
	{
		auto first = grb::utils::iterators::make_filtered_iterator(
			v.begin(), v.end(),
			[] (const size_t val) { return val != 0; }
		);
		const auto second = grb::utils::iterators::make_filtered_iterator(
			v.end(), v.end(),
			[] (const size_t val) { return val != 0; }
		);
		expected = 3;
		testOnlyOne( n, 0, expected, first, second, local_rc );
		if( local_rc != grb::SUCCESS ) {
			std::cerr << "Test 7 FAILED" << grb::toString( local_rc ) << "\n";
			rc = grb::FAILED;
		}
	}

	// filter all duplicates
	{
		auto first = grb::utils::iterators::make_filtered_iterator(
			v.begin(), v.end(),
			[] (const size_t val) { return val == 0; }
		);
		const auto second = grb::utils::iterators::make_filtered_iterator(
			v.end(), v.end(),
			[] (const size_t val) { return val == 0; }
		);
		expected = n - 3;
		testOneOut( n, 0, expected, first, second, local_rc );
		if( local_rc != grb::SUCCESS ) {
			std::cerr << "Test 8 FAILED" << grb::toString( local_rc ) << "\n";
			rc = grb::FAILED;
		}
	}

	// same test but using move-assigned iterators
	{
		auto first = grb::utils::iterators::make_filtered_iterator(
			v.begin(), v.end(),
			[] (const size_t val) { return val == 0; }
		);
		const auto second = grb::utils::iterators::make_filtered_iterator(
			v.end(), v.end(),
			[] (const size_t val) { return val == 0; }
		);
		auto begin = std::move( first );
		auto end = std::move( second );
		expected = n - 3;
		testOneOut( n, 0, expected, begin, end, local_rc );
		if( local_rc != grb::SUCCESS ) {
			std::cerr << "Test 9 FAILED" << grb::toString( local_rc ) << "\n";
			rc = grb::FAILED;
		}
	}

	// same test as number 6, but then using move constructor
	{
		auto first = grb::utils::iterators::make_filtered_iterator(
			v.begin(), v.end(),
			[] (const size_t val) { return val != 0; }
		);
		const auto second = grb::utils::iterators::make_filtered_iterator(
			v.end(), v.end(),
			[] (const size_t val) { return val != 0; }
		);
		grb::utils::iterators::IteratorFilter<
			std::vector< size_t >::iterator
		> begin( std::move( first ) );
		grb::utils::iterators::IteratorFilter<
			std::vector< size_t >::iterator
		> end( std::move( second ) );
		expected = 3;
		local_rc = grb::SUCCESS;
		if( begin == end ) {
			std::cerr << "Begin iterator matches end iterator, while it should iterate "
				<< "over 3 elements\n";
			local_rc = grb::FAILED;
		} else if( !(begin != end) ) {
			std::cerr << "Equality operator behaviour mismatches that of the inequality "
				<< "operator (III)\n";
			local_rc = grb::FAILED;
		}
		if( local_rc == grb::SUCCESS ) {
			testOnlyOne( n, 0, expected, begin, end, local_rc );
		}
		if( local_rc != grb::SUCCESS ) {
			std::cerr << "Test 10 FAILED\n";
			rc = grb::FAILED;
		}
	}

	// tests mixture of prefix and postfix increments
	{
		auto first = grb::utils::iterators::make_filtered_iterator(
			v.begin(), v.end(),
			[] (const size_t val) { return val != 0; }
		);
		const auto second = grb::utils::iterators::make_filtered_iterator(
			v.end(), v.end(),
			[] (const size_t val) { return val != 0; }
		);
		expected = 3;
		local_rc = grb::SUCCESS;
		if( first == second ) {
			std::cerr << "Expected three elements, got zero\n";
			local_rc = grb::FAILED;
		}
		if( local_rc == grb::SUCCESS ) {
			auto begin = first++;
			testOnlyOne( n, 0, expected, begin, second, local_rc );
		}
		if( local_rc != grb::SUCCESS ) {
			std::cerr << "Same test as done under no. 6 but after calling postfix "
				<< "increment operator failed. The postfix operator is *not* tested "
				<< "further\n";
		} else {
			if( first == second ) {
				std::cerr << "Expected three elements, got one\n";
				local_rc = grb::FAILED;
			} else {
				size_t count = 1;
				while( first != second ) {
					(void) first++;
					(void) ++count;
				}
				if( count != expected ) {
					std::cerr << "Expected " << expected << " elements, "
						<< "got " << count << " instead.\n";
					local_rc = grb::FAILED;
				}
			}
		}
		if( local_rc != grb::SUCCESS ) {
			std::cerr << "Test 11 FAILED" << grb::toString( local_rc ) << "\n";
			rc = grb::FAILED;
		}
	}

	// done
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 100;

	// error checking
	if( argc > 2 ) {
		printUsage = true;
	}
	if( argc == 2 ) {
		size_t read;
		std::istringstream ss( argv[ 1 ] );
		if( ! ( ss >> read ) ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( !ss.eof() ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( read < 3 ) {
			std::cerr << "Given value for n is smaller than 3\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an integer larger than 2.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< grb::AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, in, out, true ) != grb::SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != grb::SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED (" << grb::toString( out ) << ")\n" << std::endl;
	} else {
		std::cout << "Test OK\n" << std::endl;
	}
	return 0;
}

