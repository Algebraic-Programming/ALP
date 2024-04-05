
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
#include <sstream>
#include <iostream>

#include <graphblas.hpp>

#include <graphblas/utils/iterators/adapter.hpp>
#include <graphblas/utils/iterators/regular.hpp>


void grb_program( const size_t &n, grb::RC &rc ) {
	rc = grb::SUCCESS;

	//empty containers
	std::cout << "Subtest 1\n";
	{
		std::vector< double > stl;
		grb::utils::containers::ConstantVector< int > v( 3, 0 );
		grb::utils::containers::Range< double > r( 0, 0, 1, 0 );
		{
			auto it = grb::utils::iterators::make_adapter_iterator(
				stl.begin(), stl.end(),
				[]( const double v ) { return 2*v; }
			);
			auto end = grb::utils::iterators::make_adapter_iterator(
				stl.end(), stl.end(),
				[]( const double v ) { return 2*v; }
			);
			if( it != end ) {
				std::cerr << "Expected empty iterator (I)\n";
				rc = grb::FAILED;
			}
		}
		{
			auto it = grb::utils::iterators::make_adapter_iterator(
				stl.cbegin(), stl.cend(),
				[]( const double v ) { return 2*v; }
			);
			auto end = grb::utils::iterators::make_adapter_iterator(
				stl.cend(), stl.cend(),
				[]( const double v ) { return 2*v; }
			);
			if( it != end ) {
				std::cerr << "Expected empty iterator (II)\n";
				rc = grb::FAILED;
			}
		}
		{
			auto it = grb::utils::iterators::make_adapter_iterator(
				v.begin(), v.end(),
				[]( const int x ) { return 2*x; }
			);
			auto end = grb::utils::iterators::make_adapter_iterator(
				v.end(), v.end(),
				[]( const int x ) { return 2*x; }
			);
			if( it != end ) {
				std::cerr << "Expected empty iterator (III)\n";
				rc = grb::FAILED;
			}
		}
		{
			auto it = grb::utils::iterators::make_adapter_iterator(
				v.cbegin(), v.cend(),
				[]( const int x ) { return 2*x; }
			);
			auto end = grb::utils::iterators::make_adapter_iterator(
				v.cend(), v.cend(),
				[]( const int x ) { return 2*x; }
			);
			if( it != end ) {
				std::cerr << "Expected empty iterator (IV)\n";
				rc = grb::FAILED;
			}
		}
		{
			auto it = grb::utils::iterators::make_adapter_iterator(
				r.begin(), r.end(),
				[]( const double x ) { return 2*x; }
			);
			auto end = grb::utils::iterators::make_adapter_iterator(
				r.end(), r.end(),
				[]( const double x ) { return 2*x; }
			);
			if( it != end ) {
				std::cerr << "Expected empty iterator (V)\n";
				rc = grb::FAILED;
			}
		}
		{
			auto it = grb::utils::iterators::make_adapter_iterator(
				r.cbegin(), r.cend(),
				[]( const double x ) { return 2*x; }
			);
			auto end = grb::utils::iterators::make_adapter_iterator(
				r.cend(), r.cend(),
				[]( const double x ) { return 2*x; }
			);
			if( it != end ) {
				std::cerr << "Expected empty iterator (VI)\n";
				rc = grb::FAILED;
			}
		}
	}

	// test adapter over non-empty containers
	std::cout << "Subtest 2\n";
	{
		std::vector< size_t > stl( n, 7 );
		grb::utils::containers::ConstantVector< size_t > v( 7, n );
		grb::utils::containers::Range< size_t > r( 0, n );
		{
			auto it = grb::utils::iterators::make_adapter_iterator(
				stl.cbegin(), stl.cend(),
				[]( const size_t x ) { return 2*x; }
			);
			const auto end = grb::utils::iterators::make_adapter_iterator(
				stl.cend(), stl.cend(),
				[]( const size_t x ) { return 2*x; }
			);
			size_t count = 0;
			for( ; it != end; ++it ) {
				(void) ++count;
				if( *it != 14 ) {
					std::cerr << "Expected value 14, not " << (*it) << " (I).\n";
					rc = grb::FAILED;
				}
			}
			if( count != n ) {
				std::cerr << "Expected " << n << " entries, not " << count << " (I).\n";
				rc = grb::FAILED;
			}
		}
		{
			auto it = grb::utils::iterators::make_adapter_iterator(
				v.cbegin(), v.cend(),
				[]( const size_t x ) { return x/2+1; }
			);
			const auto end = grb::utils::iterators::make_adapter_iterator(
				v.cend(), v.cend(),
				[]( const size_t x ) { return x/2+1; }
			);
			size_t count = 0;
			for( ; it != end; ++it ) {
				(void) ++count;
				if( *(it.operator->()) != 4 ) {
					std::cerr << "Expected value 4, not " << (*it) << " (II).\n";
					rc = grb::FAILED;
				}
			}
			if( count != n ) {
				std::cerr << "Expected " << n << " entries, not " << count << " (II).\n";
				rc = grb::FAILED;
			}
		}
		{
			auto it = grb::utils::iterators::make_adapter_iterator(
				r.cbegin(), r.cend(),
				[]( const size_t x ) { return 3*x; }
			);
			const auto end = grb::utils::iterators::make_adapter_iterator(
				r.cend(), r.cend(),
				[]( const size_t x ) -> size_t { return x*3; }
			);
			size_t count = 0;
			for( ; it != end; ++it ) {
				if( *it != count * 3 ) {
					std::cerr << "Expected value " << (count * 3) << ", "
						<< "not " << (*it) << " (III).\n";
					rc = grb::FAILED;
				}
				(void) ++count;
			}
			if( count != n ) {
				std::cerr << "Expected " << n << " entries, not " << count << " (III).\n";
				rc = grb::FAILED;
			}
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
		if( !(ss >> read) ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( !ss.eof() ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( read == 0 ) {
			std::cerr << "Given value for n is zero\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): the test size, must be "
			<< "larger than zero.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< grb::AUTOMATIC > launcher;
	grb::RC out;
	const grb::RC launch_rc = launcher.exec( &grb_program, in, out, true );
	if( launch_rc != grb::SUCCESS ) {
		std::cerr << "Launch test failed\n";
		out = launch_rc;
	}
	if( out != grb::SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED (" << grb::toString( out ) << ")\n" << std::endl;
	} else {
		std::cout << "Test OK\n" << std::endl;
	}
	return 0;
}

