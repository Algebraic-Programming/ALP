
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

#include <graphblas.hpp>

#include <graphblas/utils/iterators/regular.hpp>


void grb_program( const size_t &n, grb::RC &rc ) {
	rc = grb::SUCCESS;

	//empty containers
	std::cout << "Subtest 1\n";
	{
		grb::utils::containers::ConstantVector< int > v( 3, 0 );
		if( v.begin() != v.end() ) {
			std::cerr << "Expected empty constant vector (I)\n";
			rc = grb::FAILED;
		}
		if( v.cbegin() != v.cend() ) {
			std::cerr << "Expected empty constant vector (II)\n";
			rc = grb::FAILED;
		}
		for( const int &x : v ) {
			std::cerr << "Expected empty constant vector, but found " << x << "\n";
			rc = grb::FAILED;
		}
	}
	std::cout << "Subtest 2\n";
	{
		grb::utils::containers::Range< double > r( 0, 0, 1, 0 );
		if( r.begin() != r.end() ) {
			std::cerr << "Expected empty range (I)\n";
			rc = grb::FAILED;
		}
		if( r.cbegin() != r.cend() ) {
			std::cerr << "Expected empty range (II)\n";
			rc = grb::FAILED;
		}
		for( const double &x : r ) {
			std::cerr << "Expected empty range, but found " << x << "\n";
			rc = grb::FAILED;
		}
	}
	std::cout << "Subtest 3\n";
	{
		grb::utils::containers::ConstantVector< int > v( 3, 0 );
		if( v.begin( 0, 3 ) != v.end( 0, 3 ) ) {
			std::cerr << "Expected empty constant vector (I/I)\n";
			rc = grb::FAILED;
		}
		if( v.begin( 1, 3 ) != v.end( 1, 3 ) ) {
			std::cerr << "Expected empty constant vector (I/II)\n";
			rc = grb::FAILED;
		}
		if( v.begin( 2, 3 ) != v.end( 2, 3 ) ) {
			std::cerr << "Expected empty constant vector (I/III)\n";
			rc = grb::FAILED;
		}
		if( v.cbegin( 0, 3 ) != v.cend( 0, 3 ) ) {
			std::cerr << "Expected empty constant vector (II/I)\n";
			rc = grb::FAILED;
		}
		if( v.cbegin( 1, 3 ) != v.cend( 1, 3 ) ) {
			std::cerr << "Expected empty constant vector (II/II)\n";
			rc = grb::FAILED;
		}
		if( v.cbegin( 2, 3 ) != v.cend( 2, 3 ) ) {
			std::cerr << "Expected empty constant vector (II/III)\n";
			rc = grb::FAILED;
		}
	}
	std::cout << "Subtest 4\n";
	{
		grb::utils::containers::Range< double > r( 0, 0, 1, 0 );
		if( r.begin( 0, 3 ) != r.end( 0, 3 ) ) {
			std::cerr << "Expected empty range (I/I)\n";
			rc = grb::FAILED;
		}
		if( r.begin( 1, 3 ) != r.end( 1, 3 ) ) {
			std::cerr << "Expected empty range (I/II)\n";
			rc = grb::FAILED;
		}
		if( r.begin( 2, 3 ) != r.end( 2, 3 ) ) {
			std::cerr << "Expected empty range (I/III)\n";
			rc = grb::FAILED;
		}
		if( r.cbegin( 0, 3 ) != r.cend( 0, 3 ) ) {
			std::cerr << "Expected empty range (II/I)\n";
			rc = grb::FAILED;
		}
		if( r.cbegin( 1, 3 ) != r.cend( 1, 3 ) ) {
			std::cerr << "Expected empty range (II/II)\n";
			rc = grb::FAILED;
		}
		if( r.cbegin( 2, 3 ) != r.cend( 2, 3 ) ) {
			std::cerr << "Expected empty range (II/III)\n";
			rc = grb::FAILED;
		}
	}

	// non-empty containers, cut in 3 and 4 parts
	std::cout << "Subtest 5\n";
	{
		grb::utils::containers::ConstantVector< unsigned int > v( 7, n );
		{
			auto rit = v.begin();
			auto cit = v.cbegin();
			for( size_t k = 0; k < n; ++k ) {
				if( rit == v.end() ) {
					std::cerr << "Expected an entry at position " << k << ", found none (I)\n";
					rc = grb::FAILED;
				}
				if( cit == v.cend() ) {
					std::cerr << "Expected an entry at position " << k << ", found none (II)\n";
					rc = grb::FAILED;
				}
				if( *rit != 7 ) {
					std::cerr << "Expected value 7, got " << (*rit) << " (I)\n";
					rc = grb::FAILED;
				}
				if( *cit != 7 ) {
					std::cerr << "Expected value 7, got " << (*cit) << " (II)\n";
					rc = grb::FAILED;
				}
			}
		}
		for( unsigned int cut = 3; cut < 5; ++cut ) {
			size_t checkSum = 0;
			for( unsigned int s = 0; s < cut; ++s ) {
				for( auto it = v.begin( s, cut ); it != v.end( s, cut ); ++it ) {
					(void) ++checkSum;
					if( *it != 7 ) {
						std::cerr << "Expected value 7, got " << (*it) << " (I)\n";
						rc = grb::FAILED;
					}
				}
			}
			if( checkSum != n ) {
				std::cerr << "Expected " << n << " elements when iterating over all " << cut
					<< " iterators, found " << checkSum << " elements instead (I)\n";
				rc = grb::FAILED;
			}
			checkSum = 0;
			for( unsigned int s = 0; s < cut; ++s ) {
				for( auto it = v.cbegin( s, cut ); it != v.cend( s, cut ); ++it ) {
					(void) ++checkSum;
					if( *it != 7 ) {
						std::cerr << "Expected value 7, got " << (*it) << " (II)\n";
						rc = grb::FAILED;
					}
				}
			}
			if( checkSum != n ) {
				std::cerr << "Expected " << n << " elements when iterating over all " << cut
					<< " iterators, found " << checkSum << " elements instead (II)\n";
				rc = grb::FAILED;
			}
		}
	}
	std::cout << "Subtest 6\n";
	{
		grb::utils::containers::Range< size_t > r( 0, n );
		{
			auto rit = r.begin();
			auto cit = r.cbegin();
			for( size_t k = 0; k < n; ++k, ++rit, ++cit ) {
				if( rit == r.end() ) {
					std::cerr << "Expected an entry at position " << k << ", found none (I)\n";
					rc = grb::FAILED;
				}
				if( cit == r.cend() ) {
					std::cerr << "Expected an entry at position " << k << ", found none (II)\n";
					rc = grb::FAILED;
				}
				if( *rit != k ) {
					std::cerr << "Expected value " << k << ", "
						<< "got " << (*rit) << " (I)\n";
					rc = grb::FAILED;
				}
				if( *cit != k ) {
					std::cerr << "Expected value " << k << ", "
						<< "got " << (*cit) << " (II)\n";
					rc = grb::FAILED;
				}
			}
		}
		for( unsigned int cut = 3; cut < 5; ++cut ) {
			size_t checkSum = 0;
			size_t curVal = 0;
			for( unsigned int s = 0; s < cut; ++s ) {
				for( auto it = r.begin( s, cut ); it != r.end( s, cut ); ++it ) {
					(void) ++checkSum;
					if( *it != curVal ) {
						std::cerr << "Expected value " << curVal << ", "
							<< "got " << (*it) << " (I)\n";
						rc = grb::FAILED;
					}
					(void) ++curVal;
				}
			}
			if( checkSum != n ) {
				std::cerr << "Expected " << n << " elements when iterating over all " << cut
					<< " iterators, found " << checkSum << " elements instead (I)\n";
				rc = grb::FAILED;
			}
			checkSum = curVal = 0;
			for( unsigned int s = 0; s < cut; ++s ) {
				for( auto it = r.cbegin( s, cut ); it != r.cend( s, cut ); ++it ) {
					(void) ++checkSum;
					if( *it != curVal ) {
						std::cerr << "Expected value " << curVal << ", "
							<< "got " << (*it) << " (II)\n";
						rc = grb::FAILED;
					}
					(void) ++curVal;
				}
			}
			if( checkSum != n ) {
				std::cerr << "Expected " << n << " elements when iterating over all " << cut
					<< " iterators, found " << checkSum << " elements instead (II)\n";
				rc = grb::FAILED;
			}
		}
	}

	// non-empty containers including which a non-trivial range, cut in 2 parts,
	// exactly divisable
	std::cout << "Subtest 7\n";
	{
		grb::utils::containers::ConstantVector< long > v( -4, 2 * n );
		{
			auto rit = v.begin();
			auto cit = v.cbegin();
			for( size_t k = 0; k < 2 * n; ++k ) {
				if( rit == v.end() ) {
					std::cerr << "Expected an entry at position " << k << ", found none (I)\n";
					rc = grb::FAILED;
				}
				if( cit == v.cend() ) {
					std::cerr << "Expected an entry at position " << k << ", found none (II)\n";
					rc = grb::FAILED;
				}
				if( *rit != -4 ) {
					std::cerr << "Expected value -4, got " << (*rit) << " (I)\n";
					rc = grb::FAILED;
				}
				if( *cit != -4 ) {
					std::cerr << "Expected value -4, got " << (*cit) << " (II)\n";
					rc = grb::FAILED;
				}
			}
		}
		unsigned int cut = 2;
		size_t checkSum = 0;
		for( unsigned int s = 0; s < cut; ++s ) {
			for( auto it = v.begin( s, cut ); it != v.end( s, cut ); ++it ) {
				(void) ++checkSum;
				if( *it != -4 ) {
					std::cerr << "Expected value -4, got " << (*it) << " (I)\n";
					rc = grb::FAILED;
				}
			}
		}
		if( checkSum != 2 * n ) {
			std::cerr << "Expected " << (2*n) << " elements when iterating over all "
				<< cut << " iterators, found " << checkSum << " elements instead (I)\n";
			rc = grb::FAILED;
		}
		checkSum = 0;
		for( unsigned int s = 0; s < cut; ++s ) {
			for( auto it = v.cbegin( s, cut ); it != v.cend( s, cut ); ++it ) {
				(void) ++checkSum;
				if( *it != -4 ) {
					std::cerr << "Expected value -4, got " << (*it) << " (II)\n";
					rc = grb::FAILED;
				}
			}
		}
		if( checkSum != 2 * n ) {
			std::cerr << "Expected " << (2*n) << " elements when iterating over all "
				<< cut << " iterators, found " << checkSum << " elements instead (II)\n";
			rc = grb::FAILED;
		}
	}
	std::cout << "Subtest 8\n";
	{
		grb::utils::containers::Range< size_t > r( 1, 2 * n, 7, 2 );
		size_t items = 0;
		{
			size_t curVal = 1;
			bool rep = true;
			for( const size_t &x : r ) {
				(void) ++items;
				if( x != curVal ) {
					std::cerr << "Expected value " << curVal << ", "
						<< "got " << x << " (I)\n";
					rc = grb::FAILED;
				}
				if( !rep ) {
					curVal += 7;
					if( curVal > 2 * n ) {
						curVal = 1;
					}
				}
				rep = !rep;
			}
		}
		unsigned int cut = 2;
		size_t checkSum = 0;
		size_t curVal = 1;
		bool rep = true;
		for( unsigned int s = 0; s < cut; ++s ) {
			for( auto it = r.begin( s, cut ); it != r.end( s, cut ); ++it ) {
				(void) ++checkSum;
				if( *it != curVal ) {
					std::cerr << "Expected value " << curVal << ", "
						<< "got " << (*it) << " (II)\n";
					rc = grb::FAILED;
				}
				if( !rep ) {
					curVal += 7;
					if( curVal > 2 * n ) {
						curVal = 1;
					}
				}
				rep = !rep;
			}
		}
		if( checkSum != items ) {
			std::cerr << "Expected " << items << " elements when iterating over all "
				<< cut << " iterators, found " << checkSum << " elements instead (I)\n";
			rc = grb::FAILED;
		}
		checkSum = 0;
		curVal = 1;
		rep = true;
		for( unsigned int s = 0; s < cut; ++s ) {
			for( auto it = r.cbegin( s, cut ); it != r.cend( s, cut ); ++it ) {
				(void) ++checkSum;
				if( *it != curVal ) {
					std::cerr << "Expected value " << curVal << ", "
						<< "got " << (*it) << " (III)\n";
					rc = grb::FAILED;
				}
				if( !rep ) {
					curVal += 7;
					if( curVal > 2 * n ) {
						curVal = 1;
					}
				}
				rep = !rep;
			}
		}
		if( checkSum != items ) {
			std::cerr << "Expected " << items << " elements when iterating over all "
				<< cut << " iterators, found " << checkSum << " elements instead (II)\n";
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

