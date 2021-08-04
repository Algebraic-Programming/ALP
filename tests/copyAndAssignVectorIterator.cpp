
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

using namespace grb;

void grb_program( const size_t & n, grb::RC & rc ) {
	grb::Vector< double > vector( n );
	rc = grb::set( vector, 1.5 ); // vector = 1.5 everywhere
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}

	// retrieve master iterators
	auto start = vector.cbegin();
	const auto end = vector.cend();

	// test copy-construction
	try {
		auto iterator( start );
		const auto end_copy( end );
		size_t count = 0;
		for( ; iterator != end_copy; ++iterator, ++count ) {
			const auto pair = *iterator;
			if( pair.second != 1.5 ) {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						  << " ), expected value 1.5 after copy-constructing "
							 "iterators\n";
				rc = FAILED;
			}
		}
		grb::collectives<>::allreduce( count, grb::operators::add< size_t >() );
		if( count != n ) {
			std::cerr << "\tunexpected number of nonzeroes " << count << ", expected " << n << " after copy-constructing iterators\n";
			rc = FAILED;
		}
	} catch( ... ) {
		std::cerr << "\tcopy constructor FAILED\n";
		rc = FAILED;
	}

	// test copy-assignment
	{
		typename grb::Vector< double >::const_iterator iterator;
		typename grb::Vector< double >::const_iterator end_copy;
		iterator = start;
		end_copy = end;
		size_t count = 0;
		for( ; iterator != end_copy; ++iterator, ++count ) {
			const auto pair = *iterator;
			if( pair.second != 1.5 ) {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						  << " ), expected value 1.5 after copy-assigning "
							 "iterators\n";
				rc = FAILED;
			}
		}
		grb::collectives<>::allreduce( count, grb::operators::add< size_t >() );
		if( count != n ) {
			std::cerr << "\tunexpected number of nonzeroes " << count << ", expected " << n << " after copy-assigning iterators\n";
			rc = FAILED;
		}
	}

	// test move-constructor
	{
		typename grb::Vector< double >::const_iterator iterator( vector.cbegin() );
		const typename grb::Vector< double >::const_iterator moved_end( vector.cend() );
		size_t count = 0;
		for( ; iterator != moved_end; ++iterator, ++count ) {
			const auto pair = *iterator;
			if( pair.second != 1.5 ) {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						  << " ), expected value 1.5 after move-constructing "
							 "iterators\n";
				rc = FAILED;
			}
		}
		grb::collectives<>::allreduce( count, grb::operators::add< size_t >() );
		if( count != n ) {
			std::cerr << "\tunexpected number of nonzeroes " << count << ", expected " << n << " after move-constructing iterators\n";
			rc = FAILED;
		}
	}

	// test move-assignment
	auto iterator = std::move( start );
	const auto moved_end = std::move( end );
	size_t count = 0;
	for( ; iterator != moved_end; ++iterator, ++count ) {
		const auto pair = *iterator;
		if( pair.second != 1.5 ) {
			std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					  << " ), expected value 1.5 after move-assigning "
						 "iterators\n";
			rc = FAILED;
		}
	}
	grb::collectives<>::allreduce( count, grb::operators::add< size_t >() );
	if( count != n ) {
		std::cerr << "\tunexpected number of nonzeroes " << count << ", expected " << n << " after move-assigning iterators\n";
		rc = FAILED;
	}

	// done
	return;
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
		} else if( ! ss.eof() ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an even integer, the "
					 "test size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cerr << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}
