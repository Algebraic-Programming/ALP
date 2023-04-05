
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

#include <array>
#include <iostream>

#include <graphblas.hpp>


struct input {
	bool check;
	std::array< size_t, 3 > values;
};

struct output {
	grb::RC rc;
	std::array< size_t, 3 > IDs;
};

// test grb::getID on vectors
void grb_program1( const struct input &in, struct output &out ) {
	grb::RC &rc = out.rc;
	assert( rc == grb::SUCCESS );
	if( grb::spmd<>::pid() == 0 ) {
		if( in.check ) {
			std::cerr << "\t in vector check, phase 4/4\n";
		} else {
			std::cerr << "\t in initial vector test, phase 1/4\n";
		}
	}

	grb::Vector< std::pair< int, float > > one( 10 );
	grb::Vector< size_t > two( 500 );
	const size_t oneID = grb::getID( one ); out.IDs[ 0 ] = oneID;
	const size_t twoID = grb::getID( two ); out.IDs[ 1 ] = twoID;
	if( oneID == twoID ) {
		std::cerr << "\t two calls to getID on different containers result in the "
			<< "same ID (I)\n";
		rc = grb::FAILED;
		return;
	}
	if( oneID != grb::getID( one ) ) {
		std::cerr << "\t two calls to getID on the same container produce different "
			<< "IDs (I)\n";
		rc = grb::FAILED;
		return;
	}
	if( twoID != grb::getID( two ) ) {
		std::cerr << "\t two calls to getID on the same container produce different "
			<< "IDs (II)\n";
		rc = grb::FAILED;
		return;
	}

	if( in.check ) {
		if( oneID != in.values[ 0 ] ) {
			std::cerr << "\t container ID is not consistent with previous run (IV)\n";
			rc = grb::FAILED;
			return;
		}
		if( twoID != in.values[ 1 ] ) {
			std::cerr << "\t container ID is not consistent with previous run (V)\n";
			rc = grb::FAILED;
			return;
		}
	}

	grb::Vector< size_t > three( two );
	const size_t threeID = grb::getID( three ); out.IDs[ 2 ] = threeID;
	if( threeID != grb::getID( three ) ) {
		std::cerr << "\t two calls to getID on the same container produce different "
			<< "IDs (III): " << threeID << " vs. " << grb::getID( three ) << "\n";
		rc = grb::FAILED;
		return;
	}
	if( oneID == threeID ) {
		std::cerr << "\t two calls to getID on different containers result in the "
			<< "same ID (II)\n";
		rc = grb::FAILED;
		return;
	}
	if( twoID == threeID ) {
		std::cerr << "\t two calls to getID on different containers result in the "
			<< "same ID (III)\n";
		rc = grb::FAILED;
		return;
	}

	if( in.check ) {
		if( threeID != in.values[ 2 ] ) {
			std::cerr << "\t container ID is not consistent with previous run (VI)\n";
			rc = grb::FAILED;
			return;
		}
	}

	std::swap( two, three );
	if( twoID != grb::getID( three ) ) {
		std::cerr << "\t two calls to getID on the same container after an std::swap "
			<< "produce different IDs (I)\n";
		rc = grb::FAILED;
		return;
	}
	if( threeID != grb::getID( two ) ) {
		std::cerr << "\t two calls to getID on the same container after an std::swap "
			<< "produce different IDs (II)\n";
		rc = grb::FAILED;
		return;
	}

}

// test grb::getID on matrices
void grb_program2( const struct input &in, struct output &out ) {
	grb::RC &rc = out.rc;
	assert( rc == grb::SUCCESS );
	if( grb::spmd<>::pid() == 0 ) {
		if( in.check ) {
			std::cerr << "\t in matrix check, phase 3/4\n";
		} else {
			std::cerr << "\t in initial matrix test, phase 2/4\n";
		}
	}

	grb::Matrix< void > one( 5000, 714 );
	grb::Matrix< std::pair< size_t, double > > two( 129, 3343 );
	const size_t oneID = grb::getID( one ); out.IDs[ 0 ] = oneID;
	const size_t twoID = grb::getID( two ); out.IDs[ 1 ] = twoID;
	if( oneID == twoID ) {
		std::cerr << "\t two calls to getID on different containers result in the "
			<< "same ID (IV)\n";
		rc = grb::FAILED;
		return;
	}
	if( oneID != grb::getID( one ) ) {
		std::cerr << "\t two calls to getID on the same container produce different "
			<< "IDs (IV)\n";
		rc = grb::FAILED;
		return;
	}
	if( twoID != grb::getID( two ) ) {
		std::cerr << "\t two calls to getID on the same container produce different "
			<< "IDs (V)\n";
		rc = grb::FAILED;
		return;
	}

	if( in.check ) {
		if( oneID != in.values[ 0 ] ) {
			std::cerr << "\t container ID is not consistent with previous run (I)\n";
			rc = grb::FAILED;
			return;
		}
		if( twoID != in.values[ 1 ] ) {
			std::cerr << "\t container ID is not consistent with previous run (II)\n";
			rc = grb::FAILED;
			return;
		}
	}

	grb::Matrix< std::pair< size_t, double > > three( two );
	const size_t threeID = grb::getID( three ); out.IDs[ 2 ] = threeID;
	if( threeID != grb::getID( three ) ) {
		std::cerr << "\t two calls to getID on the same container produce different "
			<< "IDs (VI)\n";
		rc = grb::FAILED;
		return;
	}
	if( oneID == threeID ) {
		std::cerr << "\t two calls to getID on different containers result in the "
			<< "same ID (V)\n";
		rc = grb::FAILED;
		return;
	}
	if( twoID == threeID ) {
		std::cerr << "\t two calls to getID on different containers result in the "
			<< "same ID (VI)\n";
		rc = grb::FAILED;
		return;
	}

	if( in.check ) {
		if( threeID != in.values[ 2 ] ) {
			std::cerr << "\t container ID is not consistent with previous run (III)\n";
			rc = grb::FAILED;
			return;
		}
	}

	std::swap( two, three );
	if( twoID != grb::getID( three ) ) {
		std::cerr << "\t two calls to getID on the same container after an std::swap "
			<< "produce different IDs (III)\n";
		rc = grb::FAILED;
		return;
	}
	if( threeID != grb::getID( two ) ) {
		std::cerr << "\t two calls to getID on the same container after an std::swap "
			<< "produce different IDs (IV)\n";
		rc = grb::FAILED;
		return;
	}

}

// NOTE:
//  the spec does not promise anything when called on empty containers such as
//  grb::Vector< T > empty_vector( 0 ) or grb::Matrix< T > empty_matrix( 0 ),
//  therefore we cannot unit test the behaviour of grb::getID on such
//  containers.

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;

	// error checking
	if( argc != 1 ) {
		printUsage = true;
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << "\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< grb::AUTOMATIC > launcher;
	struct input in_vector{ false, {0,0,0} };
	struct input in_matrix{ false, {0,0,0} };
	struct output out;
	out.rc = grb::SUCCESS;
	in_vector.check = in_matrix.check = false;

	if( launcher.exec( &grb_program1, in_vector, out, true ) != grb::SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test 1 FAILED (launcher error)" << std::endl;
		return 255;
	}
	if( out.rc != grb::SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test 1 FAILED (" << grb::toString( out.rc ) << ")" << std::endl;
		return 255;
	}
	std::copy( out.IDs.begin(), out.IDs.end(), in_vector.values.begin() );

	assert( out.rc == grb::SUCCESS );
	if( launcher.exec( &grb_program2, in_matrix, out, true ) != grb::SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test 2 FAILED (launcher error)" << std::endl;
		return 255;
	}
	if( out.rc != grb::SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test 2 FAILED (" << grb::toString( out.rc ) << ")" << std::endl;
		return 255;
	}
	std::copy( out.IDs.begin(), out.IDs.end(), in_matrix.values.begin() );

	in_matrix.check = true;
	assert( out.rc == grb::SUCCESS );
	if( launcher.exec( &grb_program2, in_matrix, out, true ) != grb::SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test 3 FAILED (launcher error)" << std::endl;
		return 255;
	}
	if( out.rc != grb::SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test 3 FAILED (" << grb::toString( out.rc ) << ")" << std::endl;
		return 255;
	}

	in_vector.check = true;
	assert( out.rc == grb::SUCCESS );
	if( launcher.exec( &grb_program1, in_vector, out, true ) != grb::SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test 4 FAILED (launcher error)" << std::endl;
		return 255;
	}
	if( out.rc != grb::SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test 4 FAILED (" << grb::toString( out.rc ) << ")" << std::endl;
		return 255;
	}

	std::cout << "Test OK" << std::endl;
	return 0;
}

