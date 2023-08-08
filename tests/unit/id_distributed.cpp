
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

	grb::Vector< std::pair< int, float > > one( 1000000 );
	grb::Vector< size_t > two( 5000000 );
	const size_t oneLocalID = grb::getID( grb::internal::getLocal( one ) );
	out.IDs[ 0 ] = oneLocalID;
	const size_t twoLocalID = grb::getID( grb::internal::getLocal( two ) );
	out.IDs[ 1 ] = twoLocalID;
	if( oneLocalID == twoLocalID ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on different containers result in the "
			<< "same ID (I)\n";
		rc = grb::FAILED;
		return;
	}
	if( oneLocalID != grb::getID( grb::internal::getLocal( one ) ) ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on the same container produce different "
			<< "IDs (I)\n";
		rc = grb::FAILED;
		return;
	}
	if( twoLocalID != grb::getID( grb::internal::getLocal( two ) ) ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on the same container produce different "
			<< "IDs (II)\n";
		rc = grb::FAILED;
		return;
	}

	if( in.check ) {
		if( oneLocalID != in.values[ 0 ] ) {
			std::cerr << "\t container ID is not consistent with previous run (IV)\n";
			rc = grb::FAILED;
			return;
		}
		if( twoLocalID != in.values[ 1 ] ) {
			std::cerr << "\t container ID is not consistent with previous run (V)\n";
			rc = grb::FAILED;
			return;
		}
	}

	grb::Vector< size_t > three( two );
	const size_t threeLocalID = grb::getID( grb::internal::getLocal( three ) );
	out.IDs[ 2 ] = threeLocalID;
	if( threeLocalID != grb::getID( grb::internal::getLocal( three ) ) ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on the same container produce different "
			<< "IDs (III): " << threeLocalID << " vs. " << grb::getID( grb::internal::getLocal( three ) ) << "\n";
		rc = grb::FAILED;
		return;
	}
	if( oneLocalID == threeLocalID ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on different containers result in the "
			<< "same ID (II)\n";
		rc = grb::FAILED;
		return;
	}
	if( twoLocalID == threeLocalID ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on different containers result in the "
			<< "same ID (III)\n";
		rc = grb::FAILED;
		return;
	}

	if( in.check ) {
		if( threeLocalID != in.values[ 2 ] ) {
			std::cerr << "\t container ID is not consistent with previous run (VI): "
				<< threeLocalID << " vs. " << in.values[ 2 ] << "\n";
			rc = grb::FAILED;
			return;
		}
	}

	std::swap( two, three );
	if( twoLocalID != grb::getID( grb::internal::getLocal( three ) ) ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on the same container after an std::swap "
			<< "produce different IDs (I)\n";
		rc = grb::FAILED;
		return;
	}
	if( threeLocalID != grb::getID( grb::internal::getLocal( two ) ) ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on the same container after an std::swap "
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
			std::cerr << "\t in matrix check, phase 4/4\n";
		} else {
			std::cerr << "\t in initial matrix test, phase 1/4\n";
		}
	}

	grb::Matrix< std::pair< int, float > > one( 1000000, 100000 );
	grb::Matrix< size_t > two( 5000000, 100000 );
	const size_t oneLocalID = grb::getID( grb::internal::getLocal( one ) );
	out.IDs[ 0 ] = oneLocalID;
	const size_t twoLocalID = grb::getID( grb::internal::getLocal( two ) );
	out.IDs[ 1 ] = twoLocalID;
	if( oneLocalID == twoLocalID ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on different containers result in the "
			<< "same ID (I)\n";
		rc = grb::FAILED;
		return;
	}
	if( oneLocalID != grb::getID( grb::internal::getLocal( one ) ) ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on the same container produce different "
			<< "IDs (I)\n";
		rc = grb::FAILED;
		return;
	}
	if( twoLocalID != grb::getID( grb::internal::getLocal( two ) ) ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on the same container produce different "
			<< "IDs (II)\n";
		rc = grb::FAILED;
		return;
	}

	if( in.check ) {
		if( oneLocalID != in.values[ 0 ] ) {
			std::cerr << "\t container ID is not consistent with previous run (IV)\n";
			rc = grb::FAILED;
			return;
		}
		if( twoLocalID != in.values[ 1 ] ) {
			std::cerr << "\t container ID is not consistent with previous run (V)\n";
			rc = grb::FAILED;
			return;
		}
	}

	grb::Matrix< size_t > three( two );
	const size_t threeLocalID = grb::getID( grb::internal::getLocal( three ) );
	out.IDs[ 2 ] = threeLocalID;
	if( threeLocalID != grb::getID( grb::internal::getLocal( three ) ) ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on the same container produce different "
			<< "IDs (III): " << threeLocalID << " vs. " << grb::getID( grb::internal::getLocal( three ) ) << "\n";
		rc = grb::FAILED;
		return;
	}
	if( oneLocalID == threeLocalID ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on different containers result in the "
			<< "same ID (II)\n";
		rc = grb::FAILED;
		return;
	}
	if( twoLocalID == threeLocalID ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on different containers result in the "
			<< "same ID (III)\n";
		rc = grb::FAILED;
		return;
	}

	if( in.check ) {
		if( threeLocalID != in.values[ 2 ] ) {
			std::cerr << "\t container ID is not consistent with previous run (VI): "
				<< threeLocalID << " vs. " << in.values[ 2 ] << "\n";
			rc = grb::FAILED;
			return;
		}
	}

	std::swap( two, three );
	if( twoLocalID != grb::getID( grb::internal::getLocal( three ) ) ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on the same container after an std::swap "
			<< "produce different IDs (I)\n";
		rc = grb::FAILED;
		return;
	}
	if( threeLocalID != grb::getID( grb::internal::getLocal( two ) ) ) {
		std::cerr << "\t two calls to getID(getLocal(mat)) on the same container after an std::swap "
			<< "produce different IDs (II)\n";
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

