
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

#include <graphblas/utils/parser.hpp>

#include <graphblas.hpp>


using namespace grb;

constexpr const size_t MAX_FN_LENGTH = 500;
static_assert( MAX_FN_LENGTH > 0, "MAX_FN_LENGTH must be larger than 0" );

struct Input {
	char filename[ MAX_FN_LENGTH + 1 ];
	bool indirect;
};

void grb_program( const Input &in, grb::RC &rc ) {
	// read input file and basic checks
	grb::utils::MatrixFileReader< double > matrixFile( in.filename, in.indirect );
	const size_t m = matrixFile.m();
	const size_t n = matrixFile.n();
	if( m != n ) {
		std::cerr << "\n test only works for square input matrices; FAILED\n";
		rc = FAILED;
		return;
	}

	// create check vectors
	double one[ n ], two[ n ], three[ n ], four[ n ];
	size_t maxAccum[ n ];
	for( size_t i = 0; i < n; ++i ) {
		one[ i ] = 1.5;
		maxAccum[ i ] = 0;
	}
	for( const auto &nz : matrixFile ) {
		const auto &coordinates = nz.first;
		const double &value = nz.second;
		if( coordinates.first % 2 == 0 && coordinates.second % 2 == 1 ) {
			assert( one[ coordinates.second ] == 1.5 );
			one[ coordinates.first ] += 1.5 * value;
			(void) ++(maxAccum[ coordinates.first ]);
		}
	}
	size_t maxAccumOne = 0;
	for( size_t i = 0; i < n; ++i ) {
		two[ i ] = one[ i ];
		if( maxAccum[ i ] > maxAccumOne ) {
			maxAccumOne = maxAccum[ i ];
		}
		maxAccum[ i ] = 0;
	}
	maxAccumOne *= 2;

	for( const auto &nz : matrixFile ) {
		const auto &coordinates = nz.first;
		const double &value = nz.second;
		if( coordinates.first % 2 == 1 && coordinates.second % 2 == 0 ) {
			two[ coordinates.first ] += one[ coordinates.second ] * value;
			(void) ++( maxAccum[coordinates.first] );
		}
	}
	size_t maxAccumTwo = 0;
	for( size_t i = 0; i < n; ++i ) {
		three[ i ] = two[ i ];
		if( maxAccum[ i ] > maxAccumTwo ) {
			maxAccumTwo = maxAccum[ i ];
		}
		maxAccum[ i ] = 0;
	}
	maxAccumTwo = maxAccumOne * maxAccumTwo + maxAccumTwo;

	for( const auto &nz : matrixFile ) {
		const auto &coordinates = nz.first;
		const double &value = nz.second;
		if( coordinates.first % 2 == 0 && coordinates.second % 2 == 1 ) {
			three[ coordinates.second ] += two[ coordinates.first ] * value;
			(void) ++(maxAccum[ coordinates.second ]);
		}
	}
	size_t maxAccumThree = 0;
	for( size_t i = 0; i < n; ++i ) {
		four[ i ] = three[ i ];
		if( maxAccum[ i ] > maxAccumThree ) {
			maxAccumThree = maxAccum[ i ];
		}
		maxAccum[ i ] = 0;
	}
	maxAccumThree = maxAccumTwo * maxAccumThree + maxAccumThree;

	for( const auto &nz : matrixFile ) {
		const auto &coordinates = nz.first;
		const double &value = nz.second;
		if( coordinates.first % 2 == 1 && coordinates.second % 2 == 0 ) {
			four[ coordinates.second ] += three[ coordinates.first ] * value;
			(void) ++(maxAccum[ coordinates.second ]);
		}
	}
	size_t maxAccumFour = maxAccum[ 0 ];
	for( size_t i = 1; i < n; ++i ) {
		if( maxAccum[ i ] > maxAccumFour ) {
			maxAccumFour = maxAccum[ i ];
		}
	}
	maxAccumFour = maxAccumThree * maxAccumFour + maxAccumFour;

	// create vectors
	grb::Vector< double > vector( n );
	grb::Vector< size_t > temp( n );
	grb::Vector< bool > even_mask( n );
	grb::Vector< bool > odd_mask( n );
	rc = grb::set( vector, 1.5 );
	rc = rc ? rc : grb::set< grb::descriptors::use_index >( temp, 0 );
	rc = rc ? rc : grb::eWiseLambda( [&temp] (const size_t i) {
			if( temp[ i ] % 2 == 0 ) {
				temp[ i ] = 1;
			} else {
				temp[ i ] = 0;
			}
		}, temp );
	rc = rc ? rc : grb::set( even_mask, temp, true );
	rc = rc ? rc : grb::set< grb::descriptors::invert_mask >( odd_mask, temp, true );
	// build matrix
	grb::Matrix< double > matrix( n, n );
	if( rc == SUCCESS ) {
		rc = grb::buildMatrixUnique( matrix, matrixFile.cbegin(), matrixFile.cend(),
			SEQUENTIAL );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t initialisation FAILED\n";
		return;
	}

#ifdef _DEBUG
	std::cout << "\nRB Gauss-Seidel step 1...\n";
#endif

	// do one step of red-black Gauss-Seidel
	rc = grb::mxv< grb::descriptors::safe_overlap >(
		vector, even_mask,
		matrix,
		vector, odd_mask,
		grb::Semiring<
			grb::operators::add< double >, grb::operators::mul< double >,
			grb::identities::zero, grb::identities::one
		>()
	);
	if( rc != SUCCESS ) {
		std::cerr << "\t step 1 of RB Gauss-Seidel FAILED\n";
		return;
	}
	for( const auto &entry : vector ) {
#ifdef _DEBUG
		std::cout << "( " << entry.first << ", " << entry.second << " )\n";
#endif
		if( !grb::utils::equals( entry.second, one[ entry.first ], maxAccumOne ) ) {
			std::cerr << "\t entry ( " << entry.first << ", " << entry.second << " ) "
				<< "does not equal expected value " << one[ entry.first ]
				<< " in step one\n";
			rc = FAILED;
		}
	}
	if( rc == FAILED ) {
		return;
	}

#ifdef _DEBUG
	std::cout << "\nRB Gauss-Seidel step 2...\n";
#endif

	// do second step of red-black Gauss-Seidel
	rc = grb::mxv< grb::descriptors::safe_overlap >(
		vector, odd_mask,
		matrix,
		vector, even_mask,
		grb::Semiring<
			grb::operators::add< double >, grb::operators::mul< double >,
			grb::identities::zero, grb::identities::one
		>()
	);
	if( rc != SUCCESS ) {
		std::cerr << "\t step 2 of RB Gauss-Seidel FAILED\n";
		return;
	}
	for( const auto &entry : vector ) {
#ifdef _DEBUG
		std::cout << "( " << entry.first << ", " << entry.second << " )\n";
#endif
		if( !grb::utils::equals( entry.second, two[ entry.first ], maxAccumTwo ) ) {
			std::cerr << "\t entry ( " << entry.first << ", " << entry.second << " ) "
				<< "does not equal expected value " << two[ entry.first ] << " in step 2\n";
			rc = FAILED;
		}
	}
	if( rc == FAILED ) {
		return;
	}

#ifdef _DEBUG
	std::cout << "\nRB Gauss-Seidel step 3 (on transpose matrix)...\n";
#endif
	rc = grb::vxm< grb::descriptors::safe_overlap >(
		vector, odd_mask,
		vector, even_mask,
		matrix,
		grb::Semiring<
			grb::operators::add< double >, grb::operators::mul< double >,
			grb::identities::zero, grb::identities::one
		>()
	);
	if( rc != SUCCESS ) {
		std::cerr << "\t step 3 of RB Gauss-Seidel FAILED\n";
		return;
	}
	for( const auto &entry : vector ) {
#ifdef _DEBUG
		std::cout << "( " << entry.first << ", " << entry.second << " )\n";
#endif
		if( !grb::utils::equals( entry.second, three[ entry.first ], maxAccumThree ) ) {
			std::cerr << "\t entry ( " << entry.first << ", " << entry.second << " ) "
				<< "does not equal expected value " << three[ entry.first ]
				<< " in step 3\n";
			rc = FAILED;
		}
	}
	if( rc == FAILED ) {
		return;
	}

#ifdef _DEBUG
	std::cout << "\nRB Gauss-Seidel step 4 (on transpose matrix)...\n";
#endif
	rc = grb::vxm< grb::descriptors::safe_overlap >(
		vector, even_mask,
		vector, odd_mask,
		matrix,
		grb::Semiring<
			grb::operators::add< double >, grb::operators::mul< double >,
			grb::identities::zero, grb::identities::one
		>()
	);
	if( rc != SUCCESS ) {
		std::cerr << "\t step 4 of RB Gauss-Seidel FAILED\n";
		return;
	}
	for( const auto &entry : vector ) {
#ifdef _DEBUG
		std::cout << "( " << entry.first << ", " << entry.second << " )\n";
#endif
		if( ! grb::utils::equals( entry.second, four[ entry.first ], maxAccumFour ) ) {
			std::cerr << "\t entry ( " << entry.first << ", " << entry.second << " ) "
				<< "does not equal expected value " << four[ entry.first ]
				<< " in step 4\n";
			std::cerr << "\t\t number of epsilons applied is " << ( 2 * maxAccumFour )
				<< "\n";
			rc = FAILED;
		}
	}
	if( rc == FAILED ) {
		return;
	}

	// done
	return;
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	Input in;
	in.filename[ 0 ] = '\0';
	in.indirect = false;

	// error checking
	if( argc <= 1 || argc > 3 ) {
		printUsage = true;
	}
	if( argc > 1 ) {
		if( strlen( argv[ 1 ] ) > MAX_FN_LENGTH ) {
			std::cerr << "Given file name too long (please use a shorter path or "
				<< "increase MAX_FN_LENGTH)\n";
			printUsage = true;
		}
	}
	if( argc == 3 ) {
		if( strncmp( argv[ 2 ], "indirect", 9 ) == 0 ) {
			in.indirect = true;
		} else {
			std::cerr << "Unrecognised second argument passed\n";
			printUsage = true;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " <file name> [indirect]\n";
		std::cerr << "  -file name: path to a matrix file. Path has a maximum "
					 "size of "
				  << MAX_FN_LENGTH << " chars.\n";
		std::cerr << "  -indirect (optional): required when the input matrix "
					 "has an indirect mapping.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out = SUCCESS;
	(void)strncpy( in.filename, argv[ 1 ], MAX_FN_LENGTH );
	if( launcher.exec( &grb_program, in, out, true ) != SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED (launcher error)" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
		return 255;
	}
	std::cout << "Test OK" << std::endl;
	return 0;
}

