
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
	grb::Vector< double > dst( n ), src( n );

	// test set
	rc = grb::set( src, 1.5 ); // src = 1.5 everywhere
	if( rc != SUCCESS ) {
		std::cerr << "\tset-to-value FAILED\n";
	} else {
		if( nnz( src ) != n ) {
			std::cerr << "\t (set-to-value) unexpected number of nonzeroes " << nnz( src ) << ", expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto & pair : src ) {
			if( pair.second != 1.5 ) {
				std::cerr << "\t (set-to-value) unexpected entry ( " << pair.first << ", " << pair.second << " ), expected value 1.5.\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test set-to-index
	rc = grb::set< grb::descriptors::use_index >( dst, 2 );
	if( rc != SUCCESS ) {
		std::cerr << "\tset-to-index FAILED\n";
	} else {
		if( nnz( dst ) != n ) {
			std::cerr << "\t (set-to-index) unexpected number of nonzeroes " << nnz( dst ) << ", expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto & pair : dst ) {
			if( pair.second != pair.first ) {
				std::cerr << "\t (set-to-index) unexpected entry ( " << pair.first << ", " << pair.second << " ), expected value " << static_cast< double >( pair.first ) << ".\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test set overwrite
	rc = grb::set( dst, src );
	if( rc != SUCCESS ) {
		std::cerr << "\t Set-overwrite FAILED with error code " << grb::toString( rc ) << "\n";
	} else {
		if( nnz( dst ) != n ) {
			std::cerr << "\t (set-overwrite) unexpected number of nonzeroes " << nnz( dst ) << ", expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto & pair : dst ) {
			if( pair.second != 1.5 ) {
				std::cerr << "\t (set-overwrite) unexpected entry ( " << pair.first << ", " << pair.second << " ), expected value 1.5\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test set into cleared
	rc = grb::clear( dst );
	if( rc == SUCCESS ) {
		rc = grb::set( dst, src );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t Set-into-cleared FAILED with error code " << grb::toString( rc ) << "\n";
	} else {
		if( nnz( dst ) != n ) {
			std::cerr << "\t (set-into-cleared) unexpected number of nonzeroes " << nnz( dst ) << ", expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto & pair : dst ) {
			if( pair.second != 1.5 ) {
				std::cerr << "\t (set-into-cleared) unexpected entry ( " << pair.first << ", " << pair.second << " ), expected value 1.5\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test masked set
	rc = grb::setElement( src, 0, n / 2 );
	if( rc == SUCCESS ) {
		rc = grb::clear( dst );
	}
	if( rc == SUCCESS ) {
		rc = grb::set( dst, src, src );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t Masked-set FAILED with error code " << grb::toString( rc ) << "\n";
	} else {
		if( nnz( dst ) != n - 1 ) {
			std::cerr << "\t (masked-set) unexpected number of nonzeroes " << nnz( dst ) << ", expected " << ( n - 1 ) << "\n";
			rc = FAILED;
		}
		for( const auto & pair : dst ) {
			if( pair.first != n / 2 && pair.second != 1.5 ) {
				std::cerr << "\t (masked-set) unexpected entry ( " << pair.first << ", " << pair.second << " ), expected value 1.5\n";
				rc = FAILED;
			}
			if( pair.first == n / 2 ) {
				std::cerr << "\t (masked-set) unexpected entry ( " << pair.first << ", " << pair.second << " ), expected no entry at this position\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test inverted-mask set
	rc = grb::clear( dst );
	if( rc == SUCCESS ) {
		rc = grb::set< grb::descriptors::invert_mask >( dst, src, src );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t Inverted-mask set FAILED with error code " << grb::toString( rc ) << "\n";
	} else {
		if( nnz( dst ) != 1 ) {
			std::cerr << "\t (inverted-mask-set) unexpected number of "
						 "nonzeroes "
					  << nnz( dst ) << ", expected 1.\n";
			rc = FAILED;
		}
		for( const auto & pair : dst ) {
			if( pair.first == n / 2 && pair.second != 0 ) {
				std::cerr << "\t (inverted-mask-set) unexpected entry ( " << pair.first << ", " << pair.second << ": expected value 0\n";
				rc = FAILED;
			}
			if( pair.first != n / 2 ) {
				std::cerr << "\t (inverted-mask-set) unexpected entry ( " << pair.first << ", " << pair.second << " ): expected no entry at this position\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test sparse mask set
	rc = grb::clear( dst );
	if( rc == SUCCESS ) {
		rc = grb::clear( src );
	}
	if( rc == SUCCESS ) {
		rc = grb::setElement( src, 1.5, n / 2 );
	}
	if( rc == SUCCESS ) {
		rc = grb::set( dst, src, src );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t Sparse-mask set FAILED with error code " << grb::toString( rc ) << "\n";
	} else {
		if( nnz( dst ) != 1 ) {
			std::cerr << "\t (sparse-mask-set) unexpected number of nonzeroes " << nnz( dst ) << ", expected 1.\n";
			rc = FAILED;
		}
		for( const auto & pair : dst ) {
			if( pair.first == n / 2 && pair.second != 1.5 ) {
				std::cerr << "\t (sparse-mask-set) unexpected entry ( " << pair.first << ", " << pair.second << " ): expected value 1.5\n";
				rc = FAILED;
			}
			if( pair.first != n / 2 ) {
				std::cerr << "\t (sparse-mask-set) unexpected entry ( " << pair.first << ", " << pair.second << " ): expected no entry at this position\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test re-entrant mask set
	rc = grb::clear( src );
	rc = rc ? rc : grb::setElement( src, 1.5, 0 );
	rc = rc ? rc : grb::set( dst, src, src );
	if( rc != SUCCESS ) {
		std::cerr << "\t Sparse-mask set (re-entrance) FAILED with error code " << grb::toString( rc ) << "\n";
	} else {
		if( nnz( dst ) != 1 ) {
			std::cerr << "\t (sparse-mask-set-reentrant) unexpected number of nonzeroes " << nnz( dst ) << ", expected 1.\n";
			rc = FAILED;
		}
		for( const auto &pair : dst ) {
			if( (pair.first == 0 || pair.first == n/2)  && pair.second != 1.5 ) {
				std::cerr << "\t (sparse-mask-set-reentrant) unexpected entry ( " << pair.first << ", " << pair.second << " ): expected value 1.5\n";
				rc = FAILED;
			}
			if( pair.first != 0 && pair.first != n/2 ) {
				std::cerr << "\t (sparse-mask-set-reentrant) unexpected entry ( " << pair.first << ", " << pair.second << " ): expected no entry at this position\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test sparse mask set to scalar
	rc = grb::clear( dst );
	if( rc == SUCCESS ) {
		rc = grb::clear( src );
	}
	if( rc == SUCCESS ) {
		rc = grb::setElement( src, 1.5, n / 2 );
	}
	if( rc == SUCCESS ) {
		rc = grb::set( dst, src, 3.0 );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t Sparse-mask set to scalar FAILED with error code " << grb::toString( rc ) << "\n";
	} else {
		if( nnz( dst ) != 1 ) {
			std::cerr << "\t (sparse-mask-set-scalar) unexpected number of nonzeroes " << nnz( dst ) << ", expected 1.\n";
			rc = FAILED;
		}
		for( const auto & pair : dst ) {
			if( pair.first == n / 2 && pair.second != 3.0 ) {
				std::cerr << "\t (sparse-mask-set-to-scalar) unexpected entry ( " << pair.first << ", " << pair.second << " ): expected value 3.0\n";
				rc = FAILED;
			}
			if( pair.first != n / 2 ) {
				std::cerr << "\t (sparse-mask-set-to-scalar) unexpected entry ( " << pair.first << ", " << pair.second << " ): expected no entry at this position\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test re-entrant mask set to scalar
	rc = grb::clear( src );
	rc = rc ? rc : grb::setElement( src, 1.5, 0 );
	rc = rc ? rc : grb::set( dst, src, 3.0 );
	if( rc != SUCCESS ) {
		std::cerr << "\t Sparse-mask set to scalar (re-entrant) FAILED with error code " << grb::toString( rc ) << "\n";
	} else {
		if( nnz( dst ) != 1 ) {
			std::cerr << "\t (sparse-mask-set-scalar-reentrant) unexpected number of nonzeroes " << nnz( dst ) << ", expected 1.\n";
			rc = FAILED;
		}
		for( const auto &pair : dst ) {
			if( (pair.first == 0 || pair.first == n/2) && pair.second != 3.0 ) {
				std::cerr << "\t (sparse-mask-set-scalar-reentrant) unexpected entry ( " << pair.first << ", " << pair.second << " ): expected value 3.0\n";
				rc = FAILED;
			}
			if( pair.first != 0 && pair.first != n/2 ) {
				std::cerr << "\t (sparse-mask-set-scalar-reentrant) unexpected entry ( " << pair.first << ", " << pair.second << " ): expected no entry at this position\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test sparse inverted mask set to empty
	rc = grb::clear( dst );
	if( rc == SUCCESS ) {
		rc = grb::set< grb::descriptors::invert_mask >( dst, src, src );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t Sparse-inverted-mask set to empty FAILED with error "
					 "code "
				  << grb::toString( rc ) << "\n";
	} else {
		if( nnz( dst ) != 0 ) {
			std::cerr << "\t (sparse-inverted-mask-set-empty) unexpected "
						 "number of nonzeroes "
					  << nnz( dst ) << ", expected 0.\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test sparse inverted mask set
	grb::Vector< bool > mask( n );
	rc = grb::clear( dst );
	if( rc == SUCCESS ) {
		rc = grb::setElement( mask, true, n / 2 );
	}
	if( rc == SUCCESS ) {
		rc = grb::set( src, 1.5 );
	}
	if( rc == SUCCESS ) {
		rc = grb::set< grb::descriptors::invert_mask >( dst, mask, src );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t Sparse inverted-mask set FAILED with error code " << grb::toString( rc ) << "\n";
	} else {
		if( nnz( dst ) != n - 1 ) {
			std::cerr << "\t (sparse-inverted-mask-set) unexpected number of "
						 "nonzeroes "
					  << nnz( dst ) << ", expected " << ( n - 1 ) << ".\n";
			rc = FAILED;
		}
		for( const auto & pair : dst ) {
			if( pair.first == n / 2 ) {
				std::cerr << "\t (sparse-inverted-mask-set) unexpected entry ( " << pair.first << ", " << pair.second << " ): this position should have been empty\n";
				rc = FAILED;
			} else if( pair.second != 1.5 ) {
				std::cerr << "\t (sparse-inverted-mask-set) unexpected entry ( " << pair.first << ", " << pair.second << " ): expected value 1.5.\n";
				rc = FAILED;
			}
		}
	}
	// if( rc != SUCCESS ) { return; }

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
		} else if( read % 2 != 0 ) {
			std::cerr << "Given value for n is odd\n";
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
		std::cout << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}
