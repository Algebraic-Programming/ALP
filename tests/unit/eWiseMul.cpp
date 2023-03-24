
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

void grb_program( const size_t &n, RC &rc ) {
	// for the subtests that return ILLEGAL due to incorrect usage of the dense
	// descriptor in the case of nonblocking execution, the ouput vector must be
	// reset in order to cope with side effects
	constexpr bool nonblocking_execution = Properties<>::isNonblockingExecution;

	Semiring<
		operators::add< double >, operators::mul< double >,
		identities::zero, identities::one
	> ring;

	// repeatedly used containers
	Vector< bool > even_mask( n ), odd_mask( n );
	Vector< size_t > temp( n );
	Vector< double > out( n ), left( n ), right( n );

	// create masks
	rc = set< descriptors::use_index >( temp, 0 );
	rc = rc ? rc : eWiseLambda( [&temp] (const size_t i) {
			if( temp[ i ] % 2 == 0 ) {
				temp[ i ] = 1;
			} else {
				temp[ i ] = 0;
			}
		}, temp );
	rc = rc ? rc : set( even_mask, temp, true );
	rc = rc ? rc : set< descriptors::invert_mask >(
		odd_mask, even_mask, true );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "\t initialisation of masks FAILED\n";
		return;
	}

	// test eWiseMul on dense vectors
	std::cout << "Test 1: ";
	rc = rc ? rc : set( out, 0 );
	rc = rc ? rc : set( left, 1 );
	rc = rc ? rc : set( right, 2 );
	rc = rc ? rc : eWiseMul( out, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != n ) {
		std::cerr << "returns " << nnz( out ) << " nonzeroes, "
			<< "expected " << n << "\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.second != 2 ) {
			std::cerr << "\t got ( " << pair.first << ", " << pair.second << " ), "
				<< "expected entries with value 2 only\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// test in-place behaviour of eWiseMul
	std::cout << "\b\b 2: ";
	rc = eWiseMul( out, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != n ) {
		std::cerr << "returns " << nnz( out ) << " nonzeroes, "
			<< "expected " << n << "\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.second != 4 ) {
			std::cerr << "\t got ( " << pair.first << ", " << pair.second << " ), "
				<< "expected entries with value 4 only\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// test in-place with dense descriptor
	std::cout << "\b\b 3: ";
	rc = eWiseMul< descriptors::dense >( out, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != n ) {
		std::cerr << "returns " << nnz( out ) << " nonzeroes, "
			<< "expected " << n << "\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.second != 6 ) {
			std::cerr << "\t got ( " << pair.first << ", " << pair.second << " ), "
				<< "expected entries with value 4 only\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// test illegal with dense descriptor
	std::cout << "\b\b 4: ";
	rc = clear( out );
	rc = rc ? rc : eWiseMul< descriptors::dense >( out, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != ILLEGAL ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected ILLEGAL\n";
		rc = FAILED;
	} else {
		rc = SUCCESS;
	}
	if( rc != SUCCESS ) { return; }
	if( nonblocking_execution ) {
		rc = rc ? rc : clear( out );
		if( rc != SUCCESS ) {
			std::cerr << " unexpected failure of grb::clear( out )\n";
			rc = FAILED;
		}
	}

	std::cout << "\b\b 5: ";
	rc = eWiseMul< descriptors::dense >( left, out, right, ring );
	rc = rc ? rc : wait();
	if( rc != ILLEGAL ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected ILLEGAL\n";
		rc = FAILED;
	} else {
		rc = SUCCESS;
	}
	if( rc != SUCCESS ) { return; }
	if( nonblocking_execution ) {
		rc = rc ? rc : set( left, 1 );
		if( rc != SUCCESS ) {
			std::cerr << " unexpected failure of grb::set( left, 1)\n";
			rc = FAILED;
		}
	}

	std::cout << "\b\b 6: ";
	rc = eWiseMul< descriptors::dense >( left, right, out, ring );
	rc = rc ? rc : wait();
	if( rc != ILLEGAL ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected ILLEGAL\n";
		rc = FAILED;
	} else {
		rc = SUCCESS;
	}
	if( rc != SUCCESS ) { return; }
	if( nonblocking_execution ) {
		rc = rc ? rc : set( left, 1 );
		if( rc != SUCCESS ) {
			std::cerr << " unexpected failure of grb::set( left, 1)\n";
			rc = FAILED;
		}
	}

	std::cout << "\b\b 7: ";
	rc = clear( left );
	rc = rc ? rc : eWiseMul< descriptors::dense >( right, left, out, ring );
	rc = rc ? rc : wait();
	if( rc != ILLEGAL ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected ILLEGAL\n";
		rc = FAILED;
	} else {
		rc = SUCCESS;
	}
	if( rc != SUCCESS ) { return; }
	if( nonblocking_execution ) {
		rc = rc ? rc : set( right, 2 );
		if( rc != SUCCESS ) {
			std::cerr << " unexpected failure of grb::set( right, 2)\n";
			rc = FAILED;
		}
	}

	std::cout << "\b\b 8: ";
	rc = eWiseMul< descriptors::dense >( left, right, out, ring );
	rc = rc ? rc : wait();
	if( rc != ILLEGAL ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected ILLEGAL\n";
		rc = FAILED;
	} else {
		rc = SUCCESS;
	}
	if( rc != SUCCESS ) { return; }
	if( nonblocking_execution ) {
		rc = rc ? rc : set( left, 1 );
		if( rc != SUCCESS ) {
			std::cerr << " unexpected failure of grb::set( left, 1)\n";
			rc = FAILED;
		}
	}

	std::cout << "\b\b 9: ";
	rc = eWiseMul< descriptors::dense >( left, out, right, ring );
	rc = rc ? rc : wait();
	if( rc != ILLEGAL ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected ILLEGAL\n";
		rc = FAILED;
	} else {
		rc = SUCCESS;
	}
	if( rc != SUCCESS ) { return; }
	if( nonblocking_execution ) {
		rc = rc ? rc : set( left, 1 );
		if( rc != SUCCESS ) {
			std::cerr << " unexpected failure of grb::set( left, 1)\n";
			rc = FAILED;
		}
	}

	// test sparse unmasked
	std::cout << "\b\b 10: ";
	rc = clear( out );
	rc = rc ? rc : clear( left );
	rc = rc ? rc : setElement( left, 3, n / 2 );
	rc = rc ? rc : eWiseMul( out, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != 1 ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected 1\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.first != n / 2 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected no entries at positions other than " << (n/2) << "\n";
			rc = FAILED;
		}
		if( pair.second != 6 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected an entry with value 6 only\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// same test, now testing also for in-place semantics
	std::cout << "\b\b 11: ";
	rc = eWiseMul( out, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != 1 ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected 1\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.first != n / 2 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected no entries at positions other than " << (n/2) << "\n";
			rc = FAILED;
		}
		if( pair.second != 12 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected an entry with value 12 only\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// test sparse unmasked, other side
	std::cout << "\b\b 12: ";
	std::swap( left, right );
	rc = clear( out );
	rc = rc ? rc : eWiseMul( out, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != 1 ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected 1\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.first != n / 2 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected no entries at positions other than " << (n/2) << "\n";
			rc = FAILED;
		}
		if( pair.second != 6 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected an entry with value 6 only\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// same test, now testing also for in-place semantics
	std::cout << "\b\b 13: ";
	rc = eWiseMul( out, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != 1 ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected 1\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.first != n / 2 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected no entries at positions other than " << (n/2) << "\n";
			rc = FAILED;
		}
		if( pair.second != 12 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected an entry with value 12 only\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// sparse masked test
	std::cout << "\b\b 14: ";
	rc = clear( out );
	rc = rc ? rc : eWiseMul( out, even_mask, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	const bool halfLengthIsOdd = (n/2) % 2 == 1;
	if( halfLengthIsOdd ) {
		if( nnz( out ) != 0 ) {
			std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
				<< "expected 0\n";
			rc = FAILED;
		}
	} else {
		if( nnz( out ) != 1 ) {
			std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
				<< "expected 1\n";
			rc = FAILED;
		}
	}
	for( const auto &pair : out ) {
		if( halfLengthIsOdd ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected no entries\n";
			rc = FAILED;
		} else {
			if( pair.first != n / 2 ) {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected no entries at positions other than " << (n/2) << "\n";
				rc = FAILED;
			}
			if( pair.second != 6 ) {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected an entry with value 6 only\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) { return; }

	// same test, possibly also checking for in-place semantics
	std::cout << "\b\b 15: ";
	rc = eWiseMul( out, odd_mask, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != 1 ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected 1\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.first != n / 2 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected no entries at positions other than " << (n/2) << "\n";
			rc = FAILED;
		}
		if( pair.second != 6 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected an entry with value 6 only\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// sparse masked test, switch sides and mask order
	std::cout << "\b\b 16: ";
	std::swap( left, right );
	rc = clear( out );
	rc = rc ? rc : eWiseMul( out, odd_mask, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( halfLengthIsOdd ) {
		if( nnz( out ) != 1 ) {
			std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
				<< "expected 1\n";
			rc = FAILED;
		}
	} else {
		if( nnz( out ) != 0 ) {
			std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
				<< "expected 0\n";
			rc = FAILED;
		}
	}
	for( const auto &pair : out ) {
		if( halfLengthIsOdd ) {
			if( pair.first != n / 2 ) {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected no entries at positions other than " << (n/2) << "\n";
				rc = FAILED;
			}
			if( pair.second != 6 ) {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected an entry with value 6 only\n";
				rc = FAILED;
			}
		} else {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected no entries\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// same test, possibly also checking for in-place semantics
	std::cout << "\b\b 17: ";
	rc = eWiseMul( out, even_mask, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != 1 ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected 1\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.first != n / 2 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected no entries at positions other than " << (n/2) << "\n";
			rc = FAILED;
		}
		if( pair.second != 6 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected an entry with value 6 only\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// masked dense test
	std::cout << "\b\b 18: ";
	rc = grb::set( left, 3.0 );
	rc = rc ? rc : grb::set( right, 2.0 );
	rc = rc ? rc : eWiseMul( out, odd_mask, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( halfLengthIsOdd ) {
		if( nnz( out ) != n / 2 ) {
			std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
				<< "expected " << (n/2) << "\n";
			rc = FAILED;
		}
	} else {
		if( nnz( out ) != n / 2 + 1 ) {
			std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
				<< "expected " << (n/2+1) << "\n";
			rc = FAILED;
		}
	}
	for( const auto &pair : out ) {
		if( pair.first % 2 == 1 && pair.first != n / 2 && pair.second != 6 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected entry with value 6 here\n";
			rc = FAILED;
		}
		if( pair.first % 2 == 1 && pair.first == n / 2 && pair.second != 12 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected entries with value 12 at this position\n";
			rc = FAILED;
		}
		if( pair.first % 2 == 0 ) {
			if( pair.first == n / 2 ) {
				if( pair.second != 6 ) {
					std::cerr << "primitive returns an entry ( "
						<< pair.first << ", " << pair.second << " ), "
						<< "expected entries with value 6 at this position\n";
					rc = FAILED;
				}
			} else {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected no entry at this position\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) { return; }

	// now use complementary mask to generate a dense vector
	std::cout << "\b\b 19: ";
	rc = eWiseMul( out, even_mask, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != n ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected " << n << "\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.first != n / 2 && pair.second != 6 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected entry with value 6 here\n";
			rc = FAILED;
		}
		if( pair.first == n / 2 && pair.second != 12 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected entry with value 12 here\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// test scalar on the right, sparse, unmasked
	std::cout << "\b\b 20: ";
	const double scalar = 2.0;
	rc = clear( out );
	rc = rc ? rc : clear( left );
	rc = rc ? rc : setElement( left, 3.0, n / 2 );
	rc = rc ? rc : eWiseMul( out, left, scalar, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != 1 ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected 1\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.first != n / 2 && pair.second != 6 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected a single entry with value 6 at position " << (n/2) << "\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// test scalar on the right, dense, unmasked
	std::cout << "\b\b 21: ";
	rc = set( right, 2.0 );
	rc = rc ? rc : eWiseMul( out, right, scalar, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != n ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected " << n << "\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.first == n / 2 ) {
			if( pair.second != 10 ) {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected value 10 at this position\n";
				rc = FAILED;
			}
		} else if( pair.second != 4 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected value 4 at this position\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// test scalar on the left, sparse, unmasked
	std::cout << "\b\b 22: ";
	rc = eWiseMul( out, scalar, left, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != n ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected " << n << "\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.first == n / 2 ) {
		       if( pair.second != 16 ) {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected an entry with value 16 at this position\n";
				rc = FAILED;
		       }
		} else if( pair.second != 4 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected value 4 at this position\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// test scalar on the left, dense, unmasked
	std::cout << "\b\b 23: ";
	rc = eWiseMul( out, scalar, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != n ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected " << n << "\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.first == n / 2 ) {
			if( pair.second != 20 ) {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected an entry with value 20 at this position\n";
				rc = FAILED;
			}
		} else if( pair.second != 8 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected value 8 at this position\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// test scalars on both sides, unmasked and masked
	{
		std::cout << "\b\b 24: ";
		double anotherScalar = 3.0;
		rc = clear( out );
		rc = rc ? rc : eWiseMul( out, scalar, anotherScalar, ring );
		rc = rc ? rc : wait();
		if( rc != SUCCESS ) {
			std::cerr << "primitive returns " << toString( rc ) << ", "
				<< "expected SUCCESS\n";
			rc = FAILED;
			return;
		}
		if( nnz( out ) != n ) {
			std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
				<< "expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.second != 6 ) {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected an entry with value 6 only\n";
				rc = FAILED;
			}
		}
		if( rc != SUCCESS ) { return; }
		std::cout << "\b\b 25: ";
		rc = clear( out );
		rc = rc ? rc : eWiseMul( out, odd_mask, scalar, anotherScalar, ring );
		rc = rc ? rc : wait();
		if( rc != SUCCESS ) {
			std::cerr << "primitive returns " << toString( rc ) << ", "
				<< "expected SUCCESS\n";
			rc = FAILED;
			return;
		}
		if( nnz( out ) != n/2 ) {
			std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
				<< "expected " << (n/2) << "\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first % 2 == 0 ) {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected entries at odd positions only\n";
				rc = FAILED;
			}
			if( pair.first % 2 == 1 && pair.second != 6 ) {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected entries with value 6 only\n";
				rc = FAILED;
			}
		}
		if( rc != SUCCESS ) { return; }
	}

	// check masked sparse vector<-vector<-scalar
	std::cout << "\b\b 26: ";
	if( halfLengthIsOdd ) {
#ifdef _DEBUG
		std::cerr << "performing test with odd mask, while n/2=" << (n/2) << "\n";
#endif
		rc = eWiseMul( out, odd_mask, left, scalar, ring );
	} else {
#ifdef _DEBUG
		std::cerr << "performing test with even mask, while n/2=" << (n/2) << "\n";
#endif
		rc = eWiseMul( out, even_mask, left, scalar, ring );
	}
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( halfLengthIsOdd ) {
		if( nnz( out ) != n / 2 ) {
			std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
				<< "expected " << (n/2) << "\n";
			rc = FAILED;
		}
	} else {
		if( nnz( out ) != n/2+1 ) {
			std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
				<< "expected " << (n/2+1) << "\n";
			rc = FAILED;
		}
	}
	for( const auto &pair : out ) {
		if( pair.first == n / 2 ) {
			if( pair.first % 2 == 1 ) {
				if( pair.second != 12 ) {
					std::cerr << "primitive returns an entry ( "
						<< pair.first << ", " << pair.second << " ), "
						<< "expected this entry to have value 12\n";
					rc = FAILED;
				}
			} else {
				if( pair.second != 6 ) {
					std::cerr << "primitive returns an entry ( "
						<< pair.first << ", " << pair.second << " ), "
						<< "expected this entry to have value 6\n";
					rc = FAILED;
				}
			}
		} else if( pair.first % 2 == 1 ) {
			if( pair.second != 6 ) {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected this entry to have value 6\n";
				rc = FAILED;
			}
		} else {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected no entry at this position\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// check masked dense vector<-vector<-scalar
	std::cout << "\b\b 27: ";
	rc = clear( out );
	rc = rc ? rc : eWiseMul( out, even_mask, right, scalar, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != n/2 ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected " << (n/2) << "\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.first % 2 == 1 ) {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected no entry at this position\n";
			rc = FAILED;
		} else {
			assert( pair.first % 2 == 0 );
			if( pair.second != 4 ) {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected the value 4\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) { return; }

	// check masked sparse vector<-scalar<-vector
	std::cout << "\b\b 28: ";
	if( halfLengthIsOdd ) {
		rc = eWiseMul( out, odd_mask, scalar, left, ring );
	} else {
		rc = eWiseMul( out, even_mask, scalar, left, ring );
	}
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( halfLengthIsOdd ) {
		if( nnz( out ) != n/2+1 ) {
			std::cerr << "primitives returns " << nnz( out ) << " nonzeroes, "
				<< "expected " << (n/2+1) << "\n";
			rc = FAILED;
		}
	} else {
		if( nnz( out ) != n/2 ) {
			std::cerr << "primitives returns " << nnz( out ) << " nonzeroes, "
				<< "expected " << (n/2) << "\n";
			rc = FAILED;
		}
	}
	for( const auto &pair : out ) {
		if( pair.first == n / 2 ) {
			if( halfLengthIsOdd ) {
				if( pair.second != 6 ) {
					std::cerr << "primitive returns an entry ( "
						<< pair.first << ", " << pair.second << " ), "
						<< "expected the value 6 at this position\n";
					rc = FAILED;
				}
			} else {
				if( pair.second != 10 ) {
					std::cerr << "primitive returns an entry ( "
						<< pair.first << ", " << pair.second << " ), "
						<< "expected the value 10 at this position\n";
					rc = FAILED;
				}
			}
		} else if( pair.first % 2 != 1 ) {
			if( pair.second != 4 ) {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected the value 4 at this position\n";
				rc = FAILED;
			}
		} else {
			assert( pair.first % 2 == 0 );
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected no entry at this position\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// check masked dense vector<scalar<-vector
	std::cout << "\b\b 29: ";
	rc = eWiseMul( out, odd_mask, scalar, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != n ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected " << n << "\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		if( pair.first == n / 2 ) {
			if( pair.second != 10 ) {
				std::cerr << "primitive returns an entry ( "
					<< pair.first << ", " << pair.second << " ), "
					<< "expected the value 10 at this position\n";
				rc = FAILED;
			}
		} else if( pair.second != 4 ) {
			std::cerr << "primitive returns an entry ( "
				<< pair.first << ", " << pair.second << " ), "
				<< "expected the value 4 at this position\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// check masked with empty vector
	std::cout << "\b\b 30: ";
	rc = clear( out );
	rc = rc ? rc : clear( left );
	rc = rc ? rc : eWiseMul( out, even_mask, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != 0 ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected 0\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		std::cerr << "primitive returns an entry ( "
			<< pair.first << ", " << pair.second << " ), "
			<< "expected no entries\n";
		rc = FAILED;
	}
	if( rc != SUCCESS ) { return; }
	std::cout << "\b\b 31: ";
	std::swap( left, right );
	rc = eWiseMul( out, odd_mask, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != 0 ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected 0\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		std::cerr << "primitive returns an entry ( "
			<< pair.first << ", " << pair.second << " ), "
			<< "expected no entries\n";
		rc = FAILED;
	}
	if( rc != SUCCESS ) { return; }

	// the same without masks
	std::cout << "\b\b 32: ";
	std::swap( left, right );
	rc = eWiseMul( out, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != 0 ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected 0\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		std::cerr << "primitive returns an entry ( "
			<< pair.first << ", " << pair.second << " ), "
			<< "expected no entries\n";
		rc = FAILED;
	}
	if( rc != SUCCESS ) { return; }
	std::cout << "\b\b 33: ";
	std::swap( left, right );
	rc = eWiseMul( out, left, right, ring );
	rc = rc ? rc : wait();
	if( rc != SUCCESS ) {
		std::cerr << "primitive returns " << toString( rc ) << ", "
			<< "expected SUCCESS\n";
		rc = FAILED;
		return;
	}
	if( nnz( out ) != 0 ) {
		std::cerr << "primitive returns " << nnz( out ) << " nonzeroes, "
			<< "expected 0\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		std::cerr << "primitive returns an entry ( "
			<< pair.first << ", " << pair.second << " ), "
			<< "expected no entries\n";
		rc = FAILED;
	}
	if( rc != SUCCESS ) { return; }

	// done
	std::cout << "\b\b OK\n";
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
	Launcher< AUTOMATIC > launcher;
	RC out;
	if( launcher.exec( &grb_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cerr << "Test FAILED (" << toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}

