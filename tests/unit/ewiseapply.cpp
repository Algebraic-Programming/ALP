
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

void grb_program( const size_t &n, grb::RC &rc ) {
	grb::Vector< double > out( n ), left( n ), right( n );
	grb::Vector< bool > mask( n );
	grb::Vector< size_t > temp( n );
	rc = grb::set< grb::descriptors::use_index >( temp, 0 );
	rc = rc ? rc : grb::eWiseLambda( [&temp] (const size_t i) {
			if( temp[ i ] % 2 == 0 ) {
				temp[ i ] = 1;
			} else {
				temp[ i ] = 0;
			}
		}, temp );

	// left = 1.5 everywhere
	// right = 0.25 at every even index
	rc = rc ? rc : grb::set( left, 1.5 );
	rc = rc ? rc : grb::set( right, temp, 0.25 );

	// mask = true at the lower half
	rc = rc ? rc : grb::set< grb::descriptors::use_index >( temp, 0 );
	rc = rc ? rc : grb::eWiseLambda( [&temp,&n] (const size_t i) {
			if( temp[ i ] < n / 2 ) {
				temp[ i ] = 1;
			} else {
				temp[ i ] = 0;
			}
		}, temp );
	rc = rc ? rc : grb::set( mask, temp, true );
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}

	grb::Monoid< grb::operators::add< double >, grb::identities::zero > plusM;

	// test operator versions first, dense vectors only, without masks
	rc = grb::eWiseApply( out, left, 0.25, plusM.getOperator() );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != size( out ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << " ), "
				<< "expected " << size( out ) << " ) at subtest 1\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.second != 1.75 ) {
				std::cerr << "\tunexpected entry ( " << pair.first << ", "
					<< pair.second << " ); expected ( " << pair.first << ", 1.75 ) "
					<< "at subtest 1\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, 0.25, left, plusM.getOperator() );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != size( out ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << ", "
				<< "expected " << size( out ) << " ) at subtest 2\n";
			rc = FAILED;
		}
		for( const auto pair : out ) {
			if( pair.second != 1.75 ) {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< "; expected ( " << pair.first << ", 1.75 ) at subtest 2\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, left, left, plusM.getOperator() );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != size( out ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << ", "
				<< "expected " << size( out ) << " ) at subtest 3\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.second != static_cast< double >( 3 ) ) {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< "; expected ( " << pair.first << ", 3 ) at subtest 3\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	// operator versions, dense vectors only, with masks
	rc = grb::eWiseApply( out, mask, left, 0.25, plusM.getOperator() );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( grb::nnz( out ) != grb::nnz( mask ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << grb::nnz( out )
				<< " != " << grb::nnz( mask ) << " ) at subtest 4\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first < n / 2 ) {
				if( pair.second != 1.75 ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< " ); expected ( " << pair.first << ", 1.75 ) at subtest 4\n";
					rc = FAILED;
					return;
				}
			} else {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< " ); expected this index to be empty) at subtest 4\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, mask, 0.25, left, plusM.getOperator() );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( grb::nnz( out ) != grb::nnz( mask ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << grb::nnz( out )
				<< " != " << grb::nnz( mask ) << " ) at subtest 5\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first < n / 2 ) {
				if( pair.second != 1.75 ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< "; expected ( " << pair.first << ", 1.75 ) at subtest 5\n";
					rc = FAILED;
				}
			} else {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< "; expected this index to be empty) at subtest 5\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, mask, left, left, plusM.getOperator() );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( grb::nnz( out ) != grb::nnz( mask ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << grb::nnz( out )
				<< " != " << grb::nnz( mask ) << " ) at subtest 6\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first < n / 2 ) {
				if( pair.second != static_cast< double >( 3 ) ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< "; expected ( " << pair.first << ", 3 ) at subtest 6\n";
					rc = FAILED;
				}
			} else {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< "; expected this index to be empty) at subtest 6\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	// monoid version, dense vectors, unmasked
	rc = grb::eWiseApply( out, left, 0.25, plusM );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != size( out ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << ", "
				<< "expected " << size( out ) << " ) at subtest 7\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.second != 1.75 ) {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< "; expected ( " << pair.first << ", 1.75 ) at subtest 7\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, 0.25, left, plusM );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != size( out ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << ", "
				<< "expected " << size( out ) << " ) at subtest 8\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.second != 1.75 ) {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< "; expected ( " << pair.first << ", 1.75 ) at subtest 8\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, left, left, plusM.getOperator() );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != size( out ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out )
				<< ", expected " << size( out ) << " ) at subtest 9\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.second != static_cast< double >( 3 ) ) {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< "; expected ( " << pair.first << ", 3 ) at subtest 9\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	// monoid versions, dense vectors, with masks
	rc = grb::eWiseApply( out, mask, left, 0.25, plusM );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( grb::nnz( out ) != grb::nnz( mask ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << grb::nnz( out )
				<< " != " << grb::nnz( mask ) << " ) at subtest 10\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first < n / 2 ) {
				if( pair.second != 1.75 ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< "; expected ( " << pair.first << ", 1.75 ) at subtest 10\n";
					rc = FAILED;
				}
			} else {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< "; expected this index to be empty) at subtest 10\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, mask, 0.25, left, plusM );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( grb::nnz( out ) != grb::nnz( mask ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << grb::nnz( out )
				<< " != " << grb::nnz( mask ) << " ) at subtest 11\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first < n / 2 ) {
				if( pair.second != 1.75 ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< "; expected ( " << pair.first << ", 1.75 ) at subtest 11\n";
					rc = FAILED;
				}
			} else {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< "; expected this index to be empty) at subtest 11\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, mask, left, left, plusM );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( grb::nnz( out ) != grb::nnz( mask ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << grb::nnz( out )
				<< " != " << grb::nnz( mask ) << " ) at subtest 12\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first < n / 2 ) {
				if( pair.second != static_cast< double >( 3 ) ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< "; expected ( " << pair.first << ", 3 ) at subtest 12\n";
					rc = FAILED;
				}
			} else {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< "; expected this index to be empty) at subtest 12\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	// monoid version, sparse vectors, unmasked
	rc = grb::eWiseApply( out, right, 0.25, plusM );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != size( out ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << ", "
				<< "expected " << size( out ) << " ) at subtest 13\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first % 2 == 0 ) {
				if( pair.second != 0.5 ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< "; expected ( " << pair.first << ", 0.5 ) at subtest 13\n";
					rc = FAILED;
				}
			} else {
				if( pair.second != 0.25 ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< "; expected ( " << pair.first << ", 0.25 ) at subtest 13\n";
					rc = FAILED;
				}
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, 0.25, right, plusM );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != size( out ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << " ), "
				<< "expected " << size( out ) << " ) at subtest 14\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first % 2 == 0 ) {
				if( pair.second != 0.5 ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< " ); expected ( " << pair.first << ", 0.5 ) at subtest 14\n";
					rc = FAILED;
				}
			} else {
				if( pair.second != 0.25 ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< " ); expected ( " << pair.first << ", 0.25 ) at subtest 14\n";
					rc = FAILED;
				}
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, left, right, plusM );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != size( out ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << " ), "
				<< "expected " << size( out ) << " ) at subtest 15\n";
			rc = FAILED;
		}
		for( const auto pair : out ) {
			if( pair.first % 2 == 0 ) {
				if( pair.second != static_cast< double >( 1.75 ) ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< " ); expected ( " << pair.first << ", 1.75 ) at subtest 15\n";
					rc = FAILED;
				}
			} else {
				if( pair.second != static_cast< double >( 1.5 ) ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< " ); expected ( " << pair.first << ", 1.5 ) at subtest 15\n";
					rc = FAILED;
				}
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, right, left, plusM );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != size( right ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << ", "
				<< "expected " << size( right ) << " ) at subtest 16\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first % 2 == 0 ) {
				if( pair.second != static_cast< double >( 1.75 ) ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< "; expected ( " << pair.first << ", 1.75 ) at subtest 16\n";
					rc = FAILED;
				}
			} else {
				if( pair.second != static_cast< double >( 1.5 ) ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< "; expected ( " << pair.first << ", 1.5 ) at subtest 16\n";
					rc = FAILED;
				}
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, right, right, plusM );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != nnz( right ) ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << ", "
				<< "expected " << nnz( right ) << " ) at subtest 17\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first % 2 == 0 ) {
				if( pair.second != static_cast< double >( .5 ) ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< "; expected ( " << pair.first << ", 0.5 ) at subtest 17\n";
					rc = FAILED;
				}
			} else {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< "; expected nothing at this entry) at subtest 17\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	// monoid version, sparse vectors, with masks
	rc = grb::eWiseApply( out, mask, right, 0.25, plusM );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != size( out ) / 2 ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << ", "
			       << "expected " << size( out ) / 2 << " ) at subtest 18\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first < n / 2 ) {
				if( pair.first % 2 == 0 ) {
					if( pair.second != 0.5 ) {
						std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
							<< "; expected ( " << pair.first << ", 0.5 ) at subtest 18\n";
						rc = FAILED;
					}
				} else {
					if( pair.second != 0.25 ) {
						std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
							<< "; expected ( " << pair.first << ", 0.25 ) at subtest 18\n";
						rc = FAILED;
					}
				}
			} else {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< "; expected nothing at this index) at subtest 18\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, mask, 0.25, right, plusM );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != size( out ) / 2 ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << ", "
				<< "expected " << size( out ) / 2 << " ) at subtest 19\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first < n / 2 ) {
				if( pair.first % 2 == 0 ) {
					if( pair.second != 0.5 ) {
						std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
							<< "; expected ( " << pair.first << ", 0.5 ) at subtest 19\n";
						rc = FAILED;
					}
				} else {
					if( pair.second != 0.25 ) {
						std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
							<< "; expected ( " << pair.first << ", 0.25 ) at subtest 19\n";
						rc = FAILED;
					}
				}
			} else {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< "; expected nothing at this index) at subtest 19\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, mask, left, right, plusM );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != size( out ) / 2 ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << ", "
				<< "expected " << size( out ) / 2 << " ) at subtest 20\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first < n / 2 ) {
				if( pair.first % 2 == 0 ) {
					if( pair.second != static_cast< double >( 1.75 ) ) {
						std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
							<< "; expected ( " << pair.first << ", 1.75 ) at subtest 20\n";
						rc = FAILED;
					}
				} else {
					if( pair.second != static_cast< double >( 1.5 ) ) {
						std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
							<< "; expected ( " << pair.first << ", 1.5 ) at subtest 20\n";
						rc = FAILED;
					}
				}
			} else {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< "; expected nothing at this index) at subtest 20\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, mask, right, left, plusM );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != size( right ) / 2 ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << ", "
				<< "expected " << size( right ) / 2 << " ) at subtest 21\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first < n / 2 ) {
				if( pair.first % 2 == 0 ) {
					if( pair.second != static_cast< double >( 1.75 ) ) {
						std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
							<< "; expected ( " << pair.first << ", 1.75 ) at subtest 21\n";
						rc = FAILED;
					}
				} else {
					if( pair.second != static_cast< double >( 1.5 ) ) {
						std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
							<< "; expected ( " << pair.first << ", 1.5 ) at subtest 21\n";
						rc = FAILED;
					}
				}
			} else {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< "; expected nothing at this index) at subtest 21\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, mask, right, right, plusM );
	assert( rc == SUCCESS );
	const bool halfIsOdd = ((n / 2) % 2) == 1;
	if( rc == SUCCESS ) {
		const size_t expected = nnz( right ) / 2 + (halfIsOdd ? 1 : 0);
		if( nnz( out ) != expected ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << ", "
				<< "expected " << expected << " ) at subtest 22\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first < n / 2 ) {
				if( pair.first % 2 == 0 ) {
					if( pair.second != static_cast< double >( .5 ) ) {
						std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
							<< " ), expected ( " << pair.first << ", 0.5 ) at subtest 22\n";
						rc = FAILED;
					}
				} else {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< " ), expected nothing at this entry at subtest 22\n";
					rc = FAILED;
				}
			} else {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< " ), expected nothing at this index at subtest 22\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = clear( right );
	rc = rc ? rc : clear( left );
	rc = rc ? rc : setElement( right, 2.17, 0 );
	rc = rc ? rc : setElement( right, 2.0, n/2 );
	rc = rc ? rc : setElement( right, 3.14, n-1 );
	rc = rc ? rc : setElement( left,  1.0, n-1 );
	rc = rc ? rc : setElement( left, -1.0, 0 );
	rc = eWiseApply( out, mask, left, right, plusM );
	assert( n % 2 == 0 );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		const size_t expect = 1;
		if( nnz( out ) != expect ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << ", "
				<< "), expected " << expect << " ) at subtest 23\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first == 0 ) {
				if( pair.second != 1.17 ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< " ), expected ( " << pair.first << ", 1.17 ) at subtest 23\n";
					rc = FAILED;
				}
			} else {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< " ), expected ( " << pair.first << ", " << (n/2) << " ) "
					<< "at subtest 23\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	rc = grb::eWiseApply( out, left, right, plusM );
	assert( rc == SUCCESS );
	if( rc == SUCCESS ) {
		if( nnz( out ) != 3 ) {
			std::cerr << "\tunexpected number of nonzeroes ( " << nnz( out ) << ", "
				<< "expected " << 3 << " ) at subtest 24\n";
			rc = FAILED;
		}
		for( const auto &pair : out ) {
			if( pair.first == 0 ) {
				if( pair.second != 1.17 ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< " ), expected ( " << pair.first << ", 1.17 ) at subtest 24\n";
					rc = FAILED;
				}
			} else if( pair.first == n / 2 ) {
				if( pair.second != 2.0 ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< " ), expected ( " << pair.first << ", 2.0 ) at subtest 24\n";
					rc = FAILED;
				}
			} else if( pair.first == n - 1 ) {
				if( !utils::equals( pair.second, 4.14, 1 ) ) {
					std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
						<< " ), expected ( " << pair.first << ", 4.14 ) at subtest 24\n";
					rc = FAILED;
				}
			} else {
				std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second
					<< ", expected nothing at this index at subtest 24\n";
				rc = FAILED;
			}
		}
		if( rc == FAILED ) {
			return;
		}
	} else {
		return;
	}

	// those were all variants:)
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
		std::cerr << std::flush;
		std::cout << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}

