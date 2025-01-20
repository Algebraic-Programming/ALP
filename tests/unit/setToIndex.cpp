
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


static bool expect_success( const grb::RC rc ) {
	if( rc != grb::SUCCESS ) {
		std::cerr << " expected SUCCESS, got " << grb::toString( rc ) << "\n";
		return false;
	}
	return true;
}

static grb::RC expect_full(
	grb::Vector< double > &dst
) {
	grb::RC ret = grb::SUCCESS;
	if( grb::nnz( dst ) != grb::size( dst ) ) {
		std::cerr << " expected " << grb::size( dst ) << " values, got "
			<< grb::nnz( dst ) << "\n";
		ret = grb::FAILED;
	}
	for( const auto &pair : dst ) {
		if( pair.first != pair.second ) {
			std::cerr << " unexpected output pair ( " << pair.first << ", "
				<< pair.second << " ); expected index to match value\n";
			ret = grb::FAILED;
		}
	}
	return ret;
}

static grb::RC expect_none(
	grb::Vector< double > &dst
) {
	grb::RC ret = grb::SUCCESS;
	if( grb::nnz( dst ) != 0 ) {
		std::cerr << " expected zero values, got " << grb::nnz( dst ) << "\n";
		ret = grb::FAILED;
	}
	for( const auto &pair : dst ) {
		std::cerr << " unexpected output pair ( " << pair.first << ", "
			<< pair.second << " ); expected none\n";
		ret = grb::FAILED;
	}
	return ret;
}

static grb::RC expect_one(
	grb::Vector< double > &dst,
	const size_t expected_index,
	const double expected_value
) {
	grb::RC ret = grb::SUCCESS;
	if( grb::nnz( dst ) != 1 ) {
		std::cerr << " expected one value, got " << grb::nnz( dst ) << "\n";
		ret = grb::FAILED;
	}
	for( const auto &pair : dst ) {
		if( pair.first != expected_index || pair.second != expected_value ) {
			std::cerr << " unexpected output pair ( " << pair.first << ", "
				<< pair.second << " ); expected index " << expected_index
				<< " and value " << expected_value << "\n";
			ret = grb::FAILED;
		}
	}
	return ret;
}

static grb::RC expect_all_but_one(
	grb::Vector< double > &dst,
	const size_t unexpected_index
) {
	grb::RC ret = grb::SUCCESS;
	if( grb::nnz( dst ) + 1 != grb::size( dst ) ) {
		std::cerr << " expected " << (grb::size( dst ) - 1) << " values, got "
			<< grb::nnz( dst ) << "\n";
		ret = grb::FAILED;
	}
	for( const auto &pair : dst ) {
		if( pair.first == unexpected_index ) {
			std::cerr << " unexpected output pair ( " << pair.first << ", "
				<< pair.second << " ); unexpected index\n";
			ret = grb::FAILED;
		} else if( pair.first != pair.second ) {
			std::cerr << " unexpected output pair ( " << pair.first << ", "
				<< pair.second << " ); expected index to match value\n";
			ret = grb::FAILED;
		}
	}
	return ret;
}

static grb::RC expect_constant(
	grb::Vector< double > &dst,
	const double expected_value
) {
	grb::RC ret = grb::SUCCESS;
	if( grb::nnz( dst ) != grb::size( dst ) ) {
		std::cerr << " expected " << grb::size( dst ) << " values, got "
			<< grb::nnz( dst ) << "\n";
		ret = grb::FAILED;
	}
	for( const auto &pair : dst ) {
		if( pair.second != expected_value ) {
			std::cerr << " unexpected output pair ( " << pair.first << ", "
				<< pair.second << " ); expected value " << expected_value << "\n";
			ret = grb::FAILED;
		}
	}
	return ret;
}

static grb::RC dense_tests(
	grb::Vector< double > &dst,
	grb::Vector< double > &src
) {
	// for the subtests that return ILLEGAL due to incorrect usage of the dense
	// descriptor and in the case of nonblocking execution, the output vector may
	// be modified due to side effects. Therefore, for some of the subtests below,
	// the ouput vector is reset explicitly
	constexpr bool nonblocking_execution =
		grb::Properties<>::isNonblockingExecution;
	constexpr auto dense = grb::descriptors::dense | grb::descriptors::use_index;

	assert( grb::size( dst ) == grb::size( src ) );
	grb::Vector< bool > full_mask( grb::size( dst ) );
	grb::Vector< bool > one_mask( grb::size( dst ) );
	grb::RC ret = grb::set( full_mask, false );
	ret = ret ? ret : grb::setElement( one_mask, false, size( dst ) / 2 );
	ret = ret ? ret : grb::clear( src );
	ret = ret ? ret : grb::clear( dst );
	if( ret != grb::SUCCESS ) {
		std::cerr << "\t initialisation of dense tests FAILED\n";
		return ret;
	}

	// subtest overview (all have use_index as additional descriptor to the below-
	// mentioned):
	//  1) dense, sparse output, no mask, scalar input value: expects ILLEGAL
	//  2) dense, sparse output, one_mask, scalar input value: expects ILLEGAL
	//  3) dense, sparse output, full_mask, scalar input value: expects ILLEGAL
	//  4) dense, sparse output, no mask, sparse input vector: expects ILLEGAL
	//  5) dense, sparse output, one_mask, sparse input vector: expects ILLEGAL
	//  6) dense, sparse output, full_mask, sparse input vector: expects ILLEGAL
	//  7) dense, sparse output, no mask, dense input vector: expects ILLEGAL
	//  8) dense, sparse output, one_mask, dense input vector: expects ILLEGAL
	//  9) dense, sparse output, full_mask, dense input vector: expects ILLEGAL
	// 10) dense, dense output, no mask, scalar input value: expects OK
	// 11) dense, dense output, one_mask, scalar input value: expects ILLEGAL
	// 12) dense, dense output, full_mask, scalar input value: expects OK
	// 13) dense, dense output, no mask, sparse input vector: expects ILLEGAL
	// 14) dense, dense output, one_mask, sparse input vector: expects ILLEGAL
	// 15) dense, dense output, full_mask, sparse input vector: expects ILLEGAL
	// 16) dense, dense output, no mask, dense input vector: expects OK
	// 17) dense, dense output, one_mask, dense input vector: expects ILLEGAL
	// 18) dense, dense output, full_mask, dense input vector: expects OK
	//
	// 19) dense+invert, sparse output, one_mask, scalar input value: expects ILL
	// 20) dense+invert, sparse output, full mask, scalar input value: expects ILL
	// 21) dense+invert, sparse output, one_mask, sparse input vector: expects ILL
	// 22) dense+invert, sparse output, full_mask, sparse input vector: expects ILL
	// 23) dense+invert, sparse output, one_mask, dense input vector: expects ILL
	// 24) dense+invert, sparse output, full_mask, dense input vector: expects ILL
	// 25) dense+invert, dense output, one_mask, scalar input value: expects ILL
	// 26) dense+invert, dense output, full mask, scalar input value: expects OK
	// 27) dense+invert, dense output, one_mask, sparse input vector: expects ILL
	// 28) dense+invert, dense output, full_mask, sparse input vector: expects ILL
	// 29) dense+invert, dense output, one_mask, dense input vector: expects ILL
	// 30) dense+invert, dense output, full_mask, dense input vector: expects OK

	std::cerr << "\t dense subtest 1:";
	ret = grb::set< dense >( dst, 1.0 );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::clear( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected failure of grb::clear( dst )\n";
			return grb::FAILED;
		}
	} else {
		if( grb::nnz( dst ) != 0 ) {
			std::cerr << " expected 0, got " << grb::nnz( dst ) << "\n";
			return grb::FAILED;
		}
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 2:";
	ret = grb::set< dense >( dst, one_mask, 1.0 );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::clear( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected failure of grb::clear( dst )\n";
			return grb::FAILED;
		}
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 3:";
	ret = grb::set< dense >( dst, full_mask, 1.0 );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::clear( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected failure of grb::clear( dst )\n";
			return grb::FAILED;
		}
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 4:";
	ret = grb::set< dense >( dst, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::clear( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected failure of grb::clear( dst )\n";
			return grb::FAILED;
		}
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 5:";
	ret = grb::set< dense >( dst, one_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::clear( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected failure of grb::clear( dst )\n";
			return grb::FAILED;
		}
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 6:";
	ret = grb::set< dense >( dst, full_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::clear( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected failure of grb::clear( dst )\n";
			return grb::FAILED;
		}
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 7:";
	ret = grb::set( src, 3.14 );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::SUCCESS ) {
		std::cerr << " could not initialise test:" << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	ret = expect_constant( src, 3.14 );
	if( ret != grb::SUCCESS ) {
		std::cerr << "\t could not initialise test\n";
		return grb::FAILED;
	}
	ret = grb::set< dense >( dst, src );
	ret = ret ? ret : grb::wait();
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::clear( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected error at grb::clear: " << grb::toString( ret )
				<< "\n";
			return grb::FAILED;
		}
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 8:";
	ret = grb::set< dense >( dst, one_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::clear( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected error at grb::clear(dst): " << grb::toString( ret )
				<< "\n";
			return grb::FAILED;
		}
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 9:";
	ret = grb::set< dense >( dst, full_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::clear( dst );
		ret = ret ? ret : grb::wait( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << "unexpected failure at grb::clear: " << grb::toString( ret )
				<< "\n";
			return grb::FAILED;
		}
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 10:";
	ret = grb::set( dst, 0 );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::SUCCESS ) {
		std::cerr << " error initialising test: " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	ret = grb::set< dense >( dst, 3.14 );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::SUCCESS ) {
		std::cerr << " expected SUCCESS, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	ret = expect_full( dst );
	if( ret != grb::SUCCESS ) { return ret; }

	std::cerr << "\b 11:";
	ret = grb::set< dense >( dst, one_mask, 1.0 );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::set< grb::descriptors::use_index >( dst, 100 );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected failure of grb::set\n";
			return grb::FAILED;
		}
	}
	ret = expect_full( dst );
	if( ret != grb::SUCCESS ) { return ret; }

	std::cerr << "\b 12:";
	ret = grb::set< dense >( dst, full_mask, 1.0 );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::SUCCESS ) {
		std::cerr << " expected SUCCESS, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return ret; }

	std::cerr << "\b 13:";
	ret = grb::set( dst, 0 );
	ret = ret ? ret : grb::clear( src );
	ret = ret ? ret : grb::setElement( src, 3.14, grb::size( src ) / 2 );
	ret = ret ? ret : grb::wait( dst, src );
	ret = ret ? ret : expect_one( src, grb::size( src ) / 2, 3.14 );
	if( ret != grb::SUCCESS ) {
		std::cerr << " unexpected failure at test initialisation: "
			<< grb::toString( ret ) << "\n";
	}
	ret = grb::set< dense >( dst, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::set( dst, 0 );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected failure of grb::set( dst, 0 )\n";
			return grb::FAILED;
		}
	}
	ret = expect_constant( dst, 0 );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 14:";
	ret = grb::set< dense >( dst, one_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::set( dst, 0 );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected failure of grb::set( dst, 0 )\n";
			return grb::FAILED;
		}
	}
	ret = expect_constant( dst, 0 );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 15:";
	ret = grb::set< dense >( dst, full_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::set( dst, 0 );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected failure of grb::set( dst, 0 )\n";
			return grb::FAILED;
		}
	}
	ret = expect_constant( dst, 0 );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 16: ";
	ret = grb::set( src, 3.14 );
	ret = ret ? ret : grb::wait( src );
	ret = ret ? ret : expect_constant( src, 3.14 );
	if( ret != grb::SUCCESS ) {
		std::cerr << " could not initialise test: " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	ret = grb::set< dense >( dst, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::SUCCESS ) {
		std::cerr << " expected SUCCESS, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	ret = expect_full( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 17:";
	ret = grb::set< dense >( dst, one_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::set< grb::descriptors::use_index >( dst, 3.14 );
		ret = ret ? ret : grb::wait( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected error at grb::set< use_index >( dst ): "
				<< grb::toString( ret ) << "\n";
			return grb::FAILED;
		}
	}
	ret = expect_full( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 18:";
	ret = grb::set< dense >( dst, full_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::SUCCESS ) {
		std::cerr << " expected SUCCESS, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 19:";
	ret = grb::clear( dst );
	ret = ret ? ret : grb::wait( dst );
	ret = ret ? ret : expect_none( dst );
	if( ret != grb::SUCCESS ) {
		std::cerr << " unexpected error at grb::clear( dst ): "
			<< grb::toString( ret ) << "\n";
		return ret;
	}
	ret = grb::set< dense | grb::descriptors::invert_mask >( dst, one_mask, 3.14 );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		ret = grb::FAILED;
		return ret;
	}
	if( nonblocking_execution ) {
		ret = grb::clear( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected error at grb::clear: " << grb::toString( ret )
				<< "\n";
			ret = grb::FAILED;
			return ret;
		}
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 20:";
	ret = grb::set< dense | grb::descriptors::invert_mask >( dst, full_mask, 1.0 );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::clear( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected error at grb::clear: " << grb::toString( ret )
				<< "\n";
			ret = grb::FAILED;
			return ret;
		}
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return ret; }

	std::cerr << "\b 21:";
	ret = grb::clear( src );
	ret = ret ? ret : grb::setElement( src, 100, grb::size( src ) / 2 );
	ret = ret ? ret : grb::wait( src );
	ret = ret ? ret : expect_one( src, grb::size( src ) / 2, 100 );
	if( ret != grb::SUCCESS ) {
		std::cerr << " unexpected error initialising test: "
			<< grb::toString( ret ) << "\n";
		return ret;
	}
	ret = grb::set< dense | grb::descriptors::invert_mask >( dst, one_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		ret = grb::FAILED;
		return ret;
	}
	if( nonblocking_execution ) {
		ret = grb::clear( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected error at grb::clear: " << grb::toString( ret )
				<< "\n";
			ret = grb::FAILED;
			return ret;
		}
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 22:";
	ret = grb::set< dense | grb::descriptors::invert_mask >( dst, full_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::clear( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected error at grb::clear: " << grb::toString( ret )
				<< "\n";
			ret = grb::FAILED;
			return ret;
		}
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return ret; }

	std::cerr << "\b 23:";
	ret = grb::set( src, 1.17 );
	ret = ret ? ret : grb::wait( src );
	ret = ret ? ret : expect_constant( src, 1.17 );
	if( ret != grb::SUCCESS ) {
		std::cerr << " unexpected error initialising test: "
			<< grb::toString( ret ) << "\n";
		return ret;
	}
	ret = grb::set< dense | grb::descriptors::invert_mask >( dst, one_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		ret = grb::FAILED;
		return ret;
	}
	if( nonblocking_execution ) {
		ret = grb::clear( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected error at grb::clear: " << grb::toString( ret )
				<< "\n";
			ret = grb::FAILED;
			return ret;
		}
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 24:";
	ret = grb::set< dense | grb::descriptors::invert_mask >( dst, full_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::clear( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected error at grb::clear: " << grb::toString( ret )
				<< "\n";
			ret = grb::FAILED;
			return ret;
		}
	}
	ret = expect_none( dst );
	if( ret != grb::SUCCESS ) { return ret; }

	std::cerr << "\b 25:";
	ret = grb::set( dst, 0 );
	ret = ret ? ret : grb::wait( dst );
	ret = ret ? ret : expect_constant( dst, 0 );
	if( ret != grb::SUCCESS ) {
		std::cerr << " unexpected error at grb::set( dst, 0 ): "
			<< grb::toString( ret ) << "\n";
		return ret;
	}
	ret = grb::set< dense | grb::descriptors::invert_mask >( dst, one_mask, 3.14 );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		ret = grb::FAILED;
		return ret;
	}
	if( nonblocking_execution ) {
		ret = grb::set( dst, 0 );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected error at grb::set: " << grb::toString( ret )
				<< "\n";
			ret = grb::FAILED;
			return ret;
		}
	}
	ret = expect_constant( dst, 0 );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 26:";
	ret = grb::set< dense | grb::descriptors::invert_mask >( dst, full_mask, 1.0 );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::SUCCESS ) {
		std::cerr << " expected SUCCESS, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	ret = expect_full( dst );
	if( ret != grb::SUCCESS ) { return ret; }

	std::cerr << "\b 27:";
	ret = grb::clear( src );
	ret = ret ? ret : grb::setElement( src, 100, grb::size( src ) / 2 );
	ret = ret ? ret : grb::wait( src );
	ret = ret ? ret : expect_one( src, grb::size( src ) / 2, 100 );
	if( ret != grb::SUCCESS ) {
		std::cerr << " unexpected error initialising test: "
			<< grb::toString( ret ) << "\n";
		return ret;
	}
	ret = grb::set< dense | grb::descriptors::invert_mask >( dst, one_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		ret = grb::FAILED;
		return ret;
	}
	if( nonblocking_execution ) {
		ret = grb::set< grb::descriptors::use_index >( dst, 77 );
		ret = ret ? ret : grb::wait( dst );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected error at grb::set< use_index >( dst ): "
				<< grb::toString( ret ) << "\n";
			ret = grb::FAILED;
			return ret;
		}
	}
	ret = expect_full( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 28:";
	ret = grb::set< dense | grb::descriptors::invert_mask >( dst, full_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	if( nonblocking_execution ) {
		ret = grb::set< grb::descriptors::use_index >( dst, 0 );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected error at grb::set< use_index >( dst ): "
				<< grb::toString( ret ) << "\n";
			return grb::FAILED;
		}
	}
	ret = expect_full( dst );
	if( ret != grb::SUCCESS ) { return ret; }

	std::cerr << "\b 29:";
	ret = grb::set( src, 2.17 );
	ret = ret ? ret : grb::wait( src );
	ret = ret ? ret : expect_constant( src, 2.17 );
	if( ret != grb::SUCCESS ) {
		std::cerr << " unexpected error initialising test: "
			<< grb::toString( ret ) << "\n";
		return ret;
	}
	ret = grb::set< dense | grb::descriptors::invert_mask >( dst, one_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::ILLEGAL ) {
		std::cerr << " expected ILLEGAL, got " << grb::toString( ret ) << "\n";
		ret = grb::FAILED;
		return ret;
	}
	if( nonblocking_execution ) {
		ret = grb::set< grb::descriptors::use_index >( dst, 1 );
		if( ret != grb::SUCCESS ) {
			std::cerr << " unexpected error at grb::set< use_index >( dst ) "
				<< grb::toString( ret ) << "\n";
			ret = grb::FAILED;
			return ret;
		}
	}
	ret = expect_full( dst );
	if( ret != grb::SUCCESS ) { return grb::FAILED; }

	std::cerr << "\b 30:";
	ret = grb::set( dst, 0 );
	ret = ret ? ret : grb::wait( dst );
	ret = ret ? ret : expect_constant( dst, 0 );
	if( ret != grb::SUCCESS ) {
		std::cerr << " error initialising test: " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	ret = grb::set< dense | grb::descriptors::invert_mask >( dst, full_mask, src );
	ret = ret ? ret : grb::wait( dst );
	if( ret != grb::SUCCESS ) {
		std::cerr << " expected SUCCESS, got " << grb::toString( ret ) << "\n";
		return grb::FAILED;
	}
	ret = expect_full( dst );
	if( ret != grb::SUCCESS ) { return ret; }

	std::cerr << "\b OK\n";
	return grb::SUCCESS;
}

void grb_program( const size_t &n, grb::RC &rc ) {
	grb::Vector< double > dst( n ), src( n );
	grb::Vector< bool > one_mask( n ), full_mask( n );

	// subtest overview (all have use_index as additional descriptor to the below-
	// mentioned):
	//  1) no_operation, sparse output, no mask, scalar input value: expects OK
	//  2) no_operation, sparse output, one_mask, scalar input value: expects OK
	//  3) no_operation, sparse output, full_mask, scalar input value: expects OK
	//  4) no_operation, sparse output, no mask, sparse input vector: expects OK
	//  5) no_operation, sparse output, one_mask, sparse input vector: expects OK
	//  6) no_operation, sparse output, full_mask, sparse input vector: expects OK
	//  7) no_operation, sparse output, no mask, dense input vector: expects OK
	//  8) no_operation, sparse output, one_mask, dense input vector: expects OK
	//  9) no_operation, sparse output, full_mask, dense input vector: expects OK
	// 10) no_operation, dense output, no mask, scalar input value: expects OK
	// 11) no_operation, dense output, one_mask, scalar input value: expects OK
	// 12) no_operation, dense output, full_mask, scalar input value: expects OK
	// 13) no_operation, dense output, no mask, sparse input vector: expects OK
	// 14) no_operation, dense output, one_mask, sparse input vector: expects OK
	// 15) no_operation, dense output, full_mask, sparse input vector: expects OK
	// 16) no_operation, dense output, no mask, dense input vector: expects OK
	// 17) no_operation, dense output, one_mask, dense input vector: expects OK
	// 18) no_operation, dense output, full_mask, dense input vector: expects OK
	//
	// 19) invert, sparse output, one_mask, scalar input value: expects OK
	// 20) invert, sparse output, full mask, scalar input value: expects OK
	// 21) invert, sparse output, one_mask, sparse input vector: expects OK
	// 22) invert, sparse output, full_mask, sparse input vector: expects OK
	// 23) invert, sparse output, one_mask, dense input vector: expects OK
	// 24) invert, sparse output, full_mask, dense input vector: expects OK
	// 25) invert, dense output, one_mask, scalar input value: expects OK
	// 26) invert, dense output, full mask, scalar input value: expects OK
	// 27) invert, dense output, one_mask, sparse input vector: expects OK
	// 28) invert, dense output, full_mask, sparse input vector: expects OK
	// 29) invert, dense output, one_mask, dense input vector: expects OK
	// 30) invert, dense output, full_mask, dense input vector: expects OK

	constexpr auto use_index = grb::descriptors::use_index;
	constexpr auto invert = grb::descriptors::invert_mask | use_index;

	std::cerr << "\t general subtest 1:";
	rc = grb::clear( dst );
	rc = rc ? rc : grb::clear( src );
	rc = rc ? rc : grb::setElement( dst, 0, 0 );
	rc = rc ? rc : grb::setElement( src, 3.14, grb::size( src ) / 2 );
	rc = rc ? rc : grb::wait( dst, src );
	rc = rc ? rc : expect_one( dst, 0, 0 );
	rc = rc ? rc : expect_one( src, grb::size( src ) / 2, 3.14 );
	rc = rc ? rc : grb::setElement( one_mask, true, grb::size( one_mask ) / 2 );
	rc = rc ? rc : grb::set( full_mask, false );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED: " << grb::toString( rc ) << "\n";
		return;
	}
	rc = grb::set< use_index >( dst, 1.0 );
	rc = rc ? rc : grb::wait( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 2:";
	rc = grb::clear( dst );
	rc = rc ? rc : grb::wait();
	rc = rc ? rc : expect_none( dst );
	if( rc != grb::SUCCESS ) {
		std::cerr << " error initialising test: " << grb::toString( rc ) << "\n";
		return;
	}
	rc = grb::set< use_index >( dst, one_mask, src );
	rc = rc ? rc : grb::wait();
	if( !expect_success( rc ) ) { return; }
	const size_t half_size = grb::size( one_mask ) / 2;
	rc = expect_one( dst, half_size, half_size );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 3:";
	rc = grb::clear( dst );
	rc = rc ? rc : grb::set( dst, one_mask, 3.14 );
	rc = rc ? rc : grb::wait( dst );
	rc = rc ? rc : expect_one( dst, half_size, 3.14 );
	if( rc != grb::SUCCESS ) {
		std::cerr << " error initialising test\n";
		return;
	}
	rc = grb::set< use_index >( dst, full_mask, 7.0 );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_none( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 4:";
	rc = grb::set< use_index >( dst, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_one( dst, half_size, half_size );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 5:";
	rc = grb::clear( dst );
	rc = rc ? rc : grb::setElement( dst, 3.14, grb::size( dst ) - 1 );
	rc = rc ? rc : grb::wait( dst );
	rc = rc ? rc : expect_one( dst, grb::size( dst ) - 1, 3.14 );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED: " << grb::toString( rc ) << "\n";
		return;
	}
	rc = grb::set< use_index >( dst, one_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_one( dst, half_size, half_size );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 6:";
	rc = grb::clear( dst );
	rc = rc ? rc : grb::wait( dst );
	rc = rc ? rc : expect_none( dst );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED: " << grb::toString( rc ) << "\n";
		return;
	}
	rc = grb::set< use_index >( dst, full_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_one( dst, half_size, half_size );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 7:";
	rc = grb::set( src, 3.14 );
	rc = rc ? rc : grb::wait( src );
	rc = rc ? rc : expect_constant( src, 3.14 );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED: " << grb::toString( rc ) << "\n";
		return;
	}
	rc = grb::set< use_index >( dst, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_full( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 8:";
	rc = grb::clear( dst );
	rc = rc ? rc : grb::setElement( dst, 2.71, 0 );
	rc = rc ? rc : grb::wait( dst );
	rc = rc ? rc : expect_one( dst, 0, 2.17 );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED: " << grb::toString( rc ) << "\n";
		return;
	}
	rc = grb::set< use_index >( dst, one_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_one( dst, half_size, half_size );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 9:";
	rc = grb::set< use_index >( dst, full_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_none( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 10:";
	rc = grb::set( dst, 1.17 );
	rc = rc ? rc : grb::wait( dst );
	rc = rc ? rc : expect_constant( dst, 1.17 );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED: " << grb::toString( rc ) << "\n";
		return;
	}
	rc = grb::set< use_index >( dst, 10.0 );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_full( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 11:";
	rc = grb::set< use_index >( dst, one_mask, 7.0 );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_one( dst, half_size, half_size );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 12:";
	rc = grb::set( dst, 3.14 );
	rc = rc ? rc : grb::wait( dst );
	rc = rc ? rc : expect_constant( dst, 3.14 );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED: " << grb::toString( rc ) << "\n";
		return;
	}
	rc = grb::set< use_index >( dst, full_mask, 1.0 );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_none( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 13:";
	rc = grb::clear( src );
	rc = rc ? rc : grb::setElement( src, 0, 0 );
	rc = rc ? rc : grb::set( dst, 3.14 );
	rc = rc ? rc : grb::wait( src, dst );
	rc = rc ? rc : expect_one( src, 0, 0 );
	rc = rc ? rc : expect_constant( dst, 3.14 );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED: " << grb::toString( rc ) << "\n";
		return;
	}
	rc = grb::set< use_index >( dst, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_full( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 14:";
	rc = grb::set< use_index >( dst, one_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_one( dst, half_size, half_size );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 15:";
	rc = grb::set( dst, 0 );
	rc = rc ? rc : grb::wait( dst );
	rc = rc ? rc : expect_constant( dst, 0 );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED: " << grb::toString( rc ) << "\n";
		return;
	}
	rc = grb::set< use_index >( dst, full_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_none( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 16:";
	rc = grb::set( dst, 1 );
	rc = rc ? rc : grb::set( src, 3.14 );
	rc = rc ? rc : grb::wait( dst, src );
	rc = rc ? rc : expect_constant( dst, 1 );
	rc = rc ? rc : expect_constant( src, 3.14 );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED: " << grb::toString( rc ) << "\n";
		return;
	}
	rc = grb::set< use_index >( dst, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_full( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 17:";
	rc = grb::setElement( dst, grb::size( dst ), half_size );
	rc = rc ? rc : grb::wait( dst );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED\n";
		return;
	}
	if( grb::nnz( dst ) != grb::size( dst ) ) {
		std::cerr << " expected " << grb::size( dst ) << " values, got "
			<< grb::nnz( dst ) << ". Test initialisation FAILED\n";
		rc = grb::FAILED;
	}
	for( const auto &pair : dst ) {
		if( pair.first == half_size ) {
			if( pair.second != grb::size( dst ) ) {
				std::cerr << " unexpected pair ( " << pair.first << ", "
					<< pair.second << " ); expected value " << grb::size( dst ) << ". "
					<< "Test initialisation FAILED\n";
				rc = grb::FAILED;
			}
		} else {
			if( pair.second != pair.first ) {
				std::cerr << " unexpected pair ( " << pair.first << ", "
					<< pair.second << " ); expected value matching index. "
					<< "Test initialisation FAILED\n";
				rc = grb::FAILED;
			}
		}
	}
	if( rc != grb::SUCCESS ) { return; }
	rc = grb::set< use_index >( dst, one_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_one( dst, half_size, half_size );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 18:";
	rc = grb::set( dst, 1.0 );
	rc = rc ? rc : grb::wait( dst );
	rc = rc ? rc : expect_constant( dst, 1.0 );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED\n";
		return;
	}
	rc = grb::set< use_index >( dst, full_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_none( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 19:";
	rc = rc ? rc : grb::clear( dst );
	rc = rc ? rc : grb::setElement( dst, 15.3, 0 );
	rc = rc ? rc : grb::wait( dst );
	{
		bool initFailed = rc == grb::SUCCESS;
		if( !initFailed && grb::nnz( dst ) != 1 ) {
			initFailed = true;
		}
		if( !initFailed && (dst.cbegin())->first != 0 ) {
			initFailed = true;
		}
		if( !initFailed && (dst.cbegin())->second != 15.3 ) {
			initFailed = true;
		}
		if( !initFailed && (++(dst.cbegin())) != dst.cend() ) {
			initFailed = true;
		}
		if( initFailed ) {
			if( rc == grb::SUCCESS ) {
				rc = grb::FAILED;
			}
			std::cerr << " test initialisation FAILED\n";
			return;
		}
	}
	rc = grb::set< invert >( dst, one_mask, 3.14 );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_all_but_one( dst, half_size );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 20:";
	rc = grb::set< invert >( dst, full_mask, 7.17 );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_none( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr<< "\b 21A:";
	rc = grb::clear( src );
	rc = rc ? rc : grb::wait( src );
	if( rc != grb::SUCCESS || grb::nnz( src ) != 0 ) {
		std::cerr << " test initialisation FAILED\n";
		if( rc != grb::SUCCESS ) {
			rc = grb::FAILED;
		}
		return;
	}
	rc = grb::set< invert >( dst, one_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_none( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\n 21B:";
	rc = grb::setElement( src, 3.14, half_size );
	rc = rc ? rc : grb::wait( src );
	{
		bool initFailed = rc != grb::SUCCESS;
		if( grb::nnz( src ) != 1 ) { initFailed = true; }
		if( (src.cbegin())->first != half_size ) { initFailed = true; }
		if( (src.cbegin())->second != 3.14 ) { initFailed = true; }
		if( (++(src.cbegin())) != src.cend() ) { initFailed = true; }
		if( initFailed && rc == grb::SUCCESS ) { rc = grb::FAILED; }
		if( initFailed ) {
			std::cerr << " test initialisation FAILED\n";
			return;
		}
	}
	rc = grb::set< invert >( dst, one_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_none( dst );
	if( rc != grb::SUCCESS ) { return ; }

	std::cerr << "\b 21C:";
	rc = grb::setElement( src, 7.17, 0 );
	rc = rc ? rc : grb::wait( src );
	{
		bool initFailed = grb::SUCCESS != rc;
		if( grb::nnz( src ) != 2 ) { initFailed = true; }
		auto it = src.cbegin();
		if( it->first != 0 || it->first != half_size ) { initFailed = true; }
		if( it->first == 0 && it->second != 7.17 ) { initFailed = true; }
		if( it->first == half_size && it->second != 3.14 ) { initFailed = true; }
		(void) ++it;
		if( it->first != 0 || it->first != half_size ) { initFailed = true; }
		if( it->first == 0 && it->second != 7.17 ) { initFailed = true; }
		if( it->first == half_size && it->second != 3.14 ) { initFailed = true; }
		if( (++it) != src.cend() ) { initFailed = true; }
		if( initFailed && rc == grb::SUCCESS ) { rc = grb::FAILED; }
		if( initFailed ) {
			std::cerr << " test initialisation FAILED\n";
			return;
		}
	}
	rc = grb::set< invert >( dst, one_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_one( dst, 0, 0 );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 22:";
	rc = grb::set< invert >( dst, full_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_none( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 23:";
	rc = grb::set( src, 3.14 );
	rc = rc ? rc : grb::wait( src );
	{
		bool initFailed = grb::SUCCESS != rc;
		if( !initFailed && grb::nnz( src ) != grb::size( src ) ) { initFailed = true; }
		if( expect_constant( src, 3.14 ) != grb::SUCCESS ) { initFailed = true; }
		if( initFailed && rc == grb::SUCCESS ) { rc = grb::FAILED; }
		if( initFailed ) {
			std::cerr << " test initialisation FAILED\n";
			return;
		}
	}
	rc = grb::set< invert >( dst, one_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_one( dst, 0, 0 );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 24:";
	rc = grb::set< invert >( dst, full_mask, src );
	rc = rc ? rc : grb::wait();
	if( !expect_success( rc ) ) { return; }
	rc = expect_none( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 25:";
	rc = grb::set( dst, 3.14 );
	rc = rc ? rc : grb::wait( dst );
	rc = rc ? rc : expect_constant( dst, 3.14 );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED\n";
		return;
	}
	rc = grb::set< invert >( dst, one_mask, 7.17 );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_all_but_one( dst, half_size );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 26:";
	rc = grb::set( dst, 3.14 );
	rc = rc ? rc : grb::wait( dst );
	rc = rc ? rc : expect_constant( dst, 3.14 );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED\n";
		return;
	}
	rc = grb::set< invert >( dst, full_mask, 7.17 );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_none( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 27A:";
	rc = grb::set( dst, 3.14 );
	rc = rc ? rc : grb::clear( src );
	rc = rc ? rc : grb::wait( dst, src );
	rc = rc ? rc : expect_constant( dst, 3.14 );
	rc = rc ? rc : expect_none( src );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED\n";
		return;
	}
	rc = grb::set< invert >( dst, one_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_none( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 27B:";
	rc = grb::set( dst, 3.14 );
	rc = rc ? rc : grb::setElement( src, 7.17, 0 );
	rc = rc ? rc : grb::setElement( src, 7.07, half_size );
	rc = rc ? rc : grb::wait( dst, src );
	rc = rc ? rc : expect_constant( dst, 3.14 );
	if( rc == grb::SUCCESS ) {
		if( grb::nnz( src ) != 2 ) {
			std::cerr << " expected two entries in src\n";
			rc = grb::FAILED;
		}
		for( const auto &pair : src ) {
			if( pair.first != 0 && pair.first != half_size ) {
				std::cerr << " expected only entries at position 0 or " << half_size
					<< "\n";
				rc = grb::FAILED;
			}
			if( pair.first == 0 && pair.second != 7.17 ) {
				std::cerr << " expected value 7.17 at position 0\n";
				rc = grb::FAILED;
			}
			if( pair.first == half_size && pair.second != 7.07 ) {
				std::cerr << " expected value 7.07 at position " << half_size << "\n";
				rc = grb::FAILED;
			}
		}
	}
	if( rc == grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED\n";
		return;
	}
	rc = grb::set< invert >( dst, one_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_one( dst, 0, 0 );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 28:";
	rc = grb::set( dst, 3.14 );
	rc = rc ? rc : grb::wait( dst );
	rc = rc ? rc : expect_constant( dst, 3.14 );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED\n";
		return;
	}
	rc = grb::set< invert >( dst, full_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_none( dst );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 29:";
	rc = grb::set( dst, 3.14 );
	rc = rc ? rc : grb::set( src, 7.17 );
	rc = rc ? rc : expect_constant( dst, 3.14 );
	rc = rc ? rc : expect_constant( src, 7.17 );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED\n";
		return;
	}
	rc = grb::set< invert >( dst, one_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_all_but_one( dst, half_size );
	if( rc != grb::SUCCESS ) { return; }

	std::cerr << "\b 30:";
	rc = grb::set( dst, 3.14 );
	rc = rc ? rc : expect_constant( dst, 3.14 );
	if( rc != grb::SUCCESS ) {
		std::cerr << " test initialisation FAILED\n";
		return;
	}
	rc = grb::set< invert >( dst, full_mask, src );
	rc = rc ? rc : grb::wait( dst );
	if( !expect_success( rc ) ) { return; }
	rc = expect_none( dst );
	if( rc != grb::SUCCESS ) { return; }

	// test behaviour under dense descriptor
	rc = dense_tests( dst, src );

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
		if( !( ss >> read ) ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( !ss.eof() ) {
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
		std::cerr << "  -n (optional, default is 100): an even integer, "
			<< "the test size.\n";
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

