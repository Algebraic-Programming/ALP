
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

#include "graphblas/utils/parser.hpp"

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";
	grb::utils::MatrixFileReader< double, unsigned short int > west( "datasets/west0497.mtx" );
	int ret = 0;
	if( west.m() != 497 ) {
		std::cerr << "west0497 has 497 rows, not " << west.m() << std::endl;
		ret = 1;
	}
	if( west.n() != 497 ) {
		std::cerr << "west0497 has 497 columns, not " << west.n() << std::endl;
		ret = 2;
	}
	if( west.nz() != 1727 ) {
		std::cerr << "west0497 has 1727 nonzeroes, not " << west.nz() << std::endl;
		ret = 3;
	}
	if( west.isPattern() == true ) {
		std::cerr << "west0497 is not a pattern matrix, yet it is detected to "
					 "be one."
				  << std::endl;
		ret = 4;
	}
	if( west.isSymmetric() == true ) {
		std::cerr << "west0497 is not a symmetric matrix, yet it is detected "
					 "to be one."
				  << std::endl;
		ret = 5;
	}
	if( west.usesDirectAddressing() == false ) {
		std::cerr << "west0497 should be read with direct addressing, not an "
					 "indirect one."
				  << std::endl;
		ret = 6;
	}
	size_t count = 0;
	for( auto nonzero : west ) {
		(void)nonzero;
		++count;
	}
	if( count != west.nz() ) {
		std::cerr << "Iterator does not contain " << west.nz() << " nonzeroes. It instead iterated over " << count << " nonzeroes." << std::endl;
		ret = 7;
	}
	auto base_it = west.begin( grb::SEQUENTIAL, []( double & val ) {
		val = 1;
	} );
	count = 0;
	size_t count_converted = 0;
	for( ; base_it != west.end(); ++base_it ) {
		count_converted += static_cast< size_t >( ( *base_it ).second );
		++count;
	}
	if( count != west.nz() ) {
		std::cerr << "Iterator (non-auto) does not contain " << west.nz() << " nonzeroes. It instead iterated over " << count << " nonzeroes." << std::endl;
		ret = 8;
	}
	if( count != count_converted ) {
		std::cerr << "Reader converter failed." << std::endl;
		ret = 9;
	}

	if( ret == 0 ) {
		std::cout << "Test OK.\n" << std::endl;
	} else {
		std::cout << "Test FAILED.\n" << std::endl;
	}
	return ret;
}
