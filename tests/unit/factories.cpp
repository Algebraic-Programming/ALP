
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

#include <algorithm>
#include <functional>
#include <iostream>

#include <graphblas.hpp>
#include <graphblas/algorithms/matrix_factory.hpp>


using namespace grb;
using namespace grb::algorithms;

namespace {
	RC error( const std::string &msg ) {
		std::cerr << "Test FAILED: " << msg << std::endl;
		return FAILED;
	}
} // namespace

static RC test_factory_empty( const size_t &n ) {
	{ // matrices< void >::empty of size: [0,0]
		Matrix< void > M = matrices< void >::empty( 0, 0 );
		if( nnz( M ) != 0 ) {
			return error( "matrices< void >::empty, size=(0,0): nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			return error( "matrices< void >::empty, size=(0,0): nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			return error( "matrices< void >::empty, size=(0,0): ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			return error( "matrices< void >::empty, size=(0,0): found a value" );
		}
	}

	{ // matrices< int >::empty of size: [0,0]
		Matrix< int > M = matrices< int >::empty( 0, 0 );
		if( nnz( M ) != 0 ) {
			return error( "matrices< int >::empty, size=(0,0): nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			return error( "matrices< int >::empty, size=(0,0): nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			return error( "matrices< int >::empty, size=(0,0): ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			return error( "matrices< int >::empty, size=(0,0): found a value" );
		}
	}

	{ // matrices< void >::empty of size: [n,n]
		Matrix< void > M = matrices< void >::empty( n, n );
		if( nnz( M ) != 0 ) {
			return error( "matrices< void >::empty, size=(n,n): nnz != 0" );
		}
		if( nrows( M ) != n ) {
			return error( "matrices< void >::empty, size=(n,n): nrows != n" );
		}
		if( ncols( M ) != n ) {
			return error( "matrices< void >::empty, size=(n,n): ncols != n" );
		}
		for( const auto &e : M ) {
			(void) e;
			return error( "matrices< void >::empty, size=(n,n): found a value" );
		}
	}

	{ // matrices< int >::empty of size: [n,n]
		Matrix< int > M = matrices< int >::empty( n, n );
		if( nnz( M ) != 0 ) {
			return error( "matrices< int >::empty, size=(n,n): nnz != 0" );
		}
		if( nrows( M ) != n ) {
			return error( "matrices< int >::empty, size=(n,n): nrows != n" );
		}
		if( ncols( M ) != n ) {
			return error( "matrices< int >::empty, size=(n,n): ncols != n" );
		}
		for( const auto &e : M ) {
			(void) e;
			return error( "matrices< int >::empty, size=(n,n): found a value" );
		}
	}

	return SUCCESS;
}

static RC test_factory_identity( const size_t &n, const long &offset ) {
	const size_t i_offset = offset > 0 ? offset : 0;
	const size_t j_offset = offset < 0 ? (-offset) : 0;
	const size_t expected_nnz = i_offset + j_offset < n
		? n - i_offset - j_offset
		: 0;
	{ // matrices< void >::identity of size: [0,0]
		Matrix< void > M = matrices< void >::identity( 0, offset );
		if( nnz( M ) != 0 ) {
			return error( "matrices< void >::identity, size=(0,0): nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			return error( "matrices< void >::identity, size=(0,0): nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			return error( "matrices< void >::identity, size=(0,0): ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			return error( "matrices< void >::identity, size=(0,0): found a value" );
		}
	}

	{ // matrices< int >::identity of size: [0,0]
		Matrix< int > M = matrices< int >::identity( 0, 2, offset );
		if( nnz( M ) != 0 ) {
			return error( "matrices< int >::identity, size=(0,0): nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			return error( "matrices< int >::identity, size=(0,0): nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			return error( "matrices< int >::identity, size=(0,0): ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			return error( "matrices< int >::identity, size=(0,0): found a value" );
		}
	}

	{ // matrices< void >::identity
		Matrix< void > M = matrices< void >::identity( n, offset );
		if( nnz( M ) != expected_nnz ) {
			return error( "matrices< void >::identity: nnz != n-abs(k)" );
		}
		if( nrows( M ) != n ) {
			return error( "matrices< void >::identity: nrows != n" );
		}
		if( ncols( M ) != n ) {
			return error( "matrices< void >::identity: ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.first + i_offset != e.second + j_offset ) {
				std::cerr << "matrices< void >::identity: incorrect coordinate ( "
					<< i_offset << "|"
					<< e.first << ", " << e.second << " )\n";
				return FAILED;
			}
		}
	}

	{ // matrices< int >::identity
		Matrix< int > M = matrices< int >::identity( n, 2, offset );
		if( nnz( M ) != expected_nnz ) {
			return error( "matrices< int >::identity: nnz != n-abs(k)" );
		}
		if( nrows( M ) != n ) {
			return error( "matrices< int >::identity: nrows != n" );
		}
		if( ncols( M ) != n ) {
			return error( "matrices< int >::identity: ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.first.first + i_offset != e.first.second + j_offset ) {
				return error( "matrices< int >::identity: incorrect coordinate" );
			}
			if( e.second != 2 ) {
				return error( "matrices< int >::identity: incorrect value" );
			}
		}
	}

	return SUCCESS;
}

static RC test_factory_eye( const size_t &n ) {
	{ // matrices< void >::eye of size: [0,0]
		Matrix< void > M = matrices< void >::eye( 0, 0 );
		if( nnz( M ) != 0 ) {
			return error( "matrices< void >::eye, size=(0,0): nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			return error( "matrices< void >::eye, size=(0,0): nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			return error( "matrices< void >::eye, size=(0,0): ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			return error( "matrices< void >::eye, size=(0,0): found a value" );
		}
	}

	{ // matrices< int >::eye of size: [0,0]
		Matrix< int > M = matrices< int >::eye( 0, 0 );
		if( nnz( M ) != 0 ) {
			return error( "matrices< int >::eye, size=(0,0): nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			return error( "matrices< int >::eye, size=(0,0): nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			return error( "matrices< int >::eye, size=(0,0): ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			return error( "matrices< int >::eye, size=(0,0): found a value" );
		}
	}

	{ // matrices< void >::eye of size: [n,n]
		Matrix< void > M = matrices< void >::eye( n, n );
		if( nnz( M ) != n ) {
			return error( "matrices< void >::eye, size=(n,n): nnz != n" );
		}
		if( nrows( M ) != n ) {
			return error( "matrices< void >::eye, size=(n,n): nrows != n" );
		}
		if( ncols( M ) != n ) {
			return error( "matrices< void >::eye, size=(n,n): ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.first != e.second ) {
				return error( "matrices< void >::eye, size=(n,n): incorrect coordinate" );
			}
		}
	}

	{ // matrices< int >::eye of size: [n,n]
		Matrix< int > M = matrices< int >::eye( n, n, 2 );
		if( nnz( M ) != n ) {
			return error( "matrices< int >::eye, size=(n,n): nnz != n" );
		}
		if( nrows( M ) != n ) {
			return error( "matrices< int >::eye, size=(n,n): nrows != n" );
		}
		if( ncols( M ) != n ) {
			return error( "matrices< int >::eye, size=(n,n): ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.first.first != e.first.second ) {
				return error( "matrices< int >::eye, size=(n,n): incorrect coordinate" );
			}
			if( e.second != 2 ) {
				return error( "matrices< int >::eye, size=(n,n): incorrect value" );
			}
		}
	}

	{ // matrices< int >::eye of size: [1,n]
		Matrix< int > M = matrices< int >::eye( 1, n, 2 );
		if( nnz( M ) != 1 ) {
			return error( "matrices< int >::eye, size=(1,n): nnz != 1" );
		}
		if( nrows( M ) != 1 ) {
			return error( "matrices< int >::eye, size=(1,n): nrows != 1" );
		}
		if( ncols( M ) != n ) {
			return error( "matrices< int >::eye, size=(1,n): ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.first.first != 0 ) {
				return error( "matrices< int >::eye, size=(1,n): incorrect coordinate" );
			}
			if( e.second != 2 ) {
				return error( "matrices< int >::eye, size=(1,n): incorrect value" );
			}
		}
	}

	{ // matrices< int >::eye of size: [n,1]
		Matrix< int > M = matrices< int >::eye( n, 1, 2 );
		if( nnz( M ) != 1 ) {
			return error( "matrices< int >::eye, size=(n,1): nnz != 1" );
		}
		if( nrows( M ) != n ) {
			return error( "matrices< int >::eye, size=(n,1): nrows != n" );
		}
		if( ncols( M ) != 1 ) {
			return error( "matrices< int >::eye, size=(n,1): ncols != 1" );
		}
		for( const auto &e : M ) {
			if( e.first.second != 0 ) {
				return error( "matrices< int >::eye, size=(n,1): "
					"incorrect column coordinate" );

			}
			if( e.second != 2 ) {
				return error( "matrices< int >::eye, size=(n,1): incorrect value" );
			}
		}
	}

	return SUCCESS;
}

template< typename VoidFactoryFunc, typename IntFactoryFunc >
RC test_factory_full_templated(
	const size_t &n,
	const std::string &factoryName,
	const VoidFactoryFunc &voidFactory,
	const IntFactoryFunc &intFactory
) {
	{ // matrices::voidFactory of size: [0,0]
		Matrix< void > M = voidFactory( 0, 0 );
		if( nnz( M ) != 0 ) {
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"found a value" );
		}
	}

	{ // matrices::intFactory of size: [0,0]
		Matrix< int > M = intFactory( 0, 0, 2 );
		if( nnz( M ) != 0 ) {
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"found a value" );
		}
	}

	{ // matrices::voidFactory of size: [n,n]
		Matrix< void > M = voidFactory( n, n );
		if( nnz( M ) != n * n ) {
			return error( "matrices::" + factoryName + "<void>, size=(n,n): "
				"nnz != n*n" );
		}
		if( nrows( M ) != n ) {
			return error( "matrices::" + factoryName + "<void>, size=(n,n): "
				"nrows != n" );
		}
		if( ncols( M ) != n ) {
			return error( "matrices::" + factoryName + "<void>, size=(n,n): "
				"ncols != n" );
		}
	}

	{ // matrices::intFactory of size: [n,n]
		Matrix< int > M = intFactory( n, n, 2 );
		if( nnz( M ) != n * n ) {
			return error( "matrices::" + factoryName + "<int>, size=(n,n): "
				"nnz != n*n" );
		}
		if( nrows( M ) != n ) {
			return error( "matrices::" + factoryName + "<int>, size=(n,n): "
				"nrows != n" );
		}
		if( ncols( M ) != n ) {
			return error( "matrices::" + factoryName + "<int>, size=(n,n): "
				"ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.second != 2 ) {
				return error( "matrices::" + factoryName + "<int>, size=(n,n): "
					"incorrect value" );
			}
		}
	}

	{ // matrices::intFactory of size: [1,n]
		Matrix< int > M = intFactory( 1, n, 2 );
		if( nnz( M ) != n ) {
			return error( "matrices::" + factoryName + "<int>, size=(1,n): "
				"nnz != n" );
		}
		if( nrows( M ) != 1 ) {
			return error( "matrices::" + factoryName + "<int>, size=(1,n): "
				"nrows != 1" );
		}
		if( ncols( M ) != n ) {
			return error( "matrices::" + factoryName + "<int>, size=(1,n): "
				"ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.first.first != 0 ) {
				return error( "matrices::" + factoryName + "<int>, size=(1,n): "
					"incorrect row coordinate" );
			}
			if( e.second != 2 ) {
				return error( "matrices::" + factoryName + "<int>, size=(1,n): "
					"incorrect value" );
			}
		}
	}

	{ // matrices::intFactory of size: [n,1]
		Matrix< int > M = intFactory( n, 1, 2 );
		if( nnz( M ) != n ) {
			return error( "matrices::" + factoryName + "<int>, size=(n,1): "
				"nnz != n" );
		}
		if( nrows( M ) != n ) {
			return error( "matrices::" + factoryName + "<int>, size=(n,1): "
				"nrows != n" );
		}
		if( ncols( M ) != 1 ) {
			return error( "matrices::" + factoryName + "<int>, size=(n,1): "
				"ncols != 1" );
		}
		for( const auto &e : M ) {
			if( e.first.second != 0 ) {
				return error( "matrices::" + factoryName + "<int>, size=(n,1): "
					"incorrect column coordinate" );
			}
			if( e.second != 2 ) {
				return error( "matrices::" + factoryName + "<int>, size=(n,1): "
					"incorrect value" );
			}
		}
	}

	return SUCCESS;
}

static RC test_factory_dense( const size_t &n ) {
	return test_factory_full_templated(
		n, "dense",
		[&](const size_t r, const size_t c ) {
			return matrices< void >::dense( r, c );
		},
		[&](const size_t r, const size_t c, const int v ) {
			return matrices< int >::dense( r, c, v );
		} );
}

static RC test_factory_full( const size_t & n ) {
	return test_factory_full_templated(
		n, "full",
		[&](const size_t r, const size_t c ) {
			return matrices< void >::full( r, c );
		},
		[&](const size_t r, const size_t c, const int v ) {
			return matrices< int >::full( r, c, v );
		} );
}

template< typename VoidFactoryFunc, typename IntFactoryFunc >
static RC test_factory_dense_valued(
	const size_t &n,
	const std::string &factoryName,
	const VoidFactoryFunc &voidFactory,
	const IntFactoryFunc &intFactory,
	const int expectedValue
) {
	{ // matrices::voidFactory of size: [0,0]
		Matrix< void > M = voidFactory( 0, 0 );
		if( nnz( M ) != 0 ) {
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"found a value" );
		}
	}

	{ // matrices::intFactory of size: [0,0]
		Matrix< int > M = intFactory( 0, 0 );
		if( nnz( M ) != 0 ) {
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			return error( "matrices::" + factoryName + "<void>, size=(0,0): "
				"found a value" );
		}
	}

	{ // matrices::voidFactory of size: [n,n]
		Matrix< void > M = voidFactory( n, n );
		if( nnz( M ) != n * n ) {
			return error( "matrices< void >::" + factoryName + ", size=(n,n): "
				"nnz = " + std::to_string(nnz( M )) + " != n*n = " + std::to_string(n*n) );
		}
		if( nrows( M ) != n ) {
			return error( "matrices< void >::" + factoryName + ", size=(n,n): "
				"nrows != n" );
		}
		if( ncols( M ) != n ) {
			return error( "matrices< void >::" + factoryName + ", size=(n,n): "
				"ncols != n" );
		}
	}

	{ // matrices::intFactory of size: [n,n]
		Matrix< int > M = intFactory( n, n );
		if( nnz( M ) != n * n ) {
			return error( "matrices::" + factoryName + "<int>, size=(n,n): "
				"nnz != n*n" );
		}
		if( nrows( M ) != n ) {
			return error( "matrices::" + factoryName + "<int>, size=(n,n): "
				"nrows != n" );
		}
		if( ncols( M ) != n ) {
			return error( "matrices::" + factoryName + "<int>, size=(n,n): "
				"ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.second != expectedValue ) {
				return error( "matrices::eye<int>, size=(n,n): incorrect value" );
			}
		}
	}

	{ // matrices::intFactory of size: [1,n]
		Matrix< int > M = intFactory( 1, n );
		if( nnz( M ) != n ) {
			return error( "matrices::" + factoryName + "<int>, size=(1,n): "
				"nnz != n" );
		}
		if( nrows( M ) != 1 ) {
			return error( "matrices::" + factoryName + "<int>, size=(1,n): "
				"nrows != 1" );
		}
		if( ncols( M ) != n ) {
			return error( "matrices::" + factoryName + "<int>, size=(1,n): "
				"ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.first.first != 0 ) {
				return error( "matrices::" + factoryName + "<int>, size=(1,n): "
					"incorrect row coordinate" );
			}
			if( e.second != expectedValue ) {
				return error( "matrices::" + factoryName + "<int>, size=(1,n): "
					"incorrect value" );
			}
		}
	}

	{ // matrices::intFactory of size: [n,1]
		Matrix< int > M = intFactory( n, 1 );
		if( nnz( M ) != n ) {
			return error( "matrices::" + factoryName + "<int>, size=(n,1): "
				"nnz != n" );
		}
		if( nrows( M ) != n ) {
			return error( "matrices::" + factoryName + "<int>, size=(n,1): "
				"nrows != n" );
		}
		if( ncols( M ) != 1 ) {
			return error( "matrices::" + factoryName + "<int>, size=(n,1): "
				"ncols != 1" );
		}
		for( const auto &e : M ) {
			if( e.first.second != 0 ) {
				return error( "matrices::" + factoryName + "<int>, size=(n,1): "
					"incorrect column coordinate" );
			}
			if( e.second != expectedValue ) {
				return error( "matrices::" + factoryName + "<int>, size=(n,1): "
					"incorrect value" );
			}
		}
	}

	return SUCCESS;
}

static RC test_factory_zeros( const size_t &n ) {
	return test_factory_dense_valued(
		n, "zeros",
		[&](const size_t r, const size_t c ) {
			return matrices< void >::zeros( r, c );
		},
		[&](const size_t r, const size_t c ) {
			return matrices< int >::zeros( r, c );
		},
		0 );
}

static RC test_factory_ones( const size_t &n ) {
	return test_factory_dense_valued(
		n, "ones",
		[&](const size_t r, const size_t c ) {
			return matrices< void >::ones( r, c );
		},
		[&](const size_t r, const size_t c ) {
			return matrices< int >::ones( r, c );
		},
		1 );
}

void grb_program( const size_t &n, grb::RC &rc ) {
	rc = SUCCESS;
	std::cout << "Testing matrices::empty\n";
	rc = rc != SUCCESS ? rc : test_factory_empty( n );
	std::cout << "Testing matrices::identity\n";
	rc = rc != SUCCESS ? rc : test_factory_identity( n, 0 );
	std::cout << "Testing matrices::identity (1 offset)\n";
	rc = rc != SUCCESS ? rc : test_factory_identity( n, 1 );
	std::cout << "Testing matrices::identity (-2 offset)\n";
	rc = rc != SUCCESS ? rc : test_factory_identity( n, -2 );
	std::cout << "Testing matrices::identity (n offset)\n";
	rc = rc != SUCCESS ? rc : test_factory_identity( n, n );
	std::cout << "Testing matrices::eye\n";
	rc = rc != SUCCESS ? rc : test_factory_eye( n );
	std::cout << "Testing matrices::dense (direct)\n";
	rc = rc != SUCCESS ? rc : test_factory_dense( n );
	std::cout << "Testing matrices::full\n";
	rc = rc != SUCCESS ? rc : test_factory_full( n );
	std::cout << "Testing matrices::zeros\n";
	rc = rc != SUCCESS ? rc : test_factory_zeros( n );
	std::cout << "Testing matrices::ones\n";
	rc = rc != SUCCESS ? rc : test_factory_ones( n );
}

int main(const int argc, char ** argv ) {
	// defaults
	size_t in = 100;

	// error checking
	if( argc > 2 ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is " << in << "): "
			<< "a positive integer.\n";
		return 1;
	}
	if( argc >= 2 ) {
		in = std::strtoul( argv[ 1 ], nullptr, 0 );
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out;
	const grb::RC launch_rc = launcher.exec( &grb_program, in, out, true );
	if( launch_rc != grb::SUCCESS ) {
		std::cerr << "Test launcher reported error during call to exec\n";
		out = launch_rc;
	}
	if( out != SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED (" << grb::toString( out ) << ")\n" << std::endl;
	} else {
		std::cout << "Test OK\n" << std::endl;
	}
	return 0;
}

