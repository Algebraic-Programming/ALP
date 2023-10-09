
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
#include <sstream>

#include <graphblas/algorithms/matrix_factory.hpp>

#include <graphblas.hpp>

using namespace grb;

namespace {
	static RC error( const std::string & msg ) {
		std::cerr << "Test FAILED:" << msg << std::endl;
		return FAILED;
	}
} // namespace

static RC test_factory_empty( const size_t & n ) {
	{ // grb::factory::empty<void> of size: [0,0]
		Matrix< void > M = factory::empty< void >( 0, 0, IOMode::SEQUENTIAL );
		if( nnz( M ) != 0 ) {
			return error( "grb::factory::empty<void>, size=(0,0): nnz != 0" );
		} else if( nrows( M ) != 0 ) {
			return error( "grb::factory::empty<void>, size=(0,0): nrows != 0" );
		} else if( ncols( M ) != 0 ) {
			return error( "grb::factory::empty<void>, size=(0,0): ncols != 0" );
		}
		for( const auto & e : M ) {
			(void)e;
			return error( "grb::factory::empty<void>, size=(0,0): found a value" );
		}
	}

	{ // grb::factory::empty<int> of size: [0,0]
		Matrix< int > M = factory::empty< int >( 0, 0, IOMode::SEQUENTIAL );
		if( nnz( M ) != 0 ) {
			return error( "grb::factory::empty<void>, size=(0,0): nnz != 0" );
		} else if( nrows( M ) != 0 ) {
			return error( "grb::factory::empty<void>, size=(0,0): nrows != 0" );
		} else if( ncols( M ) != 0 ) {
			return error( "grb::factory::empty<void>, size=(0,0): ncols != 0" );
		}
		for( const auto & e : M ) {
			(void)e;
			return error( "grb::factory::empty<void>, size=(0,0): found a value" );
		}
	}

	{ // grb::factory::empty<void> of size: [n,n]
		Matrix< void > M = factory::empty< void >( n, n, IOMode::SEQUENTIAL );
		if( nnz( M ) != 0 ) {
			return error( "grb::factory::empty<void>, size=(n,n): nnz != 0" );
		} else if( nrows( M ) != n ) {
			return error( "grb::factory::empty<void>, size=(n,n): nrows != n" );
		} else if( ncols( M ) != n ) {
			return error( "grb::factory::empty<void>, size=(n,n): ncols != n" );
		}
		for( const auto & e : M ) {
			(void)e;
			return error( "grb::factory::empty<void>, size=(n,n): found a value" );
		}
	}

	{ // grb::factory::empty<int> of size: [n,n]
		Matrix< int > M = factory::empty< int >( n, n, IOMode::SEQUENTIAL );
		if( nnz( M ) != 0 ) {
			return error( "grb::factory::empty<void>, size=(n,n): nnz != 0" );
		} else if( nrows( M ) != n ) {
			return error( "grb::factory::empty<void>, size=(n,n): nrows != n" );
		} else if( ncols( M ) != n ) {
			return error( "grb::factory::empty<void>, size=(n,n): ncols != n" );
		}
		for( const auto & e : M ) {
			(void)e;
			return error( "grb::factory::empty<void>, size=(n,n): found a value" );
		}
	}

	return SUCCESS;
}

static RC test_factory_identity( const size_t & n ) {
	{ // grb::factory::identity<void> of size: [0,0]
		Matrix< void > M = factory::identity< void >( 0, IOMode::SEQUENTIAL );
		if( nnz( M ) != 0 ) {
			return error( "grb::factory::identity<void>, size=(0,0): nnz != 0" );
		} else if( nrows( M ) != 0 ) {
			return error( "grb::factory::identity<void>, size=(0,0): nrows != 0" );
		} else if( ncols( M ) != 0 ) {
			return error( "grb::factory::identity<void>, size=(0,0): ncols != 0" );
		}
		for( const auto & e : M ) {
			(void)e;
			return error( "grb::factory::identity<void>, size=(0,0): found a value" );
		}
	}

	{ // grb::factory::identity<int> of size: [0,0]
		Matrix< int > M = factory::identity< int >( 0, IOMode::SEQUENTIAL, 2 );
		if( nnz( M ) != 0 ) {
			return error( "grb::factory::identity<void>, size=(0,0): nnz != 0" );
		} else if( nrows( M ) != 0 ) {
			return error( "grb::factory::identity<void>, size=(0,0): nrows != 0" );
		} else if( ncols( M ) != 0 ) {
			return error( "grb::factory::identity<void>, size=(0,0): ncols != 0" );
		}
		for( const auto & e : M ) {
			(void)e;
			return error( "grb::factory::identity<void>, size=(0,0): found a value" );
		}
	}

	{ // grb::factory::identity<void>
		Matrix< void > M = factory::identity< void >( n, IOMode::SEQUENTIAL );
		if( nnz( M ) != n ) {
			return error( "grb::factory::identity<void>: nnz != n" );
		} else if( nrows( M ) != n ) {
			return error( "grb::factory::identity<void>: nrows != n" );
		} else if( ncols( M ) != n ) {
			return error( "grb::factory::identity<void>: ncols != n" );
		}
		std::vector< bool > touched( n, false );
		for( const auto & e : M ) {
			touched[ e.first ] = true;
			if( e.first != e.second ) {
				return error( "grb::factory::identity<void>: incorrect coordinate" );
			}
		}
		if( std::find( touched.cbegin(), touched.cend(), false ) != touched.cend() ) {
			return error( "grb::factory::identity<void>: not touched every elements" );
		}
	}

	{ // grb::factory::identity<int>
		Matrix< int > M = factory::identity< int >( n, IOMode::SEQUENTIAL, 2 );
		if( nnz( M ) != n ) {
			return error( "grb::factory::identity<int>: nnz != n" );
		} else if( nrows( M ) != n ) {
			return error( "grb::factory::identity<int>: nrows != n" );
		} else if( ncols( M ) != n ) {
			return error( "grb::factory::identity<int>: ncols != n" );
		}
		std::vector< bool > touched( n, false );
		for( const auto & e : M ) {
			touched[ e.first.first ] = true;
			if( e.first.first != e.first.second ) {
				return error( "grb::factory::identity<int>: incorrect coordinate" );
			} else if( e.second != 2 ) {
				return error( "grb::factory::identity<int>: incorrect value" );
			}
		}
		if( std::find( touched.cbegin(), touched.cend(), false ) != touched.cend() ) {
			return error( "grb::factory::identity<void>: not touched every elements" );
		}
	}

	return SUCCESS;
}

static RC test_factory_eye( const size_t & n ) {
	{ // grb::factory::eye<void> of size: [0,0]
		Matrix< void > M = factory::eye< void >( 0, 0, IOMode::SEQUENTIAL );
		if( nnz( M ) != 0 ) {
			return error( "grb::factory::eye<void>, size=(0,0): nnz != 0" );
		} else if( nrows( M ) != 0 ) {
			return error( "grb::factory::eye<void>, size=(0,0): nrows != 0" );
		} else if( ncols( M ) != 0 ) {
			return error( "grb::factory::eye<void>, size=(0,0): ncols != 0" );
		}
		for( const auto & e : M ) {
			(void)e;
			return error( "grb::factory::eye<void>, size=(0,0): found a value" );
		}
	}

	{ // grb::factory::eye<int> of size: [0,0]
		Matrix< int > M = factory::eye< int >( 0, 0, IOMode::SEQUENTIAL );
		if( nnz( M ) != 0 ) {
			return error( "grb::factory::eye<void>, size=(0,0): nnz != 0" );
		} else if( nrows( M ) != 0 ) {
			return error( "grb::factory::eye<void>, size=(0,0): nrows != 0" );
		} else if( ncols( M ) != 0 ) {
			return error( "grb::factory::eye<void>, size=(0,0): ncols != 0" );
		}
		for( const auto & e : M ) {
			(void)e;
			return error( "grb::factory::eye<void>, size=(0,0): found a value" );
		}
	}

	{ // grb::factory::eye<void> of size: [n,n]
		Matrix< void > M = factory::eye< void >( n, n, IOMode::SEQUENTIAL );
		if( nnz( M ) != n ) {
			return error( "grb::factory::eye<void>, size=(n,n): nnz != n" );
		} else if( nrows( M ) != n ) {
			return error( "grb::factory::eye<void>, size=(n,n): nrows != n" );
		} else if( ncols( M ) != n ) {
			return error( "grb::factory::eye<void>, size=(n,n): ncols != n" );
		}
		std::vector< bool > touched( n, false );
		for( const auto & e : M ) {
			touched[ e.first ] = true;
			if( e.first != e.second ) {
				return error( "grb::factory::eye<void>, size=(n,n): incorrect coordinate" );
			}
		}
		if( std::find( touched.cbegin(), touched.cend(), false ) != touched.cend() ) {
			return error( "grb::factory::eye<void>, size=(n,n): not touched every elements" );
		}
	}

	{ // grb::factory::eye<int> of size: [n,n]
		Matrix< int > M = factory::eye< int >( n, n, IOMode::SEQUENTIAL, 2 );
		if( nnz( M ) != n ) {
			return error( "grb::factory::eye<int>, size=(n,n): nnz != n" );
		} else if( nrows( M ) != n ) {
			return error( "grb::factory::eye<int>, size=(n,n): nrows != n" );
		} else if( ncols( M ) != n ) {
			return error( "grb::factory::eye<int>, size=(n,n): ncols != n" );
		}
		std::vector< bool > touched( n, false );
		for( const auto & e : M ) {
			touched[ e.first.first ] = true;
			if( e.first.first != e.first.second ) {
				return error( "grb::factory::eye<int>, size=(n,n): incorrect coordinate" );
			} else if( e.second != 2 ) {
				return error( "grb::factory::eye<int>, size=(n,n): incorrect value" );
			}
		}
		if( std::find( touched.cbegin(), touched.cend(), false ) != touched.cend() ) {
			return error( "grb::factory::eye<void>, size=(n,n): not touched every elements" );
		}
	}

	{ // grb::factory::eye<int> of size: [1,n]
		Matrix< int > M = factory::eye< int >( 1, n, IOMode::SEQUENTIAL, 2 );
		if( nnz( M ) != 1 ) {
			return error( "grb::factory::eye<int>, size=(1,n): nnz != 1" );
		} else if( nrows( M ) != 1 ) {
			return error( "grb::factory::eye<int>, size=(1,n): nrows != 1" );
		} else if( ncols( M ) != n ) {
			return error( "grb::factory::eye<int>, size=(1,n): ncols != n" );
		}
		std::vector< bool > touched( 1, false );
		for( const auto & e : M ) {
			touched[ e.first.first ] = true;
			if( e.first.first != 0 ) {
				return error( "grb::factory::eye<int>, size=(1,n): incorrect coordinate" );
			} else if( e.second != 2 ) {
				return error( "grb::factory::eye<int>, size=(1,n): incorrect value" );
			}
		}
		if( std::find( touched.cbegin(), touched.cend(), false ) != touched.cend() ) {
			return error( "grb::factory::eye<void>, size=(1,n): not touched every elements" );
		}
	}

	{ // grb::factory::eye<int> of size: [n,1]
		Matrix< int > M = factory::eye< int >( n, 1, IOMode::SEQUENTIAL, 2 );
		if( nnz( M ) != 1 ) {
			return error( "grb::factory::eye<int>, size=(n,1): nnz != 1" );
		} else if( nrows( M ) != n ) {
			return error( "grb::factory::eye<int>, size=(n,1): nrows != n" );
		} else if( ncols( M ) != 1 ) {
			return error( "grb::factory::eye<int>, size=(n,1): ncols != 1" );
		}
		std::vector< bool > touched( 1, false );
		for( const auto & e : M ) {
			touched[ e.first.first ] = true;
			if( e.first.second != 0 ) {
				return error( "grb::factory::eye<int>, size=(n,1): incorrect column coordinate" );

			} else if( e.second != 2 ) {
				return error( "grb::factory::eye<int>, size=(n,1): incorrect value" );
			}
		}
		if( std::find( touched.cbegin(), touched.cend(), false ) != touched.cend() ) {
			return error( "grb::factory::eye<int>, size=(n,1): not touched every elements" );
		}
	}

	return SUCCESS;
}

template< typename VoidFactoryFunc, typename IntFactoryFunc >
RC test_factory_full_templated( const size_t & n, const std::string & factoryName, const VoidFactoryFunc & voidFactory, const IntFactoryFunc & intFactory ) {
	{ // grb::factory::voidFactory of size: [0,0]
		Matrix< void > M = voidFactory( 0, 0, IOMode::SEQUENTIAL );
		if( nnz( M ) != 0 ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): nnz != 0" );
		} else if( nrows( M ) != 0 ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): nrows != 0" );
		} else if( ncols( M ) != 0 ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): ncols != 0" );
		}
		for( const auto & e : M ) {
			(void)e;
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): found a value" );
		}
	}

	{ // grb::factory::intFactory of size: [0,0]
		Matrix< int > M = intFactory( 0, 0, IOMode::SEQUENTIAL, 2 );
		if( nnz( M ) != 0 ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): nnz != 0" );
		} else if( nrows( M ) != 0 ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): nrows != 0" );
		} else if( ncols( M ) != 0 ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): ncols != 0" );
		}
		for( const auto & e : M ) {
			(void)e;
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): found a value" );
		}
	}

	{ // grb::factory::voidFactory of size: [n,n]
		Matrix< void > M = voidFactory( n, n, IOMode::SEQUENTIAL );
		if( nnz( M ) != n * n ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(n,n): nnz != n*n" );
		} else if( nrows( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(n,n): nrows != n" );
		} else if( ncols( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(n,n): ncols != n" );
		}
		std::vector< bool > touched( n * n, false );
		for( const auto & e : M ) {
			touched[ e.first * n + e.second ] = true;
		}
		if( std::find( touched.cbegin(), touched.cend(), false ) != touched.cend() ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(n,n): not touched every elements" );
		}
	}

	{ // grb::factory::intFactory of size: [n,n]
		Matrix< int > M = intFactory( n, n, IOMode::SEQUENTIAL, 2 );
		if( nnz( M ) != n * n ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,n): nnz != n*n" );
		} else if( nrows( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,n): nrows != n" );
		} else if( ncols( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,n): ncols != n" );
		}
		std::vector< bool > touched( n * n, false );
		for( const auto & e : M ) {
			touched[ e.first.first * n + e.first.second ] = true;
			if( e.second != 2 ) {
				return error( "grb::factory::" + factoryName + "<int>, size=(n,n): incorrect value" );
			}
		}
		if( std::find( touched.cbegin(), touched.cend(), false ) != touched.cend() ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,n): not touched every elements" );
		}
	}

	{ // grb::factory::intFactory of size: [1,n]
		Matrix< int > M = intFactory( 1, n, IOMode::SEQUENTIAL, 2 );
		if( nnz( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(1,n): nnz != n" );
		} else if( nrows( M ) != 1 ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(1,n): nrows != 1" );
		} else if( ncols( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(1,n): ncols != n" );
		}
		std::vector< bool > touched( n, false );
		for( const auto & e : M ) {
			touched[ e.first.second ] = true;
			if( e.first.first != 0 ) {
				return error( "grb::factory::" + factoryName + "<int>, size=(1,n): incorrect row coordinate" );
			} else if( e.second != 2 ) {
				return error( "grb::factory::" + factoryName + "<int>, size=(1,n): incorrect value" );
			}
		}
		if( std::find( touched.cbegin(), touched.cend(), false ) != touched.cend() ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(1,n): not touched every elements" );
		}
	}

	{ // grb::factory::intFactory of size: [n,1]
		Matrix< int > M = intFactory( n, 1, IOMode::SEQUENTIAL, 2 );
		if( nnz( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,1): nnz != n" );
		} else if( nrows( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,1): nrows != n" );
		} else if( ncols( M ) != 1 ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,1): ncols != 1" );
		}
		std::vector< bool > touched( n, false );
		for( const auto & e : M ) {
			touched[ e.first.first ] = true;
			if( e.first.second != 0 ) {
				return error( "grb::factory::" + factoryName + "<int>, size=(n,1): incorrect column coordinate" );
			} else if( e.second != 2 ) {
				return error( "grb::factory::" + factoryName + "<int>, size=(n,1): incorrect value" );
			}
		}
		if( std::find( touched.cbegin(), touched.cend(), false ) != touched.cend() ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,1): not touched every elements" );
		}
	}

	return SUCCESS;
}

static RC test_factory_dense( const size_t & n ) {
	return test_factory_full_templated(
		n, "dense",
		[ & ]( size_t r, size_t c, IOMode m ) {
			return factory::dense< void >( r, c, m );
		},
		[ & ]( size_t r, size_t c, IOMode m, int v ) {
			return factory::dense< int >( r, c, m, v );
		} );
}

static RC test_factory_full( const size_t & n ) {
	return test_factory_full_templated(
		n, "full",
		[ & ]( size_t r, size_t c, IOMode m ) {
			return factory::full< void >( r, c, m );
		},
		[ & ]( size_t r, size_t c, IOMode m, int v ) {
			return factory::full< int >( r, c, m, v );
		} );
}

template< typename VoidFactoryFunc, typename IntFactoryFunc >
static RC test_factory_dense_valued( const size_t & n, const std::string & factoryName, const VoidFactoryFunc & voidFactory, const IntFactoryFunc & intFactory, const int expectedValue ) {
	{ // grb::factory::voidFactory of size: [0,0]
		Matrix< void > M = voidFactory( 0, 0, IOMode::SEQUENTIAL );
		if( nnz( M ) != 0 ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): nnz != 0" );
		} else if( nrows( M ) != 0 ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): nrows != 0" );
		} else if( ncols( M ) != 0 ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): ncols != 0" );
		}
		for( const auto & e : M ) {
			(void)e;
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): found a value" );
		}
	}

	{ // grb::factory::intFactory of size: [0,0]
		Matrix< int > M = intFactory( 0, 0, IOMode::SEQUENTIAL );
		if( nnz( M ) != 0 ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): nnz != 0" );
		} else if( nrows( M ) != 0 ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): nrows != 0" );
		} else if( ncols( M ) != 0 ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): ncols != 0" );
		}
		for( const auto & e : M ) {
			(void)e;
			return error( "grb::factory::" + factoryName + "<void>, size=(0,0): found a value" );
		}
	}

	{ // grb::factory::voidFactory of size: [n,n]
		Matrix< void > M = voidFactory( n, n, IOMode::SEQUENTIAL );
		if( nnz( M ) != n * n ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(n,n): nnz != n*n" );
		} else if( nrows( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(n,n): nrows != n" );
		} else if( ncols( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(n,n): ncols != n" );
		}
		std::vector< bool > touched( n * n, false );
		for( const auto & e : M ) {
			touched[ e.first * n + e.second ] = true;
		}
		if( std::find( touched.cbegin(), touched.cend(), false ) != touched.cend() ) {
			return error( "grb::factory::" + factoryName + "<void>, size=(n,n): not touched every elements" );
		}
	}

	{ // grb::factory::intFactory of size: [n,n]
		Matrix< int > M = intFactory( n, n, IOMode::SEQUENTIAL );
		if( nnz( M ) != n * n ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,n): nnz != n*n" );
		} else if( nrows( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,n): nrows != n" );
		} else if( ncols( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,n): ncols != n" );
		}
		std::vector< bool > touched( n * n, false );
		for( const auto & e : M ) {
			touched[ e.first.first * n + e.first.second ] = true;
			if( e.second != expectedValue ) {
				return error( "grb::factory::eye<int>, size=(n,n): incorrect value" );
			}
		}
		if( std::find( touched.cbegin(), touched.cend(), false ) != touched.cend() ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,n): not touched every elements" );
		}
	}

	{ // grb::factory::intFactory of size: [1,n]
		Matrix< int > M = intFactory( 1, n, IOMode::SEQUENTIAL );
		if( nnz( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(1,n): nnz != n" );
		} else if( nrows( M ) != 1 ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(1,n): nrows != 1" );
		} else if( ncols( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(1,n): ncols != n" );
		}
		std::vector< bool > touched( n, false );
		for( const auto & e : M ) {
			touched[ e.first.second ] = true;
			if( e.first.first != 0 ) {
				return error( "grb::factory::" + factoryName + "<int>, size=(1,n): incorrect row coordinate" );
			} else if( e.second != expectedValue ) {
				return error( "grb::factory::" + factoryName + "<int>, size=(1,n): incorrect value" );
			}
		}
		if( std::find( touched.cbegin(), touched.cend(), false ) != touched.cend() ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(1,n): not touched every elements" );
		}
	}

	{ // grb::factory::intFactory of size: [n,1]
		Matrix< int > M = intFactory( n, 1, IOMode::SEQUENTIAL );
		if( nnz( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,1): nnz != n" );
		} else if( nrows( M ) != n ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,1): nrows != n" );
		} else if( ncols( M ) != 1 ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,1): ncols != 1" );
		}
		std::vector< bool > touched( n, false );
		for( const auto & e : M ) {
			touched[ e.first.first ] = true;
			if( e.first.second != 0 ) {
				return error( "grb::factory::" + factoryName + "<int>, size=(n,1): incorrect column coordinate" );
			} else if( e.second != expectedValue ) {
				return error( "grb::factory::" + factoryName + "<int>, size=(n,1): incorrect value" );
			}
		}
		if( std::find( touched.cbegin(), touched.cend(), false ) != touched.cend() ) {
			return error( "grb::factory::" + factoryName + "<int>, size=(n,1): not touched every elements" );
		}
	}

	return SUCCESS;
}

static RC test_factory_zeros( const size_t & n ) {
	return test_factory_dense_valued(
		n, "zeros",
		[ & ]( size_t r, size_t c, IOMode m ) {
			return factory::zeros< void >( r, c, m );
		},
		[ & ]( size_t r, size_t c, IOMode m ) {
			return factory::zeros< int >( r, c, m );
		},
		1 );
}

static RC test_factory_ones( const size_t & n ) {
	return test_factory_dense_valued(
		n, "ones",
		[ & ]( size_t r, size_t c, IOMode m ) {
			return factory::ones< void >( r, c, m );
		},
		[ & ]( size_t r, size_t c, IOMode m ) {
			return factory::ones< int >( r, c, m );
		},
		1 );
}

void grb_program( const size_t & n, grb::RC & rc ) {
	rc = SUCCESS;
	rc = rc != SUCCESS ? rc : test_factory_empty( n );
	rc = rc != SUCCESS ? rc : test_factory_identity( n );
	rc = rc != SUCCESS ? rc : test_factory_eye( n );
	rc = rc != SUCCESS ? rc : test_factory_dense( n );
	rc = rc != SUCCESS ? rc : test_factory_full( n );
	rc = rc != SUCCESS ? rc : test_factory_zeros( n );
	rc = rc != SUCCESS ? rc : test_factory_ones( n );
}

int main( int argc, char ** argv ) {
	// defaults
	size_t in = 100;

	// error checking
	if( argc > 2 ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is " << in << "): a positive integer.\n";
		return 1;
	}
	if( argc == 2 ) {
		in = atoi( argv[ 1 ] );
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
