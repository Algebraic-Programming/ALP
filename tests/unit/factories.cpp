
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
#include <utils/assertions.hpp>
#include <graphblas/algorithms/matrix_factory.hpp>


using namespace grb;
using namespace grb::algorithms;

#define ERROR( rc, msg ) {										\
	std::cerr << "  [Line " << __LINE__ << "] Test FAILED: "	\
		<< (msg) << std::endl;									\
	(rc) = FAILED;												\
}

/**
 * This function tests the 'empty' factory function for matrices.
 *
 * The 'empty' factory function creates an empty matrix of a given size.
 * This test function checks the correctness of the created matrix by verifying
 * its properties such as the number of non-zero elements, the number
 * of rows and columns.
 *
 * @param rc The return code of the test.
 * @param n The size of the matrix to be tested.
 *
 * @return Returns a RC (Return Code) indicating the success or failure
 * of the test.
 * If the test is successful, it returns SUCCESS.
 * If the test fails, it returns FAILED.
 */
static void test_factory_empty( RC &rc, const size_t &n ) {
	{ // matrices< void >::empty of size: [0,0]
		const Matrix< void > M = matrices< void >::empty( 0, 0 );
		if( nnz( M ) != 0 ) {
			ERROR( rc, "matrices< void >::empty, size=(0,0): nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			ERROR( rc, "matrices< void >::empty, size=(0,0): nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			ERROR( rc, "matrices< void >::empty, size=(0,0): ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			ERROR( rc, "matrices< void >::empty, size=(0,0): found a value" );
		}
	}

	{ // matrices< int >::empty of size: [0,0]
		const Matrix< int > M = matrices< int >::empty( 0, 0 );
		if( nnz( M ) != 0 ) {
			ERROR( rc, "matrices< int >::empty, size=(0,0): nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			ERROR( rc, "matrices< int >::empty, size=(0,0): nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			ERROR( rc, "matrices< int >::empty, size=(0,0): ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			ERROR( rc, "matrices< int >::empty, size=(0,0): found a value" );
		}
	}

	{ // matrices< void >::empty of size: [n,n]
		const Matrix< void > M = matrices< void >::empty( n, n );
		if( nnz( M ) != 0 ) {
			ERROR( rc, "matrices< void >::empty, size=(n,n): nnz != 0" );
		}
		if( nrows( M ) != n ) {
			ERROR( rc, "matrices< void >::empty, size=(n,n): nrows != n" );
		}
		if( ncols( M ) != n ) {
			ERROR( rc, "matrices< void >::empty, size=(n,n): ncols != n" );
		}
		for( const auto &e : M ) {
			(void) e;
			ERROR( rc, "matrices< void >::empty, size=(n,n): found a value" );
		}
	}

	{ // matrices< int >::empty of size: [n,n]
		const Matrix< int > M = matrices< int >::empty( n, n );
		if( nnz( M ) != 0 ) {
			ERROR( rc, "matrices< int >::empty, size=(n,n): nnz != 0" );
		}
		if( nrows( M ) != n ) {
			ERROR( rc, "matrices< int >::empty, size=(n,n): nrows != n" );
		}
		if( ncols( M ) != n ) {
			ERROR( rc, "matrices< int >::empty, size=(n,n): ncols != n" );
		}
		for( const auto &e : M ) {
			(void) e;
			ERROR( rc, "matrices< int >::empty, size=(n,n): found a value" );
		}
	}
}

/**
 * This function tests the 'identity' factory function for matrices.
 *
 * The 'identity' factory function creates an identity matrix of a given size
 * with a given offset. This test function checks the correctness of the
 * created matrix by verifying its properties such as the number of non-zero
 * elements, the number of rows and columns.
 *
 * @param rc The return code of the test.
 * @param n The size of the matrix to be tested.
 *
 * @return Returns a RC (Return Code) indicating the success or failure
 * of the test.
 * If the test is successful, it returns SUCCESS.
 * If the test fails, it returns FAILED.
 */
static void test_factory_identity( RC &rc, const size_t &n ) {
	const size_t expected_nnz = n;
	{ // matrices< void >::identity of size: [0,0]
		Matrix< void > M = matrices< void >::identity( 0 );
		if( nnz( M ) != 0 ) {
			ERROR( rc, "matrices< void >::identity, size=(0,0): nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			ERROR( rc, "matrices< void >::identity, size=(0,0): nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			ERROR( rc, "matrices< void >::identity, size=(0,0): ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			ERROR( rc, "matrices< void >::identity, size=(0,0): found a value" );
		}
	}

	{ // matrices< int >::identity of size: [0,0]
		Matrix< int > M = matrices< int >::identity( 0 );
		if( nnz( M ) != 0 ) {
			ERROR( rc, "matrices< int >::identity, size=(0,0): nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			ERROR( rc, "matrices< int >::identity, size=(0,0): nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			ERROR( rc, "matrices< int >::identity, size=(0,0): ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			ERROR( rc, "matrices< int >::identity, size=(0,0): found a value" );
		}
	}

	{ // matrices< void >::identity
		Matrix< void > M = matrices< void >::identity( n );
		if( nnz( M ) != expected_nnz ) {
			ERROR( rc, "matrices< void >::identity: nnz != n" );
		}
		if( nrows( M ) != n ) {
			ERROR( rc, "matrices< void >::identity: nrows != n" );
		}
		if( ncols( M ) != n ) {
			ERROR( rc, "matrices< void >::identity: ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.first != e.second ) {
				ERROR( rc, "matrices< void >::identity: incorrect coordinate ( "
					+ std::to_string(e.first) + ", " + std::to_string(e.second) + " )\n" );
			}
		}
	}

	{ // matrices< int >::identity
		Matrix< int > M = matrices< int >::identity( n );
		if( nnz( M ) != expected_nnz ) {
			ERROR( rc, "matrices< int >::identity: nnz != n" );
		}
		if( nrows( M ) != n ) {
			ERROR( rc, "matrices< int >::identity: nrows != n" );
		}
		if( ncols( M ) != n ) {
			ERROR( rc, "matrices< int >::identity: ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.first.first != e.first.second ) {
				ERROR( rc, "matrices< int >::identity: incorrect coordinate" );
			}
			if( e.second != 1 ) {
				ERROR( rc, "matrices< int >::identity: incorrect value" );
			}
		}
	}

	{ // matrices< double >::identity with non-standard semiring, empty
		Semiring<
			operators::min< double >, operators::add< double >,
			identities::infinity, identities::zero
		> minPlusFP64;
		Matrix< double > M = matrices< double >::identity( 0, minPlusFP64 );
		if( nnz( M ) != 0 ) {
			ERROR( rc, "matrices< double >::identity: nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			ERROR( rc, "matrices< double >::identity:: nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			ERROR( rc, "matrices< double >::identity:: ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			ERROR( rc, "matrices< double >::identity, size=(0,0): found a value" );
		}
	}

	{ // matrices< double >::identity with non-standard semiring
		Semiring<
			operators::min< double >, operators::add< double >,
			identities::infinity, identities::zero
		> minPlusFP64;
		Matrix< double > M = matrices< double >::identity( n, minPlusFP64 );
		if( nnz( M ) != expected_nnz ) {
			ERROR( rc, "matrices< double >::identity: nnz != n" );
		}
		if( nrows( M ) != n ) {
			ERROR( rc, "matrices< double >::identity:: nrows != n" );
		}
		if( ncols( M ) != n ) {
			ERROR( rc, "matrices< double >::identity:: ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.first.first != e.first.second ) {
				ERROR( rc, "matrices< double >::identity: incorrect coordinate" );
			}
			if( e.second != 0 ) {
				ERROR( rc, "matrices< double >::identity: incorrect value, got "
					+ std::to_string(e.second) + ", expected zero\n" );
			}
		}
	}
}

/**
 * This function tests the 'eye' factory function for matrices.
 *
 * The 'eye' factory function creates an identity matrix with a given offset.
 * This test function checks the correctness of the created matrix by verifying
 * its properties such as the number of non-zero elements, the number of rows
 * and columns, and the values of the elements.
 *
 * @param rc The return code of the test.
 * @param n The size of the matrix to be tested.
 * @param offset The offset for the identity matrix. Positive values shift the
 * identity diagonal to the right, while negative values shift it to the left.
 *
 * @return Returns a RC (Return Code) indicating the success or failure
 * of the test.
 * If the test is successful, it returns SUCCESS.
 * If the test fails, it returns FAILED.
 */
static void test_factory_eye( RC &rc, const size_t &n, const long &offset ) {
	const size_t i_offset = offset > 0 ? offset : 0;
	const size_t j_offset = offset < 0 ? (-offset) : 0;
	const size_t expected_nnz = i_offset + j_offset < n
		? n - i_offset - j_offset
		: 0;
	{ // matrices< void >::eye of size: [0,0]
		Matrix< void > M = matrices< void >::eye( 0, 0, offset );
		if( nnz( M ) != 0 ) {
			ERROR( rc, "matrices< void >::eye, size=(0,0): nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			ERROR( rc, "matrices< void >::eye, size=(0,0): nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			ERROR( rc, "matrices< void >::eye, size=(0,0): ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			ERROR( rc, "matrices< void >::eye, size=(0,0): found a value" );
		}
	}

	{ // matrices< int >::eye of size: [0,0]
		Matrix< int > M = matrices< int >::eye( 0, 0, 1, offset );
		if( nnz( M ) != 0 ) {
			ERROR( rc, "matrices< int >::eye, size=(0,0): nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			ERROR( rc, "matrices< int >::eye, size=(0,0): nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			ERROR( rc, "matrices< int >::eye, size=(0,0): ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			ERROR( rc, "matrices< int >::eye, size=(0,0): found a value" );
		}
	}

	{ // matrices< void >::eye of size: [n,n]
		Matrix< void > M = matrices< void >::eye( n, n, offset );
		if( nnz( M ) != expected_nnz ) {
			ERROR( rc, "matrices< void >::eye, size=(n,n): nnz != n" );
		}
		if( nrows( M ) != n ) {
			ERROR( rc, "matrices< void >::eye, size=(n,n): nrows != n" );
		}
		if( ncols( M ) != n ) {
			ERROR( rc, "matrices< void >::eye, size=(n,n): ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.first + i_offset != e.second + j_offset ) {
				ERROR( rc, "matrices< void >::eye, size=(n,n): incorrect coordinate" );
			}
		}
	}

	{ // matrices< int >::eye of size: [n,n]
		Matrix< int > M = matrices< int >::eye( n, n, 2, offset );
		if( nnz( M ) != expected_nnz ) {
			ERROR( rc, "matrices< int >::eye, size=(n,n): nnz != n-abs(k)" );
		}
		if( nrows( M ) != n ) {
			ERROR( rc, "matrices< int >::eye, size=(n,n): nrows != n" );
		}
		if( ncols( M ) != n ) {
			ERROR( rc, "matrices< int >::eye, size=(n,n): ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.first.first + i_offset != e.first.second + j_offset ) {
				ERROR( rc, "matrices< int >::eye, size=(n,n): incorrect coordinate" );
			}
			if( e.second != 2 ) {
				ERROR( rc, "matrices< int >::eye, size=(n,n): incorrect value" );
			}
		}
	}

	{ // matrices< int >::eye of size: [1,n]
		Matrix< int > M = matrices< int >::eye( 1, n, 2, offset );
		if( offset < 0 || i_offset >= n ) {
			if( nnz( M ) != 0 ) {
				ERROR( rc, "matrices< int >::eye, size(1,n): nnz != 0" );
			}
		} else if( nnz( M ) != 1 ) {
			ERROR( rc, "matrices< int >:eye, size=(1,n), offset=" + std::to_string(offset)
				+ ": nnz != 1 (it reads " + std::to_string(nnz( M )) + " instead)\n")
		}
		if( nrows( M ) != 1 ) {
			ERROR( rc, "matrices< int >::eye, size=(1,n): nrows != 1" );
		}
		if( ncols( M ) != n ) {
			ERROR( rc, "matrices< int >::eye, size=(1,n): ncols != n" );
		}
		for( const auto &e : M ) {
			if( offset < 0 ) {
				ERROR( rc, "matrices< int >::eye, size(1,n), offset=" + std::to_string(offset)
					+ ": incorrect coordinate\n" );
			} else {
				if( e.first.first != 0 ) {
					ERROR( rc, "matrices< int >::eye, size=(1,n): incorrect row index" );
				}
				if( e.first.second != i_offset ) {
					ERROR( rc, "matrices< int >::eye, size=(1,n): incorrect column "
						"coordinate" );
				}
				if( e.second != 2 ) {
					ERROR( rc, "matrices< int >::eye, size=(1,n): incorrect value" );
				}
			}
		}
	}

	{ // matrices< int >::eye of size: [n,1]
		Matrix< int > M = matrices< int >::eye( n, 1, 2, offset );
		if( offset > 0 || j_offset > n ) {
			if( nnz( M ) != 0 ) {
				ERROR( rc, "matrices< int >::eye, size=(n,1): nnz != 0" );
			}
		} else if( nnz( M ) != 1 ) {
			ERROR( rc, "matrices< int >::eye, size=(n,1): nnz != 1" );
		}
		if( nrows( M ) != n ) {
			ERROR( rc, "matrices< int >::eye, size=(n,1): nrows != n" );
		}
		if( ncols( M ) != 1 ) {
			ERROR( rc, "matrices< int >::eye, size=(n,1): ncols != 1" );
		}
		for( const auto &e : M ) {
			if( offset > 0 ) {
				ERROR( rc, "matrices< int >::eye, size=(n,1), offset=" + std::to_string(offset)
					+ ": incorrect coordinate\n" );
			} else {
				if( e.first.first != j_offset ) {
					ERROR( rc, "matrices< int >::eye, size=(n,1), offset=" + std::to_string(offset)
						+ ": incorrect row coordinate\n" );
				}
				if( e.first.second != 0 ) {
					ERROR( rc, "matrices< int >::eye, size=(n,1), offset=" + std::to_string(offset)
						+ ": incorrect column coordinate\n" );
				}
				if( e.second != 2 ) {
					ERROR( rc, "matrices< int >::eye, size=(n,1), offset=" + std::to_string(offset)
						+ ": incorrect value\n" );
				}
			}
		}
	}
}

template< typename VoidFactoryFunc, typename IntFactoryFunc >
void test_factory_full_templated(
	RC &rc,
	const size_t &n,
	const std::string &factoryName,
	const VoidFactoryFunc &voidFactory,
	const IntFactoryFunc &intFactory
) {
	{ // matrices::voidFactory of size: [0,0]
		Matrix< void > M = voidFactory( 0, 0 );
		if( nnz( M ) != 0 ) {
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"found a value" );
		}
	}

	{ // matrices::intFactory of size: [0,0]
		Matrix< int > M = intFactory( 0, 0, 2 );
		if( nnz( M ) != 0 ) {
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"found a value" );
		}
	}

	{ // matrices::voidFactory of size: [n,n]
		Matrix< void > M = voidFactory( n, n );
		if( nnz( M ) != n * n ) {
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(n,n): "
				"nnz != n*n" );
		}
		if( nrows( M ) != n ) {
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(n,n): "
				"nrows != n" );
		}
		if( ncols( M ) != n ) {
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(n,n): "
				"ncols != n" );
		}
	}

	{ // matrices::intFactory of size: [n,n]
		Matrix< int > M = intFactory( n, n, 2 );
		if( nnz( M ) != n * n ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,n): "
				"nnz != n*n" );
		}
		if( nrows( M ) != n ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,n): "
				"nrows != n" );
		}
		if( ncols( M ) != n ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,n): "
				"ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.second != 2 ) {
				ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,n): "
					"incorrect value" );
			}
		}
	}

	{ // matrices::intFactory of size: [1,n]
		Matrix< int > M = intFactory( 1, n, 2 );
		if( nnz( M ) != n ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(1,n): "
				"nnz != n" );
		}
		if( nrows( M ) != 1 ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(1,n): "
				"nrows != 1" );
		}
		if( ncols( M ) != n ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(1,n): "
				"ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.first.first != 0 ) {
				ERROR( rc, "matrices::" + factoryName + "<int>, size=(1,n): "
					"incorrect row coordinate" );
			}
			if( e.second != 2 ) {
				ERROR( rc, "matrices::" + factoryName + "<int>, size=(1,n): "
					"incorrect value" );
			}
		}
	}

	{ // matrices::intFactory of size: [n,1]
		Matrix< int > M = intFactory( n, 1, 2 );
		if( nnz( M ) != n ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,1): "
				"nnz != n" );
		}
		if( nrows( M ) != n ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,1): "
				"nrows != n" );
		}
		if( ncols( M ) != 1 ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,1): "
				"ncols != 1" );
		}
		for( const auto &e : M ) {
			if( e.first.second != 0 ) {
				ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,1): "
					"incorrect column coordinate" );
			}
			if( e.second != 2 ) {
				ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,1): "
					"incorrect value" );
			}
		}
	}
}

/**
 * This function tests the 'dense' factory function for matrices.
 *
 * The 'dense' factory function creates a dense matrix of a given size.
 * This test function checks the correctness of the created matrix by verifying
 * its properties such as the number of non-zero elements, the number
 * of rows and columns.
 *
 * @param rc The return code of the test.
 * @param n The size of the matrix to be tested.
 *
 * @return Returns a RC (Return Code) indicating the success or failure
 * of the test.
 * If the test is successful, it returns SUCCESS.
 * If the test fails, it returns FAILED.
 */
static void test_factory_dense( RC &rc, const size_t &n ) {
	test_factory_full_templated(
		rc, n, "dense",
		[&](const size_t r, const size_t c ) {
			return matrices< void >::dense( r, c );
		},
		[&](const size_t r, const size_t c, const int v ) {
			return matrices< int >::dense( r, c, v );
		} );
}

/**
 * This function tests the 'full' factory function for matrices.
 *
 * The 'full' factory function creates a full matrix of a given size.
 * This test function checks the correctness of the created matrix by verifying
 * its properties such as the number of non-zero elements, the number
 * of rows and columns.
 *
 * @param rc The return code of the test.
 * @param n The size of the matrix to be tested.
 *
 * @return Returns a RC (Return Code) indicating the success or failure
 * of the test.
 * If the test is successful, it returns SUCCESS.
 * If the test fails, it returns FAILED.
 */
static void test_factory_full( RC &rc, const size_t & n ) {
	return test_factory_full_templated(
		rc, n, "full",
		[&](const size_t r, const size_t c ) {
			return matrices< void >::full( r, c );
		},
		[&](const size_t r, const size_t c, const int v ) {
			return matrices< int >::full( r, c, v );
		} );
}

template< typename VoidFactoryFunc, typename IntFactoryFunc >
static void test_factory_dense_valued(
	RC &rc,
	const size_t &n,
	const std::string &factoryName,
	const VoidFactoryFunc &voidFactory,
	const IntFactoryFunc &intFactory,
	const int expectedValue
) {
	{ // matrices::voidFactory of size: [0,0]
		Matrix< void > M = voidFactory( 0, 0 );
		if( nnz( M ) != 0 ) {
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"found a value" );
		}
	}

	{ // matrices::intFactory of size: [0,0]
		Matrix< int > M = intFactory( 0, 0 );
		if( nnz( M ) != 0 ) {
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"nnz != 0" );
		}
		if( nrows( M ) != 0 ) {
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"nrows != 0" );
		}
		if( ncols( M ) != 0 ) {
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"ncols != 0" );
		}
		for( const auto &e : M ) {
			(void) e;
			ERROR( rc, "matrices::" + factoryName + "<void>, size=(0,0): "
				"found a value" );
		}
	}

	{ // matrices::voidFactory of size: [n,n]
		Matrix< void > M = voidFactory( n, n );
		if( nnz( M ) != n * n ) {
			ERROR( rc, "matrices< void >::" + factoryName + ", size=(n,n): "
				"nnz = " + std::to_string(nnz( M )) + " != n*n = " + std::to_string(n*n) );
		}
		if( nrows( M ) != n ) {
			ERROR( rc, "matrices< void >::" + factoryName + ", size=(n,n): "
				"nrows != n" );
		}
		if( ncols( M ) != n ) {
			ERROR( rc, "matrices< void >::" + factoryName + ", size=(n,n): "
				"ncols != n" );
		}
	}

	{ // matrices::intFactory of size: [n,n]
		Matrix< int > M = intFactory( n, n );
		if( nnz( M ) != n * n ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,n): "
				"nnz != n*n" );
		}
		if( nrows( M ) != n ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,n): "
				"nrows != n" );
		}
		if( ncols( M ) != n ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,n): "
				"ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.second != expectedValue ) {
				ERROR( rc, "matrices::eye<int>, size=(n,n): incorrect value" );
			}
		}
	}

	{ // matrices::intFactory of size: [1,n]
		Matrix< int > M = intFactory( 1, n );
		if( nnz( M ) != n ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(1,n): "
				"nnz != n" );
		}
		if( nrows( M ) != 1 ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(1,n): "
				"nrows != 1" );
		}
		if( ncols( M ) != n ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(1,n): "
				"ncols != n" );
		}
		for( const auto &e : M ) {
			if( e.first.first != 0 ) {
				ERROR( rc, "matrices::" + factoryName + "<int>, size=(1,n): "
					"incorrect row coordinate" );
			}
			if( e.second != expectedValue ) {
				ERROR( rc, "matrices::" + factoryName + "<int>, size=(1,n): "
					"incorrect value" );
			}
		}
	}

	{ // matrices::intFactory of size: [n,1]
		Matrix< int > M = intFactory( n, 1 );
		if( nnz( M ) != n ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,1): "
				"nnz != n" );
		}
		if( nrows( M ) != n ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,1): "
				"nrows != n" );
		}
		if( ncols( M ) != 1 ) {
			ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,1): "
				"ncols != 1" );
		}
		for( const auto &e : M ) {
			if( e.first.second != 0 ) {
				ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,1): "
					"incorrect column coordinate" );
			}
			if( e.second != expectedValue ) {
				ERROR( rc, "matrices::" + factoryName + "<int>, size=(n,1): "
					"incorrect value" );
			}
		}
	}
}

/**
 * This function tests the 'zeros' factory function for matrices.
 *
 * The 'zeros' factory function creates a matrix of a given size filled with zeros.
 * This test function checks the correctness of the created matrix by verifying
 * its properties such as the number of non-zero elements, the number
 * of rows and columns.
 *
 * @param rc The return code of the test.
 * @param n The size of the matrix to be tested.
 *
 * @return Returns a RC (Return Code) indicating the success or failure
 * of the test.
 * If the test is successful, it returns SUCCESS.
 * If the test fails, it returns FAILED.
 */
static void test_factory_zeros( RC &rc, const size_t &n ) {
	test_factory_dense_valued(
		rc, n, "zeros",
		[&](const size_t r, const size_t c ) {
			return matrices< void >::zeros( r, c );
		},
		[&](const size_t r, const size_t c ) {
			return matrices< int >::zeros( r, c );
		},
		0 );
}

/**
 * This function tests the 'ones' factory function for matrices.
 *
 * The 'ones' factory function creates a matrix of a given size filled with ones.
 * This test function checks the correctness of the created matrix by verifying
 * its properties such as the number of non-zero elements, the number
 * of rows and columns.
 *
 * @param rc The return code of the test.
 * @param n The size of the matrix to be tested.
 *
 * @return Returns a RC (Return Code) indicating the success or failure
 * of the test.
 * If the test is successful, it returns SUCCESS.
 * If the test fails, it returns FAILED.
 */
static void test_factory_ones( RC &rc, const size_t &n ) {
	test_factory_dense_valued(
		rc, n, "ones",
		[&](const size_t r, const size_t c ) {
			return matrices< void >::ones( r, c );
		},
		[&](const size_t r, const size_t c ) {
			return matrices< int >::ones( r, c );
		},
		1 );
}

void grb_program( const size_t &n, RC &rc ) {
	rc = SUCCESS;
	std::cout << "Testing matrices::empty\n";
	test_factory_empty( rc, n );
	std::cout << "Testing matrices::identity\n";
	test_factory_identity( rc, n );
	std::cout << "Testing matrices::eye\n";
	test_factory_eye( rc, n, 0 );
	std::cout << "Testing matrices::eye (1 offset)\n";
	test_factory_eye( rc, n, 1 );
	std::cout << "Testing matrices::eye (2 offset)\n";
	test_factory_eye( rc, n, 2 );
	std::cout << "Testing matrices::eye (n offset)\n";
	test_factory_eye( rc, n, static_cast<long>(n) );
	std::cout << "Testing matrices::eye (-1 offset)\n";
	test_factory_eye( rc, n, -1 );
	std::cout << "Testing matrices::eye (-2 offset)\n";
	test_factory_eye( rc, n, -2 );
	std::cout << "Testing matrices::eye (-2*n offset)\n";
	test_factory_eye( rc, n, -2*static_cast<long>(n) );
	std::cout << "Testing matrices::dense (direct)\n";
	test_factory_dense( rc, n );
	std::cout << "Testing matrices::full\n";
	test_factory_full( rc, n );
	std::cout << "Testing matrices::zeros\n";
	test_factory_zeros( rc, n );
	std::cout << "Testing matrices::ones\n";
	test_factory_ones( rc, n );

	// Check return code for distributed backends
	ASSERT_RC_SUCCESS(
		collectives<>::allreduce( rc, grb::operators::any_or< RC >() )
	);
}

int main( int argc, char ** argv ) {
	// defaults
	constexpr const size_t n_max = static_cast< size_t >(
		std::numeric_limits< long >::max() );
	size_t in = 100;

	// error checking
	if( argc > 2 ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is " << in << "): "
			<< "a positive integer smaller or equal to "
			<< n_max << ".\n";
		return 10;
	}
	if( argc >= 2 ) {
		in = std::strtoul( argv[ 1 ], nullptr, 0 );
		if( in > n_max ) {
			std::cerr << "Given value for n is too large\n";
			return 20;
		}
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

