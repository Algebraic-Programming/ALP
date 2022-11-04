
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

#include <alp.hpp>

using namespace alp;

typedef double T1;

const T1 testval1 = 1.5;
const T1 testval2 = -1;
const T1 testval3 = 2.;

void alp_program( const size_t &n, alp::RC &rc ) {

	// test 1
	// foldl( vector, scalar, mul_op)
	// foldr( scalar, vector, mul_op)
	{
		rc = SUCCESS;

		alp::Vector< T1 > x_l( n );

		alp::Semiring<
			alp::operators::add< double >, alp::operators::mul< double >,
			alp::identities::zero, alp::identities::one
		> ring;

		std::vector< T1 > x_data( n );

		// test 1 foldl , init
		std::fill( x_data.begin(), x_data.end(), testval1 );

		rc = rc ? rc : alp::buildVector( x_l, x_data.begin(), x_data.end() );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 1 (foldl( vector, scalar, mul_op )): initialisation FAILED\n";
			return;
		}

		// test 1 foldl, exec
		Scalar< T1 > out_r( testval2 );
		rc = alp::foldl( x_l, out_r, ring.getMultiplicativeOperator() );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 1 (foldl( vector, scalar, mul_op )): foldl FAILED\n";
			return;
		}

		// test 1 foldl, check
#ifdef DEBUG
		std::cout << "x_l = ";
#endif
		if( *out_r != testval2 ) {
			std::cerr << "\t test 1 ( foldl( vector, scalar, mul_op )): unexpected output ";
			std::cerr << "\t scalar should not me modified here\n";
			rc = FAILED;
			return;
		}
		for( size_t i = 0; i < alp::size( x_l ); ++i ) {
			if( x_l[ i ] !=  testval1 * testval2 ) {
				std::cerr << "\t test 1 ( foldl( vector, scalar, mul_op )): unexpected output "
					  << "vector [" <<  i << " ] ( " << x_l[ i ] << ", expected "
					  << ( static_cast< T1 >( testval1 * testval2 ) )
					  << " )\n";
				rc = FAILED;
				return;
			}
#ifdef DEBUG
			if( i < 10 ) {
				std::cout << x_l[ i ] << " ";
			} else if( i + 10 > alp::size( x_l ) ) {
				std::cout << x_l[ i ] << " ";
			} else if( i == 10 ) {
				std::cout << " ...  ";
			}
#endif
		}

		// test 1 foldr, exec
		alp::Vector< T1 > x_r( n );
		*out_r = testval2;

		// test 1 foldr, init
		std::fill( x_data.begin(), x_data.end(), testval1 );
		rc = rc ? rc : alp::buildVector( x_r, x_data.begin(), x_data.end() );

		// test 1 foldr, exec
		rc = alp::foldr( out_r, x_r, ring.getMultiplicativeOperator() );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 1 (foldr( scalar, vector, mul_op )): foldr FAILED\n";
			return;
		}
		// test 1 foldr, check
#ifdef DEBUG
		std::cout << "x_r = ";
#endif
		for( size_t i = 0; i < alp::size( x_r ); ++i ) {
			if( x_r[ i ] !=  testval1 * testval2 ) {
				std::cerr << "\t test 1 ( foldr( scalar, vector, mul_op )): unexpected output "
					  << "vector [" <<  i << " ] ( " << x_r[ i ] << ", expected "
					  << ( static_cast< T1 >( testval1 * testval2 ) )
					  << " )\n";
				rc = FAILED;
				return;
			}
#ifdef DEBUG
			if( i < 10 ) {
				std::cout << x_r[ i ] << " ";
			} else if( i + 10 > alp::size( x_r ) ) {
				std::cout << x_r[ i ] << " ";
			} else if( i == 10 ) {
				std::cout << " ...  ";
			}
#endif
		}

	}

	// test 2
	// foldl( scalar, vector, add_op)
	// foldr( vector, scalar, add_op)
	{
		alp::Vector< T1 > x_l( n );

		//test 2, init
		alp::Semiring<
			alp::operators::add< double >, alp::operators::mul< double >,
			alp::identities::zero, alp::identities::one
		> ring;

		rc = SUCCESS;
		{
			// temp initialization
			std::vector< T1 > x_data( n );
			std::fill( x_data.begin(), x_data.end(), static_cast< T1 >( testval2 ) );
			rc = rc ? rc : alp::buildVector( x_l, x_data.begin(), x_data.end() );
		}
		// rc = rc ? rc : alp::set( x_l, Scalar< T1 >( 0 ) ); // needs an implementation

		if( rc != SUCCESS ) {
			std::cerr << "\t test 2 (foldl( scalar, vector, add_op )) "
				  << "initialisation FAILED\n";
			return;
		}

		// test 2 foldl, exec
		Scalar< T1 > out( testval3 );
		rc = alp::foldl( out, x_l, ring.getAdditiveMonoid() );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 2 (foldl( scalar, vector, monoid )) foldl FAILED\n";
			return;
		}

		// test 2 foldl, check
		if( testval3 + testval2 * static_cast< T1 >( n ) != *out ) {
			std::cerr << "\t test 2 (foldl( scalar, vector, monoid)), "
				  << "unexpected output: " << *out << ", expected "
				  << testval3 + testval2 * static_cast< T1 >( n )
				  << ".\n";
			rc = FAILED;
			return;
		}

		// test 2 foldr, exec
		rc = alp::foldr( x_l, out, ring.getAdditiveMonoid() );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 2 (foldr( vector, scalar, monoid )) foldr FAILED\n";
			return;
		}

		// test 2 foldr, check
		if( testval3 + testval2 * static_cast< T1 >( n * 2 ) != *out ) {
			std::cerr << "\t test 2 (foldl( scalar, vector, monoid)), "
				  << "unexpected output: " << *out << ", expected "
				  << testval3 + testval2 * static_cast< T1 >( n * 2 )
				  << ".\n";
			rc = FAILED;
			return;
		}

#ifdef DEBUG
		std::cout << "x_l = ";
#endif
		for( size_t i = 0; i < alp::size( x_l ); ++i ) {
			if( x_l[ i ] !=  testval2 ) {
				std::cerr << "\t test 2 ( foldl/r): unexpected output, vector x_l should not be modified "
					  << "vector [" <<  i << " ] ( " << x_l[ i ] << ", expected "
					  << ( static_cast< T1 >( testval1 * testval2 ) )
					  << " )\n";
				rc = FAILED;
				return;
			}
#ifdef DEBUG
			if( i < 10 ) {
				std::cout << x_l[ i ] << " ";
			} else if( i + 10 > alp::size( x_l ) ) {
				std::cout << x_l[ i ] << " ";
			} else if( i == 10 ) {
				std::cout << " ...  ";
			}
#endif
		}

		// test 3
		// test 3 (foldl( scalar, vector_view, add_op))
		// test 3 (foldr( vector_view, scalar, add_op))
		auto x_view_even_l = alp::get_view( x_l, alp::utils::range( 0, n, 2 ) );
		*out = testval3;
		rc = alp::foldl( out, x_view_even_l, ring.getAdditiveMonoid() );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 3 (foldl( scalar, vector_view, monoid )) foldl FAILED\n";
			return;
		}

		// test 3 foldl, check
		if( testval3 + testval2 * static_cast< T1 >( n / 2 ) != *out ) {
			std::cerr << "\t test 3 (foldl( scalar, vector_view, monoid )), "
				  << "unexpected output: " << *out << ", expected "
				  << testval3 + testval2 * static_cast< T1 >( n / 2 )
				  << ".\n";
			rc = FAILED;
			return;
		}

		// test 3 (foldr( vector_view, scalar, add_op))
		auto x_view_even_r = alp::get_view( x_l, alp::utils::range( 0, n, 2 ) );
		*out = testval3;
		rc = alp::foldr( x_view_even_r, out, ring.getAdditiveMonoid() );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 3 (foldr( vector_view, scalar, monoid )) foldr FAILED\n";
			return;
		}

		// test 3 foldr, check
		if( testval3 + testval2 * static_cast< T1 >( n / 2 ) != *out ) {
			std::cerr << "\t test 3 (foldr( vector_view, scalar, monoid )), "
				  << "unexpected output: " << *out << ", expected "
				  << testval3 + testval2 * static_cast< T1 >( n / 2 )
				  << ".\n";
			rc = FAILED;
			return;
		}

#ifdef DEBUG
		std::cout << "x_l = ";
#endif
		for( size_t i = 0; i < alp::size( x_l ); ++i ) {
			if( x_l[ i ] !=  testval2 ) {
				std::cerr << "\t test 3 ( foldl/r): unexpected output, vector x_l should not be modified "
					  << "vector [" <<  i << " ] ( " << x_l[ i ] << ", expected "
					  << ( static_cast< T1 >( testval1 * testval2 ) )
					  << " )\n";
				rc = FAILED;
				return;
			}
#ifdef DEBUG
			if( i < 10 ) {
				std::cout << x_l[ i ] << " ";
			} else if( i + 10 > alp::size( x_l ) ) {
				std::cout << x_l[ i ] << " ";
			} else if( i == 10 ) {
				std::cout << " ...  ";
			}
#endif
		}
	}

	// test 4
	// test 4 (foldl( vector, vector, add_op))
	// test 4 (foldr( vector, vector, add_op))
	{
		alp::Semiring<
			alp::operators::add< double >, alp::operators::mul< double >,
			alp::identities::zero, alp::identities::one
		> ring;

		alp::Vector< T1 > x( n );
		alp::Vector< T1 > y( n );

		std::vector< T1 > data( n );

		std::fill( data.begin(), data.end(), static_cast< T1 >( testval2 ) );
		rc = rc ? rc : alp::buildVector( x, data.begin(), data.end() );

		std::fill( data.begin(), data.end(), static_cast< T1 >( testval3 ) );
		rc = rc ? rc : alp::buildVector( y, data.begin(), data.end() );

		if( rc != SUCCESS ) {
			std::cerr << "\t test 4 alp::buildVector FAILED\n";
			return;
		}

		// test 4 (foldl( vector, vector, add_op))
		rc = alp::foldl( x, y, ring.getAdditiveOperator() );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 4 (foldl( vector, vector, monoid )) foldl FAILED\n";
			return;
		}

		for( size_t i = 0; i < alp::size( x ); ++i ) {
			if( x[ i ] !=  testval2 +  testval3 ) {
				std::cerr << "\t test 4 ( foldl): unexpected output "
					  << "vector [" <<  i << " ] ( " << x[ i ] << ", expected "
					  << ( static_cast< T1 >( testval2 + testval3 ) )
					  << " )\n";
				rc = FAILED;
				return;
			}

			if( y[ i ] !=  testval3 ) {
				std::cerr << "\t test 4 ( foldl): unexpected output, vector y should not be modified "
					  << "vector [" <<  i << " ] ( " << y[ i ] << ", expected "
					  << ( static_cast< T1 >( testval3 ) )
					  << " )\n";
				rc = FAILED;
				return;
			}
		}

		// test 4 (foldr( vector, vector, add_op))
		rc = alp::foldr( x, y, ring.getAdditiveOperator() );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 4 (foldr( vector, vector, monoid )) foldr FAILED\n";
			return;
		}

		for( size_t i = 0; i < alp::size( x ); ++i ) {
			if( x[ i ] !=  testval2 +  testval3 ) {
				std::cerr << "\t test 4 ( foldr): unexpected output, vector x should not be modified "
					  << "vector [" <<  i << " ] ( " << x[ i ] << ", expected "
					  << ( static_cast< T1 >( testval2 + testval3 ) )
					  << " )\n";
				rc = FAILED;
				return;
			}

			if( y[ i ] !=  testval2 +  2 * testval3 ) {
				std::cerr << "\t test 4 ( foldr): unexpected output, "
					  << "vector [" <<  i << " ] ( " << y[ i ] << ", expected "
					  << ( static_cast< T1 >( testval2 +  2 * testval3 ) )
					  << " )\n";
				rc = FAILED;
				return;
			}
		}

	}

	// test 5
	// test 5 (foldl( matrix, scalar, add_monoid))
	// test 5 (foldl( matrix, matrix, add_monoid))
	{
		alp::Matrix< T1, alp::structures::General > A( n, n );
		alp::Matrix< T1, alp::structures::General > B( n, n );
		alp::Scalar< T1 > alpha( testval1 );

		//test 5, init
		alp::Semiring<
			alp::operators::add< double >, alp::operators::mul< double >,
			alp::identities::zero, alp::identities::one
		> ring;

		rc = SUCCESS;
		rc = rc ? rc : alp::set( A, alp::Scalar< T1 >( testval2 ) );
		rc = rc ? rc : alp::set( B, alp::Scalar< T1 >( testval3 ) );

		if( rc != SUCCESS ) {
			std::cerr << "\t test 5 (foldl( matrix, scalar, add_op )) "
				  << "initialisation FAILED\n";
			return;
		}

		// test 5 (foldl( matrix, scalar, add_monoid)) exec
		rc = alp::foldl( A, alpha, ring.getAdditiveMonoid() );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 5 (foldl( matrix, scalar, monoid )) foldl FAILED\n";
			return;
		}

		// test 5 (foldl( matrix, matrix, add_monoid)) exec
		rc = alp::foldl( A, B, ring.getAdditiveMonoid() );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 5 (foldl( matrix, matrix, monoid )) foldl FAILED\n";
			return;
		}

		// test 2, check
		const auto A_val = alp::internal::access( A, alp::internal::getStorageIndex( A, 0, 0 ) );
		if( testval1 + testval2 + testval3 != A_val ) {
			std::cerr << "\t test 5 (foldl( matrix, scalar, monoid ), foldl( matrix, matrix, monoid )), "
				  << "unexpected output: " << A_val << ", expected "
				  << testval1 + testval2 + testval3
				  << ".\n";
			rc = FAILED;
			return;
		}
	}

	// test 6
	// test 6 (foldr( scalar, matrix, add_monoid ))
	// test 6 (foldr( matrix, matrix, add_monoid ))
	{
		alp::Matrix< T1, alp::structures::General > A( n, n );
		alp::Matrix< T1, alp::structures::General > B( n, n );
		alp::Scalar< T1 > alpha( testval1 );

		//test 6, init
		alp::Semiring<
			alp::operators::add< double >, alp::operators::mul< double >,
			alp::identities::zero, alp::identities::one
		> ring;

		rc = SUCCESS;
		rc = rc ? rc : alp::set( A, alp::Scalar< T1 >( testval2 ) );
		rc = rc ? rc : alp::set( B, alp::Scalar< T1 >( testval3 ) );

		if( rc != SUCCESS ) {
			std::cerr << "\t test 6 (foldr( scalar, matrix, monoid ), foldr( matrix, matrix, monoid )) "
				  << "initialisation FAILED\n";
			return;
		}

		// test 6 (foldr( scalar, matrix, add_monoid )) exec
		rc = alp::foldr( alpha, B, ring.getAdditiveMonoid() );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 6 (foldr( matrix, scalar, monoid )) foldl FAILED\n";
			return;
		}

		// test 6 (foldr( matrix, matrix, add_monoid )) exec
		rc = alp::foldr( A, B, ring.getAdditiveMonoid() );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 6 (foldr( matrix, matrix, monoid )) foldl FAILED\n";
			return;
		}

		// test 2, check
		const auto B_val = alp::internal::access( B, alp::internal::getStorageIndex( B, 0, 0 ) );
		if( testval1 + testval2 + testval3 != B_val ) {
			std::cerr << "\t test 6 (foldr( scalar, matrix, monoid ), foldr( matrix, matrix, monoid )), "
				  << "unexpected output: " << B_val << ", expected "
				  << testval1 + testval2 + testval3
				  << ".\n";
			rc = FAILED;
			return;
		}
	}
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
	alp::Launcher< AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alp_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cerr << "Test FAILED (" << alp::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}
