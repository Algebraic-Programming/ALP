
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

/*
 * A practical graph corresponding to a 5 by 5 matrix with 10 nonzeroes.
 *
 * -# Flight prices correspond to the cheapest round trip price with departure
 *    on 1/10/2016 and return on 8/10/2016 when booked on 10/8/2016 according to
 *    Google Flights.
 * -# Distances are as determined by Google Maps.
 * -# All edges are directed.
 *
 * @author: A. N. Yzelman
 * @date: 11th August, 2016.
 */

#include <climits>
#include <cstdio>
#include <string>
#include <vector>

#include <graphblas.hpp>

using namespace grb;

//! [Example Data]
static const char * const vertex_ids[ 5 ] = { "Shenzhen", "Hong Kong", "Santa Clara", "London", "Paris" };

static const double distances[ 10 ] = { 8.628, 8.964, 11.148, .334, 9.606, 9.610, .017, .334, .017, .334 };
static const int price[ 10 ] = { 723, 956, 600, 85, 468, 457, 333, 85, 50, 150 };
static const double timeliness[ 10 ] = { 0.9, 0.7, 0.99, 0.9, 0.9, 0.7, 0.99, 0.7, .99, 0.99 };
static const std::string mode[ 10 ] = { "air", "air", "air", "air", "air", "air", "air", "air", "land", "land" };

static const size_t I[ 10 ] = { 3, 4, 2, 3, 3, 4, 1, 4, 1, 4 };
static const size_t J[ 10 ] = { 2, 2, 1, 4, 1, 1, 0, 3, 0, 3 };
//! [Example Data]

//! [Example function taking arbitrary semirings]
template< typename Ring >
grb::Vector< typename Ring::D4 >
shortest_path( const grb::Matrix< typename Ring::D2 > & A, const grb::Vector< typename Ring::D1 > & initial_state, const size_t hops = 1, const Ring & ring = Ring() ) {
	const size_t size = grb::size( initial_state );
	grb::Vector< typename Ring::D4 > ret( size );
	grb::Vector< typename Ring::D4 > new_state( size );
	grb::set( ret, initial_state );
	vxm( ret, initial_state, A, ring );
	for( size_t i = 1; i < hops; ++i ) {
		grb::set( new_state, ret );
		vxm( ret, new_state, A, ring );
	}
	return ret;
}
//! [Example function taking arbitrary semirings]

int main( int argc, char ** argv ) {
	(void)argc;
	(void)printf( "Illustration executable: %s\n\n", argv[ 0 ] );

	(void)printf( "This is not a functional or performance test, but rather an illustration of some of the GraphBLAS usefulness.\n\n" );

	(void)printf( "Create distance graph as a 5 x 5 matrix with 10 nonzeroes:\n"
				  "-->grb::Matrix< double > dist( 5, 5 );\n" );
	//! [Example matrix allocation]
	grb::Matrix< double > dist( 5, 5 );
	resize( dist, 10 );
	//! [Example matrix allocation]

	(void)printf( "Load distance graph:\n"
				  "-->dist.buildMatrixUnique( dist, &(I[0]), &(J[0]), distances, 10 "
				  ");\n" );
	//! [Example matrix assignment]
	buildMatrixUnique( dist, &( I[ 0 ] ), &( J[ 0 ] ), distances, 10, SEQUENTIAL );
	//! [Example matrix assignment]

	(void)printf( "Create new vectors x and y:\n"
				  "-->grb::Vector< int > x( 5 );\n"
				  "-->grb::Vector< int > y( 5 );\n" );
	//! [Example vector allocation]
	grb::Vector< double > x( 5 );
	grb::Vector< double > y( 5 );
	//! [Example vector allocation]

	(void)printf( "The five vertices stand for the following cities:\n" );
	for( size_t i = 0; i < 5; ++i ) {
		(void)printf( "--> city %zd: %s\n", i, vertex_ids[ i ] );
	}
	(void)printf( "Let us calculate which cities are reachable from %s by taking one "
				  "air or land route:\n-->"
				  "x.set( INFINITY );\n-->"
				  "x.setElement( 0, 4 );\n-->"
				  "y.set( x );\n-->"
				  "typedef grb::Semiring< grb::operators::min< double >, "
				  "grb::operators::add< double >, grb::identities::infinity, "
				  "grb::identitites::zero > shortest_path_double;\n-->"
				  "vxm( y, x, dist, shortest_path_double );\n",
		vertex_ids[ 4 ] );
	//! [Example vector assignment]
	grb::set( x, INFINITY );
	grb::setElement( x, 0.0, 4 );
	grb::set( y, x );
	//! [Example vector assignment]
	//! [Example semiring definition]
	grb::Semiring< grb::operators::min< double >, grb::operators::add< double >, grb::identities::infinity, grb::identities::zero > shortest_path_double;
	//! [Example semiring definition]
	//! [Example semiring use: sparse vector times matrix multiplication]
	grb::vxm( y, x, dist, shortest_path_double );
	//! [Example semiring use: sparse vector times matrix multiplication]
	(void)printf( "We can reach the following cities within one trip:\n" );
	for( const std::pair< size_t, double > & pair : y ) {
		const double val = pair.second;
		if( val < INFINITY ) {
			(void)printf( "--> %s at distance %lf thousand kilometres.\n", vertex_ids[ pair.first ], val );
		}
	}

	(void)printf( "Let us calculate which cities we can reach after one more trip. "
				  "To do this, we first copy y into x, thus effectively computing "
				  "y=A(Ax).\n"
				  "-->grb( x, y );\n"
				  "-->grb::vxm( y, x, dist, shortest_path_double );\n" );
	grb::set( x, y );
	grb::operators::add< double, double, double > add_operator;
	grb::vxm( y, x, dist, shortest_path_double );
	(void)printf( "We can reach the following cities within two trips:\n" );
	for( const std::pair< size_t, double > & pair : y ) {
		const double val = pair.second;
		if( val < INFINITY ) {
			(void)printf( "--> %s at distance %lf\n", vertex_ids[ pair.first ], val );
		}
	}

	(void)printf( "We put the above in a templated function so we can call the same "
				  "shortest-paths calculation on different input and using different "
				  "semirings:\n"
				  "-->template< typename ring >\n"
				  "-->grb::Vector< typename ring::D4 > shortest_path( const "
				  "grb::Matrix< typename ring::D2 > &A, const grb::Vector< typename "
				  "ring::D1 > &initial_state, const size_t n, const size_t hops = 1 "
				  ") {\n"
				  "-->	grb::Vector< typename ring::D4 > ret( n );\n"
				  "-->    grb::set( ret, initial_state );\n"
				  "-->	grb::vxm( ret, initial_state, A, ring );\n"
				  "-->	for( size_t i = 1; i < hops; ++i ) {\n"
				  "-->		grb::Vector< typename ring::D4 > new_state( n );\n"
				  "-->            grb::set( new_state, ret );\n"
				  "-->		grb::vxm( ret, new_state, A, ring );\n"
				  "-->	}\n"
				  "-->	return ret;\n"
				  "-->}\n" );

	(void)printf( "Now let us calculate the price of flying instead of the distance. "
				  "The price is in Euros so now we use integers instead of doubles, "
				  "resulting in different domains the semiring which otherwise "
				  "remains identical:\n"
				  "-->typedef grb::Semiring< grb::operators::min< int >, "
				  "grb::operators::add< int >, grb::identities::infinity, "
				  "grb::identities::zero > shortest_path_ints;\n" );

	typedef grb::Semiring< grb::operators::min< int >, grb::operators::add< int >, grb::identities::infinity, grb::identities::zero > shortest_path_ints;
	(void)printf( "We continue in one go:"
				  "-->grb::Matrix< int > prices( 5, 5 );\n"
				  "-->grb::Vector< int > initial_trip_price( 5 );\n"
				  "-->buildMatrixUnique( prices, &(I[0]), &(J[0]), air, 10 );\n"
				  "-->before_trip_price.set( 9999 ); //all prices initially unknown. Integers have no infinite, however, so just pick a big number that doesn't overflow)\n"
				  "-->before_trip_price.setElement( 0, 4 );   //except that of our start position, which is free\n"
				  "-->grb::Vector< int > trip_prices = shortest_path< shortest_path_ints >( prices, initial_trip_price, 2 );\n" );
	grb::Matrix< int > prices( 5, 5 );
	resize( prices, 10 );
	grb::Vector< int > initial_trip_price( 5 );
	buildMatrixUnique( prices, &( I[ 0 ] ), &( J[ 0 ] ), price, 10, SEQUENTIAL );
	grb::set( initial_trip_price, 9999 );        // all prices initially unknown. Integers have no infinite, however, so just pick a big number (that doesn't overflow)
	grb::setElement( initial_trip_price, 0, 4 ); // except that of our start position, which is free
	                                             //! [Example function call while passing a semiring]
	grb::Vector< int > trip_prices = shortest_path< shortest_path_ints >( prices, initial_trip_price, 2 );
	//! [Example function call while passing a semiring]

	(void)printf( "We can go from Paris to the following cities, within two separate trips:\n" );
	for( const std::pair< size_t, int > & pair : trip_prices ) {
		const size_t i = pair.first;
		const int val = pair.second;
		if( val < INFINITY ) {
			(void)printf( "--> %s at cost %d\n", vertex_ids[ i ], val );
		}
	}

	(void)printf( "We might also be interested in the probability we will arrive on "
				  "time. Instead of distances or prices, we now assign probabilities "
				  "to the edges; e.g., flights from Santa Clara to Hong Kong have a "
				  "`timeliness' of 0.99, meaning that with 99 percent certainty, the "
				  "flight will be on time.\n" );
	(void)printf( "For the sake of example, we count flights going out from Paris as "
				  "having only a 70 percent probability of being on time due to "
				  "strikes, while flights going out of Heathrow London are slightly "
				  "more often late, at 90 percent. Trains between London and Paris "
				  "run at .99 timeliness.\n" );
	(void)printf( "We can now compute the best combination of trip legs in terms of "
				  "timeliness when using the following semiring:\n" );
	(void)printf( "-->typedef grb::Semiring< grb::operators::mul< double >, "
				  "grb::operators::max< double >, grb::identities::one, "
				  "grb::identities::negative_infinity > mul_max_double;\n" );
	typedef grb::Semiring< grb::operators::max< double >, grb::operators::mul< double >, grb::identities::negative_infinity, grb::identities::one > mul_max_double;
	(void)printf( "Let us use this semi-ring:\n"
				  "-->grb::Matrix< double > T( 5, 5 );\n"
				  "-->buildMatrixUnique( T, &(I[0]), &(J[0]), timeliness, 10 );\n"
				  "-->grb::Vector< double > initial_timeliness( 5 );\n"
				  "-->initial_timeliness.set( 0.0 );\n"
				  "-->initial_timeliness.setElement( 1.0, 4 );\n"
				  "-->grb::Vector< int > trip_timeliness = shortest_path< "
				  "mul_max_double >( T, initial_timeliness, 2 );\n" );
	//! [Example shortest-paths with semiring adapted to find the most reliable route instead]
	grb::Matrix< double > T( 5, 5 );
	resize( T, 10 );
	buildMatrixUnique( T, &( I[ 0 ] ), &( J[ 0 ] ), timeliness, 10, SEQUENTIAL );
	grb::Vector< double > initial_timeliness( 5 );
	grb::set( initial_timeliness, 0.0 );
	grb::setElement( initial_timeliness, 1.0, 4 );
	const grb::Vector< double > trip_timeliness2 = shortest_path< mul_max_double >( T, initial_timeliness, 2 );

	(void)printf( "If we take a maximum of two separate trips, we can go from Paris "
				  "to the following cities timeliness as follows:\n" );
	for( const std::pair< size_t, double > & pair : trip_timeliness2 ) {
		const size_t i = pair.first;
		const double val = pair.second;
		if( val > 0 ) {
			(void)printf( "--> %s with %lf percent probability of arriving on time\n", vertex_ids[ i ], val * 100.0 );
		}
	}
	//! [Example shortest-paths with semiring adapted to find the most reliable route instead]

	(void)printf( "If we allow a maximum of three separate trips, however, the "
				  "probability of us arriving in Shenzhen increases "
				  "dramatically:\n" );
	const grb::Vector< double > trip_timeliness3 = shortest_path< mul_max_double >( T, initial_timeliness, 3 );
	for( const std::pair< size_t, double > & pair : trip_timeliness3 ) {
		const size_t i = pair.first;
		const double val = pair.second;
		if( val > 0 ) {
			(void)printf( "--> %s with %lf percent probability of arriving on time\n", vertex_ids[ i ], val * 100.0 );
		}
	}

	return EXIT_SUCCESS;
}
