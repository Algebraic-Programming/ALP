
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
//#include "../include/graphblas/base/blas3.hpp"

using namespace grb;

//! [Example Data]

static const int adj[ 50 ] = { 1, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

static const size_t I[ 50 ] = { 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 1, 3, 5, 7, 9, 1, 3, 5, 7, 9, 1, 3, 5, 7, 9, 1, 3, 5, 7, 9, 1, 3, 5, 7, 9};
static const size_t J[ 50 ] = { 1, 3, 5, 7, 9, 1, 3, 5, 7, 9, 1, 3, 5, 7, 9, 1, 3, 5, 7, 9, 1, 3, 5, 7, 9, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8};

static const int M_adj[ 6 ] = { 1, 1, 1, 1, 1, 1};

static const size_t M_I[ 6 ] = { 0, 2, 1, 3, 6, 7};
static const size_t M_J[ 6 ] = { 1, 3, 0, 2, 7, 6};

static const int Al_adj[ 6 ] = { 2, 7, 10, 2, 7, 10};

static const size_t Al_I[ 6 ] = { 5, 4, 1, 0, 3, 2 };
static const size_t Al_J[ 6 ] = { 0, 3, 2, 5, 4, 1 };


/*

Function that flips values in a matching matrix M, according to a matrix Alternating, which specifies edges that need to be added
A describes a graph, while z is a full-rank vector filled with 1's

*/
void flip(grb::Matrix< int > &M, const grb::Matrix< int > &Alternating, const grb::Matrix< int > &A, const grb::Vector< int > &z) {
	const size_t n = grb::size( z );
	
	grb::Semiring< grb::operators::add< int >, grb::operators::mul< int >, grb::identities::zero, grb::identities::one > standard;


	/*
	Vector m will contain all matched vertices
	*/
	grb::Vector< int > m( n ); 
	grb::mxv( m, M, z, standard ); 


	/*
	Vector a will contain vertices from an alternating path
	*/
	grb::Vector< int > a( n );
	grb::mxv( a, Alternating, z, standard );


	/*
	Vector r will contain vertices both in an alternating path and in a matching
	*/
	grb::Vector< int > r( n );
	grb::eWiseMul( r, m, a, standard );

	/*(void)printf("Nonzero count in r is: %lu\n\n",nnz(r));

	for(auto it: r) {
		(void)printf("%lu = %d\n",it.first, it.second);
	}*/

	grb::Matrix< int > R( n, n );
	grb::Matrix< int > ToSubtract( n, n );
	grb::Matrix< int > TempR( n, n );

	/*
	R contains nonzeros on edges that need to be removed
	*/

	Monoid<
		grb::operators::add< int >,
		grb::identities::zero
	> addition_monoid;

	grb::maskedOuter( TempR, M, r, z, grb::operators::mul< int >(),Phase::RESIZE );
	grb::maskedOuter( TempR, M, r, z, grb::operators::mul< int >() ); 

	grb::eWiseApply< descriptors::transpose_right >( R, TempR, TempR, addition_monoid, Phase::RESIZE );
	grb::eWiseApply< descriptors::transpose_right >( R, TempR, TempR, addition_monoid );

	/*(void)printf("Nonzero count in R is: %lu\n\n",nnz(R));

	for(auto it: R) {
		(void)printf("(%lu,%lu) = %d\n",it.first.first,it.first.second, it.second);
	}*/
	//return;
	
	
	/*
	ToSubtract contains values of M on edges that need to be removed

	We take nonzeroes from both M and R and create TempR

	We extend TempR by zeroes from M.

	*/

	Monoid<
			grb::operators::right_assign< int >,
			grb::identities::zero
		> right_assignment_monoid;

	grb::eWiseApply( TempR, R, M, grb::operators::right_assign< int >(), Phase::RESIZE );
	grb::eWiseApply( TempR, R, M, grb::operators::right_assign< int >() ) ;

	grb::eWiseApply( ToSubtract, M, TempR, right_assignment_monoid, Phase::RESIZE );
	grb::eWiseApply( ToSubtract, M, TempR, right_assignment_monoid ) ;

	/*(void)printf("Nonzero count in ToSubtract is: %lu\n\n",nnz(ToSubtract));

	for(auto it: ToSubtract) {
		(void)printf("(%lu,%lu) = %d\n",it.first.first,it.first.second, it.second);
	}*/

	/*
	TempM is M with unnecessary edges removed
	*/

	grb::Matrix< int > TempM( n, n );
	grb::eWiseApply( TempM, M, ToSubtract, grb::operators::subtract< int >(), Phase::RESIZE );
	grb::eWiseApply( TempM, M, ToSubtract, grb::operators::subtract< int >() );
	

	

	/*
	AlternatingA contains the weights of edges that need to be added
	*/

	grb::Matrix< int > AlternatingA( n, n );

	grb::eWiseApply( AlternatingA, Alternating, A, grb::operators::right_assign< int >());

	grb::Matrix< int > PrefinalM( n, n );

	/*
	PrefinalM contains all the correct values, however it isn't filtered to contain only nonzeroes
	*/
	grb::eWiseApply( PrefinalM, TempM, AlternatingA, addition_monoid, Phase::RESIZE );
	grb::eWiseApply( PrefinalM, TempM, AlternatingA, addition_monoid );


	grb::select( M, PrefinalM, grb::operators::is_nonzero< int, int, int >(), Phase::RESIZE );
	grb::select( M, PrefinalM, grb::operators::is_nonzero< int, int, int >() );

}



/*
A function returning masking the matrix B to only the maximal value in each row.
In a case of ties, the highest column is picked
*/
void maxPerRow(grb::Matrix< int > &RowMax, const grb::Matrix< int > &B, const grb::Vector< int > &z) {
	const size_t n = grb::size( z );

	if( nnz(B)==0 ) {
		grb::set(RowMax, B);
		return;
	}

	grb::Vector< int > b( n );
	grb::set( b, 0 );
	grb::internal::foldl_unmasked( b, B, grb::operators::max< int >() );


	grb::Matrix< int > C( n, n );
	grb::Matrix< int > C_tmp( n, n );
	grb::Matrix< int > tmp( n, n );
	grb::maskedOuter( tmp, B, b, z, grb::operators::left_assign< int >(), Phase::RESIZE );
	grb::maskedOuter( tmp, B, b, z, grb::operators::left_assign< int >() );
	grb::eWiseApply( C_tmp, B, tmp, grb::operators::equal< int, int >(), Phase::RESIZE );
	grb::eWiseApply( C_tmp, B, tmp, grb::operators::equal< int, int >() );

	grb::select( C, C_tmp, grb::operators::is_nonzero< int, int, int >(), Phase::RESIZE );
	grb::select( C, C_tmp, grb::operators::is_nonzero< int, int, int >() );

	grb::set( C_tmp, C, Phase::RESIZE );
	grb::set( C_tmp, C );

	/*(void)printf("Nonzero count in K is: %lu\n\n",nnz(K));

	for(auto it: K) {
		(void)printf("(%lu,%lu) = %d\n",it.first.first,it.first.second, it.second);
	}*/

	grb::eWiseLambda( [&C_tmp]( const size_t i, const size_t j, int& nz ) { nz = j; }, C_tmp );

	grb::Vector< int > c( n );
	grb::set( c, -1 );
	grb::internal::foldl_unmasked( c, C_tmp, grb::operators::max< int >() );

	grb::Matrix< int > K( n, n );
	grb::set( K, C, Phase::RESIZE );
	grb::set( K, C );

	/*(void)printf("Nonzero count in K is: %lu\n\n",nnz(K));

	for(auto it: K) {
		(void)printf("(%lu,%lu) = %d\n",it.first.first,it.first.second, it.second);
	}*/

	grb::eWiseLambda( [&K]( const size_t i, const size_t j, int& nz ) { nz = j; }, K );

	/*(void)printf("Nonzero count in K is: %lu\n\n",nnz(K));

	for(auto it: K) {
		(void)printf("(%lu,%lu) = %d\n",it.first.first,it.first.second, it.second);
	}*/


	grb::Matrix< int > B_mask( n, n );
	grb::maskedOuter( tmp, C, c, z, grb::operators::left_assign< int >(), Phase::RESIZE );
	grb::maskedOuter( tmp, C, c, z, grb::operators::left_assign< int >() );
	grb::eWiseApply( B_mask, K, tmp, grb::operators::equal< int, int >(), Phase::RESIZE );
	grb::eWiseApply( B_mask, K, tmp, grb::operators::equal< int, int >() );


	grb::select( tmp, B_mask, grb::operators::is_nonzero< int, int, int >(), Phase::RESIZE );
	grb::select( tmp, B_mask, grb::operators::is_nonzero< int, int, int >() );


	grb::eWiseApply( RowMax, B, tmp, grb::operators::left_assign< int >(), Phase::RESIZE );
	grb::eWiseApply( RowMax, B, tmp, grb::operators::left_assign< int >() );
}

void searchOneAugmentations ( grb::Matrix< int > &G1, grb::Matrix< int > &D1, const grb::Matrix< int > &M, const grb::Matrix< int > &A, const grb::Vector< int > &z ) {
	
	const size_t n = grb::size( z );

	grb::Matrix< int > Temp( n, n );

	grb::Matrix< int > Unmatched( n, n );

	Monoid<
			grb::operators::right_assign< int >,
			grb::identities::zero
		> right_assignment_monoid;

	grb::eWiseApply( Temp, A, M, right_assignment_monoid, Phase::RESIZE );
	grb::eWiseApply( Temp, A, M, right_assignment_monoid, Phase::EXECUTE );

	grb::eWiseApply( Unmatched, A, Temp, grb::operators::subtract< int >(), Phase::RESIZE );
	grb::eWiseApply( Unmatched, A, Temp, grb::operators::subtract< int >(), Phase::EXECUTE );
	
	grb::Semiring< grb::operators::add< int >, grb::operators::mul< int >, grb::identities::zero, grb::identities::one > standard;

	grb::Vector< int > m( n );
	grb::set( m, 0 );
	grb::mxv( m, M, z, standard , Phase::RESIZE );
	grb::mxv( m, M, z, standard , Phase::EXECUTE );
	


	grb::maskedOuter( Temp, Unmatched, m, m, grb::operators::add< int >(),Phase::RESIZE );
	grb::maskedOuter( Temp, Unmatched, m, m, grb::operators::add< int >() );
	


	grb::Matrix< int > G1_temp( n, n );

	grb::eWiseApply( G1_temp, Unmatched, Temp, grb::operators::subtract< int >(), Phase::RESIZE );
	grb::eWiseApply( G1_temp, Unmatched, Temp, grb::operators::subtract< int >(), Phase::EXECUTE );

	grb::select( G1, G1_temp, grb::operators::is_positive<int,int,int>(), Phase::RESIZE );
	grb::select( G1, G1_temp, grb::operators::is_positive<int,int,int>() );

	if( nnz( G1 ) ) {
		maxPerRow( D1, G1, z );
		//maxPerRow( D1, G1, z );
	}
	else {
		grb::clear( D1 );
		grb::clear( G1 );
	}
}

void findOneAugmentations( grb::Matrix< int > &Augmentation, const grb::Matrix< int > &M, const grb::Matrix< int > &A, const grb::Vector< int > &z ) {

	const size_t n = grb::size( z );
	
	grb::Matrix< int > G1( n, n );
	grb::Matrix< int > D1( n, n );

	searchOneAugmentations( G1, D1, M, A, z );

	if( nnz( G1 ) ) {
		grb::eWiseApply<descriptors::transpose_right>( Augmentation, D1, D1, grb::operators::mul< int >(), Phase::RESIZE);
		grb::eWiseApply<descriptors::transpose_right>( Augmentation, D1, D1, grb::operators::mul< int >(), Phase::EXECUTE);
	}
	else {
		grb::clear( Augmentation );
	}

}

void selectSubmatrix( grb::Matrix< int > &B, const grb::Matrix< int > &A, const grb::Vector< int > &rows, const grb::Vector< int > &cols ) {
	const size_t n = grb::size( rows );
	grb::Matrix< int > Tmp( n, n );

	grb::maskedOuter( Tmp, A, rows, cols, grb::operators::add< int >(),Phase::RESIZE );
	grb::maskedOuter( Tmp, A, rows, cols, grb::operators::add< int >() );

	grb::eWiseApply( B, Tmp, A, grb::operators::right_assign< int >(), Phase::RESIZE );
	grb::eWiseApply( B, Tmp, A, grb::operators::right_assign< int >() );
}

/*void selectRows( grb::Matrix< int > &B, const grb::Matrix< int > &A, const grb::Vector< int > &m, const grb::Vector< int > &z ) {
	const size_t n = grb::size( z );
	grb::Matrix< int > Tmp( n, n );

	grb::maskedOuter( Tmp, A, m, z, grb::operators::add< int >(),Phase::RESIZE );
	grb::maskedOuter( Tmp, A, m, z, grb::operators::add< int >() );

	grb::eWiseApply( B, Tmp, A, grb::operators::right_assign< int >(), Phase::RESIZE );
	grb::eWiseApply( B, Tmp, A, grb::operators::right_assign< int >() );
}

void selectColumns( grb::Matrix< int > &B, const grb::Matrix< int > &A, const grb::Vector< int > &m, const grb::Vector< int > &z ) {
	const size_t n = grb::size( z );
	grb::Matrix< int > Tmp( n, n );

	grb::maskedOuter( Tmp, A, z, m, grb::operators::add< int >(),Phase::RESIZE );
	grb::maskedOuter( Tmp, A, z, m, grb::operators::add< int >() );

	grb::eWiseApply( B, Tmp, A, grb::operators::right_assign< int >(), Phase::RESIZE );
	grb::eWiseApply( B, Tmp, A, grb::operators::right_assign< int >() );
}*/


void findCycle2Augmentations( grb::Matrix< int > &G2C, grb::Matrix< int > &D2C, const grb::Matrix< int > &A, const grb::Matrix< int > &M, const grb::Vector< int > &z ) {
	const size_t n = grb::size( z );

	grb::Semiring< grb::operators::add< int >, grb::operators::mul< int >, grb::identities::zero, grb::identities::one > standard;


	grb::Vector< int > m( n ); 
	grb::mxv( m, M, z, standard ); 

	grb::Matrix< int > M_Lower( n, n );
	grb::Matrix< int > M_Upper( n, n );

	grb::select( M_Lower, M, grb::operators::is_strictly_lower< int, int, int >(), Phase::RESIZE );
	grb::select( M_Lower, M, grb::operators::is_strictly_lower< int, int, int >() );

	grb::select( M_Upper, M, grb::operators::is_strictly_upper< int, int, int >(), Phase::RESIZE );
	grb::select( M_Upper, M, grb::operators::is_strictly_upper< int, int, int >() );

	grb::Vector< int > ml( n );
	grb::mxv( ml, M_Lower, z, standard ); 

	grb::Vector< int > mu( n );
	grb::mxv( mu, M_Upper, z, standard ); 

	grb::Vector< int > mw( n );
	grb::set( mw, 0 );
	grb::mxv( mw, M, z, standard , Phase::RESIZE );
	grb::mxv( mw, M, z, standard , Phase::EXECUTE );
	

	grb::Matrix< int > Temp( n, n );

	selectSubmatrix( Temp, A, m, z );

	grb::Matrix< int > A_m( n, n );

	selectSubmatrix( A_m, Temp, z, m );

	grb::Matrix< int > P_m( n, n );

	grb::Matrix< int > A_mM( n, n );

	grb::set( P_m, M, 1, Phase::RESIZE );
	grb::set( P_m, M, 1, Phase::EXECUTE );

	grb::mxm( Temp, P_m, A_m, Phase::RESIZE );
	grb::mxm( Temp, P_m, A_m, Phase::EXECUTE );

	grb::mxm( A_mM, Temp, P_m, Phase::RESIZE );
	grb::mxm( A_mM, Temp, P_m, Phase::EXECUTE );

	grb::Matrix< int > C( n, n );
	grb::Matrix< int > Temp_C( n, n );
	grb::Matrix< int > D2C1( n, n );
	grb::Matrix< int > D2C2( n, n );

	grb::eWiseApply( Temp_C, A_mM, A_m, grb::operators::add< int >(), Phase::RESIZE );
	grb::eWiseApply( Temp_C, A_mM, A_m, grb::operators::add< int >(), Phase::EXECUTE );

	grb::maskedOuter( Temp, Temp_C, mw, mw, grb::operators::add< int >(),Phase::RESIZE );
	grb::maskedOuter( Temp, Temp_C, mw, mw, grb::operators::add< int >() );

	grb::eWiseApply( C, Temp_C, Temp, grb::operators::subtract< int >(), Phase::RESIZE );
	grb::eWiseApply( C, Temp_C, Temp, grb::operators::subtract< int >(), Phase::EXECUTE );

	selectSubmatrix( D2C1, C, mu, z );

	maxPerRow( Temp, D2C1, z );

	grb::select( D2C1, Temp, grb::operators::is_positive<int,int,int>(), Phase::RESIZE );
	grb::select( D2C1, Temp, grb::operators::is_positive<int,int,int>(), Phase::EXECUTE );

	if( nnz( D2C1 ) ) {

		grb::mxm( Temp, P_m, D2C1, Phase::RESIZE );
		grb::mxm( Temp, P_m, D2C1, Phase::EXECUTE );

		grb::mxm( D2C2, Temp, P_m, Phase::RESIZE );
		grb::mxm( D2C2, Temp, P_m, Phase::EXECUTE );

		Monoid<
			grb::operators::add< int >,
			grb::identities::zero
		> addition_monoid;

		grb::eWiseApply( D2C, D2C1, D2C2, addition_monoid, Phase::RESIZE );
		grb::eWiseApply( D2C, D2C1, D2C2, addition_monoid );

		grb::Vector< int > r( n ); 
		grb::mxv( r, M, z, standard ); 

		grb::maskedOuter( G2C, M, r, z, grb::operators::left_assign< int >() );
	}
	else {

		grb::clear( D2C );
		grb::clear( G2C );

	}

}

void findPath2Augmentations( grb::Matrix< int > &G2P, grb::Matrix< int > &D2P, const grb::Matrix< int > &A, const grb::Matrix< int > &M, const grb::Vector< int > &z ) {
	const size_t n = grb::size( z );

	grb::Semiring< grb::operators::add< int >, grb::operators::mul< int >, grb::identities::zero, grb::identities::one > standard;


	grb::Vector< int > m( n ); 
	grb::mxv( m, M, z, standard ); 

	grb::Matrix< int > M_Lower( n, n );
	grb::Matrix< int > M_Upper( n, n );

	grb::select( M_Lower, M, grb::operators::is_strictly_lower< int, int, int >(), Phase::RESIZE );
	grb::select( M_Lower, M, grb::operators::is_strictly_lower< int, int, int >() );

	grb::select( M_Upper, M, grb::operators::is_strictly_upper< int, int, int >(), Phase::RESIZE );
	grb::select( M_Upper, M, grb::operators::is_strictly_upper< int, int, int >() );

	grb::Vector< int > ml( n );
	grb::mxv( ml, M_Lower, z, standard ); 

	grb::Vector< int > mu( n );
	grb::mxv( mu, M_Upper, z, standard ); 

	grb::Vector< int > mw( n );
	grb::set( mw, 0 );
	grb::mxv( mw, M, z, standard , Phase::RESIZE );
	grb::mxv( mw, M, z, standard , Phase::EXECUTE );

	grb::Matrix< int > D1( n, n );
	grb::Matrix< int > G1( n, n );

	searchOneAugmentations( G1, D1, M, A, z );

	grb::Matrix< int > D2P_tmp( n, n );

	selectSubmatrix( D2P_tmp, D1, m, z );

	grb::Matrix< int > D2P_tmp_a( n, n );
	
	grb::set( D2P_tmp_a, D2P_tmp, Phase::RESIZE );
	grb::set( D2P_tmp_a, D2P_tmp );

	grb::eWiseLambda( [&D2P_tmp_a]( const size_t i, const size_t j, int& nz ) { nz = j; }, D2P_tmp_a );

	grb::Vector< int > a( n );
	grb::mxv( a, D2P_tmp_a, z, standard ); 

	grb::Vector< int > a_g( n );
	grb::mxv( a_g, D2P_tmp, z, standard ); 

	grb::Matrix< int > G2P_tmp1( n, n );

	grb::maskedOuter(G2P_tmp1, M, a_g, a_g, grb::operators::add< int >(), Phase::RESIZE );
	grb::maskedOuter(G2P_tmp1, M, a_g, a_g, grb::operators::add< int >(), Phase::EXECUTE );

	grb::Matrix< int > G2P_tmp2( n, n );

	grb::select( G2P_tmp2, G2P_tmp1, grb::operators::is_positive<int,int,int>(), Phase::RESIZE );
	grb::select( G2P_tmp2, G2P_tmp1, grb::operators::is_positive<int,int,int>(), Phase::EXECUTE );
	

	if( nnz( G2P_tmp2 ) ) {

		grb::Matrix< int > P_m( n, n );

		grb::set( P_m, M, 1, Phase::RESIZE );
		grb::set( P_m, M, 1, Phase::EXECUTE );

		grb::Vector< int > v( n );

		grb::mxv( v, P_m, a, Phase::RESIZE );
		grb::mxv( v, P_m, a, Phase::EXECUTE );

		
		
	}
	else {

		grb::clear( D2P );
		grb::clear( G2P );

	}

}

/*
A function that finds the 1/2 approximation of maximum weight bipartite matching in A.
*/
void oneOverTwoApproximation( grb::Matrix< int > &M, const grb::Matrix< int > &A ) {

	const size_t n = grb::nrows( A );
	grb::Vector< int > z( n );

	grb::set( z , 1 );

	grb::Matrix< int > Augment( n, n );


	//Find the set of potential 1-augmentations
	findOneAugmentations( Augment, M, A, z );

	int iter = 1;

	while( nnz(Augment) ) { //While such a set is not empty

		/*(void)printf("Iteration %d: Nonzero count in Augmentation is: %lu\n\n", iter,nnz(Augment));

		for(auto it: Augment) {
			(void)printf("\t (%lu,%lu) = %d\n",it.first.first,it.first.second, it.second);
		}*/

		//Apply them to M
		flip( M, Augment, A, z );


		grb::clear(Augment);
		findOneAugmentations( Augment, M, A, z );


		/*(void)printf("Iteration %d: Nonzero count in M is: %lu\n\n", iter,nnz(M));

		for(auto it: M) {
			(void)printf("\t (%lu,%lu) = %d\n",it.first.first,it.first.second, it.second);
		}*/

		iter += 1;
	}

}

int main( int argc, char ** argv ) {
	(void)argc;
	(void)printf( "Illustration executable: %s\n\n", argv[ 0 ] );

	


	grb::Matrix< int > M( 10, 10 );
	grb::Matrix< int > A( 10, 10 );

	buildMatrixUnique( A, &( I[ 0 ] ), &( J[ 0 ] ), adj, 50, SEQUENTIAL );
	buildMatrixUnique( M, &( M_I[ 0 ] ), &( M_J[ 0 ] ), M_adj, 6, SEQUENTIAL );

	/*

	grb::Matrix< int > C( 10, 10 );

	grb::internal::maxPerRow(C, A, Phase::RESIZE );
	grb::internal::maxPerRow(C, A, Phase::EXECUTE );

	(void)printf("Nonzero count in C is: %lu\n\n",nnz(C));

	for(auto it: C) {
		(void)printf("(%lu,%lu) = %d\n",it.first.first,it.first.second, it.second);
	}
	
	
	*/

	oneOverTwoApproximation( M, A );

	(void)printf("Nonzero count in M is: %lu\n\n",nnz(M));

	for(auto it: M) {
		(void)printf("(%lu,%lu) = %d\n",it.first.first,it.first.second, it.second);
	}
	
	
	return EXIT_SUCCESS;
}

