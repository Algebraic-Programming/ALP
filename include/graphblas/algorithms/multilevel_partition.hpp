#ifndef _H_GRB_MULTILEVEL_PARTITION
#define _H_GRB_MULTILEVEL_PARTITION

#include <random>
#include <graphblas.hpp>
#include <graphblas/algorithms/spec_part_utils.hpp>

#include <set>

using namespace grb;
using namespace algorithms;

// Gabriel's code refactored for the new GraphBLAS version
namespace grb {
	namespace algorithms {

		template < typename pType >
		RC m_zero( 
			pType &z,
			Vector< pType > &M,
			pType &r
			) {
			
				const grb::Semiring<
				     grb::operators::add< pType >,
					grb::operators::mul< pType >,
					grb::identities::zero, 
					grb::identities::one
			    > standard_sr;
			Vector< pType > R( grb::size( M ) );
			grb::setElement( R, 1, r );
			grb::dot( z, M, R, standard_sr );


			return SUCCESS;

		}

		

		template< typename IOType, typename pType >
		RC update_weight_matrix(
			Matrix< IOType > &Aw,
			pType v,
			pType i_max
		) {

			const grb::Semiring<
				     grb::operators::add< IOType >,
					grb::operators::mul< IOType >,
					grb::identities::zero, 
					grb::identities::one
			    > standard_sr;
			// std::vector<int> Ivec, Jvec;
			// std::vector<double> Vvec;
			int n_vertices = grb::ncols(Aw);
			// for( int i = 0; i < n_vertices; ++i) {
			// 	Vector< IOType > col( n_vertices );
			// 	if (i != i_max) {
			// 		grb::setElement( col, 1, i );
			// 	}
			// 	if (i == v) {
			// 		grb::setElement( col, 1, i_max );
			// 	}
			// 	Vector< IOType > vi( n_vertices );
			// 	grb::mxv( vi, Aw, col, standard_sr );
			// 	for ( const std::pair< size_t, IOType > &pair1 : vi ) {
			// 		double val1 = pair1.second;
			// 		if (val1 != 0) {
			// 			int i1 = pair1.first;
			// 			Ivec.push_back( i1 );
			// 			Jvec.push_back( i );
			// 			Vvec.push_back( val1 );
			// 		}
			// 	}
			// }
			std::vector<int> Ivec2, Jvec2;
			std::vector<double> Vvec2;
			for( int i = 0; i < n_vertices; ++i ) {
				
				if ( i != i_max ) {
					Ivec2.push_back( i );
					Jvec2.push_back( i );
					Vvec2.push_back( 1 );
				}
				if ( i == v ) {
					Ivec2.push_back( i_max );
					Jvec2.push_back( i );
					Vvec2.push_back( 1 );
				}
			}

			int* I2 = &Ivec2[0];
			int* J2 = &Jvec2[0];
			double* V2 = &Vvec2[0];

			// int* I = &Ivec[0];
			// int* J = &Jvec[0];
			// double* V = &Vvec[0];

			Matrix< IOType > one_( n_vertices, n_vertices );
			grb::resize( one_, Vvec2.size() );
			grb::buildMatrixUnique( one_, &(I2[0]), &(J2[0]), &(V2[0]), Vvec2.size(), PARALLEL );
			// Matrix< IOType > tmp( grb::nrows( Aw ), grb::ncols( Aw ) );
			grb::mxm( Aw, Aw, one_, standard_sr );
			
			// grb::resize( tmp, Vvec.size() ) ;
			// grb::buildMatrixUnique( tmp, &(I[0]), &(J[0]), &(V[0]), Vvec.size(), SEQUENTIAL );

			
			return SUCCESS;

		}


		template< typename IOType, typename pType >
		RC coarsening_step(
			Matrix< IOType > &Aw, 
			Vector< pType > &M,
			std::vector< Matrix< pType >* > &T,
			size_t &N
		) {
			
			const grb::Semiring<
				     grb::operators::add< IOType >,
					grb::operators::mul< IOType >,
					grb::identities::zero, 
					grb::identities::one
			    > standard_sr;
			int n = grb::ncols( Aw );
			int m = grb::nrows( Aw );

			T.push_back( new Matrix< pType > ( n,n ) );

			std::vector< int > Ivec;
			std::vector< int > Jvec;
			std::vector< int > Vvec;
			std::cout << "sanity check n " << n << std::endl;
			for ( int i = 0; i < n; ++i ) {
				Ivec.push_back( i );
				Jvec.push_back( i );
				Vvec.push_back( 1 );
			}

			Vector< pType > Mtmp( grb::size( M ) );
			grb::set( Mtmp, M );

			float N_old = N / 1.7; // i dont get thiss ssssd

			while ( N > N_old ) {
				// std::uniform_int_distribution<> distr(0,n-1);
				size_t seed_uniform = std::chrono::system_clock::now().time_since_epoch().count();
				std::default_random_engine random_generator( seed_uniform );
				std::uniform_int_distribution< size_t > uniform( 0, n - 1 );

				int r;
				int z;
				do {
					r = uniform( random_generator ); 
					m_zero( z, Mtmp, r );
				
				} while(!z);
				int v_i = r;
				grb::Vector< int > v_pos( n );
				grb::setElement( v_pos, 1, v_i );
				Vector< int > v( m );
				grb::mxv( v, Aw, v_pos, standard_sr );
				
				Vector< int > edgew( n );
				// grb::mxv< grb::descriptors::transpose_matrix >( edgew, Mtmp, standard_sr.getAdditiveMonoid(), Aw, v, standard_sr);
				grb::mxv< grb::descriptors::transpose_matrix >( edgew, Mtmp, Aw, v, standard_sr );

				int max = 0, i_max = -1;
				// i_max = -1
				for ( const std::pair< size_t, double > &pair: edgew ) {
					const double val = pair.second;
					if ( val >= max && pair.first != v_i ) {
						max = val;
						i_max = pair.first;
					}
				}

				update_weight_matrix( Aw, v_i, i_max );
				// if (i_max != -1) {
				Ivec.push_back( i_max );
				Jvec.push_back( v_i );
				Vvec.push_back( 1 );
				//}
				grb::setElement( M, 0, i_max );
				grb::setElement( Mtmp, 0, i_max );
				grb::setElement( Mtmp, 0, v_i );

				N = N - 1;

			}

			int* I = &Ivec[0];
			int* J = &Jvec[0];
			int* V = &Vvec[0];
			grb::resize( *(T.back()), Vvec.size() );

			grb::buildMatrixUnique( *(T.back()), &(I[0]), &(J[0]), &(V[0]), Vvec.size(), SEQUENTIAL );
			return SUCCESS;

		}


		template< typename IOType, typename pType >
		RC coarsening( Matrix< IOType > &Aw, Vector< pType > &M, std::vector< Matrix< pType >* > &T, pType k ) {
			size_t min_size = 100*k;
			size_t n = grb::ncols( Aw );
			while( n > min_size ) {
				coarsening_step( Aw, M, T, n );
			}

			coarsening_step( Aw, M, T, n );
			return SUCCESS;
		}

		template< typename IOType, typename pType > 
		RC initial_partition( 
			Vector< pType > &M, 
			Vector< pType > &P, 
			const pType &k, 
			std::vector< IOType > &sizes 
			) {

				// std::uniform_int_distribution<> distr( 1, k );
				size_t seed_uniform = std::chrono::system_clock::now().time_since_epoch().count();
				std::default_random_engine random_generator( seed_uniform );
				std::uniform_int_distribution< size_t > uniform( 1, k );
				for( const std::pair< size_t, double > &pair : M ) {
					IOType val = pair.second;
					if ( val != 0 ) {
						int p = uniform( random_generator );
						grb::setElement( P, p, pair.first );
						sizes[p]++;
					}
				}

			return SUCCESS;

		}


	RC modified_mxm( grb::Matrix< double > &Aw, grb::Vector< int > &w, grb::Matrix< int > &A) {
	// Variables to build matrix Aw later
	std::vector< int > Ivec, Jvec;
	const grb::Semiring<
				     grb::operators::add< double >,
					grb::operators::mul< double >,
					grb::identities::zero, 
					grb::identities::one
			    > standard_sr;
	std::vector< double > Vvec;
	int n_nets = grb::size( w );
	// build the w vectors
	for( const std::pair< size_t, int > &pair : w ) {
		// Build column vector of i-th entry of w
		int i = pair.first;
		double val = pair.second;
		val = ( val <= 1 ) ? 0 : ( 1 / ( val - 1 ) ); 
		grb::Vector< double > wi( n_nets );
		grb::setElement( wi, val, i );
		// Calculate vi = wi * A
		grb::Vector< double > vi( grb::ncols( A ) );
		auto rc = grb::vxm( vi, wi, A, standard_sr );
		assert( rc == grb::SUCCESS );

		// Copy the values of vi into matrix variables
		for( const std::pair< size_t, double > &pair1 : vi ) {
			double val1 = pair1.second;
			if ( val1 != 0 ) {
				int i1 = pair1.first;
				Ivec.push_back( i );
				Jvec.push_back( i1 );
				Vvec.push_back( val1 );
			}
		}
	}

	// Build matrix Aw from matrix variables
	int* I = &Ivec[0];
	int* J = &Jvec[0];
	double* V = &Vvec[0];
	grb::resize( Aw, Vvec.size() );
	grb::buildMatrixUnique( Aw, &(I[0]), &(J[0]), &(V[0]), Vvec.size(), SEQUENTIAL );

	return SUCCESS;
}


		template< typename IOType, typename pType > 
		RC uncoarsening_step( 
			Matrix< IOType > &Aw,
			Vector< pType > &M,
			Vector< pType > &P,
			std::vector< IOType > &sizes,
			const double &c, 
			const int &k
		) {

			
			const grb::Semiring<
				     grb::operators::add< IOType >,
					grb::operators::mul< IOType >,
					grb::identities::zero, 
					grb::identities::one
			    > standard_sr;

			const grb::Semiring< 
				grb::operators::logical_and< pType >, 
				grb::operators::logical_or< pType >, 
				grb::identities::logical_true, 
				grb::identities::logical_false > ao_sr;

			RC rc;
			int cursize = 0;
			for ( const std::pair< size_t, IOType > &pair : M ) {
				if ( pair.second != 0 ) cursize += 1;

			}
			std::cout << "curr size is " << cursize << std::endl;
			double maxsize = c * cursize/k;
			
			for( const std::pair< size_t, double > &pair : M ) {
				if (pair.second == 0) continue;
				Vector< double > v_pos( grb::ncols( Aw ) );
				grb::setElement( v_pos, 1, pair.first );
				Vector< IOType > v( grb::nrows( Aw ) );
				grb::set( v, 0 );
				// rc = grb::mxv( v, grb::operators::add< double, double, double >(), Aw, v_pos, standard_sr );
				rc = grb::mxv( v, Aw, v_pos, standard_sr );
				int a;

				for( const std::pair< size_t, IOType > &pair1 : P ) {
					if( pair1.first == pair.first ) a = pair1.second;
				}

				Vector< IOType > nvt( grb::ncols( Aw ) );
				Vector< IOType > nv( grb::ncols( Aw ) );
				grb::set( nvt, 0 );
				grb::set( nv, 0 );
				// rc = grb::vxm(nvt, M, grb::operators::add< IOType, IOType, IOType >(), v, Aw, standard_sr);
				// rc = grb::vxm( nvt, v, Aw, standard_sr );
				rc = grb::vxm( nvt, M, v, Aw, standard_sr );
				assert( v == grb::SUCCESS );
				for( const std::pair< size_t, double > &pairn : nvt ) {
					const double val = pairn.second;
					if ( val == 0 ) {
						grb::setElement( nv, 0, pairn.first );
					} else {
						grb::setElement( nv, 1, pairn.first );
					}
				}

				Vector< pType > Ia( grb::ncols( Aw ) );
				grb::set( Ia, 0 );
				rc = grb::eWiseApply< grb::descriptors::dense > ( Ia, P, a, grb::operators::equal< pType, pType, pType >() );
				assert( rc == grb::SUCCESS );
				Vector< pType > internal( grb::ncols( Aw ));
				grb::set( internal, 0 );
				grb::Vector< pType > nv_helper( grb::size( nv ) );

				grb::set( nv_helper, nv );

				rc = grb::eWiseApply( internal, nv_helper, Ia, nv, grb::operators::equal< pType, pType, pType >() );
				assert( rc == grb::SUCCESS );

				bool is_internal = true;

				for( const std::pair< size_t, pType > &pair1 : internal ) {
					if ( pair.second == 0 ) {
						is_internal = false;
					}
				}
				if (!is_internal) continue;


				Vector< pType > neighboring_parts( grb::ncols( Aw ) );
				grb::set( neighboring_parts, 0 );

				rc = grb::eWiseApply( neighboring_parts, M, nv, P, grb::operators::mul< pType, pType, pType >() );
				assert( rc == grb::SUCCESS );
				std::set< pType, std::greater< pType > > Nv;
				for( const std::pair< size_t, int > &pairnp : neighboring_parts ) {
					if( pairnp.second != 0 ) Nv.insert( pairnp.second );
				}

				int minb; 
				int mincost = 100000;
				for( auto b : Nv ) {
					// Check if moving to b violates load balancing constraints
					if( sizes[b] > maxsize ) continue;

					// Compute the relevant cost of moving v to b
					grb::Vector< int > relcost( grb::nrows( Aw ) );
					grb::set( relcost, 0 );
						for( auto c : Nv ) {
							if( c == b )
								grb::setElement( P, b, pair.first );
							// Build vector that tells us which vertices take part of c
							grb::Vector< int > Ic( grb::ncols( Aw ) );
							grb::set( Ic, 0 ); // making Ia dense
							rc = grb::eWiseApply( Ic, M, P, c, grb::operators::equal< int, int, int>() ); // Need a mask here
							assert( rc == grb::SUCCESS );

							grb::Vector< int > spansc( grb::nrows( Aw ) );
							grb::set( spansc, 0 );
							// Needs to be masked
							// rc = grb::vxm< grb::descriptors::transpose_matrix >( spansc, v, grb::operators::add<double, double, double>(), Ic, Aw, ao_sr);
							rc = grb::vxm< grb::descriptors::transpose_matrix >( spansc, v, Ic, Aw, ao_sr );
							assert( rc == grb::SUCCESS );

							rc = foldl( relcost, spansc, grb::operators::add< int, int, int >() );
							assert( rc == grb::SUCCESS );
							grb::setElement( P, a, pair.first );
						}
			
						int cost = 0;
						for( const std::pair< size_t, int > &pairnp : relcost ) 
			       			cost += pairnp.second;	

						if( cost < mincost ){
							mincost = cost;
							minb = b;
						}
				}

				if( minb == 0 ) continue;
				grb::setElement( P, minb, pair.first );
				sizes[a]--;
				sizes[minb]++;
			}
			return SUCCESS;
		}



		template< typename IOType >
		RC uncoarsen_weight_matrix(
			Matrix< IOType > &Aw, 
			Matrix< int > &Ts
		) {
			const grb::Semiring<
				     grb::operators::add< IOType >,
					grb::operators::mul< IOType >,
					grb::identities::zero, 
					grb::identities::one
			    > standard_sr;
			std::vector< int > Ivec, Jvec;
			std::vector< IOType > Vvec;

			size_t n = grb::ncols( Ts );
			size_t m = grb::nrows( Ts );
			for( int i = 0; i < n; ++i ) {
				Vector< int > vi( m );
				grb::setElement( vi, -1, i );
				Vector< int > Ti( n );
				grb::mxv( Ti, Ts, vi, standard_sr );

				if( i < m ) grb::setElement( Ti, 1, i );
				Vector< IOType > tmp( grb::nrows( Aw ) );
				RC rc = grb::mxv( tmp, Aw, Ti, standard_sr );

				for( const std::pair< size_t, IOType > &pair1: tmp ) {
					double val1 = pair1.second;
					if(val1 != 0) {
						int i1 = pair1.first;
						Ivec.push_back( i1 );
						Jvec.push_back( i );
						Vvec.push_back( val1 );
					}
				}
			}

			int* I = &Ivec[0];
			int* J = &Jvec[0];
			double* V = &Vvec[0];
			grb::resize( Aw, Vvec.size() );
			grb::buildMatrixUnique( Aw, &(I[0]), &(J[0]), &(V[0]), Vvec.size(), SEQUENTIAL );

			return SUCCESS;

		}


		template< typename IOType, typename pType > 
		RC uncoarsening( Matrix< IOType > &Aw, Vector< pType > &M, Vector< pType > &P, std::vector< Matrix< pType >* > &T,
		std::vector< IOType > &sizes
		) {
			const grb::Semiring<
				     grb::operators::add< IOType >,
					grb::operators::mul< IOType >,
					grb::identities::zero, 
					grb::identities::one
			    > standard_sr;
			Vector< pType > M_temp( grb::size( M ) );
			Vector< pType > P_temp( grb::size( P ) );
			RC rc = SUCCESS;

			for( int s = 0; s < T.size(); ++s ) {
				rc = grb::mxv( M_temp, *(T[s]), M, standard_sr );
				rc = grb::mxv( P_temp, *(T[s]), P, standard_sr );
				grb::set( M, M_temp );
				grb::set( P, P_temp );
				uncoarsen_weight_matrix( Aw, *(T[s]) );
			}
			const double c = 1.1;
			const int k = 2;
			
			for( const std::pair< size_t, double > &pair : P ) {
				std::cout << "P[" << pair.first << "] = " << pair.second << std::endl;

			}
			// nothing is written to P i think 
			uncoarsening_step( Aw, M, P, sizes, c, k );

			return SUCCESS;
		}


		template< typename IOType, typename pType >
		RC partition(
			Matrix< pType > &A, 
			const pType &k,
			const IOType &c
		) {

			const grb::Semiring<
				    grb::operators::add< IOType >,
					grb::operators::mul< IOType >,
					grb::identities::zero, 
					grb::identities::one
			    > standard_sr;
			Vector< pType > M( grb::ncols( A ) );
			grb::set( M, 1 );
			Vector< pType > P( grb::ncols( A ) );
			grb::set( P, 0 );

			std::vector< Matrix< int >* > T;
			int n = grb::ncols( A );
			int m = grb::nrows( A );
			
			Vector< IOType > ones( n );
			grb::set( ones, 1 );
			
			// Vector< IOType > w( m );
			Vector< pType > w( m );
			grb::mxv( w, A, ones, standard_sr );
			Matrix< IOType > Aw( m, n );
			

			// A: int
			// w: double
			// Aw: double 
			// convert A to double
			// Matrix < IOType > W( m, m );
			// Matrix < IOType > Aio( n, m );
			// grb::set( Aio, A );
			//grb::mxv(Aw, w, Aio, standard_sr);
			// buildMatrix
			// Matrix< IOType > W( m, m );
			// auto converter = grb::utils::makeVectorToMatrixConverter< int, IOType >( w, []( const int & ind, const IOType & val ) {
			// 		return std::make_pair( val, ind );
			// 	} );
			// 
			// grb::buildMatrixUnique(W,converter.begin(), converter.end(), SEQUENTIAL);
			// std::vector< int > Ivec, Jvec;
			// std::vector< IOType > Vvec;
			// for ( const std::pair< size_t, IOType > &pair : w ) {
			// 	int i = pair.first;
			// 	IOType val = pair.second;
			// 	for ( int f = 0; f < m; ++f ) {
			// 		Ivec.push_back( i );
			// 		Jvec.push_back( f );
			// 		Vvec.push_back( val );
			// 	}
			// }
			// grb::resize(W,m*m);
			// grb::buildMatrixUnique(W, &(Ivec[0]), &(Ivec[m-1]),&(Jvec[0]), &(Jvec[m-1]),&(Vvec[0]), &(Vvec[m-1]),SEQUENTIAL);
			// grb::buildMatrixUnique( W, &(Ivec[0]), &(Jvec[0]), &(Vvec[0]), Vvec.size(), PARALLEL );
			// for(const std::pair< std::pair< size_t, size_t >, double > &pair : A) {
			// 	std::cout << "enenen" << pair.first.first << " " << pair.first.second << " " << pair.second << std::endl;
			// }
			// grb::mxm( Aw, W, A, standard_sr );	

			// grb::mxm< grb::descriptors::dense >( Aw, A, W, standard_sr );
			
			int n_nets = grb::size( w );
			Matrix< IOType > W( n_nets, n_nets );
			std::vector< int > WI, WJ;
			std::vector< IOType > WV;
			Vector< IOType > w_vals( m );
			int num_elem = 1;
			for ( const std::pair< size_t, IOType > &pair : w ) {
				int i = pair.first;
				IOType val = pair.second;
				val = ( val <= 1 ) ? 0 : 1 / ( val - 1 );
				if (val !=  0 ) {
					WJ.push_back( i );
					WI.push_back( i );
					WV.push_back( val );		
					grb::setElement( w_vals, val, i );
					num_elem++;
				}
			}
			
			int* I = &WI[0];
			int* J = &WJ[0];
			double* V = &WV[0];
			grb::resize( W, WV.size() );
			// auto converter = grb::utils::makeVectorToMatrixConverter< void, IOType >( w_vals, []( const std::pair< size_t, IOType > &pair ) {
			// 	return std::make_pair( pair.first, pair.second );
			// } );
			// auto converter = grb::utils::makeVectorToMatrixConverter< void, IOType >( w_vals, []( const size_t &ind, const IOType &val ) {
			// 	return std::make_pair( ind, val );
			// });
			
			// grb::buildMatrixUnique( W, converter.begin(), converter.end(), PARALLEL );
			// grb::buildMatrixUnique(W, w_vals.begin(), w_vals.end(), SEQUENTIAL );
			std::cout << "A rows: " << grb::nrows( A ) << " A cols: " << grb::ncols( A ) << std::endl;
			std::cout << "W rows: " << grb::nrows( W ) << " W cols: " << grb::ncols( W ) << std::endl;

			grb::buildMatrixUnique( W, &(I[0]), &(J[0]), &(V[0]), WV.size(), SEQUENTIAL );
		
	
			std::cout << "here" << std::endl;

			// RC diditfuckup = grb::mxm( Aw, W, A, standard_sr );
			
			modified_mxm( Aw, w, A );
			// for (const std::pair< std::pair< size_t, size_t >, double> &pair : Aw ) {
			// 	std::cout << "i: " << pair.first.first << " j: " << pair.first.second << " v: " << pair.second << std::endl;
			// }
			// modified_mxm( Aw, w, A );
			std::vector< IOType > sizes( k );
		
			coarsening( Aw, M , T , k );

			initial_partition( M, P, k, sizes );
			uncoarsening( Aw, M , P , T, sizes );

			
			for( const std::pair< size_t, IOType > &pair : P ) {
				std::cout << "P[" << pair.first << "] = " << pair.second << std::endl;
			}

			return SUCCESS;


		}


	}
}
#endif
