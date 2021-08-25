#include <random>
#include <graphblas.hpp>

#include <set>

using namespace grb;
using namespace algorithms;

// Gabriel's code refactored for the new GraphBLAS version
namespace grb {
	namespace algorithms {
		/* 
		grb::Semiring< int, int, int, int,
		 grb::operators::logical_and, 
		 grb::operators::logical_or, 
		 grb::identities::logical_true, 
		 grb::identities::logical_false > ao_sr;

		*/


		template <typename IOType, typename pType>
		RC m_zero( 
			pType &z, 
			Vector<IOType> &M, 
			const pType r, 
			const grb::Semiring<
				     grb::operators::add< IOType >,
					grb::operators::mul< IOType >,
					grb::identities::zero, 
					grb::identities::one
			    > &standard_sr) {

			Vector< pType > R( grb::size( M ) );
			grb::setElement( R, 1, r);
			grb::dot( z, M, R, standard_sr);
			return SUCCESS;

		}



		template< typename IOType, typename pType>
		RC update_weight_matrix(
			Matrix< IOType > &Aw,
			pType v,
			pType i_max,
			const grb::Semiring<
				     grb::operators::add< IOType >,
					grb::operators::mul< IOType >,
					grb::identities::zero, 
					grb::identities::one
			    > &standard_sr
		) {
			std::vector<int> Ivec, Jvec;
			std::vector<double> Vvec;
			int n_vertices = grb::ncols(Aw);
			for( int i = 0; i < n_vertices; ++i) {
				Vector< IOType > col(n_vertices);
				if (i != i_max) {
					grb::setElement(col, 1, i);
				}
				if (i == v) {
					grb::setElement(col, 1, i_max);
				}

				Vector< IOType > vi( n_vertices );
				grb::mxv(vi, Aw, col, standard_sr);

				for (const std::pair< size_t, IOType > &pair1 : vi) {
					double val1 = pair.second;
					if (val1 != 0) {
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
			grb::resize( Aw, Vvec.size()) ;
			grb::buildMatrixUnique( Aw, &(I[0]), &(J[0]), &(V[0]), Vvec.size(), SEQUENTIAL );

			return SUCCESS;

		}


		template<typename IOType, typename pType>
		RC coarsening_step(
			Matrix< IOType > &Aw, 
			Vector< pType > &M,
			std::vector<Matrix<pType>*> &T,
			pType &N,
			const grb::Semiring<
				     grb::operators::add< IOType >,
					grb::operators::mul< IOType >,
					grb::identities::zero, 
					grb::identities::one
			    > &standard_sr
		) {
			int n = grb::ncols(Aw);
			int m = grb::nrows(Aw);

			T.push_back(new Matrix< pType > (n,n));

			std::vector< int > Ivec;
			std::vector< int > Jvec;
			std::vector< int > Vvec;

			for (int i = 0; i < n; ++i ) {
				Ivec.push_back(i);
				Jvec.push_back(i);
				Vvec.push_back(1);
			}

			Vector< pType > Mtmp(grb::size( M ));
			grb::set(Mtmp, M);

			float N_old = N / 1.7;

			while (N > N_old) {
				std::uniform_int_distribution<> distr(0,n-1);
				int r;
				do {
					r = distr(gen); // TODO
				} while(!m_zero(Mtmp, r));

				int v_i = r;
				grb::Vector< int > v_pos( n );
				grb::setElement( v_pos, 1, v_i );
				Vector< int > v(m);
				grb::mxv(v, Aw, v_pos, standard_sr);

				Vector< int > edgew( n );
				grb::mxv< grb::descriptors::transpose_matrix >( edgew, Mtmp, standard_sr.getAdditiveMonoid(), Aw, v, standard_sr);

				int max = 0, i_max = -1;
				for (const std::pair<size_t, double > &pair: edgew) {
					const double val = pair.second;
					if (val >= max && pair.first != v_i) {
						max = val;
						i_max = pair.first;
					}
				}

				update_weight_matrix(Aw, v_i, i_max);
				Ivec.push_back(i_max);
				Jvec.push_back(v_i);
				Vvec.push_back(1);

				grb::setElement(M, 0, i_max);
				grb::setElement(Mtmp, 0, i_max);
				grb::setElement(Mtmtp, 0, v_i);

				N = N - 1;

			}

			int* I = &Ivec[0];
			int* J = &Jvec[0];
			int* V = &Vvec[0];
			grb::resize( *(T.back()), Vvec.size());
			grb::buildMatrixUnique( *(T.back()), &(I[0]), &(J[0]), &(V[0]), Vvec.size(), SEQUENTIAL);

			return SUCCESS;

		}


		template< typename IOType, typename pType >
		RC coarsening( Matrix< IOType > &Aw, Vector< pType > &M, std::vector< Matrix< pType >* > &T, pType k ) {
			size_t min_size = 100*k;
			size_t n = grb::ncols( Aw );
			while( n > min_size ) {
				coarsening_step( Aw, M, T, n );
			}

			coarsening_step( Aw, M, T, n);

			return SUCCESS;
		}

		template< typename IOType, typename pType > 
		RC initial_partition( 
			Vector< pType > &M, 
			Vector< pType > &P, 
			pType &K, 
			std::vector< IOType > &sizes 
			) {

				std::uniform_int_distribution<> distr( 1, k );
				for( const std::pair< size_t, double > &pair : M ) {
					IOType val = pair.second;
					if ( val != 0 ) {
						int p = distr( gen );
						grb::setElement( P, p, pair.first );
						sizes[p]++;
					}
				}

			return SUCCESS;

		}



		template< typename IOType, typename pType > 
		RC uncoarsening_step( 
			Matrix< IOType > &Aw,
			Vector< pType > &M,
			Vector< pType > &P,
			std::vector< IOType > &sizes,
			const grb::Semiring<
				     grb::operators::add< IOType >,
					grb::operators::mul< IOType >,
					grb::identities::zero, 
					grb::identities::one
			    > &standard_sr
		) {
			RC rc;
			int cursize = 0;
			for ( const std::pair< size_t, IOType > &pair : M) {
				if (pair.second != 0) cursize += 1;

			}
			double maxsize = c_g * cursize/k_g;

			for( const std::pair< size_t, double > &pair : M ) {
				if (pair.second == 0) continue;
				Vector< double > v_pos( grb::ncols( Aw ));
				grb::setElement(v_pos, 1, pair.first);
				Vector< IOType > v(grb::nrows( Aw ));
				grb::set( v, 0 );
				rc = grb::mxv( v, grb::operators::add< double, double, double >(), Aw, v_pos, standard_sr );
				
				int a;

				for( const std::pair< size_t, IOType > &pair1 : P ) {
					if(pair1.first == pair.first) a = pair1.second;
				}

				Vector< IOType > nvt( grb::ncols( Aw ));
				Vector< IOType > nvt( grb::ncols( Aw ));
				grb::set(nvt,0);
				grb::set(nv,0);
				rc = grb::vxm(nvt, M, grb::operators::add< IOType, IOType, IOType >(), v, Aw, standard_sr);

				for(const std::pair< size_t, double > &pairn : nvt ) {
					const double val = pairn.second;
					if (val == 0 ) {
						grb::setElement(nv, 0, pairn.first);
					} else {
						grb::setElement(nv, 1, pairn.first);
					}
				}

				Vector< pType > Ia( grb::ncols(Aw) );
				grb::set( Ia, 0 );
				rc = grb::eWiseApply< grb::descriptors::dense > (Ia, P, a, grb::operators::equal< pType, pType, pType >() );

				Vector< pType > internal( grb::ncols( Aw ));
				grb::set(internal, 0);
				grb::Vector<pType> nv_helper(grb::size(nv));

				grb::set(nv_helper, nv);

				rc = grb::eWiseApply( internal, nv_helper, Ia, nv, grb::operators::equal< pType, pType, pType >());

				bool is_internal = true;

				for( const std::pair< size_t, pType > &pair1 : internal ) {
					if ( pair.second == 0 ) {
						is_internal = false;
					}
				}
				if (!is_internal) continue;


				Vector< pType > neighboring_parts( grb::ncols( Aw ));
				grb::set( neighboring_parts, 0 );

				rc = grb::eWiseApply(neighboring_parts, M, nv, P, grb::operators::mul< pType, pType, pType >());
				std::set< pType, std::greater< pType >> Nv;
				for( const std::pair< size_t, int > &pairnp : neighboring_parts ) {
					if( pairnp.second != 0 ) Nv.insert(pairnp.second);
				}

				int minb; 
				int mincost = 100000;

				for ( auto b : Nv ) {
					if (sizes[b] > maxsize ) continue;

					Vector< pType > relcost( grb::nrows( Aw ));

					grb::set(Ic, 0);
					rc = grb::eWiseApply(Ic, M, P, c, grb::operators::equal<pType, pType, pType>());

					Vector<int> spansc(grb::nrows(Aw));

					grb::set(spans,0);

					rc = grb::vxm<grb::descriptors::transpose_matrix >( spansc, v, grb::operators::add<IOType, IOType, IOType> (), Ic, Aw, ao_sr);

					rc = foldl(relcost, spansc, grb::operators::add<int,int,int>());
					grb::setElement(P,a,pair.first);
				}

				int cost = 0;

				if (cost < mincost ) {
					mincost = cost;
					minb = b;
				}

				if( minb == 0 ) continue;
				grb::setElement( P, minb, pair.first );
				sizes[a]--;
				sizes[minb]++;
			}
			return SUCCESS;
		}

		template< typename IOType>
		RC uncoarsen_weight_matrix(
			Matrix< IOType > &Aw, 
			Matrix< int > &Ts,
			const grb::Semiring<
				     grb::operators::add< IOType >,
					grb::operators::mul< IOType >,
					grb::identities::zero, 
					grb::identities::one
			    > &standard_sr
		) {
			std::vector<int> Ivec, Jvec;
			std::vector< IOType > Vvec;

			size_t n = grb::ncols( Ts );
			size_t m = grb::nrows( Ts );
			for( int i = 0; i < n; ++i ) {
				Vector<int> vi(m);
				grb::setElement( vi, -1, i );
				Vector< int > Ti(n);
				grb::mxv(Ti, Ts, vi, standard_sr);

				if(i < m) grb::setElement(Ti,1,i);
				Vector<IOType> tmp(grb::nrows( Aw ));
				RC rc = grb::mxv( tmp, Aw, Ti, standard_sr);

				for(const std::pair<size_t, IOType> &pair1: tmp) {
					double val1 = pair1.second;
					if(val1 != 0) {
						int i1 = pair1.first;
						Ivec.push_back(i1);
						Jvec.push_back(i);
						Vvec.push_back(val1);
					}
				}
			}

			int* I = &Ivec[0];
			int* J = &Jvec[0];
			double* V = &Vvec[0];
			grb::resize( Aw, Vvec.size());
			grb::buildMatrixUnique( Aw, &(I[0]), &(J[0]), &(V[0]), Vvec.size(), SEQUENTIAL);

			return SUCCESS;

		}


		template< typename IOType, typename pType> 
		RC uncoarsening( Matrix< IOType > &Aw, Vector< pType > &M, Vector< pType > &P, std::vector< Matrix< pType >*> &T,
		std::vector< IOType > &sizes,
		const grb::Semiring<
				     grb::operators::add< IOType >,
					grb::operators::mul< IOType >,
					grb::identities::zero, 
					grb::identities::one
			    > &standard_sr) {
			Vector< pType > M_temp(grb::size( M ));
			Vector< pType > P_temp(grb::size( P ));
			RC rc = SUCCESS;

			for( int s = 0; s < T.size(); ++s ) {
				rc = grb::mxv( M_temp, *(T[s]), M, standard_sr);
				rc = grb::mxv( P_temp, *(T[s]), P, standard_sr);
				grb::set(M, M_temp);
				grb::set(P, P_temp);
				uncoarsen_weight_matrix(Aw, *(T[s]));
			}
			uncoarsening_step(Aw, M, P, sizes);

			return SUCCESS;
		}


		template< typename IOType, typename pType>
		RC partition(
			Matrix< pType > &A, 
			pType &k,
			IOType &c,
			const grb::Semiring<
				     grb::operators::add< IOType >,
					grb::operators::mul< IOType >,
					grb::identities::zero, 
					grb::identities::one
			    > &standard_sr
		) {
			Vector< pType > M( grb::ncols( A ));
			grb::set(M, 1);

			Vector< pType > P( grb::ncols( A ));
			grb::set( P, 0 );

			std::vector< Matrix< int >* > T;
			int n = grb::ncols( A );
			int m = grb::nrows( A );

			Vector< int > ones( n );
			grb::set( ones, 1 );
			
			Vector< pType > w(m);
			grb::mxv( w, A, ones, standard_sr );

			Matrix< IOType > Aw( m, n );

			grb::mxm(Aw, w, A, standard_sr);

			std::vector< IOType > sizes( k );

			coarsening( Aw, M , T , k);
			initial_partition( M, P, k, sizes );
			uncoarsening( Aw, M , P , T, sizes );


			for( const std::pair< size_t, IOType > &pair : P ) {
				std::cout << "P[" << pair.first << "] = " << pair.second << std::endl;
			}

			return SUCCESS;


		}


	}
}
