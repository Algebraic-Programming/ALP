/*
* Verner Vlacic, Huawei Zurich Research Center, 25 Feb 2021
*/

#ifndef _H_GRB_SPECPART_UTILS
#define _H_GRB_SPECPART_UTILS

#include <graphblas.hpp>

using namespace grb;
namespace grb::algorithms::spec_part_utils{};

namespace grb {

	//returns the p norm of a vector

	template <typename IOType, typename pType>

	IOType p_norm(
		Vector<IOType> x,
		const pType &p,
		const grb::Monoid<
			grb::operators::add< IOType >,
			grb::identities::zero
		> &mono_add
	) { //Can we somehow use the Vector buffer instead of copying?
		eWiseMap([&p](const IOType u){
			return std::pow(std::abs(u),p);
		}, x );

		IOType ret = static_cast<IOType>(0);
		grb::foldl(ret,x,mono_add);

		return std::pow(ret,1/p);
	}

	template <typename IOType, typename pType>
	IOType p_norm_to_p(
		Vector<IOType> x,
		const pType &p,
		const grb::Monoid<
				grb::operators::add< IOType >,
				grb::identities::zero
			> &mono_add
		) { //Can we somehow use the Vector buffer instead of copying?
		eWiseMap([&p](const IOType u){
			return std::pow(std::abs(u),p);
		}, x );

		IOType ret = static_cast<IOType>(0);
		grb::foldl(ret,x,mono_add);

		return ret;
	}

	namespace algorithms{

		namespace spec_part_utils{

			//applies the laplacian to a vector in the form y = Dx - Ax, where A is the adjacency matrix
			template <typename IOType>
			RC compute_Laplacian(Matrix<IOType> &L, const Matrix<IOType> &A) {
				int n = grb::nrows(A);
				int counts[n];
				std::vector<double> vals;
				std::vector<int> I, J;
				for(int i = 0; i < n; ++i) {
					counts[i] = 0;
				}
				for(const std::pair< std::pair< size_t, size_t>, IOType > & pair : A) {
					int i = pair.first.first;
					counts[i]++;
				}
				for(const std::pair< std::pair< size_t, size_t>, IOType > & pair : A) {
					int i = pair.first.first;
					int j = pair.first.second;
					if(i==j) {
						vals.push_back(counts[i]);
						I.push_back(i);
						J.push_back(j);
					} else {
						vals.push_back(-1);
						I.push_back(i);
						J.push_back(j);

						vals.push_back(-1);
						I.push_back(j);
						J.push_back(i);
					}

				}
				// for(int i = 0; i < n; ++i) {
				//	 vals.push_back(counts[i]);
				// }
				int* I2 = &I[0];
				int* J2 = &J[0];
				double* V2 = &vals[0];
				grb::resize( L, vals.size() );
				grb::buildMatrixUnique( L, &(I2[0]), &(J2[0]), &(V2[0]), vals.size(), SEQUENTIAL );

				return SUCCESS;
			}

			template <typename IOType>
			RC apply_Laplacian(
				Vector<IOType> &Lx,
				const Vector<IOType> &x,
				const Vector<IOType> &diag,
				const Matrix<void> &A,
				const size_t n,
				const Semiring<
					grb::operators::add< IOType >,
					grb::operators::right_assign< bool, IOType, IOType >,
					grb::identities::zero,
					grb::identities::logical_true
				> &pattern_ring ) {

				Vector<IOType> diag_x(n), Ax(n);
				grb::set(diag_x,0);
				grb::set(Ax,0);
				grb::eWiseApply( diag_x, diag, x, operators::mul<IOType>() ); //CHANGE TO apply ONCE ALBERT-JAN RENAMES eWiseApply
				grb::mxv( Ax, A, x, pattern_ring );

				grb::eWiseApply( Lx, diag_x, Ax, operators::subtract<IOType>() );

				return SUCCESS;
			}

			//generalized "rounding function", can be used to obtain a 0-1 partition according to the signs of vector, or to implement the elementwise sign function
			template <
				typename IOType,
				typename IntegerT, //used for entries in the incidence matrix, doing integer manipulations etc. so proably long int
				typename IntegerT1
			>
			RC general_rounding( Vector<IntegerT> &par, const Vector<IOType> &x, IntegerT1 &&up_val, IntegerT1 &&down_val ){
				return eWiseLambda([ &x, &par, &up_val, &down_val ]( const size_t i ){
					if ( x[i] >= 0 ){
						grb::setElement( par, static_cast<IntegerT>(up_val), i );
					}
					else{
						grb::setElement( par, static_cast<IntegerT>(down_val), i );
					}
				}, x, par );
			}


			//computes the ratio Cheeger cut
			template < typename IOType, typename IntegerT >
			RC ratio_cheeger_cut(
				IOType &cut,
				const Vector<IntegerT> &par,
				const Matrix<IntegerT> &A,
				const size_t m,
				const size_t n,
				const Semiring< 
				grb::operators::add< IntegerT >,
				grb::operators::mul< IntegerT >,
				grb::identities::zero,
				grb::identities::one
			> &integers_ring
			) {

				//compute edges=Ax
				Vector<IntegerT> edges(m);

				grb::mxv( edges, A, par, integers_ring );

				cut = static_cast<IOType>(p_norm(edges, (bool) 1, integers_ring.getAdditiveMonoid())) / std::min(
					p_norm(par, (bool) 1, integers_ring.getAdditiveMonoid()),
					static_cast<IntegerT>(n - p_norm(par, (bool) 1, integers_ring.getAdditiveMonoid()))
				);

				return SUCCESS;
			}

			template < typename IOType, typename pType >

			//single-element phi_p function

			IOType phi_p_scalar ( IOType &u, const pType &p ){
				return (u>0) ? std::pow(std::fabs(u),p-1) : -std::pow(std::fabs(u),p-1);
			}

			//in-place application of the elementwise phi_p function x -> |x|^{p-1}sgn(x), p should satisfy p>1

			template < typename IOType, typename pType >

			RC phi_p ( Vector<IOType> &x, const pType &p ){
				eWiseMap( [ &x, &p ]( const IOType u ){
					return phi_p_scalar( u, p );
				}, x);
				return SUCCESS;
			}

			template < typename IOType, typename pType >

			//in-place phi_p normalisation
			RC phi_p_normalize(
				Vector<IOType> &x,
				const pType &p,
				const size_t &n, 
				const grb::Monoid<
					grb::operators::add< IOType >,
					grb::identities::zero
				> &mono_add
			) {
				//compute constant to be subtracted
				IOType to_subtract = 0;
				for( const std::pair< size_t, IOType > &pair : x ){
					to_subtract += phi_p_scalar( pair.second, p );
				} 
				to_subtract=to_subtract/n;

				phi_p( x, p );
				// maybe here?!!
				foldl( x, -to_subtract, mono_add );

				phi_p( x, p/(p-1) );

				return SUCCESS;
			}

			RC RCut( double &rcut,
				const Matrix< double > &W,
				const Vector< size_t > &x,
				const size_t k
			) {
				size_t n = grb::ncols( W );

				if ( grb::size( x ) != n ) {
					return MISMATCH;
				}

				Semiring< 
					grb::operators::add< double >,
					grb::operators::right_assign< bool, double, double >,
					grb::identities::zero,
					grb::identities::logical_true
				> pattern_sum;

				// vector of degrees
				grb::Vector< bool > ones( n );
				grb::set( ones, true );
				grb::Vector< double > degs( n );
				grb::vxm( degs, ones, W, pattern_sum );

				// cluster indicators for the computation of the ratio cut
				std::vector< grb::Vector< bool > * > cluster_indic( k );
				for ( size_t i = 0; i < k; ++i ) {
					cluster_indic[ i ] = new grb::Vector< bool >( n );
				}

				// parallelise this once we have random-access iterators
				for ( const auto &pair : x ) {
					grb::setElement( *(cluster_indic[ pair.second ]), true, pair.first );
				}

				rcut = 0;

				grb::Vector< double > aux( n );
				for ( size_t i = 0; i < k; ++i ) {
					double to_add = 0;

					// in case of an empty cluster
					if ( grb::nnz( *(cluster_indic[ i ]) ) == 0) {
						rcut = std::numeric_limits< double >::max();
						return SUCCESS;
					}

					// W part
					grb::clear( aux );
					grb::set(aux,0);
					grb::vxm( aux, *(cluster_indic[ i ]), W, pattern_sum );

					eWiseLambda( [ &cluster_indic, &aux, &i ] ( size_t j ){
						//std::cout << (*(cluster_indic[ i ]))[ j ] << " " << aux[ j ] << "\n";
					}, (*(cluster_indic[ i ])) );

					grb::dot( to_add, *(cluster_indic[ i ]), aux, pattern_sum );
					//to_add = - to_add;

					//std::cout << to_add <<  "\n";

					// degs part
					double to_add_2 = 0;

					grb::dot( to_add_2, *(cluster_indic[ i ]), degs, pattern_sum );

					//std::cout << to_add_2 <<  "\n";

					rcut += (to_add_2 - to_add) /static_cast< double >( grb::nnz( *(cluster_indic[ i ]) ) );
				}

				for ( size_t i = 0; i < k; ++i ) {
					delete cluster_indic[i];
				}

				return SUCCESS;
			}

			/** 
			 * Identify current incidence matrix from hyperincidence
			 */
			template < typename IOType >
			RC HyperToIncid( Matrix< IOType > &A, //resulting incidence matrix
				const Matrix< IOType > &A_hyper, //hypergraph matrix,
				const Vector< IOType > &x //at which vector
			) { 
				// define index-value type
				typedef std::pair< size_t, IOType > indexIOType;

				// // argmin monoid
				Monoid<
					grb::operators::argmin< size_t, IOType >,
					grb::identities::infinity
				> argmin_monoid;

				// // argmax monoid
				Monoid<
					grb::operators::argmax< size_t, IOType >,
					grb::identities::negative_infinity
				> argmax_monoid;

				// check sizes make sense
				size_t m = grb::nrows( A_hyper );
				size_t n = grb::ncols( A_hyper );

				if ( grb::nrows( A ) != m ||
						grb::ncols( A ) != n ||
						grb::size( x ) != n
				) {
					return MISMATCH;
				}

				// running error code
				RC ret = SUCCESS;

				// add labels to x
				Vector< size_t > labels( n );
				ret = ret ? ret : grb::set< grb::descriptors::use_index >( labels, 0 );


				// FIX THIS
				// declare vector of selected vertices
				Vector< indexIOType > max_selected( m ), min_selected( m );
				for ( size_t i = 0; i < m; ++i ) {
					ret = ret ? ret : grb::setElement( max_selected, std::pair< size_t, IOType >( 0, -std::numeric_limits< IOType >::max() ), i );
					ret = ret ? ret : grb::setElement( min_selected, std::pair< size_t, IOType >( 0, std::numeric_limits< IOType >::max() ), i );
				}
				//ret = ret ? ret : grb::set< descriptors::no_casting >( max_selected, std::pair< size_t, IOType >( (size_t)0, std::numeric_limits< IOType >::min() ) );
				//ret = ret ? ret : grb::set< descriptors::no_casting >( min_selected, std::pair< size_t, IOType >( (size_t)0, std::numeric_limits< IOType >::max() ) );

				// compute vector of selected max and min indices
				// first set A_{ij} to x_j
				ret = ret ? ret : grb::set( A, A_hyper );
				ret = ret ? ret : grb::eWiseLambda(
					[ &A, &x ]( const size_t i, const size_t j, IOType &v ) {
						v = x[ j ];
					}, A );
				// compute max_selected
				ret = ret ? ret : vxm< descriptors::transpose_matrix >(
					max_selected, labels, A, argmax_monoid,
					operators::zip< size_t, IOType >()
				);
				// compute min_selected
				ret = ret ? ret : vxm< descriptors::transpose_matrix >(
					min_selected, labels, A, argmin_monoid,
					operators::zip< size_t, IOType >()
				);

				// compute incidence matrix
				ret = ret ? ret : grb::set( A, A_hyper );
				ret = ret ? ret : grb::eWiseLambda(
					[ &A, &max_selected, &min_selected ]( const size_t i, const size_t j, IOType &v ) {
						v *=   ( static_cast< int >( j == (max_selected[i]).first )
							  - static_cast< int >( j == (min_selected[i]).first ) );
					}, A );

				// Semiring< 
				//	 grb::operators::add< double >,
				//	 grb::operators::mul< double >,
				//	 grb::identities::zero,
				//	 grb::identities::one
				// > reals_ring;

				// for ( const auto &pair : max_selected ) {
				//	 std::cout << "i = " << pair.first <<  " max_selected[i] = " << pair.second.first << " " << pair.second.second << std::endl;
				// }

				// for ( const auto &pair : min_selected ) {
				//	 std::cout << "i = " << pair.first <<  " max_selected[i] = " << pair.second.first << " " << pair.second.second << std::endl;
				// }

				// for ( size_t i = 0; i < m; ++i ){
				//	 grb::Vector< double > v( m ), u( n );
				//	 grb::setElement( v, 1.0, i );
				//	 grb::vxm( u, v, A, reals_ring );
				//	 for ( const auto &pair : u ) {
				//		 std::cout << "i = " << i <<  " j = " << pair.first << " v = " << pair.second << std::endl;
				//	 }
				// }
				// std::cin.get();

				return ret;
			}

			RC RCutAdj( double &rcut,
				const Matrix< double > &A_hyper,
				const Vector< size_t > &x,
				std::vector< double >  &cluster_cuts,
				const size_t k
			) {
				size_t n = grb::ncols( A_hyper );
				size_t m = grb::nrows( A_hyper );

				if ( grb::size( x ) != n ) {
					return MISMATCH;
				}

				// incidence matrix derived from A_hyper
				Matrix< double > A( m, n );

				Semiring< 
					grb::operators::add< double >,
					grb::operators::mul< double >,
					grb::identities::zero,
					grb::identities::one
				> reals_ring;

				Semiring< 
					grb::operators::add< double >,
					grb::operators::right_assign< bool, double, double >,
					grb::identities::zero,
					grb::identities::logical_true
				> pattern_sum;

				// cluster indicators for the computation of the ratio cut
				std::vector< grb::Vector< double > * > cluster_indic( k );
				for ( size_t i = 0; i < k; ++i ) {
					cluster_indic[ i ] = new grb::Vector< double >( n );
					grb::set( *(cluster_indic[ i ]), 0 );
				}

				// parallelise this once we have random-access iterators
				std::vector< size_t > sizes( k, 0 );
				for ( const auto &pair : x ) {
					grb::setElement( *(cluster_indic[ pair.second ]), 1.0, pair.first );
					++sizes[ pair.second ];
				}

				rcut = 0;

				grb::Vector< double > aux( m );
				for ( size_t i = 0; i < k; ++i ) {
					double to_add = 0;

					// in case of an empty cluster
					if ( sizes[ i ] == 0) {
						rcut = std::numeric_limits< double >::max();
						return SUCCESS;
					}

					// A part
					HyperToIncid( A, A_hyper, *(cluster_indic[ i ]) );

					grb::clear( aux );
					grb::set( aux, 0 );
					grb::mxv( aux, A, *(cluster_indic[ i ]), reals_ring );

					grb::eWiseMap( [] ( double &u ){
						return std::fabs( u );
					}, aux );

					grb::foldl( to_add, aux, reals_ring.getAdditiveMonoid() );

					rcut += to_add /sizes[ i ];

					cluster_cuts[ i ] = to_add /sizes[ i ];
				}

                for ( size_t i = 0; i < k; ++i ) {
					delete cluster_indic[i];
				}

				return SUCCESS;
			}

			/*
			*   power iteration to obtain top k orthogonal eigenvectors (with largest eigenvalues) 
			*
			*/
			template < typename IOType >
			RC PowerIter( const Matrix< IOType > &A_hyper, //hyper-incidence matrix
				double p,
				std::vector< grb::Vector< IOType >* > &Eigs, //contains the initial guess for the k eigenvectors and the final k eigenvectors, should be nonzero
				std::vector< double > &Cj, // constant for offsetting the pLaplacian
				const IOType conv = 1e-8 // convergence tolerance
			) {
				//nothing to do
				if ( Eigs.empty() ) {
					return SUCCESS;
				}

				// check sizes make sense
				size_t k = Eigs.size();
				size_t n = grb::ncols( A_hyper );
				size_t m = grb::nrows( A_hyper );
				for ( size_t i = 1; i < k; ++i ) {
					if ( grb::size(*Eigs[i]) != n ) {
						return MISMATCH;
					}
				}

				//declare the real numbers
				Semiring<
					grb::operators::add< IOType >,
					grb::operators::mul< IOType >,
					grb::identities::zero,
					grb::identities::one
				> reals_ring;

				// declare the max monoid for residual computation
				Monoid<
					grb::operators::max< IOType >,
					grb::identities::negative_infinity
				> max_monoid;

				// declare pattern sum ring
				Semiring< 
					grb::operators::add< IOType >,
					grb::operators::right_assign_if< IOType, IOType, IOType >,
					grb::identities::zero,
					grb::identities::logical_true
				> pattern_sum;

				// declare derived incidence matrix
				Matrix< IOType > A( m, n ); 

				//define the 2-norm
				const auto eucNorm = [ &n, &reals_ring ]( double &out, grb::Vector< double > &in ) {
					grb::RC rc = SUCCESS;
					out = 0;
					rc = rc ? rc : grb::dot( out, in, in, reals_ring );
					out = std::sqrt( out );
					return rc;
				};
				//define the inf-norm
				const auto infNorm = [ &n, &max_monoid, &p ]( double &out, grb::Vector< double > &in ) {
					grb::RC rc = SUCCESS;
					out = 0;
					grb::Vector< IOType > temp( n );
					rc = rc ? rc : grb::set( temp, in );
					rc = rc ? rc : grb::eWiseMap( [](double u){return std::fabs(u);}, temp );
					rc = rc ? rc : grb::foldl( out, temp, max_monoid );
					return rc;
				};

				// running error code
				RC ret = SUCCESS;

				// working vectors
				grb::Vector< IOType > w( n ), w_old( n ), w_compare( n );
				grb::Vector< IOType > temp( n ), temp2( m );

				// working scalars
				IOType residual, residual_old, norm;


				// number of iterations before increasing C
				size_t iter = 0;
				size_t C_iter = 500;
				double C_factor = 1.05;
				double norm_sols = 1; //minimum supnorm of all solutions, used for relative error 



				// determine the maximum degree of the graph
				// COULD BE COMPUTED OUTSIDE OF PowerIter

				//C = maxdeg * (p-1) * std::pow( conv * norm_sols, p - 2);
				//std::vector< double > Cj( k, C );


				for ( size_t j = 0; j < k; j++ ) {

					// load initial guess into w_old and w_compare and normalise
					ret = ret ? ret : grb::set( w_old, *Eigs[ j ] );
					ret = ret ? ret : infNorm( norm, w_old );
					ret = ret ? ret : grb::foldl( w_old, 1/norm, reals_ring.getMultiplicativeOperator() );
					ret = ret ? ret : grb::set( w_compare, w_old );

					size_t l = 0;
					iter = 0;

					// this choice of C seems to work for unweighted matrices, weighted TBD
					//C = maxdeg * (p-1) * std::pow( conv * norm_sols, p - 2);

					// create history vector
					size_t history_length = 500;
					std::vector< IOType > residual_history( history_length, 1 );
					//double history_oscillation = 0.01;
					IOType history_average = 1, history_average_old = 1;
					double minimum_tolerance = conv;

					do {
						// hit w_old with C*Id - pLaplacian to get w
						ret = ret ? ret : grb::set( temp, w_old );
						ret = ret ? ret : spec_part_utils::phi_p( temp, p/(p-1) );
						ret = ret ? ret : grb::set( temp2, 0 );

						//compute A from A_hyper
						ret = ret ? ret : grb::clear( A );
						ret = ret ? ret : HyperToIncid( A, A_hyper, temp );

						// continue computing C*Id - pLaplacian
						ret = ret ? ret : grb::mxv( temp2, A, temp, reals_ring );
						ret = ret ? ret : spec_part_utils::phi_p( temp2, p );
						ret = ret ? ret : grb::set( w, 0 );
						ret = ret ? ret : grb::vxm( w, temp2, A, reals_ring );

						ret = ret ? ret : eWiseLambda( [ &w, &w_old, &Cj, &j ]( size_t i ) {
							w[ i ] =   Cj[ j ]*w_old [ i ] - w[ i ];
						}, w_old, w );

						// reorthogonalise w.r.t. seen vectors
						for ( size_t l = 0; l < j; ++l ) {
						//if ( j > 0 ) { 
							IOType innerprod;
							ret = ret ? ret : grb::dot< descriptors::dense >( innerprod, w, *Eigs[ l ], reals_ring );
							//innerprod /= ( out_norms[ l ] * out_norms[ l ] );
							ret = ret ? ret : eWiseLambda( [ &w, &Eigs, &innerprod, &l ]( size_t i ){
								w[ i ] -= innerprod * (*Eigs[ l ])[ i ];
							}, w, *Eigs[ l ] );
						}

						// normalise w.r.t. euclidean norm
						ret = ret ? ret : eucNorm( norm, w );
						ret = ret ? ret : grb::foldl( w, 1/norm, reals_ring.getMultiplicativeOperator() );

						ret = ret ? ret : infNorm( norm, w );
						norm_sols = std::min( norm_sols, norm );

						//replace w_old with current w
						ret = ret ? ret : grb::set( w_old, w );

						//they will enter a cycle of length j, update residual with this distance
						//if ( l == 0 ) {
							residual_old = residual;
							residual = 0;
							ret = ret ? ret : grb::dot( residual, w, w_compare, max_monoid, grb::operators::abs_diff< IOType >() );

							// history_average_old = history_average;
							// history_average = history_average + ( residual - residual_history[ iter % history_length ] ) / history_length;
							// residual_history[ iter % history_length ] = residual;

							ret = ret ? ret : grb::set( w_compare, w );

							//std::cout << "p = " << p  << " j = " << j << " residual = " << residual << " history_average = " << history_average << " C[j] = " << Cj[j] << " iter= "<< iter << std::endl;
							//std::cout << "p = " << p  << " j = " << j << " residual = " << residual << " C = " << C << std::endl;
						//}

						// if ( history_average >= history_average_old ) {
						//	 Cj[ j ]*=C_factor;
						// }

						// update l and C ( if converging slowly )
						 //if ( j > 0 ) l = ( l + 1 ) %  j;
						 ++iter;
						 if ( iter > C_iter ) {
							 Cj[ j ]*=C_factor;
							 iter = 0;
						 }

					} while ( ret==SUCCESS && residual > conv * norm_sols );

					std::cout << "final C[ "  <<  j << " ] = " << Cj[ j ] << std::endl;

					// write w to *Eigs[ j ]
					ret = ret ? ret : grb::set( *Eigs[ j ], w );

					// apply "postprocessing" map to *Eigs[ j ] and normalise w.r.t. euclidean norm
					//ret = ret ? ret : spec_part_utils::phi_p( *Eigs[ j ], p/(p-1) );

					ret = ret ? ret : eucNorm( norm, *Eigs[ j ] );
					ret = ret ? ret : grb::foldl( *Eigs[ j ], 1/norm, reals_ring.getMultiplicativeOperator() );

					//ret = ret ? ret : eucNorm( norm, *Eigs[ j ] );
					//std::cout << "norm of evector = " << norm  << std::endl;
					//std::cin.get(); 
				}

				if ( ret != SUCCESS ) {
					std::cout << "\tPowerIter finished with unexpected return code! " << grb::toString( ret ) << std::endl;
				}

				return ret;
			}

		} //end namespace spec_part_utils

	} //end namespace algorithms

} //end namespace grb

#endif

