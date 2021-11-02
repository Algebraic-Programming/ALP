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

            template <typename IOType>

            //applies the laplacian to a vector in the form y = Dx - Ax, where A is the adjacency matrix

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
                //     vals.push_back(counts[i]);
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

            template <
                typename IOType,
                typename IntegerT, //used for entries in the incidence matrix, doing integer manipulations etc. so proably long int
                typename IntegerT1
            >

            //generalized "rounding function", can be used to obtain a 0-1 partition according to the signs of vector, or to implement the elementwise sign function

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


            template < typename IOType, typename IntegerT >

            //computes the ratio Cheeger cut
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
            ){
                
                //compute edges=Ax    
                Vector<IntegerT> edges(m);

                grb::mxv( edges, A, par, integers_ring );

                cut = static_cast<IOType>(p_norm(edges, (bool) 1, integers_ring.getAdditiveMonoid()))/std::min(p_norm(par, (bool) 1, integers_ring.getAdditiveMonoid()), static_cast<IntegerT>(n - p_norm(par, (bool) 1, integers_ring.getAdditiveMonoid())));

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

                return SUCCESS;
            }

            RC RCutAdj( double &rcut,
                const Matrix< double > &A,
                const Vector< size_t > &x,
                const size_t k
            ) {
                size_t n = grb::ncols( A );
                size_t m = grb::nrows( A );

                if ( grb::size( x ) != n ) {
                    return MISMATCH;
                }

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
                }

                // parallelise this once we have random-access iterators
                for ( const auto &pair : x ) {
                    grb::setElement( *(cluster_indic[ pair.second ]), 1.0, pair.first );
                }

                rcut = 0;

                grb::Vector< double > aux( m );
                for ( size_t i = 0; i < k; ++i ) {
                    double to_add = 0;

                    // in case of an empty cluster
                    if ( grb::nnz( *(cluster_indic[ i ]) ) == 0) {
                        rcut = std::numeric_limits< double >::max();
                        return SUCCESS;
                    }

                    // W part
                    grb::clear( aux );
                    grb::set( aux, 0 );
                    grb::mxv( aux, A, *(cluster_indic[ i ]), reals_ring );
                    
                    grb::eWiseMap( [] ( double &u ){
                        return std::fabs( u );
                    }, aux );

                    grb::foldl( to_add, aux, reals_ring.getAdditiveMonoid() );

                    rcut += to_add /grb::nnz( *(cluster_indic[ i ]) );
                }

                return SUCCESS;
            }

            /*  
            * power iteration to obtain top k orthogonal eigenvectors (with largest eigenvalues) 
            */
            template < typename IOType >
            RC PowerIter( const Matrix< IOType > &A, //incidence matrix
                          double p,
                          std::vector< grb::Vector< IOType >* > &Eigs,          //contains the initial guess for the k eigenvectors
                                                                                //  and the final k eigenvectors, should be nonzero
                          const IOType conv = 1e-8,        // convergence tolerance
                          double C = 20             // constant for offsetting the pLaplacian
                          )
            {
                //nothing to do
                if ( Eigs.empty() ) {
                    return SUCCESS;
                }

                // check sizes make sense
                size_t k = Eigs.size();
                size_t n = grb::ncols( A );
                size_t m = grb::nrows( A );
                for ( size_t i = 1; i < k; ++i ) {
                    if ( grb::size(*Eigs[i]) != n ) {
                        return MISMATCH;
                    }
                }

                //declare the real numbers
                grb::Semiring<
                        grb::operators::add< IOType >,
                        grb::operators::mul< IOType >,
                        grb::identities::zero,
                        grb::identities::one>
                    reals_ring;

                // declare the max monoid for residual computation
                Monoid<
                        grb::operators::max< IOType >,
                        grb::identities::negative_infinity
                    > max_monoid;

                // declare pattern sum ring
                Semiring< 
                        grb::operators::add< IOType >,
                        grb::operators::right_assign_if< IOType, bool, IOType >,
                        grb::identities::zero,
                        grb::identities::logical_true
                    > pattern_sum;

                //define the 2-norm
                const auto eucNorm = [ &n, &reals_ring ]( double &out, grb::Vector< double > &in ) {
                        grb::RC rc = SUCCESS;
                        rc = rc ? rc : grb::dot( out, in, in, reals_ring );
                        out = std::sqrt( out );
                        return rc;
                };
                //define the inf-norm
                const auto infNorm = [ &n, &max_monoid, &p ]( double &out, grb::Vector< double > &in ) {
                        grb::RC rc = SUCCESS;
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
                //double C_base = C;
                double C_factor = 1.01;
                double norm_sols = 1; //minimum supnorm of all solutions, used for relative error 

                // determine the maximum degree of the graph
                // COULD BE COMPUTED OUTSIDE OF PowerITer
                size_t maxdeg;
                grb::Vector< IOType > ones( m ), degs( n );
                ret = ret ? ret : grb::set( ones, 1 );
                ret = ret ? ret : grb::vxm( degs, ones, A, pattern_sum );
                ret = ret ? ret : grb::foldl( maxdeg, degs, max_monoid );
                //std::cout << " maxdeg = "<< maxdeg << std::endl;

                for ( size_t j = 0; j < k; j++ ) {

                    // load initial guess into w_old and w_compare and normalise
                    ret = ret ? ret : grb::set( w_old, *Eigs[ j ] );
                    ret = ret ? ret : eucNorm( norm, w_old );
                    ret = ret ? ret : grb::foldl( w_old, 1/norm, reals_ring.getMultiplicativeOperator() );
                    ret = ret ? ret : grb::set( w_compare, w_old );

                    size_t l = 0;
                    iter = 0;
                    //C = C_base;
                    C = maxdeg * (p-1) * std::pow( conv * norm_sols, p - 2);
                    do {
                        // hit w_old with C*Id - pLaplacian to get w
                        ret = ret ? ret : grb::set( temp, w_old );
                        ret = ret ? ret : spec_part_utils::phi_p( temp, p/(p-1) );
                        ret = ret ? ret : grb::set( temp2, 0 );
                        ret = ret ? ret : grb::mxv( temp2, A, temp, reals_ring );
                        ret = ret ? ret : spec_part_utils::phi_p( temp2, p );
                        ret = ret ? ret : grb::set( w, 0 );
                        ret = ret ? ret : grb::vxm( w, temp2, A, reals_ring );
                        
                        ret = ret ? ret : eWiseLambda( [ &w, &w_old, &C ]( size_t i ) {
                            w[ i ] = C*w_old [ i ] - w[ i ];
                        }, w_old, w );

                        // reorthogonalise w.r.t. seen vectors
                        //for ( size_t l = 0; l < j; ++l ) {
                        if ( j > 0 ) { 
                            IOType innerprod;
                            ret = ret ? ret : grb::dot< descriptors::dense >( innerprod, w, *Eigs[ l ], reals_ring );
                            ret = ret ? ret : eWiseLambda( [ &w, &Eigs, &innerprod, &l ]( size_t i ){
                                w[ i ] -= innerprod * (*Eigs[ l ])[ i ];
                            }, w, *Eigs[ l ] );
                        }

                        // normalise
                        ret = ret ? ret : eucNorm( norm, w );
                        ret = ret ? ret : grb::foldl( w, 1/norm, reals_ring.getMultiplicativeOperator() );
                        
                        ret = ret ? ret : infNorm( norm, w );
                        norm_sols = std::min( norm_sols, norm );
                        
                        //replace w_old with current w
                        ret = ret ? ret : grb::set( w_old, w );

                        //they will enter a cycle of length j, update residual with this distance
                        if ( l == 0 ) {
                            //residual_old = residual;
                            residual = 0;
                            ret = ret ? ret : grb::dot( residual, w, w_compare, max_monoid, grb::operators::abs_diff< IOType >() );
                            ret = ret ? ret : grb::set( w_compare, w );
                            //std::cout << "p = " << p  << " j = " << j << " residual = " << residual << " C = " << C << std::endl;
                        }

                        // update l and C ( if converging slowly )
                        if ( j > 0 ) l = ( l + 1 ) %  j;
                        ++iter;
                        if ( iter > C_iter ) {
                            C*=C_factor;
                            iter = 0;
                        }
                    
                    } while ( ret==SUCCESS && residual > conv * norm_sols );

                    std::cout << "final C = " << C << std::endl;
                    
                    // write w to *Eigs[ j ]
                    ret = ret ? ret : grb::set( *Eigs[ j ], w );

                    // apply "postprocessing" map to *Eigs[ j ] and normalise
                    //ret = ret ? ret : spec_part_utils::phi_p( *Eigs[ j ], p/(p-1) );
                    ret = ret ? ret : eucNorm( norm, *Eigs[ j ] );
                    ret = ret ? ret : grb::foldl( *Eigs[ j ], 1/norm, reals_ring.getMultiplicativeOperator() );
                    ret = ret ? ret : eucNorm( norm, *Eigs[ j ] );
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