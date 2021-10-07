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

            RC get_degree_mat(Matrix<IOType> D, const Matrix<IOType> &A) {
                int n = grb::nrows(A);
                int counts[n];
                std::vector<double> vals;
                std::vector<int> I;
                for(int i = 0; i < n; ++i) {
                    counts[i] = 0;
                }
                for(const std::pair< std::pair< size_t, size_t>, IOType > & pair : A) {
                    int i = pair.first.first;
                    I.push_back(i);
                
                    counts[i]++;
                }
                for(int i = 0; i < n; ++i) {
                    vals.push_back(counts[i]);
                }
                int* I2 = &I[0];
	            double* V2 = &vals[0];
	            grb::resize( D, vals.size() );
	            grb::buildMatrixUnique( D, &(I2[0]), &(I2[0]), &(V2[0]), vals.size(), SEQUENTIAL );

                return SUCCESS;
            }
            RC compute_Laplacian(
                Matrix<IOType> &L,
                const Matrix<IOType> &A,
                const Matrix<IOType> &diag,

            ) {
                grb::eWiseApply( L, diag, A, operators::subtract<IOType>());
                return SUCCESS;
            }
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



        } //end namepace spec_part_utils

    } //end namespace algorithms

} //end namespace grb

#endif