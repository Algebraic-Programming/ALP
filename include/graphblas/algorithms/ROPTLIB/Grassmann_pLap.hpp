#ifndef GRASSMANN_PLAP_H
#define GRASSMANN_PLAP_H

#include <graphblas.hpp>
#include <graphblas/utils/Timer.hpp>

#include "Manifolds/Grassmann.h"
#include "Problems/Problem.h"
#include "Solvers/Solvers.h"
#include "Others/def.h"

#include <cmath>

#undef Vector

namespace ROPTLIB {

    class Grass_pLap : public ROPTLIB::Problem {

        using ROPTLIB::Problem::NumGradHess;

        private:
            const grb::Matrix< double > &W;
            grb::Vector< double > ones;
            const size_t n, k;
            const double p;
            mutable std::vector< grb::Vector< double > * > Columns, Etax, Res;

            const grb::Semiring< 
				grb::operators::add< double >,
				grb::operators::mul< double >,
				grb::identities::zero,
				grb::identities::one
			>  reals_ring;

            const double Hess_approx_thresh = 1e-10;


            // performance measurement

            mutable grb::utils::Timer timer;
            mutable double io_time = 0, grb_time = 0;


            // function that converts a ROPTLIB nxk matrix to k graphblas vectors of length n
            void ROPTLIBtoGRB(
                const ROPTLIB::Variable &x,
                std::vector< grb::Vector< double > * > &grb_x
            ) const {
                grb::RC rc = grb::SUCCESS;

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp parallel for schedule(static,config::CACHE_LINE_SIZE::value())
#endif
                const double *xPtr = x.ObtainReadData();

                // does this paralllel for make sense together with distributed memory backends?
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp parallel for schedule(static,config::CACHE_LINE_SIZE::value())
#endif
                for ( size_t i = 0; i < k; ++i ) {
                    rc = rc ? rc : grb::buildVector( *(grb_x[ i ]), xPtr + i*n, xPtr + ( i + 1 )*n, SEQUENTIAL );
                }
            }

            // function that writes a set of k graphblas vectors of length n to a ROPTLIB nxk matrix
            void GRBtoROPTLIB(
                const std::vector< grb::Vector< double > * > &grb_x,
                ROPTLIB::Element * result
            ) const {
                double *resPtr = result->ObtainWriteEntireData();

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp parallel for schedule(static,config::CACHE_LINE_SIZE::value())
#endif
                for ( size_t i = 0; i < k; ++i ) {
                    // once we have random-access Vector iterators can parallelise this 
                    for ( const auto &pair : *(grb_x[ i ]) ) {
                        resPtr[ i*n + pair.first ] = pair.second;
                    }
                }
            }

            double summandEvalNum( const size_t l ) const {
                grb::Matrix< double > Wuu( n, n );
                grb::resize( Wuu, grb::nnz( W ) );
                grb::Vector< double > vec( n );
                double s = 0;

                grb::set( Wuu, W );
                grb::eWiseLambda( [&Wuu, &l, this ]( const size_t i, const size_t j, double &v ) {
                    v = v * std::pow( std::fabs( (*(this->Columns[ l ]))[ i ] - (*(this->Columns[ l ]))[ j ] ), this->p );
                }, Wuu );
                
                grb::vxm( vec, ones, Wuu, reals_ring );
                grb::dot( s, vec, ones, reals_ring ); 

                return s;
            }

            double pPowSum( const size_t l ) const {
                grb::Vector< double > vec( n );
                double s = 0;

                //working with orthonormal columns
                if ( p == 2 ) return 1.0;

                grb::set( vec, *Columns[ l ] );
                grb::eWiseMap( [ this ] ( const double u ) {
                    return std::pow( std::fabs( u ), this->p );
                }, vec );
                grb::foldl( s, vec, reals_ring.getAdditiveMonoid() );

                return s;
            }

            double phi_p( const double &u ) const {
                return u > 0 ? std::pow( u, p - 1 ) : -std::pow( -u, p - 1 );
            }

        public:
            Grass_pLap( const grb::Matrix< double > &inW, size_t in_n, size_t in_k, double p_in ) :
                W(inW),
                ones( in_n ),
                n(in_n),
                k(in_k),
                p(p_in)
            {
                NumGradHess = false;
                
                grb::set( ones, 1 );

                Columns.resize( k );
                Etax.resize( k );
                Res.resize( k );

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
			#pragma omp parallel for schedule(static,config::CACHE_LINE_SIZE::value())
#endif
                for ( size_t i = 0; i < k; ++i ) {
                    Columns[ i ] = new grb::Vector< double >( n );
                    Etax[ i ] = new grb::Vector< double >( n );
                    Res[ i ] = new grb::Vector< double >( n );
                }
            }

            virtual ~Grass_pLap() {}

            virtual double f( const ROPTLIB::Variable &x ) const {
                // convert to k Graphblas vectors
                timer.reset();
                ROPTLIBtoGRB( x, Columns );

                /*
                const double * OPtr = x.ObtainReadData();
                for ( size_t i = 0; i < n*k; ++i ) {
                    if ( i%n == 0 ) std::cout << std::endl;
                    std::cout << OPtr[ i ] << " ";
                }
                for ( size_t l = 0; l < k; ++l ) {
                    for ( const auto & pair : *Columns[ l ] ){
                        if ( isnan(pair.second) ) {
                            std::cerr << "in f";
                        }
                        assert( !isnan(pair.second) );
                        //std::cout << pair.second << " ";
                    }
                } 
                std::cout << "\n";
                */
                io_time += timer.time();

                // evaluate objective function in graphblas

                timer.reset();
                double result = 0;
                for ( size_t l = 0; l < k; ++l ) {
                    result += summandEvalNum( l ) / ( 2*pPowSum( l ) );
                    //std::cout << "num = " << summandEvalNum( l ) << " den = " << 2*pPowSum( l ) << "\n";
                }
                grb_time += timer.time();

                //std::cout << " F(U)= "<< result << "\n";
                return result;
            }

            virtual ROPTLIB::Element& EucGrad(
                const ROPTLIB::Variable &x,
                ROPTLIB::Element * result
            ) const {
                // convert to k Graphblas vectors
                timer.reset();
                ROPTLIBtoGRB( x, Columns );

                for ( size_t l = 0; l < k; ++l ) {
                    for ( const auto & pair : *Columns[ l ] ){
                        if ( isnan(pair.second) ) {
                            std::cerr << "in eucgrad";
                        }
                        assert( !isnan(pair.second) );
                        //std::cout << pair.second << " ";
                    }
                } 
                io_time += timer.time();

                // evaluate euclidean gradient in graphblas
                timer.reset();
                for ( size_t l = 0; l < k; ++l ) { 

                    grb::Matrix< double > Wphiu( n, n );
                    grb::resize( Wphiu, grb::nnz( W ) );
                    grb::Vector< double > vec( n );
                    
                    grb::set( Wphiu, W );
                    grb::eWiseLambda( [&Wphiu, &l, this ]( const size_t i, const size_t j, double &v ) {
                        v = v * phi_p( (*(this->Columns[ l ]))[ i ] - (*(this->Columns[ l ]))[ j ] );
                    }, Wphiu );

                    

                    grb::set( vec, 0 );
                    grb::vxm( vec, ones, Wphiu, reals_ring );
                    double powsum = pPowSum( l );
                    double factor = summandEvalNum( l ) / ( 2*powsum );

                    grb::set( *(Res[ l ]), 0 );
                    grb::eWiseLambda( [ &vec, &powsum, &factor, &l, this ]( const size_t i ) {
                        (*(this->Res[ l ]))[ i ] =
                            -( (this->p) / powsum ) *
                            ( vec[ i ] - factor * phi_p( (*(this->Columns[ l ]))[ i ] ) );
                    }, vec );
                }
                grb_time += timer.time();

                // write data back to ROPTLIB format
                timer.reset();
                GRBtoROPTLIB( Res, result );
                io_time += timer.time();

                for ( size_t l = 0; l < k; ++l ) {  
                    for ( const auto & pair : *Res[ l ] ){
                            if ( isnan(pair.second) ) {
                                std::cerr << "in eucgrad";
                            }
                            assert( !isnan(pair.second) );
                    }
                }

                return *result;
            }

            virtual ROPTLIB::Element& EucHessianEta(
                const ROPTLIB::Variable &x,
                const ROPTLIB::Element &etax,
                ROPTLIB::Element * result
            ) const {
                // convert to k Graphblas vectors
                timer.reset();
                ROPTLIBtoGRB( x, Columns );
                for ( size_t l = 0; l < k; ++l ) {
                    for ( const auto & pair : *Columns[ l ] ){
                        if ( isnan(pair.second) ) {
                            std::cerr << "in euchessianeta";
                        }
                        assert( !isnan(pair.second) );
                        //std::cout << pair.second << " ";
                    }
                }   
                ROPTLIBtoGRB( etax, Etax );
                io_time += timer.time();

                // evaluate hessian*vector in graphblas
                timer.reset();
                for ( size_t l = 0; l < k; ++l ) { 
                    grb::Matrix< double > Wuu( n, n );
                    grb::resize( Wuu, grb::nnz( W ) );
                    grb::Vector< double > vec1( n ), vec2( n );

                    grb::set( Wuu, W );
                    grb::eWiseLambda( [&Wuu, &l, this ]( const size_t i, const size_t j, double &v ) {
                        //std::cout << v << " " << (*(this->Columns[ l ]))[ i ] << " " << (*(this->Columns[ l ]))[ j ] << "\n";
                        v = v *
                            std::pow(
                                std::fabs(
                                    std::max(
                                        Hess_approx_thresh,
                                        (*(this->Columns[ l ]))[ i ] - (*(this->Columns[ l ]))[ j ] 
                                    )
                                ),
                                p - 2
                            );
                        //std::cout << v << "\n";
                    }, Wuu );
                    
                    grb::set( vec1, 0 );
                    grb::vxm( vec1, ones, Wuu, reals_ring );
                    grb::vxm( vec2, *(this->Etax[ l ]), Wuu, reals_ring );

                    //for ( const auto & pair : vec1 ){
                    //    std::cout << pair.second << " "; 
                    //}

                    double powsum = pPowSum( l );

                    grb::set( *(Res[ l ]), 0 );
                    grb::eWiseLambda( [ &vec1, &vec2, &powsum, &l, this ]( const size_t i ) {
                        (*(this->Res[ l ]))[ i ] = 
                            ( (this->p) * ( this->p - 1 )  / powsum ) *
                            ( vec1[ i ] * (*(this->Etax[ l ]))[ i ] - vec2[ i ] );
                        if ( isnan((*(this->Res[ l ]))[ i ]) ){
                            std::cout << powsum << " " << vec1[ i ] << " " << (*(this->Etax[ l ]))[ i ] <<  " " << vec2[ i ] << "\n";
                        }
                    }, vec1 );
                }
                grb_time += timer.time();

                timer.reset();
                GRBtoROPTLIB( Res, result );
                io_time += timer.time();

                for ( size_t l = 0; l < k; ++l ) {  
                    for ( const auto & pair : *Res[ l ] ){
                        if ( isnan(pair.second) ) {
                            std::cerr << "in euchessianeta";
                        }
                        assert( !isnan(pair.second) );
                    }
                }

                return *result;
            }

            double getIOtime() {
                return io_time;
            }
            
            double getGRBtime() {
                return grb_time;
            }
    };

/*
    double LinesearchInput(
        integer iter, const ROPTLIB::Variable &x1, const ROPTLIB::Element &exeta1, double initialstepsize,
        double initialslope, const ROPTLIB::Problem *prob, const ROPTLIB::Solvers *solver
    ) {
        return 1;
    }
*/

/*
    bool StoppingCriterion(
        const ROPTLIB::Variable &x, const ROPTLIB::Element &funSeries, integer lengthSeries,
        double finalval, double initval, const ROPTLIB::Problem *prob, const ROPTLIB::Solvers *solver
    ) {
        return ( finalval / initval < 0.000001 );
    }
*/

} // end namespace ROPTLIB

#endif