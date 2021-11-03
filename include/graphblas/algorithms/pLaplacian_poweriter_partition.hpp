/*
* Verner Vlacic, Huawei Zurich Research Center, 1 Nov 2021
*/

#ifndef _H_GRB_PLAP_POWERITER
#define _H_GRB_PLAP_POWERITER

#include <graphblas.hpp>
#include <graphblas/algorithms/spec_part_utils.hpp>
#include <graphblas/algorithms/kmeans.hpp>

#include <iostream>
#include <time.h>
#include <chrono>
#include <random>

using namespace grb;
using namespace algorithms;

namespace grb
{
    namespace algorithms
    {

        RC pLaplacian_poweriter(
            Vector< size_t > &x,       //vectors corresponding to the final clusters
            const Matrix< double > &A, // incidence matrix
            const size_t k,                      // number of clusters
            const double final_p = 1.1,          //Final value of p
            const double factor = 0.9,           //Factor for the reduction of p
            const size_t kmeans_ortho_reps = 30, // repetitions of kmeans clustering with orthogonal initialisation
            const size_t kmeans_kpp_reps = 30    // repetitions of kmeans clustering with k++ initialisation
        )
        {

            
            //declare the reals ring
            grb::Semiring<
                        grb::operators::add< double >,
                        grb::operators::mul< double >,
                        grb::identities::zero,
                        grb::identities::one>
                    reals_ring;

            //get number of vertices and edges
            const size_t m = nrows( A );
            const size_t n = ncols( A );

            if (size(x) != n)
            {
                return MISMATCH;
            }

            //running error code
			grb::RC ret = SUCCESS;

            size_t iter = 0; //iteration count
            double p = 2;    //Initialize p value

            // matrix to contain final p-eigenvectors for classification using kmeans
            Matrix< double > X(k, n);
            // matrix to contain the k means as row vectors
            Matrix< double > K(k, k);
            // vector to contain final cluster labels and distances to the cluster centroids
            Vector< std::pair< size_t, double > > clusters_and_distances(n);

            // set up data structure of eigenvectors and initialise it by standard normal random entries
            std::vector< grb::Vector< double >* > Eigs( k );
            std::normal_distribution< double > distribution( 0.0, 1.0 );
            std::default_random_engine generator( 0 );

            for ( size_t l = 0; l < k; ++l ) {
                Eigs[ l ] = new grb::Vector< double >( n );
// #ifdef _H_GRB_REFERENCE_OMP_BLAS3
// #pragma omp parallel for schedule(static, config::CACHE_LINE_SIZE::value())
// #endif
                for ( size_t i = 0; i < n; ++i ) {
                    generator.seed( std::chrono::system_clock::now().time_since_epoch().count() );   
                    ret = ret ? ret : grb::setElement( *Eigs[ l ], distribution( generator ) , i ); 
                }
            }

            // Initialize timers
            grb::utils::Timer timer;
            double grb_time = 0, kmeans_time = 0;

            p = p / factor;
            do
            {
                p = std::max(factor * p, final_p);
                ++iter;

                std::cout << "#######################################" << std::endl;
                std::cout << "#             Solving at p = " << p << "   #" << std::endl;
                std::cout << "#######################################" << std::endl;

                timer.reset();

                // run the power method
                
                std::cout << "Running the power method with p = " << p << std::endl;

                //CURRENTLY C IS AUTOMATIC
                //double C = 1000;
                // convexification number, should be at least the operator norm of the gradient of the laplacian
                double precision = ( p == final_p || p == 2) ? 1e-6 : 1e-6;
                ret = ret ? ret : spec_part_utils::PowerIter( A, p, Eigs, precision );
    
                grb_time += timer.time();

                //  Strategy to reduce the value of p
                // p = 1 + factor * (p - 1);

            } while (p > final_p);

            // place the solution into the rows of a graphblas matrix for kmeans classification

            grb::resize(X, n * k);
            grb::resize(K, k * k);

            size_t *I = new size_t[n * k];
            size_t *J = new size_t[n * k];
            double *V = new double[n * k];

            // #ifdef _H_GRB_REFERENCE_OMP_BLAS3
            // #pragma omp parallel for schedule(static, config::CACHE_LINE_SIZE::value())
            // #endif
            for (size_t i = 0; i < n * k; ++i)
            {
                I[i] = i / n;
                J[i] = i % n;
                V[i] = (*Eigs[ I[i] ])[ J[i] ];
                std::cout << V[i] << ", ";
                //if ( J[i] == 0 ) {
                //    std::cout << std::endl << std::endl;
                //    std::cin.get();
                //}
            }

            grb::buildMatrixUnique( X, I, J, V, n * k, PARALLEL );

            // classify using the graphblas kmeans implementation
            timer.reset();

            double best_rcut = std::numeric_limits<double>::max();
            for (size_t i = 0; i < kmeans_ortho_reps + kmeans_kpp_reps; ++i)
            {

                grb::clear(K);

                if (i < kmeans_ortho_reps)
                {
                    grb::algorithms::korth_initialisation(K, X);
                }
                else
                {
                    grb::algorithms::kpp_initialisation(K, X);
                }

                grb::algorithms::kmeans_iteration(K, clusters_and_distances, X);

                double rcut = 0;
                grb::Vector<size_t> x_temp(n);
                for (const auto &pair : clusters_and_distances)
                {
                    grb::setElement(x_temp, pair.second.first, pair.first);
                }

                // compute the ratio cut
                grb::algorithms::spec_part_utils::RCutAdj(rcut, A, x_temp, k);
                //std::cout << "rcut = " << rcut << std::endl;

                // rcut could be zero in the degenerate case of only one cluster being populated
                if (rcut > 0 && rcut < best_rcut)
                {
                    best_rcut = rcut;
                    grb::set(x, x_temp);
                }
            }
            kmeans_time += timer.time();

            std::vector< size_t > cluster_sizes(k);
            for (const auto &pair : x)
            {
                ++cluster_sizes[pair.second];
            }
            std::cout << "===========" << std::endl;
            std::cout << "Statistics" << std::endl;
            std::cout << "===========" << std::endl;
            std::cout << "Final p_value:" << final_p << std::endl;
            std::cout << "RCut value:" << best_rcut << std::endl;

            for (size_t i = 0; i < k; ++i)
            {
                std::cout << "\t" << cluster_sizes[i] << " nodes in cluster " << i << std::endl;
            }

            std::cout << "grb time (msec) = " << grb_time << std::endl;
            std::cout << "kmeans time (msec) = " << kmeans_time << std::endl;
            std::cout << "total time (msec) = " << grb_time + kmeans_time << std::endl;

            return SUCCESS;

        } //end pLaplacian_poweriter

    } //end namespace algorithms

} //end namespace grb

#endif