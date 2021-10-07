/*
* Pouya Pourjafar Kolaei, Huawei Zurich Research Center, Oct 2021
*/

// Additions for eigenvalue initial computation
#include <armadillo>

// #ifndef _H_GRB_PLAP_SPECPART
// #define _H_GRB_PLAP_SPECPART

#include <graphblas.hpp>
#include <graphblas/algorithms/spec_part_utils.hpp>
#include <graphblas/algorithms/kmeans.hpp>

#include <iostream>
#include <time.h>
using namespace arma;
using namespace grb;
using namespace algorithms;


namespace grb 
{
    namespace algorithms
    {
        template < typename IOType, typename IntegerT >

        RC spectral_graph_partitioner(
            Vector<IntegerT> &x,            //vectors corresponding to the final clusters
            const Matrix<IOType> &A,      // adjacency matrix
            const size_t k,               // number of clusters
            const size_t kmeans_reps = 30 // repetitions of kmeans clustering                        
            // const arma::Mat<double> &V       
            //const double conv_inner=0.00000001,        //convergence tolerance for the internal loop
            //const size_t cons_outer = 0.000001       //convergence tolerance for the external loop
        ) {
            // build graph Laplacian
            int n = grb::nrows(A);
            Matrix<IOType> L(n,n);
            grb::algorithms::spec_part_utils::compute_Laplacian(L, A);

            // compute Eigendecomposition of Laplacian using Arma
            //      1. convert L to arma matrix

            arma::Mat<IOType> arma_L(n,n);
            for(const std::pair<std::pair<size_t,size_t>,IOType> &p : L) {
                int i = p.first.first;
                int j = p.first.second;
                IOType v = p.second;
                arma_L[i,j] = v;
            }

            // 2. schön für dich, ich hab gehört du fährst mercedes
            arma::Col<IOType> eigen_vals(n);
            arma::Mat<IOType> eigen_vecs(n,n);
            arma::eig_sym(eigen_vals, eigen_vecs, arma_L);

            // Do k-means on them
            //      1. Convert Back to GraphBLAS

            Matrix<IOType> EigenVecs(k,n);
            Matrix<IOType> K(k,k);
            std::vector<int> Ivec, Jvec;
            std::vector<IOType> Vvec;
            for(int i = 0; i < k; ++i) {
                for(int j = 0; j < n; ++j) {
                    Ivec.push_back(i);
                    Jvec.push_back(j);
                    Vvec.push_back(eigen_vecs[i,j]);
                }
            }

            int* I = &Ivec[0];
            int* J = &Jvec[0];
	        IOType* V = &Vvec[0];
	        grb::resize( EigenVecs, Vvec.size() );
	        grb::buildMatrixUnique( EigenVecs, &(I[0]), &(J[0]), &(V[0]), Vvec.size(), SEQUENTIAL );


            // use verners kmeans thing
            Vector<std::pair<size_t, IOType>> clusters_and_distances(n);
            double best_rcut = std::numeric_limits<double>::max();

            for (size_t i = 0; i < kmeans_reps; ++i)
            {

                grb::clear(K);
                grb::algorithms::kpp_initialisation(K, EigenVecs);
                grb::algorithms::kmeans_iteration(K, clusters_and_distances, EigenVecs);
                double rcut = 0;
                grb::Vector<size_t> x_temp(n);
                for (const auto &pair : clusters_and_distances)
                {
                    grb::setElement(x_temp, pair.second.first, pair.first);
                }

                grb::algorithms::spec_part_utils::RCut(rcut, A, x_temp, k);

                // rcut could be zero in the degenerate case of only one cluster being populated
                if (rcut > 0 && rcut < best_rcut)
                {
                    best_rcut = rcut;
                    grb::set(x, x_temp);
                }
            }
            std::vector<size_t> cluster_sizes(k);
            for (const auto &pair : x)
            {
                ++cluster_sizes[pair.second];
            }
            std::cout << "===========" << std::endl;
            std::cout << "Statistics" << std::endl;
            std::cout << "===========" << std::endl;
            std::cout << "RCut value:"    << best_rcut << std::endl;

            for (size_t i = 0; i < k; ++i)
            {
                std::cout << "\t" << cluster_sizes[i] << " nodes in cluster " << i << std::endl;
            }




            return SUCCESS;
        }
    }
}