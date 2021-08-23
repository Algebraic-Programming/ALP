/*
* Verner Vlacic, Huawei Zurich Research Center, 25 Feb 2021
*/

// Additions for eigenvalue initial computation
#include <armadillo>

#ifndef _H_GRB_PLAP_SPECPART
#define _H_GRB_PLAP_SPECPART

#include <graphblas.hpp>
#include <graphblas/algorithms/spec_part_utils.hpp>
#include "Solvers/RNewton.h"
#include "Others/randgen.h"
#include <graphblas/algorithms/ROPTLIB/Grassmann_pLap.hpp>
#include <graphblas/algorithms/kmeans.hpp>
//#include </home/pasadakis/dkm/include/dkm_parallel.hpp>
//#include </home/pasadakis/dkm/include/dkm.hpp>
//#include </home/pasadakis/dkm/include/dkm_utils.hpp>





//blame ROPTLIB for this abomination
#undef Vector

#include <iostream>
#include <time.h>
using namespace arma;
using namespace grb;
using namespace algorithms;

namespace grb
{

    namespace algorithms
    {

        template <
            typename IOType,
            typename IntegerT>

        RC pLaplacian_bisection(
            Vector<IOType> &x,             //Vector corresponding to initial and final partition
            const Matrix<IntegerT> &A,     //Incidence matrix
            const IOType b_max = 2,        //Load balancing parameter
            const IOType beta = 2,         //Final value of p is 1+exp(-beta)
            const IOType conv = 0.0000001, //convergence tolerance for the internal loop
            const size_t max_iter = 1000   //number of iterations for the external loop
        )
        {

            //RINGS AND MONOIDS

            //declare the real mul/add ring
            grb::Semiring<
                grb::operators::add<IOType>,
                grb::operators::mul<IOType>,
                grb::identities::zero,
                grb::identities::one>
                reals_ring;

            //declare the integer mul/add ring
            grb::Semiring<
                grb::operators::add<IntegerT>,
                grb::operators::mul<IntegerT>,
                grb::identities::zero,
                grb::identities::one>
                integers_ring;

            //declare the oneNorm ring
            grb::Semiring<
                grb::operators::add<double>,
                grb::operators::abs_diff<double>,
                grb::identities::zero,
                grb::identities::zero>
                oneNormDiff;

            //get number of vertices and edges
            const size_t n = ncols(A);
            const size_t m = nrows(A);

            //initialize partition vector and best vector seen
            Vector<IntegerT> par(n);
            spec_part_utils::general_rounding(par, x, 1, 0);
            Vector<IOType> x_min(n);
            set(x_min, x);

            //compute initial ratio Cheeger cut of current estimate of eigenvector
            IOType r_cheeg_min, r_cheeg;
            spec_part_utils::ratio_cheeger_cut(r_cheeg_min, par, A, m, n, integers_ring);

            //control variables
            size_t iter = 0;    //iteration count
            IOType residual, p; //accuracy residual, and Laplacian parameter p

            //auxiliary variables for the computation of the gradient
            Vector<IOType> aux_1(m), aux_2(n), aux_3(n), grad(n); //phi_p(Ax), phi_p(x), A^T phi_p(Ax), gradient
            IOType aux_4, aux_5;                                  // x^T phi_p(x), x^T A^T phi_p(Ax)
            set(grad, 0);                                         //to ensure grad is a dense vector
            set(aux_1,0);
            set(aux_2,0);
            set(aux_3,0);
            set(aux_4,0);

            //external loop, evolving p
            do
            {
                //reset residual, set p
                residual = 0;
                p = 1 + std::exp(-beta * iter / max_iter);

                //internal loop, finding the p-eigenvector
                do
                {

                    spec_part_utils::phi_p_normalize(x, p, n, reals_ring.getAdditiveMonoid());

                    spec_part_utils::general_rounding(par, x, 1, 0);

                    spec_part_utils::ratio_cheeger_cut(r_cheeg, par, A, m, n, integers_ring);

                    if (r_cheeg <= r_cheeg_min /*ratio Cheeger cut better than before*/
                        && std::fabs(2 * p_norm(par, (bool)1, integers_ring.getAdditiveMonoid()) / static_cast<IOType>(n) - 1) < b_max)
                    {                          //load balance constraint
                        grb::set(x_min, x);    //save best solution
                        r_cheeg_min = r_cheeg; //save best cheeger const so far
                    }

                    //compute auxiliary variables for the gradient

                    grb::mxv(aux_1, A, x, reals_ring);
                    spec_part_utils::phi_p(aux_1, p);

                    grb::set(aux_2, x);
                    spec_part_utils::phi_p(aux_2, p);

                    grb::mxv<grb::descriptors::transpose_matrix>(aux_3, A, aux_1, reals_ring);

                    grb::dot(aux_4, x, aux_2, reals_ring);

                    grb::dot(aux_5, x, aux_3, reals_ring);

                    eWiseLambda([&grad, &aux_2, &aux_3, &aux_4, &aux_5, &p](const size_t i)
                                { grad[i] = p * (aux_3[i] / aux_4 - (aux_5 / (aux_4 * aux_4)) * aux_2[i]); },
                                grad, aux_2, aux_3);

                    //LATER DO LINE SEARCH, NOW ONLY GRADIENT DESCENT

                    IOType alpha = 0.1; //GRADIENT DESCENT PARAMETER

                    eWiseLambda([&x, &grad, &alpha](const size_t i)
                                { x[i] = x[i] - alpha * grad[i]; },
                                x, grad);

                    //print current iteration
                    std::cout << "value of p: " << p << std::endl;
                    std::cout << "iteration " << iter << " of the external loop" << std::endl;

                    std::cout << "Current x: ";
                    for (const std::pair<size_t, IOType> &pair : x)
                    {
                        std::cout << pair.second << " ";
                    }
                    std::cout << std::endl;
                    std::cout << "rcheeg: " << r_cheeg << std::endl;
                    std::cout << "Current x_min: ";
                    for (const std::pair<size_t, IOType> &pair : x_min)
                    {
                        std::cout << pair.second << " ";
                    }
                    std::cout << std::endl;
                    std::cout << "rcheeg_min: " << r_cheeg_min << std::endl;

                    std::cout << "residual: " << residual << std::endl;
                    std::cout << std::endl;

                    //compute residual
                    dot<descriptors::dense>(residual, x, x_min, oneNormDiff);

                } while (residual > conv);

                //reset x to current best
                set(x, x_min);

                ++iter;

            } while (iter < max_iter);

            return SUCCESS;
        } //end RC pLaplacian_bisection

        RC pLaplacian_multi(
            Vector<size_t> &x,            //vectors corresponding to the final clusters
            const Matrix<double> &W,      // adjacency matrix
            arma::Mat<double> &V,         // eigenvecs matrix
            const size_t k,               // number of clusters
            const double final_p = 2,       //Final value of p
            const double factor = 0.9,    //Factor for the reduction of p
            const size_t kmeans_reps = 30 // repetitions of kmeans clustering                        
            // const arma::Mat<double> &V       
            //const double conv_inner=0.00000001,        //convergence tolerance for the internal loop
            //const size_t cons_outer = 0.000001       //convergence tolerance for the external loop
        )
        {

            //get number of vertices and edges
            const size_t n = nrows(W);

            if (size(x) != n)
            {
                return MISMATCH;
            }

            size_t iter = 0; //iteration count
            double p    = 1.8; //Initialize p value

            // matrix to contain final p-eigenvectors for classification using kmeans
            Matrix<double> X(k, n);
            // matrix to contain the k means as row vectors
            Matrix<double> K(k, k);
            // vector to contain final cluster labels and distances to the cluster centroids
            Vector<std::pair<size_t, double>> clusters_and_distances(n);

            // Define the Grassmann manifold
            ROPTLIB::Grassmann Domain(n, k);
            // ROPTLIB::Grassmann TEMP(n, k);

            double* V_mem = V.memptr(); // get the pointer to memory for the eigenvectors

            // ARMA eigenvecs
            std::cout << "-------------" << std::endl;
            std::cout << "|The ARMA eigenvecs|" << std::endl;   
            V.brief_print("Eigenvectors of the graph Laplacian");
            std::cout << "-------------" << std::endl;
        

            // genrandseed(time(NULL));
            // ROPTLIB::Variable GrassInit = Domain.RandominManifold(); // Random initial guess for the 
            // std::cout << "----------------------" << std::endl;    
            // std::cout << "Random Initial Guess" << std::endl;    
            // std::cout << GrassInit << std::endl;      
            // std::cout << "----------------------" << std::endl;    
            // std::cin.get();

            ROPTLIB::Variable GrassInit(n , k);
            double *temp = GrassInit.ObtainWriteEntireData();
            for (int i = 0; i < n*k; i++)
            {
                temp[i] = V_mem[i]; // Use the input at p = 2 as initial guess
            }
            std::cout << "ARMA Initial Guess" << std::endl;
            std::cout << "-------------" << std::endl;    
            std::cout << GrassInit << std::endl;            
            std::cin.get();
            // std::cin.get();         

            // ROPTLIB variable for the solution vector
            ROPTLIB::Variable Optimizer;
            // ROPTLIB solver used                          
            ROPTLIB::RNewton *RNewtonSolver;

            grb::utils::Timer timer;
            double io_time = 0, grb_time = 0, grbropt_time = 0, kmeans_time = 0;

            do
            {
                ++iter;

                std::cout << "#######################################" << std::endl;            
                std::cout << "#             Solving at p = " << p << "   #" << std::endl;            
                std::cout << "#######################################" << std::endl;            

                timer.reset();

                // Define the p-spectral clustering problem
                ROPTLIB::Grass_pLap Prob(W, n, k, p);

                // Set the domain of the problem to be the Grassmann manifold
                Prob.SetDomain(&Domain);

                // output the parameters of the manifold of domain
                Domain.CheckParams();

                if (iter == 1)
                    RNewtonSolver = new ROPTLIB::RNewton(&Prob, &GrassInit);
                 else
                    RNewtonSolver = new ROPTLIB::RNewton(&Prob, &Optimizer);
                


                // can provide custom line search rule by setting LineSearch_LS to LSSM_INPUTFUN
                // RNewtonSolver->LinesearchInput = &LinesearchInput;
                // RNewtonSolver->IsPureLSInput = false;

                // custom stopping criterion
                //RNewtonSolver->StopPtr = &StoppingCriterion;

                RNewtonSolver->Verbose        = ROPTLIB::ITERRESULT;
                RNewtonSolver->LineSearch_LS  = ROPTLIB::LSSM_ARMIJO;
                RNewtonSolver->OutputGap      = 10;
                RNewtonSolver->Max_Iteration  = 20;
                RNewtonSolver->Minstepsize    = 1e-10;
                RNewtonSolver->Max_Inner_Iter = 100;
                RNewtonSolver->Tolerance      = 1e-6;
                // RNewtonSolver->Stop_Criterion = 1;
                RNewtonSolver->CheckParams();
                RNewtonSolver->Run();
                
                // ROPTLIB variable for the solution vector
                Optimizer = RNewtonSolver->GetXopt();
                // Check the actions of gradient and Hessian
                Prob.CheckGradHessian(Optimizer);

                delete RNewtonSolver;

                io_time      += Prob.getIOtime();
                grb_time     += Prob.getGRBtime();
                grbropt_time += timer.time();

                //const double * OPtr = Optimizer.ObtainReadData();
                //for ( size_t i = 0; i < n*k; ++i ) {
                //    if ( i%n == 0 ) std::cout << std::endl;
                //    std::cout << OPtr[ i ] << " ";
                //}

                p = 1 + factor * (p - 1);
            } while (p > final_p);

            timer.reset();

            // place the optimiser into the rows of a graphblas matrix for kmeans classification
            const double *OptPtr = Optimizer.ObtainReadData();
            // Optimizer = GrassInit;
            std::cout << "-------------" << std::endl; 
            std::cout << "The final solution" << std::endl;
            std::cout << "-------------" << std::endl; 
            std::cout << Optimizer - GrassInit << std::endl;
            std::cout << "-------------" << std::endl;            
            std::cin.get();
            

            /* Print the pointer values
            std::cout << "@@@@@@@@@@@@@@@@@@@@@" << std::endl;
            for (int i = 0; i < 10; i++)
            {
                std::cout << OptPtr[i] << std::endl;
            }
            std::cout << "@@@@@@@@@@@@@@@@@@@@@" << std::endl;
            std::cin.get(); */
            
            /* ---- KMEANS TEST 
            auto data         = dkm::load_csv<float, 4>("/home/pasadakis/dkm/build/iris.data.csv");
            auto cluster_data = dkm::kmeans_lloyd_parallel(data, k);

            std::cout << "Means:" << std::endl;
            for (const auto& mean : std::get<0>(cluster_data)) {
                std::cout << "\t(" << mean[0] << "," << mean[1] << ")" << std::endl;
            }
            std::cout << "\nCluster labels:" << std::endl;
            std::cout << "\tPoint:";
            for (const auto& point : data) {
                std::stringstream value;
                value << "(" << point[0] << "," << point[1] << ")";
                std::cout << std::setw(14) << value.str();
            }
            std::cout << std::endl;
            std::cout << "\tLabel:";
            for (const auto& label : std::get<1>(cluster_data)) {
                std::cout << std::setw(14) << label;
            }
            std::cout << std::endl;
            KMEANS TEST END ---- */


            // classify using the graphblas kmeans implementation
            grb::resize(X, n * k);
            grb::resize(K, k * k);

            size_t *I = new size_t[n * k];
            size_t *J = new size_t[n * k];

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
#pragma omp parallel for schedule(static, config::CACHE_LINE_SIZE::value())
#endif
            for (size_t i = 0; i < n * k; ++i)
            {
                I[i] = i / n;
                J[i] = i % n;
            }

            grb::buildMatrixUnique(X, I, J, OptPtr, n * k, SEQUENTIAL);
            io_time += timer.time();

            timer.reset();

            double best_rcut = std::numeric_limits<double>::max();
            for (size_t i = 0; i < 1; ++i)
            {

                grb::clear(K);
                grb::algorithms::kpp_initialisation(K, X);
                grb::algorithms::kmeans_iteration(K, clusters_and_distances, X);
                //grb::algorithms::localsearchpp(K, clusters_and_distances, X);
                double rcut = 0;
                grb::Vector<size_t> x_temp(n);
                for (const auto &pair : clusters_and_distances)
                {
                    grb::setElement(x_temp, pair.second.first, pair.first);
                }

                grb::algorithms::spec_part_utils::RCut(rcut, W, x_temp, k);

                // rcut could be zero in the degenerate case of only one cluster being populated
                if (rcut > 0 && rcut < best_rcut)
                {
                    best_rcut = rcut;
                    grb::set(x, x_temp);
                }
            }
            kmeans_time += timer.time();

            std::vector<size_t> cluster_sizes(k);
            for (const auto &pair : x)
            {
                ++cluster_sizes[pair.second];
            }
            std::cout << "===========" << std::endl;
            std::cout << "Statistics" << std::endl;
            std::cout << "===========" << std::endl;
            std::cout << "Final p_value:" << final_p << std::endl;
            std::cout << "RCut value:"    << best_rcut << std::endl;

            for (size_t i = 0; i < k; ++i)
            {
                std::cout << "\t" << cluster_sizes[i] << " nodes in cluster " << i << std::endl;
            }


            std::cout << "io time (msec) = " << io_time << std::endl;
            std::cout << "grb time (msec) = " << grb_time << std::endl;
            std::cout << "grb+roptlib time (msec) = " << grbropt_time << std::endl;
            std::cout << "kmeans time (msec) = " << kmeans_time << std::endl;
            std::cout << "total time (msec) = " << grbropt_time + kmeans_time << std::endl;

            return SUCCESS;
        } //end RC pLaplacian_bisection

    } //end namespace algorithms

} //end namespace grb

#endif
