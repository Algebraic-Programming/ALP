#ifndef GRASSMANN_PLAP_H
#define GRASSMANN_PLAP_H

#include <graphblas.hpp>
#include <graphblas/utils/Timer.hpp>

#include "Manifolds/Grassmann.h"
#include "Manifolds/Stiefel.h"
#include "Problems/Problem.h"
#include "Solvers/Solvers.h"
#include "Others/def.h"

#include <cmath>

#undef Vector

namespace ROPTLIB
{

    class Grass_pLap : public ROPTLIB::Problem
    {

        using ROPTLIB::Problem::NumGradHess;

    private:
        const grb::Matrix<double> &W;
        grb::Vector<double> ones;
        const size_t n, k;
        const double p;
        mutable std::vector<grb::Vector<double> *> Columns, Etax, Res;

        const grb::Semiring<
            grb::operators::add<double>,
            grb::operators::mul<double>,
            grb::identities::zero,
            grb::identities::one>
            reals_ring;

        const double Hess_approx_thresh = 1e-160;

        // performance measurement

        mutable grb::utils::Timer timer;
        mutable double io_time = 0, grb_time = 0;

        // function that converts a ROPTLIB nxk matrix to k graphblas vectors of length n
        void ROPTLIBtoGRB(
            const ROPTLIB::Variable &x,
            std::vector<grb::Vector<double> *> &grb_x) const
        {
            grb::RC rc = grb::SUCCESS;

// #ifdef _H_GRB_REFERENCE_OMP_BLAS3
// #pragma omp parallel for schedule(static, config::CACHE_LINE_SIZE::value())
// #endif
            const double *xPtr = x.ObtainReadData();

            // does this paralllel for make sense together with distributed memory backends?
// #ifdef _H_GRB_REFERENCE_OMP_BLAS3
// #pragma omp parallel for schedule(static, config::CACHE_LINE_SIZE::value())
// #endif
            for (size_t i = 0; i < k; ++i)
            {
                rc = rc ? rc : grb::buildVector(*(grb_x[i]), xPtr + i * n, xPtr + (i + 1) * n, SEQUENTIAL);
            }
        }

        // function that writes a set of k graphblas vectors of length n to a ROPTLIB nxk matrix
        void GRBtoROPTLIB(
            const std::vector<grb::Vector<double> *> &grb_x,
            ROPTLIB::Element *result) const
        {
            double *resPtr = result->ObtainWriteEntireData();

// #ifdef _H_GRB_REFERENCE_OMP_BLAS3
// #pragma omp parallel for schedule(static, config::CACHE_LINE_SIZE::value())
// #endif
            for (size_t i = 0; i < k; ++i)
            {
                // once we have random-access Vector iterators can parallelise this
                for (const auto &pair : *(grb_x[i]))
                {
                    resPtr[i * n + pair.first] = pair.second;
                }
            }
        }

        double summandEvalNum(const size_t l) const
        {
            grb::Matrix<double> Wuu(n, n);
            grb::resize(Wuu, grb::nnz(W));
            grb::Vector<double> vec(n);
            double s = 0;

            grb::set(Wuu, W);
            grb::eWiseLambda([&Wuu, &l, this](const size_t i, const size_t j, double &v)
                             { v = v * std::pow(std::fabs((*(this->Columns[l]))[i] - (*(this->Columns[l]))[j]), this->p); },
                             Wuu);

            grb::vxm(vec, ones, Wuu, reals_ring);
            grb::dot(s, vec, ones, reals_ring);

            return s;
        }

        double pPowSum(const size_t l) const
        {
            grb::Vector<double> vec(n);
            double s = 0;

            //working with orthonormal columns
            if (p == 2)
                return 1.0;

            grb::set(vec, *Columns[l]);
            grb::eWiseMap([this](const double u)
                          { return std::pow(std::fabs(u), this->p); },
                          vec);
            grb::foldl(s, vec, reals_ring.getAdditiveMonoid());

            return s;
        }

        double phi_p(const double &u) const
        {
            return u > 0 ? std::pow(u, p - 1) : -std::pow(-u, p - 1);
        }

    public:
        Grass_pLap(const grb::Matrix<double> &inW, size_t in_n, size_t in_k, double p_in) : W(inW),
                                                                                            ones(in_n),
                                                                                            n(in_n),
                                                                                            k(in_k),
                                                                                            p(p_in)
        {
            NumGradHess = false;

            grb::set(ones, 1);

            Columns.resize(k);
            Etax.resize(k);
            Res.resize(k);

// #ifdef _H_GRB_REFERENCE_OMP_BLAS3
// #pragma omp parallel for schedule(static, config::CACHE_LINE_SIZE::value())
// #endif
            for (size_t i = 0; i < k; ++i)
            {
                Columns[i] = new grb::Vector<double>(n);
                Etax[i] = new grb::Vector<double>(n);
                Res[i] = new grb::Vector<double>(n);
            }
        }

        virtual ~Grass_pLap() {}
        // Objective function, p-norm
        virtual double f(const ROPTLIB::Variable &x) const
        {
            // convert to k Graphblas vectors
            timer.reset();
            ROPTLIBtoGRB(x, Columns);

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

            // ============================================== //
            // Evaluating the objective function in graphblas //
            // ============================================== //

            timer.reset();
            double result = 0;
            for (size_t l = 0; l < k; ++l)
            {
                result += summandEvalNum(l) / (2 * pPowSum(l));
                // Print result. This is the
                // function evaluation. It is a double
                //std::cout << "num = " << summandEvalNum( l ) << " den = " << 2*pPowSum( l ) << "\n";
            }
            grb_time += timer.time();

            //std::cout << " F(U)= "<< result << "\n";
            return result;
        }

        virtual ROPTLIB::Element &EucGrad(
            const ROPTLIB::Variable &x,
            ROPTLIB::Element *result) const
        {
            // convert to k Graphblas vectors
            timer.reset();
            ROPTLIBtoGRB(x, Columns);

            for (size_t l = 0; l < k; ++l)
            {
                for (const auto &pair : *Columns[l])
                {
                    if (isnan(pair.second))
                    {
                        std::cerr << "in eucgrad";
                    }
                    assert(!isnan(pair.second));
                    //std::cout << pair.second << " ";
                }
            }
            io_time += timer.time();

            // ============================================== //
            // Evaluating the euclidean gradient in graphblas //
            // ============================================== //

            timer.reset();
            for (size_t l = 0; l < k; ++l)
            {

                grb::Matrix<double> Wphiu(n, n);
                grb::resize(Wphiu, grb::nnz(W));
                grb::Vector<double> vec(n);

                // Print the entries of the input matrix W    
                // for ( size_t i=0; i<n; ++i) {
                //     grb::Vector<double> v1(n), v2(n);
                //     grb::setElement( v1, 1, i );
                //     grb::set( v2, 0 );
                //     grb::vxm( v2, v1, W, reals_ring );
                //     for (const auto &elt : v2) {
                //         if (elt.second != 0) std::cout << i << " " << elt.first << "\n"; 
                //     }
                // }
                // std::cout << "==============================" << std::endl;
                // End Print


                grb::set(Wphiu, W);
                grb::eWiseLambda([&Wphiu, &l, this](const size_t i, const size_t j, double &v)
                                 { v = v * phi_p((*(this->Columns[l]))[j] - (*(this->Columns[l]))[i]); 
                                 },
                                 Wphiu);

                // Print the entries of the resulting matrix Wphiu
                // for ( size_t i=0; i<n; ++i) {
                //     grb::Vector<double> v1(n), v2(n);
                //     grb::setElement( v1, 1, i );
                //     grb::set( v2, 0 );
                //     grb::vxm( v2, v1, Wphiu, reals_ring );
                //     for (const auto &elt : v2) {
                //         if (elt.second != 0) std::cout << i << " " << elt.first << "\n"; 
                //     }
                // }
                // std::cin.get();
                // End Print

                grb::set(vec, 0);
                grb::vxm(vec, ones, Wphiu, reals_ring);

                double powsum = pPowSum(l);
                double factor = summandEvalNum(l) / (2 * powsum);

                grb::set(*(Res[l]), 0);
                // std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
                grb::eWiseLambda([&vec, &powsum, &factor, &l, this](const size_t i)
                                 {
                                     // == Version defining the grb::Element. ==
                                    //   grb::setElement(*(this->Res[l]), ((this->p) / powsum) *
                                    //       (vec[i] - factor * phi_p((*(this->Columns[l]))[i])), i);

                                     //  Print out all components of the gradient
                                     // std::cout << (this->p)/powsum << "||" << vec[i] << "||" << factor << "||" << phi_p((*(this->Columns[l]))[i]) << std::endl;

                                     // == Version without defining the grb::Element. It should be identical. ==
                                     (*(this->Res[l]))[i] =
                                         ((this->p) / powsum) *
                                         (vec[i] - factor * phi_p((*(this->Columns[l]))[i]));
                                 },
                                 vec);
                // std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
                // This is a printout of the euclidean gradient vector
                // std::cout << "factor " << factor << std::endl;
                // for (const std::pair<size_t, double> &vec_elem : vec)
                // {
                //     if (vec_elem.first < 20)
                //     {
                //         std::cout << phi_p((*(this->Columns[l]))[vec_elem.first]) << std::endl;
                //     }
                // }
                // std::cin.get();
            }
            grb_time += timer.time();

            // write data back to ROPTLIB format
            timer.reset();
            GRBtoROPTLIB(Res, result);

            //  ++++ This is a print out of the full nxk matrix for the gradient in ROPTLIB +++
            // std::cout << "-------------" << std::endl;
            // std::cout << "grb Res" << std::endl;
            // std::cout << "-------------" << std::endl;
            // for(int i = 0; i < k; ++i ) {
            //     for(const auto elem : *(Res[i])) {
            //         std::cout << elem.second << " ";
            //     }
            //     std::cout << std::endl;
            // }
            // std::cout << "-------------" << std::endl;
            // std::cin.get();

            //  ++++ This is a print out of the full nxk matrix for the gradient in ROPTLIB +++
            // std::cout << "-------------" << std::endl;
            // std::cout << "The final solution" << std::endl;
            // std::cout << "-------------" << std::endl;
            // std::cout <<   *result     << std::endl;
            // std::cout << "-------------" << std::endl;
            // std::cin.get();

            io_time += timer.time();

            for (size_t l = 0; l < k; ++l)
            {
                for (const auto &pair : *Res[l])
                {
                    // Print pair.second here (double)
                    // these are the components of the gradient
                    if (isnan(pair.second))
                    {
                        std::cerr << "in eucgrad";
                    }
                    assert(!isnan(pair.second));
                }
            }

            return *result;
        }

        // ============================================== //
        // Evaluating the Hessian  in graphblas           //
        // ============================================== //

        virtual ROPTLIB::Element &EucHessianEta(
            const ROPTLIB::Variable &x,
            const ROPTLIB::Element &etax,
            ROPTLIB::Element *result) const
        {
            // convert to k Graphblas vectors
            timer.reset();
            ROPTLIBtoGRB(x, Columns);
            for (size_t l = 0; l < k; ++l)
            {
                for (const auto &pair : *Columns[l])
                {
                    if (isnan(pair.second))
                    {
                        std::cerr << "Nan in  Hessian * eta" << std::endl;
                    }
                    assert(!isnan(pair.second));
                    //std::cout << pair.second << " ";
                }
            }
            ROPTLIBtoGRB(etax, Etax);
            io_time += timer.time();

            // evaluate hessian*vector in graphblas
            timer.reset();

            grb::Matrix<double> Wuu(n, n);
            grb::resize(Wuu, grb::nnz(W));
            grb::Vector<double> vec1(n), vec2(n);

            for (size_t l = 0; l < k; ++l)
            {
                grb::set(Wuu, W);

                // If the 1st eigenvector is constant, we have a division by zero when l = 0
                // 
                //  { v = std::pow(1e-16, p - 2); },

                // OUT: hardcode the values of the component for l = 0
                // if (l == 0)
                // {
                //     grb::eWiseLambda([this](const size_t i){
                //         (*(this->Res[0]))[i] = (*(this->Etax[0]))[i] * 1e+2;
                //     },
                //     *(Res[0]), *(Etax[0]) );
                // }
                // else
                // {
                    grb::eWiseLambda([&Wuu, &l, this](const size_t i, const size_t j, double &v)
                                     {
                                        //  if (verbose_level = 5){
                                        //  std::cout << "|x|^(p-2), with |x|:  " << std::fabs((*(this->Columns[l]))[i] - (*(this->Columns[l]))[j]) << std::endl;
                                        // }
                                         v = v * std::pow(std::max(Hess_approx_thresh,std::fabs(                                                                                                                 
                                                         (*(this->Columns[l]))[i] - (*(this->Columns[l]))[j]) 
                                                         ),
                                                     p - 2);

                                         //  std::cout << "i: " << i << " || j: " << j << " || w_ij: "<< v << std::endl;
                                         //  std::cout << "u_li - u_lj: " <<  (*(this->Columns[l]))[i] - (*(this->Columns[l]))[j] <<
                                         //  "|| max(t): " << std::fabs(

                                         //                      (*(this->Columns[l]))[i] - (*(this->Columns[l]))[j])  << "|| p-pow(t): " <<
                                         //                      std::pow(
                                         //              std::fabs(

                                         //                      (*(this->Columns[l]))[i] - (*(this->Columns[l]))[j]),
                                         //              p - 2)
                                         //                       << std::endl;
                                     },
                                     Wuu);
                

                grb::set(vec1, 0);
                grb::set(vec2, 0);
                grb::vxm(vec1, ones, Wuu, reals_ring);
                grb::vxm(vec2, *(this->Etax[l]), Wuu, reals_ring);
                double powsum = pPowSum(l);

                grb::set(*(Res[l]), 0);
                grb::eWiseLambda([&vec1, &vec2, &powsum, &l, this](const size_t i)
                                 {
                                     (*(this->Res[l]))[i] =
                                         ((this->p) * (this->p - 1) / powsum) *
                                         (vec1[i] * (*(this->Etax[l]))[i] - vec2[i]);

                                     //  std::cout << "-------------" << std::endl;
                                     //  std::cout <<  vec1[i] << " || " << std::endl;

                                     if (isnan((*(this->Res[l]))[i]))
                                     {
                                         std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
                                         std::cout << " Nan in the Hessian computation. Printing components " << std::endl;
                                         std::cout << "Vector No is: " << l << std::endl;
                                         std::cout <<  "p(p-1)/||u||^p_p = "  << powsum << std::endl;                                         
                                         std::cout << "vec1 * eta = " <<  vec1[i] * (*(this->Etax[l]))[i] << std::endl;
                                         std::cout << "vec2 = " << vec2[i] << std::endl;
                                         std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
                                        //  std::cin.get();
                                     }
                                 },
                                 vec1);
                // }
                // std::cout << "p: " << p << " || "
                //           << "p*(p-1)/denom: " << ((this->p) * (this->p - 1) / powsum) << " ||" << std::endl;
                // std::cin.get();
                
            }
            grb_time += timer.time();

            timer.reset();
            GRBtoROPTLIB(Res, result);

            // //  ++++ This is a print out of the full nxk matrix for the gradient in ROPTLIB +++
            // std::cout << "-------------" << std::endl;
            // std::cout << "The Hessian is" << std::endl;
            // std::cout << "-------------" << std::endl;
            // std::cout << *result << std::endl;
            // std::cout << "-------------" << std::endl;
            // std::cin.get();

            io_time += timer.time();

            for (size_t l = 0; l < k; ++l)
            {
                for (const auto &pair : *Res[l])
                {
                    if (isnan(pair.second))
                    {
                        std::cerr << "Nan in  Hessian * eta" << std::endl;
                    }
                    assert(!isnan(pair.second));
                }
            }

            return *result;
        }

        double getIOtime()
        {
            return io_time;
        }

        double getGRBtime()
        {
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

}; // end namespace ROPTLIB

#endif