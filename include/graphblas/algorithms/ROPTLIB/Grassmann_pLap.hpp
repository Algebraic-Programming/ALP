#ifndef GRASSMANN_PLAP_H
#define GRASSMANN_PLAP_H

#include <graphblas/utils/Timer.hpp>

#include "Manifolds/Grassmann.h"
#include "Manifolds/Stiefel.h"
#include "Problems/Problem.h"
#include "Solvers/Solvers.h"
#include "Others/def.h"

#include <cmath>

#undef Vector

#include <graphblas.hpp>


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
        mutable std::vector<grb::Vector<double> *> Columns, Etax, Res, Prev;
		mutable std::vector< grb::Matrix< double > * > UiUj;

        mutable grb::Matrix<double> Wuu, BUF;
        mutable grb::Vector<double> vec, vec2, vec_aux;

        mutable std::vector<double> sums, pows;
        mutable std::vector<bool> mats;

        mutable bool updated;
            

        const grb::Semiring<
            grb::operators::add<double>,
            grb::operators::mul<double>,
            grb::identities::zero,
            grb::identities::one>
            reals_ring;

        const double Hess_approx_thresh = 1e-160;

        // performance measurement

        mutable grb::utils::Timer timer;
        mutable double io_time = 0, grb_time = 0, ropttgrb = 0, grbtropt = 0, hessT=0, gradT=0, fT=0;

        // function that converts a ROPTLIB nxk matrix to k graphblas vectors of length n
        void ROPTLIBtoGRB(
            const ROPTLIB::Variable &x,
            std::vector<grb::Vector<double> *> &grb_x) const
        {
            timer.reset();
            grb::RC rc = grb::SUCCESS;

            const double *xPtr = x.ObtainReadData();

            // does this parallel for make sense together with distributed memory backends?
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
#pragma omp parallel for schedule(static, config::CACHE_LINE_SIZE::value())
#endif
            for (size_t i = 0; i < k; ++i)
            {
                grb::set(*(grb_x[i]), 0);
                rc = rc ? rc : grb::buildVector(*(grb_x[i]), xPtr + i * n, xPtr + (i + 1) * n, SEQUENTIAL);
                if(rc != grb::SUCCESS || isnan(*xPtr)){
                    std::cout << "Result: " << grb::toString(rc)<<std::endl;
                    std::cin.get();
                }
                //Check if same
                if(!updated && grb_x == Columns){
                    //std::cout << "Columns\n";
                    if(grb::nnz(*(grb_x[i])) != grb::nnz(*(Prev[i]))){
                        clear();
                        updated = true;
                      //  std::cout << "Diff input: "  << std::endl;

                    } else {
                        for (size_t j = 0; j < n; j++)
                        {
                            if((*(grb_x[i]))[j] != (*(Prev[i]))[j]){
                                updated = true;
                                clear();
                                break;
                            }
                            if(isnan((*(grb_x[i]))[j])){
                               std::cout << "num = " << (*(grb_x[i]))[j] << " den = " << j << ", l = " << i << "\n";
                                std::cin.get();
                            }
                        }

                    }
                    grb::set(*(Prev[i]), *(grb_x[i]));
                }

            }
            updated = false;
            ropttgrb += timer.time();
        }

        // function that writes a set of k graphblas vectors of length n to a ROPTLIB nxk matrix
        void GRBtoROPTLIB(
            const std::vector<grb::Vector<double> *> &grb_x,
            ROPTLIB::Element *result) const
        {
            timer.reset();
            double *resPtr = result->ObtainWriteEntireData();

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
#pragma omp parallel for schedule(static, config::CACHE_LINE_SIZE::value())
#endif
            for (size_t i = 0; i < k; ++i)
            {
                // once we have random-access Vector iterators can parallelise this
                for (const auto &pair : *(grb_x[i]))
                {
                    resPtr[i * n + pair.first] = pair.second;
                }
            }
            grbtropt += timer.time();
        }

        void setDerivativeMatrices(const size_t l) const{
            if(mats[l]){
                return;
            }

            grb::Vector<double> u = *(this->Columns[l]);

            grb::set(*(UiUj[l]), W);

			grb::eWiseLambda(
				[ this, &u ]( const size_t i, const size_t j, double & v ) {
					v =  u[ i ] - u[ j ];
				},
				*(this->UiUj[l]) );	

            mats[l] = true;
        }

        double summandEvalNum(const size_t l) const
        {

            if(sums[l] != 0){
               return sums[l];
            }

            double s = 0;


            powMat(l, BUF, this->p);	
            
            grb::set( vec_aux, 0 );

            //grb::eWiseApply(Wuu, Wuu, BUF, reals_ring.getMultiplicativeOperator());

            grb::vxm(vec_aux, ones, BUF, reals_ring);
            grb::dot(s, vec_aux, ones, reals_ring);

            sums[l] = s;
            return s;
        }

        void powMat( const size_t l, grb::Matrix<double> &B, double pow, double threshold = 0.0) const{
            setDerivativeMatrices(l);

            grb::set(B, *(this->UiUj[l]));
            grb::eWiseLambda(
				[ pow, this, threshold]( const size_t i, const size_t j, double & v ) {
                    v = std::pow(std::max(threshold, std::fabs(v)), pow);
				},
				B );	
        }

        double pPowSum(const size_t l) const
        {
          
            //working with orthonormal columns
            if (p == 2.0)
                return 1.0;

            if(pows[l] != 0){
                return pows[l];
            }
            
            double s = 0;
            grb::set( vec_aux, *Columns[l] );
            grb::eWiseMap([this](const double u)
                          { return std::pow(std::fabs(u), this->p); },
                          vec_aux);
            grb::foldl( s, vec_aux, reals_ring.getAdditiveMonoid() );

            pows[l] = s;
            return s;
        }

        double phi_p(const double &u) const
        {
            return u > 0 ? std::pow(u, p - 1) : -std::pow(-u, p - 1);
        }
        void clear() const{
            for (size_t l = 0; l < k; l++)
            {
               sums[l] = 0;
               pows[l] = 0;
               mats[l] = false;
            }

        }

    public:
        Grass_pLap(const grb::Matrix<double> &inW, size_t in_n, size_t in_k, double p_in) : W(inW),
                                                                                            ones(in_n),
                                                                                            n(in_n),
                                                                                            k(in_k),
                                                                                            p(p_in),
                                                                                            Wuu(in_n,in_n),
                                                                                            BUF(in_n,in_n),
                                                                                            vec(in_n),
                                                                                            vec2(in_n),
                                                                                            vec_aux(in_n)
        {
            NumGradHess = false;
            //timer.reset();

            grb::set(ones, 1);

            Columns.resize(k);
            Etax.resize(k);
            Res.resize(k);
            Prev.resize(k);
    		UiUj.resize( k );
            pows = std::vector<double>(k, 0);
            sums = std::vector<double>(k, 0);
            mats = std::vector<bool>(k, false);

            grb::resize(Wuu, grb::nnz(W));
   			grb::resize( BUF, grb::nnz( W ) );


#ifdef _H_GRB_REFERENCE_OMP_BLAS3
#pragma omp parallel for schedule(static, config::CACHE_LINE_SIZE::value())
#endif
            for (size_t i = 0; i < k; ++i)
            {
                Columns[i] = new grb::Vector<double>(n);
                Etax[i] = new grb::Vector<double>(n);
                Res[i] = new grb::Vector<double>(n);
                Prev[i] = new grb::Vector<double>(n);
   				UiUj[ i ] = new grb::Matrix< double >( n, n,  grb::nnz( W ));
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
            //clear();
            for (size_t l = 0; l < k; ++l)
            {
                double s = summandEvalNum(l) / (2 * pPowSum(l));
                if(isnan(s)){
                    std::cout << "num = " << summandEvalNum( l ) << " den = " << 2*pPowSum( l ) << ", l = " << l << "\n";
                }
                result += s;
                //sums[l] = s;
                // Print result. This is the
                // function evaluation. It is a double
                //std::cout << "num = " << summandEvalNum( l ) << " den = " << 2*pPowSum( l ) << "\n";
            }
            grb_time += timer.time();
            fT += timer.time();

            std::cout << " F(U)= "<< result << "\n";
            // for (size_t l = 0; l < k; ++l)
            // {
            //     std::cout << "F" << l << " : ";
            //     int skl =0;
            //     for (const auto &pair : *(this->Columns[l]))
            //     {
            //         if(skl>10)break;
            //         // Print pair.second here (double)
            //         // these are the components of the gradient
            //         if (isnan(pair.second))
            //         {
            //             std::cerr << "in eucgrad";
            //         }
            //         assert(!isnan(pair.second));
  			// 		std::cout << ", " << pair.second;
            //         skl++;
            //     }
            //     std::cout << std::endl;
            // }
            return result;
        }

        virtual ROPTLIB::Element &EucGrad(
            const ROPTLIB::Variable &x,
            ROPTLIB::Element *result) const
        {
  			std::cout << "EucGrad CALLED \n"; 

            // convert to k Graphblas vectors
            timer.reset();
            ROPTLIBtoGRB(x, Columns);

            // for (size_t l = 0; l < k; ++l)
            // {
            //     for (const auto &pair : *Columns[l])
            //     {
            //         if (isnan(pair.second))
            //         {
            //             std::cerr << "in eucgrad";
            //         }
            //         assert(!isnan(pair.second));
            //         std::cout << pair.second << " ";
            //     }
            // }
            io_time += timer.time();

            // ============================================== //
            // Evaluating the euclidean gradient in graphblas //
            // ============================================== //

            timer.reset();
            for (size_t l = 0; l < k; ++l)
            {


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


                double powsum = pPowSum(l);
                double factor = summandEvalNum(l) / (2 * powsum);
                 grb::set(BUF, *(UiUj[l]));
                grb::eWiseLambda(
				[ this, &l ]( const size_t i, const size_t j, double & v ) {
					v = phi_p(v);
				},
				BUF );
                

                //Print the entries of the resulting matrix Wphiu
                // for ( size_t i=0; i<n; ++i) {
                //     grb::Vector<double> v1(n), v2(n);
                //     grb::setElement( v1, 1, i );
                //     grb::set( v2, 0 );
                //     grb::vxm( v2, v1, Wuu, reals_ring );
                //     for (const auto &elt : v2) {
                //         if (elt.second != 0) std::cout << i << " " << elt.first << "\n"; 
                //     }
                // }
                //std::cin.get();
                //End Print

                grb::set(vec, 0);
                grb::vxm<grb::descriptors::transpose_matrix>(vec, ones, BUF, reals_ring);

                grb::set(*(Res[l]), 0);
                // std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
                grb::eWiseLambda([ &powsum, &factor, &l, this](const size_t i)
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
            gradT += timer.time();

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

            // for (size_t l = 0; l < k; ++l)
            // {
            //     std::cout << "L" << l << " : ";
            //     int skl =0;
            //     for (const auto &pair : *(this->Columns[l]))
            //     {
            //         if(skl>10)break;
            //         // Print pair.second here (double)
            //         // these are the components of the gradient
            //         if (isnan(pair.second))
            //         {
            //             std::cerr << "in eucgrad";
            //         }
            //         assert(!isnan(pair.second));
  			// 		std::cout << ", " << pair.second;
            //         skl++;
            //     }
            //     std::cout << std::endl;
            // }

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
            //std::cout << "Hessian CALLED\n";
			
            // convert to k Graphblas vectors
            timer.reset();
            ROPTLIBtoGRB(x, Columns);
            // for (size_t l = 0; l < k; ++l)
            // {
            //     for (const auto &pair : *Columns[l])
            //     {
            //         if (isnan(pair.second))
            //         {
            //             std::cerr << "Nan in  Hessian * eta" << std::endl;
            //         }
            //         assert(!isnan(pair.second));
            //         //std::cout << pair.second << " ";
            //     }
            // }
            ROPTLIBtoGRB(etax, Etax);
            io_time += timer.time();

            // evaluate hessian*vector in graphblas
            timer.reset();

            for (size_t l = 0; l < k; ++l)
            {

                setDerivativeMatrices(l);
                powMat(l, Wuu, this->p-2, Hess_approx_thresh);	

                grb::set(vec, 0);
                grb::set(vec2, 0);
                grb::vxm(vec, ones, Wuu, reals_ring);
                grb::vxm(vec2, *(this->Etax[l]), Wuu, reals_ring);
                double powsum = (this->p) * (this->p - 1) / pPowSum(l);

                grb::set(*(Res[l]), 0);
                grb::eWiseLambda([ &powsum, &l, this](const size_t i)
                                 {
                                     (*(this->Res[l]))[i] =
                                         (powsum) *
                                         (vec[i] * (*(this->Etax[l]))[i] - vec2[i]);

                                     //  std::cout << "-------------" << std::endl;
                                     //  std::cout <<  vec1[i] << " || " << std::endl;

                                    //  if (isnan((*(this->Res[l]))[i]))
                                    //  {
                                    //      std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
                                    //      std::cout << " Nan in the Hessian computation. Printing components " << std::endl;
                                    //      std::cout << "Vector No is: " << i << std::endl;
                                    //      std::cout <<  "p(p-1)/||u||^p_p = "  << powsum << std::endl;                                         
                                    //      std::cout << "vec1 * eta = " <<  vec[i] * (*(this->Etax[l]))[i] << std::endl;
                                    //      std::cout << "vec1 = " <<  vec[i] << ", e =" << (*(this->Etax[l]))[i] << std::endl;
                                    //      std::cout << "vec2 = " << vec2[i] << std::endl;
                                    //      std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
                                    //       std::cin.get();
                                    //  }
                                 },
                                 vec);
                // }
                // std::cout << "p: " << p << " || "
                //           << "p*(p-1)/denom: " << ((this->p) * (this->p - 1) / powsum) << " ||" << std::endl;
                // std::cin.get();
                
            }
            grb_time += timer.time();
            hessT += timer.time();

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

            // for (size_t l = 0; l < k; ++l)
            // {
            //     for (const auto &pair : *Res[l])
            //     {
            //         if (isnan(pair.second))
            //         {
            //             std::cerr << "Nan in  Hessian * eta" << std::endl;
            //         }
            //         assert(!isnan(pair.second));
            //     }
            // }

            return *result;
        }

        double getIOtime()
        {
            return io_time;
        }

        double getGRBtime()
        {
            std::cout << "R2G: " << ropttgrb << ", G2R" << grbtropt << std::endl; 
            std::cout << "f: " << fT << ", grad: " << gradT << ", hess: " << hessT << std::endl; 
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