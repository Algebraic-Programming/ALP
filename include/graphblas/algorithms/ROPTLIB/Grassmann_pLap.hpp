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
        mutable std::vector<grb::Vector<double> *> Columns, Etax, Res, Prev, Diag;
		mutable std::vector< grb::Matrix< double > * > UiUj;
		mutable std::vector< grb::Matrix< double > * > Hess;

        mutable grb::Matrix<double> Wuu, BUF;
        mutable grb::Vector<double> vec, vec2, vec_aux;
            
        mutable std::vector<double> facs;
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
        mutable size_t ropt_to_grb_count = 0, grb_to_ropt_count = 0, f_eval_count = 0, grad_eval_count = 0, hessian_eval_count = 0;

        // function that converts a ROPTLIB nxk matrix to k graphblas vectors of length n
        void ROPTLIBtoGRB(
            const ROPTLIB::Variable &x,
            std::vector<grb::Vector<double> *> &grb_x) const
        {
            ropt_to_grb_count++;
            timer.reset();
            grb::RC rc = grb::SUCCESS;

            const double *xPtr = x.ObtainReadData();

            for (size_t i = 0; i < k; ++i)
            {
                grb::set(*(grb_x[i]), 0);
                rc = rc ? rc : grb::buildVector(*(grb_x[i]), xPtr + i * n, xPtr + (i + 1) * n, SEQUENTIAL);
                if(rc != grb::SUCCESS || isnan(*xPtr)){
                    std::cout << "Result: " << grb::toString(rc)<<std::endl;
                    std::cin.get();
                }

                //Check if the vectors are updated
                if(grb_x == Columns){

                    for (size_t j = 0; j < n; j++)
                    {
                        if((*(grb_x[i]))[j] != (*(Prev[i]))[j]){
                            mats[i] = false;
                            break;
                        }
                    }
                    grb::set(*(Prev[i]), *(grb_x[i]));
                }
            }

            ropttgrb += timer.time();
        }

        // function that writes a set of k graphblas vectors of length n to a ROPTLIB nxk matrix
        void GRBtoROPTLIB(
            const std::vector<grb::Vector<double> *> &grb_x,
            ROPTLIB::Element *result) const
        {
            grb_to_ropt_count++;
            timer.reset();
            double *resPtr = result->ObtainWriteEntireData();


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
            //Skip if already calculated
            if(mats[l]) return;

            grb::Vector<double> u = *(this->Columns[l]);

            grb::set(*(UiUj[l]), W);

            //For each input vector u
            // U_ij = u_i - u_j
			grb::eWiseLambda(
				[ this, &u ]( const size_t i, const size_t j, double & v ) {
					v =  u[ i ] - u[ j ];
				},
				*(this->UiUj[l]) );

            mats[l] = true;
        }

        // Calculates w_ij|u_i-u_j|^p
        double summandEvalNum(const size_t l) const
        {
            double s = 0;
            
            // for each elem v do |v|^p
            powMat(l, BUF, this->p, Hess_approx_thresh);	
            
            grb::set( vec_aux, 0 );

            //Multiply wheights
            // v_ij *= w_ij
            grb::eWiseApply(BUF, BUF, W, reals_ring.getMultiplicativeOperator());

            //Sum all rows and cols
            grb::vxm(vec_aux, ones, BUF, reals_ring);

            //THIS PRIMITIVE SINGLE-THREADED!!
            // int thr;
            // #pragma omp parallel
			// {
            //     thr = omp_get_num_threads();
            // }
            // omp_set_num_threads(1);
            //grb::dot(s, vec_aux, ones, reals_ring);
            grb::foldl( s, vec_aux, reals_ring.getAdditiveMonoid() ); 
            // omp_set_num_threads(thr);

            return s;
        }

        //Takes a matrix and for each element v does v = |v|^pow
        //Uses a threshold to avoid negative values
        void powMat( const size_t l, grb::Matrix<double> &B, double pow, double threshold = 0.0) const{
            setDerivativeMatrices(l);

            grb::set(B, *(this->UiUj[l]));
            //Do v = |v|^pow
            grb::eWiseLambda(
				[ pow, this, threshold]( const size_t i, const size_t j, double & v ) {
                    v = std::pow(std::max(threshold, std::fabs(v)), pow);
                },
				B );	
        }

        //For a vector u do norm to the power of p, ||u||^p
        double pPowSum(const size_t l) const
        {

            //working with orthonormal columns
            if (p == 2.0)
                return 1.0;
            
            double s = 0;
            grb::set( vec_aux, *Columns[l] );
            //Do the abs and power
            grb::eWiseLambda([this](const size_t i)
                          { vec_aux[i] = std::pow(std::fabs(vec_aux[i]), this->p); },
                          vec_aux);
            //Sum all values

            //THIS PRIMITIVE SINGLE-THREADED!!
            // int thr;
            // #pragma omp parallel
			// {
            //     thr = omp_get_num_threads();
            // }
            // omp_set_num_threads(1);
            grb::foldl( s, vec_aux, reals_ring.getAdditiveMonoid() ); //CULPRIT!
            // omp_set_num_threads(thr);
            
            return s;
        }

        //Returns the funcion phi() to a value
        double phi_p(const double &u) const
        {
            return u > 0 ? std::pow(u, p - 1) : -std::pow(-u, p - 1);
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
            Diag.resize(k);
    		UiUj.resize( k );
    		Hess.resize( k );
            Diag.resize(k);

            facs = std::vector<double>(k, 1.0);
            mats = std::vector<bool>(k, false);

            grb::resize(Wuu, grb::nnz(W));
   			grb::resize( BUF, grb::nnz( W ) );

            for (size_t i = 0; i < k; ++i)
            {
                Columns[i] = new grb::Vector<double>(n);
                grb::set(*Columns[i], 0);

                Etax[i] = new grb::Vector<double>(n);
                grb::set(*Etax[i], 0);

                Res[i] = new grb::Vector<double>(n);
                grb::set(*Res[i], 0);

                Prev[i] = new grb::Vector<double>(n);
                grb::set(*Prev[i], 0);

                Diag[i] = new grb::Vector<double>(n);
                grb::set(*Diag[i], 0);

   				UiUj[i] = new grb::Matrix< double >( n, n,  grb::nnz( W ));
   				Hess[i] = new grb::Matrix< double >( n, n,  grb::nnz( W ));
            }
        }

        virtual ~Grass_pLap() {
            for (size_t i = 0; i < k; ++i)
            {
                delete Columns[i];
                delete Etax[i];
                delete Res[i];
                delete Prev[i];
                delete Diag[i];
   				delete UiUj[i];
   				delete Hess[i];
            }            
        }

        // Objective function, p-norm
        virtual double f(const ROPTLIB::Variable &x) const
        {   
            

            // convert to k Graphblas vectors
            timer.reset();
            ROPTLIBtoGRB(x, Columns);
            

            io_time += timer.time();

            // ============================================== //
            // Evaluating the objective function in graphblas //
            // ============================================== //

            f_eval_count++;
            timer.reset();
            double result = 0;
            //clear();
            for (size_t l = 0; l < k; ++l)
            {
                double s = summandEvalNum(l) / (2 * pPowSum(l));
                result += s;
                // Print result. This is the
                // function evaluation. It is a double
                //std::cout << "num = " << summandEvalNum( l ) << " den = " << 2*pPowSum( l ) << "\n";
            }
            grb_time += timer.time();
            fT += timer.time();

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


        void calculateHessian(const size_t l) const {

            grb::Matrix<double>* H = (Hess[l]); 
            //Make sure UiUj is updated
            setDerivativeMatrices(l);
    
            powMat(l, *H, p-2, Hess_approx_thresh);
            grb::eWiseApply(*H, *H, W, reals_ring.getMultiplicativeOperator());
            set(Wuu, *H);

            //Save factor to reuse later 
            facs[l] = (this->p) * (this->p - 1) / pPowSum(l);

            grb::set(*(Diag[l]), 0);
            //Calculate the diagonals
            grb::vxm(*(Diag[l]), ones, *H, reals_ring);
        };

        virtual ROPTLIB::Element &EucGrad(
            const ROPTLIB::Variable &x,
            ROPTLIB::Element *result) const
        {
            
            // convert to k Graphblas vectors
            
            timer.reset();
            ROPTLIBtoGRB(x, Columns);

            
        // DEBUG OUTPUT
            // for (size_t l = 0; l < k; ++l)
            // {
            //     for (const auto &pair : *Columns[l])
            //     {
            //         if (isnan(pair.second))
            //         {
            //             std::cerr << "in eucgrad";
            //         }
            //         assert(!isnan(pair.second));
            //         std::cout << pair.second << " ";sudo sshfs -o allow_other,default_permissions,IdentityFile=/home/anderhan/.ssh/id_rsa ahansson@login.huaweirc.ch:/home/ahansson/ /mnt/slurm/ -p 2222
            //     }
            // }
            io_time += timer.time();

            // ============================================== //
            // Evaluating the euclidean gradient in graphblas //
            // ============================================== //

            grad_eval_count++;
            timer.reset();
            for (size_t l = 0; l < k; ++l)
            {

          // DEBUG OUTPUT
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

                

                grb::eWiseApply(BUF, BUF, W, reals_ring.getMultiplicativeOperator());

                
        // DEBUG OUTPUT
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
            // FOR POSSIBLE USE IN FUTURE
            // Scale with constant eigenvector
                //double s;
                //grb::foldl(s, *(Res[l]), reals_ring.getAdditiveMonoid());
                //grb::eWiseAdd(*(Res[l]), *(Res[l]), s/-n, reals_ring);

        // DEBUG OUTPUT
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

                //Update and save Hessian for later 
                

                calculateHessian(l);

            }
            grb_time += timer.time();
            gradT += timer.time();

            timer.reset();

            // write data back to ROPTLIB format
            GRBtoROPTLIB(Res, result);

        // DEBUG OUTPUT
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
        
        // DEBUG OUTPUT
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

        // DEBUG OUTPUT
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
            hessian_eval_count++;
            timer.reset();
            

            for (size_t l = 0; l < k; ++l)
            {

                grb::set(vec, 0);
                grb::set(vec2, 0);
                 
                // Res = (D - H) * etax * factor
                // vec = etax*H
                grb::vxm(vec, *(Etax[l]), *(Hess[l]), reals_ring);
                // vec2 = diag.*etax 
                grb::eWiseApply(vec2, *(Diag[l]), *(Etax[l]), reals_ring.getMultiplicativeOperator());

                // res = (diag.*etax) - (etax*H)
                grb::eWiseApply(*(Res[l]), vec2, vec, grb::operators::subtract<double>());

                // int thr;
                // #pragma omp parallel
                // {
                //     thr = omp_get_num_threads();
                // }
                //omp_set_num_threads(1);
                // res = ((diag.*etax) - (etax*H)) * factor
                grb::foldl(*(Res[l]), facs[l], reals_ring.getMultiplicativeOperator());
                //omp_set_num_threads(thr);

            // FOR POSSIBLE USE IN FUTURE
            // Scale with constant eigenvector
                // double s;
                // grb::foldl(s, *(Res[l]), reals_ring.getAdditiveMonoid());
                //grb::eWiseAdd(*(Res[l]), *(Res[l]), s/-n, reals_ring);


                
            }
            grb_time += timer.time();
            hessT += timer.time();

            timer.reset();

            GRBtoROPTLIB(Res, result);

        // DEBUG OUTPUT
            // //  ++++ This is a print out of the full nxk matrix for the gradient in ROPTLIB +++
            // std::cout << "-------------" << std::endl;
            // std::cout << "The Hessian is" << std::endl;
            // std::cout << "-------------" << std::endl;
            // std::cout << *result << std::endl;
            // std::cout << "-------------" << std::endl;
            // std::cin.get();

            io_time += timer.time();

        // DEBUG OUTPUT
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
            // DEBUG OUTPUT
            //std::cout << "R2G: " << ropttgrb << ", G2R" << grbtropt << std::endl; 
            //std::cout << "f: " << fT << ", grad: " << gradT << ", hess: " << hessT << std::endl; 
            return grb_time;
        }

        double get_ropt_to_grb_count()
        {
            return ropt_to_grb_count;
        }

        double get_grb_to_ropt_count()
        {
            return grb_to_ropt_count;
        }

        double get_f_eval_count()
        {
            return f_eval_count;
        }

        double get_grad_eval_count()
        {
            return grad_eval_count;
        }

        double get_hessian_eval_count()
        {
            return hessian_eval_count;
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
