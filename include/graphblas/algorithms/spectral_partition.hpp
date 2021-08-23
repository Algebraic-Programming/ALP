/*
* Verner Vlacic, Huawei Zurich Research Center, 25 Feb 2021
*/

#ifndef _H_GRB_PLAP_SPECPART
#define _H_GRB_PLAP_SPECPART

#include <graphblas.hpp>
#include <graphblas/algorithms/spec_part_utils.hpp>

#include <iostream>

using namespace grb;

namespace grb {

    namespace algorithms {

        template <
            Descriptor descr = descriptors::no_operation,
            typename IOType,
            typename IntegerT
        >

        RC Fiedler_vector_incidence(
            Vector<IOType> &x,                  //Vector corresponding to initial and final partition
            const Matrix<IntegerT> &A,          //Incidence matrix
            const IOType conv=0.0000001,        //convergence tolerance for the loop
            const size_t max = 10000,            //maximum allowed number of iterations
            size_t * const iterations = NULL,                       //Optional algo call stats
			double * const quality = NULL
        ){
            
            //RINGS AND MONOIDS

            //declare the real mul/add ring
            grb::Semiring<IOType> reals_ring;

            //declare the integer mul/add ring
			grb::Semiring<IntegerT> integers_ring;

            //declare the oneNorm ring
			grb::Semiring<
				double, double, double, double,
				grb::operators::add,
				grb::operators::abs_diff,
				grb::identities::zero,
				grb::identities::zero
			> oneNormDiff;

            //get number of vertices and edges
            const size_t n = ncols(A);
            const size_t m = nrows(A);

            //initialize partition vector and vector from previous iteration seen
            Vector<IntegerT> par(n);
            Vector<IOType> x_prev(n);

            //control variables
            IOType residual; //accuracy residual
            size_t iter = 0; //iteration count

            //auxiliary variables for the computation of the gradient
            Vector<IOType> aux_1(m), aux_3(n), grad(n); //Ax, A^T Ax, gradient
            IOType aux_4, aux_5; // x^T x, x^T A^T Ax
            set(grad,0); //make grad dense

                //loop: project the estimated Fiedler vector and do gradient descent on the Rayleigh ratio
                do {
                    set( x_prev, x );
                    iter++;

                    IOType to_subtract = 0;
                    foldl( to_subtract, x, reals_ring.getAdditiveMonoid() ); //sum the compnents of x into to_subtract
                    to_subtract=to_subtract/n;
                    foldl( x, -to_subtract, reals_ring.getAdditiveMonoid() ); //subtract to_subtract from every component of x

                    //compute auxiliary variables for the gradient

                    grb::mxv( aux_1, A, x, reals_ring );

                    grb::mxv<grb::descriptors::transpose_matrix>( aux_3, A, aux_1, reals_ring );

                    grb::dot( aux_4, x, x, reals_ring );

                    grb::dot( aux_5, x, aux_3, reals_ring );

                    eWiseLambda( [&x, &grad, &aux_3, &aux_4, &aux_5](const size_t i){
                        grad[i] = 2*(aux_3[i]/aux_4 - (aux_5/(aux_4*aux_4))*x[i]);
                    }, x, grad, aux_3);

                    //LATER DO LINE SEARCH, NOW ONLY GRADIENT DESCENT
                    
                    IOType alpha = 2; //GRADIENT DESCENT PARAMETER

                    eWiseLambda( [&x,&grad,&alpha](const size_t i){
                        x[i] = x[i] - alpha*grad[i];
                    }, x, grad );

                    //print current iteration

                    std::cout << "Current estimate: ";
                    for (const std::pair< size_t, IOType > &pair : x ){
                        std::cout << pair.second << " ";
                    }
                    std::cout << std::endl;

                    //compute residual
                    RC ret = dot<descriptors::dense>( residual, x, x_prev, oneNormDiff );
                    assert( ret == SUCCESS );
                    (void) ret;

                    std::cout << "residual: " << residual << " iteration: " << iter << std::endl; 
                    std::cout << std::endl;

                } while ( residual > conv && iter < max );

                //check if the user requested any stats
                if( iterations != NULL ) {
                    *iterations = iter;
                }
                if( quality != NULL ) {
                    *quality = residual;
                }

                if ( iter >= max ) {
                    std::cout << "Did not converge after " << iter << " iterations." << std::endl;
                    return FAILED; //not converged
				}
                
            return SUCCESS;
        } //end Fiedler_vector_incidence

        template <
            Descriptor descr = descriptors::no_operation,
            typename IOType,
            typename IntegerT = long
        >

        RC Fiedler_vector_laplacian(
            Vector<IOType> &x,                  //Vector corresponding to initial and final partition
            const Matrix<void> &A,          //Adjacency matrix
            const IOType conv = 0.0000001,       //convergence tolerance for the loop
            const size_t max = 10000,            //maximum allowed number of iterations
            size_t * const iterations = NULL,                       //Optional algo call stats
			double * const quality = NULL
        ){
            
            //RINGS AND MONOIDS

            //declare the real mul/add ring
            grb::Semiring<IOType> reals_ring;

            //declare the pattern "ring"
            Semiring<
                IOType, IOType, IOType, IOType,
                operators::add,
                operators::right_assign
            > pattern_ring;

            //declare the integer mul/add ring
			grb::Semiring<IntegerT> integers_ring;

            //declare the oneNorm ring
			grb::Semiring<
				double, double, double, double,
				grb::operators::add,
				grb::operators::abs_diff,
				grb::identities::zero,
				grb::identities::zero
			> oneNormDiff;

            //get dimension of the Laplacian
            const size_t n = ncols(A);

            //initialize partition vector and vector from previous iteration seen
            Vector<IntegerT> par(n);
            Vector<IOType> x_prev(n);

            //control variables
            IOType residual; //accuracy residual
            size_t iter = 0; //iteration count

            //auxiliary variables for the computation of the gradient
            Vector<IOType> diag(n); // diagonal of the laplacian
            Vector<IOType> Lx(n); // Lx
            Vector<IOType> grad(n); // gradient
            Vector<IOType> all_ones(n); // a vector of all ones for the computation of the diagonal
            IOType xx; // x^T x
            IOType xLx; // x^T L x
            set( all_ones, 1 );
            set( grad, 0 ); //make grad dense
            grb::mxv( diag, A, all_ones, integers_ring );

                //loop: project the estimated Fiedler vector and do gradient descent on the Rayleigh ratio
                do {
                    set( x_prev, x );
                    iter++;

                    IOType to_subtract = 0;
                    foldl( to_subtract, x, reals_ring.getAdditiveMonoid() ); //sum the components of x into to_subtract
                    to_subtract = to_subtract/n;
                    foldl( x, -to_subtract, reals_ring.getAdditiveMonoid() ); //subtract to_subtract from every component of x

                    //compute auxiliary variables for the gradient

                    grb::algorithms::spec_part_utils::apply_Laplacian( Lx, x, diag, A, n, pattern_ring );

                    grb::dot( xx, x, x, reals_ring );

                    grb::dot( xLx, x, Lx, reals_ring );


                    eWiseLambda( [&x, &grad, &Lx, &xx, &xLx](const size_t i){
                        grad[i] = 2*(Lx[i]- (xLx/xx)*x[i])/xx;
                    }, x, grad, Lx);

                    //LATER DO LINE SEARCH, NOW ONLY GRADIENT DESCENT
                    
                    IOType alpha = 2; //GRADIENT DESCENT PARAMETER

                    eWiseLambda( [&x,&grad,&alpha](const size_t i){
                        x[i] = x[i] - alpha*grad[i];
                    }, x, grad );

                    //print current iteration

                    std::cout << "Current estimate: ";
                    for (const std::pair< size_t, IOType > &pair : x ){
                        std::cout << pair.second << " ";
                    }
                    std::cout << std::endl;

                    //compute residual
                    RC ret = dot<descriptors::dense>( residual, x, x_prev, oneNormDiff );
                    assert( ret == SUCCESS );
                    (void) ret;

                    std::cout << "residual: " << residual << " iteration: " << iter << std::endl; 
                    std::cout << std::endl;

                } while ( residual > conv && iter < max );

                //check if the user requested any stats
                if( iterations != NULL ) {
                    *iterations = iter;
                }
                if( quality != NULL ) {
                    *quality = residual;
                }

                if ( iter >= max ) {
                    std::cout << "Did not converge after " << iter << " iterations." << std::endl;
                    return FAILED; //not converged
				}
                
            return SUCCESS;
        } //end Fiedler_vector_laplacian
   
    } //end namespace algorithms

} //end namespace grb

#endif