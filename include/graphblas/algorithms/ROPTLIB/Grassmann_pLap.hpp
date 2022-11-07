#ifndef GRASSMANN_PLAP_H
#define GRASSMANN_PLAP_H

#include <cmath>

#include <graphblas/utils/Timer.hpp>

#include "Manifolds/Grassmann.h"
#include "Manifolds/Stiefel.h"
#include "Others/def.h"
#include "Problems/Problem.h"
#include "Solvers/Solvers.h"

#undef Vector

#include <graphblas.hpp>

#include <chrono>


namespace ROPTLIB {
	class Grass_pLap : public ROPTLIB::Problem {
		using ROPTLIB::Problem::NumGradHess;

	private:
		const grb::Matrix< double > & W;
		grb::Vector< double > ones;
		const size_t n, k;
		const double p;
		mutable std::vector< grb::Vector< double > * > Columns, Etax, Res;

		mutable grb::Matrix< double > Wuu;
		mutable grb::Vector< double > vec, vec2, vec_aux;
		mutable grb::Matrix<double> AbsUiUj;
		mutable std::vector< grb::Matrix< double > * > UiUj;
		mutable bool copied = false;


		mutable std::vector<double> sums, pows, uus;

		const grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > reals_ring;

		const double Hess_approx_thresh = 1e-160;

		// performance measurement
		mutable grb::utils::Timer timer;
		

		// function that converts a ROPTLIB nxk matrix to k graphblas vectors of length n
		void ROPTLIBtoGRB( const ROPTLIB::Variable & x, std::vector< grb::Vector< double > * > & grb_x ) const {

			grb::RC rc = grb::SUCCESS;
			const double * xPtr = x.ObtainReadData();
			//std::cout << "**** " << xPtr << std::endl;

		// does this parallel for make sense together with distributed memory backends?
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
	#pragma omp parallel for schedule( static, config::CACHE_LINE_SIZE::value() )
#endif
			for( size_t i = 0; i < k; ++i ) {
				// std::cout << "**** " << omp_get_num_threads() << std::endl;
				rc = rc ? rc : grb::buildVector( *( grb_x[ i ] ), xPtr + i * n, xPtr + ( i + 1 ) * n, grb::SEQUENTIAL );
				//pows[i] = 0;
				//sums[i] = 0;
				//uus[i] = 0;
				
				//TODO:See if vectors can be wrapped
				//*(grb_x[i]) = grb::internal::template
				//	wrapRawVector< double >( n, xPtr + i * n);
			}
		}

		// function that writes a set of k graphblas vectors of length n to a ROPTLIB nxk matrix
		void GRBtoROPTLIB( const std::vector< grb::Vector< double > * > & grb_x, ROPTLIB::Element * result ) const {
			double * resPtr = result->ObtainWriteEntireData();

#ifdef _H_GRB_REFERENCE_OMP_BLAS3
	#pragma omp parallel for schedule( static, config::CACHE_LINE_SIZE::value() )
#endif
			for( size_t i = 0; i < k; ++i ) {
				// once we have random-access Vector iterators can parallelise this
				for( const auto & pair : *( grb_x[ i ] ) ) {
					resPtr[ i * n + pair.first ] = pair.second;
				}
			}
		}

		// function that evaluates the sum in the numerator ...

		double summandEvalNum( const size_t l ) const {

			//std::cerr << "Summand eval: " << l <<std::endl;
			if(sums[l] != 0){
				//return sums[l];
			}
			double s = 0;
			grb::set( Wuu, W );
			grb::clear( vec );
			//grb::set( vec_aux, 0 );
			grb::eWiseLambda(
				[ &l, this ]( const size_t i, const size_t j, double & v ) {
					v = v * std::pow( std::fabs( ( *( this->Columns[ l ] ) )[ i ] - ( *( this->Columns[ l ] ) )[ j ] ), this->p );
				},
				this->Wuu );


			grb::vxm( vec, ones, Wuu, reals_ring );
			grb::dot( s, vec, ones, reals_ring );
			sums[l] = s;
			return s;
		}

		// function that evaluates Σ||

		double pPowSum( const size_t l ) const {
			//std::cerr << "pPowSum: " << l <<std::endl;

			if(pows[l] != 0){
				//return pows[l];
			}
			// working with orthonormal columns
			//if( p == 2 )
				//return 1.0;

			double s = 0;
			// grb::Vector<double> vec(n);
			//grb::clear(vec_aux);
			grb::set(vec, *Columns[l]);
			grb::eWiseMap(
				[this](const double u) {
					return std::pow( std::fabs( u ), this->p );
				},
				vec );
			grb::foldl( s, vec, reals_ring.getAdditiveMonoid() );

			pows[l] = s;
			return s;
		}

		// function that evaluates \phi

		double phi_p( const double & u, double p ) const {
			return u > 0 ? std::pow( u, p - 1 ) : -std::pow( -u, p - 1 );
		}

		inline double sgn(double v) const{
			return v > 0 ? 1 : v < 0 ? -1 : 0;
		}

		/**Aux functions that was implemented in MATLAB*/
		// get Sparse Derivative Matrix
		void getSparseDerivativeMatrix(grb::Vector<double> &u, size_t l)const {
			//std::cout << "u: " << grb::getID(u) << " p:" << p << " l:"<< l<<std::endl;

			//if(uus[l] != 0) return;

			// UiUj, matrix
			//grb::outer(UiUj, u, u, grb::operators::subtract<double>());
			grb::set(*(UiUj[l]), W);
			//std::cout << "UIUJ: " << grb::nnz(UiUj) <<std::endl;
			//std::cout << "W: " << grb::nnz(W) <<std::endl;
			grb::eWiseLambda(
				[ this, &u ]( const size_t i, const size_t j, double & v ) {
					v = ( u[ i ] - u[ j ] );
				},
				*(this->UiUj[l]) );		
			//uus[l] = 1;
		}

		//abs power
		void computeAbsPower(grb::Matrix<double> &UiUj, double p) const{
			//grb::Matrix<double> AbsUiUj(n,n);
			set(AbsUiUj, UiUj);
			grb::eWiseLambda(
				[ this, p ]( const size_t i, const size_t j, double & v ) {
					v = std::pow( std::fabs( v ), p );
				},
				this->AbsUiUj);	

			//std::cout << "U: " << grb::nnz(UiUj) << " AU:" << grb::nnz(AbsUiUj);	
		}

		// pnormPow()
		double pNormPow(grb::Vector<double> &up, double p) const{
			grb::eWiseMap(
				[ p ]( const double d) {
					return std::pow( std::fabs( d ), p );
				},
				up);
			double s = 0;
			grb::foldl( s, up, reals_ring.getAdditiveMonoid() );

			//pows[l] = s;
			return s;
		}

	public:
		Grass_pLap( const grb::Matrix< double > & inW, size_t in_n, size_t in_k, double p_in ) :
			W( inW ), ones( in_n ), n( in_n ), k( in_k ), p( p_in ), Wuu( in_n, in_n ), AbsUiUj( in_n, in_n ), vec( in_n ), vec2( in_n ), vec_aux( in_n ) {
			NumGradHess = false;

			grb::set( ones, 1 );
			Columns.resize( k );
			Etax.resize( k );
			Res.resize( k );
			UiUj.resize( k );
			pows = std::vector<double>(k, 0);
			sums = std::vector<double>(k, 0);
			uus = std::vector<double>(k, 0);
			grb::resize( Wuu, grb::nnz( W ) );
			grb::resize( AbsUiUj, grb::nnz( W ) );


#ifdef _H_GRB_REFERENCE_OMP_BLAS3
#pragma omp parallel for schedule( static, config::CACHE_LINE_SIZE::value() )
#endif
			for( size_t i = 0; i < k; ++i ) {
				Columns[ i ] = new grb::Vector< double >( n );
				Etax[ i ] = new grb::Vector< double >( n );
				Res[ i ] = new grb::Vector< double >( n );
				UiUj[ i ] = new grb::Matrix< double >( n, n );
				grb::resize( *(UiUj[i]), grb::nnz( W ) );

			}
		}

		virtual ~Grass_pLap() {}

		mutable double io_time = 0, grb_time = 0, obj_time = 0, grad_time = 0, hess_time = 0, hess_set_time = 0, hess_vxm_time = 0,
		hess_EW_time = 0;
		virtual double f( const ROPTLIB::Variable & x ) const {
			//std::cout << "f CALLED\n";

			timer.reset();
			ROPTLIBtoGRB( x, Columns ); // convert to k Graphblas vectors
			io_time += timer.time();

			// ============================================== //
			// Evaluating the objective function in graphblas //
			// ============================================== //

			timer.reset();
			double result = 0;
			for(size_t l = 0; l < k; ++l) {

				getSparseDerivativeMatrix(*( this->Columns[ l ] ), l);
				computeAbsPower(*(this->UiUj[l]), this->p);
				//std::cout << "NNZs, W, Abs, l : " << grb::nnz(W) << " " << grb::nnz(AbsUiUj) << " " << l << std::endl;
				grb::eWiseApply(this->Wuu, this->W, this->AbsUiUj, grb::operators::mul<double>());
				double s = 0;
				clear(vec_aux);
				grb::mxv( vec_aux, Wuu, ones, reals_ring );
				grb::foldl( s, vec_aux, reals_ring.getAdditiveMonoid() );
				double denom = pNormPow(*( this->Columns[ l ] ), this->p) * 2;
				uus[l] = s/ denom;
				result += uus[l];//ummandEvalNum( l ) / ( 2 * pPowSum( l ) );
                //result += summandEvalNum(l) / (2 * pPowSum(l));

			}
			obj_time += timer.time(); // stop time the summandEvalNum
						
			std::cout << "f CALLED " << result << "\n"; 
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

		virtual ROPTLIB::Element & EucGrad( const ROPTLIB::Variable & x, ROPTLIB::Element * result ) const {
			std::cout << "EucGrad CALLED\n";
			timer.reset();
			ROPTLIBtoGRB(x, Columns); // convert to k Graphblas vectors
			io_time += timer.time();
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

			// ============================================== //
			// Evaluating the euclidean gradient in graphblas //
			// ============================================== //
			//std::cout << "Result from previous: "<< this->result << std::endl;
			
			for(size_t l = 0; l < k; ++l) {

				timer.reset();
				grb::set(Wuu, W);
				//std::cout << "NNZs, W, Abs, l : " << grb::nnz(Wuu) << " " << grb::nnz(W) << " " << l << std::endl;

				hess_set_time += timer.time();

				timer.reset();
				//grb::Vector<double> uu(*( this->Columns[ l ] ));
				grb::set(vec_aux, *( this->Columns[ l ] ));

				getSparseDerivativeMatrix( vec_aux, l);

				grb::Matrix<double> U(*(this-> UiUj[l]));
				grb::Matrix<double> BUF(grb::nrows(U), grb::ncols(U), grb::nnz(U));
				computeAbsPower(U, p - 1.0);

				//std::cout << "NNZs, W, Abs, l : " << grb::nnz(W) << " " << grb::nnz(U) << " " << l << std::endl;
                // grb::set(Wuu, W);
                // grb::eWiseLambda([ &l, this](const size_t i, const size_t j, double &v)
                //                  { v = v * phi_p((*(this->Columns[l]))[j] - (*(this->Columns[l]))[i], this ->p); 
                //                  },
                //                  Wuu);
				grb::eWiseLambda(
					[ this ]( const size_t i, const size_t j, double & v ) {
						v = this->sgn(v);
					},
					U);
				grb::eWiseApply(BUF, AbsUiUj, U, grb::operators::mul<double>());
				//grb::set(AbsUiUj, BUF);
				grb::eWiseApply(Wuu, W, BUF, grb::operators::mul<double>());
				//std::cout << "NNZs, W, Abs, l : " << grb::nnz(Wuu) << " " << grb::nnz(BUF) << " " << l << std::endl;
				grb::set(AbsUiUj, Wuu);
				grb::eWiseApply<grb::descriptors::transpose_right>(BUF, Wuu, AbsUiUj, grb::operators::subtract<double>());
				//std::cout << "NNZs, W, Abs, l : " << grb::nnz(Wuu) << " " << grb::nnz(AbsUiUj) << " " << l << std::endl;
				set(Wuu, BUF);
				hess_EW_time += timer.time();

				timer.reset();
				//grb::set(vec, 0);
				//clear(vec);
				//clear(vec_aux);
				hess_set_time += timer.time();
				
				timer.reset();
				//grb::set(vec, 0);
				grb::clear(vec);
                grb::vxm(vec, ones, Wuu, reals_ring);
				hess_vxm_time += timer.time();

				timer.reset();
				//double powsum = pPowSum(l);
				//double factor = summandEvalNum( l ) / ( 2 * powsum );
				//double denom = pNormPow(uu, p);
				obj_time += timer.time(); 

				timer.reset();
				grb::set(*( Res[ l ] ), 0);
				//grb::clear(*( Res[ l ] ));
				hess_set_time += timer.time();

				timer.reset();
                // grb::eWiseLambda([ &powsum, &factor, &l, this](const size_t i)
                //                  {
                //                      // == Version defining the grb::Element. ==
                //                     //   grb::setElement(*(this->Res[l]), ((this->p) / powsum) *
                //                     //       (vec[i] - factor * phi_p((*(this->Columns[l]))[i])), i);

                //                      //  Print out all components of the gradient
                //                      // std::cout << (this->p)/powsum << "||" << vec[i] << "||" << factor << "||" << phi_p((*(this->Columns[l]))[i]) << std::endl;

                //                      // == Version without defining the grb::Element. It should be identical. ==
                //                      (*(this->Res[l]))[i] =
                //                          ((this->p) / powsum) *
                //                          (vec[i] - factor * phi_p((*(this->Columns[l]))[i], this->p));
                //                  },
                //                  vec);

				grb::set(vec2, vec_aux);

				grb::eWiseMap(
					[&l, this](const double d){
						return this->uus[l] * phi_p(d, p);//sgn(std::pow(std::fabs(d), p - 1.0));
					},
					vec_aux
				);
				double fac = p / (pNormPow(vec2, p));

				grb::eWiseApply(vec, vec, vec_aux, grb::operators::subtract<double>());

				grb::eWiseApply(*( this->Res[ l ] ), vec, fac, grb::operators::mul<double>());
				

				// grb::Vector<double> u2(uu);
				// grb::eWiseMap(
				// 	[this, &l](const double d){
				// 		return this->uus[l] * phi_p(d, p);//sgn(std::pow(std::fabs(d), p - 1.0));
				// 	},
				// 	u2
				// );
				//double fac = p / (pNormPow(uu, p));

				//if(uus[l] != factor)
				//	std::cout << " fac: " << factor << ", powsum: " << uus[l] << "sum: " << factor - uus[l] << std::endl;
				//exit(0);


				//timer.reset();

				//obj_time += timer.time(); 
				// //grb::clear(uu);
				// grb::eWiseApply(vec, vec, u2, grb::operators::subtract<double>());
				// grb::eWiseApply(*( this->Res[ l ] ), vec, (this->p / powsum), grb::operators::mul<double>());
				//std::cout << "u2: " << grb::nnz(*( this->Res[ l ] )) << std::endl;

				hess_EW_time += timer.time();	

			}
			grb_time += timer.time();

			// write data back to ROPTLIB format
			timer.reset();
			GRBtoROPTLIB( Res, result );
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
		// Evaluating the Hessian*vec in graphblas        //
		// ============================================== //

		virtual ROPTLIB::Element & EucHessianEta( const ROPTLIB::Variable & x, const ROPTLIB::Element & etax, ROPTLIB::Element * result ) const {
			//std::cout << "Hessian CALLED\n";
			//exit(0);
			timer.reset();
			ROPTLIBtoGRB(x, Columns); // convert to k Graphblas vectors
			ROPTLIBtoGRB(etax, Etax); // convert to k Graphblas vectors
			io_time += timer.time();
					
			for( size_t l = 0; l < k; ++l ) {

				timer.reset();
				grb::set(Wuu, W);
				hess_set_time += timer.time();	

				timer.reset();
				grb::eWiseLambda(
					[ &l, this ]( const size_t i, const size_t j, double & v ) {
						v = v * std::pow( std::max( Hess_approx_thresh, std::fabs( ( *( this->Columns[ l ] ) )[ i ] - ( *( this->Columns[ l ] ) )[ j ] ) ), p - 2 );
					},
					this->Wuu);
				hess_EW_time += timer.time();

				timer.reset();
				grb::set(vec, 0);
				grb::set(vec2, 0);
				hess_set_time += timer.time();	

				timer.reset();
				grb::vxm(vec, ones, Wuu, reals_ring);
				grb::vxm(vec2, *( this->Etax[ l ] ), Wuu, reals_ring);
				double powsum = pPowSum(l);
				hess_vxm_time += timer.time();

				timer.reset();
				grb::set(*( Res[ l ]), 0);
				hess_set_time += timer.time();

				timer.reset();
				grb::eWiseLambda(
					[ &powsum, &l, this ]( const size_t i ) {
						( *( this->Res[ l ] ) )[ i ] = ( ( this->p ) * ( this->p - 1 ) / powsum ) * ( ( this->vec )[ i ] * ( *( this->Etax[ l ] ) )[ i ] - ( this->vec2 )[ i ] );
					},
					this->vec );
				hess_EW_time += timer.time();
			}

			timer.reset();			
			GRBtoROPTLIB(Res, result);
			io_time += timer.time();

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

}; // end namespace ROPTLIB

#endif