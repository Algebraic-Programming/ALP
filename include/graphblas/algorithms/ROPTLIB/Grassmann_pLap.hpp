#ifndef GRASSMANN_PLAP_H
#define GRASSMANN_PLAP_H

#include <cmath>

#include <graphblas/utils/Timer.hpp>

#include "Manifolds/Grassmann.h"
#include "Manifolds/Stiefel.h"
#include "Others/def.h"
#include "Problems/Problem.h"
#include "Solvers/Solvers.h"


#include <graphblas.hpp>

#include <chrono>

#undef Vector

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

		const grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > reals_ring;

		const double Hess_approx_thresh = 1e-160;

		// performance measurement
		mutable grb::utils::Timer timer;
		

		// function that converts a ROPTLIB nxk matrix to k graphblas vectors of length n
		void ROPTLIBtoGRB( const ROPTLIB::Variable & x, std::vector< grb::Vector< double > * > & grb_x ) const {
			grb::RC rc = grb::SUCCESS;
			// std::cout << "**** " << omp_get_num_threads() << std::endl;
			const double * xPtr = x.ObtainReadData();

		// does this parallel for make sense together with distributed memory backends?
#ifdef _H_GRB_REFERENCE_OMP_BLAS3
#pragma omp parallel for schedule( static, config::CACHE_LINE_SIZE::value() )
#endif
			for( size_t i = 0; i < k; ++i ) {
				// std::cout << "**** " << omp_get_num_threads() << std::endl;
				rc = rc ? rc : grb::buildVector( *( grb_x[ i ] ), xPtr + i * n, xPtr + ( i + 1 ) * n, grb::SEQUENTIAL );
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

        double summandEvalNum(const size_t l) const
        {
            grb::Matrix<double> Wuu(n, n);
            grb::resize(Wuu, grb::nnz(W));
            grb::Vector<double> vec(n);
            double s = 0;

			grb::set( Wuu, W );
			// grb::clear( vec );
			grb::set( vec_aux, 0 );
			grb::eWiseLambda(
				[ &l, this ]( const size_t i, const size_t j, double & v ) {
					v = v * std::pow( std::fabs( ( *( this->Columns[ l ] ) )[ i ] - ( *( this->Columns[ l ] ) )[ j ] ), this->p );
				},
				this->Wuu );


			grb::vxm( vec_aux, ones, Wuu, reals_ring );
			grb::dot( s, vec_aux, ones, reals_ring );

			return s;
		}

		// function that evaluates Σ||

		double pPowSum( const size_t l ) const {
			// working with orthonormal columns
			if( p == 2 )
				return 1.0;

			double s = 0;
			// grb::Vector<double> vec(n);
			// grb::clear(vec_aux);
			grb::set(vec_aux, *Columns[l]);
			grb::eWiseMap(
				[this](const double u) {
					return std::pow( std::fabs( u ), this->p );
				},
				this->vec_aux );
			grb::foldl( s, vec_aux, reals_ring.getAdditiveMonoid() );

			return s;
		}

		// function that evaluates \phi

		double phi_p( const double & u ) const {
			return u > 0 ? std::pow( u, p - 1 ) : -std::pow( -u, p - 1 );
		}

	public:
		Grass_pLap( const grb::Matrix< double > & inW, size_t in_n, size_t in_k, double p_in ) :
			W( inW ), ones( in_n ), n( in_n ), k( in_k ), p( p_in ), Wuu( in_n, in_n ), vec( in_n ), vec2( in_n ), vec_aux( in_n ) {
			NumGradHess = false;

			grb::set( ones, 1 );
			Columns.resize( k );
			Etax.resize( k );
			Res.resize( k );

			grb::resize( Wuu, grb::nnz( W ) );


#ifdef _H_GRB_REFERENCE_OMP_BLAS3
#pragma omp parallel for schedule( static, config::CACHE_LINE_SIZE::value() )
#endif
			for( size_t i = 0; i < k; ++i ) {
				Columns[ i ] = new grb::Vector< double >( n );
				Etax[ i ] = new grb::Vector< double >( n );
				Res[ i ] = new grb::Vector< double >( n );
			}
		}

		virtual ~Grass_pLap() {}

		mutable double io_time = 0, grb_time = 0, obj_time = 0, grad_time = 0, hess_time = 0, hess_set_time = 0, hess_vxm_time = 0,
		hess_EW_time = 0;
		virtual double f( const ROPTLIB::Variable & x ) const {
			
			timer.reset();
			ROPTLIBtoGRB( x, Columns ); // convert to k Graphblas vectors
			io_time += timer.time();

			// ============================================== //
			// Evaluating the objective function in graphblas //
			// ============================================== //

			timer.reset();
			double result = 0;
			for(size_t l = 0; l < k; ++l) {
				result += summandEvalNum( l ) / ( 2 * pPowSum( l ) );
			}
			obj_time += timer.time(); // stop time the summandEvalNum
			

			return result;
		}

		virtual ROPTLIB::Element & EucGrad( const ROPTLIB::Variable & x, ROPTLIB::Element * result ) const {
			
			timer.reset();
			ROPTLIBtoGRB(x, Columns); // convert to k Graphblas vectors
			io_time += timer.time();

			// ============================================== //
			// Evaluating the euclidean gradient in graphblas //
			// ============================================== //

			
			for(size_t l = 0; l < k; ++l) {

				timer.reset();
				grb::set(Wuu, W);
				hess_set_time += timer.time();

				timer.reset();
				grb::eWiseLambda(
					[ &l, this ]( const size_t i, const size_t j, double & v ) {
						v = v * phi_p( ( *( this->Columns[ l ] ) )[ j ] - ( *( this->Columns[ l ] ) )[ i ] );
					},
					this->Wuu);
				hess_EW_time += timer.time();

				timer.reset();
				grb::set(vec, 0);
				hess_set_time += timer.time();
				
				timer.reset();
				grb::vxm(vec, ones, Wuu, reals_ring);
				hess_vxm_time += timer.time();

				timer.reset();
				double powsum = pPowSum(l);
				double factor = summandEvalNum( l ) / ( 2 * powsum );
				obj_time += timer.time(); 

				timer.reset();
				grb::set(*( Res[ l ] ), 0);
				hess_set_time += timer.time();

				timer.reset();
				grb::eWiseLambda(
					[ &powsum, &factor, &l, this ]( const size_t i ) {
						// == Version without defining the grb::Element.
						( *( this->Res[ l ] ) )[ i ] = ( ( this->p ) / powsum ) * ( this->vec[ i ] - factor * phi_p( ( *( this->Columns[ l ] ) )[ i ] ) );
					},
					this->vec );
				hess_EW_time += timer.time();	
			}

			// write data back to ROPTLIB format
			timer.reset();
			GRBtoROPTLIB( Res, result );
			io_time += timer.time();

			return *result;
		}

		// ============================================== //
		// Evaluating the Hessian*vec in graphblas        //
		// ============================================== //

		virtual ROPTLIB::Element & EucHessianEta( const ROPTLIB::Variable & x, const ROPTLIB::Element & etax, ROPTLIB::Element * result ) const {
			
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