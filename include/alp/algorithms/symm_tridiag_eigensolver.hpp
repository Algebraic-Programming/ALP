/*
 *   Copyright 2021 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <sstream>

#include <alp.hpp>
#include <graphblas/utils/iscomplex.hpp> // use from grb
#include "../tests/utils/print_alp_containers.hpp"

// TEMPDISABLE should be removed in the final version
#define TEMPDISABLE

namespace alp {

	namespace algorithms {

		/**
		 * find zero of secular equation in interval <a,b>
		 * using bisection
		 * this is not an optimal algorithm and there are many
		 * more efficient implantation
		 */
		template<
			typename D,
			typename VecView1,
			typename VecImfR1,
			typename VecImfC1,
			typename VecView2,
			typename VecImfR2,
			typename VecImfC2,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >
		>
		RC bisec_sec_eq(
			Scalar< D > &lambda,
			const Vector<	D, structures::General,	Dense, VecView1, VecImfR1, VecImfC1 > &d,
			// Vector v should be const, but that would disable eWiseLambda, to be resolved in the future
			Vector<	D, structures::General,	Dense, VecView2, VecImfR2, VecImfC2 > &v,
			const Scalar< D > &a,
			const Scalar< D > &b,
			const D tol=1.e-12,
			const Ring & ring = Ring(),
			const Minus & minus = Minus(),
			const Divide & divide = Divide()
		) {
			RC rc = SUCCESS;

#ifdef DEBUG
			std::cout << " a = " << *a << " ";
			std::cout << " b = " << *b << " ";
#endif

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );
			Scalar< D > x0( ( *a + *b ) / ( 2 ) );

			if( std::abs( *a - *b ) < tol ) {
				alp::set( lambda, x0 );
				return rc;
			}

			//fx0=1+sum(v**2/(d-x0))
			Scalar< D > fx0( one );
			rc = rc ? rc : eWiseLambda(
				[ &d, &x0, &fx0, &ring, &minus, &divide ]( const size_t i, D &val ) {
					Scalar< D > alpha( val );
					Scalar< D > beta( d[ i ] );
					foldl( alpha, Scalar< D > ( val ), ring.getMultiplicativeOperator() );
					foldl( beta, x0, minus );
					foldl( alpha, beta, divide );
					foldl( fx0, alpha, ring.getAdditiveOperator() );
				},
				v
			);

#ifdef DEBUG
			std::cout << " x0 = " << *x0 << " ";
			std::cout << " fx0 = " << *fx0 << "\n";
#endif

			if( std::abs( *fx0 ) < tol ) {
				alp::set( lambda, x0 );
				return rc;
			}

			if( *fx0 < *zero ) {
				rc = rc ? rc : bisec_sec_eq( lambda, d, v, x0, b, tol );
			} else {
				rc = rc ? rc : bisec_sec_eq( lambda, d, v, a, x0, tol );
			}

			return rc;
		}


		/**
		 * Calcualte eigendecomposition of system D + vvt
		 *        \f$D = diag(d)$ is diagonal matrix and
		 *        \a vvt outer product outer(v,v)
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type minus operator used in the computation
		 * @tparam Divide   Type of divide operator used in the computation
		 * @param[out] Egvecs    output orthogonal matrix contaning eigenvectors
		 * @param[out] egvals    output vector containg eigenvalues
		 * @param[in]  ring A semiring for operations
		 * @return RC       SUCCESS if the execution was correct
		 *
		 */
		template<
			typename D,
			typename VecView1,
			typename VecImfR1,
			typename VecImfC1,
			typename VecView2,
			typename VecImfR2,
			typename VecImfC2,
			typename VecView3,
			typename VecImfR3,
			typename VecImfC3,
			typename OrthogonalType,
			typename OrthViewType,
			typename OrthViewImfR,
			typename OrthViewImfC,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >
		>
		RC eigensolveDiagPlusOuter(
			Vector<	D, structures::General,	Dense, VecView1, VecImfR1, VecImfC1 > &egvals,
			Matrix< D, OrthogonalType, Dense, OrthViewType, OrthViewImfR, OrthViewImfC > &Egvecs,
			Vector<	D, structures::General, Dense, VecView2, VecImfR2, VecImfC2 > &d,
			Vector< D, structures::General, Dense, VecView3, VecImfR3, VecImfC3 > &v,
			const Ring & ring = Ring(),
			const Minus & minus = Minus(),
			const Divide & divide = Divide()
		) {
			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );
			const size_t n = nrows( Egvecs );
			const double eps = 1.e-7;

			// all egvec/val are trivial when the corresponding
			// element of v is zero
			size_t count_direct_egvc = 0;
			size_t count_non_direct_egvc = 0;

			std::vector< size_t > direct_egvc_indx( n, 0 );
			std::vector< size_t > non_direct_egvc_indx( n, 0 );

			for( size_t i = 0; i < n; i++ ) {
				if( std::abs( v[ i ] ) < eps ) {
					//simple egval formula ;
					direct_egvc_indx[ count_direct_egvc ] = i ;
					++count_direct_egvc;
				} else {
					//complicated egval formula ;
					non_direct_egvc_indx[ count_non_direct_egvc ] = i;
					++count_non_direct_egvc;
				}
			}
			direct_egvc_indx.resize( count_direct_egvc );
			non_direct_egvc_indx.resize( count_non_direct_egvc );
			alp::Vector< size_t > select_direct( count_direct_egvc );
			alp::Vector< size_t > select_non_direct( count_non_direct_egvc );
			alp::buildVector( select_direct, direct_egvc_indx.begin(), direct_egvc_indx.end() );
			alp::buildVector( select_non_direct, non_direct_egvc_indx.begin(), non_direct_egvc_indx.end() );

#ifdef DEBUG
			std::cout << " count_direct_egvc = " << count_direct_egvc << "\n";
			std::cout << " count_non_direct_egvc = " << count_non_direct_egvc << "\n";
#endif
			auto egvals_direct = get_view< alp::structures::General >( egvals, select_direct );
			auto egvals_non_direct = get_view< alp::structures::General >( egvals, select_non_direct );

			auto Egvecs_direct = alp::get_view< alp::structures::Orthogonal >(
				Egvecs, select_direct, select_direct
			);
			auto Egvecs_non_direct = alp::get_view< alp::structures::Orthogonal >(
				Egvecs, select_non_direct, select_non_direct
			);


			// Vector< D, structures::General, Dense > d_nontrv( n );
			// Vector< D, structures::General, Dense > v_nontrv( n );
			// alp::set( d_nontrv, zero );
			// alp::set( v_nontrv, zero );

			// for( size_t i = 0; i < n; i++ ) {
			// 	if( std::abs( v[ i ] ) < eps ) {
			// 		//simple egval formula ;
			// 		//set eigenvalue
			// 		egvals[ i ] = d[ i ];
			// 		//set eigenvector
			// 		//(could be done without temp vector by using eWiseLambda)
			// 		Vector< D, structures::General, Dense > dvec( n );
			// 		alp::set( dvec, zero );
			// 		dvec[ i ] = *one;
			// 		auto Egvecs_vec_view = get_view( Egvecs, utils::range( 0, n ), i );
			// 		rc = rc ? rc : alp::set( Egvecs_vec_view, dvec );
			// 	} else {
			// 		//complicated egval formula ;
			// 		d_nontrv[ non_trivial_egvc_count ] = d[ i ];
			// 		v_nontrv[ non_trivial_egvc_count ] = v[ i ];
			// 		++non_trivial_egvc_count;
			// 	}
			// }

			// auto d_nontrv_nnz_view = get_view( d_nontrv, utils::range( 0, non_trivial_egvc_count ) );
			// auto v_nontrv_nnz_view = get_view( v_nontrv, utils::range( 0, non_trivial_egvc_count ) );

			// Vector< D, structures::General, Dense > egvals_nontrv( non_trivial_egvc_count );
			// Matrix< D, OrthogonalType, Dense > Egvecs_nontrv( non_trivial_egvc_count );
			// rc = rc ? rc : alp::set( egvals_nontrv, zero );
			// rc = rc ? rc : alp::set( Egvecs_nontrv, zero );

// #ifdef DEBUG
// 			print_vector( "eigensolveDiagPlusOuter: d ", d );
// 			print_vector( "eigensolveDiagPlusOuter: v ", v );
// 			print_vector( "eigensolveDiagPlusOuter: d_nontrv_nnz_view ", d_nontrv_nnz_view );
// 			print_vector( "eigensolveDiagPlusOuter: v_nontrv_nnz_view ", v_nontrv_nnz_view );

// 			// for( size_t i = 0; i < non_trivial_egvc_count; ++i ) {
// 			// 	std::cout << " ============ i= " << i << "  ============\n";

// 			// 	std::cout << " d = array([";
// 			// 	for( size_t i = 0; i < non_trivial_egvc_count; ++i ) {
// 			// 		std::cout << d_nontrv_nnz_view[ i ] << ", ";
// 			// 	}
// 			// 	std::cout << " ])\n";
// 			// 	std::cout << " v = array([";
// 			// 	for( size_t i = 0; i < non_trivial_egvc_count; ++i ) {
// 			// 		std::cout << v_nontrv_nnz_view[ i ] << ", ";
// 			// 	}
// 			// 	std::cout << " ])\n";

// 			// 	Scalar< D > a( d_nontrv_nnz_view[ i ] );
// 			// 	Scalar< D > b( d_nontrv_nnz_view[ i ] );
// 			// 	std::cout << "0 a,b=" << *a << " " << *b << "\n";
// 			// 	if( i + 1 < non_trivial_egvc_count  ) {
// 			// 		std::cout << " alp::set b to <<" << d_nontrv_nnz_view[ i + 1 ] << ">> \n";
// 			// 		rc = alp::set( b, Scalar< D >( d_nontrv_nnz_view[ i + 1 ] ) );
// 			// 		if( rc != SUCCESS ) {
// 			// 			std::cout << " **** alp::set failed ***** \n";
// 			// 		}
// 			// 		std::cout << "1  a,b=" << *a << " " << *b << "\n";
// 			// 	} else {
// 			// 		Scalar< D > alpha( zero );
// 			// 		rc = rc ? rc : norm2( alpha, v, ring );
// 			// 		foldl( b, alpha, ring.getAdditiveOperator() );
// 			// 		std::cout << "2 a,b=" << *a << " " << *b << "\n";
// 			// 	}
// 			// 	Scalar< D > lambda( ( *a - *b ) / 2 );

// 			// 	std::cout << " a,lambda,b=" << *a << " " << *lambda << " " << *b << "\n";

// 			// 	rc = rc ? rc : bisec_sec_eq( lambda, d_nontrv_nnz_view, v_nontrv_nnz_view, a, b );
// 			// 	std::cout << " lambda (" << i << ") = " << *lambda << "\n";
// 			}
// #endif


// #ifdef TEMPDISABLE
// 			for( size_t i = 0; i < non_trivial_egvc_count; i++ ) {
// 				egvals_nontrv[ i ] = d_nontrv[ i ];
// 				Vector< D, structures::General, Dense > dvec( non_trivial_egvc_count );
// 				alp::set( dvec, zero );
// 				dvec[ i ] = *one;
// 				auto Egvecs_nontrv_vec_view = get_view( Egvecs_nontrv, utils::range( 0, non_trivial_egvc_count ), i );
// 				rc = rc ? rc : alp::set( Egvecs_nontrv_vec_view, dvec );
// 			}
// #else
// 			not implemented;
// #endif

// 			//copy egvals_nontrv and Egvecs_nontrv into egvals and Egvecs
// 			size_t k = 0;
// 			for( size_t i = 0; i < n; i++ ) {
// 				if( !( std::abs( v[ i ] ) < eps ) ) {
// 					egvals[ i ] = egvals_nontrv[ k ];
// 					//resolve this with select/permute view
// #ifdef TEMPDISABLE
// 					auto Egvecs_nontrv_vec_view = get_view( Egvecs_nontrv, utils::range( 0, non_trivial_egvc_count ), k );
// 					auto Egvecs_vec_view = get_view( Egvecs, utils::range( 0, non_trivial_egvc_count ), i ); // this is wrong
// 					rc = rc ? rc : alp::set( Egvecs_vec_view, Egvecs_nontrv_vec_view );
// #endif
// 				}
// 			}

			return rc;
		}


		/**
		 * Calcualte eigendecomposition of symmetric tridiagonal matrix T
		 *        \f$T = Qdiag(d)Q^T\f$ where
		 *        \a T is real symmetric tridiagonal
		 *        \a Q is orthogonal (columns are eigenvectors).
		 *        \a d is vector containing eigenvalues.
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type minus operator used in the computation
		 * @tparam Divide   Type of divide operator used in the computation
		 * @param[out] Q    output orthogonal matrix contaning eigenvectors
		 * @param[out] d    output vector containg eigenvalues
		 * @param[in]  T    input symmetric tridiagonal matrix
		 * @param[in]  ring A semiring for operations
		 * @return RC       SUCCESS if the execution was correct
		 *
		 */
		template<
			typename D,
			typename SymmOrHermTridiagonalType,
			typename OrthogonalType,
			typename SymmHermTrdiViewType,
			typename OrthViewType,
			typename SymmHermTrdiImfR,
			typename SymmHermTrdiImfC,
			typename OrthViewImfR,
			typename OrthViewImfC,
			typename VecViewType,
			typename VecImfR,
			typename VecImfC,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >
		>
		RC symm_tridiag_dac_eigensolver(
			Matrix<
				D,
				SymmOrHermTridiagonalType,
				Dense,
				SymmHermTrdiViewType,
				SymmHermTrdiImfR,
				SymmHermTrdiImfC
			> &T,
			Matrix<
				D,
				OrthogonalType,
				Dense,
				OrthViewType,
				OrthViewImfR,
				OrthViewImfC
			> &Q,
			Vector<
				D,
				structures::General,
				Dense,
				VecViewType,
				VecImfR,
				VecImfC
			> &d,
			const Ring & ring = Ring(),
			const Minus & minus = Minus(),
			const Divide & divide = Divide()
		) {
			(void)ring;
			(void)minus;
			(void)divide;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			RC rc = SUCCESS;

			const size_t n = nrows( T );
			const size_t m = n / 2;

			if( n == 1 ) {
				//d=T[0];
				rc = rc ? rc : eWiseLambda(
					[ &d ]( const size_t i, const size_t j, D &val ) {
						(void) i;
						(void) j;
						alp::set( d, Scalar< D > ( val ) );
					},
					T
				);
				// Q=array([[1]]);
				rc = rc ? rc : alp::set( Q, one );

				return rc;
			}


			Vector< D, structures::General, Dense > v( n );
			rc = rc ? rc : set( v, zero );
			rc = rc ? rc : eWiseLambda(
				[ &T, &m, &ring ]( const size_t i, D &val ) {
					if( i ==  m - 1 ) {
						val = ring.template getOne< D >();
					}
					if( i ==  m) {
						val = internal::access( T, internal::getStorageIndex( T, m - 1, m ) );
					}
				},
				v
			);
#ifdef DEBUG
			print_vector( " v = ", v );
#endif
			Matrix< D, SymmOrHermTridiagonalType, Dense > Atmp( n );
			rc = rc ? rc : alp::set( Atmp, T );
			auto vvt =  alp::outer( v, ring.getMultiplicativeOperator() ) ;

#ifdef DEBUG
			print_matrix( " Atmp(0) = ", Atmp );
			print_matrix( " vvt = ", vvt );
#endif
			rc = rc ? rc : alp::foldl( Atmp, vvt, minus );

#ifdef DEBUG
			print_matrix( " Atmp(1) = ", Atmp );
#endif

			auto Ttop = get_view< SymmOrHermTridiagonalType >( Atmp, utils::range( 0, m ), utils::range( 0, m ) );
			auto Tdown = get_view< SymmOrHermTridiagonalType >( Atmp, utils::range( m, n ), utils::range( m, n ) );

#ifdef DEBUG
			print_matrix( " Ttop = ", Ttop );
			print_matrix( " Tdown = ", Tdown );
#endif

			Vector< D, structures::General, Dense > dtmp( n );
			rc = rc ? rc : alp::set( dtmp, zero );
			auto dtop = get_view( dtmp, utils::range( 0, m ) );
			auto ddown = get_view( dtmp, utils::range( m, n ) );

			Matrix< D, OrthogonalType, Dense > U( n );
			rc = rc ? rc : alp::set( U, zero );

			auto Utop = get_view< OrthogonalType >( U, utils::range( 0, m ), utils::range( 0, m ) );
			auto Udown = get_view< OrthogonalType >( U, utils::range( m, n ), utils::range( m, n ) );

			// rc = rc ? rc : symm_tridiag_dac_eigensolver( Ttop, Utop, dtop, ring );
			// rc = rc ? rc : symm_tridiag_dac_eigensolver( Tdown, Udown, ddown, ring );
			std::cout << " --> ust one iteration\n";

#ifdef DEBUG
			std::cout << " after symm_tridiag_dac_eigensolver call:\n";
			print_matrix( " Utop = ", Utop );
			print_matrix( " Udown = ", Udown );
			print_matrix( " U = ", U );
#endif

			Vector< D, structures::General, Dense > z( n );
			rc = rc ? rc : alp::set( z, zero );

#ifdef DEBUG
			print_vector( "  v  ", v );
			print_vector( "  z  ", z );
#endif

#ifdef TEMPDISABLE
			// while mxv does not support vectors/view
			// we cast vector->matrix and use mxm
			auto z_mat_view = get_view< view::matrix >( z );
			auto v_mat_view = get_view< view::matrix >( v );
			rc = rc ? rc : mxm(
				z_mat_view,
				alp::get_view< alp::view::transpose >( U ),
				v_mat_view,
				ring
			);
#else
			//z=U^T.dot(v)
			rc = rc ? rc : mxv(
				z,
				alp::get_view< alp::view::transpose >( U ),
				v,
				ring
			);
#endif

#ifdef DEBUG
			print_vector( "  z  ", z );
#endif

			// permutations which sort dtmp
			std::vector< size_t > isort_dtmp( n, 0 );
			for( size_t i = 0 ; i < n ; ++i ) {
				isort_dtmp[ i ] = i;
			}
			std::sort(
				isort_dtmp.begin(),
				isort_dtmp.end(),
				[ &dtmp ]( const size_t &a, const size_t &b ) {
					return ( dtmp[ a ] < dtmp[ b ] );
				}
			);
			alp::Vector< size_t > permutation_vec( n );
			alp::buildVector( permutation_vec, isort_dtmp.begin(), isort_dtmp.end() );

#ifdef DEBUG
			print_vector( "  dtmp  ", dtmp );
			// std::cout << " sort(dtmp) = \n";
			// std::cout << "    [ ";
			// for( size_t i = 0 ; i < n ; ++i ) {
			// 	std::cout << "\t" << dtmp[ isort_dtmp[ i ] ];
			// }
			// std::cout << " ]\n";
#endif

			auto dtmp2 = alp::get_view< alp::structures::General >(
				dtmp,
				permutation_vec
			);
			auto ztmp2 = alp::get_view< alp::structures::General >(
				z,
				permutation_vec
			);
#ifdef DEBUG
			print_vector( "  dtmp2  ", dtmp2 );
			print_vector( "  ztmp2  ", ztmp2 );
#endif


			rc = rc ? rc : alp::set( d, zero );
			Matrix< D, OrthogonalType, Dense > QdOuter( n );
			rc = rc ? rc : alp::set( QdOuter, zero );
			auto QdOuter2 = alp::get_view< alp::structures::Orthogonal >(
				QdOuter, permutation_vec, permutation_vec
			);

#ifdef DEBUG
			print_matrix( "  QdOuter(in)  ", QdOuter );
#endif
			rc = rc ? rc : eigensolveDiagPlusOuter( d, QdOuter2, dtmp2, ztmp2 );
#ifdef DEBUG
			print_matrix( "  QdOuter(out)  ", QdOuter );
#endif


			// Q=U.dot((V[:,iiinv]).T)
			rc = rc ? rc : alp::set( Q, zero );
			rc = rc ? rc : mxm(
				Q,
				U,
				alp::get_view< alp::view::transpose >( QdOuter ),
				ring
			);

			return rc;
		}
	} // namespace algorithms
} // namespace alp
