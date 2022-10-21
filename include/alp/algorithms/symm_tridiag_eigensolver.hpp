/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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
#ifdef DEBUG
#include "../tests/utils/print_alp_containers.hpp"
#endif

// TEMPDISABLE should be removed in the final version
#define TEMPDISABLE

namespace alp {

	namespace algorithms {

		/**
		 * find zero of secular equation in interval <a,b>
		 * using bisection
		 * this is not an optimal algorithm and there are many
		 * more efficient implementation
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
			// Vector v should be const, but that would disable eWiseLambda,
			// to be resolved in the future
			Vector<	D, structures::General,	Dense, VecView2, VecImfR2, VecImfC2 > &v,
			const Scalar< D > &a,
			const Scalar< D > &b,
			const D tol = 1.e-10,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			RC rc = SUCCESS;

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
					alp::foldl( alpha, Scalar< D > ( val ), ring.getMultiplicativeOperator() );
					alp::foldl( beta, x0, minus );
					alp::foldl( alpha, beta, divide );
					alp::foldl( fx0, alpha, ring.getAdditiveOperator() );
				},
				v
			);

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
		 * Calculate eigendecomposition of system D + vvt
		 *        \f$D = diag(d)$ is diagonal matrix and
		 *        \a vvt outer product outer(v,v)
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type of minus operator used in the computation
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
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
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
			// the following loop should be replaced by ALP primitives
			// since v is not sorted it seems that another sort is needed
			// currently there is no simple way to impement this in ALP
			for( size_t i = 0; i < n; i++ ) {
				if( std::abs( v[ i ] ) < eps ) {
					// in these cases equals are canonical vectors
					// and eigenvalues are d[i]
					direct_egvc_indx[ count_direct_egvc ] = i ;
					++count_direct_egvc;
				} else {
					// these cases require complicated egval formula
					// and for cases where egval is close to the singular point
					// different algorithm for eigenvectors needs to be implemented
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
			std::cout << " ---->     count_direct_egvc = " << count_direct_egvc << "\n";
			std::cout << " ----> count_non_direct_egvc = " << count_non_direct_egvc << "\n";
#endif
			auto egvals_direct = alp::get_view< alp::structures::General >( egvals, select_direct );
			auto egvals_non_direct = alp::get_view< alp::structures::General >( egvals, select_non_direct );

			auto Egvecs_non_direct = alp::get_view< alp::structures::Orthogonal >(
				Egvecs, select_non_direct, select_non_direct
			);

			// copy d -> egvals for direct part
			rc = rc ? rc : alp::set(
				egvals_direct,
				get_view< alp::structures::General >( d, select_direct )
			);

			auto d_view = alp::get_view< alp::structures::General >( d, select_non_direct );
			auto v_view = alp::get_view< alp::structures::General >( v, select_non_direct );

#ifdef DEBUG
			print_vector( "eigensolveDiagPlusOuter: d ", d );
			print_vector( "eigensolveDiagPlusOuter: v ", v );
			print_vector( "eigensolveDiagPlusOuter: d_view ", d_view );
			print_vector( "eigensolveDiagPlusOuter: v_view ", v_view );
#endif

			// vec_b = {d_view[1], d_view[2], ... , d_view[N-1], d_view[N]+dot(v,v) }
			size_t nn = alp::getLength( d_view );
			alp::Vector< D > vec_b( nn );
			auto v1 = alp::get_view( vec_b, utils::range( 0, nn - 1 ) );
			auto v2 = alp::get_view( d_view, utils::range( 1, nn ) );
			rc = rc ? rc : alp::set( v1, v2 );
			auto v3 = alp::get_view( vec_b, utils::range( nn - 1, nn ) );
			auto v4 = alp::get_view( d_view, utils::range( nn - 1, nn ) );
			rc = rc ? rc : alp::set( v3, v4 );

			// eWiseLambda currently does not work with select view
			// dot() does not work with select view
			// as a (temp) solution we use temp vectors
			alp::Vector< D > vec_temp_egvals( nn );
			alp::Vector< D > vec_temp_d( nn );
			alp::Vector< D > vec_temp_v( nn );

			rc = rc ? rc : alp::set( vec_temp_egvals, zero );
			rc = rc ? rc : alp::set( vec_temp_d, d_view );
			rc = rc ? rc : alp::set( vec_temp_v, v_view );

			Scalar< D > alpha( zero );
			// there is a bug in dot() when called on select views
			//rc = rc ? rc : alp::dot( alpha, d_view, d_view, ring );
			rc = rc ? rc : alp::dot( alpha, vec_temp_v, vec_temp_v, ring );

			auto v5 = alp::get_view( vec_b, utils::range( alp::getLength( vec_b ) - 1, alp::getLength( vec_b ) ) );
			rc = rc ? rc : alp::foldl( v5, alpha, ring.getAdditiveOperator() );

			rc = rc ? rc : alp::eWiseLambda(
				[ &d_view, &vec_temp_v, &vec_b ]( const size_t i, D &val ) {
					Scalar< D > a( d_view[ i ] );
					Scalar< D > b( vec_b[ i ] );
					Scalar< D > w( ( *a + *b ) / 2 );
					bisec_sec_eq( w, d_view, vec_temp_v, a, b );
					val = *w;
				},
				vec_temp_egvals
			);
			rc = rc ? rc : alp::set( egvals_non_direct, vec_temp_egvals );

			Matrix< D, structures::General, Dense > tmp_egvecs( nn, nn );
			Matrix< D, structures::General, Dense > tmp_denominator( nn, nn );

			alp::Vector< D > ones( nn );
			rc = rc ? rc : alp::set( ones, one );
			rc = rc ? rc : alp::set(
				tmp_egvecs,
				alp::outer( vec_temp_v, ones, ring.getMultiplicativeOperator() )
			);

			auto ddd = alp::outer( vec_temp_d, ones, ring.getMultiplicativeOperator() );
			auto lll = alp::outer( ones, egvals_non_direct, ring.getMultiplicativeOperator() );
			rc = rc ? rc : alp::set( tmp_denominator, ddd );
			rc = rc ? rc : alp::foldl( tmp_denominator, lll, minus );
			rc = rc ? rc : alp::foldl( tmp_egvecs, tmp_denominator, divide );

			// while fold matrix -> vector would be a solution to
			// normalize columns in tmp_egvecs,
			// here we abuse the syntax and use eWiseLambda.
			// Once fold matrix -> vector implemented, the next section should be rewritten
			rc = rc ? rc : alp::eWiseLambda(
				[ &tmp_egvecs, &nn, &ring, &divide, &zero ]( const size_t i, D &val ) {
					(void) val;
					auto egvec_i = get_view( tmp_egvecs, utils::range( 0, nn ), i );
					Scalar< D > norm_i( zero );
					alp::norm2( norm_i, egvec_i, ring );
					alp::foldl( egvec_i, norm_i , divide );
				},
				ones
			);

			// update results
			auto egvecs_view = alp::get_view( Egvecs_non_direct, utils::range( 0, nn ), utils::range( 0, nn ) );
			auto tmp_egvecs_orth_view = alp::get_view< OrthogonalType >( tmp_egvecs );
			rc = rc ? rc : alp::set( egvecs_view, tmp_egvecs_orth_view );

			return rc;
		}


		/**
		 * Calculate eigendecomposition of symmetric tridiagonal matrix T
		 *        \f$T = Qdiag(d)Q^T\f$ where
		 *        \a T is real symmetric tridiagonal
		 *        \a Q is orthogonal (columns are eigenvectors).
		 *        \a d is vector containing eigenvalues.
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type of minus operator used in the computation
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
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			(void) ring;
			(void) minus;
			(void) divide;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			RC rc = SUCCESS;

			const size_t n = nrows( T );
			const size_t m = n / 2;

			if( n == 1 ) {
				//d=T[0];
				rc = rc ? rc : alp::eWiseLambda(
					[ &d ]( const size_t i, const size_t j, D &val ) {
						(void) i;
						(void) j;
						alp::set( d, Scalar< D > ( val ) );
					},
					T
				);
				// Q=[[1]]; a 1x1 matrix
				rc = rc ? rc : alp::set( Q, one );

				return rc;
			}


			Vector< D, structures::General, Dense > v( n );
			rc = rc ? rc : alp::set( v, zero );
			rc = rc ? rc : alp::eWiseLambda(
				[ &T, &m, &ring ]( const size_t i, D &val ) {
					if( i == m - 1 ) {
						val = ring.template getOne< D >();
					}
					if( i == m) {
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
			print_matrix( " vvt = ", vvt );
#endif
			rc = rc ? rc : alp::foldl( Atmp, vvt, minus );

#ifdef DEBUG
			print_matrix( " Atmp(updated)  ", Atmp );
#endif

			auto Ttop = alp::get_view< SymmOrHermTridiagonalType >( Atmp, utils::range( 0, m ), utils::range( 0, m ) );
			auto Tdown = alp::get_view< SymmOrHermTridiagonalType >( Atmp, utils::range( m, n ), utils::range( m, n ) );

#ifdef DEBUG
			print_matrix( " Ttop = ", Ttop );
			print_matrix( " Tdown = ", Tdown );
#endif

			Vector< D, structures::General, Dense > dtmp( n );
			rc = rc ? rc : alp::set( dtmp, zero );
			auto dtop = alp::get_view( dtmp, utils::range( 0, m ) );
			auto ddown = alp::get_view( dtmp, utils::range( m, n ) );

			Matrix< D, OrthogonalType, Dense > U( n );
			rc = rc ? rc : alp::set( U, zero );

			auto Utop = alp::get_view< OrthogonalType >( U, utils::range( 0, m ), utils::range( 0, m ) );
			auto Udown = alp::get_view< OrthogonalType >( U, utils::range( m, n ), utils::range( m, n ) );

			rc = rc ? rc : symm_tridiag_dac_eigensolver( Ttop, Utop, dtop, ring );
			rc = rc ? rc : symm_tridiag_dac_eigensolver( Tdown, Udown, ddown, ring );
			//std::cout << " --> ust one iteration\n";

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
			auto z_mat_view = alp::get_view< view::matrix >( z );
			auto v_mat_view = alp::get_view< view::matrix >( v );
			rc = rc ? rc : alp::mxm(
				z_mat_view,
				alp::get_view< alp::view::transpose >( U ),
				v_mat_view,
				ring
			);
#else
			//z=U^T.dot(v)
			rc = rc ? rc : alp::mxv(
				z,
				alp::get_view< alp::view::transpose >( U ),
				v,
				ring
			);
#endif

#ifdef DEBUG
			print_vector( "  d  ", dtmp );
			print_vector( "  z  ", z );
#endif

			// permutations which sort dtmp
			std::vector< size_t > isort_dtmp( n, 0 );
			std::vector< size_t > no_permute_data( n, 0 );
			for( size_t i = 0; i < n; ++i ) {
				isort_dtmp[ i ] = i;
				no_permute_data[ i ] = i;
			}
			std::sort(
				isort_dtmp.begin(),
				isort_dtmp.end(),
				[ &dtmp ]( const size_t &a, const size_t &b ) {
					return ( dtmp[ a ] < dtmp[ b ] );
				}
			);
			alp::Vector< size_t > permutation_vec( n );
			alp::Vector< size_t > no_permutation_vec( n );
			alp::buildVector( permutation_vec, isort_dtmp.begin(), isort_dtmp.end() );
			alp::buildVector( no_permutation_vec, no_permute_data.begin(), no_permute_data.end() );

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
			auto QdOuter_diag = alp::get_view< alp::view::diagonal >( QdOuter );
			rc = rc ? rc : alp::set(
				QdOuter_diag,
				one
			);

			auto QdOuter2 = alp::get_view< alp::structures::Orthogonal >(
				QdOuter, permutation_vec, no_permutation_vec
			);

			rc = rc ? rc : eigensolveDiagPlusOuter( d, QdOuter2, dtmp2, ztmp2 );
#ifdef DEBUG
			print_vector( "  d(out)  ", d );
			print_matrix( "  QdOuter(out)  ", QdOuter );
			print_matrix( "  U  ", U );
#endif

			rc = rc ? rc : alp::set( Q, zero );
			rc = rc ? rc : alp::mxm( Q, U, QdOuter, ring	);

#ifdef DEBUG
			print_matrix( "  Q = U x Q   ", Q );
#endif

			return rc;
		}
	} // namespace algorithms
} // namespace alp
