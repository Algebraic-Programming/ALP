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

#ifdef DEBUG
			print_vector( " dtop = ", dtop );
			print_vector( " ddown = ", ddown );
#endif

			Matrix< D, OrthogonalType, Dense > U( n );
			rc = rc ? rc : alp::set( U, zero );
#ifdef DEBUG
			print_matrix( " U = ", U );
#endif
			auto Utop = get_view< OrthogonalType >( U, utils::range( 0, m ), utils::range( 0, m ) );

#ifdef DEBUG
			print_matrix( " Utop = ", Utop );
#endif
			auto Udown = get_view< OrthogonalType >( U, utils::range( m, n ), utils::range( m, n ) );

#ifdef DEBUG
			print_matrix( " Udown = ", Udown );
#endif

			rc = rc ? rc : symm_tridiag_dac_eigensolver( Ttop, Utop, dtop, ring );
			rc = rc ? rc : symm_tridiag_dac_eigensolver( Tdown, Udown, ddown, ring );

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
#ifdef DEBUG
			print_vector( "  dtmp  ", dtmp );
			std::cout << " sort(dtmp) = \n";
			std::cout << "    [ ";
			for( size_t i = 0 ; i < n ; ++i ) {
				std::cout << "\t" << dtmp[ isort_dtmp[ i ] ];
			}
			std::cout << " ]\n";
#endif

			// temp solution:
			// dtmp2 and ztmp2 in the final version should be
			// permutations views of dtmp and z, respectively
			// instead, here we materialize permutations
			Vector< D, structures::General, Dense > dtmp2( n );
			Vector< D, structures::General, Dense > ztmp2( n );
			rc = rc ? rc : alp::set( dtmp2, zero );
			rc = rc ? rc : alp::set( ztmp2, zero );
			for( size_t i = 0 ; i < n ; ++i ) {
				dtmp2[ i ] = dtmp[ isort_dtmp[ i ] ];
				ztmp2[ i ] = z[ isort_dtmp[ i ] ];
			}
#ifdef DEBUG
			print_vector( "  dtmp2  ", dtmp2 );
			print_vector( "  ztmp2  ", ztmp2 );
#endif


#ifdef TEMPDISABLE
			// *********** diagDpOuter not implemented ***********
			//nummerical wrong, missing diagDpOuter implementation
			rc = rc ? rc : alp::set( d, dtmp2 );
			//nummerical wrong, missing diagDpOuter implementation
			Matrix< D, OrthogonalType, Dense > QdOuter( n );
			rc = rc ? rc : alp::set( QdOuter, zero );
			auto QdOuter_diag = alp::get_view< alp::view::diagonal >( QdOuter );
			rc = rc ? rc : alp::set( QdOuter_diag, one );
			// ***************************************************
#else
			D,V= diagDpOuter( dtmp2, ztmp2 );
#endif

#ifdef DEBUG
			print_matrix( "  QdOuter  ", QdOuter );
#endif

			// temp
			// unpermute cols into QdOuterUnpermuted
			Matrix< D, OrthogonalType, Dense > QdOuterUnpermuted( n );
			rc = rc ? rc : alp::set( QdOuterUnpermuted, zero );
			for( size_t i = 0 ; i < n ; ++i ) {
				auto vin = get_view( QdOuter, utils::range( 0, n ), i );
				auto vout = get_view( QdOuterUnpermuted, utils::range( 0, n ), isort_dtmp[ i ] );
				rc = rc ? rc : alp::set( vout, vin );
			}

#ifdef DEBUG
			print_matrix( "  QdOuterUnpermuted  ", QdOuterUnpermuted );
#endif

			// Q=U.dot((V[:,iiinv]).T)
			rc = rc ? rc : alp::set( Q, zero );
			rc = rc ? rc : mxm(
				Q,
				U,
				alp::get_view< alp::view::transpose >( QdOuterUnpermuted ),
				ring
			);

			return rc;
		}
	} // namespace algorithms
} // namespace alp
