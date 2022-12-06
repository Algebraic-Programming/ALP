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
#ifdef DEBUG
#include "../tests/utils/print_alp_containers.hpp"
#endif

namespace alp {

	namespace algorithms {

		/**
		 * Given a general matrix H perform an inplace Householder reflections in order to
		 * eliminate column elements H[i+d:,i] (below diagonal or subdigonal), H = U H.
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type minus operator used in the computation
		 * @tparam Divide   Type of divide operator used in the computation
		 * @param[in,out]   U updated orthogonal matrix
		 * @param[in,out]   H updated general matrix with column
		 *                         elements H[i+d:,i] eliminated
		 * @param[in]       i column to eliminate
		 * @param[in]       d offset from diagonal to eliminate, default 0
		 * @param[in]       ring A semiring for operations
		 * @return RC       SUCCESS if the execution was correct
		 */
		template<
			typename MatH,
			typename MatU,
			typename IndexType,
			typename D = typename MatH::value_type,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >,
			std::enable_if_t<
				structures::is_a< typename MatH::structure, structures::General >::value &&
				structures::is_a< typename MatU::structure, structures::Orthogonal >::value &&
				is_semiring< Ring >::value &&
				is_operator< Minus >::value &&
				is_operator< Divide >::value
			> * = nullptr
		>
		RC elminate_below_ith_diag(
			const IndexType i,
			MatH &H,
			MatU &U,
			const IndexType d = 0,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			const IndexType m = nrows( H );
			const IndexType n = ncols( H );

			// v=copy(A0[i+d:,i])
			auto a = get_view( H, utils::range( i + d, m ), i );
			Vector< D > v( m - ( i + d ) );
			rc = rc ? rc : alp::set( v, a );

			// alpha=v[0]/abs(v[0])
			Scalar< D > alpha( zero );
			auto v0 = get_view( v, utils::range( 0, 1 ) );
			rc = rc ? rc : foldl( alpha, v0, ring.getAdditiveMonoid() );
			rc = rc ? rc : foldl( alpha, Scalar< D >( std::abs( *alpha ) ), divide );

			// alpha=alpha*norm(v)
			Scalar< D > norm_v1( zero );
			rc = rc ? rc : norm2( norm_v1, v, ring );
			rc = rc ? rc : foldl( alpha, norm_v1, ring.getMultiplicativeOperator() );

			// v[0]=v[0]-alpha
			rc = rc ? rc : foldl( v0, alpha, minus );

			// v=v/norm(v)
			Scalar< D > norm_v2( zero );
			rc = rc ? rc : norm2( norm_v2, v, ring );
			rc = rc ? rc : foldl( v, norm_v2, divide );

			//P1=zeros((m-(i+d),m-(i+d))).astype(complex)
			//P1=P1-2*outer(v,conjugate(v))
			auto vvh = outer( v, ring.getMultiplicativeOperator() );
			Matrix< D, typename decltype( vvh )::structure, Dense > reflector( m - ( i + d ) );
			rc = rc ? rc : alp::set( reflector, vvh );
			rc = rc ? rc : foldl( reflector, Scalar< D > ( -2 ), ring.getMultiplicativeOperator() );

			// A0=P.dot(A0)
			auto Hupdate = get_view( H, utils::range( i + d, m ), utils::range( 0, n ) );
			Matrix< D, structures::General, Dense > Temp1( m - ( i + d ) , n );
			rc = rc ? rc : alp::set( Temp1, Hupdate );
			rc = rc ? rc : mxm( Hupdate, reflector, Temp1, ring );

			// Uk=Uk.dot(P)
			auto Uupdate = get_view< structures::OrthogonalColumns >( U, utils::range( 0, m ), utils::range( i + d, m ) );
			Matrix< D, structures::OrthogonalColumns, Dense > Temp2( m, m - ( i + d ) );
			rc = rc ? rc : alp::set( Temp2, Uupdate );
			rc = rc ? rc : mxm( Uupdate, Temp2, reflector, ring );

			return rc;
		}

		/**
		 * Computes Householder (inplace) bidiagonalisation of general matrix \f$H = U B V \f$
		 * where \a H is general (complex or real),
		 * \a U orthogonal, \a B is bidiagonal and  \a V orthogonal.
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type minus operator used in the computation
		 * @tparam Divide   Type of divide operator used in the computation
		 * @param[in,out]   U updated orthogonal matrix
		 * @param[in,out]   V updated orthogonal matrix
		 * @param[in,out]   H input general matrix, output bidiagonal matrix (B)
		 * @param[in]       ring A semiring for operations
		 * @return RC       SUCCESS if the execution was correct
		 *
		 */
		template<
			typename MatH,
			typename D = typename MatH::value_type,
			typename MatU,
			typename MatV,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >,
			std::enable_if_t<
				is_matrix< MatH >::value &&
				is_matrix< MatU >::value &&
				is_matrix< MatV >::value &&
				structures::is_a< typename MatH::structure, structures::General >::value &&
				structures::is_a< typename MatU::structure, structures::Orthogonal >::value &&
				structures::is_a< typename MatV::structure, structures::Orthogonal >::value &&
				is_semiring< Ring >::value &&
				is_operator< Minus >::value &&
				is_operator< Divide >::value
			> * = nullptr
		>
		RC householder_bidiag(
			MatU &U,
			MatH &H,
			MatV &V,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			const size_t m = nrows( H );
			const size_t n = ncols( H );

			// check sizes
			if(
				( ncols( U ) != nrows( H ) ) ||
				( ncols( H ) != nrows( V ) )
			) {
				std::cerr << "Incompatible sizes in householder_bidiag.\n";
				return FAILED;
			}

			//for i in range(min(n,m)):
			for( size_t i = 0; i < std::min( n, m ); ++i ) {
				// eliminate column elements below ith diagonal element
				if( i < std::min( n, m - 1 ) ) {
					rc = rc ? rc : elminate_below_ith_diag( i, H, U, static_cast< size_t >( 0 ), ring, minus, divide );
				}
				// eliminate row elements to the right from (i+1)th diagonal element
				if( i < std::min( n - 2, m ) ) {
					auto HT = get_view< alp::view::transpose >( H );
					auto VT = get_view< alp::view::transpose >( V );
					rc = rc ? rc : elminate_below_ith_diag( i, HT, VT, static_cast< size_t >( 1 ), ring, minus, divide );
				}
			}

			return rc;

		}
	} // namespace algorithms
} // namespace alp
