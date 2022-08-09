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

/**
 *
 * @file
 *
 * This file registers available index mapping functions (IMFs).
 * IMFs are maps between integer intervals and can be used to define
 * affine \em access transformations in the form of access matrices.
 * For example, an access matrix \f$G_f\in R^{N\times N}\f$
 * parametrized by the IMF \f$f\f$ such that
 * \f[G_f = \sum_{i=0}^{n-1} e_i^n\left(e_{f(i)}^N\right)^T\f]
 * could be used to access a group of $n\eN$ rows of matrix
 * \f$A\in R^{N\times N}\f$
 * according to \f$f\f$ by multiplying \f$A\f$ by \f$G_f\f$ from the left:
 * \f[\tilde{A} = G_f\cdot A,\quad \tilde{A}\in R^{n\times N}\f]
 *
 * \note The idea of parametrized matrices to express matrix accesses at
 *       a higher level of mathematical abstractions is inspired by the
 *       SPIRAL literature (Franchetti et al. SPIRAL: Extreme Performance Portability.
 *       http://spiral.net/doc/papers/08510983_Spiral_IEEE_Final.pdf).
 *       Similar affine formulations are also used in the polyhedral
 *       compilation literature to express concepts such as access
 *       relations.
 *       In this draft we use integer maps. A symbolic version of them could be
 *       defined using external libraries such as the Integer Set Library (isl
 *       \link https://libisl.sourceforge.io/).
 *
 */

#ifndef _H_ALP_IMF
#define _H_ALP_IMF

#include <memory>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace alp {

	namespace imf {

		class IMF {

			public:

				const size_t n;
				const size_t N;

				IMF( const size_t n, const size_t N ): n( n ), N( N ) {}

			protected:

				template< typename OtherImf >
				bool isSame( const OtherImf & other ) const {
					//static_assert( std::is_same< decltype( *this ), decltype( other ) >::value );
					return n == other.n && N == other.N;
				}

			private:

				/** Implements the mapping function of the IMF */
				size_t map( const size_t ) const;

		}; // class IMF

		/**
		 * The strided IMF.
		 * \f$I_n =[0, n), I_N =[0, N)\f$
		 * \f$Strided_{b, s} = I_n \rightarrow I_N; i \mapsto b + si\f$
		 */

		class Strided: public IMF {

			public:

				const size_t b;
				const size_t s;

				size_t map( const size_t i ) const {
#ifdef _DEBUG
					std::cout << "Calling Strided map\n";
#endif
					return b + s * i;
				}

				Strided( const size_t n, const size_t N, const size_t b, const size_t s ): IMF( n, N ), b( b ), s( s ) { }

				template< typename OtherIMF >
				bool isSame( const OtherIMF &other ) const {
					return IMF::isSame( other ) &&
						b == static_cast< const Strided & >( other ).b &&
						s == static_cast< const Strided & >( other ).s;
				}
		};

		/**
		 * The identity IMF.
		 * \f$I_n = [0, n)\f$
		 * \f$Id = I_n \rightarrow I_n; i \mapsto i\f$
		 */

		class Id: public Strided {

			public:

				Id( const size_t n ) : Strided( n, n, 0, 1 ) {}
		};

		class Select: public IMF {

			public:

				std::vector< size_t > select;

				size_t map( const size_t i ) const {
#ifdef _DEBUG
					std::cout << "Calling Select map.\n";
#endif
					return select.at( i );
				}

				Select( size_t N, std::vector< size_t > &select ): IMF( select.size(), N ), select( select ) {
					//if ( *std::max_element( select.cbegin(), select.cend() ) >= N) {
					//	throw std::runtime_error("IMF Select beyond range.");
					//}
				}

				Select( size_t N, std::vector< size_t > &&select ): IMF( select.size(), N ), select( select ) {
#ifdef _DEBUG
					std::cout << "Select move constructor\n";
#endif
					//if ( *std::max_element( select.cbegin(), select.cend() ) >= N) {
					//	throw std::runtime_error("IMF Select beyond range.");
					//}
				}

				template< typename OtherIMF >
				bool isSame( const OtherIMF &other ) const {
					return IMF::isSame( other ) && select == static_cast< const Select & >( other ).select;
				}
		};

		/**
		 * A composition of two IMFs.
		 * \f$I_{g,n} =[0, n), I_{g,N} =[0, N)\f$
		 * \f$I_{f,n} =[0, n), I_{f,N} =[0, N)\f$
		 * \f$Composed_{f, g} = I_{g,n} \rightarrow I_{f,N}; i \mapsto f( g( i ) )\f$
		 *
		 * \tparam LeftIMF  The left function of the composition operator
		 *                  (applied second, i.e., \f$g\f$ )
		 * \tparam RightIMF The right function of the composition operator
		 *                  (applied first, i.e., \f$f\f$ )
		 *
		 * For specific combinations of the IMF types, there are specializations
		 * that avoid nested function calls by fusing two functions into one.
		 */

		template< typename LeftImf, typename RightImf >
		class Composed: public IMF {

			public:
				const LeftImf &f;
				const RightImf &g;

				size_t map( const size_t i ) const {
#ifdef _DEBUG
						std::cout << "Calling Composed< IMF, IMF>::map()\n";
#endif
						return f.map( g.map( i ) );
				}

				Composed( const LeftImf &f, const RightImf &g ):
					IMF( g.n, f.N ), f( f ), g( g ) {
#ifdef _DEBUG
						std::cout << "Creating composition of IMFs that cannot be composed into a"
						             "single mapping function. Consider the effect on performance.\n";
#endif
					}

		};

		template< typename LeftImf, typename RightImf >
		struct composed_type {
			typedef Composed< LeftImf, RightImf > type;
		};

		template<>
		struct composed_type< Strided, Strided > {
			typedef Strided type;
		};

		template<>
		struct composed_type< Id, Strided > {
			typedef Strided type;
		};

		template<>
		struct composed_type< Strided, Id > {
			typedef Strided type;
		};

		template<>
		struct composed_type< Id, Id > {
			typedef Id type;
		};

		/**
		 * Creates the composed IMF from two provided input IMFs.
		 * Depending on the input IMF types, the factory may
		 * specialize the returned IMF type.
		 */

		struct ComposedFactory {

			template< typename LeftImf, typename RightImf >
			static typename composed_type< LeftImf, RightImf >::type create( const LeftImf &, const RightImf & );

		};

		template<>
		Strided ComposedFactory::create( const Id &f, const Strided &g ) {
			return Strided( g.n, f.N, g.b, g.s );
		}

		template<>
		Strided ComposedFactory::create( const Strided &f, const Strided &g ) {
			return Strided( g.n, f.N, f.s * g.b + f.b, f.s * g.s );
		}

		template<>
		Strided ComposedFactory::create( const Strided &f, const Id &g ) {
			return Strided( g.n, f.N, f.b, f.s );
		}

		/** Composition of two Id IMFs is an Id Imf */
		template<>
		Id ComposedFactory::create( const Id &f, const Id &g ) {
#ifdef NDEBUG
			(void)f;
#endif
			// The first function's co-domain must be equal to the second function's domain.
			assert( g.N == f.n );
			return Id( g.n );
		}

		template<>
		Composed< Strided, Select > ComposedFactory::create( const Strided &f1, const Select &f2 ) {
			return Composed< Strided, Select >( f1, f2 );
		}

	}; // namespace imf

} // namespace alp

#endif // _H_ALP_IMF
