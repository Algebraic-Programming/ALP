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

	template<
		typename T,
		typename Structure,
		enum Density density,
		typename View,
		typename ImfR,
		typename ImfC,
		enum Backend backend
	>
	class Vector;

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

				Strided( const Strided &other ) : IMF( other.n, other.N ), b( other.b ), s( other.s ) { }

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

				explicit Id( const size_t n ) : Strided( n, n, 0, 1 ) {}
		};

		/**
		 * The constant-mapping IMF.
		 * \f$I_n = [0, n)\f$
		 * \f$Constant = I_n \rightarrow I_N; i \mapsto const\f$ with \f$const in I_N\f$
		 */

		class Constant: public Strided {

			public:

				explicit Constant( const size_t n, const size_t N, const size_t value ) : Strided( n, N, value, 0 ) {}
		};

		/**
		 * The zero IMF.
		 * \f$I_n = [0, n)\f$
		 * \f$Zero = I_n \rightarrow I_1; i \mapsto 0\f$
		 */

		class Zero: public Strided {

			public:

				explicit Zero( const size_t n ) : Strided( n, 1, 0, 0 ) {}
		};


		class Select: public IMF {

			public:

				/** \internal \todo Change to ALP vector */
				std::vector< size_t > select;

				size_t map( const size_t i ) const {
					return select.at( i );
				}

				template< typename T, typename Structure, enum Density density, typename View, typename ImfR, typename ImfC, enum Backend backend >
				Select(
					size_t N,
					const alp::Vector< T, Structure, density, View, ImfR, ImfC, backend > &select
				): IMF( getLength( select ), N ), select( getLength( select ) ) {

					/** \internal \todo Use set when this->select becomes ALP vector */
					//set( this->select, select );
					for( size_t i = 0; i < getLength( select ); ++i ) {
						this->select[ i ] = select[ i ];
					}
#ifdef DEBUG
					// Check that select vector does not map outside of range [0,N)
					for( size_t i = 0; i < getLength( select ); ++i ) {
						if ( select[ i ] >= N ) {
							throw std::runtime_error("Provided select vector mapping beyond the provided range.");
						}
					}
#endif
				}

				Select( const Select &other ) : IMF( other.select.size(), other.N ), select( other.select ) {
#ifdef _DEBUG
					std::cout << "Select copy constructor\n";
#endif
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
				const LeftImf f;
				const RightImf g;

				size_t map( const size_t i ) const {
#ifdef _DEBUG
						std::cout << "Calling Composed< IMF, IMF>::map()\n";
#endif
						return f.map( g.map( i ) );
				}

				Composed( const LeftImf &f, const RightImf &g ):
					IMF( g.n, f.N ), f( f ), g( g ) {
#ifdef DEBUG
						std::cout << "Creating composition of IMFs that cannot be composed into a"
						             "single mapping function. Consider the effect on performance.\n";
#endif
					}

		};

		namespace internal {

			/**
			 * Ensures that the range of the right IMF matches the domain of the left.
			 * If the condition is not satisfied, throws an exception
			 *
			 * @tparam LeftImf   The type of the left IMF
			 * @tparam RightImf  The type of the right IMF
			 *
			 * @param[in] left_imf   The left IMF
			 * @param[in] right_imf  The right IMF
			 *
			 */
			template< typename LeftImf, typename RightImf >
			static void ensure_imfs_match( const LeftImf &left_imf, const RightImf &right_imf ) {
				if( !( right_imf.N == left_imf.n ) ) {
					throw std::runtime_error( "Cannot compose two IMFs with non-matching range and domain" );
				}
			}

		} // namespace internal

		/**
		 * Exposes the type and creates the composed IMF from two provided input IMFs.
		 *
		 * For certain combinations of IMFs, the resulting composed IMF is
		 * one of the fundamental types. In these cases, the factory is
		 * specialized to produce the appropriate type and object.
		 */
		template< typename LeftImf, typename RightImf >
		struct ComposedFactory {

			typedef Composed< LeftImf, RightImf > type;

			static type create( const LeftImf &f, const RightImf &g ) {
				internal::ensure_imfs_match( f, g );
				return type( f, g );
			}
		};

		template< typename RightImf >
		struct ComposedFactory< Id, RightImf > {

			typedef RightImf type;

			static type create( const Id &f, const RightImf &g ) {
				internal::ensure_imfs_match( f, g );
				return RightImf( g );
			}
		};

		template< typename LeftImf >
		struct ComposedFactory< LeftImf, Id > {

			typedef LeftImf type;

			static type create( const LeftImf &f, const Id &g ) {
				internal::ensure_imfs_match( f, g );
				return LeftImf( f );
			}
		};

		template<>
		struct ComposedFactory< Id, Id > {

			typedef Id type;

			static type create( const Id &f, const Id &g ) {
				internal::ensure_imfs_match( f, g );
				return type( f.n );
			}
		};

		template<>
		struct ComposedFactory< Strided, Strided >{

			typedef Strided type;

			static type create( const Strided &f, const Strided &g ) {
				internal::ensure_imfs_match( f, g );
				return type( g.n, f.N, f.s * g.b + f.b, f.s * g.s );
			}
		};

		template<>
		struct ComposedFactory< Strided, Constant > {

			typedef Constant type;

			static type create( const Strided &f, const Constant &g ) {
				internal::ensure_imfs_match( f, g );
				return type( g.n, f.N, f.b + f.s * g.b );
			}
		};

	}; // namespace imf

} // namespace alp

#endif // _H_ALP_IMF