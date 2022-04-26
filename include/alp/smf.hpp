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
 * This file registers available storage mapping functions (SMFs).
 * SMFs are maps between logical and physical storage space.
 *
 */

#ifndef _H_ALP_SMF
#define _H_ALP_SMF

#include <memory>

namespace alp {

	namespace smf {

		namespace polynoms {
			/** Implements a*i + b*j + c */
			class FirstOrder {

				private:

					const size_t a;
					const size_t b;
					const size_t c;

				public:

					FirstOrder( const size_t a, const size_t b, const size_t c ):
						a( a ), b( b ), c( c ) {}

					size_t f( const size_t i, const size_t j ) const {
						return a * i + b * j + c;
					}

			}; // FirstOrder

			/** Implements a*i^2 + b*i + c*j^2 + d*j + e */
			class SecondOrder {

				private:

					const double a;
					const double b;
					const double c;
					const double d;
					const double e;

				public:

					SecondOrder( const double a, const double b, const double c, const double d, const double e ):
						a( a ), b( b ), c( c ), d( d ), e( e ) {}

					size_t f( const size_t i, const size_t j ) const {
						return static_cast< size_t >( a * i * i + b * i + c * j * j + d * j + e );
					}

			}; // SecondOrder
		};

		class GeneralStorage : private polynoms::FirstOrder {
			private:
				const size_t M;
				const size_t N;
			public:
				GeneralStorage( const size_t M, const size_t N ) :
					polynoms::FirstOrder( N, 1, 0 ), M( M ), N( N ) {}

				size_t map( const size_t i, const size_t j ) const {
					std::cout << "Calling GeneralStorage:::map()\n";
					assert( i < M );
					assert( j < N );
					return polynoms::FirstOrder::f( i, j );
				}
		};

		class PackedStorage : private polynoms::SecondOrder {
			private:
				const size_t M;
				const size_t N;
			public:
				PackedStorage( const size_t M, const size_t N ) :
					polynoms::SecondOrder( 0.5, -0.5, 0, 1, 0 ), M( M ), N( N ) {}

				size_t map( const size_t i, const size_t j ) const {
					std::cout << "Calling PackedStorage::map()\n";
					assert( i < M );
					assert( j < N );
					return polynoms::SecondOrder::f( i, j );
				}
		};

		class BandStorage : private polynoms::FirstOrder {
			private:
				const size_t M;
				const size_t N;
				const size_t kl;
				const size_t ku;
			public:
				BandStorage( const size_t M, const size_t N, const size_t kl, const size_t ku ) :
					polynoms::FirstOrder( /* TODO */ 0, 0, 0 ), M( M ), N( N ), kl( kl ), ku( ku ) {

					assert( kl <= M );
					assert( ku <= N );
				}

		};

		template< typename ImfL, typename ImfR, typename Smf >
		class AMF {
			ImfL imf_l;
			ImfR imf_r;
			Smf smf;
			AMF( ImfL &&imf_l, ImfR &&imf_r, Smf &&smf ) : imf_l( imf_l ), imf_r( imf_r ), smf( smf ) {}	

			size_t map( const size_t i, const size_t j ) const {
				std::cout << "Calling AMF::map()\n";
				return smf.map( imf_l.map( i ), imf_r.map( j ) );
			}
		};

		template<>
		class AMF< imf::Strided, imf::Strided, GeneralStorage > : private polynoms::FirstOrder {
			/** For size checks */
			const size_t M;
			const size_t N;

			AMF( const size_t M, const size_t N, const imf::Strided &imf_l, const imf::Strided &imf_r, const GeneralStorage &smf ) :
				polynoms::FirstOrder( imf_l.s * N, imf_r.s, imf_l.b * N + imf_r.b ), M( M ), N( N ) {
				(void)smf;
			}

			size_t map( const size_t i, const size_t j ) const {
				std::cout << "Calling AMF::map()\n";
				assert( i < M );
				assert( j < N );
				return polynoms::FirstOrder::f( i, j );
			}
		};
	}; // namespace smf

} // namespace alp

#endif // _H_ALP_SMF
