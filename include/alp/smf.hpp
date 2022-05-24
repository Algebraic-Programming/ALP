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
#include "structures.hpp"

namespace alp {

	namespace storage {

		namespace polynomials {

			/**
			 * Implements the polynomial
			 * ( A*a*x^2 + B*b*y^2 + C*c*x*y + D*d*x + E*e*y + F*f ) / Denominator
			 * where uppercase coefficients are compile-time constant,
			 * lowercase coefficients are run-time constant,
			 * and x and y are variables.
			 * All coefficients, variables are integers and all operations are integer
			 * operations.
			 *
			 * The purpose of compile-time constant coefficients is to allow compile-time
			 * optimizations for zero factors.
			 */
			template< size_t Ax2, size_t Ay2, size_t Axy, size_t Ax, size_t Ay, size_t A0, size_t Denominator >
			struct BivariateQuadratic {

				static_assert( Denominator != 0, "Denominator cannot be zero (division by zero).");

				const size_t ax2, ay2, axy, ax, ay, a0;

				BivariateQuadratic(
					const size_t ax2, const size_t ay2, const size_t axy,
					const size_t ax, const size_t ay,
					const size_t a0 ) :
					ax2( ax2 ), ay2( ay2 ), axy( axy ),
					ax( ax ), ay( ay ),
					a0( a0 ) {}

				size_t f( const size_t x, const size_t y ) const {
					return (Ax2 * ax2 * x * x +
						Ay2 * ay2 * y * y +
						Axy * axy * x * y +
						Ax * ax * x +
						Ay * ay * y +
						A0 * a0) / Denominator;
				}

			}; // BivariateQuadratic

		};

		enum Scheme { NONE, FULL_ROW_MAJOR, FULL_COLUMN_MAJOR, PACKED, BANDED };

		/**
		 * Encapsulate type and methods associated with a specific
		 * storage scheme.
		 */
		template< enum Scheme storage_scheme >
		struct SMF {

			/** A type associated with this specific storage scheme. */
			typedef polynomials::BivariateQuadratic< 0, 0, 0, 0, 0, 0, 1 > type;

			/** Creates an instance of this storage scheme. */
			static type Instance() {
				return type( 0, 0, 0, 0, 0, 0 );
			}

			/** Returns the storage size corresponding to this storage scheme. */
			static size_t getStorageSize() {
				return 0;
			}
		};

		/**
		 * Implements full storage scheme with row-major ordering.
		 */
		template<>
		struct SMF< Scheme::FULL_ROW_MAJOR > {

			typedef polynomials::BivariateQuadratic< 0, 0, 0, 1, 1, 0, 1 > type;

			/**
			 * For row-major storage, dim represents number of columns.
			 * For column-major storage, dim represents number of rows.
			 */
			static type Instance( const size_t rows, const size_t cols ) {
				(void) rows;
				return type( 0, 0, 0, cols, 1, 0 );
			}

			/**
			 * @param[in] M  number of rows
			 * @param[in] N  number of columns
			 */
			static size_t getStorageSize( const size_t M, const size_t N ) {
				return M * N;
			}
		};

		/**
		 * Implements full storage scheme with column-major ordering.
		 *
		 * Mapping polynomial and getStorageSize are inherited from
		 * the specialization for full row-major storage scheme.
		 */
		template<>
		struct SMF< Scheme::FULL_COLUMN_MAJOR > : SMF< Scheme::FULL_ROW_MAJOR > {

			static type Instance( const size_t rows, const size_t cols ) {
				(void) cols;
				return type( 0, 0, 0, rows, 1, 0 );
			}
		};

		template<>
		struct SMF< Scheme::PACKED > {

			typedef polynomials::BivariateQuadratic< 0, 0, 0, 0, 0, 0, 1 /* TODO */ > type;

			static type Instance( size_t dim ) {
				(void) dim;
				return type( 0, 0, 0, 0, 0, 0 /* TODO */ );
			}

			/**
			 * @param[in] M  number of rows
			 * @param[in] N  number of columns
			 */
			static size_t getStorageSize( const size_t M, const size_t N ) {
				(void) M;
				(void) N;
				return 0; // TODO
			}
		};

		template<>
		struct SMF< Scheme::BANDED > {

			typedef polynomials::BivariateQuadratic< 0, 0, 0, 0, 0, 0, 1 /* TODO */ > type;

			static type Instance( size_t rows, size_t cols, size_t kl, size_t ku ) {
				(void) rows;
				(void) cols;
				(void) kl;
				(void) ku;
				return type( 0, 0, 0, 0, 0, 0 /* TODO */ );
			}

			/**
			 * @param[in] M  number of rows
			 * @param[in] N  number of columns
			 */
			static size_t getStorageSize( const size_t M, const size_t N, size_t kl, size_t ku ) {
				(void) M;
				(void) N;
				(void) kl;
				(void) ku;
				return 0; // TODO
			}
		};

		/**
		 * Provides a type of composed Access Mapping Function
		 * expressed as a BivariateQuadratic polynomial depending
		 * on the types of the IMFs and the SMF.
		 */
		template< typename ImfR, typename ImfC, enum Scheme storage_scheme >
		struct Composition {
			typedef polynomials::BivariateQuadratic< 1, 1, 1, 1, 1, 1, 1 > type;
		};

		template<>
		struct Composition< imf::Strided, imf::Strided, Scheme::FULL_ROW_MAJOR > {
			typedef polynomials::BivariateQuadratic< 0, 0, 0, 1, 1, 1, 1 > type;
		};

		/**
		 * Access Mapping Function (AMF) maps a logical matrix coordinates (i, j)
		 * to a corresponding matrix element's location in the physical container.
		 * AMF take into account the index mapping function associated to each
		 * coordinate (rows and columns) and the storage mapping function that
		 * maps logical coordinates to physical ones.
		 *
		 * For certain combinations of IMFs and SMFs it is possible to fuse the
		 * index computation in a single function call.
		 */
		template< typename ImfR, typename ImfC, enum Scheme storage >
		class AMF {

			template< typename T, typename Structure, enum Density density, typename View, typename ImfRT, typename ImfCT, enum Backend backend >
			friend class alp::Matrix;

			private:

				const ImfR imf_r;
				const ImfC imf_c;
				typedef typename SMF< storage >::type smf_type;
				const smf_type smf;

			public:

				AMF( ImfR &&imf_r, ImfC &&imf_c, smf_type smf ) : imf_r( imf_r ), imf_c( imf_c ), smf( smf ) {}

				/**
				 * Returns dimensions of the logical layout of the associated container.
				 *
				 * @return  A pair of two values, number of rows and columns, respectively.
				 */
				std::pair< size_t, size_t> getLogicalDimensions() const {
					return std::make_pair( imf_r.n, imf_c.n );
				}

				/**
				 * Returns dimensions of the physical layout of the associated container.
				 *
				 * @return  The size of the physical container.
				 */
				std::size_t getStorageDimensions() const {
					return 0;
				}

				/**
				 * Returns a storage index based on the coordinates in the
				 * logical iteration space.
				 *
				 * @param[in] i  row-coordinate
				 * @param[in] j  column-coordinate
				 * @param[in] s  current process ID
				 * @param[in] P  total number of processes
				 *
				 * @return  storage index corresponding to the provided logical
				 *          coordinates and parameters s and P.
				 */
				size_t getStorageIndex( const size_t i, const size_t j, const size_t s, const size_t P ) const {
#ifdef _DEBUG
					std::cout << "Calling AMF::getStorageIndex()\n";
#endif
					(void)s;
					(void)P;
					return smf.f( imf_r.map( i ), imf_c.map( j ) );
				}

				/**
				 * Returns coordinates in the logical iteration space based on
				 * the storage index.
				 *
				 * @param[in] storageIndex  storage index in the physical
				 *                          iteration space
				 * @param[in] s             current process ID
				 * @param[in] P             total number of processes
				 *
				 * @return  a pair of row- and column-coordinates in the
				 *          logical iteration space.
				 */
				std::pair< size_t, size_t > getCoords( const size_t storageIndex, const size_t s, const size_t P ) const;
		};

		template< enum Scheme storage >
		class AMF< imf::Strided, imf::Strided, storage > {

			template< typename T, typename Structure, enum Density density, typename View, typename ImfRT, typename ImfCT, enum Backend backend >
			friend class alp::Matrix;

			private:

				/** For size checks */
				const imf::Strided imf_r;
				const imf::Strided imf_c;
				typedef typename SMF< storage >::type smf_type;
				const smf_type smf;
				typedef typename Composition< imf::Strided, imf::Strided, storage >::type Composition_t;
				const Composition_t amf;

				Composition_t fusion(
					const imf::Strided &imf_r,
					const imf::Strided &imf_c,
					const smf_type &smf
				) const {
					return Composition_t(
						smf.ax2 * imf_r.s * imf_r.s, // ax2 ( for x^2 )
						smf.ay2 * imf_c.s * imf_c.s, // ay2 ( for y*2 )
						smf.axy * imf_r.s * imf_c.s, // axy ( for x * y )
						imf_r.s * ( 2 * smf.ax2 * imf_r.b + smf.axy * imf_c.b + smf.ax ), // ax ( for x )
						imf_c.s * ( 2 * smf.ay2 * imf_c.b + smf.axy * imf_r.b + smf.ay ), // ay ( for y )
						smf.ax2 * imf_r.b * imf_r.b + smf.ay2 * imf_c.b * imf_c.b +
						smf.axy * imf_r.b * imf_c.b + smf.ax * imf_r.b + smf.ay * imf_c.b + smf.a0 // a0
					);
				}

			public:

				AMF( const imf::Strided &imf_r, const imf::Strided &imf_c, const smf_type &smf ) :
					imf_r( imf_r ), imf_c( imf_c ), smf( smf ), amf( fusion( imf_r, imf_c, smf ) ) {
				}

				std::pair< size_t, size_t> getLogicalDimensions() const {
					return std::make_pair( imf_r.n, imf_c.n );
				}

				std::size_t getStorageDimensions() const {
					return SMF< storage >::getStorageDimensions( imf_r.n, imf_c.n );
				}

				size_t getStorageIndex( const size_t i, const size_t j, const size_t s, const size_t P ) const {
#ifdef _DEBUG
					std::cout << "Calling AMF::getStorageIndex()\n";
#endif
					// TODO: Maybe these asserts should be pushed to debug mode
					// for performance reasons.
					(void)s;
					(void)P;
					assert( i < imf_r.n );
					assert( j < imf_c.n );
					return amf.f( i, j );
				}

				std::pair< size_t, size_t > getCoords( const size_t storageIndex, const size_t s, const size_t P ) const;

		}; // class AMF< imf::Strided, imf::Strided, storage >

	}; // namespace storage

} // namespace alp

#endif // _H_ALP_SMF
