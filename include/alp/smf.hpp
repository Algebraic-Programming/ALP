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

				size_t evaluate( const size_t x, const size_t y ) const {
					return (Ax2 * ax2 * x * x +
						Ay2 * ay2 * y * y +
						Axy * axy * x * y +
						Ax * ax * x +
						Ay * ay * y +
						A0 * a0) / Denominator;
				}

			}; // BivariateQuadratic

			typedef BivariateQuadratic< 0, 0, 0, 0, 0, 0, 1 > None_type;
			typedef BivariateQuadratic< 0, 0, 0, 1, 1, 0, 1 > Full_type;
			typedef BivariateQuadratic< 0, 0, 0, 0, 0, 0, 1 > Packed_type; // TODO
			typedef BivariateQuadratic< 0, 0, 0, 0, 0, 0, 1 > Band_type; // TODO

			/**
			 * Polynomial factory method
			 */
			template< typename PolynomialType, typename... Args >
			PolynomialType Create( Args... args ) {
				return PolynomialType( args... );
			}

			/** Specialization for Full storage of type i * dim + j */
			template<>
			Full_type Create< Full_type >( size_t dim ) {
				return Full_type( 0, 0, 0, dim, 1, 0 );
			}

		};


		/**
		 * Provides a type of composed Access Mapping Function
		 * expressed as a BivariateQuadratic polynomial depending
		 * on the types of the IMFs and the SMF.
		 */
		template< typename ImfR, typename ImfC, typename MappingPolynomial >
		struct Composition {
			typedef polynomials::BivariateQuadratic< 1, 1, 1, 1, 1, 1, 1 > type;
		};

		template<>
		struct Composition< imf::Strided, imf::Strided, polynomials::Full_type > {
			typedef polynomials::BivariateQuadratic< 0, 0, 0, 1, 1, 1, 1 > type;
		};

		/** Forward declaration */
		class AMFFactory;

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
		template< typename ImfR, typename ImfC, typename MappingPolynomial >
		class AMF {

			friend AMFFactory;

			private:

				const ImfR imf_r;
				const ImfC imf_c;
				const MappingPolynomial map_poly;
				const size_t storage_dimensions;

			public:

				AMF( ImfR &&imf_r, ImfC &&imf_c, MappingPolynomial map_poly, const size_t storage_dimensions ) :
					imf_r( imf_r ), imf_c( imf_c ), map_poly( map_poly ), storage_dimensions( storage_dimensions ) {}

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
					return storage_dimensions;
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
					return map_poly.evaluate( imf_r.map( i ), imf_c.map( j ) );
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

		template< typename MappingPolynomial >
		class AMF< imf::Strided, imf::Strided, MappingPolynomial > {

			friend AMFFactory;

			private:

				/** For size checks */
				const imf::Strided imf_r;
				const imf::Strided imf_c;
				const MappingPolynomial map_poly;
				typedef typename Composition< imf::Strided, imf::Strided, MappingPolynomial >::type Composition_type;
				const Composition_type amf;

				const size_t storage_dimensions;

				Composition_type fusion(
					const imf::Strided &imf_r,
					const imf::Strided &imf_c,
					const MappingPolynomial &map_poly
				) const {
					return Composition_type(
						map_poly.ax2 * imf_r.s * imf_r.s, // ax2 ( for x^2 )
						map_poly.ay2 * imf_c.s * imf_c.s, // ay2 ( for y*2 )
						map_poly.axy * imf_r.s * imf_c.s, // axy ( for x * y )
						imf_r.s * ( 2 * map_poly.ax2 * imf_r.b + map_poly.axy * imf_c.b + map_poly.ax ), // ax ( for x )
						imf_c.s * ( 2 * map_poly.ay2 * imf_c.b + map_poly.axy * imf_r.b + map_poly.ay ), // ay ( for y )
						map_poly.ax2 * imf_r.b * imf_r.b + map_poly.ay2 * imf_c.b * imf_c.b +
						map_poly.axy * imf_r.b * imf_c.b + map_poly.ax * imf_r.b + map_poly.ay * imf_c.b + map_poly.a0 // a0
					);
				}

			public:

				AMF( const imf::Strided &imf_r, const imf::Strided &imf_c, const MappingPolynomial &map_poly, const size_t storage_dimensions ) :
					imf_r( imf_r ), imf_c( imf_c ), map_poly( map_poly ), amf( fusion( imf_r, imf_c, map_poly ) ), storage_dimensions( storage_dimensions ) {
				}

				std::pair< size_t, size_t> getLogicalDimensions() const {
					return std::make_pair( imf_r.n, imf_c.n );
				}

				std::size_t getStorageDimensions() const {
					return storage_dimensions;
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
					return amf.evaluate( i, j );
				}

				std::pair< size_t, size_t > getCoords( const size_t storageIndex, const size_t s, const size_t P ) const;

		}; // class AMF< imf::Strided, imf::Strided, storage >

		class AMFFactory {

			public:

			template< typename MappingPolynomial >
			static AMF<
				imf::Id,
				imf::Id,
				MappingPolynomial
			> Create(
				imf::Id imf_r,
				imf::Id imf_c,
				MappingPolynomial map_poly,
				const size_t storage_dimensions
			) {
				return AMF< imf::Id, imf::Id, MappingPolynomial >(
					imf_r, imf_c, map_poly, storage_dimensions
				);
			}

			template<
				typename OriginalImfR, typename OriginalImfC, typename MappingPolynomial,
				typename ViewImfR, typename ViewImfC
			>
			static AMF<
				typename imf::composed_type< ViewImfR, OriginalImfR >::type,
				typename imf::composed_type< ViewImfC, OriginalImfC >::type,
				MappingPolynomial
			> Create(
				const AMF< OriginalImfR, OriginalImfC, MappingPolynomial > &original_amf,
				ViewImfR view_imf_r,
				ViewImfC view_imf_c
			) {
				return AMF<
					typename imf::composed_type< ViewImfR, OriginalImfR >::type,
					typename imf::composed_type< ViewImfC, OriginalImfC >::type,
					MappingPolynomial
				>(
					imf::ComposedFactory::create< ViewImfR, OriginalImfR >(
						view_imf_r,
						original_amf.imf_r
					),
					imf::ComposedFactory::create< ViewImfC, OriginalImfC >(
						view_imf_c,
						original_amf.imf_c
					),
					original_amf.map_poly,
					original_amf.storage_dimensions
				);
			}
		}; // class AMFFactory

	}; // namespace storage

} // namespace alp

#endif // _H_ALP_SMF
