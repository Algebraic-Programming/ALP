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
 * This file registers mechanisms for coordinate mapping between
 * logical and physical iteration spaces.
 *
 */

#ifndef _H_ALP_SMF
#define _H_ALP_SMF

#include <memory>
#include "structures.hpp"

namespace alp {

	namespace storage {

		enum StorageOrientation {
			ROW_WISE,
			COLUMN_WISE
		};

		enum StoredPart {
			UPPER,
			LOWER
		};

		/**
		 * The namespace containts polynomials used to map coordinates
		 * between logical and physical iteration spaces,
		 * associated type traits and helper classes.
		 */
		namespace polynomials {

			/**
			 * Implements the polynomial
			 * ( A*a*x^2 + B*b*y^2 + C*c*x*y + D*d*x + E*e*y + F*f ) / Denominator
			 * where uppercase coefficients are compile-time constant,
			 * lowercase coefficients are run-time constant,
			 * and x and y are variables.
			 * All coefficients and variables are integers and all operations are integer
			 * operations.
			 *
			 * The purpose of compile-time constant coefficients is to allow compile-time
			 * optimizations for zero terms/monomials.
			 *
			 * Denominator allows for implementaiton of polynomials with integer division,
			 * e.g., n * ( n + 1 ) / 2,
			 * while avoiding the need for floating point coefficients and operations.
			 *
			 * @tparam Ax2  Static coefficient corresponding to x^2
			 * @tparam Ay2  Static coefficient corresponding to y^2
			 * @tparam Axy  Static coefficient corresponding to x*y
			 * @tparam Ax   Static coefficient corresponding to x
			 * @tparam Ay   Static coefficient corresponding to y
			 * @tparam A0   Static coefficient corresponding to constant term
			 * @tparam Denominator  Static denominator dividing the whole polynomial
			 */
			template<
				size_t coeffAx2, size_t coeffAy2, size_t coeffAxy,
				size_t coeffAx, size_t coeffAy,
				size_t coeffA0,
				size_t Denominator
			>
			struct BivariateQuadratic {

				static_assert( Denominator != 0, "Denominator cannot be zero (division by zero).");
				typedef int64_t dyn_coef_t;

				static constexpr size_t Ax2 = coeffAx2;
				static constexpr size_t Ay2 = coeffAy2;
				static constexpr size_t Axy = coeffAxy;
				static constexpr size_t Ax  = coeffAx;
				static constexpr size_t Ay  = coeffAy;
				static constexpr size_t A0  = coeffA0;
				static constexpr size_t D   = Denominator;
				const dyn_coef_t ax2, ay2, axy, ax, ay, a0;

				BivariateQuadratic(
					const dyn_coef_t ax2, const dyn_coef_t ay2, const dyn_coef_t axy,
					const dyn_coef_t ax, const dyn_coef_t ay,
					const dyn_coef_t a0 ) :
					ax2( ax2 ), ay2( ay2 ), axy( axy ),
					ax( ax ), ay( ay ),
					a0( a0 ) {}

				size_t evaluate( const size_t x, const size_t y ) const {
					return (Ax2 * ax2 * x * x +
						Ay2 * ay2 * y * y +
						Axy * axy * x * y +
						Ax * ax * x +
						Ay * ay * y +
						A0 * a0) / D;
				}

			}; // BivariateQuadratic

			/** \internal Defines the interface implemented by other polynomial factories */
			struct AbstractFactory {

				/** \internal Defines the type of the polynomial returned by Create */
				typedef BivariateQuadratic< 0, 0, 0, 0, 0, 0, 1 > poly_type;

				/** \internal Instantiates a polynomial */
				static poly_type Create( const size_t rows, const size_t cols );

				/** \internal Returns the size of storage associated with the defined polynomial */
				static size_t GetStorageDimensions( const size_t rows, const size_t cols );

			}; // struct AbstractFactory

			/** p(i,j) = 0 */
			struct NoneFactory {

				typedef BivariateQuadratic< 0, 0, 0, 0, 0, 0, 1 > poly_type;

				static poly_type Create( const size_t rows, const size_t cols ) {
					(void) rows;
					(void) cols;
					return poly_type( 0, 0, 0, 0, 0, 0 );
				}

				static size_t GetStorageDimensions( const size_t rows, const size_t cols ) {
					(void) rows;
					(void) cols;
					return 0;
				}
			}; // struct NoneFactory

			/** p(i,j) = Ni + j */
			template< bool row_major = true >
			struct FullFactory {

				typedef BivariateQuadratic< 0, 0, 0, 1, 1, 0, 1 > poly_type;

				static poly_type Create( const size_t rows, const size_t cols ) {
					if( row_major ){
						return poly_type( 0, 0, 0, cols, 1, 0 );
					} else {
						return poly_type( 0, 0, 0, 1, rows, 0 );
					}
				}

				static size_t GetStorageDimensions( const size_t rows, const size_t cols ) {
					return rows * cols;
				}
			}; // struct FullFactory

			/** Implements packed, triangle-like storage */
			template< enum StoredPart stored_part, enum StorageOrientation orientation >
			struct PackedFactory;

			/** p(i,j) = (-i^2 + (2N - 1)i + 2j) / 2 */
			template<>
			struct PackedFactory< UPPER, ROW_WISE > {

				typedef BivariateQuadratic< 1, 0, 0, 1, 2, 0, 2 > poly_type;

				static poly_type Create( const size_t rows, const size_t cols ) {
#ifndef DEBUG
					(void) cols;
					(void) rows;
#endif
					assert( rows == cols );
					return poly_type( -1, 0, 0, 2 * cols - 1, 1, 0 );
				}

				static size_t GetStorageDimensions( const size_t rows, const size_t cols ) {
#ifndef DEBUG
					(void) cols;
#endif
					assert( rows == cols );
					return rows * ( rows + 1 ) / 2;
				}
			};

			/** p(i,j) = (j^2 + 2i + j) / 2 */
			template<>
			struct PackedFactory< UPPER, COLUMN_WISE > {

				typedef BivariateQuadratic< 0, 1, 0, 2, 1, 0, 2 > poly_type;

				static poly_type Create( const size_t rows, const size_t cols ) {
#ifndef DEBUG
					(void) cols;
					(void) rows;
#endif
					assert( rows == cols );
					return poly_type( 0, 1, 0, 1, 1, 0 );
				}

				static size_t GetStorageDimensions( const size_t rows, const size_t cols ) {
#ifndef DEBUG
					(void) cols;
#endif
					assert( rows == cols );
					return rows * ( rows + 1 ) / 2;
				}
			}; // struct PackedFactory

			/** p(i,j) = (i^2 + i + 2j) / 2 */
			template<>
			struct PackedFactory< LOWER, ROW_WISE > {

				typedef BivariateQuadratic< 1, 0, 0, 1, 2, 0, 2 > poly_type;

				static poly_type Create( const size_t rows, const size_t cols ) {
#ifndef DEBUG
					(void) cols;
					(void) rows;
#endif
					assert( rows == cols );
					return poly_type( 1, 0, 0, 1, 1, 0 );
				}

				static size_t GetStorageDimensions( const size_t rows, const size_t cols ) {
#ifndef DEBUG
					(void) cols;
#endif
					assert( rows == cols );
					return rows * ( rows + 1 ) / 2;
				}
			}; // struct PackedFactory

			/** p(i,j) = (-j^2 + 2i + (2M - 1)j) / 2 */
			template<>
			struct PackedFactory< LOWER, COLUMN_WISE > {

				typedef BivariateQuadratic< 0, 1, 0, 2, 1, 0, 2 > poly_type;

				static poly_type Create( const size_t rows, const size_t cols ) {
#ifndef DEBUG
					(void) rows;
					(void) cols;
#endif
					assert( rows == cols );
					return poly_type( 0, -1, 0, 1, 2 * rows - 1, 0 );
				}

				static size_t GetStorageDimensions( const size_t rows, const size_t cols ) {
#ifndef DEBUG
					(void) cols;
#endif
					assert( rows == cols );
					return rows * ( rows + 1 ) / 2;
				}
			};

			template< size_t l, size_t u, bool row_wise >
			struct BandFactory {

				typedef BivariateQuadratic< 0, 0, 0, 0, 0, 0, 1 > poly_type;

				static poly_type Create( const size_t rows, const size_t cols ) {
					(void) rows;
					(void) cols;
					throw std::runtime_error( "Needs an implementation." );
				}

				static size_t GetStorageDimensions( const size_t rows, const size_t cols ) {
					(void) rows;
					(void) cols;
					throw std::runtime_error( "Needs an implementation." );
				}
			}; // struct BandFactory

			struct ArrayFactory {
				/** p(i,j) = i */
				typedef BivariateQuadratic< 0, 0, 0, 1, 0, 0, 1 > poly_type;

				static poly_type Create( const size_t rows, const size_t cols ) {
					(void) rows;
					(void) cols;
					return poly_type( 0, 0, 0, 1, 0, 0 );
				}

				static size_t GetStorageDimensions( const size_t rows, const size_t cols ) {
					assert( ( rows == 1 ) || ( cols == 1 ) );
					return rows * cols;
				}
			};

			template< enum view::Views view, typename Polynomial >
			struct apply_view {};

			template< typename Polynomial >
			struct apply_view< view::original, Polynomial > {
				typedef Polynomial type;
			};

			template< typename Polynomial >
			struct apply_view< view::transpose, Polynomial > {
				typedef BivariateQuadratic< Polynomial::Ay2, Polynomial::Ax2, Polynomial::Axy, Polynomial::Ay, Polynomial::Ax, Polynomial::A0, Polynomial::D > type;
			};

			template< typename Polynomial >
			struct apply_view< view::diagonal, Polynomial > {
				typedef Polynomial type;
			};

			template< typename Polynomial >
			struct apply_view< view::_internal, Polynomial > {
				typedef typename NoneFactory::poly_type type;
			};

			/**
			 * Specifies the resulting IMF and Polynomial types after fusing
			 * the provided IMF and Polynomial and provides two factory methods
			 * to create the IMF and the Polynomial of the resulting types.
			 * In the general case, the fusion does not happen and the resulting
			 * types are equal to the provided types.
			 */
			template< typename Imf, typename Poly >
			struct fuse_on_i {

				typedef Imf resulting_imf_type;
				typedef Poly resulting_polynomial_type;

				static resulting_imf_type CreateImf( Imf imf ) {
					return imf;
				}

				static resulting_polynomial_type CreatePolynomial( Imf imf, Poly p ) {
					(void) imf;
					return p;
				}
			};

			/**
			 * Specialization for Id IMF.
			 */
			template< typename Poly >
			struct fuse_on_i< imf::Id, Poly > {

				/** The resulting IMF is an Id because strided IMF is fully fused into the polynomial */
				typedef imf::Id resulting_imf_type;

				/** Some static factors change after injecting strided IMF into the polynomial */
				typedef Poly resulting_polynomial_type;

				static resulting_imf_type CreateImf( imf::Id imf ) {
					return imf::Id( imf.n );
				}

				static resulting_polynomial_type CreatePolynomial( imf::Id imf, Poly p ) {
					(void)imf;
					return p;
				}
			};

			/**
			 * Specialization for strided IMF.
			 */
			template< typename Poly >
			struct fuse_on_i< imf::Strided, Poly> {

				/** The resulting IMF is an Id because strided IMF is fully fused into the polynomial */
				typedef imf::Id resulting_imf_type;

				/** Some static factors change after injecting strided IMF into the polynomial */
				typedef BivariateQuadratic<
					Poly::Ax2, Poly::Ay2, Poly::Axy,
					Poly::Ax2 || Poly::Ax, Poly::Axy || Poly::Ay,
					Poly::Ax2 || Poly::Ax || Poly::A0,
					Poly::D
				> resulting_polynomial_type;

				static resulting_imf_type CreateImf( imf::Strided imf ) {
					return imf::Id( imf.n );
				}

				static resulting_polynomial_type CreatePolynomial( imf::Strided imf, Poly p ) {
					return resulting_polynomial_type(
						p.ax2 * imf.s * imf.s, // ax2
						p.ay2,                 // ay2
						p.axy * imf.s,         // axy
						2 * Poly::Ax2 * p.ax2 * imf.s * imf.b + Poly::Ax * p.ax * imf.s, // ax
						Poly::Ay * p.ay + Poly::Axy * p.axy * imf.b,                     // ay
						Poly::Ax2 * p.ax2 * imf.b * imf.b + Poly::Ax * p.ax * imf.b + Poly::A0 * p.a0 // A0
					);
				}
			};

			/**
			 * Specialization for zero IMF.
			 */
			template< typename Poly >
			struct fuse_on_i< imf::Zero, Poly> {

				/** The resulting IMF is an Id because strided IMF is fully fused into the polynomial */
				typedef imf::Id resulting_imf_type;

				/** Some static factors change after injecting strided IMF into the polynomial */
				typedef BivariateQuadratic<
					0, Poly::Ay2, 0,
					0, Poly::Ay,
					Poly::A0,
					Poly::D
				> resulting_polynomial_type;

				static resulting_imf_type CreateImf( imf::Zero imf ) {
					return imf::Id( imf.n );
				}

				static resulting_polynomial_type CreatePolynomial( imf::Zero imf, Poly p ) {
					(void)imf;
					return resulting_polynomial_type(
						0,     // ax2
						p.ay2, // ay2
						0,     // axy
						0,     // ax
						p.ay,  // ay
						p.a0   // A0
					);
				}
			};

			template< typename Imf, typename Poly >
			struct fuse_on_j {

				typedef Imf resulting_imf_type;
				typedef Poly resulting_polynomial_type;

				static resulting_imf_type CreateImf( Imf imf ) {
					return imf;
				}

				static resulting_polynomial_type CreatePolynomial( Imf imf, Poly p ) {
					(void) imf;
					return p;
				}
			};

			/**
			 * Specialization for Id IMF.
			 */
			template< typename Poly >
			struct fuse_on_j< imf::Id, Poly > {

				/** The resulting IMF is an Id because strided IMF is fully fused into the polynomial */
				typedef imf::Id resulting_imf_type;

				/** Some static factors change after injecting strided IMF into the polynomial */
				typedef Poly resulting_polynomial_type;

				static resulting_imf_type CreateImf( imf::Id imf ) {
					return imf::Id( imf.n );
				}

				static resulting_polynomial_type CreatePolynomial( imf::Id imf, Poly p ) {
					(void)imf;
					return p;
				}
			};

			/**
			 * Specialization for strided IMF.
			 */
			template< typename Poly >
			struct fuse_on_j< imf::Strided, Poly > {

				/** The resulting IMF is an Id because strided IMF is fully fused into the polynomial */
				typedef imf::Id resulting_imf_type;

				/** Some static factors change after injecting strided IMF into the polynomial */
				typedef BivariateQuadratic<
					Poly::Ax2, Poly::Ay2, Poly::Axy,
					Poly::Axy || Poly::Ax, Poly::Ay2 || Poly::Ay,
					Poly::Ay2 || Poly::Ay || Poly::A0,
					Poly::D
				> resulting_polynomial_type;

				static resulting_imf_type CreateImf( imf::Strided imf ) {
					return imf::Id( imf.n );
				}

				static resulting_polynomial_type CreatePolynomial( imf::Strided imf, Poly p ) {
					return resulting_polynomial_type(
						p.ax2,                 // ax2
						p.ay2 * imf.s * imf.s, // ay2
						p.axy * imf.s,         // axy
						Poly::Ax * p.ax + Poly::Axy * p.axy * imf.b,                     // ax
						2 * Poly::Ay2 * p.ay2 * imf.s * imf.b + Poly::Ay * p.ay * imf.s, // ay
						Poly::Ay2 * p.ay2 * imf.b * imf.b + Poly::Ay * p.ay * imf.b + Poly::A0 * p.a0 // A0
					);
				}
			};

			/**
			 * Specialization for constant-mapping IMF.
			 */
			template< typename Poly >
			struct fuse_on_j< imf::Constant, Poly > {

				/** The resulting IMF is an Id because strided IMF is fully fused into the polynomial */
				typedef imf::Id resulting_imf_type;

				/** j factors contribute to the constant factor, while they become 0 */
				typedef BivariateQuadratic<
					Poly::Ax2, 0, 0,
					Poly::Ax || Poly::Axy, 0,
					Poly::A0 || Poly::Ay || Poly::Ay2,
					Poly::D
				> resulting_polynomial_type;

				static resulting_imf_type CreateImf( imf::Constant imf ) {
					return imf::Id( imf.n );
				}

				static resulting_polynomial_type CreatePolynomial( imf::Constant imf, Poly p ) {
					return resulting_polynomial_type(
						p.ax2,         // ax2
						0,             // ay2
						0,             // axy
						Poly::Ax * p.ax +
						Poly::Axy * p.axy * imf.b, // ax
						0,             // ay
						Poly::A0 * p.a0 +
						Poly::Ay * p.ay * imf.b +
						Poly::Ay2 * p.ay2 * imf.b * imf.b  // A0
					);
				}
			};

			/**
			 * Specialization for zero IMF.
			 */
			template< typename Poly >
			struct fuse_on_j< imf::Zero, Poly > {

				/** The resulting IMF is an Id because strided IMF is fully fused into the polynomial */
				typedef imf::Id resulting_imf_type;

				/** Some static factors change after injecting strided IMF into the polynomial */
				typedef BivariateQuadratic<
					Poly::Ax2, 0, 0,
					Poly::Ax, 0,
					Poly::A0,
					Poly::D
				> resulting_polynomial_type;

				static resulting_imf_type CreateImf( imf::Zero imf ) {
					return imf::Id( imf.n );
				}

				static resulting_polynomial_type CreatePolynomial( imf::Zero imf, Poly p ) {
					(void)imf;
					return resulting_polynomial_type(
						p.ax2, // ax2
						0,     // ay2
						0,     // axy
						p.ax,  // ax
						0,     // ay
						p.a0   // A0
					);
				}
			};

		}; // namespace polynomials

		/**
		 * Access Mapping Function (AMF) maps logical matrix coordinates (i, j)
		 * to the corresponding matrix element's location in the physical container.
		 *
		 * To calculate the mapping, the AMF first applies logical-to-logical
		 * mapping provided by one IMF per coordinate (row and column).
		 * A bivariate polynomial (called mapping polynomial) takes these two
		 * output coordinates as inputs to calculate the position is physical
		 * storage of the requested element (logical-to-physical mapping).
		 *
		 * For certain combinations of IMFs and mapping polynomial types it is
		 * possible to fuse the index computation into a single function call.
		 * AMF specializations for such IMF and polynomial types are free to do
		 * any optimizations.
		 *
		 * All AMF specializations shall expose the effective types of the IMFs
		 * and the mapping polynomial, since these may change after the fusion.
		 */
		template< typename ImfR, typename ImfC, typename MappingPolynomial >
		class AMF {

			friend class AMFFactory;

			public:

				/** Expose static properties */
				typedef ImfR imf_r_type;
				typedef ImfC imf_c_type;
				typedef MappingPolynomial mapping_polynomial_type;

			private:

				const imf_r_type imf_r;
				const imf_c_type imf_c;
				const mapping_polynomial_type map_poly;
				const size_t storage_dimensions;

				AMF( ImfR imf_r, ImfC imf_c, MappingPolynomial map_poly, const size_t storage_dimensions ) :
					imf_r( imf_r ), imf_c( imf_c ), map_poly( map_poly ), storage_dimensions( storage_dimensions ) {}

				AMF( const AMF & ) = delete;
				AMF &operator=( const AMF & ) = delete;

			public:

				AMF( AMF &&amf ) :
					imf_r( std::move( amf.imf_r ) ),
					imf_c( std::move( amf.imf_c ) ),
					map_poly( std::move( amf.map_poly ) ),
					storage_dimensions( std::move( amf.storage_dimensions ) ) {}

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
				 * @brief Returns a storage index based on the coordinates in the
				 *        logical iteration space.
				 *
				 * @tparam R  ImfR type
				 * @tparam C  ImfC type
				 *
				 * @param[in] i  row-coordinate
				 * @param[in] j  column-coordinate
				 * @param[in] s  current process ID
				 * @param[in] P  total number of processes
				 *
				 * @return  storage index corresponding to the provided logical
				 *          coordinates and parameters s and P.
				 *
				 * \note It is not necessary to call imf.map() function if the imf
				 *       has the type imf::Id. To implement SFINAE-driven selection
				 *       of the getStorageIndex, dummy parameters R and C are added.
				 *       They are set to the ImfR and ImfC by default and a static
				 *       assert ensures that external caller does not force a call
				 *       to wrong implementation by explicitly specifying values
				 *       for R and/or C.
				 *
				 */
				template<
					typename R = ImfR, typename C = ImfC,
					std::enable_if_t< !std::is_same< R, imf::Id >::value && !std::is_same< C, imf::Id >::value > * = nullptr
				>
				size_t getStorageIndex( const size_t i, const size_t j, const size_t s, const size_t P ) const {
					static_assert(
						std::is_same< R, ImfR >::value && std::is_same< C, ImfC >::value,
						"Explicit specialization of getStorageIndex is not allowed."
					);
					(void)s;
					(void)P;
					return map_poly.evaluate( imf_r.map( i ), imf_c.map( j ) );
				}

				template<
					typename R = ImfR, typename C = ImfC,
					std::enable_if_t< std::is_same< R, imf::Id >::value && !std::is_same< C, imf::Id >::value > * = nullptr
				>
				size_t getStorageIndex( const size_t i, const size_t j, const size_t s, const size_t P ) const {
					static_assert(
						std::is_same< R, ImfR >::value && std::is_same< C, ImfC >::value,
						"Explicit specialization of getStorageIndex is not allowed."
					);
					(void)s;
					(void)P;
					return map_poly.evaluate( i, imf_c.map( j ) );
				}

				template<
					typename R = ImfR, typename C = ImfC,
					std::enable_if_t< !std::is_same< R, imf::Id >::value && std::is_same< C, imf::Id >::value > * = nullptr
				>
				size_t getStorageIndex( const size_t i, const size_t j, const size_t s, const size_t P ) const {
					static_assert(
						std::is_same< R, ImfR >::value && std::is_same< C, ImfC >::value,
						"Explicit specialization of getStorageIndex is not allowed."
					);
					(void)s;
					(void)P;
					return map_poly.evaluate( imf_r.map( i ), j );
				}

				template<
					typename R = ImfR, typename C = ImfC,
					std::enable_if_t< std::is_same< R, imf::Id >::value && std::is_same< C, imf::Id >::value > * = nullptr
				>
				size_t getStorageIndex( const size_t i, const size_t j, const size_t s, const size_t P ) const {
					static_assert(
						std::is_same< R, ImfR >::value && std::is_same< C, ImfC >::value,
						"Explicit specialization of getStorageIndex is not allowed."
					);
					(void)s;
					(void)P;
					return map_poly.evaluate( i, j );
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

		}; // class AMF

		/**
		 * @brief Collects AMF factory classes
		 */
		struct AMFFactory {

			/**
			 * @brief Transforms the provided AMF by applying the gather view
			 *        represented by the given row and column IMFs
			 *
			 * Exposes the type of the resulting AMF and implements a factory
			 * method that creates objects of such type.
			 * The IMFs and the AMF may be fusedi (simplified), depending on
			 * the properties of the IMFs. For example, static IMFs (e.g. Id,
			 * Strided) are easily fused into the mapping polynomial.
			 *
			 * Fusion of the IMFs into the mapping polynomial results in a
			 * reduced amount of function calls and, potentially, less computation,
			 * on each call to the map function of the AMF. This is especially
			 * beneficial for longer chains of views.
			 *
			 * Creation of the new AMF is done in following order:
			 *   - view row IMF and target row IMF are composed
			 *   - view col IMF and target col IMF are composed
			 *   - composed row IMF is fused into the target Poly, if possible,
			 *     yielding the new intermediate polynomial
			 *   - composed col IMF is fused, if possible, into the intermediate
			 *     polynomial, obtained above. This results in the final fused
			 *     polynomial.
			 *
			 * @tparam view       The enum value of the desired view type.
			 * @tparam SourceAMF  The type of the target AMF
			 *
			 */
			template< typename ViewImfR, typename ViewImfC, typename SourceAMF >
			struct Compose {

				private:

					/** Extract target IMF and polynomial types */
					typedef typename SourceAMF::imf_r_type SourceImfR;
					typedef typename SourceAMF::imf_c_type SourceImfC;
					typedef typename SourceAMF::mapping_polynomial_type SourcePoly;

					/** Compose row and column IMFs */
					typedef typename imf::ComposedFactory< SourceImfR, ViewImfR >::type composed_imf_r_type;
					typedef typename imf::ComposedFactory< SourceImfC, ViewImfC >::type composed_imf_c_type;

					/** Fuse composed row IMF into the target polynomial */
					typedef typename polynomials::fuse_on_i<
						composed_imf_r_type,
						SourcePoly
					> fused_row;

					/** Fuse composed column IMF into the intermediate polynomial obtained above */
					typedef typename polynomials::fuse_on_j<
						composed_imf_c_type,
						typename fused_row::resulting_polynomial_type
					> fused_row_col;

					typedef typename fused_row::resulting_imf_type final_imf_r_type;
					typedef typename fused_row_col::resulting_imf_type final_imf_c_type;
					typedef typename fused_row_col::resulting_polynomial_type final_polynomial_type;

				public:

					typedef AMF< final_imf_r_type, final_imf_c_type, final_polynomial_type > amf_type;

					static amf_type Create( ViewImfR imf_r, ViewImfC imf_c, const AMF< SourceImfR, SourceImfC, SourcePoly > &amf ) {
						composed_imf_r_type composed_imf_r = imf::ComposedFactory< SourceImfR, ViewImfR >::create( amf.imf_r, imf_r );
						composed_imf_c_type composed_imf_c = imf::ComposedFactory< SourceImfC, ViewImfC >::create( amf.imf_c, imf_c );
						return amf_type(
							fused_row::CreateImf( composed_imf_r ),
							fused_row_col::CreateImf( composed_imf_c ),
							fused_row_col::CreatePolynomial(
								composed_imf_c,
								fused_row::CreatePolynomial( composed_imf_r, amf.map_poly )
							),
							amf.storage_dimensions
						);
					}

					Compose() = delete;

			}; // class Compose

			/**
			 * @brief Describes an AMF for a container that requires allocation
			 *        and exposes the AMFs type and a factory method to create it.
			 *
			 * A container that requires allocation is accompanied by Id IMFs for
			 * both row and column dimensions and the provided mapping polynomial.
			 *
			 * @tparam PolyType  Type of the mapping polynomial.
			 *
			 */
			template< typename PolyFactory >
			struct FromPolynomial {

				typedef AMF< imf::Id, imf::Id, typename PolyFactory::poly_type > amf_type;

				/**
				 * Factory method used by 2D containers.
				 *
				 * @param[in] imf_r               Row IMF
				 * @param[in] imf_c               Column IMF
				 * @param[in] poly                Mapping polynomial
				 * @param[in] storage_dimensions  Size of the allocated storage
				 *
				 * @return  An AMF object of the type \a amf_type
				 *
				 */
				static amf_type Create( imf::Id imf_r, imf::Id imf_c ) {
					return amf_type( imf_r, imf_c, PolyFactory::Create( imf_r.n, imf_c.n ), PolyFactory::GetStorageDimensions( imf_r.n, imf_c.n ) );
				}

				/**
				 * Factory method used by 1D containers.
				 *
				 * Exploits the fact that fusion of strided IMFs into the polynomial
				 * always succeeds and results in Id IMFs. As a result, the
				 * constructed AMF is of the type \a amf_type.
				 *
				 * @param[in] imf_r               Row IMF
				 * @param[in] imf_c               Column IMF
				 * @param[in] poly                Mapping polynomial
				 * @param[in] storage_dimensions  Size of the allocated storage
				 *
				 * @return  An AMF object of the type \a amf_type
				 *
				 * \note \internal To exploit existing mechanism for IMF fusion
				 *                 into the polynomial, this method creates a
				 *                 dummy AMF out of two Id IMFs and the provided
				 *                 polynomial and composes the provided Strided
				 *                 IMFs with the dummy AMF.
				 */
				static amf_type Create( imf::Id imf_r, imf::Zero imf_c ) {

					/**
					 * Ensure that the assumptions do not break upon potential
					 * future changes to AMFFactory::Compose.
					 */
					static_assert(
						std::is_same<
							amf_type,
							typename Compose< imf::Id, imf::Zero, AMF< imf::Id, imf::Id, typename PolyFactory::poly_type > >::amf_type
						>::value,
						"The factory method returns the object of different type than declared. This is a bug."
					);
					return Compose< imf::Id, imf::Zero, AMF< imf::Id, imf::Id, typename PolyFactory::poly_type > >::Create(
						imf_r, imf_c,
						FromPolynomial< PolyFactory >::Create( imf::Id( imf_r.N ), imf::Id( imf_c.N ) )
					);
				}

				FromPolynomial() = delete;

			}; // class FromPolynomial

			/**
			 * @brief Transforms the provided AMF by applying the provided View type.
			 *
			 * Exposes the type of the resulting AMF and implements a factory
			 * method that creates objects of such type.
			 *
			 * @tparam view       The enum value of the desired view type.
			 * @tparam SourceAMF  The type of the target AMF
			 *
			 */
			template< enum view::Views view, typename SourceAMF >
			struct Reshape {

				typedef SourceAMF amf_type;

				static amf_type Create( const SourceAMF &amf ) {
					throw std::invalid_argument( "Not implemented for the provided view type." );
					return amf;
				}

				Reshape() = delete;

			}; // class Reshape

			template< typename SourceAMF >
			struct Reshape< view::original, SourceAMF > {

				typedef SourceAMF amf_type;

				static amf_type Create( const SourceAMF &amf ) {
					return amf_type( amf.imf_r, amf.imf_c, amf.map_poly, amf.storage_dimensions );
				}

				Reshape() = delete;

			}; // class Reshape< original, ... >

			template< typename SourceAMF >
			struct Reshape< view::transpose, SourceAMF > {

				typedef AMF<
					typename SourceAMF::imf_c_type,
					typename SourceAMF::imf_r_type,
					typename polynomials::apply_view<
						view::transpose,
						typename SourceAMF::mapping_polynomial_type
					>::type
				> amf_type;

				static amf_type Create( const SourceAMF &amf ) {
					typedef typename polynomials::apply_view< view::transpose, typename SourceAMF::mapping_polynomial_type >::type new_mapping_polynomial_type;
					return AMF<
						typename SourceAMF::imf_c_type,
						typename SourceAMF::imf_r_type,
						new_mapping_polynomial_type
					>(
						amf.imf_c,
						amf.imf_r,
						new_mapping_polynomial_type(
							amf.map_poly.ay2, amf.map_poly.ax2, amf.map_poly.axy,
							amf.map_poly.ay, amf.map_poly.ax,
							amf.map_poly.a0
						),
						amf.storage_dimensions
					);
				}

				Reshape() = delete;

			}; // class Reshape< transpose, ... >

			/**
			 * Specialization for diagonal views
			 *
			 * Diagonal view is implemented by taking a square view over the matrix.
			 *
			 * \note \internal Converts a mapping polynomial from a bivariate-quadratic
			 *                 to univariate quadratic by summing j-factors into
			 *                 corresponding i-factors.
			 *                 Implicitely applies a largest possible square view by
			 *                 using Strided IMFs.
			 *
			 */
			template< typename SourceAMF >
			struct Reshape< view::diagonal, SourceAMF > {

				private:

					/** Short name of the original mapping polynomial type */
					typedef typename SourceAMF::mapping_polynomial_type orig_p;

					/** The type of the resulting polynomial */
					typedef polynomials::BivariateQuadratic<
						orig_p::Ax2 || orig_p::Ay2 || orig_p::Axy, 0, 0,
						orig_p::Ax || orig_p::Ay, 0,
						orig_p::A0, orig_p::D
					> new_poly_type;

				public:

					typedef AMF< imf::Id, imf::Zero, new_poly_type > amf_type;

					static amf_type Create( const SourceAMF &amf ) {
						assert( amf.getLogicalDimensions().first == amf.getLogicalDimensions().second );
						return amf_type(
							imf::Id( amf.getLogicalDimensions().first ),
							imf::Zero( 1 ),
							new_poly_type(
								orig_p::Ax2 * amf.map_poly.ax2 + orig_p::Ay2 * amf.map_poly.ay2 + orig_p::Axy * amf.map_poly.axy, 0, 0,
								orig_p::Ax * amf.map_poly.ax + orig_p::Ay * amf.map_poly.ay, 0,
								amf.map_poly.a0
							),
							amf.storage_dimensions
						);
					}

					Reshape() = delete;

			}; // class Reshape< diagonal, ... >

			/**
			 * Specialization for matrix views over vectors
			 *
			 * \note \internal The resulting AMF is equivalent to applying
			 *                 a composition with two ID IMFs.
			 *
			 */
			template< typename SourceAMF >
			struct Reshape< view::matrix, SourceAMF > {

				typedef typename AMFFactory::Compose< imf::Id, imf::Id, SourceAMF >::amf_type amf_type;

				static amf_type Create( const SourceAMF &amf ) {
					return storage::AMFFactory::Compose< imf::Id, imf::Id, SourceAMF >::Create(
						imf::Id( amf.getLogicalDimensions().first ),
						imf::Id( amf.getLogicalDimensions().second ),
						amf
					);
				}

				Reshape() = delete;

			}; // class Reshape< diagonal, ... >

		}; // class AMFFactory

	}; // namespace storage

} // namespace alp

#endif // _H_ALP_SMF
