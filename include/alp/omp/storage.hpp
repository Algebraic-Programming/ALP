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

#ifndef _H_ALP_OMP_STORAGE
#define _H_ALP_OMP_STORAGE

#include <alp/amf-based/storage.hpp>

namespace alp {

	namespace internal {

		/** Specialization for matrices */
		template< typename Structure >
		struct determine_poly_factory< Structure, imf::Id, imf::Id, omp > {

			typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for vectors */
		template< typename Structure >
		struct determine_poly_factory< Structure, imf::Id, imf::Zero, omp > {

			typedef storage::polynomials::ArrayFactory factory_type;
		};

	} // namespace internal


	/**
	 * Implements mapping between global and local iteration spaces
	 * for shared-memory parallel backend.
	 * The logical coordinates are represented as pair (i, j) of row and
	 * column positions within the matrix.
	 * The local coordinates are represented as (tr, tc, rt, br, bc, il, jl),
	 * where:
	 *  - tr is thread row-coordinate
	 *  - tc is thread column-coordinate
	 *  - rt is replication factor for thread coordinates
	 *  - br is block row-coordinate
	 *  - bc is block column-coordinate
	 *  - il is element's row-coordinate within its block
	 *  - jl is element's column-coordinate within its block
	 *
	 * This implementation assumes block-cyclic distribution of blocks
	 * among threads.
	 *
	 */
	class Distribution {

		public:

			/** Type encapsulating the global element coordinate. */
			struct GlobalCoord {

				const size_t i;
				const size_t j;

				GlobalCoord( const size_t i, const size_t j ) : i( i ), j( j ) {}

			};

			/** Type encapsulating the local element coordinate. */
			struct LocalCoord {

				const size_t tr;
				const size_t tc;
				const size_t rt;
				const size_t br;
				const size_t bc;
				const size_t il;
				const size_t jl;

				LocalCoord(
					const size_t tr, const size_t tc,
					const size_t rt,
					const size_t br, const size_t bc,
					const size_t il, const size_t jl
				) :
					tr( tr ), tc( tc ),
					rt( rt ),
					br( br ), bc( bc ),
					il( il ), jl( jl ) {}
	
			};

			/** Type encapsulating the global block coordinate. */
			struct GlobalBlockCoord {

				const size_t br;
				const size_t bc;

				GlobalBlockCoord(
					const size_t br, const size_t bc
				) :
					br( br ), bc( bc ) {}

			};

			/** Type encapsulating the local block coordinate. */
			struct LocalBlockCoord {

				const size_t tr;
				const size_t tc;
				const size_t rt;
				const size_t br;
				const size_t bc;

				LocalBlockCoord(
					const size_t tr, const size_t tc,
					const size_t rt,
					const size_t br, const size_t bc
				) :
					tr( tr ), tc( tc ),
					rt( rt ),
					br( br ), bc( bc ) {}

			};

		private:

			/** The row and column dimensions of thread grid */
			const size_t Tr;
			const size_t Tc;
			/** Replication factor in thread-coordinate space */
			const size_t Rt;
			/** The row and column dimensions of block grid */
			const size_t Br;
			const size_t Bc;

		public:

			Distribution(
				const size_t m, const size_t n,
				const size_t num_threads
			) :
				Tr( static_cast< size_t >( sqrt( num_threads ) ) ),
				Tc( num_threads / Tr ),
				Rt( config::REPLICATION_FACTOR_THREADS ),
				Br( static_cast< size_t >( std::ceil( static_cast< double >( m ) / config::BLOCK_ROW_DIM ) ) ),
				Bc( static_cast< size_t >( std::ceil( static_cast< double >( n ) / config::BLOCK_COL_DIM ) ) ) {

				if( num_threads != Tr * Tc ) {
					std::cerr << "Error\n";
				}
			}

			LocalBlockCoord mapBlockGlobalToLocal( const GlobalBlockCoord &g ) {
				(void) g;
				return LocalBlockCoord( 0, 0, 0, 0, 0 );
			}

			GlobalBlockCoord mapBlockLocalToGlobal( const LocalBlockCoord &l ) {
				const size_t block_id_r = l.br * Tr + l.tr;
				const size_t block_id_c = l.bc * Tc + l.tc;
				return GlobalBlockCoord( block_id_r, block_id_c );
			}

			LocalCoord mapGlobalToLocal( const GlobalCoord &g ) {
				(void) g;
				return LocalCoord( 0, 0, 0, 0, 0, 0, 0 );
			}

			GlobalCoord mapGlobalToLocal( const LocalCoord &l ) {
				(void) l;
				return GlobalCoord( 0, 0 );
			}

			/** Returns the total global amount of blocks */
			std::pair< size_t, size_t > getGlobalBlockGridDims() const {
				return { Br, Bc };
			}

			/** Returns the dimensions of the block grid associated to the given thread */
			std::pair< size_t, size_t > getLocalBlockGridDims( const size_t tr, const size_t tc ) const {
				// The LHS of the + operand covers the case
				// when the last block of threads is not full
				const size_t blocks_r = Br / Tr + ( tr < Br % Tr ? 1 : 0 );
				const size_t blocks_c = Bc / Tc + ( tc < Bc % Tc ? 1 : 0 );
				return { blocks_r, blocks_c };
			}

			/** Returns the global block coordinates based on the thread and local block coordinates */
			std::pair< size_t, size_t > getGlobalBlockCoords( const size_t tr, const size_t tc, const size_t br, const size_t bc ) const {
				const size_t global_br = br * Tr + tr % Tr;
				const size_t global_bc = bc * Tc + tc % Tc;
				return { global_br, global_bc };
			}

			size_t getGlobalBlockId( const size_t tr, const size_t tc, const size_t br, const size_t bc ) const {
				const auto global_coords = getGlobalBlockCoords( tr, tc, br, bc );
				return global_coords.first * Bc + global_coords.second;
			}

			/** Returns the size of the block given by the block id */
			size_t getBlockSize( const size_t block_id ) const {
				return 10;
			}

			std::pair< size_t, size_t > getThreadCoords( const size_t thread_id ) const {
				const size_t tr = thread_id / Tr;
				const size_t tc = thread_id % Tc;
				return { tr, tc };
			}
	};
		
	namespace storage {

		/**
		 * AMF for parallel shared memory backend.
		 *
		 * This implementation makes the following assumptions:
		 *  - all blocks use the same storage scheme, independent of their non-zero structure
		 *
		 * @tparam ImfR  Index-Mapping Function associated to row dimension.
		 * @tparam ImfC  Index-Mapping Function associated to column dimension.
		 * @tparam PolyFactory  The type of factory for storage polynomials
		 *                      used to construct polynomials for all blocks.
		 */
		template< typename ImfR, typename ImfC, typename PolyFactory >
		class AMF< ImfR, ImfC, PolyFactory, omp > {

			friend class AMFFactory< omp >;

			public:

				/** Expose static properties */
				typedef ImfR imf_r_type;
				typedef ImfC imf_c_type;
				typedef PolyFactory poly_factory_type;

				/** Expose types defined within the class */
				typedef struct StorageIndexType {

					size_t buffer_id;
					size_t offset;

					StorageIndexType( const size_t buffer_id, const size_t offset ) :
						buffer_id( buffer_id ), offset( offset ) {}

				} storage_index_type;

			private:

				const imf_r_type imf_r;
				const imf_c_type imf_c;

				constexpr static bool is_matrix = std::is_same< ImfC, imf::Id >::value;

				/**
				 * Number of threads used to initialize the associated container.
				 * This impacts the number of allocated blocks.
				 */
				const size_t num_threads;

				const Distribution distribution;

				AMF(
					ImfR imf_r,
					ImfC imf_c,
					size_t num_threads = config::OMP::threads()
				) :
					imf_r( imf_r ), imf_c( imf_c ),
					num_threads( num_threads ),
					distribution( imf_r.n, imf_c.n, num_threads ) {
					std::cout << "Entering AMF normal constructor\n";
				}

				AMF( const AMF & ) = delete;
				AMF &operator=( const AMF & ) = delete;

			public:

				AMF( AMF &&amf ) :
					imf_r( std::move( amf.imf_r ) ),
					imf_c( std::move( amf.imf_c ) ),
					num_threads( amf.num_threads ),
					distribution( std::move( amf.distribution ) ) {
					std::cout << "Entering OMP AMF move constructor\n";
				}

				const Distribution &getDistribution() const {
					return distribution;
				}

				/**
				 * Returns dimensions of the logical layout of the associated container.
				 *
				 * @return  A pair of two values, number of rows and columns, respectively.
				 */
				std::pair< size_t, size_t> getLogicalDimensions() const {
					return std::make_pair( imf_r.n, imf_c.n );
				}

				/**
				 * Returns dimensions of the physical layout of the associated container
				 * for the block given by its row and column coordinates.
				 *
				 * @return  The size of the physical container.
				 */
				std::size_t getStorageDimensions( const size_t br, const size_t bc ) const {
					//assert( br < Br );
					//assert( bc < Bc );
					size_t block_width = 0;
					size_t block_height = 0;

					//if( br != Br - 1 ) {
					//	block_height = config::BLOCK_ROW_DIM;
					//} else {
					//	// last block may be smaller than the configured block dimension
					//	block_height = imf_r.n - ( Br - 1 ) * config::BLOCK_ROW_DIM;
					//}

					//if( bc != Bc - 1 ) {
					//	if( is_matrix ) {
					//		block_width = config::BLOCK_ROW_DIM;
					//	} else {
					//		// vectors are stored as column vectors so their width is one
					//		block_width = 1;
					//	}
					//} else {
					//	block_width = imf_c.n - ( Bc - 1 ) * config::BLOCK_COL_DIM;
					//}

					//assert( block_width != 0 );
					//assert( block_height != 0 );
					return block_width * block_height;
				}

				size_t getBlockCount() const {
					//return Br * Bc;
					return 1;
				}

				/**
				 * Calculates the ID of the block corresponding to given thread
				 * and block coordinates.
				 *
				 * @param[in] tr  Thread row-coordinate
				 * @param[in] tc  Thread column-coordinate
				 * @param[in] br  Block row-coordinate
				 * @param[in] bc  Block column-coordinate
				 *
				 * @return The block ID
				 *
				 * \note This assumes row-major distribution of blocks from 2D
				 *       to 1D space.
				 *
				 */
				size_t getBlockId( const size_t tr, const size_t tc, const size_t br, const size_t bc ) const {
					//const size_t block_id_r = br * Tr + tr;
					//const size_t block_id_c = bc * Tc + tc;
					//return block_id_r * Bc + block_id_c;
					return 0;
				}

				size_t getOffset( const size_t i, const size_t j ) const {
					return -1;
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
				storage_index_type getStorageIndex( const size_t i, const size_t j, const size_t s, const size_t P ) const {
					(void)s;
					(void)P;
					return storage_index_type( getBlockId( i, j, s, P ), getOffset( i, j, s, P ) );
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
		 * Collects AMF factory classes.
		 */
		template<>
		struct AMFFactory< omp > {

			/**
			 * Specialization of AMFFactory for shared-memory parallel backend.
			 * 
			 * @tparam ViewImfR   The type of IMF applied to the row coordinate.
			 * @tparam ViewImfC   The type of IMF applied to the column coordinate.
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

					typedef AMF< final_imf_r_type, final_imf_c_type, final_polynomial_type, omp > amf_type;

					static amf_type Create( ViewImfR imf_r, ViewImfC imf_c, const AMF< SourceImfR, SourceImfC, SourcePoly, omp > &amf ) {
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
			template< typename Structure, typename ImfR, typename ImfC >
			struct FromPolynomial {

				// Ensure compatibility of IMF types.
				// Original Matrix has imf::Id as both IMFs.
				// Original Vector has ImfR = imf::Id and ImfC = imf::Zero.
				static_assert(
					std::is_same< ImfR, imf::Id >::value &&
					( std::is_same< ImfC, imf::Id >::value || std::is_same< ImfC, imf::Zero >::value ),
					"AMF factory FromPolynomial can only be used for an original container."
				);

				typedef typename internal::determine_poly_factory< Structure, ImfR, ImfC, omp >::factory_type PolyFactory;

				typedef AMF< imf::Id, imf::Id, PolyFactory, omp > amf_type;

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
					return amf_type( imf_r, imf_c );
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
							typename Compose< imf::Id, imf::Zero, AMF< imf::Id, imf::Id, typename PolyFactory::poly_type, omp > >::amf_type
						>::value,
						"The factory method returns the object of different type than declared. This is a bug."
					);
					return Compose< imf::Id, imf::Zero, AMF< imf::Id, imf::Id, typename PolyFactory::poly_type, omp > >::Create(
						imf_r, imf_c,
						FromPolynomial< structures::General, imf::Id, imf::Zero >::Create( imf::Id( imf_r.N ), imf::Id( imf_c.N ) )
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
					>::type,
					omp
				> amf_type;

				static amf_type Create( const SourceAMF &amf ) {
					typedef typename polynomials::apply_view< view::transpose, typename SourceAMF::mapping_polynomial_type >::type new_mapping_polynomial_type;
					return AMF<
						typename SourceAMF::imf_c_type,
						typename SourceAMF::imf_r_type,
						new_mapping_polynomial_type,
						omp
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

					typedef AMF< imf::Id, imf::Zero, new_poly_type, omp > amf_type;

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

				typedef typename Compose< imf::Id, imf::Id, SourceAMF >::amf_type amf_type;

				static amf_type Create( const SourceAMF &amf ) {
					return Compose< imf::Id, imf::Id, SourceAMF >::Create(
						imf::Id( amf.getLogicalDimensions().first ),
						imf::Id( amf.getLogicalDimensions().second ),
						amf
					);
				}

				Reshape() = delete;

			}; // class Reshape< diagonal, ... >

		}; // class AMFFactory

	} // namespace storage

} // namespace alp

#endif // _H_ALP_OMP_STORAGE