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

		/**
		 * AMF for parallel shared memory backend.
		 */
		template< typename ImfR, typename ImfC >
		class OMP_AMF {

			//friend class AMFFactory;

			public:

				/** Expose static properties */
				typedef ImfR imf_r_type;
				typedef ImfC imf_c_type;

			private:

				const imf_r_type imf_r;
				const imf_c_type imf_c;

				AMF( ImfR imf_r, ImfC imf_c ) :
					imf_r( imf_r ), imf_c( imf_c ) {}

				AMF( const AMF & ) = delete;
				AMF &operator=( const AMF & ) = delete;

			public:

				AMF( AMF &&amf ) :
					imf_r( std::move( amf.imf_r ) ),
					imf_c( std::move( amf.imf_c ) ) {}

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
					return -1;
					//return storage_dimensions;
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
				 */
				size_t getStorageIndex( const size_t i, const size_t j, const size_t s, const size_t P ) const {
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

		}; // class AMF

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

} // namespace alp

#endif // _H_ALP_OMP_STORAGE
