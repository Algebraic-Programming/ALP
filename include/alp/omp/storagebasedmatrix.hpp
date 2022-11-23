
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

#ifndef _H_ALP_OMP_STORAGEBASEDMATRIX
#define _H_ALP_OMP_STORAGEBASEDMATRIX

#include <alp/backends.hpp>
#include <alp/base/matrix.hpp>
#include <alp/config.hpp>
#include <alp/ops.hpp>
#include <alp/type_traits.hpp>
#include <alp/utils.hpp>
#include <alp/storage.hpp>

namespace alp {

	namespace internal {

		/**
		 * Matrix container specialization
		 * Implements both original containers and views on containers.
		 * @tparam requires_allocation True if the class is an original container
		 *                             False if the class is a view of another matrix
		 */
		template< typename T, typename AmfType, bool requires_allocation >
		class StorageBasedMatrix< T, AmfType, requires_allocation, omp > :
			public MatrixBase< StorageBasedMatrix< T, AmfType, requires_allocation, omp > > {

			template<
				typename MatrixType,
				std::enable_if_t< internal::is_storage_based< MatrixType >::value > *
			>
			friend size_t getStorageDimensions( const MatrixType &A ) noexcept;

			/** Get the reference to the AMF of a storage-based matrix */
			template<
				typename MatrixType,
				std::enable_if< internal::is_storage_based< MatrixType >::value > *
			>
			friend const typename MatrixType::amf_type &getAmf( const MatrixType &A ) noexcept;

			public:

				/** Expose static properties */

				typedef T value_type;
				typedef AmfType amf_type;
				typedef typename AmfType::imf_r_type imf_r_type;
				typedef typename AmfType::imf_c_type imf_c_type;
				/** Type returned by access function */
				typedef T &access_type;
				typedef const T &const_access_type;
				/** Type of the index used to access the physical storage */
				typedef typename AmfType::storage_index_type storage_index_type;

			protected:
				typedef StorageBasedMatrix< T, AmfType, requires_allocation, omp > self_type;
				friend MatrixBase< self_type >;

				typedef typename std::conditional<
					requires_allocation,
					Vector< T, omp >,
					Vector< T, omp > &
				>::type container_type;

				/** A container-type view is characterized by its association with a physical container */
				container_type container;

				/**
				 * Access mapping function maps a pair of logical coordinates
				 * into the concrete coordinate inside the actual container.
				 * \see AMF
				 */
				AmfType amf;

				/**
				 * determines the size of the matrix via the domain of
				 * the index mapping functions.
				 *
				 * @return A pair of dimensions.
				 */
				std::pair< size_t, size_t > dims() const noexcept {
					return amf.getLogicalDimensions();
				}

				size_t getStorageDimensions() const noexcept {
					return amf.getStorageDimensions();
				}

				friend const Vector< T, omp > &getContainer( const self_type &A ) {
					return A.container;
				}

				friend Vector< T, omp > &getContainer( self_type &A ) {
					return A.container;
				}

				bool getInitialized() const noexcept {
					return internal::getInitialized( container );
				}

				void setInitialized( const bool initialized ) noexcept {
					internal::setInitialized( container , initialized );
				}

				const AmfType &getAmf() const noexcept {
					return amf;
				}

				/**
				 * Returns a constant reference to the element corresponding to
				 * the provided storage index.
				 *
				 * @param storageIndex  storage index in the physical iteration
				 *                      space.
				 *
				 * @return const reference or value of the element at given position.
				 * \note This function may result in accessing memory belonging to
				 *       another thread, which may incurr performance penalty.
				 */
				const_access_type access( const storage_index_type &si ) const {
					return getRaw( getLocalContainer( container, si.buffer_id, si.block_id ) )[ si.offset ];
				}

				access_type access( const storage_index_type &si ) {
					return getRaw( getLocalContainer( container, si.buffer_id, si.block_id ) )[ si.offset ];
				}

				storage_index_type getStorageIndex( const size_t i, const size_t j, const size_t s, const size_t P ) const {
					return amf.getStorageIndex( i, j, s, P );
				}

				/**
				 * Construct a new structured matrix Base object
				 *
				 * @param rows The number of rows of the matrix.
				 * @param cols The number of columns of the matrix.
				 * @param smf  The storage mapping function assigned to this matrix.
				 */
				StorageBasedMatrix( AmfType &&amf ) :
					// \todo enable only if ImfR and ImfC are imf::Id
					container( amf.getDistribution() ),
					amf( std::move( amf ) ) {
#ifdef DEBUG
					std::cout << "Entering OMP StorageBasedMatrix constructor\n";
#endif
				}

				/** View on another container */
				StorageBasedMatrix( Vector< T, omp > &container, AmfType &&amf ) :
					container( container ),
					amf( std::move( amf ) ) {}

		}; // class StorageBasedMatrix

	} // namespace internal

} // namespace alp

#endif // end ``_H_ALP_OMP_STORAGEBASEDMATRIX''
