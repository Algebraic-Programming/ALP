
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

#ifndef _H_ALP_AMF_BASED_STORAGEBASEDMATRIX
#define _H_ALP_AMF_BASED_STORAGEBASEDMATRIX

#include <alp/backends.hpp>
#include <alp/base/matrix.hpp>
#include <alp/config.hpp>
#include <alp/ops.hpp>
#include <alp/type_traits.hpp>
#include <alp/utils.hpp>

#include "storage.hpp"
#include "vector.hpp"

#include <alp/dispatch/vector.hpp>

namespace alp {

	namespace internal {

		/** Forward declaration */
		template< typename DerivedMatrix >
		class MatrixBase;

		template< typename DerivedMatrix >
		std::pair< size_t, size_t > dims( const MatrixBase< DerivedMatrix > &A ) noexcept;

		template<
			typename MatrixType,
			std::enable_if_t< internal::is_storage_based< MatrixType >::value > * = nullptr
		>
		size_t getStorageDimensions( const MatrixType &A ) noexcept;

		template< typename MatrixType,
			std::enable_if_t< is_matrix< MatrixType>::value > * = nullptr
		>
		bool getInitialized( const MatrixType &A ) noexcept;

		template< typename MatrixType,
			std::enable_if_t< is_matrix< MatrixType>::value > * = nullptr
		>
		void setInitialized( MatrixType &, const bool ) noexcept;

		/** Forward declaration */
		template< typename T, typename AmfType, bool requires_allocation, Backend backend >
		class StorageBasedMatrix;

		/** Container reference getters used by friend functions of specialized Matrix */
		template< typename T, typename AmfType, bool requires_allocation, Backend backend >
		const Vector< T, backend > & getContainer( const StorageBasedMatrix< T, AmfType, requires_allocation, backend > & A );

		template< typename T, typename AmfType, bool requires_allocation, Backend backend >
		Vector< T, backend > & getContainer( StorageBasedMatrix< T, AmfType, requires_allocation, backend > & A );

		/** Container reference getters. Defer the call to base class friend function */
		template<
			typename T, typename Structure, enum Density density, typename View,
			typename ImfR, typename ImfC,
			Backend backend
		>
		const Vector< T, backend > & getContainer( const alp::Matrix< T, Structure, density, View, ImfR, ImfC, backend > & A ) {
			return getContainer( static_cast<
				const StorageBasedMatrix<
					T,
					typename alp::Matrix< T, Structure, density, View, ImfR, ImfC, backend >::amf_type,
					alp::Matrix< T, Structure, density, View, ImfR, ImfC, backend >::requires_allocation,
					backend
				> &
			>( A ) );
		}

		template<
			typename T, typename Structure, enum Density density, typename View,
			typename ImfR, typename ImfC,
			Backend backend
		>
		Vector< T, backend > & getContainer( alp::Matrix< T, Structure, density, View, ImfR, ImfC, backend > & A ) {
			return getContainer( static_cast<
				StorageBasedMatrix<
					T,
					typename alp::Matrix< T, Structure, density, View, ImfR, ImfC, backend >::amf_type,
					alp::Matrix< T, Structure, density, View, ImfR, ImfC, backend >::requires_allocation,
					backend
				> &
			>( A ) );
		}

		/** Returns the reference to the AMF of a storage-based matrix */
		template<
			typename MatrixType,
			std::enable_if< internal::is_storage_based< MatrixType >::value > * = nullptr
		>
		const typename MatrixType::amf_type &getAmf( const MatrixType &A ) noexcept;

		/**
		 * Matrix container specialization
		 * Implements both original containers and views on containers.
		 * @tparam requires_allocation True if the class is an original container
		 *                             False if the class is a view of another matrix
		 */
		template< typename T, typename AmfType, bool requires_allocation, Backend backend >
		class StorageBasedMatrix : public MatrixBase< StorageBasedMatrix< T, AmfType, requires_allocation, backend > > {

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
				typedef size_t storage_index_type;

			protected:
				typedef StorageBasedMatrix< T, AmfType, requires_allocation, backend > self_type;
				friend MatrixBase< self_type >;

				typedef typename std::conditional<
					requires_allocation,
					Vector< T, backend >,
					Vector< T, backend > &
				>::type container_type;

				/** A container-type view is characterized by its association with a physical container */
				container_type container;

				/**
				 * All matrix views use a pair of index mapping functions to
				 * capture the correspondence between their logical layout and the one
				 * of their underlying container. This may be another view leading to a composition
				 * of IMFs between the top matrix view and the physical container.
				 * Original matrix containers's index mapping functions are an identity mapping.
				 */
				//ImfR imf_r;
				//ImfC imf_c;

				/**
				 * The container's storage scheme. \a storage_scheme is not exposed to the user as an option
				 * but can defined by ALP at different points in the execution depending on the \a backend choice.
				 * In particular, if the structured matrix is not a temporary matrix than it is fixed at construction
				 * time when the allocation takes place.
				 * If the structured matrix is a temporary one than a storage storage scheme choice may or may not be
				 * made depending on whether a decision about instantiating the matrix is made by the framework.
				 *
				 * The specific storage scheme choice depends on the chosen backend and the structure of the matrix.
				 * \internal \todo Revisit this when introducing storage mapping functions.
				 */
				//Smf smf;

				/**
				 * Access mapping function maps a pair of logical coordinates
				 * into the concrete coordinate inside the actual container.
				 * \see AMF
				 */
				AmfType amf;
				/**
				 * @brief determines the size of the matrix via the domain of
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

				friend const Vector< T, backend > & getContainer( const self_type &A ) {
					return A.container;
				}

				friend Vector< T, backend > & getContainer( self_type &A ) {
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
				 */
				const_access_type access( const storage_index_type &storageIndex ) const {
					return container[ storageIndex ];
				}

				access_type access( const storage_index_type &storageIndex ) {
					return container[ storageIndex ];
				}

				storage_index_type getStorageIndex( const size_t i, const size_t j, const size_t s, const size_t P ) const {
					return amf.getStorageIndex( i, j, s, P );
				}

				/**
				 * @brief Construct a new structured matrix Base object assigning identity
				 * mapping functions both to the row and column dimensions.
				 *
				 * @param rows The number of rows of the matrix.
				 * @param cols The number of columns of the matrix.
				 * @param smf  The storage mapping function assigned to this matrix.
				 */
				/** (Documentation of old MatrixCotiainer. TODO: merge with the above docs.
				 * @brief Construct a new structured matrix container object.
				 *
				 * \warning \a cap is present for compatibility with other matrix specializations.
				 *          In reference backend, the number of non-zeros (i.e. capacity)
				 *          depends on the used storage scheme. Therefore, this parameter is
				 *          ignored.
				 *
				 * TODO: Add the storage scheme a parameter to the constructor
				 * so that allocation can be made accordingly, generalizing the full case.
				 */
				StorageBasedMatrix( AmfType &&amf ) :
					// enable only if ImfR and ImfC are imf::Id
					container( internal::Vector< T, backend >( amf.getStorageDimensions() ) ),
					amf( std::move( amf ) ) {}

				/** View on another container */
				StorageBasedMatrix( Vector< T, backend > &container, AmfType &&amf ) :
					container( container ),
					amf( std::move( amf ) ) {}

				/** View on another raw container */
				StorageBasedMatrix( T *buffer, const size_t buffer_size, AmfType &&amf ) :
					container( buffer, buffer_size ),
					amf( std::move( amf ) ) {}

		}; // class StorageBasedMatrix

	} // namespace internal

	/** Definitions of previously declared global methods that operate on ALP Matrix */
	namespace internal {

		/** Get the reference to the AMF of a storage-based matrix */
		template<
			typename MatrixType,
			std::enable_if< internal::is_storage_based< MatrixType >::value > * = nullptr
		>
		const typename MatrixType::amf_type &getAmf( const MatrixType &A ) noexcept {
			return A.getAmf();
		}

	} // namespace internal

	template<
		typename MatrixType,
		std::enable_if< internal::is_storage_based< MatrixType >::value > * = nullptr
	>
	size_t internal::getStorageDimensions( const MatrixType &A ) noexcept {
		static_assert( is_storage_based< MatrixType >::value, "getStorageDimensions supported only for storage-based containers.");
		return static_cast< const typename MatrixType::base_type & >( A ).getStorageDimensions();
	}

} // namespace alp

#endif // end ``_H_ALP_AMF_BASED_STORAGEBASEDMATRIX''
