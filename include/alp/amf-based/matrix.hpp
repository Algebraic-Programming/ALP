
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

/*
 * @author A. N. Yzelman
 * @date 14th of January 2022
 */

#ifndef _H_ALP_AMF_BASED_MATRIX
#define _H_ALP_AMF_BASED_MATRIX

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include <alp/backends.hpp>
#include <alp/base/matrix.hpp>
#include <alp/config.hpp>
#include <alp/ops.hpp>
#include <alp/rc.hpp>
#include <alp/type_traits.hpp>
#include <alp/utils.hpp>
#include <alp/utils/autodeleter.hpp>
//#include <alp/utils/pattern.hpp> //for help with dealing with pattern matrix input
#include <alp/vector.hpp>
#include <alp/storage.hpp>
#include <alp/structures.hpp>
#include <alp/density.hpp>
#include <alp/views.hpp>
#include <alp/imf.hpp>

namespace alp {

	/** Identifies any backend's implementation of ALP matrix as an ALP matrix. */
	template<
		typename T, typename Structure, enum Density density,
		typename View, typename ImfR, typename ImfC, enum Backend backend
	>
	struct is_matrix< Matrix< T, Structure, density, View, ImfR, ImfC, backend > > : std::true_type {};

	// Matrix-related implementation

	namespace internal {
		/** Forward declaration */
		template< typename T, typename AmfType, bool requires_allocation >
		class StorageBasedMatrix;

		/** Forward declaration */
		template< typename T, typename ImfR, typename ImfC, typename DataLambdaType >
		class FunctorBasedMatrix;

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

	} // namespace internal

	namespace internal {

		// Forward declarations
		template<
			typename MatrixType,
			std::enable_if< is_matrix< MatrixType >::value > * = nullptr
		>
		const typename MatrixType::access_type access( const MatrixType &, const typename MatrixType::storage_index_type & );

		template<
			typename MatrixType,
			std::enable_if< is_matrix< MatrixType >::value > * = nullptr
		>
		typename MatrixType::access_type access( MatrixType &, const typename MatrixType::storage_index_type & );

		template<
			typename MatrixType,
			std::enable_if< is_matrix< MatrixType >::value > * = nullptr
		>
		typename MatrixType::storage_index_type getStorageIndex( const MatrixType &A, const size_t i, const size_t j, const size_t s = 0, const size_t P = 1 );

		/** Returns the reference to the AMF of a storage-based matrix */
		template<
			typename MatrixType,
			std::enable_if< internal::is_storage_based< MatrixType >::value > * = nullptr
		>
		const typename MatrixType::amf_type &getAmf( const MatrixType &A ) noexcept;

		/**
		 * Base Matrix class containing attributes common to all Matrix specialization
		 */
		template< typename DerivedMatrix >
		class MatrixBase {

			friend std::pair< size_t, size_t > dims<>( const MatrixBase< DerivedMatrix > &A ) noexcept;

			template< typename MatrixType, std::enable_if_t< is_matrix< MatrixType>::value > * >
			friend bool getInitialized( const MatrixType &A ) noexcept;

			template< typename MatrixType, std::enable_if_t< is_matrix< MatrixType>::value > * >
			friend void setInitialized( MatrixType &A, const bool initialized ) noexcept;

			protected:

				std::pair< size_t, size_t > dims() const noexcept {
					return static_cast< const DerivedMatrix & >( *this ).dims();
				}

				template<
					typename MatrixType,
					std::enable_if< is_matrix< MatrixType >::value > *
				>
				friend const typename MatrixType::access_type access( const MatrixType &A, const typename MatrixType::storage_index_type &storageIndex );

				template<
					typename MatrixType,
					std::enable_if< is_matrix< MatrixType >::value > *
				>
				friend typename MatrixType::access_type access( MatrixType &A, const typename MatrixType::storage_index_type &storageIndex );

				template<
					typename MatrixType,
					std::enable_if< is_matrix< MatrixType >::value > *
				>
				friend typename MatrixType::storage_index_type getStorageIndex( const MatrixType &A, const size_t i, const size_t j, const size_t s, const size_t P );

				bool getInitialized() const {
					return static_cast< const DerivedMatrix & >( *this ).getInitialized();
				}

				void setInitialized( const bool initialized ) {
					static_cast< DerivedMatrix & >( *this ).setInitialized( initialized );
				}

				template< typename AccessType, typename StorageIndexType >
				const AccessType access( const StorageIndexType storageIndex ) const {
					static_assert( std::is_same< AccessType, typename DerivedMatrix::access_type >::value );
					static_assert( std::is_same< StorageIndexType, typename DerivedMatrix::storage_index_type >::value );
					return static_cast< const DerivedMatrix & >( *this ).access( storageIndex );
				}

				template< typename AccessType, typename StorageIndexType >
				AccessType access( const StorageIndexType &storageIndex ) {
					static_assert( std::is_same< AccessType, typename DerivedMatrix::access_type >::value );
					static_assert( std::is_same< StorageIndexType, typename DerivedMatrix::storage_index_type >::value );
					return static_cast< DerivedMatrix & >( *this ).access( storageIndex );
				}

				template< typename StorageIndexType >
				StorageIndexType getStorageIndex( const size_t i, const size_t j, const size_t s, const size_t P ) const {
					static_assert( std::is_same< StorageIndexType, typename DerivedMatrix::storage_index_type >::value );
					return static_cast< const DerivedMatrix & >( *this ).getStorageIndex( i, j, s, P );
				}

		};

		/**
		 * Determines the mapping polynomial type and exposes a factory method
		 * to create instances of that polynomial.
		 *
		 * All specializations of this type trait should define the factory
		 * method following the same signature. The factory method shall
		 * return an object of the type exposed as \a type.
		 *
		 * @tparam Structure  Matrix structure
		 * @tparam ImfR       Row IMF type
		 * @tparam ImfC       Column IMF type
		 * @tparam backend    The backend
		 *
		 */
		template< typename Structure, typename ImfR, typename ImfC, enum Backend backend >
		struct determine_poly_factory {};

		/**
		 * Determines the AMF type for a matrix having the provided static properties.
		 *
		 * For a matrix that requires allocation, the new AMF consists of two Id IMFs
		 * and the pre-defined mapping polynomial.
		 * For a view over another matrix, the new AMF is created from the AMF of the
		 * target matrix in one of the following ways:
		 *  - When applying gather view using IMFs, the IMFs are applied to the AMF of
		 *    the target matrix.
		 *  - When applying a different view type (e.g. transpose or diagonal), the AMF
		 *    of the target matrix is transformed according to the provided view type.
		 *
		 * @tparam View     View type
		 * @tparam ImfR     Row IMF type
		 * @tparam ImfC     Column IMF type
		 * @tparam backend  The backend
		 *
		 * The valid combinations of the input parameters are as follows:
		 *  - original view on void with Id IMFs.
		 *  - original view on ALP matrix with any type of IMFs
		 *  - other type of views (e.g. transposed, diagonal) with only Id IMFs.
		 * Invocation using incompatible parameters may result in an undefined behavior.
		 * The first parameter combination is handled by a specialization of this trait.
		 *
		 */
		template<
			typename Structure, typename View, typename ImfR, typename ImfC,
			enum Backend backend
		>
		struct determine_amf_type {

			/** Ensure that the view is not on a void type */
			static_assert(
				!std::is_same< typename View::applied_to, void >::value,
				"Cannot handle views over void type by this determine_amf_type specialization."
			);

			/** Ensure that if the view is original, the IMFs are Id */
			static_assert(
				View::type_id != view::original ||
				( View::type_id == view::original && std::is_same< imf::Id, ImfR >::value && std::is_same< imf::Id, ImfC >::value ),
				"Original view with non-ID Index Mapping Functions is not supported."
			);

			/** Ensure that if the view is transposed, the IMFs are Id */
			static_assert(
				View::type_id != view::transpose ||
				( View::type_id == view::transpose && std::is_same< imf::Id, ImfR >::value && std::is_same< imf::Id, ImfC >::value ),
				"Transposed view with non-ID Index Mapping Functions is not supported."
			);

			/** Ensure that if the view is diagonal, the row and column IMFs are Id and Zero, respectively */
			static_assert(
				View::type_id != view::diagonal ||
				( View::type_id == view::diagonal && std::is_same< imf::Id, ImfR >::value && std::is_same< imf::Zero, ImfC >::value ),
				"Diagonal view with non-Id Row and non-Zero Column Index Mapping Functions is not supported."
			);

			typedef typename std::conditional<
				View::type_id == view::gather,
				typename storage::AMFFactory::Compose<
					ImfR, ImfC, typename View::applied_to::amf_type
				>::amf_type,
				typename storage::AMFFactory::Reshape<
					View::type_id,
					typename View::applied_to::amf_type
				>::amf_type	
			>::type type;

		};

		/** Specialization for containers that allocate storage */
		template< typename Structure, typename ImfC, enum Backend backend >
		struct determine_amf_type< Structure, view::Original< void >, imf::Id, ImfC, backend > {

			static_assert(
				std::is_same< ImfC, imf::Id >::value || std::is_same< ImfC, imf::Zero >::value,
				"Incompatible combination of parameters provided to determine_amf_type."
			);

			typedef typename storage::AMFFactory::FromPolynomial<
				typename determine_poly_factory< Structure, imf::Id, ImfC, backend >::factory_type
			>::amf_type type;
		};

		/** Specialization for containers that allocate storage */
		template< typename Structure, typename ImfC, enum Backend backend, typename Lambda >
		struct determine_amf_type< Structure, view::Functor< Lambda >, imf::Id, ImfC, backend > {

			static_assert(
				std::is_same< ImfC, imf::Id >::value || std::is_same< ImfC, imf::Zero >::value,
				"Incompatible combination of parameters provided to determine_amf_type."
			);

			typedef typename storage::AMFFactory::FromPolynomial<
				storage::polynomials::NoneFactory
			>::amf_type type;
		};

		template<
			typename T,
			typename Structure,
			enum Density density,
			typename View,
			typename ImfR,
			typename ImfC,
			enum Backend backend
		>
		struct matrix_base_class {
			typedef typename std::conditional<
				internal::is_view_over_functor< View >::value,
				internal::FunctorBasedMatrix< T, ImfR, ImfC, typename View::applied_to >,
				internal::StorageBasedMatrix< T,
					typename internal::determine_amf_type< Structure, View, ImfR, ImfC, backend >::type,
					internal::requires_allocation< View >::value
				>
			>::type type;
		};

	} // namespace internal

	/**
	 * \brief An ALP structured matrix.
	 *
	 * This is an opaque data type for structured matrices.
	 *
	 * A structured matrix exposes a mathematical
	 * \em logical layout which allows to express implementation-oblivious concepts
	 * including the matrix structure itself and \em views on the matrix.
	 * The logical layout of a structured matrix maps to a physical counterpart via
	 * a storage scheme which typically depends on the chosen structure and the selected
	 * backend. alp::Matrix and alp::Vector may be used as interfaces to such a physical
	 * layout.
	 * To visualize this, you may think of a band matrix. Using a
	 * full dense or a banded storage schemes would require
	 * the use of a \a alp::Matrix container (see include/alp/density.hpp for
	 * more details about the supported storage schemes). However, the interpration of its
	 * content would differ in the two cases being a function of both the Structure
	 * information and the storage scheme combined.
	 *
	 * Views can be used to create logical \em perspectives on top of a container.
	 * For example, one may decide to refer to the transpose of a matrix or to treat
	 * for a limited part of my program a square matrix as symmetric.
	 * If a view can be expressed as a concept \em invariant of specific runtime features,
	 * such views can be defined statically (for example, one may always refer to the
	 * transpose or the diagonal of a matrix irrespective of features such as the matrix's
	 * size). Other may depend on features such as the size of a matrix
	 * (e.g., gathering/scattering the rows/columns of a matrix or permuting them).
	 *
	 * Structured matrices defined as views on other matrices do not instantiate a
	 * new container but refer to the one used by their targets.
	 * See the two specializations
	 * \a Matrix<T, structures::General, Density::Dense, View, reference >
	 * and \a Matrix<T, structures::General, Density::Dense, view::Indentity<void>, reference >
	 * as examples of structured matrix types without and with physical container, respectively.
	 *
	 *
	 * @tparam T         The type of the matrix elements. \a T shall not be a GraphBLAS
	 *                   type.
	 * @tparam Structure One of the matrix structures defined in \a alp::structures.
	 * @tparam density   Either \em enum \a Density::Dense or \em enum
	 *                   \a storage::Sparse.
	 * @tparam View      One of the matrix views in \a alp::view.
	 *                   All static views except for \a view::Original (via
	 *                   \a view::Original<void> cannot instantiate a new container
	 *                   and only allow to refer to a previously defined
	 *                   \a Matrix.
	 *                   The \a View parameter should not be used directly
	 *                   by the user but selected via \a get_view function.
	 *
	 * See examples of Matrix definitions within \a include/alp/reference/matrix.hpp
	 * and the \a dense_structured_matrix.cpp unit test.
	 *
	 */
	template< typename T, typename Structure, enum Density density, typename View, typename ImfR, typename ImfC, enum Backend backend >
	class Matrix :
		public internal::matrix_base_class< T, Structure, Density::Dense, View, ImfR, ImfC, backend >::type {

		protected:
			typedef Matrix< T, Structure, Density::Dense, View, ImfR, ImfC, backend > self_type;

			/*********************
				Storage info friends
			******************** */

			template< typename fwd_iterator >
			friend RC buildMatrix( Matrix< T, Structure, Density::Dense, View, ImfR, ImfC, backend > &A,
				const fwd_iterator & start, const fwd_iterator & end );

			template< typename fwd_iterator >
			RC buildMatrixUnique( const fwd_iterator &start, const fwd_iterator &end ) {
				std::cout << "Building Matrix<>; calling buildMatrix( Matrix<> )\n";
				return buildMatrix( *(this->_container), start, end );
			}

		public:
			/** Exposes the types and the static properties. */
			typedef Structure structure;
			/**
			 * Indicates if a matrix needs to allocate data-related memory
			 * (for the internal container or functor object).
			 * False if it is a view over another matrix or a functor.
			 */
			static constexpr bool requires_allocation = internal::requires_allocation< View >::value;

			/**
			 * Expose the base type class to enable internal functions to cast
			 * the type of objects of this class to the base class type.
			 */
			typedef typename internal::matrix_base_class< T, Structure, Density::Dense, View, ImfR, ImfC, backend >::type base_type;

			template < view::Views view_tag, bool d = false >
			struct view_type;

			template < bool d >
			struct view_type< view::original, d > {
				using type = Matrix< T, Structure, Density::Dense, view::Original< self_type >, imf::Id, imf::Id, backend >;
			};

			template < bool d >
			struct view_type< view::gather, d > {
				using type = Matrix<
					T,
					typename structures::apply_view< view::gather, Structure >::type,
					Density::Dense, view::Gather< self_type >, imf::Strided, imf::Strided, backend
				>;
			};

			template < bool d >
			struct view_type< view::transpose, d > {
				using type = Matrix<
					T,
					typename structures::apply_view< view::transpose, Structure >::type,
					Density::Dense, view::Transpose< self_type >, imf::Id, imf::Id, backend
				>;
			};

			template < bool d >
			struct view_type< view::diagonal, d > {
				using type = Vector< T, structures::General, Density::Dense, view::Diagonal< self_type >, imf::Id, imf::Zero, backend >;
			};

			/**
			 * Constructor for a storage-based matrix that allocates storage.
			 * Specialization for a matrix with not necessarily equal row and column dimensions.
			 */
			template<
				typename ThisStructure = Structure,
				std::enable_if_t<
					internal::is_view_over_storage< View >::value &&
					internal::requires_allocation< View >::value &&
					!structures::is_in< structures::Square, typename ThisStructure::inferred_structures >::value
				> * = nullptr
			>
			Matrix( const size_t rows, const size_t cols, const size_t cap = 0 ) :
				base_type(
					storage::AMFFactory::FromPolynomial<
						typename internal::determine_poly_factory< structure, ImfR, ImfC, backend >::factory_type
					>::Create(
						ImfR( rows ),
						ImfC( cols )
					)
				) {

				(void) cap;

				// This check should be performed in the class body rather than here.
				// Allocation-requiring matrix with incompatible IMFs should not be instantiable at all.
				// Here it is only forbidden to invoke this constructor for such a matrix.
				static_assert(
					std::is_same< ImfR, imf::Id >::value &&
					( std::is_same< ImfC, imf::Id >::value || std::is_same< ImfC, imf::Zero >::value ),
					"This constructor can only be used with a matrix having Id IMFs."
				);

			}

			/*
			 * Constructor for a storage-based matrix that allocates storage.
			 * Specialization for matrices with equal row and column dimensions.
			 */
			template<
				typename ThisStructure = Structure,
				std::enable_if_t<
					internal::is_view_over_storage< View >::value &&
					internal::requires_allocation< View >::value &&
					structures::is_in< structures::Square, typename ThisStructure::inferred_structures >::value
				> * = nullptr
			>
			Matrix( const size_t dim, const size_t cap = 0 ) :
				base_type(
					storage::AMFFactory::FromPolynomial<
						typename internal::determine_poly_factory< structure, ImfR, ImfC, backend >::factory_type
					>::Create(
						ImfR( dim ),
						ImfC( dim )
					)
				) {

				(void) cap;

				static_assert(
					std::is_same< ImfR, imf::Id >::value &&
					( std::is_same< ImfC, imf::Id >::value || std::is_same< ImfC, imf::Zero >::value ),
					"This constructor can only be used with a matrix having Id IMFs."
				);

			}

			/**
			 * Constructor for a view over another storage-based matrix.
			 *
			 * @tparam SourceType  The type of the target matrix.
			 *
			 */
			template<
				typename SourceType,
				std::enable_if_t<
					std::is_same< SourceType, typename View::applied_to >::value &&
					internal::is_view_over_storage< View >::value &&
					!internal::requires_allocation< View >::value
				> * = nullptr
			>
			Matrix( SourceType &source_matrix, ImfR imf_r, ImfC imf_c ) :
				base_type(
					getContainer( source_matrix ),
					storage::AMFFactory::Compose<
						ImfR, ImfC, typename SourceType::base_type::amf_type
					>::Create( imf_r, imf_c, internal::getAmf( source_matrix ) )
				) {}

			/**
			 * Constructor for a view over another matrix applying a view defined
			 * by View template parameter of the constructed matrix.
			 *
			 * @tparam SourceType  The type of the target matrix.
			 *
			 */
			template<
				typename SourceType,
				std::enable_if_t<
					std::is_same< SourceType, typename View::applied_to >::value &&
					internal::is_view_over_storage< View >::value &&
					!internal::requires_allocation< View >::value
				> * = nullptr
			>
			Matrix( SourceType &source_matrix ) :
				base_type(
					getContainer( source_matrix ),
					storage::AMFFactory::Reshape< View::type_id, typename SourceType::amf_type >::Create( internal::getAmf( source_matrix ) )
				) {}

			/**
			 * @deprecated
			 * Constructor for a view over another storage-based matrix.
			 *
			 * @tparam SourceType  The type of the target matrix.
			 * @tparam AmfType     The type of the amf used to construct the matrix.
			 *                     Used as a template parameter to benefit from
			 *                     SFINAE for the case of FunctorBasedMatrix, when
			 *                     base_type::amf_type does not exist and, therefore,
			 *                     using the expression base_type::amf_type would
			 *                     result in a hard compilation error.
			 */
			template<
				typename SourceType,
				typename AmfType,
				std::enable_if_t<
					std::is_same< SourceType, typename View::applied_to >::value &&
					internal::is_view_over_storage< View >::value &&
					!internal::requires_allocation< View >::value
				> * = nullptr
			>
			Matrix( SourceType &source_matrix, AmfType &&amf ) :
				base_type(
					getContainer( source_matrix ),
					std::forward< typename base_type::amf_type >( amf )
				) {
				static_assert(
					std::is_same< typename base_type::amf_type, AmfType >::value,
					"The AMF type of the constructor parameter needs to match the AMF type of this container specialization."
				);
			}

			/**
			 * Constructor for a functor-based matrix that allocates memory.
			 * Specialization for a matrix with non necessarily equal row and column dimensions.
			 *
			 * @tparam LambdaType  The type lambda function associated to the data.
			 *
			 */
			template<
				typename LambdaType,
				std::enable_if_t<
					std::is_same< LambdaType, typename View::applied_to >::value &&
					internal::is_view_over_functor< View >::value &&
					internal::requires_allocation< View >::value &&
					!structures::is_in< structures::Square, typename Structure::inferred_structures >::value
				> * = nullptr
			>
			Matrix( std::function< bool() > initialized, const size_t rows, const size_t cols, LambdaType lambda ) :
				base_type( initialized, imf::Id( rows ), imf::Id( cols ), lambda ) {

				static_assert(
					std::is_same< ImfR, imf::Id >::value &&
					std::is_same< ImfC, imf::Id >::value,
					"This constructor can only be used with Id IMFs."
				);

			}

			/**
			 * Constructor for a functor-based matrix that allocates memory.
			 * Specialization for a matrix with equal row and column dimensions.
			 *
			 * @tparam LambdaType  The type lambda function associated to the data.
			 *
			 */
			template<
				typename LambdaType,
				std::enable_if_t<
					std::is_same< LambdaType, typename View::applied_to >::value &&
					internal::is_view_over_functor< View >::value &&
					internal::requires_allocation< View >::value &&
					structures::is_in< structures::Square, typename Structure::inferred_structures >::value
				> * = nullptr
			>
			Matrix( std::function< bool() > initialized, const size_t dim, LambdaType lambda ) :
				base_type( initialized, imf::Id( dim ), imf::Id( dim ), lambda ) {

				static_assert(
					std::is_same< ImfR, imf::Id >::value &&
					std::is_same< ImfC, imf::Id >::value,
					"This constructor can only be used with Id IMFs."
				);

			}

			/**
			 * Constructor for a view over another functor-based matrix.
			 *
			 * @tparam SourceType  The type of the target matrix.
			 *
			 */
			template<
				typename SourceType,
				std::enable_if_t<
					std::is_same< SourceType, typename View::applied_to >::value &&
					internal::is_view_over_functor< View >::value &&
					!internal::requires_allocation< View >::value
				> * = nullptr
			>
			Matrix( SourceType &source_matrix, ImfR imf_r, ImfC imf_c ) :
				base_type( getFunctor( source_matrix ), imf_r, imf_c ) {}

			/**
			 * @deprecated
			 * Constructor for a view over another functor-based matrix.
			 *
			 * @tparam SourceType  The type of the target matrix.
			 *
			 */
			template<
				typename SourceType,
				std::enable_if_t<
					std::is_same< SourceType, typename View::applied_to >::value &&
					internal::is_view_over_functor< View >::value &&
					!internal::requires_allocation< View >::value
				> * = nullptr
			>
			Matrix( SourceType &source_matrix ) :
				Matrix( getFunctor( source_matrix ),
					imf::Id( nrows ( source_matrix ) ),
					imf::Id( ncols ( source_matrix ) )
				) {

				static_assert(
					std::is_same< ImfR, imf::Id >::value &&
					std::is_same< ImfC, imf::Id >::value,
					"This constructor can only be used with Id IMFs."
				);

			}
	}; // ALP Matrix

	namespace structures {

		/**
		 * @brief Checks if TestedStructure is a \a Structure according to the ALP's structure classification.
		 *
		 * @tparam TestedStructure   The structure to be tested.
		 * @tparam Structure 		 The structure that should be implied by \a TestedStructure.
		 */
		template< typename TestedStructure, typename Structure >
		struct is_a {

			static_assert( std::is_base_of< structures::BaseStructure, TestedStructure >::value );

			/**
			 * \a value is true iff \a Structure is implied by \a TestedStructure.
			 */
			static constexpr bool value = is_in< Structure, typename TestedStructure::inferred_structures >::value;
		};

		/**
		 * Calculates the iteration space for row-dimension for the given matrix and band index.
		 *
		 * @tparam MatrixType The type of ALP matrix
		 * @tparam band_index The index of the desired matrix band
		 *
		 * @param[in] A       ALP matrix
		 *
		 * @returns a pair of size_t values,
		 *          the first representing lower and the second upper limit.
		 *
		 * \note    Each backend shall specialize this function as its implementation
		 *          depends on the way backend handles storage of different structures.
		 */
		template<
			size_t band_index, typename MatrixType,
			std::enable_if_t<
				is_matrix< MatrixType >::value
			> * = nullptr
		>
		std::pair< size_t, size_t > calculate_row_coordinate_limits( const MatrixType &A );

		/**
		 * Calculates the iteration space for column-dimension for the given matrix, band index and row index.
		 *
		 * @tparam MatrixType The type of ALP matrix
		 * @tparam band_index The index of the desired matrix band
		 *
		 * @param[in] A       ALP matrix
		 * @param[in] row     Row index
		 *
		 * @returns a pair of size_t values,
		 *          the first representing lower and the second upper limit.
		 *
		 * \note    Each backend shall specialize this function as its implementation
		 *          depends on the way backend handles storage of different structures.
		 */
		template<
			size_t band_index, typename MatrixType,
			std::enable_if_t<
				is_matrix< MatrixType >::value
			> * = nullptr
		>
		std::pair< size_t, size_t > calculate_column_coordinate_limits( const MatrixType &A, const size_t row );

	} // namespace structures

	/**
	 *
	 * @brief Generate a view specified by \a target_view where the type is compliant with the
	 * 		  \a source matrix.
	 * 		  The function guarantees the created view is non-overlapping with other
	 *        existing views only when the check can be performed in constant time.
	 *
	 * @tparam target_view  One of the supported views listed in \a view::Views
	 * @tparam SourceMatrix The type of the source matrix
	 *
	 * @param source        The source ALP matrix
	 *
	 * @return A new \a target_view view over the source matrix.
	 *
     * \parblock
     * \par Performance semantics.
     *        -# This function performs
     *           \f$ \Theta(nref) \f$ amount of work where \f$ nref \f$ is the number
	 * 			 of available views of \a source.
     *        -# A call to this function may use \f$ \mathcal{O}(1) \f$ bytes
     *           of memory beyond the memory in use at the function call entry.
     *        -# This function may make system calls.
     * \endparblock
	 *
	 */
	template<
		enum view::Views target_view = view::original,
		typename SourceMatrix,
		std::enable_if_t<
			is_matrix< SourceMatrix >::value &&
			target_view != view::diagonal
		> * = nullptr
	>
	typename SourceMatrix::template view_type< target_view >::type
	get_view( SourceMatrix &source ) {

		using target_strmat_t = typename SourceMatrix::template view_type< target_view >::type;

		return target_strmat_t( source );
	}

	/** Specialization for diagonal view over Square matrix */
	template<
		enum view::Views target_view = view::original,
		typename SourceMatrix,
		std::enable_if_t<
			is_matrix< SourceMatrix >::value &&
			target_view == view::diagonal &&
			structures::is_in< structures::Square, typename SourceMatrix::structure::inferred_structures >::value
		> * = nullptr
	>
	typename SourceMatrix::template view_type< view::diagonal >::type
	get_view( SourceMatrix &source ) {

		using target_t = typename SourceMatrix::template view_type< view::diagonal >::type;
		return target_t( source );
	}

	/**
	 * Specialization for diagonal view over non-Square matrix.
	 * A diagonal view is created over a intermediate gather
	 * view with a square structure.
	 */
	template<
		enum view::Views target_view = view::original,
		typename SourceMatrix,
		std::enable_if_t<
			is_matrix< SourceMatrix >::value &&
			target_view == view::diagonal &&
			!structures::is_in< structures::Square, typename SourceMatrix::structure::inferred_structures >::value
		> * = nullptr
	>
	typename internal::new_container_type_from<
		typename SourceMatrix::template view_type< view::gather >::type
	>::template change_structure< structures::Square >::type
	::template view_type< view::diagonal >::type
	get_view( SourceMatrix &source ) {

		const size_t source_rows = nrows( source );
		const size_t source_cols = ncols( source );
		const size_t smaller_dimension = std::min( source_rows, source_cols );
		auto square_view = get_view< structures::Square >( source, utils::range( 0, smaller_dimension ), utils::range( 0, smaller_dimension ) );
		return get_view< view::diagonal >( square_view );
	}

	/**
	 *
	 * @brief Generate an original view where the type is compliant with the source Matrix.
	 * 		  Version where a target structure is specified. It can only generate a valid type if the target
	 * 		  structure is the same as the source's
	 * 		  or a more specialized one that would preserve its static properties (e.g., symmetric reference
	 * 		  to a square matrix -- any assumption based on symmetry would not break those based on square).
	 * 		  The function guarantees the created view is non-overlapping with other existing views only when the
	 * 		  check can be performed in constant time.
	 *
	 * @tparam TargetStructure  The target structure of the new view. It should verify
	 *                          <code> alp::is_in<Structure, TargetStructure::inferred_structures> </code>.
	 * @tparam SourceMatrix     The type of the source matrix
	 *
	 * @param source            The source ALP matrix
	 *
	 * @return A new original view over the source ALP matrix.
	 *
     * \parblock
     * \par Performance semantics.
     *        -# This function performs
     *           \f$ \Theta(nref) \f$ amount of work where \f$ nref \f$ is the number
	 * 			 of available views of \a source.
     *        -# A call to this function may use \f$ \mathcal{O}(1) \f$ bytes
     *           of memory beyond the memory in use at the function call entry.
     *        -# This function may make system calls.
     * \endparblock
	 *
	 */
	template<
		typename TargetStructure,
		typename SourceMatrix,
		std::enable_if< is_matrix< SourceMatrix >::value > * = nullptr
	>
	typename internal::new_container_type_from<
		typename SourceMatrix::template view_type< view::original >::type
	>::template change_structure< TargetStructure >::type
	get_view( SourceMatrix &source ) {

		static_assert( structures::is_in< typename SourceMatrix::structure, typename TargetStructure::inferred_structures >::value,
			"Can only create a view when the target structure is compatible with the source." );

		using target_strmat_t = typename internal::new_container_type_from<
			typename SourceMatrix::template view_type< view::original >::type
		>::template change_structure< TargetStructure >::type;

		return target_strmat_t( source );
	}

	namespace internal {

		/**
		 * Implement a gather through a View over compatible Structure using provided Index Mapping Functions.
		 * The compatibility depends on the TargetStructure, SourceStructure and IMFs, and is calculated during runtime.
		 */
		template<
			typename TargetStructure, typename TargetImfR, typename TargetImfC,
			typename SourceMatrix,
			std::enable_if_t< is_matrix< SourceMatrix >::value > * = nullptr
		>
		typename internal::new_container_type_from<
			typename SourceMatrix::template view_type< view::gather >::type
		>::template change_structure< TargetStructure >::_and_::
		template change_imfr< TargetImfR >::_and_::
		template change_imfc< TargetImfC >::type
		get_view( SourceMatrix &source, TargetImfR imf_r, TargetImfC imf_c ) {

			//if( std::dynamic_pointer_cast< imf::Select >( imf_r ) || std::dynamic_pointer_cast< imf::Select >( imf_c ) ) {
			//	throw std::runtime_error("Cannot gather with imf::Select yet.");
			//}
			// No static check as the compatibility depends on IMF, which is a runtime level parameter
			//if( ! (TargetStructure::template isInstantiableFrom< Structure >( static_cast< TargetImfR & >( imf_r ), static_cast< TargetImfR & >( imf_c ) ) ) ) {
			if( ! (structures::isInstantiable< typename SourceMatrix::structure, TargetStructure >::check( imf_r, imf_c ) ) ) {
				throw std::runtime_error("Cannot gather into specified TargetStructure from provided SourceStructure and Index Mapping Functions.");
			}

			using target_t = typename internal::new_container_type_from<
				typename SourceMatrix::template view_type< view::gather >::type
			>::template change_structure< TargetStructure >::_and_::
			template change_imfr< TargetImfR >::_and_::
			template change_imfc< TargetImfC >::type;

			return target_t( source, imf_r, imf_c );
		}
	} // namespace internal

	/**
	 *
	 * @brief Generate an original view where the type is compliant with the source Matrix.
	 * Version where a range of rows and columns are selected to form a new view with specified target
	 * structure. It can only generate a valide type if the target
	 * structure is guaranteed to preserve the static properties of the source's structure.
	 * A structural check of this kind as well as non-overlapping checks with existing views of \a source
	 * are guaranteed only when each one of them incurs constant time work.
	 *
	 * @tparam TargetStructure  The target structure of the new view. It should verify
	 *                          <code> alp::is_in<Structure, TargetStructure::inferred_structures> </code>.
	 * @tparam SourceMatrix     The type of source ALP matrix
	 *
	 * @param source            The source ALP matrix
	 * @param rng_r             A valid range of rows
	 * @param rng_c             A valid range of columns
	 *
	 * @return A new original view over the source ALP matrix.
	 *
     * \parblock
     * \par Performance semantics.
     *        -# This function performs
     *           \f$ \Theta(nref) \f$ amount of work where \f$ nref \f$ is the number
	 * 			 of available views of \a source.
     *        -# A call to this function may use \f$ \mathcal{O}(1) \f$ bytes
     *           of memory beyond the memory in use at the function call entry.
     *        -# This function may make system calls.
     * \endparblock
	 *
	 */
	template<
		typename TargetStructure,
		typename SourceMatrix,
		std::enable_if_t< is_matrix< SourceMatrix >::value > * = nullptr
	>
	typename internal::new_container_type_from<
		typename SourceMatrix::template view_type< view::gather >::type
	>::template change_structure< TargetStructure >::type
	get_view(
		SourceMatrix &source,
		const utils::range& rng_r, const utils::range& rng_c
	) {

		return internal::get_view< TargetStructure >(
			source,
			std::move( imf::Strided( rng_r.count(), nrows(source), rng_r.start, rng_r.stride ) ),
			std::move( imf::Strided( rng_c.count(), ncols(source), rng_c.start, rng_c.stride ) )
		);
	}

	/**
     *
	 * @brief Generate an original view where the type is compliant with the source Matrix.
	 * Version where no target structure is specified (in this case the structure of the source type is assumed as target)
	 * with row and column selection.
	 * A structure preserving check as well as non-overlapping checks with existing views of \a source
	 * are guaranteed only when each one of them incurs constant time work.
	 *
	 * @tparam SourceMatrix     The type of source ALP matrix
	 *
	 * @param source      The source matrix
	 * @param rng_r       A valid range of rows
	 * @param rng_c       A valid range of columns
	 *
	 * @return A new original view over the source structured matrix.
	 *
     * \parblock
     * \par Performance semantics.
     *        -# This function performs
     *           \f$ \Theta(nref) \f$ amount of work where \f$ nref \f$ is the number
	 * 			 of available views of \a source.
     *        -# A call to this function may use \f$ \mathcal{O}(1) \f$ bytes
     *           of memory beyond the memory in use at the function call entry.
     *        -# This function may make system calls.
     * \endparblock
	 *
	 */

	template<
		typename SourceMatrix,
		std::enable_if_t< is_matrix< SourceMatrix >::value > * = nullptr
	>
	typename SourceMatrix::template view_type< view::gather >::type
	get_view(
		SourceMatrix &source,
		const utils::range &rng_r,
		const utils::range &rng_c
	) {

		return internal::get_view< typename SourceMatrix::structure >(
			source,
			imf::Strided( rng_r.count(), nrows(source), rng_r.start, rng_r.stride ),
			imf::Strided( rng_c.count(), ncols(source), rng_c.start, rng_c.stride ) );
	}

	/**
	 *
	 * @brief Generate a vector view on a column of the source matrix.
	 *
	 * @tparam SourceMatrix The type of the source ALP matrix
	 *
	 * @param source        The source matrix
	 * @param rng_r         A valid range of rows
	 * @param sel_c         A valid column index
	 *
	 * @return A new gather view over the source ALP matrix.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -# This function performs
	 *           \f$ \Theta(nref) \f$ amount of work where \f$ nref \f$ is the number
	 *           of available views of \a source.
	 *        -# A call to this function may use \f$ \mathcal{O}(1) \f$ bytes
	 *           of memory beyond the memory in use at the function call entry.
	 *        -# This function may make system calls.
	 * \endparblock
	 *
	 */
	template<
		typename SourceMatrix,
		std::enable_if_t< is_matrix< SourceMatrix >::value > * = nullptr
	>
	typename internal::new_container_type_from<
		typename SourceMatrix::template view_type< view::gather >::type
	>::template change_container< alp::Vector >::_and_::
	template change_structure< structures::General >::_and_::
	template change_imfc< imf::Constant >::type
	get_view(
		SourceMatrix &source,
		const utils::range &rng_r,
		const size_t &sel_c
	) {
		using target_t = typename internal::new_container_type_from<
			typename SourceMatrix::template view_type< view::gather >::type
		>::template change_container< alp::Vector >::_and_::
		template change_structure< structures::General >::_and_::
		template change_imfc< imf::Constant >::type;

		return target_t(
			source,
			imf::Strided( rng_r.count(), nrows( source ), rng_r.start, rng_r.stride ),
			imf::Constant( 1, ncols( source ), sel_c )
		);
	}

	/**
	 *
	 * @brief Generate a vector view on a row of the source matrix.
	 *
	 * @tparam SourceMatrix The type of the source ALP matrix
	 *
	 * @param source        The source matrix
	 * @param sel_r         A valid row index
	 * @param rng_c         A valid range of columns
	 *
	 * @return A new gather view over the source ALP matrix.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -# This function performs
	 *           \f$ \Theta(nref) \f$ amount of work where \f$ nref \f$ is the number
	 *           of available views of \a source.
	 *        -# A call to this function may use \f$ \mathcal{O}(1) \f$ bytes
	 *           of memory beyond the memory in use at the function call entry.
	 *        -# This function may make system calls.
	 * \endparblock
	 *
	 * \note \internal Row-view is implemented as a column view over a
	 *                 tranposed source matrix
	 *
	 */
	template<
		typename SourceMatrix,
		std::enable_if_t< is_matrix< SourceMatrix >::value > * = nullptr
	>
	typename internal::new_container_type_from<
		typename SourceMatrix::template view_type< view::transpose >::type::template view_type< view::gather >::type
	>::template change_container< alp::Vector >::_and_::
	template change_structure< structures::General >::_and_::
	template change_imfc< imf::Constant >::type
	get_view(
		SourceMatrix &source,
		const size_t &sel_r,
		const utils::range &rng_c
	) {
		auto source_transposed = get_view< view::transpose >( source );
		return get_view( source_transposed, rng_c, sel_r );
	}

	/**
	 *
	 * Generate a dynamic gather view where the type is compliant with the source Matrix.
	 * Version where a selection of rows and columns, expressed as vectors of indices,
	 * forms a new view with specified target structure.
	 *
	 * @tparam TargetStructure The target structure of the new view. It should verify
	 *                         <code> alp::is_in<Structure, TargetStructure::inferred_structures> </code>.
	 * @tparam SourceMatrix    The type of the source ALP matrix
	 * @tparam SelectVectorR   The type of the ALP vector defining permutation for rows
	 * @tparam SelectVectorC   The type of the ALP vector defining permutation for columns
	 *
	 * @param source           The source ALP matrix
	 * @param sel_r            A valid permutation vector of a subset of row indices
	 * @param sel_c            A valid permutation vector of a subset of column indices
	 *
	 * @return A new gather view over the source ALP matrix.
	 *
	 */
	template<
		typename TargetStructure,
		typename SourceMatrix,
		typename SelectVectorR, typename SelectVectorC,
		std::enable_if_t<
			is_matrix< SourceMatrix >::value &&
			is_vector< SelectVectorR >::value &&
			is_vector< SelectVectorC >::value
		> * = nullptr
	>
	typename internal::new_container_type_from<
		typename SourceMatrix::template view_type< view::gather >::type
	>::template change_structure< TargetStructure >::_and_::
	template change_imfr< imf::Select >::_and_::
	template change_imfc< imf::Select >::type
	get_view(
		SourceMatrix &source,
		const SelectVectorR &sel_r,
		const SelectVectorC &sel_c
	) {
		return internal::get_view< TargetStructure >(
			source,
			imf::Select( nrows( source ), sel_r ),
			imf::Select( ncols( source ), sel_c )
		);
	}


	/** Definitions of previously declared global methods that operate on ALP Matrix */
	namespace internal {

		template<
			typename MatrixType,
			std::enable_if_t< is_matrix< MatrixType>::value > * = nullptr
		>
		bool getInitialized( const MatrixType &A ) noexcept {
			return static_cast< const MatrixBase< typename MatrixType::base_type > & >( A ).template getInitialized();
		}

		template<
			typename MatrixType,
			std::enable_if_t< is_matrix< MatrixType>::value > * = nullptr
		>
		void setInitialized( MatrixType &A, const bool initialized ) noexcept {
			return static_cast< MatrixBase< typename MatrixType::base_type > & >( A ).template setInitialized( initialized );
		}

		template< typename DerivedMatrix >
		std::pair< size_t, size_t > dims( const MatrixBase< DerivedMatrix > &A ) noexcept {
			return A.dims();
		}

		/** Access the matrix element.
		 *
		 * @tparam    MatrixType ALP Matrix type
		 *
		 * @param[in] A             matrix to be accessed
		 * @param[in] storageIndex  index in the physical iteration space
		 *
		 * @return For container matrices, returns a constant reference to the
		 *         element at the given physical position of matrix A.
		 *         For functor view matrices, returns a value corresponding to
		 *         the given physical position of matrix A.
		 *
		 * \note   This method may be used to access only elements local to the processor.
		 */
		template<
			typename MatrixType,
			std::enable_if< is_matrix< MatrixType >::value > * = nullptr
		>
		const typename MatrixType::access_type access( const MatrixType &A, const typename MatrixType::storage_index_type &storageIndex ) {
			return static_cast<
				const MatrixBase< typename MatrixType::base_type > &
			>( A ).template access< typename MatrixType::access_type, typename MatrixType::storage_index_type >( storageIndex );
		}

		/** Non-constant variant. **/
		template<
			typename MatrixType,
			std::enable_if< is_matrix< MatrixType >::value > * = nullptr
		>
		typename MatrixType::access_type access( MatrixType &A, const typename MatrixType::storage_index_type &storageIndex ) {
			return static_cast<
				MatrixBase< typename MatrixType::base_type > &
			>( A ).template access< typename MatrixType::access_type, typename MatrixType::storage_index_type >( storageIndex );
		}

		/** Return a storage index in the physical layout.
		 *
		 * @tparam    MatrixType ALP Matrix type
		 *
		 * @param[in] A  matrix to be accessed
		 * @param[in] i  row-index in the logical layout
		 * @param[in] j  column-index in the logical layout
		 * @param[in] s  process ID
		 * @param[in] P  total number of processors
		 *
		 * @return For container matrices, returns a constant reference to the
		 *         element at the given physical position of matrix A.
		 *         For functor view matrices, returns a value corresponding to
		 *         the given physical position of matrix A.
		 *
		 */
		template<
			typename MatrixType,
			std::enable_if< is_matrix< MatrixType >::value > * = nullptr
		>
		typename MatrixType::storage_index_type getStorageIndex( const MatrixType &A, const size_t i, const size_t j, const size_t s, const size_t P ) {
			return static_cast< const MatrixBase< typename MatrixType::base_type > & >( A ).template getStorageIndex< typename MatrixType::storage_index_type >( i, j, s, P );
		}

		/** Return a pair of coordinates in logical layout.
		 *
		 * @tparam    MatrixType ALP Matrix type
		 *
		 * @param[in] A             matrix to be accessed
		 * @param[in] storageIndex  storage index in the physical layout.
		 * @param[in] s             process ID
		 * @param[in] P             total number of processors
		 *
		 * @return Returns a pair of coordinates in logical iteration space
		 *         that correspond to the provided storage index in the
		 *         physical iteration space.
		 *
		 */
		template< typename MatrixType >
		std::pair< size_t, size_t > getCoords( const MatrixType &A, const size_t storageIndex, const size_t s, const size_t P );

		/** Get the reference to the AMF of a storage-based matrix */
		template<
			typename MatrixType,
			std::enable_if< internal::is_storage_based< MatrixType >::value > * = nullptr
		>
		const typename MatrixType::amf_type &getAmf( const MatrixType &A ) noexcept {
			return A.getAmf();
		}

	} // namespace internal

	template< typename D, typename Structure, typename View, typename ImfR, typename ImfC, enum Backend backend >
	size_t nrows( const Matrix< D, Structure, Density::Dense, View, ImfR, ImfC, backend > &A ) noexcept {
		return dims( A ).first;
	}

	template< typename D, typename Structure, typename View, typename ImfR, typename ImfC, enum Backend backend >
	size_t ncols( const Matrix< D, Structure, Density::Dense, View, ImfR, ImfC, backend > &A ) noexcept {
		return dims( A ).second;
	}

	template< typename D, typename Structure, typename View, typename ImfR, typename ImfC, enum Backend backend >
	std::pair< size_t, size_t > dims( const Matrix< D, Structure, Density::Dense, View, ImfR, ImfC, backend > &A ) noexcept {
		return internal::dims( static_cast< const internal::MatrixBase<
			typename Matrix< D, Structure, Density::Dense, View, ImfR, ImfC, backend >::base_type > & > ( A ) );
	}

	template<
		typename MatrixType,
		std::enable_if< internal::is_storage_based< MatrixType >::value > * = nullptr
	>
	size_t internal::getStorageDimensions( const MatrixType &A ) noexcept {
		static_assert( is_storage_based< MatrixType >::value, "getStorageDimensions supported only for storage-based containers.");
		return static_cast< const typename MatrixType::base_type & >( A ).getStorageDimensions();
	}

	namespace structures {

		template<
			size_t band,
			typename MatrixType,
			std::enable_if_t< is_matrix< MatrixType >::value > * = nullptr
		>
		std::ptrdiff_t get_lower_limit( const MatrixType &A ) {

			return structures::get_lower_limit< band, typename MatrixType::structure >( nrows( A ) );

		}

		template<
			size_t band,
			typename MatrixType,
			std::enable_if_t< is_matrix< MatrixType >::value > * = nullptr
		>
		std::ptrdiff_t get_upper_limit( const MatrixType &A ) {

			return structures::get_upper_limit< band, typename MatrixType::structure >( ncols( A ) );

		}

		/**
		 * Specialization for reference backend.
		 * @see alp::structures::calculate_row_coordinate_limits
		 */
		template<
			size_t band_index, typename MatrixType,
			std::enable_if_t<
				is_matrix< MatrixType >::value
			> * = nullptr
		>
		std::pair< size_t, size_t > calculate_row_coordinate_limits( const MatrixType &A ) {

			using Structure = typename MatrixType::structure;

			static_assert(
				band_index < std::tuple_size< typename Structure::band_intervals >::value,
				"Provided band index is out of bounds."
			);

			// cast matrix dimensions to signed integer to allow for comparison with negative numbers
			const std::ptrdiff_t M = static_cast< std::ptrdiff_t >( nrows( A ) );
			const std::ptrdiff_t N = static_cast< std::ptrdiff_t >( ncols( A ) );

			// band limits are negated and inverted due to different orientation
			// of coordinate system of band and matrix dimensions.
			const std::ptrdiff_t l = -structures::get_upper_limit< band_index >( A );
			const std::ptrdiff_t u = N - structures::get_lower_limit< band_index >( A );

			// fit the limits within the matrix dimensions
			const size_t lower_limit = static_cast< size_t >( std::max( std::min( l, M ), static_cast< std::ptrdiff_t >( 0 ) ) );
			const size_t upper_limit = static_cast< size_t >( std::max( std::min( u, M ), static_cast< std::ptrdiff_t >( 0 ) ) );

			assert( lower_limit <= upper_limit );

			return std::make_pair( lower_limit, upper_limit );
		}

		/**
		 * Specialization for reference backend.
		 * @see alp::structures::calculate_column_coordinate_limits
		 */
		template<
			size_t band_index, typename MatrixType,
			std::enable_if_t<
				is_matrix< MatrixType >::value
			> * = nullptr
		>
		std::pair< size_t, size_t > calculate_column_coordinate_limits( const MatrixType &A, const size_t row ) {

			using Structure = typename MatrixType::structure;

			// Declaring this to avoid static casts to std::ptrdiff_t in std::min and std::max calls
			const std::ptrdiff_t signed_zero = 0;

			static_assert(
				band_index < std::tuple_size< typename Structure::band_intervals >::value,
				"Provided band index is out of bounds."
			);

			assert( row < nrows( A ) );

			// cast matrix dimensions to signed integer to allow for comparison with negative numbers
			const std::ptrdiff_t N = static_cast< std::ptrdiff_t >( ncols( A ) );

			constexpr bool is_sym = structures::is_a< Structure, structures::Symmetric >::value;
			// Temporary until adding multiple symmetry directions
			constexpr bool sym_up = is_sym;

			// Band limits
			const std::ptrdiff_t l = structures::get_lower_limit< band_index >( A );
			const std::ptrdiff_t u = structures::get_upper_limit< band_index >( A );

			// Band limits taking into account symmetry
			const std::ptrdiff_t sym_l = is_sym && sym_up ? std::max( signed_zero, l ) : l;
			const std::ptrdiff_t sym_u = is_sym && !sym_up ? std::min( signed_zero, u ) : u;

			// column coordinate lower and upper limits considering the provided row coordinate
			const std::ptrdiff_t sym_l_row = static_cast< std::ptrdiff_t >( row ) + sym_l;
			const std::ptrdiff_t sym_u_row = sym_l_row + ( sym_u - sym_l );

			// fit the limits within the matrix dimensions
			const size_t lower_limit = static_cast< size_t >( std::max( std::min( sym_l_row, N ), signed_zero ) );
			const size_t upper_limit = static_cast< size_t >( std::max( std::min( sym_u_row, N ), signed_zero ) );

			assert( lower_limit <= upper_limit );

			return std::make_pair( lower_limit, upper_limit );
		}

	} // namespace structures
} // namespace alp

#endif // end ``_H_ALP_AMF_BASED_MATRIX''
