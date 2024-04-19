
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

#ifndef _H_ALP_DISPATCH_BLAS2
#define _H_ALP_DISPATCH_BLAS2

#include <cstddef>

#include <alp/backends.hpp>
#include <alp/config.hpp>
#include <alp/rc.hpp>
#include <alp/matrix.hpp>
#include <graphblas/utils/iscomplex.hpp>

#define NO_CAST_OP_ASSERT( x, y, z )                                           \
	static_assert( x,                                                          \
		"\n\n"                                                                 \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"*     ERROR      | " y " " z ".\n"                                    \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"* Possible fix 1 | Remove no_casting from the template parameters "   \
		"in this call to " y ".\n"                                             \
		"* Possible fix 2 | For all mismatches in the domains of input "       \
		"parameters and the operator domains, as specified in the "            \
		"documentation of the function " y ", supply an input argument of "    \
		"the expected type instead.\n"                                         \
		"* Possible fix 3 | Provide a compatible operator where all domains "  \
		"match those of the input parameters, as specified in the "            \
		"documentation of the function " y ".\n"                               \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );

namespace alp {

	namespace internal {

		/**
		 * Applies the provided function to each element of the given band.
		 * This function is called by the public eWiseLambda variant.
		 * Forward declaration. Specializations handle bound checking.
		 */
		template<
			size_t BandIndex, typename Func,
			typename DataType, typename Structure, typename View, typename ImfR, typename ImfC,
			std::enable_if_t<
				BandIndex >= std::tuple_size< typename Structure::band_intervals >::value
			> * = nullptr
		>
		RC eWiseLambda(
			const Func f,
			alp::Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, dispatch > &A
		);

		/** Specialization for an out-of-bounds band index */
		template<
			size_t BandIndex, typename Func,
			typename DataType, typename Structure, typename View, typename ImfR, typename ImfC,
			std::enable_if_t<
				BandIndex >= std::tuple_size< typename Structure::band_intervals >::value
			> * = nullptr
		>
		RC eWiseLambda(
			const Func f,
			alp::Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, dispatch > &A
		) {
			(void)f;
			(void)A;
			// nothing to do
			return SUCCESS;
		}

		/**
		 * Specialization for a within-the-range band index.
		 * Applies the provided function to each element of the given band.
		 * Upon completion, calls itself for the next band.
		 */
		template<
			size_t band_index, typename Func,
			typename DataType, typename Structure, typename View, typename ImfR, typename ImfC,
			std::enable_if_t<
				band_index < std::tuple_size< typename Structure::band_intervals >::value
			> * = nullptr
		>
		RC eWiseLambda(
			const Func f,
			alp::Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, dispatch > &A
		) {
			const auto i_limits = structures::calculate_row_coordinate_limits< band_index >( A );

			for( size_t i = i_limits.first; i < i_limits.second; ++i ) {

				const auto j_limits = structures::calculate_column_coordinate_limits< band_index >( A, i );

				for( size_t j = j_limits.first; j < j_limits.second; ++j ) {
					auto &a_val = internal::access( A, internal::getStorageIndex( A, i, j ) );
					f( i, j, a_val );
				}
			}
			return eWiseLambda< band_index + 1 >( f, A );
		}

	} // namespace internal

	/**
	 * Delegates to single-band variant.
	 *
	 * @see alp::eWiseLambda for the user-level specification.
	 */
	template<
		typename Func,
		typename DataType, typename Structure, typename View, typename ImfR, typename ImfC
	>
	RC eWiseLambda(
		const Func f,
		Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, dispatch > &A
	) {
#ifdef _DEBUG
		std::cout << "entering alp::eWiseLambda (matrices, dispatch ). A is " << alp::nrows( A ) << " by " << alp::ncols( A ) << " and holds " << alp::nnz( A ) << " nonzeroes.\n";
#endif
		return internal::eWiseLambda< 0 >( f, A );
	}

	/**
	 * This function provides dimension checking and will defer to the below
	 * function for the actual implementation.
	 *
	 * @see alp::eWiseLambda for the user-level specification.
	 */
	template<
		typename Func,
		typename DataType1, typename DataStructure1, typename DataView1, typename DataImfR1, typename DataImfC1,
		typename DataType2, typename DataStructure2, typename DataView2, typename DataImfR2, typename DataImfC2,
		typename... Args
	>
	RC eWiseLambda(
		const Func f,
		Matrix< DataType1, DataStructure1, Density::Dense, DataView1, DataImfR1, DataImfC1, dispatch > &A,
		const Vector< DataType2, DataStructure2, Density::Dense, DataView2, DataImfR2, DataImfC2, dispatch > &x,
		Args const &... args
	) {
		// do size checking
		if( !( getLength( x ) == nrows( A ) || getLength( x ) == ncols( A ) ) ) {
			std::cerr << "Mismatching dimensions: given vector of size " << size( x )
				<< " has nothing to do with either matrix dimension (" << nrows( A ) << " nor " << ncols( A ) << ").\n";
			return MISMATCH;
		}

		return eWiseLambda( f, A, args... );
	}

	namespace internal {

		/**
		 * Applies fold to all elements of the given band
		 * Depending on the values of left and scalar, performs the following variants:
		 * - left == true  && scalar == true:  C = C . alpha
		 * - left == true  && scalar == false: C = C . A
		 * - left == false && scalar == true:  C = alpha . C
		 * - left == false && scalar == false: C = A . C
		 * This variants handles out-of-bounds band index.
		 * All variants assume compatible parameters:
		 *   - matching structures
		 *   - matching dynamic sizes
		 */
		template<
			size_t band_index,
			bool left, // if true, performs foldl, otherwise foldr
			bool scalar,
			Descriptor descr,
			class Operator,
			typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
			typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
			typename InputTypeScalar, typename InputStructureScalar,
			std::enable_if_t<
				band_index >= std::tuple_size< typename IOStructure::band_intervals >::value
			> * = nullptr
		>
		RC fold_matrix_band_generic(
			alp::Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, dispatch > *C,
			const alp::Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, dispatch > *A,
			const alp::Scalar< InputTypeScalar, InputStructureScalar, dispatch > *alpha,
			const Operator &op,
			const std::enable_if_t<
				!alp::is_object< IOType >::value &&
				!alp::is_object< InputType >::value &&
				alp::is_operator< Operator >::value
			> * const = nullptr
		) {
			(void) C;
			(void) A;
			(void) alpha;
			(void) op;
			return SUCCESS;
		}

		/** Specialization for band index within the bounds */
		template<
			size_t band_index,
			bool left, // if true, performs foldl, otherwise foldr
			bool scalar,
			Descriptor descr,
			class Operator,
			typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
			typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
			typename InputTypeScalar, typename InputStructureScalar,
			std::enable_if_t<
				band_index < std::tuple_size< typename IOStructure::band_intervals >::value
			> * = nullptr
		>
		RC fold_matrix_band_generic(
			alp::Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, dispatch > *C,
			const alp::Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, dispatch > *A,
			const alp::Scalar< InputTypeScalar, InputStructureScalar, dispatch > *alpha,
			const Operator &op,
			const std::enable_if_t<
				!alp::is_object< IOType >::value &&
				!alp::is_object< InputType >::value &&
				alp::is_operator< Operator >::value
			> * const = nullptr
		) {
			// Ensure that the provided containers are compatible with static configuration
			assert( C != nullptr );
			if( scalar ) {
				assert( alpha != nullptr );
			} else {
				assert( A != nullptr );
			}

			constexpr bool is_sym_c = structures::is_a< IOStructure, structures::Symmetric >::value;
			constexpr bool is_sym_a = structures::is_a< InputStructure, structures::Symmetric >::value;

			// Temporary until adding multiple symmetry directions
			constexpr bool sym_up_c = is_sym_c;
			constexpr bool sym_up_a = is_sym_a;

			// It is assumed without checking that bands of A are a subset of bands of C. TODO: Implement proper check.
			// If input is scalar, iterating over bands of C, otherwise over bands of A
			const auto i_limits = scalar ?
				structures::calculate_row_coordinate_limits< band_index >( *C ) :
				structures::calculate_row_coordinate_limits< band_index >( *A );

			for( size_t i = i_limits.first; i < i_limits.second; ++i ) {

				const auto j_limits = scalar ?
					structures::calculate_column_coordinate_limits< band_index >( *C, i ) :
					structures::calculate_column_coordinate_limits< band_index >( *A, i );

				for( size_t j = j_limits.first; j < j_limits.second; ++j ) {
					auto &IO_val = internal::access( *C, internal::getStorageIndex( *C, i, j ) );

					if( scalar ) {
						if( left ) {
							// C = C . alpha
							(void) internal::foldl( IO_val, **alpha, op );
						} else {
							// C = alpha . C
							(void) internal::foldr( **alpha, IO_val, op );
						}
					} else {
						// C = A . C
						// Calculate indices to 'A' depending on matching symmetry with 'C'
						const size_t A_i = ( sym_up_c == sym_up_a ) ? i : j;
						const size_t A_j = ( sym_up_c == sym_up_a ) ? j : i;
						const auto &A_val = internal::access( *A, internal::getStorageIndex( *A, A_i, A_j ) );

						if( left ) {
							// C = C . A
							(void) internal::foldl( IO_val, A_val, op );
						} else {
							// C = A . C
							(void) internal::foldr( A_val, IO_val, op );
						}
					}
				}
			}
			return fold_matrix_band_generic<
				band_index + 1, left, scalar, descr
			>( C, A, alpha, op );
		}

		/**
		 * \internal general elementwise matrix application that all eWiseApply variants refer to.
		 */
		template<
			bool left, bool scalar,
			Descriptor descr,
			class Operator,
			typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
			typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
			typename InputTypeScalar, typename InputStructureScalar
		>
		RC fold_matrix_generic(
			alp::Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, dispatch > *C,
			const alp::Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, dispatch > *A,
			const alp::Scalar< InputTypeScalar, InputStructureScalar, dispatch > *alpha,
			const Operator &op,
			const std::enable_if_t<
				!alp::is_object< IOType >::value &&
				!alp::is_object< InputType >::value &&
				alp::is_operator< Operator >::value
			> * const = nullptr
		) {

#ifdef _DEBUG
			std::cout << "In alp::internal::fold_matrix_generic\n";
#endif

			// run-time checks
			// TODO: support left/right_scalar
			const size_t m = alp::nrows( *C );
			const size_t n = alp::ncols( *C );

			if( !scalar ){
				assert( A != nullptr );
				if( m != nrows( *A ) || n != ncols( *A ) ) {
					return MISMATCH;
				}
			}

			// delegate to single-band variant
			return fold_matrix_band_generic< 0, left, scalar, descr >( C, A, alpha, op );
		}

	} // namespace internal

	/** Folds element-wise B into A, operator variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Operator
	>
	RC foldl(
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, dispatch > &A,
		const Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, dispatch > &B,
		const Operator &op = Operator(),
		const std::enable_if_t<
			!alp::is_object< IOType >::value && ! alp::is_object< InputType >::value && alp::is_operator< Operator >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D1, IOType >::value ),
			"alp::foldl",
			"called on a matrix A of a type that does not match the first domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D2, InputType >::value ),
			"alp::foldl",
			"called with a matrix B of a type that does not match the second domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D3, IOType >::value ),
			"alp::foldl",
			"called on a matrix A of a type that does not match the third domain "
			"of the given operator"
		);
		// TODO: check that InputStructure's bands are identical to IOStructure's bands

		constexpr bool left = true;
		constexpr bool scalar = false;
		constexpr Scalar< InputType, structures::General, dispatch > *no_scalar = nullptr;
		return internal::fold_matrix_generic< left, scalar, descr >( &A, &B, no_scalar, op ) ;
	}

	/** Folds element-wise beta into A, operator variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Operator
	>
	RC foldl(
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, dispatch > &A,
		const Scalar< InputType, InputStructure, dispatch > &beta,
		const Operator &op = Operator(),
		const std::enable_if_t<
			!alp::is_object< IOType >::value && !alp::is_object< InputType >::value && alp::is_operator< Operator >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D1, IOType >::value ),
			"alp::foldl",
			"called on a matrix A of a type that does not match the first domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D2, InputType >::value ),
			"alp::foldl",
			"called with a scalar beta of a type that does not match the second domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D3, IOType >::value ),
			"alp::foldl",
			"called on a matrix A of a type that does not match the third domain "
			"of the given operator"
		);
		// TODO: check that InputStructure's bands are identical to IOStructure's bands

		constexpr bool left = true;
		constexpr bool scalar = true;
		constexpr Matrix< InputType, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, dispatch > *no_matrix = nullptr;
		return internal::fold_matrix_generic< left, scalar, descr >( &A, no_matrix, &beta, op ) ;
	}

	/**
	 * Returns a view over the input matrix returning conjugate of the accessed element.
	 * This avoids materializing the resulting container.
	 * The elements are calculated lazily on access.
	 *
	 * @tparam descr      	    The descriptor to be used (descriptors::no_operation
	 *                    	    if left unspecified).
	 * @tparam InputType  	    The value type of the input matrix.
	 * @tparam InputStructure   The Structure type applied to the input matrix.
	 * @tparam InputView        The view type applied to the input matrix.
	 *
	 * @param A      The input matrix
	 *
	 * @return Matrix view over a lambda function defined in this function.
	 *
	 * Specialization for non-square matrices. This distinction is necessary due
	 * to different constructor signature for square and non-square matrices.
	 *
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename Structure, typename View, typename ImfR, typename ImfC,
		std::enable_if_t<
			!structures::is_a< Structure, structures::Square >::value
		> * = nullptr
	>
	Matrix<
		DataType, Structure, Density::Dense,
		view::Functor< std::function< void( DataType &, const size_t, const size_t ) > >,
		imf::Id, imf::Id,
		dispatch
	>
	conjugate(
		const Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, dispatch > &A,
		const std::enable_if_t<
			!alp::is_object< DataType >::value
		> * const = nullptr
	) {

		std::function< void( DataType &, const size_t, const size_t ) > data_lambda =
			[ &A ]( DataType &result, const size_t i, const size_t j ) {
				result = grb::utils::is_complex< DataType >::conjugate(
					internal::access( A, internal::getStorageIndex( A, i, j ) )
				);
			};
		std::function< bool() > init_lambda =
			[ &A ]() -> bool {
				return internal::getInitialized( A );
			};

		return Matrix<
			DataType,
			Structure,
			Density::Dense,
			view::Functor< std::function< void( DataType &, const size_t, const size_t ) > >,
			imf::Id, imf::Id,
			dispatch
			>(
				init_lambda,
				nrows( A ),
				ncols( A ),
				data_lambda
			);

	}

	/** Specialization for square matrices */
	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename Structure, typename View, typename ImfR, typename ImfC,
		std::enable_if_t<
			structures::is_a< Structure, structures::Square >::value
		> * = nullptr
	>
	Matrix<
		DataType, Structure, Density::Dense,
		view::Functor< std::function< void( DataType &, const size_t, const size_t ) > >,
		imf::Id, imf::Id,
		dispatch
	>
	conjugate(
		const Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, dispatch > &A,
		const std::enable_if_t<
			!alp::is_object< DataType >::value
		> * const = nullptr
	) {

		std::function< void( DataType &, const size_t, const size_t ) > data_lambda =
			[ &A ]( DataType &result, const size_t i, const size_t j ) {
				result = grb::utils::is_complex< DataType >::conjugate(
					internal::access( A, internal::getStorageIndex( A, i, j ) )
				);
			};
		std::function< bool() > init_lambda =
			[ &A ]() -> bool {
				return internal::getInitialized( A );
			};

		return Matrix<
			DataType,
			Structure,
			Density::Dense,
			view::Functor< std::function< void( DataType &, const size_t, const size_t ) > >,
			imf::Id, imf::Id,
			dispatch
			>(
				init_lambda,
				nrows( A ),
				data_lambda
			);

	}

} // end namespace ``alp''

#undef NO_CAST_OP_ASSERT

#endif // end ``_H_ALP_DISPATCH_BLAS2''

