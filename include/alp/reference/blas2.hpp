
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

#ifndef _H_ALP_REFERENCE_BLAS2
#define _H_ALP_REFERENCE_BLAS2

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

	/**
	 * \addtogroup reference
	 * @{
	 */

	/** \internal Delegates to fully masked variant */
	template< Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType3, typename InputStructure3, typename InputView3, typename InputImfR3, typename InputImfC3,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Ring
	>
	RC vxm( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & u,
		const Vector< InputType3, InputStructure3, Density::Dense, InputView3, InputImfR3, InputImfC3, reference > & mask,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & v,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< alp::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const Vector< bool, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > empty_mask( 0 );
		return vxm< descr, true, false >( u, mask, v, empty_mask, A, ring );
	}

	/** \internal Delegates to fully masked variant */
	template< Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType3, typename InputStructure3, typename InputView3, typename InputImfR3, typename InputImfC3,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class AdditiveMonoid,
		class MultiplicativeOperator
	>
	RC vxm( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & u,
		const Vector< InputType3, InputStructure3, Density::Dense, InputView3, InputImfR3, InputImfC3, reference > & mask,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & v,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< alp::is_monoid< AdditiveMonoid >::value && alp::is_operator< MultiplicativeOperator >::value && ! alp::is_object< IOType >::value &&
				! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && ! alp::is_object< InputType3 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		const alp::Vector< bool, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > empty_mask( 0 );
		return vxm< descr, true, false >( u, mask, v, empty_mask, A, add, mul );
	}

	/** \internal Delegates to vxm_generic. */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType3, typename InputStructure3, typename InputView3, typename InputImfR3, typename InputImfC3,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType4, typename InputStructure4, typename InputView4, typename InputImfR4, typename InputImfC4,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Ring
	>
	RC vxm( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & u,
		const Vector< InputType3, InputStructure3, Density::Dense, InputView3, InputImfR3, InputImfC3, reference > & mask,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & v,
		const Vector< InputType4, InputStructure4, Density::Dense, InputView4, InputImfR4, InputImfC4, reference > & v_mask,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< alp::is_semiring< Ring >::value, void >::type * const = NULL ) {
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType1 = typename Ring::D1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2 = typename Ring::D2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2
	>
	RC vxm( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & u,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & v,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< alp::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const Vector< bool, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > empty_mask( 0 );
		return vxm< descr, false, false >( u, empty_mask, v, empty_mask, A, ring );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class AdditiveMonoid, class MultiplicativeOperator
	>
	RC vxm( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & u,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & v,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< alp::is_monoid< AdditiveMonoid >::value && alp::is_operator< MultiplicativeOperator >::value && ! alp::is_object< IOType >::value &&
				! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		const alp::Vector< bool, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > empty_mask( 0 );
		return vxm< descr, false, false >( u, empty_mask, v, empty_mask, A, add, mul );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType3 = bool, typename InputStructure3, typename InputView3, typename InputImfR3, typename InputImfC3,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		class Ring
	>
	RC mxv( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & u,
		const Vector< InputType3, InputStructure3, Density::Dense, InputView3, InputImfR3, InputImfC3, reference > & mask,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & v,
		const Ring & ring,
		const typename std::enable_if< alp::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const Vector< bool, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > empty_mask( 0 );
		return mxv< descr, true, false >( u, mask, A, v, empty_mask, ring );
	}

	/** \internal Delegates to vxm_generic */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType3, typename InputStructure3, typename InputView3, typename InputImfR3, typename InputImfC3,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType4, typename InputStructure4, typename InputView4, typename InputImfR4, typename InputImfC4,
		class Ring
	>
	RC mxv( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & u,
		const Vector< InputType3, InputStructure3, Density::Dense, InputView3, InputImfR3, InputImfC3, reference > & mask,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & v,
		const Vector< InputType4, InputStructure4, Density::Dense, InputView4, InputImfR4, InputImfC4, reference > & v_mask,
		const Ring & ring,
		const typename std::enable_if< alp::is_semiring< Ring >::value, void >::type * const = NULL ) {

		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * \internal Delegates to fully masked variant.
	 */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType2 = typename Ring::D2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		typename InputType1 = typename Ring::D1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1
	>
	RC mxv( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & u,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & v,
		const Ring & ring,
		const typename std::enable_if< alp::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const Vector< bool, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > empty_mask( 0 );
		return mxv< descr, false, false >( u, empty_mask, A, v, empty_mask, ring );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		class AdditiveMonoid, class MultiplicativeOperator
	>
	RC mxv( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & u,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & v,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< alp::is_monoid< AdditiveMonoid >::value && alp::is_operator< MultiplicativeOperator >::value && ! alp::is_object< IOType >::value &&
				! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		const alp::Vector< bool, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > empty_mask( 0 );
		return mxv< descr, false, false >( u, empty_mask, A, v, empty_mask, add, mul );
	}

	/**
	 * \internal Delegates to vxm_generic
	 */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType3, typename InputStructure3, typename InputView3, typename InputImfR3, typename InputImfC3,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType4, typename InputStructure4, typename InputView4, typename InputImfR4, typename InputImfC4,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class AdditiveMonoid,
		class MultiplicativeOperator
	>
	RC vxm( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & u,
		const Vector< InputType3, InputStructure3, Density::Dense, InputView3, InputImfR3, InputImfC3, reference > & mask,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & v,
		const Vector< InputType4, InputStructure4, Density::Dense, InputView4, InputImfR4, InputImfC4, reference > & v_mask,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< alp::is_monoid< AdditiveMonoid >::value && alp::is_operator< MultiplicativeOperator >::value && ! alp::is_object< IOType >::value &&
				! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && ! alp::is_object< InputType3 >::value && ! alp::is_object< InputType4 >::value &&
				! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {

		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * \internal Delegates to vxm_generic.
	 */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		typename InputType3, typename InputStructure3, typename InputView3, typename InputImfR3, typename InputImfC3,
		typename InputType4, typename InputStructure4, typename InputView4, typename InputImfR4, typename InputImfC4,
		class AdditiveMonoid,
		class MultiplicativeOperator
	>
	RC mxv( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & u,
		const Vector< InputType3, InputStructure3, Density::Dense, InputView3, InputImfR3, InputImfC3, reference > & mask,
		const Matrix< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & A,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & v,
		const Vector< InputType4, InputStructure4, Density::Dense, InputView4, InputImfR4, InputImfC4, reference > & v_mask,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< alp::is_monoid< AdditiveMonoid >::value && alp::is_operator< MultiplicativeOperator >::value && ! alp::is_object< IOType >::value &&
				! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && ! alp::is_object< InputType3 >::value && ! alp::is_object< InputType4 >::value &&
				! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {

		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	namespace internal {

		/**
		 * Applies the provided function to each element of the given band.
		 * This function is called by the public eWiseLambda variant.
		 * Forward declaration. Specializations handle bound checking.
		 */
		template<
			size_t BandIndex, typename Func,
			typename DataType, typename Structure, typename View, typename ImfR, typename ImfC,
			typename std::enable_if_t<
				BandIndex >= std::tuple_size< typename Structure::band_intervals >::value
			> * = nullptr
		>
		RC eWiseLambda(
			const Func f,
			alp::Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, reference > &A
		);

		/** Specialization for an out-of-bounds band index */
		template<
			size_t BandIndex, typename Func,
			typename DataType, typename Structure, typename View, typename ImfR, typename ImfC,
			typename std::enable_if_t<
				BandIndex >= std::tuple_size< typename Structure::band_intervals >::value
			> * = nullptr
		>
		RC eWiseLambda(
			const Func f,
			alp::Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, reference > &A
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
			typename std::enable_if_t<
				band_index < std::tuple_size< typename Structure::band_intervals >::value
			> * = nullptr
		>
		RC eWiseLambda(
			const Func f,
			alp::Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, reference > &A
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
		Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, reference > &A
	) {
#ifdef _DEBUG
		std::cout << "entering alp::eWiseLambda (matrices, reference ). A is " << alp::nrows( A ) << " by " << alp::ncols( A ) << " and holds " << alp::nnz( A ) << " nonzeroes.\n";
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
		Matrix< DataType1, DataStructure1, Density::Dense, DataView1, DataImfR1, DataImfC1, reference > &A,
		const Vector< DataType2, DataStructure2, Density::Dense, DataView2, DataImfR2, DataImfC2, reference > &x,
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
			typename std::enable_if_t<
				band_index >= std::tuple_size< typename IOStructure::band_intervals >::value
			> * = nullptr
		>
		RC fold_matrix_band_generic(
			alp::Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > *C,
			const alp::Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > *A,
			const alp::Scalar< InputTypeScalar, InputStructureScalar, reference > *alpha,
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
			typename std::enable_if_t<
				band_index < std::tuple_size< typename IOStructure::band_intervals >::value
			> * = nullptr
		>
		RC fold_matrix_band_generic(
			alp::Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > *C,
			const alp::Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > *A,
			const alp::Scalar< InputTypeScalar, InputStructureScalar, reference > *alpha,
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
			alp::Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > *C,
			const alp::Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > *A,
			const alp::Scalar< InputTypeScalar, InputStructureScalar, reference > *alpha,
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
	/**
	 * For all elements in a ALP Matrix \a B, fold the value \f$ \alpha \f$
	 * into each element.
	 *
	 * The original value of \f$ \alpha \f$ is used as the left-hand side input
	 * of the operator \a op. The right-hand side inputs for \a op are retrieved
	 * from the input Matrix \a B. The result of the operation is stored in \a A,
	 * thus overwriting its previous values.
	 *
	 * The value of \f$ B_i,j \f$ after a call to thus function thus equals
	 * \f$ \alpha \odot B_i,j \f$, for all \f$ i, j \in \{ 0, 1, \dots, n - 1 \} \f$.
	 *
	 * @tparam descr         The descriptor used for evaluating this function.
	 *                       By default, this is alp::descriptors::no_operation.
	 * @tparam OP            The type of the operator to be applied.
	 * @tparam InputType     The type of \a alpha.
	 * @tparam IOType        The type of the elements in \a B.
	 * @tparam IOStructure   The structure of the matrix \a B.
	 * @tparam IOView        The view applied to the matrix \a B.
	 *
	 * @param[in]     alpha The input value to apply as the left-hand side input
	 *                      to \a op.
	 * @param[in,out] B     On function entry: the initial values to be applied as
	 *                      the right-hand side input to \a op.
	 *                      On function exit: the output data.
	 * @param[in]     op    The monoid under which to perform this left-folding.
	 *
	 * @returns alp::SUCCESS This function always succeeds.
	 *
	 * \note We only define fold under monoids, not under plain operators.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * alp::descriptors::no_operation, alp::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If alp::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a IOType, 2) the second domain of \a op must match
	 * \a InputType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# (no requirements).
	 * \endparblock
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	//  *         the size of the vector \a x. The constant factor depends on the
	//  *         cost of evaluating the underlying binary operator. A good
	//  *         implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the operator used allow for this.
	//  *
	//  *      -# This call will not result in additional dynamic memory allocations.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ 2n \cdot \mathit{sizeof}(\mathit{IOType}) + \mathcal{O}(1) \f$
	//  *         bytes of data movement.
	//  * \endparblock
	 *
	 * @see alp::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Monoid
	>
	RC foldr(
		const Scalar< InputType, InputStructure, reference > &alpha,
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > &B,
		const Monoid &monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value && ! alp::is_object< IOType >::value && alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, InputType >::value ),
			"alp::foldr",
			"called with a scalar alpha of a type that does not match the first domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, IOType >::value ),
			"alp::foldr",
			"called on a matrix B of a type that does not match the second domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, IOType >::value ),
			"alp::foldr",
			"called on a matrix B of a type that does not match the third domain "
			"of the given operator"
		);
		// TODO: check that InputStructure's bands are a subset of IOStructure's bands

		// fold to the right, with scalar as input
		constexpr bool left = false;
		constexpr bool scalar = true;
		constexpr Matrix< InputType, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > *no_matrix = nullptr;
		return internal::fold_matrix_generic< left, scalar, descr >( &B, no_matrix, &alpha, monoid.getOperator() ) ;
	}

	/** Folds element-wise alpha into B, operator variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Operator
	>
	RC foldr(
		const Scalar< InputType, InputStructure, reference > &alpha,
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > &B,
		const Operator &op = Operator(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value && ! alp::is_object< IOType >::value && alp::is_operator< Operator >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D1, InputType >::value ),
			"alp::foldr",
			"called with a scalar alpha B of a type that does not match the first domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D2, IOType >::value ),
			"alp::foldr",
			"called on a matrix B of a type that does not match the second domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D3, IOType >::value ),
			"alp::foldr",
			"called on a matrix B of a type that does not match the third domain "
			"of the given operator"
		);
		// TODO: check that InputStructure's bands are identical to IOStructure's bands

		// fold to the right, with scalar as input
		constexpr bool left = false;
		constexpr bool scalar = true;
		constexpr Matrix< InputType, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > *no_matrix = nullptr;
		return internal::fold_matrix_generic< left, scalar, descr >( &B, no_matrix, &alpha, op ) ;
	}

	/** Folds element-wise A into B, monoid variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Monoid
	>
	RC foldr(
		const Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &A,
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > &B,
		const Monoid &monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value && ! alp::is_object< IOType >::value && alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, InputType >::value ),
			"alp::foldr",
			"called with a matrix A of a type that does not match the first domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, IOType >::value ),
			"alp::foldr",
			"called on a matrix B of a type that does not match the second domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, IOType >::value ),
			"alp::foldr",
			"called on a matrix B of a type that does not match the third domain "
			"of the given operator"
		);
		// TODO: check that InputStructure's bands are a subset of IOStructure's bands

		// fold to the right, with matrix as input (no scalar)
		constexpr bool left = false;
		constexpr bool scalar = false;
		constexpr Scalar< InputType, structures::General, reference > *no_scalar = nullptr;
		return internal::fold_matrix_generic< left, scalar, descr >( &B, &A, no_scalar, monoid.getOperator() ) ;
	}

	/** Folds element-wise A into B, operator variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Operator
	>
	RC foldr(
		const Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &A,
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > &B,
		const Operator &op = Operator(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value && ! alp::is_object< IOType >::value && alp::is_operator< Operator >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D1, InputType >::value ),
			"alp::foldr",
			"called with a matrix A of a type that does not match the first domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D2, IOType >::value ),
			"alp::foldr",
			"called on a matrix B of a type that does not match the second domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Operator::D3, IOType >::value ),
			"alp::foldr",
			"called on a matrix B of a type that does not match the third domain "
			"of the given operator"
		);
		// TODO: check that InputStructure's bands are identical to IOStructure's bands

		// fold to the right, with matrix as input (no scalar)
		constexpr bool left = false;
		constexpr bool scalar = false;
		constexpr Scalar< InputType, structures::General, reference > *no_scalar = nullptr;
		return internal::fold_matrix_generic< left, scalar, descr >( &B, &A, no_scalar, op ) ;
	}

	/** Folds element-wise B into A, monoid variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Monoid
	>
	RC foldl(
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > &A,
		const Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &B,
		const Monoid &monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< IOType >::value && ! alp::is_object< InputType >::value && alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, IOType >::value ),
			"alp::foldl",
			"called on a matrix A of a type that does not match the first domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, InputType >::value ),
			"alp::foldl",
			"called with a matrix B of a type that does not match the second domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, IOType >::value ),
			"alp::foldl",
			"called on a matrix A of a type that does not match the third domain "
			"of the given operator"
		);
		// TODO: check that InputStructure's bands are a subset of IOStructure's bands

		constexpr bool left = true;
		constexpr bool scalar = false;
		constexpr Scalar< InputType, structures::General, reference > *no_scalar = nullptr;
		return internal::fold_matrix_generic< left, scalar, descr >( &A, &B, no_scalar, monoid.getOperator() ) ;
	}

	/** Folds element-wise B into A, operator variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Operator
	>
	RC foldl(
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > &A,
		const Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &B,
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
		constexpr Scalar< InputType, structures::General, reference > *no_scalar = nullptr;
		return internal::fold_matrix_generic< left, scalar, descr >( &A, &B, no_scalar, op ) ;
	}

	/** Folds element-wise beta into A, monoid variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Monoid
	>
	RC foldl(
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > &A,
		const Scalar< InputType, InputStructure, reference > &beta,
		const Monoid &monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< IOType >::value && !alp::is_object< InputType >::value && alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, IOType >::value ),
			"alp::foldl",
			"called on a matrix A of a type that does not match the first domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, InputType >::value ),
			"alp::foldl",
			"called with a scalar beta of a type that does not match the second domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, IOType >::value ),
			"alp::foldl",
			"called on a matrix A of a type that does not match the third domain "
			"of the given operator"
		);
		// TODO: check that InputStructure's bands are a subset of IOStructure's bands

		constexpr bool left = true;
		constexpr bool scalar = true;
		constexpr Matrix< InputType, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > *no_matrix = nullptr;
		return internal::fold_matrix_generic< left, scalar, descr >( &A, no_matrix, &beta, monoid.getOperator() ) ;
	}

	/** Folds element-wise beta into A, operator variant */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Operator
	>
	RC foldl(
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > &A,
		const Scalar< InputType, InputStructure, reference > &beta,
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
		constexpr Matrix< InputType, structures::General, Density::Dense, view::Original< void >, imf::Id, imf::Id, reference > *no_matrix = nullptr;
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
		reference
	>
	conjugate(
		const Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, reference > &A,
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
			reference
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
		reference
	>
	conjugate(
		const Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, reference > &A,
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
			reference
			>(
				init_lambda,
				nrows( A ),
				data_lambda
			);

	}
	/** @} */

} // end namespace ``alp''

#undef NO_CAST_OP_ASSERT

#endif // end ``_H_ALP_REFERENCE_BLAS2''

