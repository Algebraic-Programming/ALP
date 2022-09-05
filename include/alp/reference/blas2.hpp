
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

namespace alp {

	/**
	 * \addtogroup reference
	 * @{
	 */

	/**
	 * Retrieve the number of nonzeroes contained in this matrix.
	 *
	 * @returns The number of nonzeroes the current matrix contains.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -# This function consitutes \f$ \Theta(1) \f$ work.
	 *        -# This function allocates no additional dynamic memory.
	 *        -# This function uses \f$ \mathcal{O}(1) \f$ memory
	 *           beyond that which was already used at function entry.
	 *        -# This function will move
	 *             \f$ \mathit{sizeof}( size\_t ) \f$
	 *           bytes of memory.
	 * \endparblock
	 */
	template< typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC >
	size_t nnz( const Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > & A ) noexcept {
		return A.nz;
	}

	/**
	 * Resizes the matrix to have at least the given number of nonzeroes.
	 * The contents of the matrix are not retained.
	 *
	 * Resizing of dense containers is not allowed as the capacity is determined
	 * by the container dimensions and the storage scheme. Therefore, this
	 * function will not change the capacity of the matrix.
	 *
	 * Even though the capacity remains unchanged, the contents of the matrix
	 * are not retained to maintain compatibility with the general specification.
	 * However, the actual memory will not be reallocated. Rather, the matrix
	 * will be marked as uninitialized.
	 *
	 * @param[in] A         The matrix to be resized.
	 * @param[in] nonzeroes The number of nonzeroes this matrix is to contain.
	 *
	 * @return SUCCESS   If \a new_nz is not larger than the current capacity
	 *                   of the matrix.
	 *         ILLEGAL   If \a new_nz is larger than the current capacity of
	 *                   the matrix.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -$ This function consitutes \f$ \Theta(1) \f$ work.
	 *        -# This function allocates \f$ \Theta(0) \f$
	 *           bytes of dynamic memory.
	 *        -# This function does not make system calls.
	 * \endparblock
	 */
	template< typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC >
	RC resize( Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &A, const size_t new_nz ) noexcept {
		(void)A;
		(void)new_nz;
		// TODO implement
		// setInitialized( A, false );
		return PANIC;
	}

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
			size_t BandIndex, typename Func,
			typename DataType, typename Structure, typename View, typename ImfR, typename ImfC,
			typename std::enable_if_t<
				BandIndex < std::tuple_size< typename Structure::band_intervals >::value
			> * = nullptr
		>
		RC eWiseLambda(
			const Func f,
			alp::Matrix< DataType, Structure, Density::Dense, View, ImfR, ImfC, reference > &A
		) {
			const size_t M = nrows( A );
			const size_t N = ncols( A );

			const std::ptrdiff_t l = structures::get_lower_bandwidth< BandIndex >( A );
			const std::ptrdiff_t u = structures::get_upper_bandwidth< BandIndex >( A );

			// In case of symmetry the iteration domain intersects the the upper
			// (or lower) domain of A
			constexpr bool is_sym_a { structures::is_a< Structure, structures::Symmetric >::value };

			// Temporary until adding multiple symmetry directions
			constexpr bool sym_up_a { is_sym_a };

			/** i-coordinate lower and upper limits considering matrix size and band limits */
			const std::ptrdiff_t i_l_lim = std::max( static_cast< std::ptrdiff_t >( 0 ), -u );
			const std::ptrdiff_t i_u_lim = std::min( M, -l + N );

			for( size_t i = static_cast< size_t >( i_l_lim ); i < static_cast< size_t >( i_u_lim ); ++i ) {
				/** j-coordinate lower and upper limits considering matrix size and symmetry */
				const std::ptrdiff_t j_sym_l_lim = is_sym_a && sym_up_a ? i : 0;
				const std::ptrdiff_t j_sym_u_lim = is_sym_a && !sym_up_a ? i + 1 : N;
				/** j-coordinate lower and upper limits, also considering the band limits in addition to the factors above */
				const std::ptrdiff_t j_l_lim = std::max( j_sym_l_lim, l );
				const std::ptrdiff_t j_u_lim = std::min( j_sym_u_lim, u );

				for( size_t j = static_cast< size_t >( j_l_lim ); j < static_cast< size_t >( j_u_lim ); ++j ) {
					auto &a_val = internal::access( A, internal::getStorageIndex( A, i, j ) );
					f( i, j, a_val );
				}
			}
			return eWiseLambda< BandIndex + 1 >( f, A );
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
		if( ! ( getLength( x ) == nrows( A ) || getLength( x ) == ncols( A ) ) ) {
			std::cerr << "Mismatching dimensions: given vector of size " << size( x )
				<< " has nothing to do with either matrix dimension (" << nrows( A ) << " nor " << ncols( A ) << ").\n";
			return MISMATCH;
		}

		return eWiseLambda( f, A, args... );
	}

	/**
	 * For all elements in a ALP Matrix \a A, fold the value \f$ \alpha \f$
	 * into each element.
	 *
	 * The original value of \f$ \alpha \f$ is used as the left-hand side input
	 * of the operator \a op. The right-hand side inputs for \a op are retrieved
	 * from the input Matrix \a A. The result of the operation is stored in \a A,
	 * thus overwriting its previous values.
	 *
	 * The value of \f$ A_i,j \f$ after a call to thus function thus equals
	 * \f$ \alpha \odot A_i,j \f$, for all \f$ i, j \in \{ 0, 1, \dots, n - 1 \} \f$.
	 *
	 * @tparam descr         The descriptor used for evaluating this function.
	 *                       By default, this is alp::descriptors::no_operation.
	 * @tparam OP            The type of the operator to be applied.
	 * @tparam InputType     The type of \a alpha.
	 * @tparam IOType        The type of the elements in \a A.
	 * @tparam IOStructure   The structure of the matrix \a A.
	 * @tparam IOView        The view applied to the matrix \a A.
	 *
	 * @param[in]     alpha The input value to apply as the left-hand side input
	 *                      to \a op.
	 * @param[in,out] A     On function entry: the initial values to be applied as
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
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Monoid
	>
	RC foldr( const Scalar< InputType, InputStructure, reference > & alpha,
		Matrix< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & A,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! alp::is_object< InputType >::value && ! alp::is_object< IOType >::value && alp::is_monoid< Monoid >::value, void >::type * const = NULL ) {
		// static sanity checks
		//NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, IOType >::value ), "alp::foldl",
		//	"called with a vector x of a type that does not match the first domain "
		//	"of the given operator" );
		//NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, InputType >::value ), "alp::foldl",
		//	"called on a vector y of a type that does not match the second domain "
		//	"of the given operator" );
		//NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, IOType >::value ), "alp::foldl",
		//	"called on a vector x of a type that does not match the third domain "
		//	"of the given operator" );
		(void)alpha;
		(void)A;
		(void)monoid;

		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/** @} */

} // end namespace ``alp''

#endif // end ``_H_ALP_REFERENCE_BLAS2''

