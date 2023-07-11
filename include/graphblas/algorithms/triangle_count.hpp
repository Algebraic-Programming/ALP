
/*
 *   Copyright 2023 Huawei Technologies Co., Ltd.
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
 * @file
 *
 * Implements the triangle counting algorithm, using different methods.
 *
 * @author Benjamin Lozes
 * @date: May 10th, 2023
 */

#ifndef _H_GRB_TRIANGLE_COUNT
#define _H_GRB_TRIANGLE_COUNT

#include <map>
#include <numeric>
#include <vector>

#include <graphblas/utils/iterators/NonzeroIterator.hpp>

#include <graphblas.hpp>

namespace grb {

	namespace algorithms {

		enum class TriangleCountAlgorithm { Burkhardt, Cohen, Sandia_TT };

		std::map< TriangleCountAlgorithm, std::string > TriangleCountAlgorithmNames = {
			{ TriangleCountAlgorithm::Burkhardt, "Burkhardt" },
			{ TriangleCountAlgorithm::Cohen, "Cohen" },
			{ TriangleCountAlgorithm::Sandia_TT, "Sandia_TT" }
		};

		template<
			class Semiring, class MulMonoid, class SumMonoid,
			Descriptor descr_mxm = descriptors::no_operation,
			Descriptor descr_ewa = descriptors::no_operation,
			Descriptor descr_reduce = descriptors::no_operation,
			typename D1, typename RIT1, typename CIT1, typename NIT1,
			typename D2, typename RIT2, typename CIT2, typename NIT2,
			typename D3, typename RIT3, typename CIT3, typename NIT3,
			typename D4, typename RIT4, typename CIT4, typename NIT4,
			typename D5, typename RIT5, typename CIT5, typename NIT5,
			typename D6
		>
		RC triangle_count_generic(
			size_t & count,
			Matrix< D1, config::default_backend, RIT1, CIT1, NIT1 > & MXM_out,
			const Matrix< D2, config::default_backend, RIT2, CIT2, NIT2 > & MXM_lhs,
			const Matrix< D3, config::default_backend, RIT3, CIT3, NIT3 > & MXM_rhs,
			Matrix< D4, config::default_backend, RIT4, CIT4, NIT4 > & EWA_out,
			const Matrix< D5, config::default_backend, RIT5, CIT5, NIT5 > & EWA_rhs,
			const D6 div_factor,
			const Semiring mxm_semiring = Semiring(),
			const MulMonoid ewiseapply_monoid = MulMonoid(),
			const SumMonoid sumreduce_monoid = SumMonoid()
		) {
			if( ( &MXM_out == &MXM_lhs ) || ( &MXM_out == &MXM_rhs ) ) {
				return ILLEGAL;
			}

			RC rc = SUCCESS;

			// Compute MXM_out = Mlhs * Mrhs
			rc = rc ? rc : mxm< descr_mxm >( MXM_out, MXM_lhs, MXM_rhs, mxm_semiring, Phase::RESIZE );
			rc = rc ? rc : mxm< descr_mxm >( MXM_out, MXM_lhs, MXM_rhs, mxm_semiring, Phase::EXECUTE );

			// Compute MXM_out .*= EWA_rhs
			// FIXME: Replace by a foldl( Matrix[in,out], Matrix[in], Monoid ) - not implemented yet
			// Will then become:
			// rc = rc ? rc : eWiseApply< descr_ewa >( MXM_out, MXM_out, EWA_rhs, ewiseapply_monoid, Phase::RESIZE );
			// rc = rc ? rc : eWiseApply< descr_ewa >( MXM_out, MXM_out, EWA_rhs, ewiseapply_monoid, Phase::EXECUTE );
			// Instead of:
			rc = rc ? rc : eWiseApply< descr_ewa >( EWA_out, MXM_out, EWA_rhs, ewiseapply_monoid, Phase::RESIZE );
			rc = rc ? rc : eWiseApply< descr_ewa >( EWA_out, MXM_out, EWA_rhs, ewiseapply_monoid, Phase::EXECUTE );

			// Compute a sum reduction over <EWA_out> into <count>
			count = static_cast< size_t >( 0 );
			rc = rc ? rc : foldl< descr_reduce >( count, EWA_out, sumreduce_monoid );

			// Apply the div_factor to the reduction result
			count /= div_factor;

			return rc;
		}

		/**
		 * Given a graph, indicates how many triangles are contained within.
		 *
		 * @tparam D 				The type of the matrix non-zero values.
		 *
		 * @param[out]    count     The number of triangles.
		 * 						    Any prior contents will be ignored.
		 * @param[in]     A         The input graph.
		 * @param[in,out] MXM_out    Buffer matrix with the same dimensions as the input
		 * 							graph. Any prior contents will be ignored.
		 * @param[in] L 		Lower triangular matrix of the input graph (optional)
		 * @param[in] U 		Lower triangular matrix of the input graph (optional)
		 *
		 *
		 * @returns #grb::SUCCESS  When the computation completes successfully.
		 * @returns #grb::MISMATCH If the dimensions of the input matrices/buffers
		 * 						   are incompatible.
		 * @returns #grb::ILLEGAL  If the given algorithm does not exist.
		 * @returns #grb::PANIC    If an unrecoverable error has been encountered. The
		 *                         output as well as the state of ALP/GraphBLAS is
		 *                         undefined.
		 *
		 * \par Performance semantics
		 *
		 *   -# This function does not allocate nor free dynamic memory, nor shall it
		 *      make any system calls.
		 *
		 * For performance semantics regarding work, inter-process data movement,
		 * intra-process data movement, synchronisations, and memory use, please see
		 * the specification of the ALP primitives this function relies on. These
		 * performance semantics, with the exception of getters such as #grb::nnz, are
		 * specific to the backend selected during compilation.
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename D1, typename RIT1, typename CIT1, typename NIT1,
			typename D2, typename RIT2, typename CIT2, typename NIT2,
			typename D3, typename RIT3, typename CIT3, typename NIT3,
			typename D4, typename RIT4, typename CIT4, typename NIT4,
			class Semiring = Semiring< operators::add< D1 >,
									   operators::mul< D1 >,
									   identities::zero,
									   identities::one >,
			class MulMonoid = Monoid< operators::mul< D1 >, 
									  identities::one >,
			class SumMonoid = Monoid< operators::add< size_t, D1, size_t >, 
									  identities::zero > 
		>
		RC triangle_count(
			const TriangleCountAlgorithm algo,
			size_t & count,
			const Matrix< D1, config::default_backend, RIT1, CIT1, NIT1 > & A,
			Matrix< D2, config::default_backend, RIT2, CIT2, NIT2 > & MXM_out,
			Matrix< D3, config::default_backend, RIT3, CIT3, NIT3 > & EWA_out,
			Matrix< D4, config::default_backend, RIT4, CIT4, NIT4 > & L = { 0, 0 },
			Matrix< D4, config::default_backend, RIT4, CIT4, NIT4 > & U = { 0, 0 }
		) {
			// Static assertions
			static_assert( std::is_integral< D1 >::value, "Type D1 must be integral" );

			// Sanity checks
			if( nrows( A ) != ncols( A ) ) {
				std::cerr << "Matrix A must be square" << std::endl;
				return MISMATCH;
			}
			if( ncols( L ) != nrows( L ) ) {
				std::cerr << "Matrix L must be square" << std::endl;
				return MISMATCH;
			}
			if( nrows( A ) != ncols( L ) ) {
				std::cerr << "Matrices A and L must have the same dimensions" << std::endl;
				return MISMATCH;
			}
			if( ncols( U ) != nrows( U ) ) {
				std::cerr << "Matrix U must be square" << std::endl;
				return MISMATCH;
			}
			if( nrows( A ) != ncols( U ) ) {
				std::cerr << "Matrices A and U must have the same dimensions" << std::endl;
				return MISMATCH;
			}
			if( ncols( MXM_out ) != nrows( MXM_out ) ) {
				std::cerr << "Matrix MXM_out must be square" << std::endl;
				return MISMATCH;
			}
			if( nrows( A ) != ncols( MXM_out ) ) {
				std::cerr << "Matrices A and MXM_out must have the same dimensions" << std::endl;
				return MISMATCH;
			}
			if( ncols( EWA_out ) != nrows( EWA_out ) ) {
				std::cerr << "Matrix EWA_out must be square" << std::endl;
				return MISMATCH;
			}
			if( nrows( A ) != ncols( EWA_out ) ) {
				std::cerr << "Matrices A and EWA_out must have the same dimensions" << std::endl;
				return MISMATCH;
			}

			// Dispatch to the appropriate algorithm
			switch( algo ) {
				case TriangleCountAlgorithm::Burkhardt: {
					return triangle_count_generic<
						Semiring, MulMonoid, SumMonoid,
						descr | descriptors::transpose_right
					>( count, MXM_out, A, A, EWA_out, A, 6UL );
				}

				case TriangleCountAlgorithm::Cohen: {
					if( nrows( L ) == 0 || ncols( L ) == 0 ) {
						std::cerr << "Matrix L must be provided for the Cohen algorithm" << std::endl;
						return MISMATCH;
					}
					if( nrows( U ) == 0 || ncols( U ) == 0 ) {
						std::cerr << "Matrix U must be provided for the Cohen algorithm" << std::endl;
						return MISMATCH;
					}

					return triangle_count_generic<
						Semiring, MulMonoid, SumMonoid
					>( count, MXM_out, L, U, EWA_out,  A, 2UL );
				}

				case TriangleCountAlgorithm::Sandia_TT: {
					if( ( nrows( U ) == 0 || ncols( U ) == 0 ) && ( nrows( L ) == 0 || ncols( L ) == 0 ) ) {
						std::cerr << "Matrix L or U must be provided for the Sandia_TT algorithm" << std::endl;
						return MISMATCH;
					}

					const Matrix< D4, config::default_backend, RIT4, CIT4, NIT4 > & T = ( nrows( U ) == 0 || ncols( U ) == 0 ) ? L : U;
					return triangle_count_generic<
						Semiring, MulMonoid, SumMonoid
					>( count, MXM_out, T, T, EWA_out, T, 1UL );
				}

				default:
					std::cerr << "Unknown TriangleCountAlgorithm enum value" << std::endl;
					return ILLEGAL;
			}

			return SUCCESS;
		}

	} // namespace algorithms

} // namespace grb

#endif // _H_GRB_TRIANGLE_COUNT
