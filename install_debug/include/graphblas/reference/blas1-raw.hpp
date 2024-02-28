
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
 * @date 11th of September, 2019
 */

#if ! defined _H_GRB_REFERENCE_BLAS1_RAW || defined _H_GRB_REFERENCE_OMP_BLAS1_RAW
#define _H_GRB_REFERENCE_BLAS1_RAW

#include "config.hpp"
#include "coordinates.hpp"
#include "vector.hpp"

#ifdef _H_GRB_REFERENCE_OMP_BLAS1_RAW
#include <omp.h>
#endif

#define NO_CAST_ASSERT( x, y, z )                                              \
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
		"* Possible fix 2 | Provide a value that matches the expected type.\n" \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );

namespace grb {
	namespace internal {

		/**
		 * Element-wise left-looking fold of a masked tall-skinny \f$ n \times K \f$
		 * matrix into a GraphBLAS vector.
		 *
		 * @tparam no_skip   If <tt>true</tt>, \a skip will be ignored.
		 * @tparam IOType    The type of the elements in the GraphBLAS vector.
		 * @tparam InputType The type of the elements in the raw matrix.
		 * @tparam MaskType  The element types.
		 * @tparam Accumulator The operator used to fold values into \a x.
		 *
		 * @param[in,out] x   The GraphBLAS vector.
		 * @param[in] to_fold The tall-skinny matrix to fold into \a x.
		 * @param[in] mask    The mask of \a to_fold.
		 * @param[in] n       The number of rows of \a to_fold.
		 * @param[in] K       The number of columns of \a to_fold.
		 * @param[in] skip    Which column of \a to_fold to skip over.
		 * @param[in] acc     The accumulator to use during the folding.
		 *
		 * The memory pointed to by \a to_fold must contain \f$ nK \f$ elements
		 * of type \a InputType. The memory pointed to by \a mask must contain
		 * \f$ nK \f$ elements of type \a MaskType.
		 *
		 * The argument \a n must equal #grb::size of \a x.
		 *
		 * The argument \a skip must be smaller or equal to \a K.
		 *
		 * @note If all columns of \a to_fold must be processed, \a skip may be
		 *       set equal to \a K.
		 *
		 * @returns #MISMATCH If \a n does not equal the size of \a x.
		 * @returns #ILLEGAL If \a K equals zero.
		 * @returns #ILLEGAL If \a skip is larger than \a K.
		 * @returns #SUCCESS When the operation completed successfully.
		 *
		 * \par Performance semantics:
		 *   -# \f$ \Theta(nK) \f$ data movement
		 *   -# \f$ \mathit{nnz}(\mathit{mask}) \f$ applications of \a acc
		 *   -# No dynamic memory allocations or other system calls
		 *   -# \f$ \Theta(K) \f$ streams
		 */
		template<
			Descriptor descr, bool no_skip,
			typename IOType, typename Coords, typename InputType, typename MaskType,
			class Accumulator
		>
		RC foldl_from_raw_matrix_to_vector( Vector< IOType, reference, Coords > &x,
			const InputType * __restrict__ const to_fold,
			const MaskType * __restrict__ const mask,
			const size_t n, const size_t K,
			const size_t skip,
			const Accumulator &acc
		) {
			NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
				std::is_same< IOType, typename Accumulator::D1 >::value ),
				"grb::foldl_from_raw_matrix_to_vector",
				"called with a grb::Vector type that does not match the given "
				"accumulator's left domain." );
			NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
				std::is_same< InputType, typename Accumulator::D2 >::value ),
				"grb::foldl_from_raw_matrix_to_vector",
				"called with a matrix type that does not match the given "
				"accumulator's right domain." );
			NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
				std::is_same< IOType, typename Accumulator::D3 >::value ),
				"grb::foldl_from_raw_matrix_to_vector",
				"called with a grb::Vector type that does not match the given "
				"accumulator's output domain." );
			NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
				std::is_same< bool, MaskType >::value ),
				"grb::foldl_from_raw_matrix_to_vector",
				"called with a non-bool mask type." );
			auto &coordinates = internal::getCoordinates( x );
			IOType * __restrict__ const raw = internal::getRaw( x );
			const size_t local_n = coordinates.size();
			if( n != local_n ) {
#ifdef _DEBUG
				std::cout << "Error in foldl_from_raw_matrix_to_vector: mismatching number of rows\n";
#endif
				return MISMATCH;
			}
			if( K == 0 ) {
#ifdef _DEBUG
				std::cout << "Error in foldl_from_raw_matrix_to_vector: 0 columns given\n";
#endif
				return ILLEGAL;
			}
			if( ! no_skip && skip > K ) {
#ifdef _DEBUG
				std::cout << "Error in foldl_from_raw_matrix_to_vector: invalid value for skip given\n";
#endif
				return ILLEGAL;
			}
			if( n == 0 ) {
				return SUCCESS;
			}

#ifdef _DEBUG
			std::cout << "Performing foldl_from_raw_matrix_to_vector.\n"
				<< "\t Initial coordinate contents: " << coordinates.nonzeroes() << " / " << coordinates.size() << ".\n"
				<< "\t The raw matrix has size " << n << " by " << K << ".\n"
				<< "\t The " << skip << "-th column of the raw matrix will be skipped.\n";
			std::cout << "\t Contents of _local at function entry:\n";
			for( size_t i = 0; i < local_n; ++i ) {
				if( coordinates.assigned( i ) ) {
					std::cout << "\t\t " << i << ", " << raw[ i ] << "\n";
				} else {
					std::cout << "\t\t " << i << ", no value here\n";
				}
			}
#endif

#ifdef _H_GRB_REFERENCE_OMP_BLAS1_RAW
			#pragma omp parallel
			{
				size_t asyncAssigns = 0;
				const size_t asyncJoinWhen = coordinates.maxAsyncAssigns();
				assert( asyncJoinWhen > 0 );
				internal::Coordinates< reference >::Update local_update = coordinates.EMPTY_UPDATE();
				size_t start, end;
				grb::config::OMP::localRange( start, end, 0, local_n );
#else
				const size_t start = 0;
				const size_t end = local_n;
#endif
				assert( start <= end );
				assert( end <= local_n );
				for( size_t i = start; i < end; ++i ) {
#ifdef _DEBUG
					std::cout << "\t Now processing row " << i << "\n";
#endif
					for( size_t k = 0; k < K; ++k ) {
						if( !no_skip && k == skip ) {
#ifdef _DEBUG
							std::cout << "\t " << k << "-th column is skipped\n";
#endif
							continue;
						}
#ifdef _DEBUG
						std::cout << "\t Now processing column " << k << "\n";
#endif
						const size_t src_i = k * local_n + i;
						if( mask[ src_i ] ) {
#ifdef _DEBUG
							std::cout << "\t Folding raw matrix nonzero at this position, which has value " << to_fold[ src_i ] << "\n";
#endif
#ifndef _H_GRB_REFERENCE_OMP_BLAS1_RAW
							if( coordinates.assign( i ) ) {
								foldl< descr >( raw[ i ], to_fold[ src_i ], acc );
							} else {
								raw[ i ] = to_fold[ src_i ];
							}
#else
							if( coordinates.asyncAssign( i, local_update ) ) {
								foldl< descr >( raw[ i ], to_fold[ src_i ], acc );
							} else {
								raw[ i ] = to_fold[ src_i ];
								(void) ++asyncAssigns;
								if( asyncAssigns == asyncJoinWhen ) {
									(void) coordinates.joinUpdate( local_update );
									asyncAssigns = 0;
								}
							}
#endif
						}
					}
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1_RAW
				while( ! coordinates.joinUpdate( local_update ) ) {}
			} // end pragma omp parallel
#endif
#ifdef _DEBUG
			std::cout << "\t Exiting foldl_from_raw_matrix_to_vector with exit code SUCCESS\n";
			std::cout << "\t\t Contents of _local at function exit:\n";
			for( size_t i = 0; i < local_n; ++i ) {
				if( coordinates.assigned( i ) ) {
					std::cout << "\t\t " << i << ", " << raw[ i ] << "\n";
				} else {
					std::cout << "\t\t " << i << ", no value here\n";
				}
			}
#endif
			return SUCCESS;
		}

		/**
		 * Folds a tall-skinny matrix into a vector in an element-wise fashion.
		 * Only defined for dense, left-looking folds.
		 *
		 * @tparam OP     The operator to use while folding.
		 * @tparam IType  The input data type (of \a to_fold).
		 * @tparam IOType The input/output data type (of \a fold_into).
		 *
		 * @param[in,out] x       The vector whose elements to fold into.
		 * @param[in]     to_fold The tall-skinny matrix whose elements to fold.
		 * @param[in]     n       The size of \a fold_into and the size of each
		 *                        column of \a to_fold.
		 * @param[in]     K       The number of columns in \a to_fold. Must be larger
		 *                        than \a 0.
		 * @param[in]     skip    Which of the \K columns in \a to_fold to skip. Must
		 *                        be smaller than or equal to \a n.
		 * @param[in]     op      The operator to use while folding.
		 *
		 * The matrix \a to_fold is assumed packed in a column-major fashion.
		 *
		 * \note If \a skip equals \a K then no columns will be skipped.
		 *
		 * @returns #SUCCESS  On successful completion of this function call.
		 */
		template< Descriptor descr,
			class OP, typename IOType, typename IType,
			typename Coords
		>
		RC foldl_from_raw_matrix_to_vector( Vector< IOType, reference, Coords > &x,
			const IType * __restrict__ const to_fold,
			const size_t n, const size_t K,
			const size_t skip,
			const OP & op
		) noexcept {
			// take at least a number of elements so that no two threads operate on the same cache line
			const constexpr size_t blocksize =
				config::SIMD_BLOCKSIZE< IOType >::value() > config::SIMD_BLOCKSIZE< IType >::value() ?
					config::SIMD_BLOCKSIZE< IOType >::value() :
					config::SIMD_BLOCKSIZE< IType >::value();

			// static checks
			static_assert( blocksize > 0,
				"Config error: zero blocksize in call to fold_from_raw_matrix_to_vector_generic!"
			);
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< IOType, typename OP::D1 >::value ), "grb::foldl_from_raw_matrix_to_vector",
				"called with a grb::Vector type that does not match the given operator's left domain." );
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< IType, typename OP::D2 >::value ), "grb::foldl_from_raw_matrix_to_vector",
				"called with a matrix type that does not match the given operator's right domain." );
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< IOType, typename OP::D3 >::value ), "grb::foldl_from_raw_matrix_to_vector",
				"called with a grb::Vector type that does not match the given accumulator's output domain." );

#ifdef _DEBUG
			std::cout << "foldl_from_raw_matrix_to_vector (unmasked) called with"
				"n = " << n << ", K = " << K << ", skip = " << skip << "\n";
#endif
			// dyanmic checks
			if( K == 0 ) {
				return ILLEGAL;
			}
			if( skip > K ) {
				return ILLEGAL;
			}

			// catch trivial case
			if( n == 0 ) {
				return SUCCESS;
			}

			// perform operation:
			IOType * __restrict__ const fold_into = internal::getRaw( x );
#ifdef _H_GRB_REFERENCE_OMP_BLAS1_RAW
			#pragma omp parallel
			{
				const size_t t = static_cast< size_t >( omp_get_num_threads() );
				const size_t s = static_cast< size_t >( omp_get_thread_num() );
				size_t range = blocksize * ( n / blocksize / t );
				const size_t start = s * range;
				if( s + 1 == t ) {
					range = n - start;
				}
				assert( start + range <= n );
				const size_t end = start + range;
#else
#ifdef _DEBUG
			const size_t s = 0;
#endif
			const size_t start = 0;
			const size_t end = n;
#endif
#ifdef _DEBUG
				std::cout << "\tfoldl_from_raw_matrix_to_vector (unmasked), "
							 "thread "
						  << s << " has range " << start << "--" << end << ". The blocksize is " << blocksize << "." << std::endl;
#endif
				assert( start < end );
				assert( end <= n );
				size_t i = start;
				while( i + blocksize - 1 < end ) {
					IOType buffer[ blocksize ];
					for( size_t j = 0; j < blocksize; ++j ) {
						buffer[ j ] = fold_into[ i + j ];
					}
					for( size_t k = 0; k < skip; ++k ) {
						IType input[ blocksize ];
						for( size_t j = 0; j < blocksize; ++j ) {
							input[ j ] = to_fold[ k * n + i + j ];
						}
						for( size_t j = 0; j < blocksize; ++j ) {
							foldl< descr >( buffer[ j ], input[ j ], op );
						}
					}
					for( size_t k = skip + 1; k < K; ++k ) {
						IType input[ blocksize ];
						for( size_t j = 0; j < blocksize; ++j ) {
							input[ j ] = to_fold[ k * n + i + j ];
						}
						for( size_t j = 0; j < blocksize; ++j ) {
							foldl< descr >( buffer[ j ], input[ j ], op );
						}
					}
					for( size_t j = 0; j < blocksize; ++j ) {
						fold_into[ i + j ] = buffer[ j ];
					}
					i += blocksize;
				}
#ifdef _DEBUG
				std::cout << "\tfoldl_from_raw_matrix_to_vector (unmasked), "
							 "thread "
						  << s << " is entering coda..." << std::endl;
#endif
				for( ; i < end; ++i ) {
					for( size_t k = 0; k < skip; ++k ) {
						foldl< descr >( fold_into[ i ], to_fold[ k * n + i ], op );
					}
					for( size_t k = skip + 1; k < K; ++k ) {
						foldl< descr >( fold_into[ i ], to_fold[ k * n + i ], op );
					}
				}
#ifdef _DEBUG
				std::cout << "\tfoldl_from_raw_matrix_to_vector (unmasked), "
							 "thread "
						  << s << " is done!" << std::endl;
#endif
#ifdef _H_GRB_REFERENCE_OMP_BLAS1_RAW
			}
#endif
			return SUCCESS;
		}

	} // namespace internal
} // namespace grb

// parse this unit again for OpenMP support
#ifdef _GRB_WITH_OMP
#ifndef _H_GRB_REFERENCE_OMP_BLAS1_RAW
#define _H_GRB_REFERENCE_OMP_BLAS1_RAW
#define reference reference_omp
#include "graphblas/reference/blas1-raw.hpp"
#undef reference
#undef _H_GRB_REFERENCE_OMP_BLAS1_RAW
#endif
#endif

#undef NO_CAST_ASSERT

#endif // end `_H_GRB_REFERENCE_BLAS1_RAW'
