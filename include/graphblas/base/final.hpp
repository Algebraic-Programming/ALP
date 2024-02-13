
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
 * @file
 *
 * Describes raw functionalities that a so-called <em>final backend</em> should
 * provide.
 *
 * A final backend is one that other backends rely on for implementing core
 * computations.
 *
 * @author A. N. Yzelman
 * @date 13th of February, 2024
 */

#ifndef _H_GRB_BASE_FINAL
#define _H_GRB_BASE_FINAL

#include <string>

#include <graphblas/rc.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/backends.hpp>
#include <graphblas/descriptors.hpp>


namespace grb::internal {

	/**
	 * This class gathers raw functionalities that non-final backends cannot
	 * implement directly because it is unaware whether final computations should
	 * occur in parallel or not, while, if it should execute in parallel, it is
	 * unaware which parallelisation scheme it should employ.
	 *
	 * The base implementation defines all functions every final backend should
	 * implement and provides a sequential implementation for each such function.
	 * Therefore, only parallel final backends should override this class.
	 */
	template< grb::Backend backend >
	class maybeParallel {

		public:

			/**
			 * Provides a basic memory copy.
			 *
			 * @param[out] out The area where to write to.
			 * @param[in]  in  The memory area to copy from.
			 * @param[in] size How many bytes should be copied.
			 *
			 * The input and output memory areas are not allowed to overlap.
			 */
			static void memcpy(
				void * const __restrict__ out,
				const void * const __restrict__ in,
				const size_t size
			) {
				(void) std::memcpy( out, in, size );
			}

			/**
			 * Folds (reduces) every column of a matrix into a vector.
			 *
			 * @tparam descr  The descriptor to be taken into account.
			 * @tparam IOType The type of vector \em and matrix elements.
			 * @tparam OP     The operator used for reduction.
			 *
			 * @param[in,out] out The output vector.
			 *
			 * Pre-existing values in \a out are reduced into.
			 *
			 * @param[in] matrix The matrix which should be column-wise reduced into
			 *                   \a out.
			 * @param[in] cols   The colums of \a matrix.
			 * @param[in] rows   The rows of \a matrix.
			 * @param[in] skip   Which column of \a matrix to skip.
			 *
			 * Taking \a skip higher or equal to \a cols will mean no column is
			 * skipped.
			 *
			 * @param[in] op The operator by which to reduce.
			 */
			template< grb::Descriptor descr, typename IOType, typename OP >
			static void foldMatrixToVector(
				IOType * const __restrict__ out,
				const IOType * const __restrict__ matrix,
				const size_t cols, const size_t rows,
				const size_t skip,
				const OP &op
			) {
				(void) op;
				if( skip >= cols ) {
					for( size_t j = 0; j < cols; ++j ) {
						OP::eWiseFoldlAA( out, matrix + j * rows, rows );
					}
				} else {
					for( size_t j = 0; j < skip; ++j ) {
						OP::eWiseFoldlAA( out, matrix + j * rows, rows );
					}
					for( size_t j = skip + 1; j < cols; ++j ) {
						OP::eWiseFoldlAA( out, matrix + j * rows, rows );
					}
				}
			}
			
	};

}

#endif // end ``_H_GRB_BASE_FINAL´´

