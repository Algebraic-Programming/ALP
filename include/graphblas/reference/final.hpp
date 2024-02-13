
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
 * Provides the final implementations for the reference_omp backend.
 *
 * (For the reference backend the default final implementations suffice.)
 *
 * @author A. N. Yzelman
 * @date 13th of February, 2024
 */

#ifndef _H_GRB_REFERENCE_FINAL
#define _H_GRB_REFERENCE_FINAL

#include <string>

#include <graphblas/base/final.hpp>

#include <graphblas/omp/config.hpp>


namespace grb::internal {

	template<>
	class maybeParallel< reference_omp > {
		
		public:

			static void memcpy(
				void * const out,
				const void * const in,
				const size_t size
			) {
				if( size < config::OMP::minLoopSize() ) {
					(void) std::memcpy( out, in, size );
				} else {
					char * const __restrict__ out_c = static_cast< char * >(out);
					const char * const __restrict__ in_c = static_cast< const char * >(in);
					#pragma omp parallel
					{
						size_t start, end;
						config::OMP::localRange( start, end, 0, size );
						assert( end >= start );
						const size_t my_size = end - start;
						if( my_size > 0 ) {
							(void) std::memcpy( out_c + start, in_c + start, my_size );
						}
					}
				}
			}

			template< grb::Descriptor descr, typename IOType, typename OP >
			static void foldMatrixToVector(
				IOType * const __restrict__ out,
				const IOType * const __restrict__ matrix,
				const size_t cols, const size_t rows,
				const size_t skip,
				const OP &op
			) {
				(void) op;
				if( rows < config::OMP::minLoopSize() ) {
					maybeParallel< reference >::foldMatrixToVector( out, matrix, cols, rows,
						skip, op );
					return;
				}
				#pragma omp parallel
				{
					size_t start, end;
					config::OMP::localRange( start, end, 0, rows );
					assert( end >= start );
					const size_t my_size = end - start;
					if( my_size > 0 ) {
						if( skip >= cols ) {
							for( size_t j = 0; j < cols; ++j ) {
								OP::eWiseFoldlAA( out + start, matrix + j * rows + start, my_size );
							}
						} else {
							for( size_t j = 0; j < skip; ++j ) {
								OP::eWiseFoldlAA( out + start, matrix + j * rows + start, my_size );
							}
							for( size_t j = skip + 1; j < cols; ++j ) {
								OP::eWiseFoldlAA( out + start, matrix + j * rows + start, my_size );
							}
						}
					}
				}
			}

	};

}

#endif // end _H_GRB_REFERENCE_FINAL

