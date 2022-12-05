
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

#ifndef _H_ALP_OMP_IO
#define _H_ALP_OMP_IO

#include <alp/base/io.hpp>

#include "matrix.hpp"
#include "scalar.hpp"
#include "storage.hpp"

// Include backend to which sequential work is delegated
#ifdef _ALP_OMP_WITH_REFERENCE
 #include <alp/reference/io.hpp>
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

namespace alp {

	/**
	 * Sets all elements of the given matrix to the value of the given scalar.
	 * C = val
	 *
	 * @tparam descr
	 * @tparam OutputType      Data type of the output matrix C
	 * @tparam OutputStructure Structure of the matrix C
	 * @tparam OutputView      View type applied to the matrix C
	 * @tparam InputType       Data type of the scalar a
	 *
	 * @param C    Matrix whose values are to be set
	 * @param val  The value to set the elements of the matrix C
	 *
	 * @return RC  SUCCESS on the successful execution of the set
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType, typename InputStructure
	>
	RC set(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, omp > &C,
		const Scalar< InputType, InputStructure, omp > &val
	) noexcept {

		static_assert(
			!std::is_same< OutputType, void >::value,
			"alp::set (set to matrix): cannot have a pattern matrix as output"
		);
#ifdef _DEBUG
		std::cout << "Called alp::set (matrix-to-value, omp)" << std::endl;
#endif
		// static checks
		NO_CAST_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< InputType, OutputType >::value ),
			"alp::set", "called with non-matching value types"
		);

		static_assert(
			!internal::is_functor_based<
				Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, omp >
			>::value,
			"alp::set cannot be called with a functor-based matrix as a destination."
		);

		if( !internal::getInitialized( val ) ) {
			internal::setInitialized( C, false );
			return SUCCESS;
		}

		const Distribution_2_5D &d = internal::getAmf( C ).getDistribution();

		RC rc = SUCCESS;

		#pragma omp parallel for
		for( size_t thread = 0; thread < config::OMP::current_threads(); ++thread ) {
			const auto t_coords = d.getThreadCoords( thread );
			const auto block_grid_dims = d.getLocalBlockGridDims( t_coords );

			RC local_rc = SUCCESS;

			for( size_t br = 0; br < block_grid_dims.first; ++br ) {
				for( size_t bc = 0; bc < block_grid_dims.second; ++bc ) {

					// Get a sequential matrix view over the block
					auto refC = internal::get_view( C, t_coords, br, bc );

					// Construct a sequential Scalar container from the input Scalar
					Scalar< InputType, InputStructure, config::default_sequential_backend > ref_val( *val );

					// Delegate the call to the sequential set implementation
					local_rc = local_rc ? local_rc : set( refC, ref_val );

					if( local_rc != SUCCESS ) {
						rc = local_rc;
					}
				}
			}
		}

		internal::setInitialized( C, true );
		return rc;
	}

} // end namespace ``alp''

#undef NO_CAST_ASSERT

#endif // end ``_H_ALP_OMP_IO''

