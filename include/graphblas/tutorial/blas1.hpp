
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
 * Provides the level-1 primitives for the tutorial backend
 *
 * @author A. N. Yzelman
 * @date 5th of December 2016
 */

#ifndef _H_GRB_TUTORIAL_BLAS1
#define _H_GRB_TUTORIAL_BLAS1

#include <graphblas/utils/suppressions.h>

#include <iostream>    //for printing to stderr
#include <type_traits> //for std::enable_if

#include <graphblas/backends.hpp>
#include <graphblas/blas0.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/internalops.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/semiring.hpp>

#include "coordinates.hpp"
#include "vector.hpp"

namespace grb {
	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, tutorial, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, tutorial, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< OP >::value, void
		>::type * const = nullptr
	) {
		std::cout << "hello" << std::endl;

		// dynamic sanity checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < size( z ) ) { return ILLEGAL; }
			if( nnz( y ) < size( y ) ) { return ILLEGAL; }
		}

		// check for trivial op
		if( n == 0 ) {
			return SUCCESS;
		}

		// check if we can dispatch
		if( getID( z ) == getID( y ) ) {

		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}

		// check for dense variant
		if( (descr & descriptors::dense) ||
			internal::getCoordinates( y ).nonzeroes() == n
		) {
			internal::getCoordinates( z ).assignAll();

		}

		// we are in the sparse variant
		internal::getCoordinates( z ).clear();
		const bool * const null_mask = nullptr;
		const Coords * const null_coors = nullptr;
		return grb::SUCCESS;
	}

} // end namespace ``grb''

#endif // end `_H_GRB_TUTORIAL_BLAS1'

