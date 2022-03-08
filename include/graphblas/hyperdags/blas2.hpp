
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
 * @file
 *
 * Implements the BLAS-2 API for the hypergraphs backend.
 *
 * @author A. Karanasiou
 * @date 3rd of March, 2022
 */

#ifndef _H_GRB_HYPERDAGS_BLAS2
#define _H_GRB_HYPERDAGS_BLAS2

#include <graphblas/matrix.hpp>

#include <graphblas/hyperdags/init.hpp>


namespace grb {
		
	template< typename InputType >
	size_t nrows( const Matrix< InputType, hyperdags > & A ) noexcept {
		return nrows(internal::getMatrix(A));
	}
	
	template< typename InputType >
	size_t ncols( const Matrix< InputType, hyperdags > & A ) noexcept {
		return ncols(internal::getMatrix(A));
	}
	
	template< typename InputType >
	size_t nnz( const Matrix< InputType, hyperdags > & A ) noexcept {
		return nnz(internal::getMatrix(A));
	}
	
	template< typename InputType >
	RC resize( Matrix< InputType, hyperdags > & A, const size_t new_nz ) noexcept {
		return resize(internal::getMatrix(A), new_nz);
	}

} // end namespace grb

#endif

