
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
 * @date 10 of August
 */

#ifndef _H_GRB_MATRIX
#define _H_GRB_MATRIX

#include "base/config.hpp"
#include "base/matrix.hpp"

// now include all specialisations contained in the backend directories:
#ifdef _GRB_WITH_REFERENCE

#endif
#ifdef _GRB_WITH_DENSEREF

#endif
#ifdef _GRB_WITH_LPF

#endif
#ifdef _GRB_WITH_BANSHEE

#endif

// specify default only if requested during compilation
#ifdef _GRB_BACKEND
namespace grb {

	/** Returns a constant reference to an Identity matrix of the provided size
	 * */
	template< typename T >
	const StructuredMatrix< T, structures::Identity, storage::Dense, view::Identity< void >, enum Backend backend = config::default_backend > &
	I( size_t n ) {
		return StructuredMatrix< T, structures::Identity, storage::Dense, view::Identity< void >, backend >( n );
	}

	template< typename T >
	const StructuredMatrix< T, structures::Zero, storage::Dense, view::Identity< void >, enum Backend backend = config::default_backend > &
	Zero() {
		return StructuredMatrix< T, structures::Zero, storage::Dense, view::Identity< void >, backend >();
	}

} // namespace grb
#endif

#endif // end ``_H_GRB_MATRIX''
