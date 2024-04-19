
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
 * @author A. N. Yzelman
 * @date 10th of August, 2016
 */

#ifndef _H_GRB_VECTOR
#define _H_GRB_VECTOR

#include "base/config.hpp"
#include "base/vector.hpp"
#include "coordinates.hpp"

// now include all specialisations contained in the backend directories:
#ifdef _GRB_WITH_REFERENCE
 #include <graphblas/reference/vector.hpp>
#endif
#ifdef _GRB_WITH_DENSEREF
 #include <graphblas/denseref/vector.hpp>
#endif
#ifdef _GRB_WITH_HYPERDAGS
 #include <graphblas/hyperdags/vector.hpp>
#endif
#ifdef _GRB_WITH_NONBLOCKING
 #include "graphblas/nonblocking/vector.hpp"
#endif
#ifdef _GRB_WITH_LPF
 #include <graphblas/bsp1d/vector.hpp>
#endif
#ifdef _GRB_WITH_BANSHEE
 #include <graphblas/banshee/vector.hpp>
#endif

// specify default only if requested during compilation
#ifdef _GRB_BACKEND
namespace grb {

	template<
		typename D,
		Backend implementation = config::default_backend,
		typename C = internal::Coordinates<
			config::IMPLEMENTATION< config::default_backend >::coordinatesBackend()
		>
	>
	class Vector;

	/*
	 * The default value of \a StorageSchemeType could also be made conditional (Dense or Sparse) depending on \a config::default_backend
	 */
	template< typename T, typename Structure = structures::General, typename StorageSchemeType = storage::Dense, typename View = view::Original< void >, enum Backend backend = config::default_backend >
	class VectorView;

}
#endif

#endif // end ``_H_GRB_VECTOR''

