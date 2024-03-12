
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
 * Collectives implementation for the Ascend backend.
 *
 * @author A. N. Yzelman
 * @date 12th of September, 2023
 */

#ifndef _H_GRB_ASCEND_COLL
#define _H_GRB_ASCEND_COLL

#include <type_traits>

#include <graphblas/backends.hpp>
#include <graphblas/base/collectives.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/rc.hpp>


namespace grb {

	/** The collectives class is based on that of the reference backend */
	template<>
	class collectives< ascend > {

		private:

			/** Disallow instantiation of this class. */
			collectives() {}


		public:

			template<
				Descriptor descr = descriptors::no_operation,
				class Operator, typename IOType
			>
			static RC allreduce( IOType &inout, const Operator op = Operator() ) {
				return collectives< reference >::allreduce< descr, Operator, IOType >(
					inout, op );
			}

			template<
				Descriptor descr = descriptors::no_operation,
				class Operator, typename IOType
			>
			static RC reduce(
				IOType &inout, const size_t root = 0, const Operator op = Operator()
			) {
				return collectives< reference >::reduce< descr, Operator, IOType >( inout,
					root, op );
			}

			template< typename IOType >
			static RC broadcast( IOType &inout, const size_t root = 0 ) {
				return collectives< reference >::broadcast< IOType >( inout, root );
			}

			template< Descriptor descr = descriptors::no_operation, typename IOType >
			static RC broadcast(
				IOType * inout, const size_t size,
				const size_t root = 0
			) {
				return collectives< reference >::broadcast< descr, IOType >( inout, size,
					root );
			}

	}; // end class `collectives< ascend >'

} // namespace grb

#endif // end ``_H_GRB_ASCEND_COLL''

