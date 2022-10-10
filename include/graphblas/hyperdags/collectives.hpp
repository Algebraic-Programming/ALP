
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
 * @author A. N. Yzelman & J. M. Nash
 * @date 12th of April, 2017
 */

#ifndef _H_GRB_HYPERDAGS_COLL
#define _H_GRB_HYPERDAGS_COLL

#include <type_traits>

#include <graphblas/base/collectives.hpp>

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
		"* Possible fix 2 | Provide a value of the same type as the first "    \
		"domain of the given operator.\n"                                      \
		"* Possible fix 3 | Ensure the operator given to this call to " y " h" \
		"as all of its domains equal to each other.\n"                         \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );


namespace grb {

	template<>
	class collectives< hyperdags > {

		private:

			/** Disallow instantiation of this class. */
			collectives() {}

		public:

			/**
			 * Implementation details: the reference implementation has a single user
			 * process, so this call is a no-op.
			 */
			template<
				Descriptor descr = descriptors::no_operation,
				class Operator, typename IOType
			>
			static RC allreduce(
				IOType &inout, const Operator op = Operator()
			) {
			return grb::collectives<grb::_GRB_WITH_HYPERDAGS_USING>::allreduce(
				inout, op
			);
		}

			/**
			 * Implementation details: the reference implementation has a single user
			 * process, so this call is a no-op.
			 */
			template<
				Descriptor descr = descriptors::no_operation,
				class Operator, typename IOType
			>
			static RC reduce(
				IOType &inout, const size_t root = 0, const Operator op = Operator()
			) {
				// static checks
				return grb::collectives< grb::_GRB_WITH_HYPERDAGS_USING >::reduce(
					inout, root, op
				);
			}

			/**
			 * Implementation details: the reference implementation has a single user
			 * process, so this call is a no-op.
			 */
			template< typename IOType >
			static RC broadcast( IOType &inout, const size_t root = 0 ) {
				return grb::collectives<grb::_GRB_WITH_HYPERDAGS_USING>::broadcast(
					inout, root
				);
			}

			/** Implementation details: in a single user processes, this is a no-op. */
			template< Descriptor descr = descriptors::no_operation, typename IOType >
			static RC broadcast(
				IOType * inout, const size_t size, const size_t root = 0
			) {
				return grb::collectives<grb::_GRB_WITH_HYPERDAGS_USING>::broadcast(
					inout, size, root
				);
			}

	}; // end class `collectives< hyperdags >'

} // namespace grb

#endif // end ``_H_GRB_HYPERDAGS_COLL''

