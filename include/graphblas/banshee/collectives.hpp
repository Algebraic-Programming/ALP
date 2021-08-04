
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

#if ! defined _H_GRB_BANSHEE_COLL || defined _H_GRB_BANSHEE_OMP_COLL
#define _H_GRB_BANSHEE_COLL

#include <type_traits>

#include "graphblas/backends.hpp"
#include "graphblas/descriptors.hpp"
#include "graphblas/forward.hpp"
#include "graphblas/rc.hpp"

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
		"as"                                                                   \
		" a"                                                                   \
		"ll"                                                                   \
		" o"                                                                   \
		"f "                                                                   \
		"it"                                                                   \
		"s "                                                                   \
		"do"                                                                   \
		"ma"                                                                   \
		"in"                                                                   \
		"s "                                                                   \
		"eq"                                                                   \
		"ua"                                                                   \
		"l "                                                                   \
		"to"                                                                   \
		" e"                                                                   \
		"ac"                                                                   \
		"h "                                                                   \
		"ot"                                                                   \
		"he"                                                                   \
		"r."                                                                   \
		"\n"                                                                   \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );

namespace grb {

	/**
	 * Implementation notes: specialisation of the collectives template for the
	 * banshee implementation. Since the banshee implementation is sequential,
	 * all of the calls to a collective function are essentially no-ops.
	 */
	template<>
	class collectives< banshee > {

	private:
		/** Disallow instantiation of this class. */
		collectives() {}

	public:
		/**
		 * Implementation details: the banshee implementation has a single user
		 * process, so this call is a no-op.
		 */
		template< Descriptor descr = descriptors::no_operation, class Operator, typename IOType >
		static RC allreduce( IOType &, const Operator = Operator() ) {
			// static checks
			NO_CAST_ASSERT( ! ( descr & no_casting ) ||
					( std::is_same< IOType, typename Operator::D1 >::value && std::is_same< IOType, typename Operator::D2 >::value && std::is_same< IOType, typename Operator::D3 >::value ),
				"collectives::allreduce", "operator types do not match input type." );
			// done
			return grb::SUCCESS;
		}

		/**
		 * Implementation details: the banshee implementation has a single user
		 * process, so this call is a no-op.
		 */
		template< Descriptor descr = descriptors::no_operation, class Operator, typename IOType >
		static RC reduce( IOType &, const size_t root = 0, const Operator = Operator() ) {
			// static checks
			NO_CAST_ASSERT( ! ( descr & descriptors::no_casting ) ||
					( std::is_same< IOType, typename Operator::D1 >::value && std::is_same< IOType, typename Operator::D2 >::value && std::is_same< IOType, typename Operator::D3 >::value ),
				"collectives::reduce", "operator types do not match input type." );
			if( root > 0 ) {
				return grb::ILLEGAL;
			}
			return grb::SUCCESS;
		}

		/**
		 * Implementation details: the banshee implementation has a single user
		 * process, so this call is a no-op.
		 */
		template< typename IOType >
		static RC broadcast( IOType &, const size_t root = 0 ) {
			if( root > 0 ) {
				return grb::ILLEGAL;
			}
			return grb::SUCCESS;
		}

		/** Implementation details: in a single user processes, this is a no-op. */
		template< Descriptor descr = descriptors::no_operation, typename IOType >
		static RC broadcast( IOType * inout, const size_t size, const size_t root = 0 ) {
			(void)inout;
			(void)size;
			if( root > 0 ) {
				return grb::ILLEGAL;
			}
			return grb::SUCCESS;
		}

	}; // end class `collectives< banshee >'
} // namespace grb

#undef NO_CAST_ASSERT

#endif // end ``_H_GRB_BANSHEE_COLL''
