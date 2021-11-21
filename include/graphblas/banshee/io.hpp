
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
 */

#if ! defined _H_GRB_BANSHEE_IO
#define _H_GRB_BANSHEE_IO

#include <graphblas/io.hpp>
#include <graphblas/utils/SynchronizedNonzeroIterator.hpp>

#include "coordinates.hpp"
#include "vector.hpp"

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
		"* Possible fix 2 | Provide a value input iterator with element "      \
		"types that match the output vector element type.\n"                   \
		"* Possible fix 3 | If applicable, provide an index input iterator "   \
		"with element types that are integral.\n"                              \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );

namespace grb {

	/**
	 * \defgroup IO Data Ingestion
	 * @{
	 */

	/** \internal No implementation notes: follows the reference implementation. */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename Coords,
		typename fwd_iterator, class Dup = operators::right_assign< InputType >
	>
	RC buildVector(
		Vector< InputType, banshee, Coords > &x,
		fwd_iterator start, const fwd_iterator end,
		const IOMode mode, const Dup &dup
	) {
		// static sanity check
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) ||
			std::is_same< InputType, decltype( *std::declval< fwd_iterator >() ) >::value ),
			"grb::buildVector (banshee implementation)",
			"Input iterator does not match output vector type while no_casting "
			"descriptor was set" );

		// in the sequential banshee implementation, the number of user processes always equals 1
		// therefore the sequential and parallel modes are equivalent
#ifndef NDEBUG
		assert( mode == SEQUENTIAL || mode == PARALLEL );
#else
		(void)mode;
#endif

		// declare temporary to meet delegate signature
		const fwd_iterator start_pos = start;

		// do delegate
		return x.template build< descr >( dup, start_pos, end, start );
	}

	/** \internal No implementation notes: follows the reference implementation. */
	template< Descriptor descr = descriptors::no_operation, typename InputType, typename Coords, typename fwd_iterator, class Dup = operators::right_assign< InputType > >
	RC buildVector( Vector< InputType, banshee, Coords > & x, fwd_iterator start, const fwd_iterator end, const IOMode mode ) {
		return buildVector( x, start, end, mode, Dup() );
	}

	/** \internal No implementation notes: follows the reference implementation. */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename Coords,
		typename fwd_iterator1, typename fwd_iterator2,
		class Dup = operators::right_assign< InputType >
	>
	RC buildVector( Vector< InputType, banshee, Coords > & x,
		fwd_iterator1 ind_start,
		const fwd_iterator1 ind_end,
		fwd_iterator2 val_start,
		const fwd_iterator2 val_end,
		const IOMode mode,
		const Dup & dup
	) {
		// static sanity check
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) ||
			std::is_same< InputType, decltype( *std::declval< fwd_iterator2 >() ) >::value ||
			std::is_integral< decltype( *std::declval< fwd_iterator1 >() ) >::value ),
			"grb::buildVector (banshee implementation)",
			"At least one input iterator has incompatible value types while "
			"no_casting descriptor was set" );

		// in the sequential banshee implementation, the number of user processes always equals 1
		// therefore the sequential and parallel modes are equivalent
#ifndef NDEBUG
		assert( mode == SEQUENTIAL || mode == PARALLEL );
#else
		(void)mode;
#endif

		// call the private member function that provides this functionality
		return x.template build< descr >( dup, ind_start, ind_end, val_start, val_end );
	}

	/** \internal No implementation notes: follows the reference implementation. */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename Coords, typename fwd_iterator1, typename fwd_iterator2,
		class Dup = operators::right_assign< InputType >
	>
	RC buildVector( Vector< InputType, banshee, Coords > &x,
		fwd_iterator1 ind_start, const fwd_iterator1 ind_end,
		fwd_iterator2 val_start, const fwd_iterator2 val_end,
		const IOMode mode
	) {
		return buildVector( x, ind_start, ind_end, val_start, val_end, mode, Dup() );
	}

	/** \internal No implementation notes: follows the reference implementation. */
	template< Descriptor descr = descriptors::no_operation, typename InputType, typename fwd_iterator >
	RC buildMatrixUnique( Matrix< InputType, banshee > &A, fwd_iterator start, const fwd_iterator end, const IOMode mode ) {
		// parallel or sequential mode are equivalent for banshee implementation
		assert( mode == PARALLEL || mode == SEQUENTIAL );
#ifdef NDEBUG
		(void)mode;
#endif
		return A.template buildMatrixUnique< descr >( start, end );
	}

	/** @} */

} // namespace grb

#undef NO_CAST_ASSERT

#endif // end ``_H_GRB_BANSHEE_IO

