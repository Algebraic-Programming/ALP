
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


#ifndef _H_GRB_TUTORIAL_VECTOR
#define _H_GRB_TUTORIAL_VECTOR

#include <graphblas/config.hpp>
#include <graphblas/reference/vector.hpp>

namespace grb {


	template< typename D, typename MyCoordinates >
	class Vector< D, tutorial, MyCoordinates > : public Vector< D, reference, MyCoordinates > {
	public:
		Vector( const size_t n ) : Vector< D, reference, MyCoordinates >( n ) {}


		template<
			Descriptor, typename InputType,
			typename fwd_iterator,
			typename Coords, class Dup
		>
		friend RC buildVector(
			Vector< InputType, tutorial, Coords > &, fwd_iterator, const fwd_iterator,
			const IOMode, const Dup &
		);

		template<
			Descriptor descr, typename InputType,
			typename fwd_iterator1, typename fwd_iterator2,
			typename Coords, class Dup
		>
		friend RC buildVector(
			Vector< InputType, tutorial, Coords > &,
			fwd_iterator1, const fwd_iterator1,
			fwd_iterator2, const fwd_iterator2,
			const IOMode, const Dup &
		);
	};


}

#endif // end ``_H_GRB_TUTORIAL_VECTOR''

