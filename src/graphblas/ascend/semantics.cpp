
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


#include <vector>

#include <graphblas/ascend/opgen.hpp>
#include <graphblas/ascend/semantics.hpp>
#include <graphblas/ascend/grid.hpp>

namespace alp
{
	namespace internal
	{
		extern iGrid *igrid;
	}
}

bool alp::internal::invalidForEachAxes( const std::vector< int > &axes ) {

	std::vector< int > forEachAxes = internal::vectorOfVectorsToVector( internal::OpGen::forEachAxes );

	for( auto it = axes.cbegin(); it != axes.cend(); ++it ) {

		if( std::find( axes.cbegin(), axes.cend(), *it ) != forEachAxes.cend() 
			&& std::find( axes.cbegin(), axes.cend(), *it ) != it ) {
			return true;
		}
		if( std::find( forEachAxes.cbegin(), forEachAxes.cend(), *it ) != forEachAxes.cend() ) {
			return true;
		}
	}

	return false;
}

bool alp::internal::invalidAxes( const std::vector< int > &axes ) {

	std::vector< int > forEachAxes = internal::vectorOfVectorsToVector( internal::OpGen::forEachAxes );
	std::vector< int > sorted_axes_copy = axes;
	std::vector< int > intersection;

	std::sort( forEachAxes.begin(), forEachAxes.end() );
	std::sort( sorted_axes_copy.begin(), sorted_axes_copy.end() );

	std::set_intersection(
		forEachAxes.begin(), forEachAxes.end(),
		sorted_axes_copy.begin(), sorted_axes_copy.end(),
		std::back_inserter( intersection )
	);

	return ( intersection.size() > 0 );
}

