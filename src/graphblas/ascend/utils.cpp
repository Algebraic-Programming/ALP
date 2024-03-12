
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

#include <graphblas/ascend/utils.hpp>

namespace alp {

	namespace internal {

		std::string getDataType( const Datatype dtype ) {
			switch( dtype ) {
				case alp::Datatype::FP16:
					return "half";
				case alp::Datatype::FP32:
					return "single";
				case alp::Datatype::VIEW_TYPE:
					return "VIEW_TYPE";
				case alp::Datatype::NO_TYPE:
					return "NO_TYPE";
			}
            std::cerr << "Unknown datatype: " << (int) dtype << std::endl;
            std::abort();
		}

		std::string getScope( const Scope scope ) {
			switch( scope ) {
				case alp::internal::Scope::GLOBAL:
					return "GLOBAL";
				case alp::internal::Scope::LOCAL:
					return "LOCAL";
				case alp::internal::Scope::TEMP:
					return "TEMP";
				case alp::internal::Scope::VIEW:
					return "VIEW";
			}
            std::cerr << "Unknown scope: " << (int) scope << std::endl;
            std::abort();
		}

		std::vector< int > vectorOfVectorsToVector( const std::vector< std::vector< int > > &vector_of_sets ) {
			std::vector< int > vec;
			for( auto it = vector_of_sets.begin(); it != vector_of_sets.end(); ++it ) {
				for( auto jt = it->begin(); jt != it->end(); ++jt ) {
					vec.push_back( *jt );
				}
			}
			return vec;
		}

		std::vector< int > vectorDifference( const std::vector< int > &vector1, const std::vector< int > &vector2 ) {
			std::vector< int > diff;
			for( auto it = vector1.begin(); it != vector1.end(); ++it ) {
				if( std::find( vector2.begin(), vector2.end(), *it ) == std::end( vector2 ) ) {
					diff.push_back( *it );
				}
			}
			return diff;
		}

		bool vectorSubset( const std::vector< int > &vector1, const std::vector< int > &vector2 ) {
			for( auto it = vector1.begin(); it != vector1.end(); ++it ) {
				if( std::find( vector2.begin(), vector2.end(), *it ) == std::end( vector2 ) ) {
					return false;
				}
			}
			return true;
		}

		std::vector< int > vectorUnion( const std::vector< int > &vector1, const std::vector< int > &vector2 ) {
			// create copies for the sorting part below
			std::vector< int > v1 = vector1;
			std::vector< int > v2 = vector2;
			std::vector< int > vec_union;

			// the vectors must be sorted here before using set_union
			// but perhaps this is not what we want
			// on the other hand is unclear which order to maintain
			std::sort( v1.begin(), v1.end() );
			std::sort( v2.begin(), v2.end() );

			std::set_union(
				v1.begin(),
				v1.end(),
				v2.begin(),
				v2.end(),
				std::inserter( vec_union, vec_union.end() )
			);

			return vec_union;
		}

		std::vector< int > intArgsToVector( const int arg ) {
			std::vector< int > set1;
			set1.push_back( arg );
			return set1;
		}
	}
}
