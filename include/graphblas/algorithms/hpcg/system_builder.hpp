
/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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
 * @dir include/graphblas/algorithms/hpcg
 * This folder contains the code specific to the HPCG benchmark implementation: generation of the physical system,
 * generation of the single point coarsener and coloring algorithm.
 */

/**
 * @file system_builders.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Utilities to build the system matrix for an HPCG simulation in a generic number of dimensions.
 */

#ifndef _H_GRB_ALGORITHMS_HPCG_SYSTEM_BUILDER
#define _H_GRB_ALGORITHMS_HPCG_SYSTEM_BUILDER

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <cstddef>
#include <iterator>

#include <graphblas/utils/multigrid/halo_matrix_generator_iterator.hpp>

namespace grb {
	namespace algorithms {

		/**
		 * Builder class to build the iterators that generate an HPCG system matrix, describing a
		 * \p DIMS -dimensional simulation mesh for Fourier-like heat propagation.
		 *
		 * @tparam DIMS dimensions of the mesh
		 * @tparam CoordType type storing the coordinates and sizes of the matrix
		 * @tparam ValueType nonzero type
		 */
		template<
			size_t DIMS,
			typename CoordType,
			typename ValueType
		> class HPCGSystemBuilder {
		public:
			struct HPCGDiagGenerator {

				HPCGDiagGenerator(
					ValueType diag,
					ValueType non_diag
				) noexcept :
					_diag( diag ),
					_non_diag( non_diag ) {}

				HPCGDiagGenerator & operator=( const HPCGDiagGenerator & ) = default;

				inline ValueType operator()( const CoordType &i, const CoordType &j ) const noexcept {
					return j == i ? _diag: _non_diag;
				}

				ValueType _diag;
				ValueType _non_diag;
			};

			using HaloSystemType = grb::utils::multigrid::LinearizedHaloNDimSystem< DIMS, CoordType >;
			using Iterator = grb::utils::multigrid::HaloMatrixGeneratorIterator< DIMS, CoordType,
				ValueType, HPCGDiagGenerator >;

			/**
			 * Construct a new HPCGSystemBuilder object from the data of the physical system.
			 *
			 * @param sizes sizes along each dimension
			 * @param halo halo size
			 * @param diag value along the diagonal, for self-interactions
			 * @param non_diag value outside the diagonal, for element-element interaction
			 */
			HPCGSystemBuilder(
				const std::array< CoordType, DIMS > &sizes,
				CoordType halo,
				ValueType diag,
				ValueType non_diag
			) :
				_system( sizes, halo ),
				_diag_generator( diag, non_diag )
			{
				if( halo <= 0 ) {
					throw std::invalid_argument( "halo should be higher than 0" );
				}
				for( const auto i : sizes ) {
					if( i < halo + 1 ) {
						throw std::invalid_argument( "Iteration halo goes beyond system sizes" );
					}
				}
			}

			HPCGSystemBuilder( const HPCGSystemBuilder< DIMS, CoordType, ValueType > & ) = default;

			HPCGSystemBuilder( HPCGSystemBuilder< DIMS, CoordType, ValueType > && ) = default;

			HPCGSystemBuilder< DIMS, CoordType, ValueType > & operator=( const HPCGSystemBuilder< DIMS, CoordType, ValueType > & ) = default;

			HPCGSystemBuilder< DIMS, CoordType, ValueType > & operator=( HPCGSystemBuilder< DIMS, CoordType, ValueType > && ) = default;

			/**
			 * Number of elements of the mesh.
			 */
			size_t system_size() const {
				return _system.base_system_size();
			}

			/**
			 * Total number of neighbors for all elements of the mesh.
			 */
			size_t num_neighbors() const {
				return _system.halo_system_size();
			}

			/**
			 * Get the generator object, i.e. the HaloSystemType object that describes the geometry
			 * of the simulation mesh.
			 */
			const HaloSystemType & get_generator() const {
				return _system;
			}

			/**
			 * Builds the beginning iterator to generate the system matrix.
			 */
			Iterator make_begin_iterator() const {
				return Iterator( _system, _diag_generator );
			}

			/**
			 * Builds the end iterator to generate the system matrix.
			 */
			Iterator make_end_iterator() const {
				Iterator result( _system, _diag_generator );
				result += num_neighbors();
				return result;
			}

			ValueType get_diag_value() const {
				return _diag_generator._diag;
			}

			ValueType get_non_diag_value() const {
				return _diag_generator._non_diag;
			}

		private:
			HaloSystemType _system;
			HPCGDiagGenerator _diag_generator;
		};

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_HPCG_SYSTEM_BUILDER

