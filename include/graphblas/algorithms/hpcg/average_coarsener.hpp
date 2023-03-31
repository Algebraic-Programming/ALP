
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
 * @file average_coarsener.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Utilities to build the coarsening matrix for an HPCG simulation.
 */

#ifndef _H_GRB_ALGORITHMS_AVERAGE_COARSENER
#define _H_GRB_ALGORITHMS_AVERAGE_COARSENER

#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <stdexcept>

#include <graphblas/utils/multigrid/array_vector_storage.hpp>
#include <graphblas/utils/multigrid/linearized_ndim_system.hpp>

namespace grb {
	namespace algorithms {

		// forward declaration
		template<
			size_t DIMS,
			typename CoordType,
			typename ValueType
		> class AverageCoarsenerBuilder;

		/**
		 * Iterator class to generate the coarsening matrix that averages over the elements of the finer
		 * domain corresponding to the element of the coarser domain.
		 *
		 * The coarsening matrix averages \b all elements that are coarsened into one.
		 *
		 * This coarsening method requires some computation but should be relatively robust to noise
		 * or to partitioning strategies to parallelize the smoother (usually run before coarsening).
		 *
		 * This iterator is random-access.
		 *
		 * @tparam DIMS number of dimensions
		 * @tparam CoordType type storing the coordinates and the sizes
		 * @tparam ValueType type of the nonzero: it must be able to represent 1 /
		 * 	<number of finer elements per coarser elements>
		 */
		template<
			size_t DIMS,
			typename CoordType,
			typename ValueType
		> struct AverageGeneratorIterator {

			friend AverageCoarsenerBuilder< DIMS, CoordType, ValueType >;

			using RowIndexType = CoordType; ///< numeric type of rows
			using ColumnIndexType = CoordType;
			using LinearSystemType = grb::utils::multigrid::LinearizedNDimSystem< CoordType,
				grb::utils::multigrid::ArrayVectorStorage< DIMS, CoordType > >;
			using LinearSystemIterType = typename LinearSystemType::Iterator;
			using SelfType = AverageGeneratorIterator< DIMS, CoordType, ValueType >;
			using ArrayType = std::array< CoordType, DIMS >;

			struct _ValueGenerator {

				friend SelfType;

				_ValueGenerator(
					RowIndexType i,
					ColumnIndexType j,
					ValueType value
				) noexcept :
					_i( i ),
					_j( j ),
					_value( value ) {}

				_ValueGenerator( const _ValueGenerator & ) = default;

				_ValueGenerator & operator=( const _ValueGenerator & ) = default;

				inline RowIndexType i() const {
					return _i;
				}
				inline ColumnIndexType j() const {
					return _j;
				}
				inline ValueType v() const {
					return _value;
				}

			private:
				RowIndexType _i;
				ColumnIndexType _j;
				ValueType _value;
			};

			// interface for std::random_access_iterator
			using iterator_category = std::random_access_iterator_tag;
			using value_type = _ValueGenerator;
			using pointer = const value_type;
			using reference = const value_type &;
			using difference_type = typename LinearSystemIterType::difference_type;

			AverageGeneratorIterator( const SelfType & o ) = default;

			AverageGeneratorIterator( SelfType && o ) = default;

			SelfType & operator=( const SelfType & ) = default;

			SelfType & operator=( SelfType && ) = default;

			/**
			 * Advances \c this by 1 in constant time.
			 */
			SelfType & operator++() noexcept {
				(void)++_subspace_iter;
				size_t subspace_position = _subspace_iter->get_linear_position();
				// std::cout << "subspace_position " << subspace_position << std::endl;
				if( subspace_position == _num_neighbors ) {
					(void)++_sys_iter;
					_subspace_iter = _finer_subspace->begin();
				}
				update_coords();
				return *this;
			}

			/**
			 * Advances \c this by \p offset in constant time.
			 */
			SelfType & operator+=( size_t offset ) {
				CoordType sub_offset = _subspace_iter->get_linear_position() + offset;
				std::ldiv_t res = std::ldiv( sub_offset, _num_neighbors );
				_sys_iter += res.quot;
				_subspace_iter = _finer_subspace->begin();
				_subspace_iter += res.rem;
				update_coords();
				return *this;
			}

			/**
			 * Computes the difference between \c this and \p o as integer.
			 */
			difference_type operator-( const SelfType & o ) const {
				return this->_sys_iter - o._sys_iter;
			}

			/**
			 * Returns whether \c this and \p o differ.
			 */
			bool operator!=( const SelfType & o ) const {
				return this->_sys_iter != o._sys_iter;
			}

			/**
			 * Returns whether \c this and \p o are equal.
			 */
			bool operator==( const SelfType & o ) const {
				return ! this->operator!=( o );
			}

			reference operator*() const {
				return _val;
			}

			pointer operator->() const {
				return &_val;
			}

			/**
			 * Returns the current row, within the coarser system.
			 */
			inline RowIndexType i() const {
				return _val.i();
			}

			/**
			 * Returns the current column, within the finer system.
			 */
			inline ColumnIndexType j() const {
				return _val.j();
			}

			/**
			 * Returns always 1, as the coarsening keeps the same value.
			 */
			inline ValueType v() const {
				return _val.v();
			}

		private:
			const LinearSystemType * _lin_sys;
			const LinearSystemType * _finer_subspace;
			const ArrayType * _steps;
			CoordType _num_neighbors;
			LinearSystemIterType _sys_iter;
			LinearSystemIterType _subspace_iter;
			value_type _val;

			/**
			 * Construct a new AverageGeneratorIterator object starting from the LinearizedNDimSystem
			 * object \p system describing the \b coarser system and the \b ratios \p steps between each finer and
			 * the corresponding corser dimension.
			 *
			 * @param system LinearizedNDimSystem object describing the coarser system
			 * @param finer_subspace LinearizedNDimSystem object describing the subspace of each element
			 *  in the finer system
			 * @param steps ratios per dimension between finer and coarser system
			 */
			AverageGeneratorIterator(
				const LinearSystemType & system,
				const LinearSystemType & finer_subspace,
				const ArrayType & steps
			) noexcept :
				_lin_sys( &system ),
				_finer_subspace( &finer_subspace ),
				_steps( &steps ),
				_num_neighbors( std::accumulate( steps.cbegin(), steps.cend(), 1UL, std::multiplies< CoordType >() ) ),
				_sys_iter( system.begin() ),
				_subspace_iter( finer_subspace.begin() ),
				_val( 0, 0, static_cast< ValueType >( 1 ) / static_cast< ValueType >( _num_neighbors ) )
			{
				update_coords();
			}

			void update_coords() noexcept {
				_val._i = _sys_iter->get_linear_position();
				_val._j = coarse_rows_to_finer_col();
			}

			/**
			 * Returns the row coordinates converted to the finer system, to compute
			 * the column value.
			 */
			ColumnIndexType coarse_rows_to_finer_col() const noexcept {
				ColumnIndexType finer = 0;
				ColumnIndexType s = 1;
				for( size_t i = 0; i < DIMS; i++ ) {
					finer += s * _subspace_iter->get_position()[ i ];
					s *= ( *_steps )[ i ];
					finer += s * _sys_iter->get_position()[ i ];
					s *= _lin_sys->get_sizes()[ i ];
				}
				return finer;
			}
		};

		/**
		 * Builder object to create iterators that generate an averaging-coarsening matrix.
		 *
		 * It is a facility to generate beginning and end iterators and abstract the logic away from users.
		 *
		 * @tparam DIMS number of dimensions
		 * @tparam CoordType type storing the coordinates and the sizes
		 * @tparam ValueType type of the nonzero: it must be able to represent 1 (the value to sample
		 *  the finer value)
		 */
		template<
			size_t DIMS,
			typename CoordType,
			typename ValueType
		> class AverageCoarsenerBuilder {
		public:
			using ArrayType = std::array< CoordType, DIMS >;
			using Iterator = AverageGeneratorIterator< DIMS, CoordType, ValueType >;
			using SelfType = AverageCoarsenerBuilder< DIMS, CoordType, ValueType >;

			/**
			 * Construct a new AverageCoarsenerBuilder object from the sizes of finer system
			 * and those of the coarser system; finer sizes must be an exact multiple of coarser sizes,
			 * otherwise an exception is raised.
			 */
			AverageCoarsenerBuilder(
				const ArrayType & _finer_sizes,
				const ArrayType & _coarser_sizes
			) :
				system( _coarser_sizes.begin(), _coarser_sizes.end() ),
				_finer_subspace( _coarser_sizes.cbegin(), _coarser_sizes.cend() ),
				steps( DIMS )
			{
				for( size_t i = 0; i < DIMS; i++ ) {
					// finer size MUST be an exact multiple of coarser_size
					std::ldiv_t ratio = std::ldiv( _finer_sizes[ i ], _coarser_sizes[ i ] );
					if( ratio.quot < 2 || ratio.rem != 0 ) {
						throw std::invalid_argument( std::string( "finer size of dimension " )
							+ std::to_string( i ) + std::string( "is not an exact multiple of coarser size" ) );
					}
					steps[ i ] = ratio.quot;
				}
				_finer_subspace.retarget( steps );
			}

			AverageCoarsenerBuilder( const SelfType & ) = delete;

			AverageCoarsenerBuilder( SelfType && ) = delete;

			SelfType & operator=( const SelfType & ) = delete;

			SelfType & operator=( SelfType && ) = delete;

			/**
			 * Returns the size of the finer system, i.e. its number of elements.
			 */
			size_t system_size() const {
				return system.system_size();
			}

			/**
			 * Produces a beginning iterator to generate the coarsening matrix.
			 */
			Iterator make_begin_iterator() {
				return Iterator( system, _finer_subspace, steps );
			}

			/**
			 * Produces an end iteratormto stop the generation of the coarsening matrix.
			 */
			Iterator make_end_iterator() {
				Iterator result( system, _finer_subspace, steps );
				result += ( system_size() * _finer_subspace.system_size() ); // do not trigger boundary checks
				// ++result;
				return result;
			}

		private:
			const grb::utils::multigrid::LinearizedNDimSystem< CoordType,
				grb::utils::multigrid::ArrayVectorStorage< DIMS, CoordType > > system;
			grb::utils::multigrid::LinearizedNDimSystem< CoordType,
				grb::utils::multigrid::ArrayVectorStorage< DIMS, CoordType > > _finer_subspace;
			///
			/// array of steps, i.e. how much each column coordinate (finer system) must be
			/// incremented when incrementing the row coordinates; it is the ratio between
			//// #finer_sizes and row_generator#physical_sizes
			grb::utils::multigrid::ArrayVectorStorage< DIMS, CoordType > steps;
		};

	} // namespace algorithms
} // namespace grb
#endif // _H_GRB_ALGORITHMS_AVERAGE_COARSENER
