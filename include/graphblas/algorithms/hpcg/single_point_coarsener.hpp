
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
 * @file single_point_coarsener.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Utilities to build the coarsening matrix for an HPCG simulation.
 */

#ifndef _H_GRB_ALGORITHMS_HPCG_SINGLE_POINT_COARSENER
#define _H_GRB_ALGORITHMS_HPCG_SINGLE_POINT_COARSENER

#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
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
		> class SinglePointCoarsenerBuilder;

		/**
		 * Iterator class to generate the coarsening matrix for an HPCG simulation.
		 *
		 * The coarsening matrix samples a single value from the finer space for every element
		 * of the coarser space; this value is the first one (i.e. the one with smallest coordinates)
		 * in the finer sub-space corresponding to each coarser element.
		 *
		 * This coarsening method is simple but can lead to unstable results, especially with certain combinations
		 * of smoothers and partitioning methods.
		 *
		 * This iterator is random-access.
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
		> struct SinglePointCoarsenerIterator {

			friend SinglePointCoarsenerBuilder< DIMS, CoordType, ValueType >;

			using RowIndexType = CoordType; ///< numeric type of rows
			using ColumnIndexType = CoordType;
			using LinearSystemType = grb::utils::multigrid::LinearizedNDimSystem< CoordType, grb::utils::multigrid::ArrayVectorStorage< DIMS, CoordType > >;
			using LinearSystemIterType = typename LinearSystemType::Iterator;
			using SelfType = SinglePointCoarsenerIterator< DIMS, CoordType, ValueType >;
			using ArrayType = std::array< CoordType, DIMS >;

			struct _HPCGValueGenerator {

				friend SelfType;

				_HPCGValueGenerator(
					RowIndexType i,
					ColumnIndexType j
				) noexcept :
					_i( i ),
					_j( j ) {}

				_HPCGValueGenerator( const _HPCGValueGenerator & ) = default;

				_HPCGValueGenerator & operator=( const _HPCGValueGenerator & ) = default;

				inline RowIndexType i() const {
					return _i;
				}
				inline ColumnIndexType j() const {
					return _j;
				}
				inline ValueType v() const {
					return static_cast< ValueType >( 1 );
				}

			private:
				RowIndexType _i;
				ColumnIndexType _j;
			};

			// interface for std::random_access_iterator
			using iterator_category = std::random_access_iterator_tag;
			using value_type = _HPCGValueGenerator;
			using pointer = const value_type;
			using reference = const value_type &;
			using difference_type = typename LinearSystemIterType::difference_type;

			SinglePointCoarsenerIterator( const SelfType & o ) = default;

			SinglePointCoarsenerIterator( SelfType && o ) = default;

			SelfType & operator=( const SelfType & ) = default;

			SelfType & operator=( SelfType && ) = default;

			/**
			 * Advances \c this by 1 in constant time.
			 */
			SelfType & operator++() noexcept {
				(void)++_sys_iter;
				update_coords();
				return *this;
			}

			/**
			 * Advances \c this by \p offset in constant time.
			 */
			SelfType & operator+=( size_t offset ) {
				_sys_iter += offset;
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
			const ArrayType * _steps;
			LinearSystemIterType _sys_iter;
			value_type _val;

			/**
			 * Construct a new SinglePointCoarsenerIterator object starting from the LinearizedNDimSystem
			 * object \p system describing the \b coarser system and the \b ratios \p steps between each finer and
			 * the corresponding corser dimension.
			 *
			 * @param system LinearizedNDimSystem object describing the coarser system
			 * @param steps ratios per dimension between finer and coarser system
			 */
			SinglePointCoarsenerIterator(
				const LinearSystemType & system,
				const ArrayType & steps
			) noexcept :
				_lin_sys( &system ),
				_steps( &steps ),
				_sys_iter( _lin_sys->begin() ),
				_val( 0, 0 )
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
					s *= ( *_steps )[ i ];
					finer += s * _sys_iter->get_position()[ i ];
					s *= _lin_sys->get_sizes()[ i ];
				}
				return finer;
			}
		};

		/**
		 * Builder object to create iterators that generate a coarsening matrix.
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
		> class SinglePointCoarsenerBuilder {
		public:
			using ArrayType = std::array< CoordType, DIMS >;
			using Iterator = SinglePointCoarsenerIterator< DIMS, CoordType, ValueType >;
			using SelfType = SinglePointCoarsenerBuilder< DIMS, CoordType, ValueType >;

			/**
			 * Construct a new SinglePointCoarsenerBuilder object from the sizes of finer system
			 * and those of the coarser system; finer sizes must be an exact multiple of coarser sizes,
			 * otherwise an exception is raised.
			 */
			SinglePointCoarsenerBuilder(
				const ArrayType & _finer_sizes,
				const ArrayType & _coarser_sizes
			) :
				system( _coarser_sizes.begin(),
				_coarser_sizes.end() )
			{
				for( size_t i = 0; i < DIMS; i++ ) {
					// finer size MUST be an exact multiple of coarser_size
					std::ldiv_t ratio = std::ldiv( _finer_sizes[ i ], _coarser_sizes[ i ] );
					if( ratio.quot < 2 || ratio.rem != 0 ) {
						throw std::invalid_argument( std::string( "finer size of dimension " ) + std::to_string( i ) + std::string( "is not an exact multiple of coarser size" ) );
					}
					steps[ i ] = ratio.quot;
				}
			}

			SinglePointCoarsenerBuilder( const SelfType & ) = delete;

			SinglePointCoarsenerBuilder( SelfType && ) = delete;

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
				return Iterator( system, steps );
			}

			/**
			 * Produces an end iterator to stop the generation of the coarsening matrix.
			 */
			Iterator make_end_iterator() {
				Iterator result( system, steps );
				result += system_size(); // do not trigger boundary checks
				return result;
			}

		private:
			const grb::utils::multigrid::LinearizedNDimSystem< CoordType,
				grb::utils::multigrid::ArrayVectorStorage< DIMS, CoordType > > system;

			ArrayType steps; ///< array of steps, i.e. how much each column coordinate (finer system) must be
							 //// incremented when incrementing the row coordinates; is is the ration between
			                 //// #finer_sizes and row_generator#physical_sizes
		};

	} // namespace algorithms
} // namespace grb
#endif // _H_GRB_ALGORITHMS_HPCG_SINGLE_POINT_COARSENER
