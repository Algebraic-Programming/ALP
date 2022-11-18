
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

#ifndef _H_GRB_ALGORITHMS_HPCG_COARSENER_BUILDER
#define _H_GRB_ALGORITHMS_HPCG_COARSENER_BUILDER

#include <cstddef>
#include <array>
#include <iterator>
#include <stdexcept>
#include <cmath>

#include <graphblas/utils/multigrid/array_vector_storage.hpp>
#include <graphblas/utils/multigrid/linearized_ndim_system.hpp>

namespace grb {
	namespace algorithms {

		template<
			size_t DIMS,
			typename CoordType,
			typename ValueType
		>
		class HPCGCoarsenerBuilder;

		/**
		 * @brief Class to generate the coarsening matrix of an underlying \p DIMS -dimensional system.
		 *
		 * This class coarsens a finer system to a coarser system by projecting each input value (column),
		 * espressed in finer coordinates, to an output (row) value espressed in coarser coordinates.
		 * The coarser sizes are assumed to be row_generator#physical_sizes, while the finer sizes are here
		 * stored inside #finer_sizes.
		 *
		 * The corresponding refinement matrix is obtained by transposing the coarsening matrix.
		 *
		 * @tparam DIMS number of dimensions of the system
		 * @tparam T type of matrix values
		 */
		template<
			size_t DIMS,
			typename CoordType,
			typename ValueType
		> struct HPCGCoarsenerGeneratorIterator {

			friend HPCGCoarsenerBuilder< DIMS, CoordType, ValueType >;

			using RowIndexType = CoordType; ///< numeric type of rows
			using ColumnIndexType = CoordType;
			using LinearSystemType = grb::utils::multigrid::LinearizedNDimSystem< CoordType,
				grb::utils::multigrid::ArrayVectorStorage< DIMS, CoordType > >;
			using LinearSystemIterType = typename LinearSystemType::Iterator;
			using SelfType = HPCGCoarsenerGeneratorIterator< DIMS, CoordType, ValueType >;
			using ArrayType = std::array< CoordType, DIMS >;

			struct _HPCGValueGenerator {

				friend SelfType;

				_HPCGValueGenerator(
					RowIndexType i,
					ColumnIndexType j
				) noexcept :
					_i( i ),
					_j( j )
				{}

				_HPCGValueGenerator( const _HPCGValueGenerator & ) = default;

				_HPCGValueGenerator & operator=( const _HPCGValueGenerator & ) = default;

				inline RowIndexType i() const { return _i; }
				inline ColumnIndexType j() const { return _j; }
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
			using reference = const value_type&;
			using difference_type = typename LinearSystemIterType::difference_type;

			HPCGCoarsenerGeneratorIterator( const SelfType &o ) = default;

			HPCGCoarsenerGeneratorIterator( SelfType &&o ) = default;

			SelfType & operator=( const SelfType & ) = default;

			SelfType & operator=( SelfType && ) = default;

			/**
			 * @brief Increments the row and the column according to the respective physical sizes,
			 * thus iterating onto the coarsening matrix coordinates.
			 *
			 * @return \code *this \endcode, i.e. the same object with the updates row and column
			 */
			SelfType & operator++() noexcept {
				(void) ++_sys_iter;
				update_coords();
				return *this;
			}

			SelfType & operator+=( size_t offset ) {
				_sys_iter += offset;
				update_coords();
				return *this;
			}

			difference_type operator-( const SelfType &o ) const {
				return this->_sys_iter - o._sys_iter;
			}

			/**
			 * @brief Returns whether \c this and \p o differ.
			 */
			bool operator!=( const SelfType &o ) const {
				return this->_sys_iter != o._sys_iter;
			}

			/**
			 * @brief Returns whether \c this and \p o are equal.
			 */
			bool operator==( const SelfType &o ) const {
				return ! this->operator!=( o );
			}

			/**
			 * @brief Operator returning the triple to directly access row, column and element values.
			 *
			 * Useful when building the matrix by copying the triple of coordinates and value,
			 * like for the BSP1D backend.
			 */
			reference operator*() const {
				return _val;
			}

			pointer operator->() const {
				return &_val;
			}

			/**
			 * @brief Returns the current row, according to the coarser system.
			 */
			inline RowIndexType i() const {
				return _val.i();
			}

			/**
			 * @brief Returns the current column, according to the finer system.
			 */
			inline ColumnIndexType j() const {
				return _val.j();
			}

			/**
			 * @brief Returns always 1, as the coarsening keeps the same value.
			 */
			inline ValueType v() const {
				return _val.v();
			}

		private:
			//// incremented when incrementing the row coordinates; is is the ration between
			//// #finer_sizes and row_generator#physical_sizes
			const LinearSystemType *_lin_sys;
			const ArrayType *_steps; ///< array of steps, i.e. how much each column coordinate (finer system) must be
			LinearSystemIterType _sys_iter;
			value_type _val;

			/**
			 * @brief Construct a new \c HPCGCoarsenerGeneratorIterator object from the coarser and finer sizes,
			 * setting its row at \p _current_row and the column at the corresponding value.
			 *
			 * Each finer size <b>must be an exact multiple of the corresponding coarser size</b>, otherwise the
			 * construction will throw an exception.
			 *
			 * @param _coarser_sizes sizes of the coarser system (rows)
			 * @param _finer_sizes sizes of the finer system (columns)
			 * @param _current_row row (in the coarser system) to set the iterator on
			 */
			HPCGCoarsenerGeneratorIterator(
				const LinearSystemType &system,
				const ArrayType &steps
			) noexcept :
				_lin_sys( &system ),
				_steps( &steps ),
				_sys_iter( _lin_sys->begin() ),
				_val(0, 0)
			{
				update_coords();
			}

			void update_coords() noexcept {
				_val._i = _sys_iter->get_linear_position();
				_val._j = coarse_rows_to_finer_col();
			}

			/**
			 * @brief Returns the row coordinates converted to the finer system, to compute
			 * the column value.
			 */
			ColumnIndexType coarse_rows_to_finer_col() const noexcept {
				ColumnIndexType finer { 0 };
				ColumnIndexType s { 1 };
				for( size_t i { 0 }; i < DIMS; i++ ) {
					s *= (*_steps)[ i ];
					finer += s * _sys_iter->get_position()[ i ];
					s *= _lin_sys->get_sizes()[ i ];
				}
				return finer;
			}
		};

		template<
			size_t DIMS,
			typename CoordType,
			typename ValueType
		> class HPCGCoarsenerBuilder {
		public:
			using ArrayType = std::array< CoordType, DIMS >;
			using Iterator = HPCGCoarsenerGeneratorIterator< DIMS, CoordType, ValueType >;
			using SelfType = HPCGCoarsenerBuilder< DIMS, CoordType, ValueType >;

			HPCGCoarsenerBuilder(
				const ArrayType &_finer_sizes,
				const ArrayType &_coarser_sizes
			) : system( _coarser_sizes.begin(), _coarser_sizes.end() ) {
				for( size_t i { 0 }; i < DIMS; i++ ) {
					// finer size MUST be an exact multiple of coarser_size
					std::ldiv_t ratio = std::ldiv( _finer_sizes[ i ], _coarser_sizes[ i ] );
					if( ratio.quot < 2 || ratio.rem != 0 ) {
						throw std::invalid_argument(
							std::string( "finer size of dimension " ) + std::to_string( i ) +
							std::string( "is not an exact multiple of coarser size" )
						);
					}
					steps[ i ] = ratio.quot;
				}
			}

			HPCGCoarsenerBuilder( const SelfType & ) = delete;

			HPCGCoarsenerBuilder( SelfType && ) = delete;

			SelfType & operator=( const SelfType & ) = delete;

			SelfType & operator=( SelfType && ) = delete;

			size_t system_size() const {
				return system.system_size();
			}

			Iterator make_begin_iterator() {
				return Iterator( system, steps );
			}

			Iterator make_end_iterator() {
				Iterator result( system, steps );
				result += system_size() - 1; // do not trigger boundary checks
				++result;
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
#endif // _H_GRB_ALGORITHMS_HPCG_COARSENER_BUILDER

