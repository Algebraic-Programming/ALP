
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
 * @dir include/graphblas/utils/multigrid
 * This folder contains various utilities to describe an N-dimensional mesh (possibly with halo)
 * and iterate through its elements and through the neighbors of each element, possible generating
 * a matrix out of this information.
 *
 * These facilities are used to generate system matrices and various inputs for multi-grid simulations.
 */

/**
 * @file halo_matrix_generator_iterator.cpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Definition of HaloMatrixGeneratorIterator.
 */

#ifndef _H_GRB_ALGORITHMS_MULTIGRID_HALO_MATRIX_GENRATOR_ITERATOR
#define _H_GRB_ALGORITHMS_MULTIGRID_HALO_MATRIX_GENRATOR_ITERATOR

#include <cstddef>

#include "array_vector_storage.hpp"
#include "linearized_halo_ndim_system.hpp"
#include "linearized_ndim_iterator.hpp"
#include "linearized_ndim_system.hpp"

namespace grb {
	namespace utils {
		namespace multigrid {

			/**
			 * Iterator type to generate a matrix on top of the couples <element>-<neighbor> of an
			 * \p DIMS -dimensional mesh.
			 *
			 * This iterator is random-access and meets the the interface of an ALP/GraphBLAS
			 * input iterator, i.e. an object of this type \a it has methods \a i(), \a j() and
			 * \a v() to describe a nonzero triplet (row index, column index and value, respectively).
			 *
			 * This data structure is based on the LinearizedHaloNDimIterator class, esentially wrapping the
			 * underlying element index as \a i() and the neighbor index as \a j(); the value \a v()
			 * is user-customizable via a functor of type \p ValueCallable, which emits the nonzero
			 * of type \p ValueType based on the passed values of \a i() and \a j().
			 *
			 * @tparam DIMS number of dimensions
			 * @tparam CoordType tyoe storing the coordinate and the system sizes along each dimension
			 * @tparam ValueType type of nonzeroes
			 * @tparam ValueCallable callable object producing the nonzero value based on \a i() and \a j()
			 */
			template<
				size_t DIMS,
				typename CoordType,
				typename ValueType,
				typename ValueCallable
			> struct HaloMatrixGeneratorIterator {

				static_assert( std::is_copy_constructible< ValueCallable >::value,
					"ValueCallable must be copy-constructible" );

				using RowIndexType = CoordType; ///< numeric type of rows
				using ColumnIndexType = CoordType;
				using LinearSystemType = LinearizedHaloNDimSystem< DIMS, RowIndexType >;
				using SelfType = HaloMatrixGeneratorIterator< DIMS, CoordType, ValueType, ValueCallable >;
				using Iterator = typename LinearSystemType::Iterator;

				struct HaloPoint {

					friend SelfType;

					HaloPoint(
						const ValueCallable & value_producer,
						RowIndexType i,
						ColumnIndexType j
					) noexcept :
						_value_producer( value_producer ),
						_i( i ),
						_j( j ) {}

					HaloPoint( const HaloPoint & ) = default;

					HaloPoint & operator=( const HaloPoint & ) = default;

					inline RowIndexType i() const {
						return _i;
					}
					inline ColumnIndexType j() const {
						return _j;
					}
					inline ValueType v() const {
						return _value_producer( _i, _j );
					}

				private:
					ValueCallable _value_producer;
					RowIndexType _i;
					ColumnIndexType _j;
				};

				// interface for std::random_access_iterator
				using iterator_category = std::random_access_iterator_tag;
				using value_type = HaloPoint;
				using pointer = value_type;
				using reference = value_type;
				using difference_type = typename Iterator::difference_type;

				/**
				 * Construct a new \c HaloMatrixGeneratorIterator object, setting the current row as \p row
				 * and emitting \p diag if the iterator has moved on the diagonal, \p non_diag otherwise.
				 *
				 * @param sizes array with the sizes along the dimensions
				 * @param _halo halo of points to iterate around; must be > 0
				 * @param diag value to emit when on the diagonal
				 * @param non_diag value to emit outside the diagonal
				 */
				HaloMatrixGeneratorIterator(
					const LinearSystemType & system,
					const ValueCallable & value_producer
				) noexcept :
					_val( value_producer, 0, 0 ),
					_lin_system( &system ),
					_sys_iter( system.begin() )
				{
					update_coords();
				}

				HaloMatrixGeneratorIterator( const SelfType & ) = default;

				SelfType & operator=( const SelfType & ) = default;

				/**
				 * Increments the iterator by moving coordinates to the next (row, column) to iterate on.
				 *
				 * This operator internally increments the columns coordinates until wrap-around, when it increments
				 * the row coordinates and resets the column coordinates to the first possible columns;
				 * this column coordinate depends on the row coordinates according to the dimensions
				 * iteration order and on the parameter \p halo.
				 *
				 * @return HaloMatrixGeneratorIterator<DIMS, T>& \c this object, with the updated state
				 */
				SelfType & operator++() noexcept {
					(void)++_sys_iter;
					update_coords();
					return *this;
				}

				SelfType & operator+=( size_t offset ) {
					_sys_iter += offset;
					update_coords();
					return *this;
				}

				difference_type operator-( const SelfType & other ) const {
					return this->_sys_iter - other._sys_iter;
				}

				/**
				 * Operator to compare \c this against \p o  and return whether they differ.
				 *
				 * @param o object to compare \c this against
				 * @return true of the row or the column is different between \p o and \c this
				 * @return false if both row and column of \p o and \c this are equal
				 */
				bool operator!=( const SelfType & o ) const {
					return this->_sys_iter != o._sys_iter;
				}

				/**
				 * Operator to compare \c this against \p o  and return whether they are equal.
				 *
				 * @param o object to compare \c this against
				 * @return true of the row or the column is different between \p o and \c this
				 * @return false if both row and column of \p o and \c this are equal
				 */
				bool operator==( const SelfType & o ) const {
					return ! operator!=( o );
				}

				/**
				 * Operator returning the triple to directly access row, column and element values.
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
				 * Returns the current row.
				 */
				inline RowIndexType i() const {
					return _val.i();
				}

				/**
				 * Returns the current column.
				 */
				inline ColumnIndexType j() const {
					return _val.j();
				}

				/**
				 * Returns the current matrix value.
				 *
				 * @return ValueType #diagonal_value if \code row == column \endcode (i.e. if \code this-> \endcode
				 * #i() \code == \endcode \code this-> \endcode #j()), #non_diagonal_value otherwise
				 */
				inline ValueType v() const {
					return _val.v();
				}

			private:
				value_type _val;
				const LinearSystemType * _lin_system;
				Iterator _sys_iter;

				void update_coords() {
					_val._i = _sys_iter->get_element_linear();
					_val._j = _sys_iter->get_neighbor_linear();
				}
			};

		} // namespace multigrid
	}     // namespace utils
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_HALO_MATRIX_GENRATOR_ITERATOR
