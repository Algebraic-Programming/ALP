
#ifndef _H_GRB_ALGORITHMS_GEOMETRY_HALO_MATRIX_GENRATOR_ITERATOR
#define _H_GRB_ALGORITHMS_GEOMETRY_HALO_MATRIX_GENRATOR_ITERATOR

#include <cstddef>

#include "linearized_halo_ndim_system.hpp"
#include "linearized_ndim_system.hpp"
#include "linearized_ndim_iterator.hpp"
#include "array_vector_storage.hpp"

namespace grb {
	namespace algorithms {
		namespace geometry {

			template<
				size_t DIMS,
				typename CoordType,
				typename ValueType,
				typename ValueCallable
			>
			struct HaloMatrixGeneratorIterator {

				static_assert( std::is_copy_constructible< ValueCallable >::value,
					"ValueCallable must be copy-constructible" );

				using RowIndexType = CoordType; ///< numeric type of rows
				using ColumnIndexType = CoordType;

				using LinearSystemType = grb::utils::geometry::LinearizedHaloNDimSystem< RowIndexType, DIMS >;
				using SelfType = HaloMatrixGeneratorIterator< DIMS, CoordType, ValueType, ValueCallable >;
				using Iterator = typename LinearSystemType::Iterator;

				struct HaloPoint {

					friend SelfType;

					HaloPoint(
						const ValueCallable &value_producer,
						RowIndexType i,
						ColumnIndexType j
					) noexcept :
						_value_producer( value_producer ),
						_i( i ),
						_j( j )
					{}

					HaloPoint( const HaloPoint & ) = default;

					HaloPoint & operator=( const HaloPoint & ) = default;

					inline RowIndexType i() const { return _i; }
					inline ColumnIndexType j() const { return _j; }
					inline ValueType v() const {
						return _value_producer( _i, _j);
					}

				private:
					// ValueType diagonal_value;     ///< value to be emitted when the object has moved to the diagonal
					// ValueType non_diagonal_value; ///< value to emit outside of the diagonal
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
				 * @brief Construct a new \c HaloMatrixGeneratorIterator object, setting the current row as \p row
				 * and emitting \p diag if the iterator has moved on the diagonal, \p non_diag otherwise.
				 *
				 * @param sizes array with the sizes along the dimensions
				 * @param _halo halo of points to iterate around; must be > 0
				 * @param diag value to emit when on the diagonal
				 * @param non_diag value to emit outside the diagonal
				 */
				HaloMatrixGeneratorIterator(
					const LinearSystemType &system,
					const ValueCallable &value_producer
				) noexcept :
					_val( value_producer, 0, 0 ),
					_lin_system( &system ),
					_sys_iter( system.begin() )
				{
					update_coords();
				}

				HaloMatrixGeneratorIterator( const SelfType & ) = default;

				// HaloMatrixGeneratorIterator( SelfType && ) = default;

				SelfType & operator=( const SelfType & ) = default;

				// SelfType & operator=( SelfType && ) = default;

				/**
				 * @brief Increments the iterator by moving coordinates to the next (row, column) to iterate on.
				 *
				 * This operator internally increments the columns coordinates until wrap-around, when it increments
				 * the row coordinates and resets the column coordinates to the first possible columns; this column coordinate
				 * depends on the row coordinates according to the dimensions iteration order and on the parameter \p halo.
				 *
				 * @return HaloMatrixGeneratorIterator<DIMS, T>& \c this object, with the updated state
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

				difference_type operator-( const SelfType &other ) const {
					return this->_sys_iter - other._sys_iter;
				}

				/**
				 * @brief Operator to compare \c this against \p o  and return whether they differ.
				 *
				 * @param o object to compare \c this against
				 * @return true of the row or the column is different between \p o and \c this
				 * @return false if both row and column of \p o and \c this are equal
				 */
				bool operator!=( const SelfType &o ) const {
					return this->_sys_iter != o._sys_iter;
				}

				/**
				 * @brief Operator to compare \c this against \p o  and return whether they are equal.
				 *
				 * @param o object to compare \c this against
				 * @return true of the row or the column is different between \p o and \c this
				 * @return false if both row and column of \p o and \c this are equal
				 */
				bool operator==( const SelfType &o ) const {
					return ! operator!=( o );
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
				 * @brief Returns current row.
				 */
				inline RowIndexType i() const {
					return _val.i();
				}

				/**
				 * @brief Returns current column.
				 */
				inline ColumnIndexType j() const {
					return _val.j();
				}

				/**
				 * @brief Returns the current matrix value.
				 *
				 * @return ValueType #diagonal_value if \code row == column \endcode (i.e. if \code this-> \endcode
				 * #i() \code == \endcode \code this-> \endcode #j()), #non_diagonal_value otherwise
				 */
				inline ValueType v() const {
					return _val.v();
				}

				const Iterator & it() const {
					return this->_sys_iter;
				}

			private:
				value_type _val;
				const LinearSystemType *_lin_system;
				Iterator _sys_iter;

				void update_coords() {
					_val._i = _sys_iter->get_element_linear();
					_val._j = _sys_iter->get_neighbor_linear();
				}
			};



		} // namespace geometry
	} // namespace utils
} // namespace grb

#endif // _H_GRB_ALGORITHMS_GEOMETRY_HALO_MATRIX_GENRATOR_ITERATOR
