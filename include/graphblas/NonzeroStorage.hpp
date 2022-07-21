
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

/**
 * Utilities to store and update nonzeroes, for both value and pattern
 * ALP matrices (non-void and void matrices, resp.).
 *
 * @author Alberto Scolari
 * @date 15/06/2022
 */

#ifndef _H_GRB_NONZEROSTORAGE
#define _H_GRB_NONZEROSTORAGE

#include <utility>
#include <type_traits>

#include <graphblas/type_traits.hpp>


namespace grb {

	namespace internal {

		/**
		 * Utility to store a nonzero with row, column and value,
		 * implemented on top of two nested std::pair instances.
		 *
		 * @tparam RowIndexT type of row index
		 * @tparam ColIndexT type of column index
		 * @tparam ValueT type of values
		 *
		 * For internal use only.
		 */
		template<
			typename RowIndexT,
			typename ColIndexT,
			typename ValueT
		>
		class NonzeroStorage :
			public std::pair< std::pair< RowIndexT, ColIndexT >, ValueT >
		{

			public:

				typedef RowIndexT RowIndexType;
				typedef ColIndexT ColumnIndexType;
				typedef ValueT ValueType;
				typedef std::pair< std::pair< RowIndexT, ColIndexT >, ValueT > StorageType;

				NonzeroStorage() = default;

				// rely on compiler to do copy elision
				NonzeroStorage(
					const RowIndexT _row, const ColIndexT _col,
					const ValueT _val
				) noexcept :
					std::pair< std::pair< RowIndexT, ColIndexT >, ValueT >(
						std::make_pair( _row, _col ), _val
					)
				{}

				NonzeroStorage( NonzeroStorage< RowIndexT, ColIndexT, ValueT > && )
					= default;

				NonzeroStorage(
					const NonzeroStorage< RowIndexT, ColIndexT, ValueT > &
				) = default;

				NonzeroStorage< RowIndexT, ColIndexT, ValueT >& operator=(
					NonzeroStorage< RowIndexT, ColIndexT, ValueT > &&
				) = default;

				NonzeroStorage< RowIndexT, ColIndexT, ValueT >& operator=(
					const NonzeroStorage< RowIndexT, ColIndexT, ValueT > &
				) = default;

				RowIndexT & i() { return this->first.first; }
				const RowIndexT & i() const { return this->first.first; }

				ColIndexT & j() { return this->first.second; }
				const ColIndexT & j() const { return this->first.second; }

				ValueT & v() { return this->second; }
				const ValueT & v() const { return this->second; }

				StorageType & storage() { return *this; }
				const StorageType & storage() const { return *this; }

		};

		/**
		 * Utility to store a nonzero with row and column for a pattern matrix,
		 * implemented on top of std::pair
		 *
		 * @tparam RowIndexT type of row index
		 * @tparam ColIndexT type of column index
		 */
		template< typename RowIndexT, typename ColIndexT >
		class NonzeroStorage< RowIndexT, ColIndexT, void > :
			public std::pair< RowIndexT, ColIndexT >
		{

			public:

				typedef RowIndexT RowIndexType;
				typedef ColIndexT ColumnIndexType;
				typedef std::pair< RowIndexT, ColIndexT > StorageType;

				NonzeroStorage() = default;

				NonzeroStorage( const RowIndexT _row, const ColIndexT _col ) noexcept :
					std::pair< RowIndexT, ColIndexT >( _row, _col )
				{}

				NonzeroStorage( NonzeroStorage< RowIndexT, ColIndexT, void >&& ) = default;

				NonzeroStorage(
					const NonzeroStorage< RowIndexT, ColIndexT, void >&
				) = default;

				NonzeroStorage< RowIndexT, ColIndexT, void >& operator=(
					const NonzeroStorage< RowIndexT, ColIndexT, void > & ) = default;

				NonzeroStorage< RowIndexT, ColIndexT, void >& operator=(
					NonzeroStorage< RowIndexT, ColIndexT, void > &&
				) = default;

				RowIndexT & i() { return this->first; }
				const RowIndexT& i() const { return this->first; }

				ColIndexT & j() { return this->second; }
				const ColIndexT& j() const { return this->second; }

				StorageType & storage() { return *this; }
				const StorageType & storage() const { return *this; }

		};

		/**
		 * Updates the coordinates of a #NonzeroStorage instance.
		 *
		 * @param[in,out] update The instance to be updated
		 * @param[in]     row    The new row instance
		 * @param[in]     col    The new column instance
		 *
		 * This is the overload for pattern nonzeroes.
		 */
		template< typename RowIndexT, typename ColIndexT >
		void updateNonzeroCoordinates(
			NonzeroStorage< RowIndexT, ColIndexT, void > &update,
			const size_t &row, const size_t &col
		) {
			update.first = row;
			update.second = col;
		}

		/**
		 * Updates the coordinates of a #NonzeroStorage instance.
		 *
		 * @param[in,out] update The instance to be updated.
		 * @param[in]     row    The new row instance
		 * @param[in]     col    The new column instance
		 *
		 * This is the overload for non-pattern nonzeroes.
		 */
		template<
			typename RowIndexT,
			typename ColIndexT,
			typename V
		>
		void updateNonzeroCoordinates(
			NonzeroStorage< RowIndexT, ColIndexT, V > &update,
			const size_t &row, const size_t &col
		) {
			update.first.first = row;
			update.first.second = col;
		}

		/**
		 * Creates a #NonzeroStorage instance from an ALP matrix iterator.
		 *
		 * @param[in] it The iterator in a valid position.
		 *
		 * @returns The requested instance.
		 *
		 * This is the non-pattern variant.
		 */
		template<
			typename RowIndexT, typename ColIndexT, typename ValueT,
			typename IterT
		>
		inline NonzeroStorage< RowIndexT, ColIndexT, ValueT > makeNonzeroStorage(
			const IterT &it,
			typename std::enable_if<
				grb::internal::iterator_has_value_method< IterT >::value, void *
			>::type = nullptr
		) {
			return NonzeroStorage< RowIndexT, ColIndexT, ValueT >(
				it.i(), it.j(), it.v()
			);
		}

		/**
		 * Creates a #NonzeroStorage instance from an ALP matrix iterator.
		 *
		 * @param[in] it The iterator in a valid position.
		 *
		 * @returns The requested instance.
		 *
		 * This is the pattern variant.
		 */
		template<
			typename RowIndexT,
			typename ColIndexT,
			typename ValueT,
			typename IterT
		>
		inline NonzeroStorage< RowIndexT, ColIndexT, void > makeNonzeroStorage(
			const IterT &it,
			typename std::enable_if<
				!grb::internal::iterator_has_value_method< IterT >::value, void *
			>::type = nullptr
		) {
			return NonzeroStorage< RowIndexT, ColIndexT, void >( it.i(), it.j() );
		}

#ifdef _DEBUG
		/**
		 * Pretty-prints a pattern nonzero.
		 *
		 * \warning Only active in _DEBUG mode.
		 */
		template<
			typename R,
			typename T,
			typename V
		>
		void nonzeroStorage_printer(
			std::ostream &s,
			const NonzeroStorage< R, T, V > &nz,
			typename std::enable_if< !std::is_same< V, void >::value >::type * = nullptr
		) {
			s << ": " << nz.v();
		}

		/**
		 * Pretty-prints a non-pattern nonzero.
		 *
		 * \warning Only active in _DEBUG mode.
		 */
		template<
			typename R,
			typename T,
			typename V
		>
		void nonzeroStorage_printer(
			std::ostream& s,
			const NonzeroStorage< R, T, V > &nz,
			typename std::enable_if< std::is_same< V, void >::value >::type * = nullptr
		) {
			(void) s;
			(void) nz;
		}

		/**
		 * STL I/O-stream overload for instances of #NonzeroStorage.
		 *
		 * \warning Only active in _DEBUG mode.
		 */
		template<
			typename R,
			typename T,
			typename V
		>
		std::ostream& operator<<(
			std::ostream& s,
			const NonzeroStorage< R, T, V > &nz
		) {
			s << "( " << nz.i() << ", " << nz.j() << " )";
				nonzeroStorage_printer( s, nz );
			return s;
		}
#endif

	} // end namespace grb::internal

} // end namespace grb

#endif // end ``_H_GRB_NONZEROSTORAGE''

