
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
 * @brief utilities to store and update nonzeroes, for both value- and
 *        pattern-matrices
 * @author Alberto Scolari
 * @date 15/06/2022
 */

#ifndef _H_GRB_UTILS_NONZEROSTORAGE
#define _H_GRB_UTILS_NONZEROSTORAGE

#include <utility>
#include <type_traits>

#include <graphblas/type_traits.hpp>


namespace grb {

	namespace utils {

		/**
		 * Utiliy to store a nonzero with row, column and value,
		 * implemented on top of two nested std::pair instances.
		 *
		 * @tparam RowIndexT type of row index
		 * @tparam ColIndexT type of column index
		 * @tparam ValueT type of values
		 */
		template<
			typename RowIndexT,
			typename ColIndexT,
			typename ValueT
		>
		struct NonzeroStorage : public std::pair< std::pair< RowIndexT, ColIndexT >, ValueT > {

			typedef RowIndexT RowIndexType;
			typedef ColIndexT ColumnIndexType;
			typedef ValueT ValueType;

			using StorageType = std::pair< std::pair< RowIndexT, ColIndexT >, ValueT >;

			NonzeroStorage() = default;

			// rely on compiler to do copy elision
			NonzeroStorage( RowIndexT _row, ColIndexT _col, ValueT _val ) noexcept :
				std::pair< std::pair< RowIndexT, ColIndexT >, ValueT >(
					std::make_pair( _row, _col ), _val
				) {}

			NonzeroStorage( NonzeroStorage< RowIndexT, ColIndexT, ValueT >&& ) = default;

			NonzeroStorage( const NonzeroStorage< RowIndexT, ColIndexT, ValueT >& ) = default;

			NonzeroStorage< RowIndexT, ColIndexT, ValueT >& operator=(
				NonzeroStorage< RowIndexT, ColIndexT, ValueT > && ) = default;

			NonzeroStorage< RowIndexT, ColIndexT, ValueT >& operator=(
				const NonzeroStorage< RowIndexT, ColIndexT, ValueT > & ) = default;

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
		 * @brief Utiliy to store a nonzero with row and column for a pattern matrix,
		 * 	implemented on top of std::pair
		 *
		 * @tparam RowIndexT type of row index
		 * @tparam ColIndexT type of column index
		 */
		template<
			typename RowIndexT,
			typename ColIndexT
		>
		struct NonzeroStorage< RowIndexT, ColIndexT, void > : public std::pair< RowIndexT, ColIndexT > {

			typedef RowIndexT RowIndexType;
			typedef ColIndexT ColumnIndexType;

			using StorageType = std::pair< RowIndexT, ColIndexT >;

			NonzeroStorage() = default;

			NonzeroStorage( const RowIndexT _row, const ColIndexT _col ) noexcept :
				std::pair< RowIndexT, ColIndexT >( _row, _col ) {}

			NonzeroStorage( NonzeroStorage< RowIndexT, ColIndexT, void >&& ) = default;

			NonzeroStorage( const NonzeroStorage< RowIndexT, ColIndexT, void >& ) = default;

			NonzeroStorage< RowIndexT, ColIndexT, void >& operator=(
				const NonzeroStorage< RowIndexT, ColIndexT, void > & ) = default;

			NonzeroStorage< RowIndexT, ColIndexT, void >& operator=(
				NonzeroStorage< RowIndexT, ColIndexT, void > && ) = default;

			RowIndexT & i() { return this->first; }
			const RowIndexT& i() const { return this->first; }

			ColIndexT & j() { return this->second; }
			const ColIndexT& j() const { return this->second; }

			StorageType & storage() { return *this; }
			const StorageType& storage() const { return *this; }

		};

		template<
			typename RowIndexT,
			typename ColIndexT
		>
		void updateNonzeroCoordinates(
			NonzeroStorage< RowIndexT, ColIndexT, void > &update,
			const size_t &row, const size_t &col
		) {
			update.first = row;
			update.second = col;
		}

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
		 * @brief makes a nonzero out of an input iterator
		 */
		template<
			typename RowIndexT,
			typename ColIndexT,
			typename ValueT,
			typename IterT
		>
		inline NonzeroStorage< RowIndexT, ColIndexT, ValueT > makeNonzeroStorage(
			const IterT &it,
			typename std::enable_if< iterator_has_value_method< IterT >::value, void * >::type = nullptr
		) {
			return NonzeroStorage< RowIndexT, ColIndexT, ValueT >( it.i(), it.j(), it.v() );
		}

		/**
		 * @brief makes a nonzero out of an input iterator for pattern matrices
		 */
		template<
			typename RowIndexT,
			typename ColIndexT,
			typename ValueT,
			typename IterT
		>
		inline NonzeroStorage< RowIndexT, ColIndexT, void > makeNonzeroStorage(
			const IterT &it,
			typename std::enable_if< ! iterator_has_value_method< IterT >::value, void * >::type = nullptr
		) {
			return NonzeroStorage< RowIndexT, ColIndexT, void >( it.i(), it.j() );
		}

#ifdef _DEBUG
		template<
			typename R,
			typename T,
			typename V
		>
		void __val_printer(
			std::ostream &s,
			const NonzeroStorage< R, T, V > &nz,
			typename std::enable_if< !std::is_same< V, void >::value >::type * = nullptr
		) {
			s << ": " << nz.v();

		}

		template<
			typename R,
			typename T,
			typename V
		>
		void __val_printer(
			std::ostream& s,
			const NonzeroStorage< R, T, V > &nz,
			typename std::enable_if< std::is_same< V, void >::value >::type * = nullptr
		) {
			(void) s;
			(void) nz;
		}

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
			__val_printer( s, nz );
            return s;
        }
#endif

	} // namespace utils

} // namespace grb

#endif // end ``_H_GRB_UTILS_NONZEROSTORAGE''

