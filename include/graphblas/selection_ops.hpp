
/*
 *   Copyright 2024 Huawei Technologies Co., Ltd.
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
 * @file
 *
 * Provides a set of standard binary selection operators.
 *
 * @author Benjamin D. Lozes
 * @date 27th of February 2024
 */

#ifndef _H_GRB_SELECTION_OPERATORS
#define _H_GRB_SELECTION_OPERATORS

#include "internalops.hpp"
#include "type_traits.hpp"


namespace grb::operators {

	/**
	 * This namespace holds various standard matrix selection operators:
	 *  - #grb::operators::select::is_diagonal,
	 *  - #grb::operators::select::is_strictly_lower,
	 *  - #grb::operators::select::is_lower_or_diagonal,
	 *  - #grb::operators::select::is_strictly_upper,
	 *  - #grb::operators::select::is_upper_or_diagonal
	 *
	 * These operators may be provided as selection operators to #grb::select.
	 */
	namespace select {

		/**
		 * A matrix selection operator that selects the matrix diagonal.
		 */
		template<
			typename D,
			typename RIT = config::RowIndexType,
			typename CIT = config::ColIndexType
		>
		struct is_diagonal : public internal::MatrixSelectionOperatorBase<
			internal::is_diagonal< D, RIT, CIT >, D
		> {};

		/**
		 * A matrix selection operator that selects the strictly lower triangular
		 * part.
		 */
		template<
			typename D,
			typename RIT = config::RowIndexType,
			typename CIT = config::ColIndexType
		>
		struct is_strictly_lower : public internal::MatrixSelectionOperatorBase<
			internal::is_strictly_lower< D, RIT, CIT >, D
		> {};

		/**
		 * A matrix selection operator that selects the lower triangular part.
		 *
		 * This includes the matrix diagonal.
		 */
		template<
			typename D,
			typename RIT = config::RowIndexType,
			typename CIT = config::ColIndexType
		>
		struct is_lower_or_diagonal : public internal::MatrixSelectionOperatorBase<
			internal::is_lower_or_diagonal< D, RIT, CIT >, D
		> {};

		/**
		 * A matrix selection operator that selects the strictly upper triangular
		 * part.
		 */
		template<
			typename D,
			typename RIT = config::RowIndexType,
			typename CIT = config::ColIndexType
		>
		struct is_strictly_upper : public internal::MatrixSelectionOperatorBase<
			internal::is_strictly_upper< D, RIT, CIT >, D
		> {};

		/**
		 * A matrix selection operator that selects the upper triangular part.
		 *
		 * This includes the matrix diagonal.
		 */
		template<
			typename D,
			typename RIT = config::RowIndexType,
			typename CIT = config::ColIndexType
		>
		struct is_upper_or_diagonal : public internal::MatrixSelectionOperatorBase<
			internal::is_upper_or_diagonal< D, RIT, CIT >, D
		> {};

	} // namespace operators::select

	template< typename D1, typename D2, typename D3 >
	struct is_matrix_selection_operator<
		operators::select::is_diagonal< D1, D2, D3 >
	> {
		static constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3 >
	struct is_matrix_selection_operator<
		operators::select::is_strictly_lower< D1, D2, D3 >
	> {
		static constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3 >
	struct is_matrix_selection_operator<
		operators::select::is_lower_or_diagonal< D1, D2, D3 >
	> {
		static constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3 >
	struct is_matrix_selection_operator<
		operators::select::is_strictly_upper< D1, D2, D3 >
	> {
		static constexpr bool value = true;
	};

	template< typename D1, typename D2, typename D3 >
	struct is_matrix_selection_operator<
		operators::select::is_upper_or_diagonal< D1, D2, D3 >
	> {
		static constexpr bool value = true;
	};

} // end namespace grb

#endif // end ``_H_GRB_SELECTION_OPERATORS''

