
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

/*
 * @author A. N. Yzelman
 * @date 25th, 26th of May, 2017
 */

#ifndef _H_GRB_UTILS_MATRIXFILE_PROPERTIES
#define _H_GRB_UTILS_MATRIXFILE_PROPERTIES

#include <map>
#include <string>


namespace grb {

	namespace utils {

		namespace internal {

			enum Symmetry {
				General = 0,
				Symmetric,
				SkewSymmetric,
				Hermitian
			};

			struct MatrixFileProperties {

				/** The various files supported for reading. */
				enum Type { MATRIX_MARKET, SNAP };

				/** The filename of the matrix file. */
				std::string _fn;

				/** Row-wise map for indirect datasets. */
				std::map< size_t, size_t > _row_map;

				/** Column-wise map for indirect datasets. */
				std::map< size_t, size_t > _col_map;

				/** The number of rows. */
				size_t _m;

				/** The number of columns. */
				size_t _n;

				/**
				 * The number of nonzeroes.
				 *
				 * Should the number of nonzeroes be unknown a priori, this value shall
				 * equal the maximum representable value in a <tt>size_t</tt>.
				 */
				size_t _nz;

				/**
				 * The number of entries in the file.
				 *
				 * \note This needs not be the same as \a _nz in case of symmetric data
				 *       files.
				 */
				size_t _entries;

				/** Whether the file to be read is pattern-only. */
				bool _pattern;

				/**
				 * Whether the file is symmetric.
				 *
				 * If yes, meaning that this field evaluates not equal to zero, it the
				 * field indicates what symmetry type it is.
				 */
				enum Symmetry _symmetric;

				/**
				 * Whether the file holds complex-valued numbers.
				 */
				bool _complex;

				/**
				 * Whether the file has direct indexing or not.
				 *
				 * If not, a consecutive indexing has to be inferred. This can happen
				 * for row and column indices separately or simultaneously; see
				 * #_symmetricmap.
				 */
				bool _direct;

				/** If true, then _row_map equals _col_map at all times. */
				bool _symmetricmap;

				/** Whether the matrix file is 1-based. */
				bool _oneBased;

				/** The type of the file. */
				Type _type;
			};

		} // namespace internal

	}     // namespace utils

} // namespace grb

#endif // end ``_H_GRB_UTILS_MATRIXFILE_PROPERTIES''

