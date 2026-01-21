
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

#ifndef _H_MATRIXFILE_PROPERTIES
#define _H_MATRIXFILE_PROPERTIES

#include <string>

namespace alp {
	namespace utils {
		namespace internal {

			struct MatrixFileProperties {

				/** The various files supported for reading. */
				enum Type { MATRIX_MARKET, SNAP };

				/** Matrix Market formats. */
				enum MMformats { COORDINATE, ARRAY };

				/** Matrix Market formats. */
				enum MMsymmetries { GENERAL, SYMMETRIC, SKEWSYMMETRIC, HERMITIAN };

				/** Matrix Market formats. */
				enum MMdatatype { REAL, COMPLEX };

				/** The filename of the matrix file. */
				std::string _fn;

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

				/** The type of the file. */
				Type _type;

				/** The type MM format. */
				MMformats _mmformat;

				/** The symmetry type MM format. */
				MMsymmetries _symmetry;

				/** The MM data format. */
				MMdatatype _datatype;

			};

		} // namespace internal
	}     // namespace utils
} // namespace alp

#endif // end ``_H_MATRIXFILE_PROPERTIES''
