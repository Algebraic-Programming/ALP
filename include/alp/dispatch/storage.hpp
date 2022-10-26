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
 *
 * @file
 *
 * This file registers mechanisms for coordinate mapping between
 * logical and physical iteration spaces.
 *
 */

#ifndef _H_ALP_REFERENCE_STORAGE
#define _H_ALP_REFERENCE_STORAGE

#include <alp/amf-based/storage.hpp>

namespace alp {

	namespace internal {

		/** Specialization for any matrix in dispatch backend */
		template< typename Structure >
		struct determine_poly_factory< Structure, imf::Id, imf::Id, dispatch > {

			typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for vectors */
		template< typename Structure >
		struct determine_poly_factory< Structure, imf::Id, imf::Zero, dispatch > {

			typedef storage::polynomials::ArrayFactory factory_type;
		};

	} // namespace internal

} // namespace alp

#endif // _H_ALP_REFERENCE_STORAGE
