
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
 * @date 14th of January 2022
 */

#ifndef _H_ALP_OMP_MATRIX
#define _H_ALP_OMP_MATRIX

#include <alp/backends.hpp>
#include <alp/storage.hpp>
#include <alp/base/matrix.hpp>
#include <alp/amf-based/matrix.hpp>
#include "storagebasedmatrix.hpp"

#ifdef _ALP_OMP_WITH_REFERENCE
 #include <alp/reference/storage.hpp> // For AMFFactory
#endif

namespace alp {

	// Currently no backend specific implementation

	namespace internal {

		/**
		 * Exposes a block of the parallel matrix as a sequential ALP matrix.
		 *
		 * The underlying container (buffer/block) is obtained from the parallel
		 * container, while the AMF is constructed based on the properties of the
		 * block and the applied gather view (i.e., the IMFs associated to it).
		 *
		 */
		template<
			enum view::Views target_view = view::original,
			typename SourceMatrix,
			std::enable_if_t<
				is_matrix< SourceMatrix >::value
			> * = nullptr
		>
		typename internal::new_container_type_from<
			typename SourceMatrix::template view_type< view::cross_backend >::type
		>::template change_backend< config::default_sequential_backend >::type
		get_view( SourceMatrix &source, const size_t tr, const size_t tc, const size_t rt, const size_t br, const size_t bc ) {

			(void) rt;

			// get the container
			const auto &distribution = getAmf( source ).getDistribution();
			const size_t thread_id = tr * distribution.getThreadGridDims().second + tc;
			const size_t block_id = br * distribution.getLocalBlockGridDims( tr, tc ).second + bc;
			auto &container = internal::getLocalContainer( internal::getContainer( source ), thread_id, block_id );

			// make an AMF
			// note: When making a view over a vector, the second imf must be imf::Zero
			const auto block_dims = distribution.getBlockDimensions();

			// Using explicit amf_type to expose its type rather than relying on auto
			// Considerations for improved implementations:
			//  - ideally, the type of AMF and factory should be provided by a type trait
			//    rather than being explicitely specified. To that end, determine_amf_type
			//    should expose the factory in addition to the AMF type. Also factory might
			//    expose the type of AMF it produces
			using amf_type = typename determine_amf_type<
				structures::General, view::CrossBackend< SourceMatrix >, imf::Id, imf::Id, config::default_sequential_backend
			>::type;
			amf_type amf = alp::storage::AMFFactory< config::default_sequential_backend >::FromPolynomial<
				structures::General, imf::Id, imf::Id
			>::Create(
				imf::Id( block_dims.first ), imf::Id( block_dims.second )
			);

			// create a sequential container with the container and AMF above
			using target_t = typename internal::new_container_type_from<
				typename SourceMatrix::template view_type< view::cross_backend >::type
			>::template change_backend< config::default_sequential_backend >::type;

			return target_t( container, amf );
		}

	} // namespace internal

} // namespace alp

#endif // end ``_H_ALP_OMP_MATRIX''
