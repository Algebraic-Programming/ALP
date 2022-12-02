
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
#include <alp/type_traits.hpp>

#include <alp/base/matrix.hpp>
#include <alp/amf-based/matrix.hpp>

#include "storage.hpp"
#include "storagebasedmatrix.hpp"
#include "vector.hpp"

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
			typename ThreadCoords,
			std::enable_if_t<
				is_matrix< SourceMatrix >::value
			> * = nullptr
		>
		typename internal::new_container_type_from<
			typename SourceMatrix::template view_type< view::gather >::type
		>::template change_backend< config::default_sequential_backend >::type
		get_view( SourceMatrix &source, const ThreadCoords t, const size_t br, const size_t bc ) {

			// get the container
			const auto &distribution = getAmf( source ).getDistribution();
			const size_t thread_id = distribution.getThreadId( t );
			const size_t block_id = distribution.getLocalBlockId( t, br, bc );
			auto &container = internal::getLocalContainer( internal::getContainer( source ), thread_id, block_id );

			// make an AMF
			// note: When making a view over a vector, the second imf must be imf::Zero

			// Type of AMF factory corresponding to the full local block's AMF
			using original_amf_factory = alp::storage::AMFFactory< config::default_sequential_backend >::FromPolynomial<
				structures::General, imf::Id, imf::Id
			>;

			// AMF factory after applying the global view
			using amf_factory = alp::storage::AMFFactory< config::default_sequential_backend >::Compose<
				imf::Strided, imf::Strided, typename original_amf_factory::amf_type
			>;

			const auto block_dims = distribution.getBlockDimensions();

			typename amf_factory::amf_type amf = amf_factory::Create(
				imf::Id( block_dims.first ),
				imf::Id( block_dims.second ),
				original_amf_factory::Create( imf::Id( block_dims.first ), imf::Id( block_dims.second ) )
			);

			// create a sequential container with the container and AMF above
			using target_t = typename internal::new_container_type_from<
				typename SourceMatrix::template view_type< view::gather >::type
			>::template change_backend< config::default_sequential_backend >::type;

			return target_t( container, amf );
		}

	} // namespace internal

} // namespace alp

#endif // end ``_H_ALP_OMP_MATRIX''
