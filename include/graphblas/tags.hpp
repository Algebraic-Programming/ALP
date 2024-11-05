
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
 * @date 5th of December, 2016
 */

#ifndef _H_GRB_TAGS
#define _H_GRB_TAGS


namespace grb {

	/**
	 * Tags: hints on container structure.
	 *
	 * Provides a backend hints on what kind of storage optimisations are
	 * applicable. For instance, the storage of a symmetric matrix can be
	 * cut in (almost) half. These tags can be provided as hints whenever
	 * a GraphBLAS container is declared. An implementation is free to
	 * ignore such hints.
	 *
	 * Tags can be combined using bit-wise operators.
	 *
	 * \internal
	 *       Tags are not yet put into the specification. Passing them as a
	 *       template to grb::Vector and grb::Matrix creates a combinatorial
	 *       explosion in the number of combinations that must be caught.
	 *       Are there better alternatives?
	 *
	 *       Update 2023: yes there are, see Spampinato et al., ARRAY '23. This
	 *       file will be removed in future releases when it is replaced by the
	 *       concept of \em views and particular that of xMFs that prevent the
	 *       feared combinatorial explosion, both introduced in the aforementioned
	 *       paper.
	 */
	namespace tags {

		/** The type of the tags defined in this namespace. */
		typedef unsigned int Tag;

		/** No specific hint. */
		static constexpr Tag none = 0;

		/**
		 * Declares the input to be symmetric.
		 *
		 * This hint is only applicable to containers of type grb::Matrix. Passing
		 * this hint to grb::Vector will simply result in the hint being ignored.
		 */
		static constexpr Tag symmetric = 1;

	} // namespace tags

} // namespace grb

#endif

