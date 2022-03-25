
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

#ifndef _H_GRB_DENSEREF_COLL
#define _H_GRB_DENSEREF_COLL

namespace grb {

	/** 
	 * \internal Since this backend only has a single user process, the below only
	 *           contains trivial implementations.
	 */
	template<>
	class collectives< reference_dense > {

		private:

			collectives() {}

		public:

			/** \internal Trivial no-op */
			template< Descriptor descr = descriptors::no_operation,
				typename OP, typename IOType
			>
			static RC allreduce( IOType &, const OP &op = OP() ) {
				(void)op;
				return SUCCESS;
			}

			/** \internal Trivial no-op */
			template< Descriptor descr = descriptors::no_operation,
				typename OP, typename IOType
			>
			static RC reduce( IOType &, const size_t root = 0, const OP &op = OP() ) {
				assert( root == 0 );
#ifdef NDEBUG
				(void)root;
#endif
				return SUCCESS;
			}

			template< typename IOType >
			static RC broadcast( IOType &, const size_t root = 0 ) {
				assert( root == 0 );
#ifdef NDEBUG
				(void)root;
#endif
				return SUCCESS;
			}

	};

} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_COLL''

