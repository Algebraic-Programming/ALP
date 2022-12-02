
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

#ifndef _H_ALP_OMP_EXEC
#define _H_ALP_OMP_EXEC

#include <alp/backends.hpp>

#include <alp/base/exec.hpp>

#ifdef _ALP_OMP_WITH_REFERENCE
 #include <alp/reference/init.hpp>
#endif
namespace alp {

	/**
	 * \internal No implementation notes.
	 */
	template< EXEC_MODE mode >
	class Launcher< mode, omp > {

		public:

			/** \internal No implementation notes. */
			Launcher(
				const size_t process_id = 0,
				const size_t nprocs = 1,
				const std::string hostname = "localhost",
				const std::string port = "0"
			) {
				(void) process_id;
				(void) nprocs;
				(void) hostname;
				(void) port;
			}

			/** \internal No implementation notes. */
			template< typename U >
			RC exec(
				void ( *alp_program )( const void *, const size_t, U & ),
				const void *data_in, const size_t in_size,
				U &data_out,
				const bool broadcast = false
			) const {
				(void)broadcast; // value doesn't matter for a single user process
				std::cerr << "Entering exec().\n";
				// intialise GraphBLAS
				RC ret = init();
				// call graphBLAS algo
				if( ret == SUCCESS ) {
					( *alp_program )( data_in, in_size, data_out );
				}
				// finalise the GraphBLAS
				if( ret == SUCCESS ) {
					ret = finalize();
				}
				// and done
				return ret;
			}

			/** \internal No implementation notes. */
			template< typename T, typename U >
			RC exec(
				void ( *alp_program )( const T &, U & ),
				const T &data_in, U &data_out,
				const bool broadcast = false
			) {
				(void)broadcast; // value doesn't matter for a single user process
				std::cerr << "Entering exec().\n";
				// intialise GraphBLAS
				RC ret = init();
				// call graphBLAS algo
				if( ret == SUCCESS ) {
					( *alp_program )( data_in, data_out );
				}
				// finalise the GraphBLAS
				if( ret == SUCCESS ) {
					ret = finalize();
				}
				// and done
				return ret;
			}

			/** \internal No implementation notes. */
			static inline RC finalize() {
				return SUCCESS;
			}

	};

} // end namespace ``alp''

#endif // end ``_H_ALP_OMP_EXEC''

