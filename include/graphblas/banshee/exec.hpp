
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
 * @author: A. N. Yzelman
 */

#if ! defined _H_GRB_BANSHEE_EXEC
#define _H_GRB_BANSHEE_EXEC

#include <graphblas/backends.hpp>
#include <graphblas/base/exec.hpp>
#include <graphblas/init.hpp>

namespace grb {

	/**
	 * No implementation notes.
	 */
	template< EXEC_MODE mode >
	class Launcher< mode, banshee > {

	public:
		/** This implementation only accepts a single user process. It ignores \a hostname and \a port. */
		Launcher( const size_t process_id = 0,        // user process ID
			const size_t nprocs = 1,                  // total number of user processes
			const std::string hostname = "localhost", // one of the user process hostnames
			const std::string port = "0"              // a free port at hostname
		) {
			// ignore hostname and port
			(void)hostname;
			(void)port;
			// sanity checks
		}

		/** No implementation notes. */
		template< typename U >
		RC exec( void ( *grb_program )( const void *, const size_t, U & ), const void * data_in, const size_t in_size, U & data_out, const bool broadcast = false ) const {
			(void)broadcast; // value doesn't matter for a single user process
			// intialise GraphBLAS
			RC ret = init();
			// call graphBLAS algo
			if( ret == SUCCESS ) {
				( *grb_program )( data_in, in_size, data_out );
			}
			// finalise the GraphBLAS
			if( ret == SUCCESS ) {
				ret = finalize();
			}
			// and done
			return ret;
		}

		/** No implementation notes. */
		template< typename T, typename U >
		RC exec( void ( *grb_program )( const T &, U & ), // user GraphBLAS program
			const T & data_in,
			U & data_out, // input & output data
			const bool broadcast = false ) {
			(void)broadcast; // value doesn't matter for a single user process
			// intialise GraphBLAS
			RC ret = init();
			// call graphBLAS algo
			if( ret == SUCCESS ) {
				( *grb_program )( data_in, data_out );
			}
			// finalise the GraphBLAS
			if( ret == SUCCESS ) {
				ret = finalize();
			}
			// and done
			return ret;
		}

		/** No implementation notes. */
		static inline RC finalize() {
			return SUCCESS;
		}
	};

} // namespace grb

#endif // end ``_H_GRB_BANSHEE_EXEC''
