
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
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_EXEC
#define _H_GRB_NONBLOCKING_EXEC

#include <graphblas/backends.hpp>
#include <graphblas/base/exec.hpp>

#include "init.hpp"


namespace grb {

	/** The Launcher class is based on that of the reference backend */
	template< EXEC_MODE mode >
	class Launcher< mode, nonblocking > {

		private:
			Launcher< mode, reference > ref;

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
				if( nprocs != 1 ) {
					throw std::invalid_argument( "Total number of user processes must be "
						"exactly one when using the nonblocking implementation."
					);
				}
				if( process_id != 0 ) {
					throw std::invalid_argument( "Process ID must always be zero in the "
						"nonblocking implementation."
					);
				}
			}

			/** No implementation notes. */
			~Launcher() {}

			/** exec is based on that of the reference backend */
			template< typename U >
			RC exec(
				void ( *grb_program )( const void *, const size_t, U & ),
				const void * data_in, const size_t in_size,
				U &data_out, const bool broadcast = false
			) const {
				return ref.exec( grb_program, data_in, in_size, data_out, broadcast );
			}

			/** exec is based on that of the reference backend */
			template< typename T, typename U >
			RC exec( void ( *grb_program )( const T &, U & ), // user GraphBLAS program
				const T &data_in, U &data_out,            // input & output data
				const bool broadcast = false
			) {
				return ref.exec( grb_program, data_in, data_out, broadcast );
			}

			/** finalize is based on that of the reference backend */
			grb::RC finalize() { return ref.finalize(); }
	};

} // namespace grb

#endif // end ``_H_GRB_NONBLOCKING_EXEC''

