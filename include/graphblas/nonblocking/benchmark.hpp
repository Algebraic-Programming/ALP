
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

#ifndef _H_GRB_NONBLOCKING_BENCH
#define _H_GRB_NONBLOCKING_BENCH

#include <graphblas/base/benchmark.hpp>
#include <graphblas/rc.hpp>

#include "exec.hpp"


namespace grb {

	/** The Benchmarker class is based on that of the reference backend */
	template< enum EXEC_MODE mode >
	class Benchmarker< mode, nonblocking > {

		private:

			Benchmarker< mode, reference > ref;

		public:

			Benchmarker( size_t process_id = 0,			// user process ID
				size_t nprocs = 1,						// total number of user processes
				std::string hostname = "localhost",		// one of the user process hostnames
				std::string port = "0"					// a free port at hostname
				)
				: ref(process_id, nprocs, hostname, port)
				{}

			template< typename U >
			RC exec( void ( *grb_program )( const void *, const size_t, U & ),
				const void * data_in, const size_t in_size,
				U &data_out,
				const size_t inner, const size_t outer,
				const bool broadcast = false
			) const {

				return ref.exec(grb_program, data_in, in_size, data_out, inner, outer, broadcast);
			}

			template< typename T, typename U >
			RC exec( void ( *grb_program )( const T &, U & ),	// user GraphBLAS program
				const T & data_in, U &data_out,					// input & output data
				const size_t inner,
				const size_t outer,
				const bool broadcast = false
			) {
				return ref.exec(grb_program, data_in, data_out, inner, outer, broadcast);
			}

	};

} // namespace grb

#endif // end ``_H_GRB_NONBLOCKING_BENCH''

