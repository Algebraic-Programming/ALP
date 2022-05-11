
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
 * @date 31st of January, 2022
 */

#ifndef _H_GRB_HYPERDAGS_EXEC
#define _H_GRB_HYPERDAGS_EXEC

#include <graphblas/backends.hpp>
#include <graphblas/base/exec.hpp>


namespace grb {

	/**
	 * No implementation notes.
	 */
	template< EXEC_MODE mode >
	class Launcher< mode, hyperdags > {

		private:

			typedef Launcher< mode, _GRB_WITH_HYPERDAGS_USING > MyLauncherType;

			MyLauncherType launcher;

		public:

			Launcher( const size_t process_id = 0, const size_t nprocs = 1,
				const std::string hostname = "localhost",
				const std::string port = "0"
			) : launcher( process_id, nprocs, hostname, port ) {}

			template< typename U >
			RC exec( void ( *grb_program )( const void *, const size_t, U & ),
				const void * data_in,
				const size_t in_size,
				U &data_out,
				const bool broadcast = false
			) {
				return launcher.exec( grb_program, data_in, in_size, data_out, broadcast );
			}

			template< typename T, typename U >
			RC exec( void ( *grb_program )( const T &, U & ),
				const T &data_in,
				U &data_out,
				const bool broadcast = false
			) {
				return launcher.exec( grb_program, data_in, data_out, broadcast );
			}

	};

}

#endif

