
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
 * @file
 *
 * Provides the Launcher for the HyperDAGs backend
 *
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
	class Launcher< mode, hyperdags > :
		public Launcher< mode, _GRB_WITH_HYPERDAGS_USING >
	{

		private:

			typedef Launcher< mode, _GRB_WITH_HYPERDAGS_USING > MyLauncherType;


		public:

			/**
			 * \internal Delegates to #grb::Launcher (reference) constructor. By
			 *           default, this reverts to the <tt>reference</tt> backend.
			 */
			using MyLauncherType::Launcher;

			/**
			 * \internal Delegates to #grb::Launcher finalize. By default, this reverts
			 *           to the <tt>reference</tt> backend.
			 */
			using MyLauncherType::finalize;

	};

}

#endif

