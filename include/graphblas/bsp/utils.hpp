
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
 * Provides shared internal utility functions that help with implementing
 * distributed backends.
 *
 * @author A. N. Yzelman
 * @date 21st of February, 2024
 */

#ifndef _H_GRB_BSP_UTILS
#define _H_GRB_BSP_UTILS

#include "graphblas/rc.hpp"


namespace grb::internal {

	/**
	 * An internal helper function to assert an error code is agreed-upon across
	 * all user processes.
	 *
	 * This function must be called collectively.
	 *
	 * In debug mode, if the assertion is violated, a standard assertion will
	 * trigger. When not in debug mode and if the assertion is violated, the
	 * corrected global error code will be returned. In either mode, if the
	 * assertion fails, an error is printed to stderr.
	 */
	RC assertSyncedRC( const RC &in );

} // end namespace grb::internal

#endif // end _H_GRB_BSP_UTILS

