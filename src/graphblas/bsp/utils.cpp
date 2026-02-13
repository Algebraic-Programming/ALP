
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

#include "graphblas/bsp/utils.hpp"

#include "graphblas/ops.hpp"
#include "graphblas/collectives.hpp"

#include <iostream>


grb::RC grb::internal::assertSyncedRC( const grb::RC &in ) {
	grb::operators::any_or< grb::RC > reduce_op;
	grb::RC global_rc = in;
	if( grb::collectives< BSP1D >::allreduce( global_rc, reduce_op )
		!= grb::RC::SUCCESS
	) {
		return grb::RC::PANIC;
	}
	if( global_rc != in ) {
		std::cerr << "Internal error: expected a globally in-sync error code, but "
			"I had (" << grb::toString( in ) << ") while someone else had ("
			<< grb::toString( global_rc ) << ").\n";
	}
	assert( global_rc == in );
	return global_rc;
}

