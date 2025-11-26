
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
 * @date 3rd of August, 2017
 */

#include "graphblas/rc.hpp"


std::string grb::toString( const grb::RC code ) {
	std::string ret;
	switch( code ) {
		case grb::SUCCESS:
			ret = "Success";
			break;
		case grb::PANIC:
			ret = "Panic (unrecoverable)";
			break;
		case grb::OUTOFMEM:
			ret = "Out-of-memory";
			break;
		case grb::MISMATCH:
			ret = "Mismatching dimensions during call";
			break;
		case grb::OVERLAP:
			ret = "Overlapping containers given while this defined to be "
				  "illegal";
			break;
		case grb::OVERFLW:
			ret = "A cast of a given argument to a given smaller data type "
				  "would result in overflow";
			break;
		case grb::UNSUPPORTED:
			ret = "The chosen backend does not support the requested call";
			break;
		case grb::ILLEGAL:
			ret = "An illegal user argument was detected";
			break;
		case grb::FAILED:
			ret = "A GraphBLAS algorithm has failed to achieve its intended "
				  "result (e.g., has not converged)";
			break;
		default:
			ret = "Uninterpretable error code detected, please notify the "
				  "developers.";
	}
	return ret;
}
