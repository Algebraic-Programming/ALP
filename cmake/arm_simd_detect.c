
/*
 *   Copyright 2024 Huawei Technologies Co., Ltd.
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

#include <stdio.h>
#include <sys/auxv.h>

int main() {

#ifdef HWCAP2_SVE2
	if( getauxval( AT_HWCAP2 ) & HWCAP2_SVE2 ) {
		printf( "SVE2\n" );
		return 0;
	}
#endif

	const unsigned long flags = getauxval( AT_HWCAP );
#ifdef HWCAP_SVE
	if( flags & HWCAP_SVE ) {
		printf("SVE");
	} else
#endif
	if ( flags & HWCAP_ASIMD ) {
		printf( "NEON" );
	} else {
		printf( "no SIMD ISA detected!\n" );
		return 1;
	}
	printf( "\n" );
	return 0;
}
