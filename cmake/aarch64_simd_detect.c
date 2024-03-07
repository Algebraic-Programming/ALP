
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

/*
 * Check the supported SIMD ISA in an ARM architecture, via getauxval():
 * https://man7.org/linux/man-pages/man3/getauxval.3.html
 * 
 * Note that support for SVE2 may be too recent for the kernel/GLIBC version in
 * use, hence the #ifdef on HWCAP2_SVE2.
 * https://docs.kernel.org/arch/arm64/elf_hwcaps.html
 *
 * Also note that SVE (and SVE2) has implementation-dependant vector size, whose
 * retrieval is currently not implemented; the build infrastructure properly
 * warns about this case.
 */

int main() {

#ifdef HWCAP2_SVE2
	if( getauxval( AT_HWCAP2 ) & HWCAP2_SVE2 ) {
		printf( "SVE2\n" );
		return 0;
	}
#endif

	int retval = 0;
	const unsigned long flags = getauxval( AT_HWCAP );
#ifdef HWCAP_SVE
	if( flags & HWCAP_SVE ) {
		printf("SVE");
	} else
#endif
	if ( flags & HWCAP_ASIMD ) {
		printf( "NEON" );
	} else {
		printf( "no SIMD ISA detected!" );
		retval = 1;
	}
	printf( "\n" );
	return retval;
}
