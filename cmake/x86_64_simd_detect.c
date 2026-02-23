
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

/*
 * This file detects the SIMD ISA for x86 architectures using compiler built-in functionalities
 * from https://gcc.gnu.org/onlinedocs/gcc/x86-Built-in-Functions.html#index-_005f_005fbuiltin_005fcpu_005fsupports-1
 * also supported in clang.
 * 
 * Note that the SIMD support can be advertised by the CPU (e.g., via the CPUID
 * instruction) despite being disabled by the Operating System.
 * The compiler's built-in functions check both conditions.
 */

int main() {
	__builtin_cpu_init ();
	int retval = 0;
	if (__builtin_cpu_supports( "avx512f" ) ) {
		printf( "AVX512" );
	} else if ( __builtin_cpu_supports( "avx2" ) ) {
		printf( "AVX2" );
	} else if ( __builtin_cpu_supports( "avx" ) ) {
		printf( "AVX" ); 
	} else if ( __builtin_cpu_supports( "sse" ) ) {
		printf( "sse" );
	} else {
		printf( "no SIMD ISA detected!" );
		retval = 1;
	}
	printf( "\n" );
	return retval;
}
