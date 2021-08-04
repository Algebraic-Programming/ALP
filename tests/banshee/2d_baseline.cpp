
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

#include <stdio.h>

#include <vector>

#include "graphblas.hpp"
#include "runtime.h"

#define N 100

int main() {
	double *a, *b, *c, *d;

	banshee_allocator< double > all;

	a = all.allocate( N );
	b = all.allocate( N );
	c = all.allocate( N );
	d = all.allocate( N );

	for( int i = 0; i < N; i++ ) {
		a[ i ] = i;
		b[ i ] = 2 * i;
	}
	//    register volatile double ft0 asm("ft0");
	//    register volatile double ft1 asm("ft1");
	//    asm volatile("" : "=f"(ft0), "=f"(ft1));

	//    pulp_ssr_loop_1d(SSR_DM0, N, N, 8, 0);
	//    pulp_ssr_loop_1d(SSR_DM1, N, N, 8, 8);
	//    pulp_ssr_read(SSR_DM0, SSR_1D, a);
	//    pulp_ssr_read(SSR_DM1, SSR_1D, b);
	//    pulp_ssr_enable();

	for( int i = 0; i < N; i++ ) {
		c[ i ] = 0;
		for( int j = 0; j < N; j++ ) {
			c[ i ] += a[ i ] * b[ j ];
			//        asm volatile("fmadd.d %[sum], ft0, ft1, %[sum]"
			//                        : [ sum ] "+f"(sum)
			//                        :
			//                        :"ft0", "ft1");
		}
	}
	//    pulp_ssr_disable();
	//    asm volatile("" ::"f"(ft0), "f"(ft1));

	for( int i = 0; i < N; i++ ) {
		d[ i ] = 0;
		for( int j = 0; j < N; j++ ) {
			d[ i ] += a[ i ] * b[ j ];
		}
	}

	int correct = 1;
	for( int i = 0; i < N; i++ ) {
		if( ! ( ( ( c[ i ] - d[ i ] ) < 0.1 ) || ( ( d[ i ] - c[ i ] ) < 0.1 ) ) ) {
			correct = 0;
		}
	}
	if( correct ) {
		printf( "Correct result.\n" );
	} else {
		printf( "Wrong result.\n" );
	}

	return 0;
}
