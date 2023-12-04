
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
 * @date 24th of January, 2017
 */

#include <iostream>

#include <stdlib.h>

#include <pthread.h>

#include <graphblas/utils/threadLocalStorage.hpp>


static constexpr size_t P = 4;
static pthread_t handles[ P ];
static int data_array[ P ];

void * spmd( void * const data_p ) {
	int & data = *( static_cast< int * >( data_p ) );

	const double pi = 3.14 + static_cast< double >( data );
	double e = 2.718 + static_cast< double >( data );
	auto store1 = grb::utils::ThreadLocalStorage< double >();
	store1.store();
	auto store2 = grb::utils::ThreadLocalStorage< const double >();
	store2.store( pi );
	auto store3 = grb::utils::ThreadLocalStorage< double >();
	store3.store( e );

	if( store1.cload() != 0 ) {
		(void)printf( "Unexpected default double value %lf. In C++, newly "
					  "constructed doubles are required to be initialised to "
					  "zero.\n",
			store1.cload() );
		data = 1;
		return data_p;
	}

	if( store2.cload() != pi ) {
		(void)printf( "Unexpected value %lf, should have been equal to %lf.\n", store2.cload(), pi );
		data = 2;
		return data_p;
	}

	if( store3.cload() != e ) {
		(void)printf( "Unexpected value %lf, should have been equal to %lf.\n", store3.cload(), e );
		data = 3;
		return data_p;
	}

	e = 2.7182818;
	if( store3.cload() != e ) {
		(void)printf( "Unexpected value %lf, should have been updated to "
					  "%lf.\n",
			store3.cload(), e );
		data = 4;
		return data_p;
	}

	store1.load() += e;
	if( store1.cload() != e ) {
		(void)printf( "Unexpected value %lf, should have been updated to "
					  "%lf.\n",
			store1.cload(), e );
		data = 5;
		return data_p;
	}
	// this should not compile, and indeed does not at the time this test was written.
	// store2.load() += 4;

	e = pi;
	store1.store( store3.load() );
	if( store1.cload() != e ) {
		(void)printf( "Unexpected value %lf, should have been updated to "
					  "%lf.\n",
			store1.cload(), e );
		data = 6;
		return data_p;
	}
	e = 2.71;
	// this also tests auto-deletion, use valgrind to check that there are indeed no memory leaks

	double two = 2;
	store3.store( two );
	if( store3.cload() != two ) {
		(void)printf( "Unexpected value %lf, should have been set to %lf.\n", store3.cload(), two );
		data = 7;
		return data_p;
	}
	if( store1.cload() != e ) {
		(void)printf( "Unexpected value %lf, should have been updated to "
					  "%lf.\n",
			store1.cload(), e );
		data = 8;
		return data_p;
	}

	data = 0;
	return data_p;
}

int main( int argc, char ** argv ) {
	(void)argc;

	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );

	int fail = 0;

	data_array[ 0 ] = 0;
	for( size_t s = 1; s < P; ++s ) {
		data_array[ s ] = static_cast< int >( s );
		const int rc = pthread_create( handles + s, NULL, &spmd, data_array + s );
		if( rc != 0 ) {
			(void)printf( "Unexpected error (%d) while creating thread number "
						  "%zd.\n",
				(int)rc, s );
			return 10;
		}
	}
	if( spmd( data_array ) != data_array ) {
		(void)printf( "Master thread does not retrieve its own data_array.\n" );
		return 12;
	}
	if( data_array[ 0 ] != 0 ) {
		(void)printf( "Master thread reports error code %d.\n", data_array[ 0 ] );
		fail = data_array[ 0 ];
	}
	for( size_t s = 1; s < P; ++s ) {
		void * ret;
		const int rc = pthread_join( handles[ s ], &ret );
		if( rc != 0 ) {
			(void)printf( "Unexpected error (%d) while joining with thread "
						  "number %zd.\n",
				(int)rc, s );
			return 14;
		}
		const int err = *( static_cast< const int * >( ret ) );
		if( err != 0 ) {
			(void)printf( "Thread %zd reports error code %d.\n", s, err );
			fail = err;
		}
	}

	if( fail ) {
		(void)printf( "Test FAILED.\n\n" );
		(void)fflush( stdout );
		(void)fflush( stderr );
		return fail; // return the last encountered error code
	}

	(void)printf( "Test OK.\n\n" );
	(void)fflush( stdout );
	return 0;
}
