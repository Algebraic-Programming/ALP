
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

#ifndef _H_GRB_TEST_UTILS_ASSERTIONS_
#define _H_GRB_TEST_UTILS_ASSERTIONS_

#include <cstdlib>
#include <iostream>

#include "assertion_engine.hpp"
#include "token_handlers.hpp"

#define __EXIT( ret )                             \
	if( assertion_engine::exit_on_violation() ) { \
		std::exit( ret );                         \
	}

#define __LOGGER_ERR ( std::cerr )
#define __PRINT_LINE( TEXT ) __LOGGER_ERR << TEXT << std::endl;

// print red message (working only on Unix consoles!)
#define __PRINT_ERR_LINE( TEXT ) __LOGGER_ERR << "\x1B[31m" << TEXT << "\033[0m" << std::endl;

#define __PRINT_DBG_LINE_ERR( text ) __PRINT_ERR_LINE( __FILE__ << ":" << __LINE__ << "  " << text );

#define __PRINT_VIOLATION( EXPR ) __PRINT_DBG_LINE_ERR( "Violated assertion:\t\"" << EXPR << "\"" )

#define PRINT_ASSERT_FAILED3( a1, op, a2, __val1 )                                           \
	__PRINT_VIOLATION( STRINGIFY( a1 ) << " " << STRINGIFY( op ) << " " << STRINGIFY( a2 ) ) \
	__PRINT_LINE( "-- Actual values: " << __val1 << ", " << a2 );                            \
	__EXIT( -1 );

#define __ASSERT_CMP( actual, CMP_OP, expected )                     \
	{                                                                \
		decltype( actual ) __val { actual };                         \
		if( ! ( __val CMP_OP( expected ) ) ) {                       \
			PRINT_ASSERT_FAILED3( actual, CMP_OP, expected, __val ); \
		}                                                            \
	}

#define ASSERT_LT( actual, expected ) __ASSERT_CMP( actual, <, expected )

#define ASSERT_LE( actual, expected ) __ASSERT_CMP( actual, <=, expected )

#define ASSERT_EQ( actual, expected ) __ASSERT_CMP( actual, ==, expected )

#define ASSERT_NE( actual, expected ) __ASSERT_CMP( actual, !=, expected )

#define ASSERT_GE( actual, expected ) __ASSERT_CMP( actual, >=, expected )

#define ASSERT_GT( actual, expected ) __ASSERT_CMP( actual, >, expected )

#define ASSERT( expr )                         \
	if( ! ( expr ) ) {                         \
		__PRINT_VIOLATION( STRINGIFY( expr ) ) \
	}

#define ASSERT_RC_SUCCESS( rc )                                                         \
	{                                                                                   \
		grb::RC __val { rc };                                                           \
		if( __val != grb::RC::SUCCESS ) {                                               \
			__PRINT_DBG_LINE_ERR( "Unsuccessful return value:\t" << toString( __val ) ) \
			__EXIT( -1 );                                                               \
		}                                                                               \
	}

#endif // _H_GRB_TEST_UTILS_ASSERTIONS_
