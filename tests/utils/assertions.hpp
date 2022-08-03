
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

#ifndef _H_GRB_TEST_UTILS_ASSERTIONS
#define _H_GRB_TEST_UTILS_ASSERTIONS

#include <cstdlib>
#include <iostream>

#include <graphblas/spmd.hpp>

#include "assertion_engine.hpp"
#include "token_handlers.hpp"


/**
 * Aborts the current process.
 *
 * Also flushes both stdout and stderr.
 *
 * \warning The interplay of aborting with distributed backends is typically
 *          implementation-defined, and depends on external (upstream)
 *          behaviour.
 *
 * \deprecated It therefore deserves strong recommendation to write all tests
 *             as a parallel program that always gracefully exits, without
 *             calling abort, even if tests fail.
 *
 * \note See also the LPF documentations regarding failure handling.
 */
#define __EXIT( ret )                             \
	if( assertion_engine::exit_on_violation() ) { \
		std::cout.flush();                        \
		std::cerr.flush();                        \
		std::abort();                             \
	}


/**
 * The logger uses std::cout since it does not flush by default: in case of
 * parallel write text chunks from multiple processes are less likely to
 * overlap.
 *
 * \warning No overlap is not guaranteed though.
 *
 * \note In ALP tests, stderr is typically reserved for textual output checks
 *       while test messages meant for human consumption indeed are directed
 *       to stdout.
 */
#define __LOGGER_ERR ( std::cout )

/**
 * Prints a line of text and flushes it.
 *
 * \warning The text is immediately flushed and thus may easily clash with
 *          output from other processes.
 */
#define __PRINT_LINE( TEXT ) __LOGGER_ERR << TEXT << std::endl;

/**
 * Prints a red message.
 *
 * \warning Red output only on Unix consoles.
 *
 * \warning The text is immediately flushed and thus may easily clash with
 *          output from other processes.
 */
#define __PRINT_DBG_LINE_ERR( text ) \
	__LOGGER_ERR << "\x1B[31m" << __FILE__ << ":" << __LINE__ << " "; \
	if( spmd<>::nprocs() > 1 ) { \
		__LOGGER_ERR << "[PID " << spmd<>::pid() << "] "; \
	} \
	__LOGGER_ERR << text << "\033[0m" << std::endl;

/**
 * Prints a red message describing a violated assertion.
 *
 * \warning Red output only on Unix consoles.
 *
 * \warning The text is immediately flushed and thus may easily clash with
 *          output from other processes.
 */
#define __PRINT_EXPR_VIOLATION( EXPR ) __PRINT_DBG_LINE_ERR( "Violated assertion:\t\"" << EXPR << "\"" )

/**
 * Prints a red message describing a violated Boolean condition.
 *
 * \warning Red output only on Unix consoles.
 *
 * \warning The text is immediately flushed and thus may easily clash with
 *          output from other processes.
 */
#define __PRINT_BOOL_FALSE( EXPR ) __PRINT_DBG_LINE_ERR( "False Boolean condition:\t\"" << EXPR << "\"" )

/** \internal A common pattern for a violated assertion. */
#define PRINT_ASSERT_FAILED3( a1, op, a2, __val1 )                                           \
	__PRINT_EXPR_VIOLATION( STRINGIFY( a1 ) << " " << STRINGIFY( op ) << " " << STRINGIFY( a2 ) ) \
	__PRINT_LINE( "-- Actual values: " << __val1 << ", " << a2 );                            \
	__EXIT( -1 );

/** \internal A common pattern triggering when bit-wise unequal */
#define __ASSERT_CMP( actual, CMP_OP, expected )                     \
	{                                                                \
		decltype( actual ) __val { actual };                         \
		if( !( __val CMP_OP ( expected ) ) ) {                      \
			PRINT_ASSERT_FAILED3( actual, CMP_OP, expected, __val ); \
		}                                                            \
	}

/**
 * Assertion that triggers when something is not less than something else.
 *
 * \warning Red output only on Unix consoles.
 *
 * \warning The text is immediately flushed and thus may easily clash with
 *          output from other processes.
 *
 * \deprecated This assertion if failed will lead into an abort. It is strongly
 *             recommended to let tests exit gracefully.
 *
 * \deprecated This assertion if failed will exit the local process with exit
 *             code -1, which is uninformative. It is strongly recommended that
 *             every item checked for in a test, if failed, should result in a
 *             unique error code.
 */
#define ASSERT_LT( actual, expected ) __ASSERT_CMP( actual, <, expected )

/**
 * Assertion that triggers when something is not less than or equal to something
 * else.
 *
 * \warning Red output only on Unix consoles.
 *
 * \warning The text is immediately flushed and thus may easily clash with
 *          output from other processes.
 *
 * \deprecated This assertion if failed will lead into an abort. It is strongly
 *             recommended to let tests exit gracefully.
 *
 * \deprecated This assertion if failed will exit the local process with exit
 *             code -1, which is uninformative. It is strongly recommended that
 *             every item checked for in a test, if failed, should result in a
 *             unique error code.
 */
#define ASSERT_LE( actual, expected ) __ASSERT_CMP( actual, <=, expected )

/**
 * Assertion that triggers when something is not equal to something else.
 *
 * \warning Red output only on Unix consoles.
 *
 * \warning The text is immediately flushed and thus may easily clash with
 *          output from other processes.
 *
 * \deprecated This assertion if failed will lead into an abort. It is strongly
 *             recommended to let tests exit gracefully.
 *
 * \deprecated This assertion if failed will exit the local process with exit
 *             code -1, which is uninformative. It is strongly recommended that
 *             every item checked for in a test, if failed, should result in a
 *             unique error code.
 */
#define ASSERT_EQ( actual, expected ) __ASSERT_CMP( actual, ==, expected )

/**
 * Assertion that triggers when something is equal to something else.
 *
 * \warning Red output only on Unix consoles.
 *
 * \warning The text is immediately flushed and thus may easily clash with
 *          output from other processes.
 *
 * \deprecated This assertion if failed will lead into an abort. It is strongly
 *             recommended to let tests exit gracefully.
 *
 * \deprecated This assertion if failed will exit the local process with exit
 *             code -1, which is uninformative. It is strongly recommended that
 *             every item checked for in a test, if failed, should result in a
 *             unique error code.
 */
#define ASSERT_NE( actual, expected ) __ASSERT_CMP( actual, !=, expected )

/**
 * Assertion that triggers when something is not greater than or equal to
 * something else.
 *
 * \warning Red output only on Unix consoles.
 *
 * \warning The text is immediately flushed and thus may easily clash with
 *          output from other processes.
 *
 * \deprecated This assertion if failed will lead into an abort. It is strongly
 *             recommended to let tests exit gracefully.
 *
 * \deprecated This assertion if failed will exit the local process with exit
 *             code -1, which is uninformative. It is strongly recommended that
 *             every item checked for in a test, if failed, should result in a
 *             unique error code.
 */
#define ASSERT_GE( actual, expected ) __ASSERT_CMP( actual, >=, expected )

/**
 * Assertion that triggers when something is not greater than something else.
 *
 * \warning Red output only on Unix consoles.
 *
 * \warning The text is immediately flushed and thus may easily clash with
 *          output from other processes.
 *
 * \deprecated This assertion if failed will lead into an abort. It is strongly
 *             recommended to let tests exit gracefully.
 *
 * \deprecated This assertion if failed will exit the local process with exit
 *             code -1, which is uninformative. It is strongly recommended that
 *             every item checked for in a test, if failed, should result in a
 *             unique error code.
 */
#define ASSERT_GT( actual, expected ) __ASSERT_CMP( actual, >, expected )

/**
 * Assertion that triggers when something is not true.
 *
 * \warning Red output only on Unix consoles.
 *
 * \warning The text is immediately flushed and thus may easily clash with
 *          output from other processes.
 *
 * \deprecated This assertion if failed will lead into an abort. It is strongly
 *             recommended to let tests exit gracefully.
 *
 * \deprecated This assertion if failed will exit the local process with exit
 *             code -1, which is uninformative. It is strongly recommended that
 *             every item checked for in a test, if failed, should result in a
 *             unique error code.
 */
#define ASSERT_TRUE( bool_cond )				\
	if( !( bool_cond ) ) {                         \
		__PRINT_BOOL_FALSE( STRINGIFY( bool_cond ) ) \
		__EXIT( -1 );								 \
	}

/**
 * Aborts after printing a generic failure message.
 *
 * \warning Red output only on Unix consoles.
 *
 * \warning The text is immediately flushed and thus may easily clash with
 *          output from other processes.
 */
#define FAIL() __PRINT_DBG_LINE_ERR( "Execution failed" )

/**
 * Assertion that triggers when a given return code is not SUCCESS.
 *
 * \warning Red output only on Unix consoles.
 *
 * \warning The text is immediately flushed and thus may easily clash with
 *          output from other processes.
 *
 * \deprecated This assertion if failed will lead into an abort. It is strongly
 *             recommended to let tests exit gracefully.
 *
 * \deprecated This assertion if failed will exit the local process with exit
 *             code -1, which is uninformative. It is strongly recommended that
 *             every item checked for in a test, if failed, should result in a
 *             unique error code.
 */
#define ASSERT_RC_SUCCESS( rc )                                                         \
	{                                                                                   \
		grb::RC __val = rc;                                                           \
		if( __val != grb::RC::SUCCESS ) {                                               \
			__PRINT_DBG_LINE_ERR( "Unsuccessful return value:\t" << toString( __val ) ) \
			__EXIT( -1 );                                                               \
		}                                                                               \
	}

#endif // _H_GRB_TEST_UTILS_ASSERTIONS

