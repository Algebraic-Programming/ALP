
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
 * @file token_handlers.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @brief Utility macros for basic manipulation of pre-processing tokens.
 * @date 2021-04-30
 */

#ifndef _H_GRB_TEST_UTILS_TOKEN_HANDLERS_
#define _H_GRB_TEST_UTILS_TOKEN_HANDLERS_

#define __HPCG_CONC( _a, _b ) _a##_b
// because of macro expansion order
#define __HPCG_CONCAT( _a, _b ) __HPCG_CONC( _a, _b )
// to concatenate tokens (after expansion)
#define CONCAT( _a, _b ) __HPCG_CONCAT( _a, _b )

// stringification
#define __HPCG_STRINGIFY( s ) #s
#define STRINGIFY( s ) __HPCG_STRINGIFY( s )

#endif // _H_GRB_TEST_UTILS_TOKEN_HANDLERS_
