
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
 * @file internal_argument_parser_defs.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @brief Utility macros to generate various elements for argument parsing.
 *          Do not edit unless you know what you are doing!
 * @date 2021-04-30
 */

#ifndef _H_INTERNAL_ARGUMENT_PARSER
#define _H_INTERNAL_ARGUMENT_PARSER
// including this header twice clean the symbol space of the defined symbols

#include "token_handlers.hpp"

// name of member for default values in union default_value_container
#define DEF_CONT_MEMBER_NAME( _t ) _t##_value

// name for static objects inside argument_parser
#define PARSER_NAME( _t ) CONCAT( _t, _parser )
#define DEF_SET_NAME( _t ) CONCAT( _t, _default_setter )
#define DEF_PRINT_NAME( _t ) CONCAT( _t, _default_printer )

#endif // _H_INTERNAL_ARGUMENT_PARSER
