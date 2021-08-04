
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
 * @file parsed_types.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @brief Types that can be parsed, with their default value.
 * @date 2021-04-30
 */

// list here the types you want to be parsed with their default, as follows
// possibly, protect with #ifndef NO_<type> clause

PARSED_TYPE( size_parse_t, 0UL )

PARSED_TYPE( str_parse_t, nullptr )

PARSED_TYPE( double_parse_t, 0.0 )

#ifndef NO_BOOL
PARSED_TYPE( bool_parse_t, false )
#endif
