
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
 * @file
 *
 * Supresses warnings that may be unnecessarily emitted by various compilers.
 *
 * Use of the macros defined in this file should always be accompanied with an
 * explanation why it is safe to ignore the compiler warnings that the macro
 * suppresses.
 *
 * @author A. N. Yzelman
 * @date 19th of January 2022
 */

#ifndef _H_GRB_UTILS_SUPRESSIONS
#define _H_GRB_UTILS_SUPRESSIONS

#if defined(__GNUC__) && __GNUC__ >= 4
 // here are the macros for GCC
 #define GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED \
  _Pragma( "GCC diagnostic push" ) ;\
  _Pragma( "GCC diagnostic ignored \"-Wmaybe-uninitialized\"" );\

 #define GRB_UTIL_IGNORE_STRING_TRUNCATION \
  _Pragma( "GCC diagnostic push" );\
  _Pragma( "GCC diagnostic ignored \"-Wstringop-truncation\"" );\

 #define GRB_UTIL_IGNORE_CLASS_MEMACCESS \
  _Pragma( "GCC diagnostic push" ) ;\
  _Pragma( "GCC diagnostic ignored \"-Wclass-memaccess\"" );\

 #define GRB_UTIL_RESTORE_WARNINGS \
  _Pragma( "GCC diagnostic pop" );\

#else
 // here are empty default macros
 #define GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED
 #define GRB_UTIL_IGNORE_STRING_TRUNCATION
 #define GRB_UTIL_IGNORE_CLASS_MEMACCESS
 #define GRB_UTIL_RESTORE_WARNINGS
#endif

#endif // end ``_H_GRB_UTILS_SUPRESSIONS''

