
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
 * @author Antoine Petitet
 * @date June, 2017
 */


/* when compiling the object code, _GNU_SOURCE must be defined
#define _GNU_SOURCE*/

#ifndef _H_HPPARSER
#define _H_HPPARSER

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <ctype.h>
 #include <pthread.h>

#ifndef _GRB_NO_PINNING
 #include <sched.h>
#endif

/* #define VERBOSE */

#define APL_SUCCESS               0
#define APL_ENOMEM                1
#define APL_ESYS                  2
#define APL_EIO                   3
#define APL_EINVAL                4
#define APL_ENOKEY                5
#define APL_EUSAGE                6
#define APL_ENOTSUP               7
#define APL_EINTERN               8
#define APL_ELASTCODE             9


#ifdef __cplusplus
extern "C" {
#endif

int                               ReadEdgeBegin
(
   const char *                   FNAM,
   const ssize_t                  RDBS,
   const int                      PSIZ,
   const int                      PTHR,
   const int                      PRNK,
   size_t *                       NROW,
   size_t *                       NCOL,
   size_t *                       NNNZ,
   void * *                       TPRD
);

int                               ReadEdge
(
   void *                         TPRD,
   size_t *                       NEDG,               /* input output */
   size_t *                       IROW,
   size_t *                       ICOL
);

int                               ReadEdgeEnd
(
   void *                         TPRD
);

int                               TprdCopy
(
   const void * const             TSRC,
   void * * const                 TDST
);

#ifdef __cplusplus
}
#endif

#endif /* end ifndef ``_H_HPPARSER'' */

