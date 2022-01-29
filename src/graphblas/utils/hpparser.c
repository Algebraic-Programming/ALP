
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
 * @date June 2017
 */


#include "graphblas/utils/hpparser.h"

#define APL_CHAR_0              '0'
#define APL_CHAR_9              '9'
#define APL_CHAR_EOL           '\n'

#define APL_TPIO_4K            4096

#define APL_MIN(   x_, y_ )           ( ( (x_) < (y_) ) ?  (x_) : (y_) )
#define APL_MAX(   x_, y_ )           ( ( (x_) > (y_) ) ?  (x_) : (y_) )
#define APL_CEIL(  x_, y_ )               ( ( (x_) + (y_) - 1 ) / (y_) )

#define APL_TPIO_PREAD( FDES_, BCUR_, LREA_, OFFS_, IERR_ )            \
   do                                                                  \
   {                                                                   \
      ssize_t nwrd_;                                                   \
                                                                       \
      do                                                               \
      {                                                                \
         if( ( nwrd_ = pread( FDES_, BCUR_, LREA_, OFFS_ ) ) != -1 )   \
         { OFFS_ += (off_t )(nwrd_); LREA_ -= nwrd_; BCUR_ += nwrd_; } \
      } while( ( nwrd_ != -1 ) && ( LREA_ > 0 ) );                     \
                                                                       \
      IERR_ = ( nwrd_ != -1 ? APL_SUCCESS : APL_EIO );                 \
   } while( 0 )

#define APL_TPIO_BOLN_COUNT( BCUR_, BEND_, NEDG_ )                     \
   do                                                                  \
   {                                                                   \
      char * bcur_ = (char *)(BCUR_);                                  \
                                                                       \
      while( bcur_ < (BEND_) )                                         \
      {                                                                \
         while( *bcur_ != APL_CHAR_EOL ) bcur_++;                      \
         (NEDG_)++;                                                    \
         bcur_++;                                                      \
      }                                                                \
   } while( 0 )

#define APL_TPIO_BOLN_ADJUST( TWRD_, BEND_, OFFS_ )                    \
   do                                                                  \
   {                                                                   \
      if( TWRD_ )                                                      \
      {                                                                \
         while( *(BEND_) != APL_CHAR_EOL )                             \
         { (BEND_)--; (OFFS_)--; (TWRD_)++; }                          \
      }                                                                \
   } while( 0 )

#define APL_TSKL_tlft( L_ )               ( ((APL_Tskl_p *)(L_))->tlft )
#define APL_TSKL_shdn( L_ )               ( ((APL_Tskl_p *)(L_))->shdn )
#define APL_TSKL_cycl( L_ )               ( ((APL_Tskl_p *)(L_))->cycl )
#define APL_TSKL_ierr( L_ )               ( ((APL_Tskl_p *)(L_))->ierr )
#define APL_TSKL_iloc( L_ )               ( ((APL_Tskl_p *)(L_))->iloc )
#define APL_TSKL_mutx( L_ )               ( ((APL_Tskl_p *)(L_))->mutx )
#define APL_TSKL_strt( L_ )               ( ((APL_Tskl_p *)(L_))->strt )
#define APL_TSKL_wait( L_ )               ( ((APL_Tskl_p *)(L_))->wait )
#define APL_TSKL_tfun( L_ )               ( ((APL_Tskl_p *)(L_))->tfun )
#define APL_TSKL_targ( L_ )               ( ((APL_Tskl_p *)(L_))->targ )
#define APL_TSKL_tprd( L_ )               ( ((APL_Tskl_p *)(L_))->tprd )

#define APL_TprdARGS(      A_ )       (  (APL_TprdArgs_p *)(A_)        )
#define APL_TprdARGS_ithr( A_ )       ( ((APL_TprdArgs_p *)(A_))->ithr )
#define APL_TprdARGS_fdes( A_ )       ( ((APL_TprdArgs_p *)(A_))->fdes )
#define APL_TprdARGS_nedg( A_ )       ( ((APL_TprdArgs_p *)(A_))->nedg )
#define APL_TprdARGS_bcur( A_ )       ( ((APL_TprdArgs_p *)(A_))->bcur )
#define APL_TprdARGS_irow( A_ )       ( ((APL_TprdArgs_p *)(A_))->irow )
#define APL_TprdARGS_icol( A_ )       ( ((APL_TprdArgs_p *)(A_))->icol )
#define APL_TprdARGS_tprd( A_ )       ( ((APL_TprdArgs_p *)(A_))->tprd )

#define APL_TPRD(      T_ )               (  (APL_Tprd_p *)(T_)        )
#define APL_TPRD_prnk( M_ )            (   ((APL_Tprd_p *)(M_))->prnk  )
#define APL_TPRD_psiz( M_ )            (   ((APL_Tprd_p *)(M_))->psiz  )
#define APL_TPRD_pthr( M_ )            (   ((APL_Tprd_p *)(M_))->pthr  )
#define APL_TPRD_nsiz( M_ )            (   ((APL_Tprd_p *)(M_))->nsiz  )
#define APL_TPRD_prvp( M_ )            (   ((APL_Tprd_p *)(M_))->prvp  )
#define APL_TPRD_tskl( M_ )            ( &(((APL_Tprd_p *)(M_))->tskl) )
#define APL_TPRD_thrd( M_ )            (   ((APL_Tprd_p *)(M_))->thrd  )
#define APL_TPRD_fsiz( T_ )               ( ((APL_Tprd_p *)(T_))->fsiz )
#define APL_TPRD_offb( T_ )               ( ((APL_Tprd_p *)(T_))->offb )
#define APL_TPRD_offe( T_ )               ( ((APL_Tprd_p *)(T_))->offe )
#define APL_TPRD_nedg( T_ )               ( ((APL_Tprd_p *)(T_))->nedg )
#define APL_TPRD_rdbs( T_ )               ( ((APL_Tprd_p *)(T_))->rdbs )
#define APL_TPRD_buff( T_ )               ( ((APL_Tprd_p *)(T_))->buff )
#define APL_TPRD_toff( T_ )               ( ((APL_Tprd_p *)(T_))->toff )
#define APL_TPRD_tlen( T_ )               ( ((APL_Tprd_p *)(T_))->tlen )
#define APL_TPRD_csum( T_ )               ( ((APL_Tprd_p *)(T_))->csum )
#define APL_TPRD_cprm( T_ )               ( ((APL_Tprd_p *)(T_))->cprm )
#define APL_TPRD_tedg( T_ )               ( ((APL_Tprd_p *)(T_))->tedg )
#define APL_TPRD_aedg( T_ )               ( ((APL_Tprd_p *)(T_))->aedg )
#define APL_TPRD_args( T_ )               ( ((APL_Tprd_p *)(T_))->args )
#define APL_TPRD_fnam( T_ )               ( ((APL_Tprd_p *)(T_))->fnam )

/* ****************************************************************** */

typedef int                       APL_TskF_p( void * );

typedef void *                    APL_Tskl_t;
typedef void *                    APL_Tprd_t;

typedef struct
{
   int                            tlft;
   int                            shdn;
   int                            cycl;
   int                            ierr;
   int                            iloc;
   pthread_mutex_t                mutx;
   pthread_cond_t                 strt;
   pthread_cond_t                 wait;
   APL_TskF_p *                   tfun;
   void * *                       targ;
   APL_Tprd_t                     tprd;
} APL_Tskl_p; /* Must be copied / reinitialised, all via constructor */

typedef void *                    APL_TprdArgs_t;

typedef struct
{
   int                            ithr;            /* Must be copied (via constructor) */
   int                            fdes;            /* Must be copied (via constructor) */
   size_t                         nedg;            /* Must be copied */
   char *                         bcur;            /* Must be copied */
   size_t *                       irow;            /* User argument, no need for copy */
   size_t *                       icol;            /* User argument, no need for copy */
   APL_Tprd_t                     tprd;            /* Must be reinitialised (via constructor) */
} APL_TprdArgs_p;

typedef struct
{
   int                            prnk;            /* Must be copied (via constructor) */
   int                            psiz;            /* Must be copied (via constructor) */
   int                            pthr;            /* Must be copied (via constructor) */
   int                            nsiz;            /* Must be copied (via constructor) */
   int                            prvp;            /* Must be copied (via constructor) */
   APL_Tskl_p                     tskl;            /* Must be copied (via constructor) */
   pthread_t *                    thrd;            /* Must be re-initialised (via constructor) */
   off_t                          fsiz;            /* Must be copied (via constructor) */
   off_t                          offb;            /* Must be copied (via constructor) */
   off_t                          offe;            /* Must be copied (via constructor) */
   size_t                         nedg;            /* Must be copied */
   ssize_t                        rdbs;            /* Read block size, must be copied (via constructor) */
   char *                         buff;            /* Must be copied (deep copy) */
   off_t *                        toff;            /* Must be copied (deep copy) */
   ssize_t *                      tlen;            /* Must be copied (deep copy) */
   size_t *                       csum;            /* This is a cache, no copy required */
   int *                          cprm;            /* This is a cache, no copy required */
   size_t *                       tedg;            /* Must be copied (deep copy) */
   size_t *                       aedg;            /* This is a cache, no copy required */
   void * *                       args;            /* Threads args, must be copied individually */
   char                           fnam[256];       /* Must be copied (via constructor) */
} APL_Tprd_p;

/* ****************************************************************** */

static int                               APL_TsklInit
(
   APL_Tprd_t                     TPRD
)
{
   int                            iept;
   APL_Tskl_t                     tskl;

   tskl = APL_TPRD_tskl( TPRD );

   APL_TSKL_shdn( tskl ) = 1;        /* bare minimum to start up with */
   APL_TSKL_tprd( tskl ) = TPRD;
                         /* Need the mutex to access shdn for example */
   iept = pthread_mutex_init( &(APL_TSKL_mutx( tskl )), NULL );

   return( ( iept == 0 ? APL_SUCCESS : APL_ESYS ) );
}

static int                               APL_TsklSelf
(
   const APL_Tskl_t               TSKL
)
{
   int                            imid, lbnd = 1, ubnd;
   pthread_t                      ptid, * thrd;
   APL_Tprd_t                     tprd;

   ptid = pthread_self();
   tprd = APL_TSKL_tprd( TSKL );
   thrd = APL_TPRD_thrd( tprd );
/*
 * In a thread loop, the process is also the main thread. Its id is
 * stored in thrd[0]. Only pthr - 1 new threads are created during
 * start up.
 */
   if( ptid == thrd[0] ) return( 0 );

   ubnd = APL_TPRD_pthr( tprd ) - 1;

   while( lbnd <= ubnd )
   {
      imid = lbnd + ( ( ubnd - lbnd ) / 2 );

      if(      thrd[imid] == ptid ) { return( imid );  }
      else if( thrd[imid] >  ptid ) { ubnd = imid - 1; }
      else                          { lbnd = imid + 1; }
   }

   return( -1 );
}

static int                               APL_TsklDrain
(
   APL_Tskl_t                     TSKL,
   const int                      IERR
)
{
   int                            cycl, iept, ierr = APL_SUCCESS, pthr, tlft;

   if( TSKL == NULL ) return( APL_EINVAL );

   if( ( pthr = APL_TPRD_pthr( APL_TSKL_tprd( TSKL ) ) ) > 1 )
   {
      if( ( iept = pthread_mutex_lock(   &(APL_TSKL_mutx( TSKL )) ) ) != 0 )
      { (void) fprintf( stderr, "mutex_lock %s\n", strerror( iept ) ); }

      cycl = APL_TSKL_cycl( TSKL );

      tlft = --(APL_TSKL_tlft( TSKL ));

      if( IERR != APL_SUCCESS )
      {                                  /* Return at least one error */
         APL_TSKL_ierr( TSKL ) = IERR;
         APL_TSKL_iloc( TSKL ) = APL_TsklSelf( TSKL );
      }

      if( tlft )
      {
         while( ( iept == 0 ) && ( cycl == APL_TSKL_cycl( TSKL ) ) )
         {
            if( ( iept = pthread_cond_wait( &(APL_TSKL_wait( TSKL )),
                                            &(APL_TSKL_mutx( TSKL )) ) ) != 0 )
            { (void) fprintf( stderr, "cond_wait %s\n", strerror( iept ) ); }
         }
      }
      else
      {                                                     /* Reload */
         APL_TSKL_tlft( TSKL )  = pthr;
         APL_TSKL_cycl( TSKL ) ^= 0x01;
         APL_TSKL_tfun( TSKL )  = NULL;

         iept = pthread_cond_broadcast( &(APL_TSKL_wait( TSKL )) );
         if( iept != 0 )
         { (void) fprintf( stderr, "cond_broadcast %s\n", strerror( iept ) ); }
      }

      if( ( iept = pthread_mutex_unlock( &(APL_TSKL_mutx( TSKL )) ) ) != 0 )
      { (void) fprintf( stderr, "mutex_unlock %s\n", strerror( iept ) ); }

      ierr = ( iept == 0 ? APL_SUCCESS : APL_ESYS );
   }
   else
   {
      APL_TSKL_ierr( TSKL ) = IERR;
      APL_TSKL_iloc( TSKL ) = 0;
   }

   if( APL_TSKL_ierr( TSKL ) == APL_SUCCESS ) return( ierr );

   return( APL_TSKL_ierr( TSKL ) );
}

static void *                            APL_TsklThread
(
   void *                         TSKL
)
{
   int                            iept = 0, ierr = APL_SUCCESS, ithr = -1;

   do
   {
      if( ( iept = pthread_mutex_lock(   &(APL_TSKL_mutx( TSKL )) ) ) != 0 )
      { (void) fprintf( stderr, "mutex_lock %s\n", strerror( iept ) ); }

      while( ( iept == 0 ) && ( APL_TSKL_tfun( TSKL ) == NULL ) &&
             ( ! APL_TSKL_shdn( TSKL ) ) )
      {
         iept = pthread_cond_wait( &(APL_TSKL_strt( TSKL )),
                                   &(APL_TSKL_mutx( TSKL )) );
      }

      if( ( iept = pthread_mutex_unlock( &(APL_TSKL_mutx( TSKL )) ) ) != 0 )
      { (void) fprintf( stderr, "mutex_unlock %s\n", strerror( iept ) ); }

      if( iept == 0 )
      {
         if( APL_TSKL_shdn( TSKL ) ) pthread_exit( NULL );

         if( APL_TSKL_tfun( TSKL ) )
         {
            if( APL_TSKL_targ( TSKL ) )
            {        /* Ensure that the loop Start is completely done */
               if( ithr == -1 ) ithr = APL_TsklSelf( TSKL );
               ierr = (*(APL_TSKL_tfun( TSKL )))( APL_TSKL_targ( TSKL )[ithr] );
            }
            else
            {
               ierr = (*(APL_TSKL_tfun( TSKL )))( NULL );
            }

            (void) APL_TsklDrain( (APL_Tskl_t)(TSKL), ierr );
         }
      }

   } while( iept == 0 );

   if( iept != 0 )
   {
      (void) fprintf( stderr, "Error in APL_TsklThread %d\n", iept );
      exit( 1 );
   }

   return( NULL );
}

static int                               APL_TsklTidCmp
(
   const void *                   A,
   const void *                   B
)
{
   pthread_t                      a = *((pthread_t *)(A)),
                                  b = *((pthread_t *)(B));

   return( ( a > b ? 1 : ( a < b ? -1 : 0 ) ) );
}

static int                               APL_TsklSort
(
   const APL_Tprd_t               TPRD
)
{
   int                            pthr;
/*
 * In a thread loop, the process is also the main thread. Its id is
 * stored in thrd[0]. Only pthr - 1 new threads are created during
 * start up.
 */
   if( ( pthr = APL_TPRD_pthr( TPRD ) ) > 1 )
   {
      qsort( &(APL_TPRD_thrd( TPRD )[1]), (size_t)(pthr - 1),
             sizeof( pthread_t ), APL_TsklTidCmp );
   }

   return( APL_SUCCESS );
}

static int                               APL_TsklStart
(
   const APL_Tprd_t               TPRD
)
{
   int                            ithr, iept = 0, pthr;
   pthread_t *                    thrd;
   APL_Tskl_t                     tskl;

   tskl = APL_TPRD_tskl( TPRD );

   APL_TSKL_tlft( tskl ) = pthr = APL_TPRD_pthr( TPRD );
   APL_TSKL_shdn( tskl ) = APL_TSKL_cycl( tskl ) =  0;

   APL_TSKL_tfun( tskl ) = NULL;
   APL_TSKL_targ( tskl ) = NULL;

   if( pthr == 1 ) return( APL_SUCCESS );
/*
 * More than one thread
 *
 * The mutex is initialized during initialization which is always
 * performed at APL instance creation.
 */
   if( iept == 0 )
   {
      if( ( iept = pthread_cond_init( &(APL_TSKL_strt( tskl )),  NULL ) ) != 0 )
      { (void) fprintf( stderr, "cond_init %s\n", strerror( iept ) ); }
   }

   if( iept == 0 )
   {
      if( ( iept = pthread_cond_init( &(APL_TSKL_wait( tskl )),  NULL ) ) != 0 )
      { (void) fprintf( stderr, "cond_init %s\n", strerror( iept ) ); }
   }
/*
 * Create (only) pthr - 1 threads
 */
   thrd = APL_TPRD_thrd( TPRD );
   thrd[0] = pthread_self();                           /* main thread */

   for( ithr = 1; ( iept == 0 ) && ( ithr != pthr ); ithr++ )
   {
      iept = pthread_create( &(thrd[ithr]), NULL, APL_TsklThread,
                             (void *)(tskl) );
      if( iept ) { (void) fprintf( stderr, "create %s\n", strerror( iept ) ); }
   }

#ifndef _GRB_NO_PINNING
/*
 * Sort the thread ids for later fast retrieval. Note that in the case
 * of the loop only the last pthr - 1 thread ids are sorted.
 *
 * Bind the newly created threads as well as the main thread.
 */
   if( iept == 0 )
   {
      int       icor;
      cpu_set_t cpuset;
      const int maxthreads = sysconf( _SC_NPROCESSORS_ONLN );

      (void) APL_TsklSort( TPRD );

      /* get current mask */
      if( sched_getaffinity( 0, sizeof( cpu_set_t ), &cpuset ) != 0 ) {
         return APL_ESYS;
      }

      icor = maxthreads - 1;
      for( ithr = 0; ithr < pthr; ithr++ )
      {
         cpu_set_t threadset;

         /* find next free hardware thread */
         do
         {
            if( ++icor >= maxthreads ) icor = 0;
         } while( CPU_ISSET( icor, &cpuset ) == 0 );

         CPU_ZERO( &threadset );
         CPU_SET( icor, &threadset );
         (void) pthread_setaffinity_np( thrd[ithr], sizeof( cpu_set_t ),
                                        &threadset );
      }
   }
#else
   if( iept == 0 )
   {
      (void) APL_TsklSort( TPRD );
   }
#endif

   return( ( iept == 0 ? APL_SUCCESS : APL_ESYS ) );
}

static int                               APL_InstTsklSpawn
(
   const APL_Tprd_t               TPRD,
   APL_TskF_p *                   TFUN,
   void * *                       TARG
)
{
   int                            ierr = APL_SUCCESS, iept = 0;
   APL_Tskl_t                     tskl;

   if( ( tskl = APL_TPRD_tskl( TPRD ) ) == NULL ) return( APL_EINVAL );

   if( APL_TSKL_shdn( tskl ) == 1 ) ierr = APL_TsklStart( TPRD );

   APL_TSKL_ierr( tskl ) = APL_SUCCESS; APL_TSKL_iloc( tskl ) = -1;

   if( ( ierr == APL_SUCCESS ) && ( APL_TPRD_pthr( TPRD ) > 1 ) )
   {                                               /* wake up threads */
      APL_TSKL_tfun( tskl ) = TFUN;
      APL_TSKL_targ( tskl ) = TARG;

      if( ( iept = pthread_mutex_lock(   &(APL_TSKL_mutx( tskl )) ) ) != 0 )
      { (void) fprintf( stderr, "mutex_lock %s\n", strerror( iept ) ); }

      if( iept == 0 )
      {
         iept = pthread_cond_broadcast( &(APL_TSKL_strt( tskl )) );

         if( iept != 0 )
         { (void) fprintf( stderr, "cond_broadcast %s\n", strerror( iept ) ); }
      }

      if( iept == 0 )
      {
         if( ( iept = pthread_mutex_unlock( &(APL_TSKL_mutx( tskl )) ) ) != 0 )
         { (void) fprintf( stderr, "mutex_unlock %s\n", strerror( iept ) ); }
      }

      ierr = ( iept == 0 ? APL_SUCCESS : APL_ESYS );
   }

   if( ierr == APL_SUCCESS ) ierr = TFUN( TARG[0] );

   ierr = APL_TsklDrain( tskl, ierr );

   return( ierr );
}

static int                               APL_InstTsklJoin
(
   const APL_Tprd_t               TPRD
)
{
   int                            ithr, iept = 0, pthr;
   pthread_t *                    thrd;
   APL_Tskl_t                     tskl;
/* ..
 * .. Executable Statements ..
 */
   tskl = APL_TPRD_tskl( TPRD );

   pthr = APL_TPRD_pthr( TPRD );
   thrd = APL_TPRD_thrd( TPRD );

   if( pthr > 1 )
   {
      if( ( iept = pthread_mutex_lock( &(APL_TSKL_mutx( tskl )) ) ) != 0 )
      { (void) fprintf( stderr, "mutex_init %s", strerror( iept ) ); }
/*
 * Is a shutdown already in progress ?
 */
      if( ( iept == 0 ) && APL_TSKL_shdn( tskl ) )
      {
         if( ( iept = pthread_mutex_unlock( &(APL_TSKL_mutx( tskl )) ) ) != 0 )
         { (void) fprintf( stderr, "mutex_unlock %s\n", strerror( iept ) ); }

         return( APL_SUCCESS );
      }

      APL_TSKL_shdn( tskl ) = 1;

      if( iept == 0 )
      {
         if( ( iept = pthread_mutex_unlock( &(APL_TSKL_mutx( tskl )) ) ) != 0 )
         { (void) fprintf( stderr, "mutex_unlock %s\n", strerror( iept ) ); }
      }

      if( iept == 0 )
      {
         iept = pthread_cond_broadcast( &(APL_TSKL_strt( tskl )) );

         if( iept != 0 )
         { (void) fprintf( stderr, "cond_broadcast %s\n", strerror( iept ) ); }
      }
/*
 * Wait for workers to exit
 */
      for( ithr = 1; ( iept == 0 ) && ( ithr < pthr ); ithr++ )
      {
         if( ( iept = pthread_join( thrd[ithr], NULL ) ) != 0 )
         { (void) fprintf( stderr, "join %s\n", strerror( iept ) ); }
      }
/*
 * The mutex is destroyed when the APL instance is finalized, so that
 * it remains possible to access (thread-)safely the task loop.
 */
      if( iept == 0 )
      {
         iept = pthread_cond_destroy( &(APL_TSKL_strt( tskl )) );
         if( iept != 0 )
         { (void) fprintf( stderr, "cond_destroy %s\n", strerror( iept ) ); }
      }

      if( iept == 0 )
      {
         iept = pthread_cond_destroy( &(APL_TSKL_wait( tskl )) );
         if( iept != 0 )
         { (void) fprintf( stderr, "cond_destroy %s\n", strerror( iept ) ); }
      }
   }
   else
   {
      APL_TSKL_shdn( tskl ) = 1;
   }

   return( ( iept == 0 ? APL_SUCCESS : APL_ESYS ) );
}

static int                               APL_TprdOpen
(
   const char *                   FNAM,
   const int                      PRVP,
   const int                      NSIZ,
   const int                      PRNK,
   const int                      PSIZ,
   int *                          FDES,
   off_t *                        FSIZ,
   off_t *                        OFFB,
   off_t *                        OFFE,
   size_t *                       NROW,
   size_t *                       NCOL,
   size_t *                       NNNZ
)
{
   char                           fchr;
   int                            fdes, idim;
   size_t                         edim;
   off_t                          fsiz, offb, offe, offh, pchk;
   struct stat                    fsta;

   if( FNAM == NULL ) return( APL_EINVAL );

   if( ( fdes = open( FNAM, O_RDONLY ) ) != -1  )
   {
      (void) fstat( fdes, &fsta ); fsiz = fsta.st_size;

      offh = 0;                          /* Skip Matrix Market header */

      while( read( fdes, &fchr, 1 ) != -1 )
      {                                              /* skip comments */
         if( fchr == '%' || fchr == '#' )
         {
            offh++;

            while( read( fdes, &fchr, 1 ) != -1 )
            { offh++; if( fchr == '\n' ) break; }
         }
         else if( NROW != NULL )
         {                            /* Skip first line (dimensions) */
            idim = 0;

            do
            {
               offh++;

               if( ( fchr >= APL_CHAR_0 ) && ( fchr <= APL_CHAR_9 ) )
               {
                  edim = 0;

                  do
                  {
                     if( ( fchr >= APL_CHAR_0 ) && ( fchr <= APL_CHAR_9 ) )
                     { edim *= 10; edim += fchr - APL_CHAR_0; }
                     else break;

                     offh++;

                  } while( read( fdes, &fchr, 1 ) != -1 );

                  if(      idim == 0 ) *NROW = edim;
                  else if( idim == 1 ) *NCOL = edim;
                  else if( idim == 2 ) *NNNZ = edim;

                  idim++;
               }

               if( fchr == '\n' ) break;
            }
            while( read( fdes, &fchr, 1 ) != -1 );

            break;
         }
         else
            break;
      }

      pchk = APL_CEIL( fsiz - offh + 1, (off_t)(PSIZ) );

      if(      ( offb = offh + (off_t)(PRVP) * pchk ) >= fsiz )
      { offb = offe = offh; }
      else if( ( offe = offb + (off_t)(NSIZ) * pchk ) >= fsiz )
      { offe = fsiz - (off_t)(1); }

      if( offb != offh )
      {                                          /* adjust if not eol */
         offb--; (void) lseek( fdes, offb, SEEK_SET );
         while( read( fdes, &fchr, 1 ) != -1 )
         { offb++; if( fchr == '\n' ) break; }
      }

      if( ( offe != offh ) && ( offe != fsiz - (off_t)(1) ) )
      {                                          /* adjust if not eol */
         offe--; (void) lseek( fdes, offe, SEEK_SET );
         while( read( fdes, &fchr, 1 ) != -1 )
         { if( fchr == '\n' ) break; offe++; }
      }
                                               /* leave the file open */
      *FDES = fdes; *FSIZ = fsiz; *OFFB = offb; *OFFE = offe;

      return( APL_SUCCESS );
   }

   (void) fprintf( stderr, "[%d] Cannot open file %s ...\n", PRNK, FNAM );

   return( APL_EIO );
}

static int                               APL_TprdSizeOf
(
   const int                      PTHR,
   const ssize_t                  RDBS,
   size_t *                       LTPR,
   size_t *                       IOFF
)
{
   int                            pthr;
   size_t                         algn, ioff[10], ltpr;

   pthr = PTHR;

   ltpr = sizeof( APL_Tprd_p );

   algn    = sizeof( char    );                               /* buff */
   ioff[0] = algn * APL_CEIL( ltpr, algn );
   ltpr    = ioff[0] + algn * (size_t)(RDBS) * (size_t)(pthr);

   algn    = sizeof( off_t   );                               /* toff */
   ioff[1] = algn * APL_CEIL( ltpr, algn );
   ltpr    = ioff[1] + algn * (size_t)(pthr);

   algn    = sizeof( ssize_t );                               /* tlen */
   ioff[2] = algn * APL_CEIL( ltpr, algn );
   ltpr    = ioff[2] + algn * (size_t)(pthr);

   algn    = sizeof( size_t );                                /* csum */
   ioff[3] = algn * APL_CEIL( ltpr, algn );
   ltpr    = ioff[3] + algn * (size_t)(pthr);

   algn    = sizeof( int    );                                /* cprm */
   ioff[4] = algn * APL_CEIL( ltpr, algn );
   ltpr    = ioff[4] + algn * (size_t)(pthr);

   algn    = sizeof( size_t );                                /* tedg */
   ioff[5] = algn * APL_CEIL( ltpr, algn );
   ltpr    = ioff[5] + algn * (size_t)(pthr);

   algn    = sizeof( size_t );                                /* aedg */
   ioff[6] = algn * APL_CEIL( ltpr, algn );
   ltpr    = ioff[6] + algn * (size_t)(pthr) * (size_t)(pthr);

   algn    = sizeof( void * );                    /* pointers to args */
   ioff[7] = algn * APL_CEIL( ltpr, algn );
   ltpr    = ioff[7] + algn * (size_t)(pthr);

   algn    = sizeof( APL_TprdArgs_p );                        /* args */
   ioff[8] = algn * APL_CEIL( ltpr, algn );
   ltpr    = ioff[8] + algn * (size_t)(pthr);

   algn    = sizeof( pthread_t );                        /* pthread_t */
   ioff[9] = algn * APL_CEIL( ltpr, algn );
   ltpr    = ioff[9] + algn * (size_t)(pthr);

   IOFF[0] = ioff[0]; IOFF[1] = ioff[1]; IOFF[2] = ioff[2];
   IOFF[3] = ioff[3]; IOFF[4] = ioff[4]; IOFF[5] = ioff[5];
   IOFF[6] = ioff[6]; IOFF[7] = ioff[7]; IOFF[8] = ioff[8];
   IOFF[9] = ioff[9];

   *LTPR   = ltpr;

   return( APL_SUCCESS );
}

static int                               APL_TprdOffset
(
   const int                      FDES,
   const off_t                    OFFB,
   const off_t                    OFFE,
   const off_t                    LPRT,
   const int                      PTHR,
   off_t *                        POFF,
   ssize_t *                      PLEN
)
{
   char                           fchr;
   int                            iprt;
   off_t                          offb = OFFB, lprt = LPRT;

   for( iprt = 0; offb < OFFE; iprt++ )
   {              /* Compute logical part offsets and length (cyclic) */
      if( OFFE - offb + (off_t)(1)  < lprt ) lprt = OFFE - offb + (off_t)(1);

      POFF[iprt] = offb; PLEN[iprt] = lprt;

      if( offb + lprt < OFFE )
      {                                      /* adjust offb + plen */
         (void) lseek( FDES, offb + lprt - (off_t)(1), SEEK_SET );

         while( read( FDES, &fchr, 1 ) != -1 )
         { if( fchr == APL_CHAR_EOL ) break; PLEN[iprt]++; }
      }
      else
      { PLEN[iprt] = OFFE - offb + (off_t)(1); }

      offb += PLEN[iprt];
   }

   for( ; iprt < PTHR; ++iprt )
   {
      PLEN[iprt] = 0;
   }
   return( APL_SUCCESS );
}

static int                               APL_TprdOpenThread
(
   void *                         DATA
)
{
   int                            fdes;

   if( ( fdes = APL_TprdARGS_fdes( DATA ) ) == - 1 )
   {                                      /* The file is not open yet */
      if( ( fdes = open( APL_TprdARGS_bcur( DATA ), O_RDONLY ) ) != -1 )
      {                                                    /* Success */
         APL_TprdARGS_fdes( DATA ) = fdes;

         return( APL_SUCCESS );
      }

      return( APL_EIO );
   }

   return( APL_SUCCESS );
}

static int                               APL_TprdFree
(
   APL_Tprd_t *                   TPRD
)
{
   APL_Tskl_t                     tskl;
   int                            ithr;
   void * *                       args;

   if( *TPRD )
   {
      for( args = APL_TPRD_args( *TPRD ), ithr = 0;
           ithr < APL_TPRD_pthr( *TPRD ); ithr++ )
      {
         if( APL_TprdARGS_fdes( args[ithr] ) != -1 )
         {
            (void) close( APL_TprdARGS_fdes( args[ithr] ) );

            APL_TprdARGS_fdes( args[ithr] ) = -1;
         }
      }

      (void) APL_InstTsklJoin( *TPRD );        /* Join the threads */
      tskl = APL_TPRD_tskl( *TPRD );
      (void) pthread_mutex_destroy( &(APL_TSKL_mutx( tskl )) );

      free( *TPRD );

      *TPRD = NULL;
   }

   return( APL_SUCCESS );
}

static int                               APL_TprdCsumSort
(
   const int                      NSUM,
   size_t *                       CSUM,
   int *                          PERM
)
{
   int                            iprm, isum, jsum;

   for( isum = 0; isum < NSUM; isum++ ) PERM[isum] = isum;

   for( isum = 0; isum < NSUM - 1; isum++ )
   {
      for( jsum = isum + 1; jsum < NSUM; jsum++ )
      {
         if( CSUM[PERM[isum]] > CSUM[PERM[jsum]] )
         { iprm = PERM[isum]; PERM[isum] = PERM[jsum]; PERM[jsum] = iprm; }
      }
   }

   return( APL_SUCCESS );
}

static int                               APL_TprdNew
(
   int                            PSIZ,
   int                            PTHR,
   int                            PRNK,
   const char *                   FNAM,
   const ssize_t                  RDBS,
   size_t *                       NROW,
   size_t *                       NCOL,
   size_t *                       NNNZ,
   APL_Tprd_t *                   TPRD
)
{
   int                            fdes, ithr, jthr, lerr, pthr;
   off_t                          fsiz, lprt, offb, offe;
   size_t                         ioff[10], ltpr, fnsz;
   void * *                       args = NULL;
   APL_TprdArgs_p *               sarg;
   APL_Tprd_t                     tprd = NULL;

   if( FNAM == NULL ) return( APL_EINVAL );

   fnsz = strlen( FNAM );
   if( fnsz > 255 ) { return( APL_EINVAL ); }

   lerr = APL_TprdOpen( FNAM, PRNK, 1,
                        PRNK, PSIZ, &fdes,
                        &fsiz, &offb, &offe,
                        NROW, NCOL, NNNZ );

   #ifdef _DEBUG
   if( NROW != NULL )
   {
       (void) fprintf( stdout,
                       "[%2d, *] nrow = %12ld, ncol = %12ld, nnnz = %12ld\n",
                       PRNK, *NROW, *NCOL, *NNNZ );
   }
   if( lerr == APL_SUCCESS )
   {
       (void) fprintf( stdout,
                   "[%2d, *] offb = %12ld, fsiz = %12ld, offe = %12ld\n",
                   PRNK, offb, fsiz, offe );
   }
   #endif

   if( lerr == APL_SUCCESS ) lerr = APL_TprdSizeOf( PTHR, RDBS, &ltpr, ioff );

   if( lerr == APL_SUCCESS )
   {
      tprd = (APL_Tprd_t) malloc( ltpr );
      lerr = ( tprd ? APL_SUCCESS : APL_ENOMEM );
   }

   if( lerr == APL_SUCCESS )
   {
      pthr = PTHR;

      APL_TPRD_psiz( tprd ) = PSIZ;
      APL_TPRD_pthr( tprd ) = PTHR;
      APL_TPRD_nsiz( tprd ) = 1;

      APL_TPRD_prnk( tprd ) = PRNK;
      APL_TPRD_prvp( tprd ) = PRNK;                       /* NSIZ = 1 */

      APL_TPRD_fsiz( tprd ) = fsiz;
      APL_TPRD_offb( tprd ) = offb; APL_TPRD_offe( tprd ) = offe;
      APL_TPRD_nedg( tprd ) = 0;    APL_TPRD_rdbs( tprd ) = RDBS;

      APL_TPRD_buff( tprd ) = (char           *)((char *)(tprd) + ioff[0]);
      APL_TPRD_toff( tprd ) = (off_t          *)((char *)(tprd) + ioff[1]);
      APL_TPRD_tlen( tprd ) = (ssize_t        *)((char *)(tprd) + ioff[2]);
      APL_TPRD_csum( tprd ) = (size_t         *)((char *)(tprd) + ioff[3]);
      APL_TPRD_cprm( tprd ) = (int            *)((char *)(tprd) + ioff[4]);
      APL_TPRD_tedg( tprd ) = (size_t         *)((char *)(tprd) + ioff[5]);
      APL_TPRD_aedg( tprd ) = (size_t         *)((char *)(tprd) + ioff[6]);
      APL_TPRD_args( tprd ) = args = (void  * *)((char *)(tprd) + ioff[7]);
      sarg                  = (APL_TprdArgs_p *)((char *)(tprd) + ioff[8]);
      APL_TPRD_thrd( tprd ) = (pthread_t      *)((char *)(tprd) + ioff[9]);

      (void) APL_TsklInit( tprd );

      for( ithr = 0; ithr < pthr; ithr++ )
      {
         APL_TprdARGS_ithr( &(sarg[ithr]) ) = ithr;
         APL_TprdARGS_fdes( &(sarg[ithr]) ) = ( ithr == 0 ? fdes : -1 );
                                     /* Initially no edges in buffers */
         APL_TprdARGS_nedg( &(sarg[ithr]) ) = 0;
         APL_TprdARGS_bcur( &(sarg[ithr]) ) = (char *)(FNAM);
         APL_TprdARGS_tprd( &(sarg[ithr]) ) = tprd;

         args[ithr] = (void *)(&(sarg[ithr]));
      }
                                  /* Each thread opens opens the file */
      lerr = APL_InstTsklSpawn( tprd, APL_TprdOpenThread, args );

      if( lerr == APL_SUCCESS )
      {
         for( ithr = 0; ithr < pthr; ithr++ )
         { APL_TprdARGS_bcur( args[ithr] ) = NULL; }
                                                    /* Thread offsets */
         lprt = offe - offb + 1;
         lprt = APL_CEIL( lprt, (off_t)(pthr) );         /* smooth it */
         if( lprt > APL_TPIO_4K )
            lprt = APL_CEIL( lprt, APL_TPIO_4K ) * (off_t)(APL_TPIO_4K);

         (void) APL_TprdOffset( fdes, offb, offe, lprt, pthr,
                                APL_TPRD_toff( tprd ), APL_TPRD_tlen( tprd ) );

         for( jthr = 0; jthr < pthr; jthr++ )
         {
            APL_TPRD_csum( tprd )[jthr] = 0;
            APL_TPRD_tedg( tprd )[jthr] = 0;

            for( ithr = 0; ithr < pthr; ithr++ )
               APL_TPRD_aedg( tprd )[jthr * pthr + ithr] = 0;
         }

         (void) APL_TprdCsumSort( pthr, APL_TPRD_csum( tprd ),
                                  APL_TPRD_cprm( tprd ) );

         (void) memcpy( APL_TPRD_fnam( tprd ), FNAM, fnsz + 1 );
#if 0
         for( ithr = 0; ithr < pthr; ithr++ )
         {
            (void) fprintf( stdout,
               "[%2d,%2d] toff = %12ld, tlen = %12ld, tend = %12ld\n",
               APL_TPRD_prnk( TPRD ), ithr, APL_TPRD_toff( tprd )[ithr],
               APL_TPRD_tlen( tprd )[ithr], APL_TPRD_toff( tprd )[ithr] +
               APL_TPRD_tlen( tprd )[ithr] - 1 );
         }
#endif
      }

      if( lerr == APL_SUCCESS ) *TPRD = tprd;
      else (void) APL_TprdFree( &tprd );
   }

   return( lerr );
}

int                               TprdCopy
(
   const void * const             TSRC,
   void * * const                 TDST
) {
   int                            lerr, ithr;
   void                           * dest, * * args = NULL;
   char                           * bsrc, * bdst;

   /*printf( "DBG: TprdCopy called. Copying %p to ", TSRC );*/

   lerr = APL_TprdNew(
      APL_TPRD_psiz( TSRC ), APL_TPRD_pthr( TSRC ), APL_TPRD_prnk( TSRC ),
      APL_TPRD_fnam( TSRC ), APL_TPRD_rdbs( TSRC ),
      NULL, NULL, NULL,
      &dest
   );

   if( lerr == APL_SUCCESS ) {
      /* copy state of the reader */
      APL_TPRD_nedg( dest ) = APL_TPRD_nedg( TSRC );
      (void) memcpy( APL_TPRD_buff( dest ), APL_TPRD_buff( TSRC ), APL_TPRD_pthr( TSRC ) * APL_TPRD_rdbs( TSRC ) );
      (void) memcpy( APL_TPRD_toff( dest ), APL_TPRD_toff( TSRC ), APL_TPRD_pthr( TSRC ) * sizeof(off_t) );
      (void) memcpy( APL_TPRD_tlen( dest ), APL_TPRD_tlen( TSRC ), APL_TPRD_pthr( TSRC ) * sizeof(ssize_t) );
      (void) memcpy( APL_TPRD_tedg( dest ), APL_TPRD_tedg( TSRC ), APL_TPRD_pthr( TSRC ) * sizeof(size_t) );

      /* now do copy of APL_TprdArgs */
      args = APL_TPRD_args( dest );
      bsrc = APL_TPRD_buff( TSRC );
      bdst = APL_TPRD_buff( dest );
      for( ithr = 0; ithr < APL_TPRD_pthr( dest ); ithr++ )
      {
         APL_TprdARGS_nedg( args[ithr] ) = APL_TprdARGS_nedg( APL_TPRD_args( TSRC )[ithr] );
         APL_TprdARGS_bcur( args[ithr] ) = bdst +
            ( APL_TprdARGS_bcur( APL_TPRD_args( TSRC ) ) - bsrc );

         bsrc += (size_t)(ithr) * APL_TPRD_rdbs( TSRC );
         bdst += (size_t)(ithr) * APL_TPRD_rdbs( dest );
      }
      *TDST = dest;
   }

   /*printf( "%p\n.", *TDST ); fflush(stdout); DBG*/

   return lerr;
}

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
)
{
   int                            ierr;

   /*printf( "DBG: ReadEdgeBegin called with %s, %zd, %d, %d, %d, %p, %p, %p, %p\n", FNAM, RDBS, PSIZ, PTHR, PRNK, NROW, NCOL, NNNZ, TPRD );*/

   /*printf( "ReadEdgeBegin called using %d threads.\n", PTHR );*/

   ierr = APL_TprdNew( PSIZ, PTHR, PRNK, FNAM, RDBS, NROW, NCOL, NNNZ, TPRD );

   return( ierr );
}

static int                               APL_TprdReadThread
(
   void *                         DATA
)
{
   int                            fdes, ithr, kthr, lerr = APL_SUCCESS;
   size_t                         icol, iedg, ipos, irow, nedg;
   ssize_t                        lrea, twrd;
   off_t                          offs;
   char *                         bcur, * bend, * buff;
   APL_Tprd_t                     tprd;

   ithr = APL_TprdARGS_ithr( DATA ); tprd = APL_TprdARGS_tprd( DATA );

   if( APL_TPRD_nedg( tprd ) == 0 )
   {
      if( ( twrd = APL_TPRD_tlen( tprd )[ithr] ) >  0 )
      {
         fdes = APL_TprdARGS_fdes( DATA );
         offs = APL_TPRD_toff( tprd )[ithr];

         twrd -= ( lrea = APL_MIN( twrd, APL_TPRD_rdbs( tprd ) ) );

         buff = APL_TPRD_buff( tprd ) + (size_t)(ithr) * APL_TPRD_rdbs( tprd );
         bend = ( bcur = buff ) + lrea - 1;

         APL_TPIO_PREAD( fdes, bcur, lrea, offs, lerr );
         APL_TPIO_BOLN_ADJUST( twrd, bend, offs );

         bcur = buff; nedg = 0; APL_TPIO_BOLN_COUNT(  bcur, bend, nedg );

         lrea = bend - buff + 1;
         APL_TPRD_tlen( tprd )[ithr] -= lrea;
         APL_TPRD_toff( tprd )[ithr] += lrea;

         APL_TprdARGS_bcur( DATA ) = buff;
         APL_TprdARGS_nedg( DATA ) = nedg;
      }
      else
      {
         APL_TprdARGS_nedg( DATA ) = 0;
      }

      return( lerr );
   }
/*
 * Fill in user arrays according to tedg. Re-start from where we left things
 */
   nedg = APL_TPRD_tedg( tprd )[ithr];

   for( ipos = 0, kthr = 0; kthr < ithr; kthr++ )
      ipos += APL_TPRD_tedg( tprd )[kthr];

   bcur = APL_TprdARGS_bcur( DATA );
                                /* nedg is the number of edges I want */
   for( iedg = 0; iedg < nedg; iedg++ )
   {
      irow = 0;
      do { irow *= 10; irow += *bcur - APL_CHAR_0; bcur++; }
      while( ( *bcur >= APL_CHAR_0 ) && ( *bcur <= APL_CHAR_9 ) );

      while( isspace( *bcur ) ) bcur++;

      icol = 0;
      do { icol *= 10; icol += *bcur - APL_CHAR_0; bcur++; }
      while( ( *bcur >= APL_CHAR_0 ) && ( *bcur <= APL_CHAR_9 ) );

      while( *bcur != APL_CHAR_EOL ) bcur++;

      bcur++;

      APL_TprdARGS_irow( DATA )[ipos] = irow;
      APL_TprdARGS_icol( DATA )[ipos] = icol;

      ipos++;
   }

   APL_TprdARGS_bcur( DATA )  = bcur;
   APL_TprdARGS_nedg( DATA ) -= nedg;

   return( lerr );
}

#ifdef VERBOSE
static int                               APL_TprdAedgShow
(
   APL_Tprd_t                     TPRD
)
{
   int                            ithr, jthr, pthr;
   size_t                         asum, rsum;
   size_t *                       aedg;

   aedg = APL_TPRD_aedg( TPRD );
   pthr = APL_TPRD_pthr( TPRD );

   for( asum = 0, ithr = 0; ithr < pthr; ithr++ )
   {
      (void) fprintf( stdout, "[%2d,%2d] ",
               APL_TPRD_prnk( TPRD ), ithr );

      for( rsum = 0, jthr = 0; jthr < pthr; jthr++ )
      {
         rsum += aedg[ithr*pthr + jthr];
         (void) fprintf( stdout, "%12ld ", aedg[ithr*pthr + jthr] );
      }

      asum += rsum;
      (void) fprintf( stdout, "(%12ld) %12ld\n", rsum, APL_TPRD_tedg( TPRD )[ithr] );
   }

   (void) fprintf( stdout, "[%2d, *] ", APL_TPRD_prnk( TPRD ) );

   for( jthr = 0; jthr < pthr; jthr++ )
   { (void) fprintf( stdout, "%12ld ", APL_TPRD_csum( TPRD )[jthr] ); }
   (void) fprintf( stdout, "(%12ld)\n", asum );

   return( APL_SUCCESS );
}
#endif

static int                               APL_TprdAedgInit
(
   APL_Tprd_t                     TPRD
)
{
   int                            ithr, jthr, pthr;
   size_t                         csum, iedg, naup, toff;
   void * *                       args;
   size_t *                       aedg;

   args = APL_TPRD_args( TPRD );
   pthr = APL_TPRD_pthr( TPRD );

   aedg = APL_TPRD_aedg( TPRD );

   for( APL_TPRD_nedg( TPRD ) = 0, ithr = 0; ithr < pthr; ithr++ )
   {
      APL_TPRD_nedg( TPRD ) += APL_TprdARGS_nedg( args[ithr] );

      for( jthr = 0; jthr < pthr; jthr++ ) aedg[ithr*pthr + jthr] = 0;
   }

   for( ithr = 0; ithr < pthr; ithr++ )
   {
      for( naup = 0, toff = 0, jthr = 0; jthr < pthr; jthr++ )
      {
         if( jthr < ithr ) toff += APL_TprdARGS_nedg( args[jthr] );

         naup += APL_TprdARGS_nedg( args[jthr] );
      }

      naup = APL_CEIL( naup, ((size_t)(pthr)) );

#if 0
      for( iedg = 0; iedg < APL_TprdARGS_nedg( args[ithr] ); iedg++ )
      {
         jthr = ( toff + iedg ) / naup;
         aedg[ithr*pthr + jthr]++;
      }
#else
      size_t inxt, kedg, nedg;

      if( ( nedg = APL_TprdARGS_nedg( args[ithr] ) ) > 0 )
      {
         jthr = ( iedg = toff ) / naup;

         while( nedg > 0 )
         {
            inxt = ( jthr + 1 ) * naup;
            if( iedg + nedg < inxt ) inxt = iedg + nedg;

            aedg[ithr*pthr + jthr] += ( kedg = inxt - iedg );
            iedg = inxt;
            jthr++;
            nedg -= kedg;
         }
      }
#endif
   }

   for( jthr = 0; jthr < pthr; jthr++ )
   {
      for( csum = 0, ithr = 0; ithr < pthr; ithr++ )
         csum += aedg[ithr*pthr + jthr];
      APL_TPRD_csum( TPRD )[jthr] = csum;
   }

   (void) APL_TprdCsumSort( pthr, APL_TPRD_csum( TPRD ),
                            APL_TPRD_cprm( TPRD ) );

   return( APL_SUCCESS );
}

static int                               APL_TprdAedgUpdate
(
   APL_Tprd_t                     TPRD,
   const size_t                   NEDG
)
{
   int                            ithr, jthr, kthr, lthr, pthr;
   size_t                         csum, nadn, naup, nfdn, nfup, nhav,
                                  ntak, nwnt;
   size_t *                       aedg, * tedg;

   pthr = APL_TPRD_pthr( TPRD );
   nhav = APL_TPRD_nedg( TPRD );
   aedg = APL_TPRD_aedg( TPRD ); tedg = APL_TPRD_tedg( TPRD );

   for( ithr = 0; ithr < pthr; ithr++ ) tedg[ithr] = 0;
/*
 * Compute the number of edges every thread will contribute
 */
   nwnt = APL_MIN( NEDG, nhav );

#ifdef VERBOSE
   (void) fprintf( stdout, "nwnt = %ld, nhav = %12ld\n", nwnt, nhav );
#endif

   if( nhav > nwnt )
   {
      nadn = NEDG / (size_t)(pthr);
      naup = APL_CEIL( NEDG, ((size_t)(pthr)) );

      nfup = NEDG - nadn * (size_t)(pthr);
      nfdn = (size_t)(pthr) - nfup;

      for( lthr = 0; lthr < pthr; lthr++ )
      {                            /* process columns in sorted order */
         jthr = APL_TPRD_cprm( TPRD )[lthr];

         csum = APL_TPRD_csum( TPRD )[jthr];

         if( nfdn > 0 ) { ntak = nadn; nfdn--; }
         else           { ntak = naup; nfup--; }

         for( kthr = 0; kthr < pthr; kthr++ )
         {
            if( ntak == 0 ) break;

            ithr = ( jthr + kthr ) % pthr;

            if( ntak >= aedg[ithr*pthr + jthr] )
            {
               ntak       -= aedg[ithr*pthr + jthr];
               tedg[ithr] += aedg[ithr*pthr + jthr];
               aedg[ithr*pthr + jthr] = 0;
            }
            else if( aedg[ithr*pthr + jthr] > 0 )
            {
               aedg[ithr*pthr + jthr] -= ntak;
               tedg[ithr]             += ntak;
               ntak = 0;
            }
         }

         for( csum = 0, ithr = 0; ithr < pthr; ithr++ )
         { csum += aedg[ithr*pthr + jthr]; }

         APL_TPRD_csum( TPRD )[jthr] = csum;
      }
   }
   else
   {                                                 /* take them all */
      for( jthr = 0; jthr < pthr; jthr++ )
      {
         APL_TPRD_csum( TPRD )[jthr] = 0;

         for( ithr = 0; ithr < pthr; ithr++ )
         {
            tedg[ithr] += aedg[ithr*pthr + jthr];
            aedg[ithr*pthr + jthr] = 0;
         }
      }
   }

   (void) APL_TprdCsumSort( pthr, APL_TPRD_csum( TPRD ),
                            APL_TPRD_cprm( TPRD ) );

   return( APL_SUCCESS );
}

int                               ReadEdge
(
   void *                         TPRD,
   size_t *                       NEDG,               /* input output */
   size_t *                       IROW,
   size_t *                       ICOL
)
{
   int                            ithr, lerr = APL_SUCCESS, pthr;
   size_t                         nedg;
   void * *                       args;

   /*printf( "DBG: ReadEdge called with %p, %p, %p, %p\n", TPRD, NEDG, IROW, ICOL );*/

   if( *NEDG == 0 ) return( APL_SUCCESS );          /* 0 size buffers */

   args = APL_TPRD_args( TPRD );
   pthr = APL_TPRD_pthr( TPRD );

   if( APL_TPRD_nedg( TPRD ) == 0 )            /* Fill in the buffers */
   {
#ifdef VERBOSE
      for( ithr = 0; ithr < pthr; ithr++ )
      {                                   /* Show the current offsets */
         (void) fprintf( stdout,
            "[%2d,%2d] toff = %12ld, tlen = %12ld, tend = %12ld\n",
            APL_TPRD_prnk( TPRD ), ithr, APL_TPRD_toff( TPRD )[ithr],
            APL_TPRD_tlen( TPRD )[ithr], APL_TPRD_toff( TPRD )[ithr] +
            APL_TPRD_tlen( TPRD )[ithr] - 1 );
      }
#endif

      lerr = APL_InstTsklSpawn( TPRD, APL_TprdReadThread, args );

      if( lerr == APL_SUCCESS )
      {
         lerr = APL_TprdAedgInit( TPRD );

         for( nedg = 0, ithr = 0; ithr < pthr; ithr++ )
         { nedg += APL_TprdARGS_nedg( args[ithr] ); }

         APL_TPRD_nedg( TPRD ) = nedg;
      }

#ifdef VERBOSE
      (void) APL_TprdAedgShow( TPRD );  /* Show the current situation */
#endif
   }
/*
 * Fill in the user arrays
 */
   if( lerr == APL_SUCCESS )
   {
      lerr = APL_TprdAedgUpdate( TPRD, *NEDG );

#ifdef VERBOSE
      (void) APL_TprdAedgShow( TPRD ); /* Show the upcoming situation */
#endif
   }

   if( lerr == APL_SUCCESS )
   {
      for( ithr = 0; ithr < pthr; ithr++ )
      {
         APL_TprdARGS_irow( args[ithr] ) = IROW;
         APL_TprdARGS_icol( args[ithr] ) = ICOL;
      }

      lerr = APL_InstTsklSpawn( TPRD, APL_TprdReadThread, args );

      if( lerr == APL_SUCCESS )
      {                 /* Number of edges to be copied in user space */
         for( nedg = 0, ithr = 0; ithr < pthr; ithr++ )
         { nedg += APL_TprdARGS_nedg( args[ithr] ); }

         *NEDG = APL_TPRD_nedg( TPRD ) - nedg;
                /* update the global number of edges still in buffers */
         APL_TPRD_nedg( TPRD ) = nedg;
      }
   }

#ifdef VERBOSE
   (void) fprintf( stdout, "[%2d, *] nedg = %12ld  on exit of ReadEdge ...\n\n",
            APL_TPRD_prnk( TPRD ), *NEDG );
#endif

   return( lerr );
}

int                               ReadEdgeEnd
(
   void *                         TPRD
)
{
   (void) APL_TprdFree( &TPRD );

   return( APL_SUCCESS );
}


#ifdef TEST_HPPARSER

int                               main( int ARGC, char * * ARGV )
{
   int                            lerr, prnk, psiz, pthr;
   ssize_t                        rdbs;
   size_t                         eblk, nedg, ncol, nnnz, nrow, ntot;
   size_t *                       icol, * irow;
   void *                         hdle = NULL;

   if( ARGC != 7 )
   {
      (void) fprintf( stderr,
                      "Usage: %s <psiz> <nthr> <rdbs> <eblk> <filename> <dimh>\n",
                      ARGV[0] );
      (void) fprintf( stderr, "    <psize> number of processes\n" );
      (void) fprintf( stderr, "    <nthr> number of threads\n" );
      (void) fprintf( stderr, "    <rdbs> read block size (in bytes)\n" );
      (void) fprintf( stderr, "    <eblk> buffer size (in bytes)\n" );
      (void) fprintf( stderr, "    <filename> input file name\n" );
      (void) fprintf( stderr, "    <dimh> 0 iff there is no dimension header line in <filename>\n" );
      exit( EXIT_FAILURE );
   }

   psiz = atoi( ARGV[1] );               /* total number of processes */
   pthr = atoi( ARGV[2] );               /* how many threads per process */
   rdbs = (ssize_t)(atoi( ARGV[3] ));    /* read block size per thread */
   eblk = (ssize_t)(atoi( ARGV[4] ));    /* read block size per thread */

   irow = (size_t *) malloc( (size_t)(2) * eblk * sizeof( size_t ) );

   if( ( lerr = ( irow ? APL_SUCCESS : APL_ENOMEM ) ) == APL_SUCCESS )
   {
      icol = irow + eblk;

      for( ntot = 0, prnk = 0; ( lerr == APL_SUCCESS ) && ( prnk < psiz );
           prnk++ )
      {
         if( lerr == APL_SUCCESS )
         {
            if( ARGV[6][0] == '0' ) {
                lerr = ReadEdgeBegin( ARGV[5], rdbs, psiz, pthr, prnk,
                                      NULL, NULL, NULL, &hdle );
            } else {
                lerr = ReadEdgeBegin( ARGV[5], rdbs, psiz, pthr, prnk,
                                      &nrow, &ncol, &nnnz, &hdle );
            }
         }

         if( lerr == APL_SUCCESS )
         {
            do
            {
               nedg = eblk;
               lerr = ReadEdge( hdle, &nedg, irow, icol );
#if 0
               size_t iedg;

               for( iedg = 0; iedg < nedg; iedg++ )
               { (void) fprintf( stdout, "%ld %ld\n", irow[iedg], icol[iedg] ); }
#endif
               ntot += nedg;
            }
            while( ( lerr == APL_SUCCESS ) && ( nedg > 0 ) );
         }

         (void) ReadEdgeEnd( hdle );
      }

      (void) fprintf( stdout, "[ *, *] ntot = %12ld\n", ntot );

      free( irow );
   }

   exit( 0 );
   return( 0 );
}

#endif /* end ifdef ``TEST_HPPARSER'' */

