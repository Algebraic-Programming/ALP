
#ifndef _H_BSP1D_EXEC_BROADCAST_ROUTINES
#define _H_BSP1D_EXEC_BROADCAST_ROUTINES

#include <stddef.h>

#include <lpf/collectives.h>
#include <lpf/core.h>

/**
 * @file exec_broadcast_routines.hpp
 * Routines used in the Launcher's for broadcasting data.
 */

namespace grb {
	namespace internal {

		/** Global internal singleton to track whether MPI was initialized. */
		extern bool grb_mpi_initialized;

		/** Initialize collective communication for broadcast. */
		lpf_err_t lpf_init_collectives_for_bradocast( lpf_t &, lpf_coll_t &, lpf_pid_t, lpf_pid_t, size_t );

		/** Register a memory area as a global one and do the broadcast */
		lpf_err_t lpf_register_and_broadcast( lpf_t &, lpf_coll_t &, void *, size_t );


	} // end internal
} // end grb


#endif // _H_BSP1D_EXEC_BROADCAST_ROUTINES
