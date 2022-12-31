
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
 * Specifies the #grb::Launcher functionalities.
 *
 * @author A. N. Yzelman
 * @date 17th of April, 2017
 */

#ifndef _H_GRB_EXEC_BASE
#define _H_GRB_EXEC_BASE

#include <stdexcept>
#include <string>

#include <graphblas/backends.hpp>
#include <graphblas/rc.hpp>

#ifndef _GRB_NO_STDIO
 #include <iostream>
#endif


namespace grb {

	/**
	 * The various ways in which the #Launcher can be used
	 * to execute a GraphBLAS program.
	 *
	 * \warning An implementation may require different linker commands
	 *          when using different modes. This is OK, since a call to
	 *          the #Launcher is required to be quite different
	 *          depending on which mode is used. The portability is in
	 *          the GraphBLAS program being launched-- that one should
	 *          never change depending on whichever mode it is used.
	 */
	enum EXEC_MODE {

		/**
		 * Automatic mode. The #Launcher can spawn user processes
		 * which will execute a given program.
		 */
		AUTOMATIC = 0,

		/**
		 * Manual mode. The user controls \a nprocs user processes
		 * which together should execute a given program, by, for
		 * example, using the #Launcher.
		 */
		MANUAL,

		/**
		 * When running from an MPI program. The user controls
		 * \a nprocs MPI programs, which, together, should execute
		 * a given GraphBLAS program.
		 */
		FROM_MPI

	};

	/**
	 * Allows an auxiliary program to run any GraphBLAS program. Input data may be
	 * passed through a user-defined type. Output data will be retrieved via the
	 * same type. For implementations that support multiple user processes, the
	 * caller may explicitly set the process ID and total number of user processes.
	 *
	 * The intended use is to `just call' grb::exec which should, in its most
	 * trivial form, compile regardless of which backend is selected.
	 *
	 * @tparam mode           Which #EXEC_MODE the Launcher should adhere to.
	 * @tparam implementation Which GraphBLAS implementation is to be used.
	 */
	template< enum EXEC_MODE mode, enum Backend implementation >
	class Launcher {

		public :

			/**
		     * Constructs a new Launcher. This constructor is a collective
		     * call; all \a nprocs processes that form a single Launcher
		     * group must make a call to this constructor at roughly the
		     * same time. There is an implementation-defined time-out for
		     * the creation of a Launcher group.
		     *
		     * @param[in]  process_id  The user process ID of the calling process.
		     *                         The value must be larger or equal to 0. This
		     *                         value must be strictly smaller than \a nprocs.
		     *                         This value must be unique to the calling
		     *                         process within this collective call across
		     *                         \em all \a nprocs user processes. This number
		     *                         \em must be strictly smaller than \a nprocs.
		     *                         Optional: the default is 0.
		     * @param[in]  nprocs      The total number of user processes making a
		     *                         collective call to this function. Optional: the
		     *                         default is 1.
		     * @param[in]  hostname    The hostname of one of the user processes.
		     *                         Optional: the default is `localhost'.
		     * @param[in]  port        A free port number at \a hostname. This port
		     *                         will be used for TCP connections to \a hostname
		     *                         if and only if \a nprocs is larger than one.
		     *                         Optional: the default value is `0'.
		     *
		     * @throws invalid_argument If #nprocs is zero.
		     * @throws invalid_argument If #process_id is greater than or
		     *                          equal to \a nprocs.
		     *
		     * \note An implementation may define further constraints on
		     *       the input arguments, such as, obviously, on \a hostname
		     *       and \a port, but also on \a nprocs and, as a result, on
		     *       \a process_id.
		     */
			Launcher( const size_t process_id = 0,        // user process ID
				const size_t nprocs = 1,                  // total number of user processes
				const std::string hostname = "localhost", // one of the user process hostnames
				const std::string port = "0"              // a free port at hostname
			) {                                           // standard does not specify any constrants on hostname and port
		                                                  // so accept (and ignore) anything
				(void)hostname; (void)port;

#ifndef _GRB_NO_EXCEPTIONS
				// sanity checks on process_id and nprocs
				if( nprocs == 0 ) { throw std::invalid_argument( "Total number of user "
																 "processes must be "
																 "strictly larger than "
																 "zero." ); }
	if( process_id >= nprocs ) {
		throw std::invalid_argument( "Process ID must be strictly smaller than "
									 "total number of user processes." );
	}
#endif
} // namespace grb

/**
 * Executes the given GraphBLAS program. This function, depending on whether
 * GraphBLAS is compiled in automatic or in manual mode, will either
 * \em spawn the maximum number of available user processes or will connect
 * exactly \a nprocs existing processes, respectively, to execute the given
 * \a grb_program.
 *
 * This is a collective function call.
 *
 * @tparam T The type of the data to pass to the GraphBLAS program.
 * @tparam U The type of the output data to pass back to the user.
 *
 * @param[in]  grb_program User GraphBLAS program to be executed.
 * @param[in]  data_in     Input data of user-defined type \a T.
 *                         When in automatic mode, the data will only be
 *                         available at user process 0 only. When in
 *                         manual mode, the data will be available to
 *                         this user process (with the below given
 *                         \a process_id) only.
 * @param[out] data_out    Output data of user-defined type \a U. The output
 *                         data should be available at user process with ID
 *                         zero.
 * @param[in]  broadcast   Whether the input should be broadcast from user
 *                         process 0 to all other user processes. Optional;
 *                         the default value is \a false.
 *
 * @return SUCCESS If the execution proceeded as intended.
 * @return PANIC   If an unrecoverable error was encountered while trying to
 *                 execute the given GraphBLAS program.
 *
 * \warning An implementation can define further constraints on the validity
 *          of input arguments. The most obvious is that implementations
 *          supporting only one user process will not accept \a nprocs larger
 *          than 1.
 *
 * All aforementioned default values shall always be legal.
 */
template< typename T, typename U >
RC exec( void ( *grb_program )( const T &, U & ), // user GraphBLAS program
	const T & data_in,
	U & data_out, // input & output data
	const bool broadcast = false ) const {
	(void)grb_program;
	(void)data_in;
	(void)data_out;
	(void)broadcast;
	// stub implementation, should be overridden by specialised implementation,
	// so return error code
	return PANIC;
}

/**
 * Variable size version of the above function.
 *
 * @param[in]  broadcast   Whether the input should be broadcast from user
 *                         process 0 to all other user processes. Optional;
 *                         the default value is \a false. This will let user
 *                         processes with ID larger than zero allocate
 *                         \a in_size bytes of memory into which the data at
 *                         process 0 will be copied.
 *
 * \todo more documentation
 */
template< typename U >
RC exec( void ( *grb_program )( const void *, const size_t, U & ), const void * data_in, const size_t in_size, U & data_out, const bool broadcast = false ) const {
	(void)grb_program;
	(void)data_in;
	(void)in_size;
	(void)data_out;
	(void)broadcast;
	return PANIC;
}

			/**
			 * Releases all ALP resources.
			 *
			 * After a call to this function, no further ALP programs may be benchmarked
			 * nor launched-- i.e., both the #grb::Launcher and #grb::Benchmarker
			 * functionalities many no longer be used.
			 *
			 * A well-behaving program calls this function, or #grb::Launcher::finalize,
			 * exactly once and just before exiting (or just before the guaranteed last
			 * invocation of an ALP program).
			 *
			 * @return SUCCESS The resources have successfully and permanently been
			 *                 released.
			 * @return PANIC   An unrecoverable error has been encountered and the user
			 *                 program is encouraged to exit as quickly as possible. The
			 *                 state of the ALP library has become undefined and should
			 *                 no longer be used.
			 *
			 * \internal This is the base implementation that should be specialised by
			 *           each backend separately.
			 */
			static RC finalize() {
				return PANIC;
			}

	}; // end class `Launcher'

} // end namespace ``grb''

#endif // end _H_GRB_EXEC_BASE

