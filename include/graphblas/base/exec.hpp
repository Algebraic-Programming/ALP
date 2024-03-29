
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
	 * Type definition for an ALP function with input type information.
	 */
	template< typename InputType, typename OutputType >
	using AlpTypedFunc = void ( * )( const InputType &, OutputType & );

	/**
	 * Type definition for an ALP function without input type information.
	 */
	template< typename OutputType >
	using AlpUntypedFunc = void ( * )( const void *, size_t, OutputType & );

	/**
	 * The various ways in which the #grb::Launcher can be used to execute an
	 * ALP program.
	 *
	 * \warning An implementation or backend may require different linker commands
	 *          when using different modes, and may require different arguments be
	 *          passed on program launch. Please see the compiler and runner
	 *          wrappers <tt>grbcxx</tt>, <tt>alpcxx</tt>, <tt>grbrun</tt>, and/or
	 *          <tt>alprun</tt> for more details; or refer to the implementation
	 *          documentation.
	 *
	 * \warning Depending on the mode given to #grb::Launcher, different parameters
	 *          to the exec function may be required.
	 *
	 * An ALP program remains unaware of which mode the launcher employs. Normally,
	 * it requires no change depending on how it is launched. An exception is when
	 * data is passed through and from the caller program:
	 *  -# if the launch mode is #AUTOMATIC, best practice is to minimise the input
	 *     data footprint that requires broadcasting to all user processes
	 *     executing the algorithm; in the base case, no input data requires
	 *     broadcasting. Output is retained only from the first user process, i.e.,
	 *     the user process for which #grb::spmd<>::pid() returns zero.
	 *  -# for any other launch mode, multiple user processes may exist before any
	 *     ALP or ALP/GraphBLAS context exists. Each pre-existing process in such
	 *     external context is then mapped to an ALP user process in a one-to-one
	 *     manner. Data, including pointer data, may be passed freely between these
	 *     two mapped processes; this may, in principle and contrary to the
	 *     automatic mode, consider large data. Output is retained at each user
	 *     process and thus is freely available to the mapped external process. In
	 *     best practice, different user processes return different parts of the
	 *     overall output, thereby achieving parallel I/O.
	 */
	enum EXEC_MODE {

		/**
		 * Automatic mode.
		 *
		 * The #grb::Launcher may spawn additional user processes which will jointly
		 * execute a given ALP program.
		 */
		AUTOMATIC = 0,

		/**
		 * Manual mode.
		 *
		 * The user controls \a nprocs external processes which jointly should form an
		 * ALP context and execute one or more given ALP programs.
		 */
		MANUAL,

		/**
		 * From MPI mode.
		 *
		 * The user controls \a nprocs external MPI processes which jointly should
		 * form an ALP context and execute one or more given ALP programs. The only
		 * difference with the manual mode is that this mode guarantees that the
		 * pre-existing external processes are MPI processes.
		 */
		FROM_MPI

	};

	/**
	 * A group of user processes that together execute ALP programs.
	 *
	 * Allows an application to run any ALP program. Input data may be passed
	 * through a user-defined type. Output data will be retrieved via another user-
	 * defined type.
	 *
	 * For backends that support multiple user processes, the caller may explicitly
	 * set the process ID and total number of user processes. If the launcher is
	 * requested to spawn new user processes, i.e., if it is constructed using the
	 * #AUTOMATIC mode, then the backend spawns an implementation-defined number of
	 * additional user processes beyond that corresponding to the process
	 * constructing the launcher instance, that then jointly execute ALP programs
	 * in parallel.
	 *
	 * The intended use is to `just call' the exec function, which must be accepted
	 * by any backend in any implementation, to execute any ALP program.
	 *
	 * @tparam mode    Which #EXEC_MODE the Launcher should adhere to.
	 * @tparam backend Which backend to use. This is a hidden template argument that
	 *                 defaults to the backend selected at compile time through
	 *                 <tt>grbcxx</tt> or <tt>alpcxx</tt>.
	 */
	template< enum EXEC_MODE mode, enum Backend backend >
	class Launcher {

		public :

			/**
			 * Constructs a new #grb::Launcher.
			 *
			 * In #AUTOMATIC mode, a single root user processes issues a call to this
			 * constructor. In all other modes, a call to this constructor is
			 * \em collective: all \a nprocs processes that are to form a single launcher
			 * group, must make a simultaneous call to this constructor and must do so
			 * with consistent arguments.
			 *
			 * \note One may note that in all modes, a call to this constructor must be
			 *       collective; it is just that in automatic mode there is but one
			 *       process involved with the collective call (i.e., \a nprocs is one).
			 *
			 * There is an implementation-defined time-out for the creation of a launcher
			 * group. The default arguments to the below are consistent with the
			 * automatic launcher mode.
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
			 * While these arguments are generic and would work with most network
			 * fabrics, some modes such as indeed #FROM_MPI may require other arguments
			 * for constructing a launcher. In terms of specification, only #AUTOMATIC
			 * and #MANUAL are required to implement this specific constructor
			 * signature, including the specified defaults for each argument. All
			 * aforementioned default values must be legal for the #AUTOMATIC and
			 * #MANUAL modes.
			 *
			 * Any other mode in #grb::EXEC_MODE, with possibly different constructor
			 * signatures from those listed here, are both optional and implementation-
			 * specific.
			 *
			 * \note An implementation or backend may define further constraints on the
			 *       input arguments, such as, obviously, on \a hostname and \a port, but
			 *       also on \a nprocs and, as a result, on \a process_id.

			 * \note The most obvious such restriction has backends supporting only one
			 *       user process not accepting \a nprocs larger than 1.
			 *
			 * @throws invalid_argument If \a nprocs is zero.
			 * @throws invalid_argument If \a process_id is greater than or equal to
			 *                          \a nprocs.
			 *
			 * @throws std::invalid_argument If \a nprocs is zero.
			 * @throws std::invalid_argument If \a process_id is larger than or equal to
			 *                               \a nprocs.
			 */
			Launcher(
				const size_t process_id = 0,
				const size_t nprocs = 1,
				const std::string hostname = "localhost",
				const std::string port = "0"
			) {
				// spec does not specify any constrants on hostname and port
				// so accept (and ignore) anything
				(void) hostname;
				(void) port;

#ifndef _GRB_NO_EXCEPTIONS
				// sanity checks on process_id and nprocs
				if( nprocs == 0 ) {
					throw std::invalid_argument( "Total number of user processes must be "
						"strictly larger than zero." );
				}
				if( process_id >= nprocs ) {
					throw std::invalid_argument( "Process ID must be strictly smaller than "
						"total number of user processes." );
				}
#endif
			}

			/**
			 * Executes a given ALP program using the user processes encapsulated by this
			 * launcher group.
			 *
			 * Calling this function, depending on whether the automatic, manual, or from
			 * MPI mode was selected, will either:
			 *  -# use processes spawned by the ALP implemenation and use those, as well
			 *     as the process which had constructed this launcher instance,
			 *     to jointly execute the given \a alp_program, \em or
			 *  -# employ the given processes that are managed by the user application
			 *     and used to construct this launcher instance to execute the given
			 *     \a alp_program.
			 *
			 * This is a collective function call-- all processes in the launcher group
			 * must make a simultaneous call to this function and must do so using
			 * consistent arguments.
			 *
			 * @tparam T The type of the data to pass to the ALP program as input. This
			 *           must be a POD type that contains no pointers.
			 *
			 * \note In fact, \a T may be standard layout and contain no pointers, or it
			 *       may be trivially copiable and contain no pointers.
			 *
			 * For calls with \a broadcast <tt>false</tt>, \a T must furthermore be
			 * default-constructible (and have meaningful default values that allow for
			 * successful multi-process execution).
			 *
			 * For programs or entry points that are solely to be called from manual or
			 * from MPI modes with \a broadcast <tt>false</tt>, there are no constraints
			 * on the type \a T since instances of \a T are only ever passed within the
			 * pre-existing user process, and never communicated across user processes.
			 *
			 * @tparam U The type of the output data to pass back to the caller. This may
			 *           be of any type.
			 *
			 * When \a mode is #AUTOMATIC, the type \a U must be default-constructible.
			 *
			 * @param[in]  alp_program The user program to be executed.
			 * @param[in]  data_in     Input data of user-defined type \a T.
			 * @param[out] data_out    Output data of user-defined type \a U.
			 * @param[in]  broadcast   Whether the input should be broadcast from user
			 *                         process 0 to all other user processes. Optional;
			 *                         the default value is <tt>false</tt>.
			 *
			 * When in automatic mode and \a broadcast is <tt>false</tt>, the input data
			 * \a data_in will only be available at user process with ID 0-- any other
			 * user processes will receive a default-constructed \a data_in instead.
			 * When in automatic mode and \a broadcast is <tt>true</tt>, the input data
			 * \a data_in will be available at all user processes instead.
			 *
			 * When in #MANUAL or #FROM_MPI mode, each user process should collectively
			 * call this function. If \a broadcast is <tt>false</tt>, the input data
			 * will be passed from the external calling process to the corresponding ALP
			 * user processes in a one-to-one manner. Should \a broadcast be
			 * <tt>true</tt>, then the initial input data passed this way is overwritten
			 * for user processes \f$ s > 0 \f$ with the \a data_in passed at user
			 * process zero.
			 *
			 * Only in #MANUAL or #FROM_MPI modes will the output of any user processes
			 * with ID \f$ s > 0 \f$ be returned to all the processes that collectively
			 * call this function.
			 *
			 * In #AUTOMATIC mode, the output at \f$ s > 0 \f$ is lost. Only the output
			 * of the first user process \f$ s = 0 \f$ will be passed back to the root
			 * process that called this function.
			 *
			 * \note The default for \a broadcast is <tt>false</tt> as it is the variant
			 *       that implies the least cost when launching a program.
			 *
			 * \note The #FROM_MPI mode is specific to this imlementation and need not
			 *       be provided as part of the specification.
			 *
			 * @return #grb::SUCCESS If the execution proceeded as intended.
			 * @return #grb::ILLEGAL If \a broadcast was <tt>false</tt> and \a mode was
			 *                       #AUTOMATIC, but \a T not default-constructible.
			 * @return #grb::PANIC   If an unrecoverable error was encountered while
			 *                       attempting to execute, attempting to terminate, or
			 *                       while executing, the given program.
			 *
			 * \warning Even if #grb::SUCCESS is returned, an algorithm may fail to
			 *          achieve its intended result-- for example, an iterative solver
			 *          may fail to converge. A good programming pattern has that \a U
			 *          either a) is an error code for the algorithm used (e.g.,
			 *          <tt>int</tt> or #grb::RC), or that b) \a U is a struct that
			 *          contains such an error code.
			 */
			template< typename T, typename U >
			RC exec(
				AlpTypedFunc< T, U > alp_program,
				const T &data_in,
				U &data_out,
				const bool broadcast = false
			) const {
				(void) alp_program;
				(void) data_in;
				(void) data_out;
				(void) broadcast;
				// stub implementation, should be overridden by specialised backend,
				// so return error code
				return PANIC;
			}

			/**
			 * Executes a given ALP program using the user processes encapsulated by this
			 * launcher group.
			 *
			 * This variant of exec has that \a data_in is of a variable byte size,
			 * instead of a fixed (POD pointer-less) type. We refer to the given function
			 * as an untyped ALP function (since the input is a raw pointer), whereas the
			 * other variant executes \em typed ALP functions instead.
			 *
			 * If \a broadcast is <tt>true</tt>, all bytes are broadcast from the user
			 * process with ID zero to all other user processes.
			 *
			 * \note When in #MANUAL or #FROM_MPI mode, this implies any arguments passed
			 *       in a process-to-process manner will be lost.
			 *
			 * If \a broadcast is <tt>false</tt> and the launcher in #AUTOMATIC mode,
			 * then the user processes with ID \f$ s > 0 \f$ will receive \a data_in
			 * equal to <tt>nullptr</tt> and \a in_size equal to zero.
			 *
			 * See the \em typed ALP exec variant for more detailed comments, which also
			 * transfer to this untyped variant.
			 *
			 * @param[in]  alp_program The (untyped) user program to be executed.
			 * @param[in]  data_in     Pointer to raw input byte data.
			 * @param[in]  in_size     The number of bytes the input data consists of.
			 * @param[out] data_out    Output data of user-defined type \a U. The output
			 *                         data should be available at user process with ID
			 *                         zero.
			 * @param[in]  broadcast   Whether the input should be broadcast from user
			 *                         process 0 to all other user processes. Optional;
			 *                         the default value is \a false.
			 *
			 * @return #grb::SUCCESS If the execution proceeded as intended.
			 * @return #grb::ILLEGAL If \a in_size is larger than zero but \a data_in is
			 *                       equal to <tt>nullptr</tt>.
			 * @return #grb::PANIC   If an unrecoverable error was encountered while
			 *                       attempting to execute, attempting to terminate, or
			 *                       while executing, the given program.
			 */
			template< typename U >
			RC exec(
				AlpUntypedFunc< U > alp_program,
				const void * data_in,
				const size_t in_size,
				U &data_out,
				const bool broadcast = false
			) const {
				(void) alp_program;
				(void) data_in;
				(void) in_size;
				(void) data_out;
				(void) broadcast;
				return PANIC;
			}

			/**
			 * Releases all ALP resources.
			 *
			 * After a call to this function, no further ALP programs may launched using
			 * \em any #grb::Launcher or #grb::Benchmarker instance. Implementations and
			 * backends shall under no circumstance require a call to this function; any
			 * use of this function shall remain purely optional.
			 *
			 * \warning After a call to this function, also any subsequent call to the
			 *          deprecated #grb::init and #grb::finalize will no longer be
			 *          accepted.
			 *
			 * \internal
			 * \todo Remove the above warning once #grb::init and #grb::finalize are
			 *       moved to an internal namespace.
			 * \endinternal
			 *
			 * After a call to this function, the only way to once again run ALP programs
			 * is to use the #grb::Launcher from a different process.
			 *
			 * \warning Therefore, use this function with care and preferably only just
			 *          before exiting the process-- or not at all.
			 *
			 * @return #grb::SUCCESS The resources have successfully and permanently been
			 *                       released.
			 * @return #grb::PANIC   An unrecoverable error has been encountered and the
			 *                       user program is encouraged to exit as quickly as
			 *                       possible. The state of the ALP library has become
			 *                       undefined and should no longer be used.
			 *
			 * \note In the terminology of the Message Passing Interface (MPI), this
			 *       function is similar to <tt>MPI_Finalize()</tt>.
			 *
			 * \warning Different from MPI, however, a call to this function at program
			 *          exit is not mandatory.
			 *
			 * \warning An application that launches ALP programs in #grb::FROM_MPI mode
			 *          that calls this function, must (afterwards) still make a call to
			 *          <tt>MPI_Finalize()</tt>.
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

