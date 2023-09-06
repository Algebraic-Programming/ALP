
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
	template< typename InputType, typename OutputType >
	using AlpUntypedFunc = void ( * )( const InputType *,
		size_t, OutputType & );

	/**
	 * The various ways in which the #grb::Launcher can be used to execute an
	 * ALP program.
	 *
	 * \warning An implementation may require different linker commands
	 *          when using different modes.
	 *
	 * \warning Depending on the mode given to #grb::Launcher, the parameters
	 *          required for the exec function may differ.
	 *
	 * \note However, the ALP program remains unaware of which mode is the launcher
	 *       employs and will not have to change.
	 */
	enum EXEC_MODE {

		/**
		 * Automatic mode. The #grb::Launcher can spawn user processes
		 * which will execute a given program.
		 */
		AUTOMATIC = 0,

		/**
		 * Manual mode. The user controls \a nprocs user processes
		 * which together should execute a given program, by, for
		 * example, using the #grb::Launcher.
		 */
		MANUAL,

		/**
		 * When running from an MPI program. The user controls
		 * \a nprocs MPI programs, which, together, should execute
		 * a given ALP program.
		 */
		FROM_MPI

	};

	/**
	 * A group of user processes that together execute ALP programs.
	 *
	 * Allows an application to run any ALP program. Input data may be passed
	 * through a user-defined type. Output data will be retrieved via the same
	 * type.
	 *
	 * For backends that support multiple user processes, the caller may
	 * explicitly set the process ID and total number of user processes.
	 *
	 * The intended use is to `just call' the exec function, which should be
	 * accepted by any backend.
	 *
	 * @tparam mode    Which #EXEC_MODE the Launcher should adhere to.
	 * @tparam backend Which backend is to be used.
	 */
	template< enum EXEC_MODE mode, enum Backend backend >
	class Launcher {

		public :

			/**
			 * Constructs a new #grb::Launcher. This constructor is a collective call;
			 * all \a nprocs processes that form a single launcher group must make a
			 * simultaneous call to this constructor.
			 *
			 * There is an implementation-defined time-out for the creation of a launcher
			 * group.
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
			 * @throws invalid_argument If \a nprocs is zero.
			 * @throws invalid_argument If \a process_id is greater than or equal to
			 *                          \a nprocs.
			 *
			 * \note An implementation or backend may define further constraints on the
			 *       input arguments, such as, obviously, on \a hostname and \a port, but
			 *       also on \a nprocs and, as a result, on \a process_id.

			 * \note The most obvious is that backends supporting only one user process
			 *       must not accept \a nprocs larger than 1.
			 *
			 * All aforementioned default values shall always be legal.
			 */
			Launcher(
				const size_t process_id = 0,
				const size_t nprocs = 1,
				const std::string hostname = "localhost",
				const std::string port = "0"
			) {
				// spec does not specify any constrants on hostname and port
				// so accept (and ignore) anything
				(void) hostname; (void) port;

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
			 *  -# \em spawn the maximum number of available user processes and \em then
			 *     execute the given program, \em or
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
			 * \note In fact, \a T may be standard layout and contain no pointers. If it
			 *       is default-constructible, then \a broadcast may be <tt>false</tt>.
			 *
			 * \warning If \a T is \em not default-constructible, then during a call to
			 *          this function, \a broadcast must equal <tt>true</tt>.
			 *
			 * @tparam U The type of the output data to pass back to the caller. This may
			 *           be of any type.
			 *
			 * \note In case of multiple user processes, if \a mode is AUTOMATIC, then
			 *       the output of type \a U at user processes \f$ s > 0 \f$ will be
			 *       lost.
			 *
			 * @param[in]  alp_program The user program to be executed.
			 * @param[in]  data_in     Input data of user-defined type \a T.
			 *
			 * When in automatic mode and \a broadcast is <tt>false</tt>, the data will
			 * only be available at user process with ID 0. When in automatic mode and
			 * \a broadcast is <tt>true</tt>, the data will be available at all user
			 * processes.
			 *
			 * When in manual mode, each user process should collectively call this
			 * function. If \a broadcast is <tt>false</tt> each user process should
			 * collectively call this function. The input data will then be passed to
			 * the corresponding ALP user processes in a one-to-one manner. Should
			 * \a broadcast be <tt>true</tt>, then the initial input data passed to user
			 * processes \f$ s > 0 \f$ will be overwritten with the data passed to user
			 * process zero.
			 *
			 * @param[out] data_out  Output data of user-defined type \a U. The output
			 *                       data should be available at user process with ID
			 *                       zero.
			 *
			 * Only in #MANUAL or #FROM_MPI modes will the output of any user processes
			 * with ID \f$ s > 0 \f$ be returned to the respective processes involved
			 * with the collective call to this function. In #AUTOMATIC mode, the output
			 * at \f$ s > 0 \f$ is ignored.
			 *
			 * @param[in]  broadcast Whether the input should be broadcast from user
			 *                       process 0 to all other user processes. Optional;
			 *                       the default value is <tt>false</tt>.
			 *
			 * \note The default is <tt>false</tt> as it is the variant that implies the
			 *       least cost.
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
			 * instead of a fixed POD type. We refer to the given function as an untyped
			 * ALP function (since the input is a raw pointer), whereas the other variant
			 * executes \em typed ALP functions instead.
			 *
			 * If \a broadcast is <tt>true</tt> and the launcher, all bytes are broadcast
			 * to all user processes.
			 *
			 * \note When in #MANUAL or #FROM_MPI mode, this implies any arguments passed
			 *       in a process-to-process manner will be lost.
			 *
			 * See the \em typed ALP exec variant for more detailed comments, which also
			 * transfer to this untyped variant.
			 *
			 * @param[in]  alp_program The user program to be executed.
			 * @param[in]  data_in     Pointer to raw input byte data.
			 * @param[in]  in_size     The number of bytes the input data consists of.
			 * @param[out] data_out  Output data of user-defined type \a U. The output
			 *                       data should be available at user process with ID
			 *                       zero.
			 * @param[in]  broadcast Whether the input should be broadcast from user
			 *                       process 0 to all other user processes. Optional;
			 *                       the default value is \a false.
			 *
			 * @return #grb::SUCCESS If the execution proceeded as intended.
			 * @return #grb::ILLEGAL If \a broadcast was <tt>false</tt> and \a mode was
			 *                       #AUTOMATIC, but \a T not default-constructible.
			 * @return #grb::PANIC   If an unrecoverable error was encountered while
			 *                       attempting to execute, attempting to terminate, or
			 *                       while executing, the given program.
			 */
			template< typename U >
			RC exec(
				AlpUntypedFunc< void, U > alp_program,
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
			 * the #grb::Launcher and #grb::Benchmarker. Also the use of #grb::init and
			 * #grb::finalize will no longer be accepted.
			 *
			 * \warning #grb::init and #grb::finalize are deprecated.
			 *
			 * \internal
			 * \todo Remove the above comments once #grb::init and #grb::finalize are
			 *       moved to an internal namespace.
			 * \endinternal
			 *
			 * After a call to this function, the only way to once again run ALP programs
			 * is to use the #grb::Launcher from a new process.
			 *
			 * \warning Therefore, use this function with care and preferably only just
			 *          before exiting the process.

			 * A well-behaving program calls this function, or
			 * #grb::Benchmarker::finalize, exactly once before its process terminates,
			 * or just after the guaranteed last invocation of an ALP program.
			 *
			 * @return #grb::SUCCESS The resources have successfully and permanently been
			 *                       released.
			 * @return #grb::PANIC   An unrecoverable error has been encountered and the
			 *                       user program is encouraged to exit as quickly as
			 *                       possible. The state of the ALP library has become
			 *                       undefined and should no longer be used.
			 *
			 * \note In the terminology of the Message Passing Interface (MPI), this
			 *       function is the ALP equivalent of the <tt>MPI_Finalize()</tt>.
			 *
			 * \note In #grb::AUTOMATIC mode when using a parallel backend that uses MPI
			 *       to auto-parallelise the ALP computations, MPI is never explicitly
			 *       exposed to the user application. This use case necessitates the
			 *       specification of this function.
			 *
			 * \note Thus, and in particular, an ALP program launched in #grb::AUTOMATIC
			 *       mode while using the #grb::BSP1D or the #grb::hybrid backends with
			 *       ALP compiled using LPF that in turn is configured to use an
			 *       MPI-based engine, should make sure to call this function before
			 *       program exit.
			 *
			 * \note An application that launches ALP programs in #grb::FROM_MPI mode
			 *       must still call this function, even though a proper such application
			 *       makes its own call to <tt>MPI_Finalize()</tt>. This does \em not
			 *       induce improper behaviour since calling this function using a
			 *       launcher instance in #grb::FROM_MPI mode translates, from an MPI
			 *       perspective, to a no-op.
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

