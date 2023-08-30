
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
 * This file contains a variant on the #grb::Launcher specialised for
 * benchmarks.
 *
 * @author J. W. Nash & A. N. Yzelman
 * @date 17th of April, 2017
 */

#ifndef _H_GRB_BENCH_BASE
#define _H_GRB_BENCH_BASE

#include <cmath>  // for sqrt
#include <limits>
#include <vector> // warning: normally should not be used in ALP backends(!)

#ifndef _GRB_NO_STDIO
 #include <ios>
 #include <chrono>
 #include <iostream>
#endif

#ifndef _GRB_NO_EXCEPTIONS
 #include <stdexcept>
#endif

#include <graphblas/backends.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/utils.hpp>
#include <graphblas/utils/TimerResults.hpp>

#include "collectives.hpp"
#include "config.hpp"
#include "exec.hpp"


/**
 * \defgroup benchmarking Benchmarking
 *
 * ALP has a specialised class for benchmarking ALP programs, grb::Benchmarker,
 * which is a variant on the #grb::Launcher. It codes a particular benchmarking
 * strategy of any given ALP program as described below.
 *
 * The program is called \a inner times \a outer times. Between every
 * \a inner repetitions there is a one-second sleep that ensures machine
 * variability is taken into account. Several statistics are measured
 * across the \a outer repetitions: the minimum, maximum, average, and the
 * (unbiased) sample standard deviation. By contrast, for the \a inner
 * repetitions, only an average is computed -- the function of \a inner
 * repetitions is solely to avoid timing programs that execute in too short
 * a time frame, meaning a time frame that is of a similar order as the time
 * it takes to actually call the system timer functionalities.
 *
 * \note As a result, \a inner should always equal \em one when benchmarking
 *       any non-trivial ALP program, while for benchmarking ALP kernels on
 *       small data \a inner may be taken (much) larger.
 *
 * \note In published experiments, \a inner is chosen such that a single
 *       outer repetition takes 10 to 100 milliseconds.
 */

namespace grb {

	namespace internal {

		/**
		 * The common functionalities used by all #grb::Benchmarker classes.
		 *
		 * \ingroup benchmarking
		 */
		class BenchmarkerBase {

			protected:

#ifndef _GRB_NO_STDIO
				/**
				 * A helper function that prints the time elapsed sinc epoch.
				 *
				 * @param[in] printHeader An optional Boolean parameter with default value
				 *                        <tt>true</tt>. If set, this function will append
				 *                        a human-readable header before outputting the
				 *                        time-since-epoch.
				 */
				static void printTimeSinceEpoch( const bool printHeader = true ) {
					const auto now = std::chrono::system_clock::now();
					const auto since = now.time_since_epoch();
					if( printHeader ) {
						std::cout << "Time since epoch (in ms.): ";
					}
					std::cout << std::chrono::duration_cast<
							std::chrono::milliseconds
						>( since ).count() << "\n";
				}
#endif

				/**
				 * Calculate inner loop performance stats
				 */
				static void benchmark_calc_inner(
					const size_t loop,
					const size_t total,
					grb::utils::TimerResults &inner_times,
					grb::utils::TimerResults &total_times,
					grb::utils::TimerResults &min_times,
					grb::utils::TimerResults &max_times,
					std::vector< grb::utils::TimerResults > &sdev_times
				) {
					inner_times.normalize( total );
					total_times.accum( inner_times );
					min_times.min( inner_times );
					max_times.max( inner_times );
					sdev_times[ loop ] = inner_times;
				}

				/**
				 * Calculate outer loop performance stats
				 */
				static void benchmark_calc_outer(
					const size_t total,
					grb::utils::TimerResults &total_times,
					grb::utils::TimerResults &min_times,
					grb::utils::TimerResults &max_times,
					std::vector< grb::utils::TimerResults > &sdev_times,
					const size_t pid
				) {
					total_times.normalize( total );
					grb::utils::TimerResults sdev;
					// compute standard dev of average times, leaving sqrt calculation until
					// the output of the values
					sdev.set( 0 );
					for( size_t i = 0; i < total; i++ ) {
						double diff = sdev_times[ i ].io - total_times.io;
						sdev.io += diff * diff;
						diff = sdev_times[ i ].preamble - total_times.preamble;
						sdev.preamble += diff * diff;
						diff = sdev_times[ i ].useful - total_times.useful;
						sdev.useful += diff * diff;
						diff = sdev_times[ i ].postamble - total_times.postamble;
						sdev.postamble += diff * diff;
					}
					// unbiased normalisation of the standard deviation
					sdev.normalize( total - 1 );

#ifndef _GRB_NO_STDIO
					// output results
					if( pid == 0 ) {
						std::cout << "Overall timings (io, preamble, useful, postamble):\n"
							<< std::scientific;
						std::cout << "Avg: " << total_times.io << ", " << total_times.preamble
							<< ", " << total_times.useful << ", " << total_times.postamble << "\n";
						std::cout << "Min: " << min_times.io << ", " << min_times.preamble << ", "
							<< min_times.useful << ", " << min_times.postamble << "\n";
						std::cout << "Max: " << max_times.io << ", " << max_times.preamble << ", "
							<< max_times.useful << ", " << max_times.postamble << "\n";
						std::cout << "Std: " << sqrt( sdev.io ) << ", " << sqrt( sdev.preamble )
							<< ", " << sqrt( sdev.useful ) << ", " << sqrt( sdev.postamble ) << "\n";
 #if __GNUC__ > 4
						std::cout << std::defaultfloat;
 #endif
						printTimeSinceEpoch();
					}
#else
					// we ran the benchmark, but may not have a way to output it in this case
					// this currently only is touched by the #grb::banshee backend, which
					// provides other timing mechanisms.
					(void) min_times;
					(void) max_times;
					(void) pid;
#endif
				}

				/**
				 * Benchmarks a given ALP program.
				 *
				 * This variant applies to input data as a user-defined POD struct and
				 * output data as a user-defined POD struct.
				 *
				 * @tparam RunnerType The type of the runner, i.e., functor object storing
				 *                    the information for running the supplied ALP function.
				 *
				 * @param[in]  runner      Functor object running the function.
				 * @param[in]  times       Data structure with timing information.
				 * @param[in]  inner       Number of inner iterations.
				 * @param[out] outer       Number of outer iterations.
				 * @param[in]  pid process Identifier of current user process.
				 *
				 * @see benchmarking
				 *
				 * @ingroup benchmarking
				 */
				template<
					enum Backend implementation,
					typename RunnerType
				>
				static RC benchmark(
					RunnerType &runner,
					grb::utils::TimerResults &times,
					const size_t inner,
					const size_t outer,
					const size_t pid
				) {
					const double inf = std::numeric_limits< double >::infinity();
					grb::utils::TimerResults total_times, min_times, max_times;
					std::vector< grb::utils::TimerResults > sdev_times( outer );
					total_times.set( 0 );
					min_times.set( inf );
					max_times.set( 0 );
					grb::RC ret = grb::SUCCESS;

					// outer loop
					for( size_t out = 0; out < outer && ret == grb::SUCCESS; ++out ) {
						grb::utils::TimerResults inner_times;
						inner_times.set( 0 );

						// inner loop
						for( size_t in = 0; in < inner && ret == grb::SUCCESS; ++in ) {
							times.set( 0 );

							runner();

							ret = ret ? ret : grb::collectives< implementation >::reduce(
								times.io, 0, grb::operators::max< double >() );
							ret = ret ? ret : grb::collectives< implementation >::reduce(
								times.preamble, 0, grb::operators::max< double >() );
							ret = ret ? ret : grb::collectives< implementation >::reduce(
								times.useful, 0, grb::operators::max< double >() );
							ret = ret ? ret : grb::collectives< implementation >::reduce(
								times.postamble, 0, grb::operators::max< double >() );

							if( ret == grb::SUCCESS ) {
								inner_times.accum( times );
							}
						}

						if( ret == grb::SUCCESS ) {
							// calculate performance stats
							benchmark_calc_inner( out, inner, inner_times, total_times, min_times,
								max_times, sdev_times );
						}

#ifndef _GRB_NO_STDIO
						// give experiment output line
						if( pid == 0 ) {
							if( ret == grb::SUCCESS ) {
								std::ios_base::fmtflags prev_cout_state( std::cout.flags() );
								std::cout << "Outer iteration #" << out << " timings "
									<< "(io, preamble, useful, postamble, time since epoch): "
									<< std::fixed
									<< inner_times.io << ", " << inner_times.preamble << ", "
									<< inner_times.useful << ", " << inner_times.postamble << ", ";
									printTimeSinceEpoch( false );
								std::cout.flags( prev_cout_state );
							} else {
								std::cerr << "Error during cross-process collection of timing results: "
									<< "\t" << grb::toString( ret ) << std::endl;
							}
						}
#endif

						// pause for next outer loop
						if( sleep( 1 ) != 0 && ret == grb::SUCCESS ) {
#ifndef _GRB_NO_STDIO
							std::cerr << "Sleep interrupted, assume benchmark is unreliable; "
								<< "exiting.\n";
#endif
							abort();
						}
					}

					if( ret == grb::SUCCESS ) {
						// calculate performance stats
						benchmark_calc_outer( outer, total_times, min_times, max_times,
							sdev_times, pid );
					}

					return ret;
				}

				/**
				 * Benchmarks a given ALP program.
				 *
				 * This variant applies to input data as a byte blob and output data as a
				 * user-defined POD struct.
				 *
				 * @tparam U       Output type of the given user program.
				 * @tparam backend Which backend the program is using.
				 *
				 * @param[in]  alp_program The user program to be benchmarked.
				 * @param[in]  data_in     Input data as a raw data blob.
				 * @param[in]  in_size     The size, in bytes, of the input data.
				 * @param[out] out_data    Output data as a plain-old-data struct \a U.
				 * @param[in]  inner       Number of inner repetitions of the benchmark.
				 * @param[in]  outer       Number of outer repetitions of the benchmark.
				 * @param[in]  pid         Unique ID of the calling user process.
				 *
				 * @see benchmarking
				 *
				 * @ingroup benchmarking
				 */
				template<
					typename U,
					enum Backend implementation
				>
				static RC benchmark(
					void ( *alp_program )( const void *, const size_t, U & ),
					const void * data_in,
					const size_t in_size,
					U &data_out,
					const size_t inner,
					const size_t outer,
					const size_t pid
				) {
					struct void_runner {
						void ( *fun )( const void *, const size_t, U & );
						const void * in;
						const size_t ins;
						U &out;

						inline void operator()() {
							fun( in, ins, out );
						}

					} runner = { alp_program, data_in, in_size, data_out };
					return benchmark< implementation >( runner, data_out.times, inner, outer,
						pid );
				}

				/**
				 * Benchmarks a given ALP program.
				 *
				 * This variant applies to input data as a user-defined POD struct and
				 * output data as a user-defined POD struct.
				 *
				 * @tparam T Input type of the given user program.
				 * @tparam U Output type of the given user program.
				 *
				 * @param[in]  alp_program The user program to be benchmarked.
				 * @param[in]  data_in     Input data as a raw data blob.
				 * @param[in]  in_size     The size, in bytes, of the input data.
				 * @param[out] out_data    Output data.
				 * @param[in]  inner       Number of inner repetitions of the benchmark.
				 * @param[in]  outer       Number of outer repetitions of the benchmark.
				 * @param[in]  pid         Unique ID of the calling user process.
				 *
				 * @see benchmarking
				 *
				 * @ingroup benchmarking
				 */
				template<
					typename T, typename U,
					enum Backend implementation
				>
				static RC benchmark(
					void ( *alp_program )( const T &, U & ),
					const T &data_in,
					U &data_out,
					const size_t inner,
					const size_t outer,
					const size_t pid
				) {
					struct typed_runner {
						void ( *fun )( const T &, U & );
						const T &in;
						U &out;

						inline void operator()() {
							fun( in, out );
						}

					} runner = { alp_program, data_in, data_out };
					return benchmark< implementation >( runner, data_out.times, inner, outer,
						pid );
				}


			public:

				BenchmarkerBase() {
#ifndef _GRB_NO_STDIO
					printTimeSinceEpoch();
#endif
				}

		};

	} // namespace internal

	/**
	 * A class that follows the API of the #grb::Launcher, but instead of launching
	 * the given ALP program once, it launches it multiple times while benchmarking
	 * its execution times.
	 *
	 * @ingroup benchmarking
	 * @see benchmarking
	 */
	template< enum EXEC_MODE mode, enum Backend implementation >
	class Benchmarker {

		public :

			/**
			 * Constructs an instance of the benchmarker class.
			 *
			 * @param[in] process_id A unique ID for the calling user process.
			 * @param[in] nprocs     The total number of user processes participating in
			 *                       the benchmark. The given \a process_id must be
			 *                       strictly smaller than this given value.
			 * @param[in] hostname   The hostname where one of the user processes
			 *                       participating in the benchmark resides.
			 * @param[in] port       A free TCP/IP port at the host corresponding to
			 *                       the given \a hostname.
			 *
			 * The \a hostname and \a port arguments are unused if \a nprocs equals one.
			 *
			 * All arguments are optional-- their defaults are:
			 *  - 0 for \a process_id,
			 *  - 1 for \a nprocs,
			 *  - \em localhost for \a hostname, and
			 *  - 0 for \a port.
			 *
			 * This constructor may throw the same errors as #grb::Launcher.
			 *
			 * @see #grb::Launcher
			 * @see benchmarking
			 *
			 * \internal This is the base class which should be overridden by given
			 *           backend implementations.
			 */
			Benchmarker(
				const size_t process_id = 0,
				size_t nprocs = 1,
				std::string hostname = "localhost",
				std::string port = "0"
			) {
				(void) process_id; (void) nprocs; (void) hostname; (void) port;
#ifndef _GRB_NO_EXCEPTIONS
				throw std::logic_error( "Benchmarker class called with unsupported mode or "
					"implementation" );
#endif
			}

			/**
			 * Benchmarks a given ALP program.
			 *
			 * This variant applies to input data as a user-defined POD struct and
			 * output data as a user-defined POD struct.
			 *
			 * @tparam T Input type of the given user program.
			 * @tparam U Output type of the given user program.
			 *
			 * @param[in]  alp_program The ALP program to be benchmarked.
			 * @param[in]  data_in     Input data as a raw data blob.
			 * @param[out] data_out    Output data.
			 * @param[in]  inner       Number of inner repetitions of the benchmark.
			 * @param[in]  outer       Number of outer repetitions of the benchmark.
			 * @param[in]  broadcast   An optional argument that dictates whether the
			 *                         \a data_in argument should be broadcast across all
			 *                         user processes participating in the benchmark,
			 *                         prior to \em each invocation of \a alp_program.
			 *
			 * The default value of \a broadcast is <tt>false</tt>.
			 *
			 * @returns #grb::SUCCESS The benchmarking has completed successfully.
			 * @returns #grb::FAILED  An error during benchmarking has occurred. The
			 *                        benchmark attempt could be retried, and an error
			 *                        for the failure is reported to the standard error
			 *                        stream.
			 * @returns #grb::PANIC   If an unrecoverable error was encountered while
			 *                        starting the benchmark, while benchmarking, or
			 *                        while aggregating the final results.
			 *
			 * @see benchmarking
			 *
			 * \internal This is the base implementation that should be specialised by
			 *           each backend separately.
			 */
			template< typename T, typename U >
			RC exec(
				void ( *alp_program )( const T &, U & ),
				const T &data_in,
				U &data_out,
				const size_t inner,
				const size_t outer,
				const bool broadcast = false
			) const {
				(void) alp_program;
				(void) data_in;
				(void) data_out;
				(void) inner;
				(void) outer;
				(void) broadcast;

				// stub implementation, should be overridden by specialised implementation.
				// furthermore, it should be impossible to call this function without
				// triggering an exception during construction of this stub class, so we
				// just return PANIC here
#ifndef _GRB_NO_STDIO
				std::cerr << "Error: base Benchmarker::exec called. An implementation-"
					<< "specific variant should have been called instead.\n";
#endif
				return PANIC;
			}

			/**
			 * Benchmarks a given ALP program.
			 *
			 * This variant applies to input data as a byte blob and output data as a
			 * user-defined POD struct.
			 *
			 * @tparam U Output type of the given user program.
			 *
			 * @param[in]  alp_program The user program to be benchmarked.
			 * @param[in]  data_in     Input data as a raw data blob.
			 * @param[in]  in_size     The size, in bytes, of the input data.
			 * @param[out] data_out    Output data.
			 * @param[in]  inner       Number of inner repetitions of the benchmark.
			 * @param[in]  outer       Number of outer repetitions of the benchmark.
			 * @param[in]  broadcast   An optional argument that dictates whether the
			 *                         \a data_in argument should be broadcast across all
			 *                         user processes participating in the benchmark,
			 *                         prior to \em each invocation of \a alp_program.
			 *
			 * The default value of \a broadcast is <tt>false</tt>.
			 *
			 * @returns #grb::SUCCESS The benchmarking has completed successfully.
			 * @returns #grb::ILLEGAL  If \a in_size is nonzero but \a data_in compares
			 *                        equal to <tt>nullptr</tt>.
			 * @returns #grb::FAILED  An error during benchmarking has occurred. The
			 *                        benchmark attempt could be retried, and an error
			 *                        for the failure is reported to the standard error
			 *                        stream.
			 * @returns #grb::PANIC   If an unrecoverable error was encountered while
			 *                        starting the benchmark, while benchmarking, or
			 *                        while aggregating the final results.
			 *
			 * @see benchmarking
			 *
			 * \internal This is the base implementation that should be specialised by
			 *           each backend separately.
			 */
			template< typename U >
			RC exec(
				void ( *alp_program )( const void *, const size_t, U & ),
				const void * data_in, const size_t in_size,
				U &data_out,
				const size_t inner, const size_t outer,
				const bool broadcast = false
			) const {
				(void) alp_program;
				(void) data_in;
				(void) in_size;
				(void) data_out;
				(void) inner;
				(void) outer;
				(void) broadcast;

				// stub implementation, should be overridden by specialised implementation.
				// furthermore, it should be impossible to call this function without
				// triggering an exception during construction of this stub class, so we
				// just return PANIC here
#ifndef _GRB_NO_STDIO
				std::cerr << "Error: base Benchmarker::exec called. An implementation-"
					<< "specific variant should have been called instead.\n";
#endif
				return PANIC;
			}

			/**
			 * Releases all ALP resources.
			 *
			 * Calling this function is equivalent to calling #grb::Launcher::finalize.
			 *
			 * After a call to this function, no further ALP programs may be benchmarked
			 * nor launched-- i.e., both the #grb::Launcher and #grb::Benchmarker
			 * functionalities many no longer be used.
			 *
			 * A well-behaving program calls this function, or #grb::Launcher::finalize,
			 * exactly once and just before exiting (or just before the guaranteed last
			 * invocation of an ALP program).
			 *
			 * @return #grb::SUCCESS The resources have successfully and permanently been
			 *                       released.
			 * @return #grb::PANIC   An unrecoverable error has been encountered and the
			 *                       user program is encouraged to exit as quickly as
			 *                       possible. The state of the ALP library has become
			 *                       undefined and should no longer be used.
			 *
			 * \internal This is the base implementation that should be specialised by
			 *           each backend separately.
			 */
			static RC finalize() {
				return Launcher< mode, implementation >::finalize();
			}

	};

} // end namespace ``grb''

#endif // end _H_GRB_BENCH_BASE

