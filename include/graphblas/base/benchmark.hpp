
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
 * @author J. W. Nash & A. N. Yzelman
 * @date 17th of April, 2017
 */

#ifndef _H_GRB_BENCH_BASE
#define _H_GRB_BENCH_BASE

#include <chrono>
#include <ios>
#include <limits>
#include <string>

#include <graphblas/backends.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/utils.hpp>
#include <graphblas/utils/TimerResults.hpp>

#include "collectives.hpp"
#include "config.hpp"
#include "exec.hpp"

#ifndef _GRB_NO_STDIO
#include <iostream>
#endif

#ifndef _GRB_NO_EXCEPTIONS
#include <stdexcept>
#endif

#include <math.h>

namespace grb {

	namespace internal {

		class BenchmarkerBase {

		protected:
#ifndef _GRB_NO_STDIO
			/** \todo TODO add documentation. */
			static void printTimeSinceEpoch( const bool printHeader = true ) {
				const auto now = std::chrono::system_clock::now();
				const auto since = now.time_since_epoch();
				if( printHeader ) {
					std::cout << "Time since epoch (in ms.): ";
				}
				std::cout << std::chrono::duration_cast< std::chrono::milliseconds >( since ).count() << "\n";
			}
#endif

			// calculate inner loop performance stats
			static void benchmark_calc_inner( const size_t loop,
				const size_t total,
				grb::utils::TimerResults & inner_times,
				grb::utils::TimerResults & total_times,
				grb::utils::TimerResults & min_times,
				grb::utils::TimerResults & max_times,
				grb::utils::TimerResults * sdev_times ) {
				inner_times.normalize( total );
				total_times.accum( inner_times );
				min_times.min( inner_times );
				max_times.max( inner_times );
				sdev_times[ loop ] = inner_times;
			}

			// calculate outer loop performance stats
			static void benchmark_calc_outer( const size_t total,
				grb::utils::TimerResults & total_times,
				grb::utils::TimerResults & min_times,
				grb::utils::TimerResults & max_times,
				grb::utils::TimerResults * sdev_times,
				const size_t pid ) {
				total_times.normalize( total );
				grb::utils::TimerResults sdev;
				// compute standard dev of average times, leaving sqrt calculation until the output of the values
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
					std::cout << "Overall timings (io, preamble, useful, "
								 "postamble):\n"
							  << std::scientific;
					std::cout << "Avg: " << total_times.io << ", " << total_times.preamble << ", " << total_times.useful << ", " << total_times.postamble << "\n";
					std::cout << "Min: " << min_times.io << ", " << min_times.preamble << ", " << min_times.useful << ", " << min_times.postamble << "\n";
					std::cout << "Max: " << max_times.io << ", " << max_times.preamble << ", " << max_times.useful << ", " << max_times.postamble << "\n";
					std::cout << "Std: " << sqrt( sdev.io ) << ", " << sqrt( sdev.preamble ) << ", " << sqrt( sdev.useful ) << ", " << sqrt( sdev.postamble ) << "\n";
#if __GNUC__ > 4
					std::cout << std::defaultfloat;
#endif
					printTimeSinceEpoch();
				}
#else
				// write to file(?)
				(void)min_times;
				(void)max_times;
				(void)pid;
#endif
			}

			template< typename U, enum Backend implementation = config::default_backend >
			static RC benchmark( void ( *grb_program )( const void *,
									 const size_t,
									 U & ), // user GraphBLAS program
				const void * data_in,
				const size_t in_size,
				U & data_out, // input & output data
				const size_t inner,
				const size_t outer,
				const size_t pid ) {
				const double inf = std::numeric_limits< double >::infinity();
				grb::utils::TimerResults total_times, min_times, max_times;
				grb::utils::TimerResults * sdev_times = new grb::utils::TimerResults[ outer ];
				total_times.set( 0 );
				min_times.set( inf );
				max_times.set( 0 );

				// outer loop
				for( size_t out = 0; out < outer; out++ ) {
					grb::utils::TimerResults inner_times;
					inner_times.set( 0 );

					// inner loop
					for( size_t in = 0; in < inner; in++ ) {
						data_out.times.set( 0 );
						( *grb_program )( data_in, in_size, data_out );
						grb::collectives< implementation >::reduce( data_out.times.io, 0, grb::operators::max< double >() );
						grb::collectives< implementation >::reduce( data_out.times.preamble, 0, grb::operators::max< double >() );
						grb::collectives< implementation >::reduce( data_out.times.useful, 0, grb::operators::max< double >() );
						grb::collectives< implementation >::reduce( data_out.times.postamble, 0, grb::operators::max< double >() );
						inner_times.accum( data_out.times );
					}

					// calculate performance stats
					benchmark_calc_inner( out, inner, inner_times, total_times, min_times, max_times, sdev_times );

#ifndef _GRB_NO_STDIO
					// give experiment output line
					if( pid == 0 ) {
						std::cout << "Outer iteration #" << out
								  << " timings (io, preamble, useful, "
									 "postamble, time since epoch): ";
						std::cout << inner_times.io << ", " << inner_times.preamble << ", " << inner_times.useful << ", " << inner_times.postamble << ", ";
						printTimeSinceEpoch( false );
					}
#endif

					// pause for next outer loop
					if( sleep( 1 ) != 0 ) {
#ifndef _GRB_NO_STDIO
						std::cerr << "Sleep interrupted, assume benchmark is "
									 "unreliable and exiting.\n";
#endif
						abort();
					}
				}

				// calculate performance stats
				benchmark_calc_outer( outer, total_times, min_times, max_times, sdev_times, pid );
				delete[] sdev_times;

				return SUCCESS;
			}

			template< typename T, typename U, enum Backend implementation = config::default_backend >
			static RC benchmark( void ( *grb_program )( const T &, U & ), // user GraphBLAS program
				const T & data_in,
				U & data_out, // input & output data
				const size_t inner,
				const size_t outer,
				const size_t pid ) {
				const double inf = std::numeric_limits< double >::infinity();
				grb::utils::TimerResults total_times, min_times, max_times;
				grb::utils::TimerResults * sdev_times = new grb::utils::TimerResults[ outer ];
				total_times.set( 0 );
				min_times.set( inf );
				max_times.set( 0 );

				// outer loop
				for( size_t out = 0; out < outer; out++ ) {
					grb::utils::TimerResults inner_times;
					inner_times.set( 0 );

					// inner loop
					for( size_t in = 0; in < inner; in++ ) {
						data_out.times.set( 0 );

						( *grb_program )( data_in, data_out );
						grb::collectives< implementation >::reduce( data_out.times.io, 0, grb::operators::max< double >() );
						grb::collectives< implementation >::reduce( data_out.times.preamble, 0, grb::operators::max< double >() );
						grb::collectives< implementation >::reduce( data_out.times.useful, 0, grb::operators::max< double >() );
						grb::collectives< implementation >::reduce( data_out.times.postamble, 0, grb::operators::max< double >() );
						inner_times.accum( data_out.times );
					}

					// calculate performance stats
					benchmark_calc_inner( out, inner, inner_times, total_times, min_times, max_times, sdev_times );

#ifndef _GRB_NO_STDIO
					// give experiment output line
					if( pid == 0 ) {
						std::cout << "Outer iteration #" << out
								  << " timings (io, preamble, useful, "
									 "postamble, time since epoch): "
								  << std::fixed;
						std::cout << inner_times.io << ", " << inner_times.preamble << ", " << inner_times.useful << ", " << inner_times.postamble << ", ";
						printTimeSinceEpoch( false );
						std::cout << std::scientific;
					}
#endif

					// pause for next outer loop
					if( sleep( 1 ) != 0 ) {
#ifndef _GRB_NO_STDIO
						std::cerr << "Sleep interrupted, assume benchmark is "
									 "unreliable and exiting.\n";
#endif
						abort();
					}
				}

				// calculate performance stats
				benchmark_calc_outer( outer, total_times, min_times, max_times, sdev_times, pid );
				delete[] sdev_times;

				return SUCCESS;
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
	 * Benchmarking function, called from an exec function.
	 * Takes the grbProgram and its input and output data and accumultes times
	 * given in the output structure.
	 */
	template< enum EXEC_MODE mode, enum Backend implementation >
	class Benchmarker {

		public :

			Benchmarker( size_t process_id = 0,     // user process ID
				size_t nprocs = 1,                  // total number of user processes
				std::string hostname = "localhost", // one of the user process hostnames
				std::string port = "0"              // a free port at hostname
			) { (void)process_id; (void)nprocs; (void)hostname; (void)port;
#ifndef _GRB_NO_EXCEPTIONS
				throw std::logic_error( "Benchmarker class called with unsupported "
										"mode or implementation" );
#endif
			}

	template< typename T, typename U >
	RC exec( void ( *grb_program )( const T &, U & ), // user GraphBLAS program
		const T & data_in,
		U & data_out, // input & output data
		const size_t inner,
		const size_t outer,
		const bool broadcast = false ) const {
		(void)grb_program;
		(void)data_in;
		(void)data_out;
		(void)inner;
		(void)outer;
		(void)broadcast;
		// stub implementation, should be overridden by specialised implementation,
		// so return error code
		return PANIC;
	}

	template< typename U >
	RC exec( void ( *grb_program )( const void *, const size_t, U & ), const void * data_in, const size_t in_size, U & data_out, const size_t inner, const size_t outer, const bool broadcast = false )
		const {
		(void)grb_program;
		(void)data_in;
		(void)in_size;
		(void)data_out;
		(void)inner;
		(void)outer;
		(void)broadcast;
		return PANIC;
	}

	/**
	 * Releases all GraphBLAS resources. After a call to this function, no
	 * GraphBLAS library functions may be called any longer.
	 *
	 * @return SUCCESS A call to this function may never fail.
	 */
	static RC finalize() {
		return Launcher< mode, implementation >::finalize();
	}

}; // namespace grb

} // end namespace ``grb''

#endif // end _H_GRB_BENCH_BASE
