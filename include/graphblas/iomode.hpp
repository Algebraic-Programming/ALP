
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
 * Defines the various I/O modes a user could employ with ALP data ingestion
 * or extraction.
 *
 * @author A. N. Yzelman
 * @date 21st of February, 2017
 */

#ifndef _H_GRB_IOMODE
#define _H_GRB_IOMODE


namespace grb {

	/**
	 * The GraphBLAS input and output functionalities can either be used in a
	 * sequential or parallel fashion. Input functions such as buildVector or
	 * buildMatrixUnique default to sequential behaviour, which means that the
	 * collective calls to either function must have the exact same arguments--
	 * that is, each user process is passed the exact same input data.
	 *
	 * \note This does not necessarily mean that all data is stored in a
	 *       replicated fashion across all user processes.
	 *
	 * This default behaviour comes with obvious performance penalties; each user
	 * process must scan the full input data set, which takes \f$ \Theta( n ) \f$
	 * time. Scalable behaviour would instead incur \f$ \Theta( n / P ) \f$ time,
	 * with \a P the number of user processes.
	 * Using a parallel IOMode provides exactly this scalable performance. On
	 * input, this means that each user process can pass different data to the
	 * same collective call to, e.g., buildVector or buildMatrixUnique.
	 *
	 * For output, which GraphBLAS provides via \a const iterators, sequential
	 * mode means that each user process retrieves an iterator over all output
	 * elements-- this requires costly all-to-all communication. Parallel mode
	 * output instead only returns those elements that do not require inter user-
	 * process communication.
	 *
	 * \note It is guaranteed the union of all output over all user processes
	 *       corresponds to all elements in the GraphBLAS container.
	 *
	 * See the respective functions and classes for full details:
	 *   -# grb::buildVector;
	 *   -# grb::buildMatrixUnique;
	 *   -# grb::Vector::const_iterator;
	 *   -# grb::Matrix::const_iterator.
	 */
	enum IOMode {

		/**
		 * Sequential mode IO.
		 *
		 * Use of this mode results in non-scalable input and output. Its use is
		 * recommended only in case of small data sets or in one-off situations.
		 */
		SEQUENTIAL = 0,

		/**
		 * Parallel mode IO.
		 *
		 * Use of this mode results in fully scalable input and output. Its use is
		 * recommended as a default. Note that this does require the user to have
		 * his or her data distributed over the various user processes on input,
		 * and requires the user to handle distributed data on output.
		 *
		 * This is the default mode on all GraphBLAS IO functions.
		 *
		 * \note The parallel mode in situations where the number of user processes
		 *       is one, for instance when choosing a sequential or data-centric
		 *       GraphBLAS implementation, IOMode::parallel is equivalent to
		 *       IOMode::sequential.
		 */
		PARALLEL
	};

} // namespace grb

#endif // end ``_H_GRB_IOMODE''

