
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
 * @author A. N. Yzelman
 * @date 12th of December, 2017
 */

#ifndef _H_GRB_UTILS_INDEXEDVECTORMAP
#define _H_GRB_UTILS_INDEXEDVECTORMAP

#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <graphblas.hpp>

namespace grb {

	namespace utils {

		/**
		 * Represents a set of vectors with unique IDs.
		 *
		 * The IDs may be any string, and the vectors must have some fixed length.
		 *
		 * @tparam ValueType The element type of the vectors.
		 */
		template< typename ValueType >
		class IndexedVectorMap {

		private:
			/** Maximum size of a single word (in number of chars). */
			static constexpr const size_t MAX_WORD_SIZE = 255;

			/** Dimension of the vectors assigned to each word. */
			size_t dimension;

			/** Small word buffer used for extracting first string of every line. */
			char wordbuf[ MAX_WORD_SIZE ];

			/** Distributed word map. */
			std::map< std::string, size_t > word2id;

			/** Global word vector store. */
			std::vector< grb::Vector< ValueType > > word2vec;

			/** The root node (or \a SIZE_MAX if there is none). */
			size_t root;

			/** There should be no use case where the default constructor is used. */
			IndexedVectorMap() {};

		public:
			/**
			 * The modes under which an instance of \a IndexedVectorMap can be used.
			 *
			 * This mainly deals with how the dictionary of strings to indices is
			 * constructed. Suppose the dictionary takes \f$ M \f$ memory. Then
			 * \a REPLICATED mode stores the full dictionary at each of the \f$ p \f$
			 * user processes for a total memory usage of \f$ pM \f$, which does not
			 * scale.
			 * The \a SEQUENTIAL mode instead stores the map only at the \a root
			 * process for a total memory usage of \f$ M \f$ bytes. This scales in
			 * memory for increasing \f$ p \f$, but assumes \f$ M \f$ fits in a single
			 * node's memory. (I.e., it does not scale for increasing \f$ M \f$.)
			 *
			 * A \a PARALLEL mode may be implemented once there is sufficient
			 * interest. This mode would achieve \f$ M / p \f$ storage per user
			 * process, thus achieving memory scalability in scaling both \f$ M \f$
			 * and \f$ p \f$.
			 *
			 * @see IndexedVectorMap
			 */
			enum Mode {

				/** The dictionary is replicated at each user process. */
				REPLICATED = 0,

				/** The dictionary is only available at a root node. */
				SEQUENTIAL
			};

			/**
			 * Constructs a string-to-vector map from reading a text file.
			 *
			 * The text file is assumed to consist of n lines. Each line starts with
			 * any string, followed by dim values of type ValueType. The separator is a
			 * single space. n does not need to be known beforehand. dim, by contrast,
			 * must be known.
			 *
			 * This is a collective function; if one user process makes a call to this
			 * function, there must be a matching function called by all of the other
			 * user processes. This is independent of which \a mode is used(!).
			 *
			 * @param[in] filename Path to the file to read in.
			 * @param[in] dim      The size of the vector associated to each word.
			 * @param[in] mode     Which mode the dictionary is parsed.
			 * @param[in] root_pid Which user process stores the dictionary. Only used
			 *                     if \a mode is \a SEQUENTIAL. Set to \a 0 by default.
			 *
			 * @throws invalid_argument When \a mode is \a SEQUENTIAL and \a root_pid
			 *                          is not smaller than the number of user
			 *                          processes.
			 *
			 * If \a mode is \a SEQUENTIAL, only the user process with id \a root_pid
			 * will have access to the dictionary. All other processes will only have
			 * a (distributed) view of the associated vectors.
			 *
			 * \parblock Performance semantics.
			 *
			 * This constructor scans the input file twice. When \a mode is
			 * \a SEQUENTIAL or \a REPLICATED, no communication between the user
			 * processes shall be incurred. If there are n lines, n GraphBLAS vectors
			 * shall be constructed. If the GraphBLAS implementation is parallel, each
			 * vector is distributed so that operations on the vectors occur in
			 * parallel. When having many queries on small vectors, consider using a
			 * sequential GraphBLAS backend either on a distributed dictionary
			 * (requires synchronisation on each query), or in a fully replicated
			 * fashion (does not scale in memory).
			 */
			IndexedVectorMap( const std::string filename, const size_t dim, const Mode mode, const size_t root_pid = 0 ) : dimension( dim ) {
				// get SPMD info
				const size_t my_id = grb::spmd<>::pid();
				const size_t P = grb::spmd<>::nprocs();
				if( mode == SEQUENTIAL ) {
					if( root_pid > P ) {
						throw std::invalid_argument(
							"root PID must be in range of current number "
							"of user processes"
						);
					}
					root = root_pid;
				}

				// declare and initialise local variables
				std::vector< ValueType > val_buffer;
				size_t counter = 0;
				val_buffer.resize( dimension );

				// open file
				std::ifstream input;
				input.open( filename );
				if( ! input ) {
					std::cerr << "distributedWordVector: cannot open wordvector file at "
						<< filename << "\n";
				}

				// build distributed word map. This is a sequential read by everyone involved.
				(void)input.get( wordbuf, MAX_WORD_SIZE, ' ' );
				if( input.gcount() == MAX_WORD_SIZE && input.peek() != ' ' ) {
					std::cerr << "Warning: the maximum word size " << MAX_WORD_SIZE
							  << " was insufficient. I am ignoring the remainder "
								 "characters. Recompile with a different value for "
								 "MAX_WORD_SIZE to fix this.\n";
				}
				while( input ) {
					const std::string word( wordbuf );
					if( ( mode == SEQUENTIAL && my_id == root ) || mode == REPLICATED ) {
						const auto it = word2id.find( word );
						if( it != word2id.end() ) {
							std::cerr << "Warning: doubly-defined key string "
										 "found. This key will not be "
										 "reachable from the dictionary.";
						} else {
							word2id[ word ] = counter;
						}
						if( ( mode == REPLICATED && my_id == 0 ) || ( mode == SEQUENTIAL && my_id == root ) ) {
							std::cout << "Registered word: " << word << " to have ID " << counter << ".\n";
						}
					}
					// increment counter and move on to next line
					(void)++counter;
					// fast-forward to end of line
					(void)input.ignore( std::numeric_limits< std::streamsize >::max(), '\n' );
					// read next word
					(void)input.get( wordbuf, MAX_WORD_SIZE, ' ' );
				}
				// initialise class variables
				grb::Vector< ValueType > modelVector( dimension );
				word2vec = std::vector< grb::Vector< ValueType > >( counter, modelVector );
				// rewind file
				input.clear();
				(void)input.seekg( std::ios_base::beg );
				if( ! input ) {
					throw std::runtime_error( "Could not rewind input file." );
				}
				// parse vectors
				counter = 0;
				// fast-forward past word
				(void)input.ignore( std::numeric_limits< std::streamsize >::max(), ' ' );
				while( input ) {
					ValueType value;
					for( size_t k = 0; k < dimension; ++k ) {
						assert( input );
						input >> value;
						val_buffer[ k ] = value;
						if( ! input ) {
							std::cerr << "Could not parse " << k << "-th value to the " << counter
									  << "-th vector. Attempting to retrieve "
										 "the erroneous line:"
									  << std::endl;
							std::string line;
							(void)input.clear();
							(void)std::getline( input, line );
							std::cerr << line << std::endl;
							throw std::runtime_error( "Parse error during "
													  "read-out of vector "
													  "elements." );
						}
					}
					const auto rc = grb::buildVector( word2vec[ counter ], val_buffer.begin(), val_buffer.end(), grb::IOMode::SEQUENTIAL );
					if( rc != SUCCESS ) {
						std::cerr << "Could not construct " << counter << "-th word vector. Error message: " << grb::toString( rc ) << std::endl;
						throw std::runtime_error( "Error during call to "
												  "grb::buildVector." );
					}
					// go to next line (if any)
					(void)input.ignore( std::numeric_limits< std::streamsize >::max(), '\n' );
					// now we are on a next line. Fast-forward past dictionary word
					(void)input.ignore( std::numeric_limits< std::streamsize >::max(), ' ' );
					// record we read a new line
					(void)++counter;
					// set eof bit if at EOF
					(void)input.peek();
				}
			}

			/**
			 * Retrieves the index of a single word.
			 *
			 * Note that for instances constructed by file, words are separated by
			 * single spaces, hence \a query should not contain any spaces in that
			 * case.
			 *
			 * @param[in] query A single word to search for in the dictionary.
			 *
			 * \warning Note that if \a mode was \a SEQUENTIAL while constructing
			 *          the current instance, then the local dictionary is empty
			 *          unless this user process is the \a root process.
			 *
			 * @return The index of the queried word, if it is in the dictionary;
			 *         or the maximum represantable number by \a size_t otherwise.
			 */
			size_t getIndex( const std::string & query ) const {
				const auto it = word2id.find( query );
				if( it == word2id.end() ) {
					return std::numeric_limits< size_t >::max();
				} else {
					return it->second;
				}
			}

			/**
			 * Retrieves the vector corresponding to a given word index.
			 *
			 * @param[in] index Index of the vector to return.
			 *
			 * @return The requested vector.
			 *
			 * @throws invalid_argument if \a index is out of range.
			 *
			 * \note This function works regardless of the \a mode chosen
			 *       during this instance's construction.
			 */
			Vector< ValueType > getVector( const size_t index ) const {
				return word2vec.at( index );
			}

			/**
			 * Retrieves the vector corresponding to a given word.
			 *
			 * If this instance was constructed in \a SEQUENTIAL \a mode, then
			 * this function may only be executed on the \a root user process.
			 * Any other use will result in an exception thrown.
			 *
			 * @param[in] query The word to search for.
			 *
			 * @return The requested vector.
			 *
			 * @throws invalid_argument If the word does not appear in the
			 *                          dictionary.
			 */
			Vector< ValueType > getVector( const std::string & query ) const {
				return getVector( getIndex( query ) );
			}

			/**
			 * Takes a sentence, splits it up and words, retrieves all corresponding
			 * vectors, and folds those into the given vector.
			 *
			 * @tparam descr      The descriptor for the grb::foldl function.
			 * @tparam Operator   The operator to pass to the grb::foldl function.
			 * @tparam OutputType The output type of the combined vector.
			 *
			 * @param[in,out] combinedVector This will act as the left argument to
			 *                               successive calls to grb::foldl.
			 * @param[in]     query          The query input sentence.
			 * @param[in]     op             The operator for use with grb::foldl.
			 *
			 * A single space is taken as the delimiter of the \a query sentence.
			 * The \a combinedVector must be a dense vector.
			 *
			 * Words that do not appear in the dictionary will result in an error
			 * message printed to std::err, but otherwise it will be as though the
			 * query did not contain the word in question.
			 *
			 * @returns MISMATCH If the dimension of \a combinedVector does not match
			 *                   the dimensions of the vectors stored in this map. It
			 *                   shall then be as though this call was never made.
			 * @returns ILLEGAL  If vector \a combinedVector was sparse on input. It
			 *                   shall then be as though this call was never made.
			 * @returns SUCCESS  On successful exection of this function.
			 *
			 * If an error other than the above occurs, this will result in a message
			 * printed to std::err and a premature exit of the function. The
			 * appropriate grb::RC error code is then returned. See @grb::foldl for
			 * such possible error conditions.
			 *
			 * \warning Note that if \a mode was \a SEQUENTIAL while constructing
			 *          the current instance, then the local dictionary is empty
			 *          unless this user process is the \a root process.
			 *
			 * \warning \a noexcept is actually only guaranteed when the user has not
			 *          overriden STL's IOStreams to throw exceptions instea dof setting
			 *          fail bits.
			 */
			template< descriptors::Descriptor descr = descriptors::no_operation, class Operator, class OutputType >
			RC foldlSentence( grb::Vector< OutputType > & combinedVector, const std::string & query, const Operator & op ) const noexcept {
				if( size( combinedVector ) != dimension ) {
					return MISMATCH;
				}
				if( nnz( combinedVector ) < size( combinedVector ) ) {
					return ILLEGAL;
				}
				std::string word;
				std::istringstream s( query );
				s >> word;
				while( s && word.size() > 0 ) {
					const size_t index = getIndex( word );
					if( index == std::numeric_limits< size_t >::max() ) {
						std::cerr << "Did not find word: " << word << ". Ignoring it.\n";
					} else {
						const RC rc = grb::foldl< descr >( combinedVector, word2vec[ index ], op );
						if( rc != SUCCESS ) {
							std::cerr << "Error while accumulating word "
										 "vector: " +
									grb::toString( rc ) + "\n";
							return rc;
						}
					}
					s >> word;
				}
				return SUCCESS;
			}

			/**
			 * Folds all vectors with the given IDs into the given combinedVector.
			 *
			 * @tparam descr      The descriptor for the grb::foldl function.
			 * @tparam Operator   The operator to pass to the grb::foldl function.
			 * @tparam OutputType The output type of the combined vector.
			 *
			 * @param[in,out] combinedVector This will act as the left argument to
			 *                               successive calls to grb::foldl.
			 * @param[in]     query          The IDs of the input vectors.
			 * @param[in]     op             The operator for use with grb::foldl.
			 *
			 * The \a combinedVector must be a dense vector.
			 *
			 * Words that do not appear in the dictionary will result in an error
			 * message printed to std::err, but otherwise it will be as though the
			 * query did not contain the word in question.
			 *
			 * @returns MISMATCH If the dimension of \a combinedVector does not match
			 *                   the dimensions of the vectors stored in this map. It
			 *                   shall then be as though this call was never made.
			 * @returns ILLEGAL  If vector \a combinedVector was sparse on input. It
			 *                   shall then be as though this call was never made.
			 * @returns SUCCESS  On successful exection of this function.
			 *
			 * If an error other than the above occurs, this will result in a message
			 * printed to std::err and a premature exit of the function. The
			 * appropriate grb::RC error code is then returned. See grb::foldl for
			 * such error conditions.
			 *
			 * \note This function works regardless of the \a mode chosen
			 *       during this instance's construction.
			 *
			 * \warning \a noexcept is actually only guaranteed when the user has not
			 *          overriden STL's IOStreams to throw exceptions instea dof setting
			 *          fail bits.
			 */
			template< descriptors::Descriptor descr = descriptors::no_operation, class Operator, class OutputType >
			RC foldlSentence( grb::Vector< OutputType > & combinedVector, const std::vector< size_t > & query, const Operator & op ) const noexcept {
				if( size( combinedVector ) != dimension ) {
					return MISMATCH;
				}
				if( nnz( combinedVector ) < size( combinedVector ) ) {
					return ILLEGAL;
				}
				for( auto id : query ) {
					const RC rc = grb::foldl< descr >( combinedVector, word2vec[ index ], op );
					if( rc != SUCCESS ) {
						std::cerr << "Error while accumulating word vector: " + grb::toString( rc ) + "\n";
						return rc;
					}
				}
				return SUCCESS;
			}

			/** Base destructor. */
			~IndexedVectorMap() noexcept {}
		};

	} // end namespace utils
} // end namespace grb

#endif // end _H_GRB_UTILS_INDEXEDVECTORMAP
