
#ifndef _H_GRB_KCORE_DECOMPOSITION
#define _H_GRB_KCORE_DECOMPOSITION

//#define GRB_NO_NOOP_CHECKS

#include <graphblas.hpp>


namespace grb {

	namespace algorithms {

		/**
		 * The \f$ k \f$-core decomposition algorithm.
		 *
		 * Divides the input matrix into subgraphs with a coreness level. The coreness
		 * level \f$ k \f$ is defined as the largest subgraph in which each node has at
		 * least \f$ k \f$ neighbors in the subgraph.
		 *
		 * @tparam IOType   The value type of the \f$ k \f$-core vectors,
		 *                  usually an integer type.
		 * @tparam NZType   The type of the nonzero elements in the matrix.
		 *
		 * @param[in] A     Matrix representing a graph with nonzero value at
		 *                  \f$ (i, j) \f$ an edge between node \f$ i \f$ and
		 *                  \f$ j \f$.
		 * @param[out] core Empty vector of size and capacity \f$ n \f$. On
		 *                  output, if #grb::SUCCESS is returned, stores the
		 *                  result of each nodes coreness level. Coreness level
		 *                  of node \f$ i \f$ is \f$ core[i] \f$.
		 * @param[out] k    The number of coreness lever that was found in the
		 *                  graph.
		 *
		 *  To operate, this algorithm requires a workspace of four vectors. The
		 *  size \em and capacities of these must equal \f$ n \f$. The contents on
		 *  input are ignored, and the contents on output are undefined. The work space
		 *  consists of the buffer vectors \a distances, \a temp, \a update, and
		 *  \a status.
		 *
		 * @param[in,out] distances Distance buffer
		 * @param[in,out] temp      First node update buffer
		 * @param[in,out] update    Second node update buffer
		 * @param[in,out] status    Finished/unfinished buffer
		 *
		 * @returns #grb::SUCCESS       If the coreness for all nodes are found.
		 * @returns #grb::ILLEGAL       If \a A is not square. All outputs are left
		 *                              untouched.
		 * @returns #grb::MISMATCH      If the dimensions of \a core or any of the buffer
		 *                              vectors does not match \a A. All outputs are left
		 *                              untouched.
		 * @returns #grb::ILLEGAL       If the capacity of one or more of \a core and the buffer
		 *                              vectors is less than \f$ n \f$.
		 * @returns #grb::PANIC         If an unrecoverable error has been encountered. The
		 *                              output as well as the state of ALP/GraphBLAS is
		 *                              undefined.
		 *
		 * \note For undirected, unweighted graphs, use pattern matrix for \a A;
		 *       i.e., use \a NZtype <tt>void</tt>
		 *
		 * \note For unweighted graphs, IOType should be a form of unsigned integer.
		 *       The value of any IOType element will be no more than the maximum
		 *       degree found in the graph \a A.
		 *
		 * \parblock
		 * \par Performance semantics
		 *
		 *   -# This function does not allocate nor free dynamic memory, nor shall it
		 *      make any system calls.
		 *
		 * For additional performance semantics regarding work, inter-process data movement,
		 * intra-process data movement, synchronisations, and memory use, please see
		 * the specification of the ALP primitives this function relies on. These
		 * performance semantics, with the exception of getters such as #grb::nnz, are
		 * specific to the backend selected during compilation.
		 * \endparblock
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename IOType, typename NZType
		>
		RC kcore_decomposition(
			grb::Matrix< NZType > &A,
			grb::Vector< IOType > &core,
			grb::Vector< IOType > &distances,
			grb::Vector< IOType > &temp,
			grb::Vector< IOType > &update,
			grb::Vector< bool >   &status,
			IOType &k
		) {
			// Add constants/expressions
			grb::Semiring<
				grb::operators::add< IOType >, grb::operators::mul< IOType >,
				grb::identities::zero, grb::identities::one
			> ring;

			// Runtime sanity checks
			const size_t n = nrows(A);
			{
				// Verify that A is square
				if( n != ncols( A )){
					return ILLEGAL;
				}
				// Verify sizes of vectors
				if( size( core ) != n ||
					size( distances ) != n ||
					size( temp ) != n ||
					size( update ) != n ||
					size( status ) != n
				) {
					return MISMATCH;
				}
				// Verify capacity
				if( capacity( core ) != n ||
					capacity( distances ) != n ||
					capacity( temp ) != n ||
					capacity( update ) != n ||
					capacity( status ) != n
				) {
					return ILLEGAL;
				}
			}
	
			// Initialise
			size_t current_k = 0; // current coreness level
			
			// Set initial values
			RC ret = grb::SUCCESS;
			ret = ret ? ret : set( temp, static_cast< IOType >( 1 ) );
			ret = ret ? ret : set( distances,  static_cast< IOType >( 0 ) );
			ret = ret ? ret : set( core,  static_cast< IOType >( 0 ) );
			ret = ret ? ret : set( status, true );
			ret = ret ? ret : clear( update );
			assert( ret == SUCCESS );
	
			ret = ret ? ret : grb::mxv< descr | descriptors::dense >(
				distances, A, temp, ring );
			assert( ret == SUCCESS );
	
			if( SUCCESS != ret ) {
				std::cerr << " Initialization of k-core decomposition failed with error "
					<< grb::toString(ret) << "\n";
				return ret;
			}
	
			size_t count = 0;
			while( count < n && SUCCESS == ret ) {
				bool flag = true;
	
				// Update filter to exclude completed nodes
				ret = ret ? ret : set( update, status, status );
	
				while( flag ) {
					flag = false;
					ret = ret ? ret : clear( temp );
	
					// Update nodes in parallel
					ret = ret ? ret : eWiseLambda( [ &, current_k ]( const size_t i ) {
							if( status[ i ] && distances[ i ] <= current_k ) {
								core[i] = current_k;
								// Remove node from checking
								status[ i ] = false;
								// Set update
								flag = true;
								#pragma omp critical
								{
									// Add node index to update neighbouts
									setElement( temp, 1, i );
								}
							}
						}, update,
						status,
						distances,
						core,
						temp
					);
					assert( ret == SUCCESS );
	
					if( flag ) {
						ret = ret ? ret : clear( update );
						// Increase number of nodes completed
						count += nnz( temp );
	
						// Get the neighbours of the updated nodes
						ret = ret ? ret : grb::mxv< descr >( update, A, temp, ring );
						assert( ret == SUCCESS );
	
						// Decrease distances of the neighbours
						ret = ret ? ret : grb::eWiseApply( distances, distances, update,
							operators::subtract<IOType>() );
						assert( ret == SUCCESS );
					}
				}
				(void) ++current_k;
			}
			
			if( SUCCESS != ret ){
				std::cerr << " Excecution of k-core decomposition failed with error "
					<< grb::toString(ret) << "\n";
			}
	
			k = current_k;
	
			return ret;
		}

	} // namespace algorithms

} // namespace grb

#endif // end _H_GRB_KCORE_DECOMPOSITION

