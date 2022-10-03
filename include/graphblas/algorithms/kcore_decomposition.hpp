
#ifndef _H_GRB_KCORE_DECOMPOSITION
#define _H_GRB_KCORE_DECOMPOSITION

//#define GRB_NO_NOOP_CHECKS

#include <graphblas.hpp>

namespace grb {

	namespace algorithms {

        /**
         * @brief   Performs the algorithm k-core decomposition. Divides the input matrix into
         *          subgraphs with a coreness level. The coreness level k is defined as the largest
         *          subgraph in which each node has at least k neighbors in the subgraph.
         * 
         * @tparam IOType       The value type of the kcore vectors, usually a type of integer
         * @tparam NZType       The type of the nonzero elements in the matrix.
         * 
         * @param[in] A         Matrix representing a graph with nonzero value at \f$ i,j \f$ is an edge
         *                      between node \f$ i \f$ and \f$ j \f$. 
         * @param[out] core     Empty vector of size and capacity \f$ n \f$. On output, if #grb::SUCCESS is returned, 
         *                      stores the result of each nodes coreness level. Coreness level of node \f$ i \f$ is \f$ 
         *                      core[i] \f$.
         * @param[out] k        The number of coreness lever that was found in the graph.
         *
         *  To operate, this algorithm requires a workspace of four vectors. The size
		 * \em and capacities of these must equal \f$ n \f$. The contents on input are
		 * ignored, and the contents on output are undefined. These are refered to as 
         * the buffer vectors ( \a distances, \a temp, \a update, and \a status ).
         * 
         * @param[in,out] distances     Buffer for the algorithm used for distances
         * @param[in,out] temp          Buffer for the algorithm used for updated nodes
         * @param[in,out] update        Buffer for the algorithm used for nodes to be updated
         * @param[in,out] status        Buffer for the algorithm used for the status of finished/unfinished nodes 
         * 
         * @returns #grb::SUCCESS       If the coreness for all nodes are found
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
         * \par Performance semantics
		 *
		 *   -# This function does not allocate nor free dynamic memory, nor shall it
		 *      make any system calls.
         *   -# For undirected, unweighted graphs, use pattern matrix for \a A, i.e.
         *      use NZtype void
         *   -# For unweighted graphs, IOType should be a form of unsigned integer. 
         *      The value of any IOType element will be no more than the maximum degree 
         *      found in the graph \a A.
		 *
		 * For additional performance semantics regarding work, inter-process data movement,
		 * intra-process data movement, synchronisations, and memory use, please see
		 * the specification of the ALP primitives this function relies on. These
		 * performance semantics, with the exception of getters such as #grb::nnz, are
		 * specific to the backend selected during compilation.
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
            IOType* k
        ){  
            //Add constants/expressions
            grb::Semiring<
				grb::operators::add< IOType >, grb::operators::mul< IOType >,
				grb::identities::zero, grb::identities::one
			> ring;    
            //Runtime sanity checks
            const size_t n = nrows(A);

            {
                //Verify A is square
                if( n != ncols( A )){
                    return ILLEGAL;
                }
                //Verify sizes of vectors
                if( size( core ) != n ||
                    size( distances ) != n ||
                    size( temp ) != n ||
                    size( update ) != n ||
                    size( status ) != n 
                ) {
                    return MISMATCH;
                }
                //Verify capacity
                if( capacity( core ) != n ||
                    capacity( distances ) != n ||
                    capacity( temp ) != n ||
                    capacity( update ) != n ||
                    capacity( status ) != n 
                ) {
                    return ILLEGAL;
                }
            }


            //Initializing
            size_t current_k = 0; //coreness level
            
            //Set initial values 
            RC ret = grb::SUCCESS;
            ret = ret ? ret : set(temp, static_cast< IOType >( 1 ));
            ret = ret ? ret : set(distances,  static_cast< IOType >( 0 ));
            ret = ret ? ret : set(core,  static_cast< IOType >( 0 ));  
            ret = ret ? ret : set(status, true);            
            clear(update);
            assert( ret == SUCCESS );

            ret = ret ? ret : grb::mxv< descr | descriptors::dense>(distances, A, temp, ring); 
            assert( ret == SUCCESS );

            if(grb::SUCCESS != ret){
                std::cerr << " Initialization of k-core decomposition returned " << grb::toString(ret) << " error\n";
                return ret;
            }

            int count = 0;
            while(count < n && grb::SUCCESS == ret){
                bool flag = true;

                //Update filter to exclude completed nodes
                ret = ret ? ret : set(update, status, status);

                while (flag)
                {
                    flag = false;
                    clear(temp);

                    //Update nodes in parallel
                    ret = ret ? ret : eWiseLambda(
                            [ &, current_k ]( const size_t i ){
                                 if(status[i] && distances[i] <= current_k){
                                    core[i] = current_k;
                                    //Remove node from checking
                                    status[i] = false;
                                    //Set update
                                    flag = true;
                                    #pragma omp critical
                                    {   
                                        //Add node index to update neighbouts
                                        setElement(temp, 1, i);
                                    }
                                 }
                            },
                            update,
                            status,
                            distances,
                            core,
                            temp
                        );
                    assert( ret == SUCCESS );
                 

                    if(flag){
                        clear(update);
                        //Increase number of nodes completed
                        count += nnz(temp);

                        //Get the neighbours of the updated nodes 
                        ret = ret ? ret : grb::mxv< descr >(update, A, temp, ring);
                        assert( ret == SUCCESS );

                        //Decrease distances of the neighbours
                        ret = ret ? ret : grb::eWiseApply(distances, distances, update, operators::subtract<IOType>());
           				assert( ret == SUCCESS );

                    }

                }
                current_k++;
            }
            
            if(grb::SUCCESS != ret){
                std::cerr << " Excecution of k-core decomposition returned " << grb::toString(ret) << " error\n";
            }

            *k = current_k;

            return ret;
        }


	} // namespace algorithms

} // namespace grb

#endif // end _H_GRB_KCORE_DECOMPOSITION
