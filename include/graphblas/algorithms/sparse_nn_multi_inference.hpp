
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
 * Implements (non-batched) sparse neural network inference.
 *
 * @author Aristeidis Mastoras
 */

#ifndef _H_GRB_ALGORITHMS_SPARSE_NN_MULTI_INFERENCE
#define _H_GRB_ALGORITHMS_SPARSE_NN_MULTI_INFERENCE

#include <limits>
#include <vector>

#include <graphblas.hpp>

namespace grb {

	namespace algorithms {

		namespace internal {

			/**
			 * \internal
			 * Tresholded and non-tresholded sparse/graph neural network inference.
			 *
			 * @tparam thresholded A compile-time parameter controlling whether the
			 *                     inference shall be thresholded.
			 * \endinternal
			 */
			template< Descriptor descr, bool thresholded, typename ThresholdType, typename IOType, typename WeightType, typename BiasType >
			grb::RC sparse_nn_multi_inference( grb::Matrix< IOType > & Y_out,
				const grb::Matrix< IOType > & Y_in,
				const std::vector< grb::Matrix< WeightType > > & W,
				const std::vector< grb::Vector< BiasType > > & biases,
				const ThresholdType threshold ) {
				static_assert( ! ( descr & descriptors::no_casting ) || ( std::is_same< IOType, WeightType >::value && std::is_same< IOType, BiasType >::value ),
					"Input containers have different domains even though the no_casting"
					"descriptor was given" );

				assert( grb::nrows( Y_in ) == grb::nrows( Y_out ) );
				assert( grb::ncols( Y_in ) == grb::ncols( Y_out ) );
				assert( W.size() == biases.size() );

				const grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > semiring;

				grb::RC rc = SUCCESS;
				std::cout.precision( 3 );
				std::cout.setf( std::ios::fixed );
				std::cout.setf( std::ios::showpos );
				for( size_t l = 0; rc == SUCCESS && l < W.size(); ++l ) {
					std::cout << "-- Layer " << l << std::endl;

					// Y = Y * W[l]
					rc = rc ? rc : mxm( Y_out, ( l == 0 ) ? Y_in : Y_out, W[ l ], semiring, grb::Phase::RESIZE );
					assert( ! rc );
					rc = rc ? rc : mxm( Y_out, ( l == 0 ) ? Y_in : Y_out, W[ l ], semiring, grb::Phase::EXECUTE );
					assert( ! rc );

					{
						std::cout << "\tAfter weights -   First 10 nonzeroes of out are: ( ";
						size_t k = 10;
						for( const std::pair< std::pair< size_t, size_t >, double > & e : Y_out ) {
							std::cout << e.second << " ";
							if( --k <= 0 )
								break;
						}
						std::cout << ")" << std::endl;
					}

					// Y(i,j) += biases[l] (j,j) for each Y(i,j)
					rc = rc ? rc :
							  grb::eWiseLambda(
								  [ biases, l ]( const size_t i, const size_t j, IOType & e ) {
									  (void)i;
									  (void)j;
									  e += biases[ l ][ j ];
								  },
								  Y_out );
					assert( ! rc );
					{
						std::cout << "\tAfter biases -    First 10 nonzeroes of out are: ( ";
						size_t k = 10;
						for( const std::pair< std::pair< size_t, size_t >, double > & e : Y_out ) {
							std::cout << e.second << " ";
							if( --k <= 0 )
								break;
						}
						std::cout << ")" << std::endl;
					}

					// Delete strictly negative values
					/** Note: Could be replaced by an eWiseApply(Matrix, Monoid) / eWiseApply(Matrix, scalar, BinaryOp)
					 *  with grb::operators::max
					 */
					rc = rc ? rc :
							  grb::eWiseLambda(
								  []( const size_t i, const size_t j, IOType & e ) {
									  (void)i;
									  (void)j;
									  e = ( e >= 0 ) ? e : static_cast< IOType >( 0 );
								  },
								  Y_out );
					assert( ! rc );
					{
						std::cout << "\tAfter zeroes -    First 10 nonzeroes of out are: ( ";
						size_t k = 10;
						for( const std::pair< std::pair< size_t, size_t >, double > & e : Y_out ) {
							std::cout << e.second << " ";
							if( --k <= 0 )
								break;
						}
						std::cout << ")" << std::endl;
					}

					// Threshold values
					/** Note: Could be replaced by an eWiseApply(Matrix, Monoid) / eWiseApply(Matrix, scalar, BinaryOp)
					 *  with grb::operators::min
					 */
					if( thresholded ) {
						rc = rc ? rc :
								  grb::eWiseLambda(
									  [ threshold ]( const size_t i, const size_t j, IOType & e ) {
										  (void)i;
										  (void)j;
										  if( e > threshold )
											  e = static_cast< IOType >( threshold );
									  },
									  Y_out );
						assert( ! rc );
						{
							std::cout << "\tAfter threshold - First 10 nonzeroes of out are: ( ";
							size_t k = 10;
							for( const std::pair< std::pair< size_t, size_t >, double > & e : Y_out ) {
								std::cout << e.second << " ";
								if( --k <= 0 )
									break;
							}
							std::cout << ")" << std::endl;
						}
					}
				}

				return rc;
			}

		} // namespace internal

		/**
		 * Performs an inference step of a single data element through a Sparse Neural
		 * Network defined by \a num_layers sparse weight matrices and \a num_layers
		 * biases. The initial single data element may be sparse also, such as common
		 * in Graph Neural Networks (GNNs).
		 *
		 * Inference here is a repeated sequence of application of a sparse linear
		 * layer, addition of a bias factor, and the application of a ReLU.
		 *
		 * We employ a linear algebraic formulation where the ReLU and the bias
		 * application are jointly applied via a max-operator.
		 *
		 * This formalism follows closely the linear algebraic approach to the
		 * related IEEE/MIT GraphChallenge problem, such as, for example, described in
		 *
		 *   Combinatorial Tiling for Sparse Neural Networks
		 *   F. Pawlowski, R. H. Bisseling, B. Uçar and A. N. Yzelman
		 *   2020 IEEE High Performance Extreme Computing (HPEC) Conference
		 *
		 * @param[out] out    The result of inference through the neural network.
		 * @param[in]  in     The input vector, may be sparse or dense.
		 * @param[in]  layers A collection of linear layers. Each layer is assumed
		 *                    to be square and of the equal size to one another.
		 *
		 * This implies that all \a layers are \f$ n \times n \f$. The vectors \a in
		 * and \a out hence must be of length \f$ n \f$.
		 *
		 * Commonly, as an input propagates through a network, the features become
		 * increasingly dense. Hence \a out is assumed to have full capacity in order
		 * to potentially store a fully dense activation vector.
		 *
		 * Inference proceeds under a set of biases, one for each layer. Activation
		 * vectors are added a constant bias value prior to applying the given
		 * \a relu function. This function does not perform tresholding.
		 *
		 * @param[in] biases An array of \a num_layers bias factors.
		 *
		 * Inference is done using a single buffer that is alternated with \a out:
		 *
		 * @param[in,out] temp A buffer of size and capacity \f$ n \f$.
		 *
		 * Finally, optional arguments define the algebraic structures under which
		 * inference proceeds:
		 *
		 * @param[in] relu The non-linear ReLU function to apply element-wise.
		 * @param[in] ring The semiring under which to perform the inference.
		 *
		 * The default algebraic structures are standard \a relu (i.e., max), \a min
		 * for tresholding, and the real (semi-) \a ring.
		 *
		 * Valid descriptors for this algorithm are:
		 *   -# descriptor::no_casting
		 *
		 * \note This algorithm applies the propagation through layers in-place.
		 *       To facilitate this, only square layers are allowed. Non-square
		 *       layers would require the use of different vectors at every
		 *       layer.
		 *
		 * @returns #grb::SUCCESS  If the inference was successful
		 * @returns #grb::ILLEGAL  If the size of \a layers does not match that of
		 *                         \a baises.
		 * @returns #grb::MISMATCH If at least one pair of dimensions between
		 *                         \a layers, \a in, \a out, and \a temp do not match.
		 * @returns #grb::ILLEGAL  If at least one layer was not square.
		 * @returns #grb::ILLEGAL  If the capacities of one or more of \a out and
		 *                         \a temp were not full.
		 *
		 * \par Performance semantics
		 *
		 *   -# This function does not allocate nor free dynamic memory, nor shall it
		 *      make any system calls.
		 *
		 * For performance semantics regarding work, inter-process data movement,
		 * intra-process data movement, synchronisations, and memory use, please see
		 * the specification of the ALP primitives this function relies on. These
		 * performance semantics, with the exception of getters such as #grb::nnz, are
		 * specific to the backend selected during compilation.
		 */
		template< Descriptor descr = descriptors::no_operation, typename IOType, typename WeightType, typename BiasType >
		grb::RC sparse_nn_multi_inference( grb::Matrix< IOType > & Y,
			const grb::Matrix< IOType > & Y0,
			const std::vector< grb::Matrix< WeightType > > & W,
			const std::vector< grb::Vector< BiasType > > & biases ) {
			static_assert( ! ( descr & descriptors::no_casting ) || ( std::is_same< IOType, WeightType >::value && std::is_same< IOType, BiasType >::value ),
				"Input containers have different domains even though the no_casting "
				"descriptor was given" );
			Monoid< grb::operators::min< IOType >, grb::identities::infinity > dummyTresholdMonoid;
			return internal::sparse_nn_multi_inference< descr, false, double >( Y, Y0, W, biases, 0.0 );
		}

		/**
		 * Performs an inference step of a single data element through a Sparse Neural
		 * Network defined by \a num_layers sparse weight matrices and \a num_layers
		 * biases. The initial single data element may be sparse also, such as common
		 * in Graph Neural Networks (GNNs).
		 *
		 * Inference here is a repeated sequence of application of a sparse linear
		 * layer, addition of a bias factor, and the application of a ReLU.
		 *
		 * We employ a linear algebraic formulation where the ReLU and the bias
		 * application are jointly applied via a max-operator.
		 *
		 * This formalism follows closely the linear algebraic approach to the
		 * related IEEE/MIT GraphChallenge problem, such as for example described in
		 *
		 *   Combinatorial Tiling for Sparse Neural Networks
		 *   F. Pawlowski, R. H. Bisseling, B. Uçar and A. N. Yzelman
		 *   2020 IEEE High Performance Extreme Computing (HPEC) Conference
		 *
		 * @param[out] out    The result of inference through the neural network.
		 * @param[in]  in     The input vector, may be sparse or dense.
		 * @param[in]  layers A collection of linear layers. Each layer is assumed
		 *                    to be square and of the equal size to one another.
		 *
		 * This implies that all \a layers are \f$ n \times n \f$. The vectors \a in
		 * and \a out hence must be of length \f$ n \f$.
		 *
		 * Commonly, as an input propagates through a network, the features become
		 * increasingly dense. Hence \a out is assumed to have full capacity in order
		 * to potentially store a fully dense activation vector.
		 *
		 * Inference proceeds under a set of biases, one for each layer. Activation
		 * vectors are added a constant bias value prior to applying the given
		 * \a relu function. After application, the resulting vector is furthermore
		 * tresholded. The treshold is assumed constant over all layers.
		 *
		 * @param[in] biases    An array of \a num_layers bias factors.
		 * @param[in] threshold The value used for thresholding.
		 *
		 * Inference is done using a single buffer that is alternated with \a out:
		 *
		 * @param[in,out] temp A buffer of size and capacity \f$ n \f$.
		 *
		 * Finally, optional arguments define the algebraic structures under which
		 * inference proceeds:
		 *
		 * @param[in] relu The non-linear ReLU function to apply element-wise.
		 * @param[in] min  Operator used for thresholding. Maximum feature value
		 *                 is hard-coded to 32, as per the GraphChallenge.
		 * @param[in] ring The semiring under which to perform the inference.
		 *
		 * The default algebraic structures are standard \a relu (i.e., max), \a min
		 * for tresholding, and the real (semi-) \a ring.
		 *
		 * Valid descriptors for this algorithm are:
		 *   -# descriptor::no_casting
		 *
		 * \note This algorithm applies the propagation through layers in-place.
		 *       To facilitate this, only square layers are allowed. Non-square
		 *       layers would require the use of different vectors at every
		 *       layer.
		 *
		 * \note Thresholding here means that feature maps as propagated through
		 *       the neural network are capped at some maximum value, \a threshold.
		 *
		 * @returns #grb::SUCCESS  If the inference was successful
		 * @returns #grb::ILLEGAL  If the size of \a layers does not match that of
		 *                         \a baises.
		 * @returns #grb::MISMATCH If at least one pair of dimensions between
		 *                         \a layers, \a in, \a out, and \a temp do not match.
		 * @returns #grb::ILLEGAL  If at least one layer was not square.
		 * @returns #grb::ILLEGAL  If the capacities of one or more of \a out and
		 *                         \a temp were not full.
		 *
		 * \par Performance semantics
		 *
		 *   -# This function does not allocate nor free dynamic memory, nor shall it
		 *      make any system calls.
		 *
		 * For performance semantics regarding work, inter-process data movement,
		 * intra-process data movement, synchronisations, and memory use, please see
		 * the specification of the ALP primitives this function relies on. These
		 * performance semantics, with the exception of getters such as #grb::nnz, are
		 * specific to the backend selected during compilation.
		 */
		template< Descriptor descr = descriptors::no_operation, typename IOType, typename WeightType, typename BiasType, typename ThresholdType = IOType >
		grb::RC sparse_nn_multi_inference( grb::Matrix< IOType > & Y,
			const grb::Matrix< IOType > & Y0,
			const std::vector< grb::Matrix< WeightType > > & W,
			const std::vector< grb::Vector< BiasType > > & biases,
			const ThresholdType threshold ) {
			static_assert( ! ( descr & descriptors::no_casting ) || ( std::is_same< IOType, WeightType >::value && std::is_same< IOType, BiasType >::value ),
				"Input containers have different domains even though the no_casting "
				"descriptor was given" );
			return internal::sparse_nn_multi_inference< descr, true >( Y, Y0, W, biases, threshold );
		}

	} // namespace algorithms

} // end namespace grb

#endif // end _H_GRB_ALGORITHMS_SPARSE_NN_MULTI_INFERENCE
