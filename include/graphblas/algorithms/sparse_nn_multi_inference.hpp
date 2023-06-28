
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
 * Implements (non-batched) sparse neural network multi-inference.
 *
 * @authors
 * 		- Aristeidis Mastoras
 *		- Benjamin Lozes
 */

#ifndef _H_GRB_ALGORITHMS_SPARSE_NN_MULTI_INFERENCE
#define _H_GRB_ALGORITHMS_SPARSE_NN_MULTI_INFERENCE

#include <limits>
#include <vector>

#include <graphblas.hpp>

constexpr bool _Debug = false;

namespace grb {

	namespace algorithms {

		namespace internal {

			template< typename D, class Iterator >
			void printSparseMatrixIterator( size_t rows, size_t cols, Iterator begin, Iterator end, const std::string & name = "", std::ostream & os = std::cout ) {
				std::cout << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
				const size_t limit = 20;
				os.precision( 3 );
				for( size_t y = 0; y < rows; y++ ) {
					if( y > limit )
						break;
					if( y >= limit ) {
						os << "   ...";
					} else {
						os << std::string( 3, ' ' );
						for( size_t x = 0; x < cols; x++ ) {
							if( x >= limit ) {
								os << " ...";
								break;
							}

							auto found_value = std::find_if( begin, end, [ y, x ]( const std::pair< std::pair< size_t, size_t >, D > & e ) {
								return e.first.first == y && e.first.second == x;
							} );
							if( found_value != end )
								os << std::scientific << found_value->second;
							else
								os << "__________";
							os << " ";
						}
					}
					os << std::endl;
				}

				os << "]" << std::endl;
			}

			template< bool Debug = _Debug, typename D >
			void printSparseMatrix( const grb::Matrix< D > & mat, const std::string & name ) {
				if( ! Debug )
					return;
				grb::wait( mat );
				printSparseMatrixIterator< D >( grb::nrows( mat ), grb::ncols( mat ), mat.cbegin(), mat.cend(), name, std::cout );
			}

			/**
			 * \internal
			 * Tresholded and non-tresholded sparse/graph neural network inference.
			 *
			 * @tparam thresholded A compile-time parameter controlling whether the
			 *                     inference shall be thresholded.
			 * \endinternal
			 */
			template< Descriptor descr = grb::descriptors::no_operation,
				bool thresholded,
				typename ThresholdType,
				typename IOType,
				typename WeightType,
				typename BiasType,
				class MinMonoid = Monoid< grb::operators::min< IOType >, grb::identities::infinity >,
				class ReluMonoid = Monoid< grb::operators::relu< IOType >, grb::identities::negative_infinity >,
				class Ring = Semiring< grb::operators::add< IOType >, grb::operators::mul< IOType >, grb::identities::zero, grb::identities::one > >
			grb::RC sparse_nn_multi_inference( grb::Matrix< IOType > & Y_out,
				const grb::Matrix< IOType > & Y_in,
				const std::vector< grb::Matrix< WeightType > > & layers,
				const std::vector< std::vector< BiasType > > & biases,
				const ThresholdType threshold,
				Matrix< IOType > & temp,
				const ReluMonoid & relu = ReluMonoid(),
				const MinMonoid & min = MinMonoid(),
				const Ring & semiring = Ring() ) {
				static_assert( ! ( descr & descriptors::no_casting ) || ( std::is_same< IOType, WeightType >::value && std::is_same< IOType, BiasType >::value ),
					"Input containers have different domains even though the no_casting"
					"descriptor was given" );

				assert( grb::nrows( Y_in ) == grb::nrows( Y_out ) );
				assert( grb::ncols( Y_in ) == grb::ncols( Y_out ) );
				assert( layers.size() == biases.size() );

				grb::RC rc = SUCCESS;
				std::cout.precision( 3 );
				std::cout.setf( std::ios::fixed );
				std::cout.setf( std::ios::showpos );

				std::cout << "Y_out: " << grb::nrows( Y_out ) << "x" << grb::ncols( Y_out ) << std::endl;
				std::cout << "Y_in: " << grb::nrows( Y_in ) << "x" << grb::ncols( Y_in ) << std::endl;
				std::cout << "temp: " << grb::nrows( temp ) << "x" << grb::ncols( temp ) << std::endl;
				//std::cout << "biases.back(): " << grb::nrows( biases.back() ) << "x" << grb::ncols( biases.back() ) << std::endl;
				std::cout << "biases.back(): " << biases.back().size() << std::endl;
				std::cout << "layers.back(): " << grb::nrows( layers.back() ) << "x" << grb::ncols( layers.back() ) << std::endl;

				/*
MATLAB code:

function Y = inferenceReLUvec (W, bias, Y0)
	% Performs ReLU inference using input feature
	% vector(s) Y0, DNN weights W, and constant bias
	Y = Y0 ;
	nlayers = length (W) ;
	% Loop through each weight layer W{layer}
	for layer = 1:nlayers
		% Propagate through layer.
		Z = Y * W{layer} ;
		% Apply bias to non-zero entries.
		Y = Z + (double(logical(Z)) .* bias {layer}) ;
		% Threshold negative values.
		Y (Y < 0) = 0 ;
		% Threshold maximum values.
		Y (Y > 32) = 32 ;
	end
				*/

				// Y_out = Y_in;
				for( size_t l = 0; l < layers.size(); l++ ) {
					std::cout << "  -- Layer " << l << std::endl;

					{ // Y_out = ( l==0 ? Y_in : Y_out ) * layers[l]
						auto Y_out_copy = ( l == 0 ? Y_in : Y_out );
						rc = grb::mxm( Y_out, Y_out_copy, layers[ l ], semiring, grb::Phase::RESIZE );
						if( rc != grb::SUCCESS ) {
							std::cerr << "grb::mxm( Y_out, Y_out_copy, layers[l], semiring, grb::Phase::RESIZE ) failed" << std::endl;
							return rc;
						}
						rc = grb::mxm( Y_out, Y_out_copy, layers[ l ], semiring, grb::Phase::EXECUTE );
						if( rc != grb::SUCCESS ) {
							std::cerr << "grb::mxm( Y_out, Y_out_copy, layers[l], semiring, grb::Phase::EXECUTE ) failed" << std::endl;
							return rc;
						}
						printSparseMatrix( Y_out, "grb::mxm( Y_out, Y_out_copy, layers[l], semiring, grb::Phase::EXECUTE )" );
					}

					{ // Y_out( i, j ) += Bias[ layer ] ( j, j ) for each Y_out( i, j )
						rc = grb::eWiseLambda(
							[ biases, l ]( const size_t i, const size_t j, IOType & y ) {
								if(i == j)
									y += biases[ l ][ i ];
							},
							Y_out );
						if( rc != grb::SUCCESS ) {
							std::cerr << "grb::fold( Y_out, biases[l], add ) failed" << std::endl;
							return rc;
						}
						printSparseMatrix( Y_out, "grb::fold( Y_out, biases[l], add )" );
					}

					{ // Remove entries of Y_out that are negative
						// rc = grb::eWiseLambda(
						// 	[]( const size_t i, const size_t j, IOType & y ) {
						// 		(void)i;
						// 		(void)j;
						// 		y = y >= 0 ? y : 0;
						// 	},
						// 	Y_out );
						rc = foldl( Y_out, static_cast<IOType>(0), grb::operators::max< IOType >() );
						if( rc != grb::SUCCESS ) {
							std::cerr << "grb::fold( Y_out, 0, max ) failed" << std::endl;
							return rc;
						}
						printSparseMatrix( Y_out, "grb::fold( Y_out, 0, max )" );
					}

					if( thresholded ) { // threshold maximum values: Y_out (Y_out > threshold) = threshold
						// rc = grb::eWiseLambda(
						// 	[ threshold ]( const size_t i, const size_t j, IOType & y ) {
						// 		(void)i;
						// 		(void)j;
						// 		y = y <= threshold ? y : threshold;
						// 	},
						// 	Y_out );
						rc = foldl( Y_out, threshold, grb::operators::min< IOType, ThresholdType, IOType >() );
						if( rc != grb::SUCCESS ) {
							std::cerr << "grb::fold( Y_out, threshold, min ) failed" << std::endl;
							return rc;
						}
						printSparseMatrix( Y_out, "grb::fold( Y_out, threshold, min )" );
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
		template< Descriptor descr = grb::descriptors::no_operation, typename IOType, typename WeightType, typename BiasType >
		grb::RC sparse_nn_multi_inference( grb::Matrix< IOType > & Y_out,
			const grb::Matrix< IOType > & Y_in,
			const std::vector< grb::Matrix< WeightType > > & layers,
			const std::vector< std::vector< BiasType > > & biases,
			Matrix< IOType > & temp ) {
			static_assert( ! ( descr & descriptors::no_casting ) || ( std::is_same< IOType, WeightType >::value && std::is_same< IOType, BiasType >::value ),
				"Input containers have different domains even though the no_casting "
				"descriptor was given" );
			return internal::sparse_nn_multi_inference< descr, false, IOType, WeightType, BiasType >( Y_out, Y_in, layers, biases, std::numeric_limits< IOType >::max(), temp );
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
		template< Descriptor descr = grb::descriptors::no_operation, typename ThresholdType, typename IOType, typename WeightType, typename BiasType >
		grb::RC sparse_nn_multi_inference( grb::Matrix< IOType > & Y_out,
			const grb::Matrix< IOType > & Y_in,
			const std::vector< grb::Matrix< WeightType > > & layers,
			const std::vector< std::vector< BiasType > > & biases,
			const ThresholdType threshold,
			Matrix< IOType > & temp ) {
			static_assert( ! ( descr & descriptors::no_casting ) || ( std::is_same< IOType, WeightType >::value && std::is_same< IOType, BiasType >::value ),
				"Input containers have different domains even though the no_casting "
				"descriptor was given" );
			std::cerr << "sparse_nn_multi_inference< descr, true, ThresholdType, IOType, WeightType, BiasType >" << std::endl;
			return internal::sparse_nn_multi_inference< descr, true, ThresholdType, WeightType, BiasType >( Y_out, Y_in, layers, biases, threshold, temp );
		}

	} // namespace algorithms

} // end namespace grb

#endif // end _H_GRB_ALGORITHMS_SPARSE_NN_MULTI_INFERENCE
