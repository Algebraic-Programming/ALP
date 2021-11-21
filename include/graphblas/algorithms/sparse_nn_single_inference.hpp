
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
 * @author Aristeidis Mastoras
 */

#ifndef _H_GRB_ALGORITHMS_SPARSE_NN_SINGLE_INFERENCE
#define _H_GRB_ALGORITHMS_SPARSE_NN_SINGLE_INFERENCE

#include <limits>
#include <graphblas.hpp>

namespace grb {

	namespace algorithms {

		namespace internal {

			/**
			 * \internal Implementation of both tresholded and non-tresholded sparse/graph NN inference.
			 *
			 * @tparam thresholded A compile-time parameter controlling whether the inference shall be
			 *                     thresholded or not.
			 */
			template< Descriptor descr,
				bool thresholded,
				typename IOType,
				typename WeightType,
				typename BiasType,
				typename ThresholdType,
				class MinMonoid,
				class ReluMonoid,
				class Ring
			>
			grb::RC sparse_nn_single_inference( grb::Vector< IOType > &out,
				const grb::Vector< IOType > &in,
				const std::vector< grb::Matrix< WeightType > > &layers,
				const std::vector< BiasType > &biases,
				const ThresholdType threshold,
				const size_t num_layers,
				grb::Vector< IOType > &temp,
				const ReluMonoid &relu,
				const MinMonoid &min,
				const Ring &ring
			) {
				static_assert( ! ( descr & descriptors::no_casting ) ||
					( std::is_same< IOType, WeightType >::value && std::is_same< IOType, BiasType >::value ),
					"Input containers have different domains even though the no_casting descriptor was given"
				);

				// run-time checks:
				if( num_layers == 0 ) {
					return ILLEGAL;
				}
				if( grb::size( in ) != grb::nrows( ( layers[ 0 ] ) ) ||
					grb::size( out ) != grb::ncols( ( layers[ num_layers - 1 ] ) ) ||
					grb::size( out ) != grb::size( temp )
				) {
					return MISMATCH;
				}
				for( size_t i = 1; i < num_layers; ++i ) {
					if( grb::ncols( ( layers[ i - 1 ] ) ) != grb::nrows( ( layers[ i ] ) ) ) {
						return MISMATCH;
					}
				}
				for( size_t i = 0; i < num_layers; ++i ) {
					if( grb::ncols( ( layers[ i ] ) ) != grb::nrows( ( layers[ i ] ) ) ) {
						return ILLEGAL;
					}
				}

				grb::RC ret = SUCCESS;

	/*
				// this is a correct implementation that does not unroll the first and the last iterations
				// we do not use it because it requires setting the input vector to the output vector
				// which results in copying data for 2*n elements

				ret = grb::set( out, in );

				for( size_t i = 1; ret == SUCCESS && i < num_layers ; ++i ) {

					std::swap( out, temp );
					ret = ret ? ret : grb::set( out, 0 );
					ret = ret ? ret : grb::vxm( out, temp, *(layers[ i - 1 ]), ring );
					ret = ret ? ret : grb::foldl< descriptors::dense >( out, biases[ i ], ring.getAdditiveMonoid() );
					ret = ret ? ret : grb::foldl< descriptors::dense >( out, 0, relu );
					if( thresholded ) {
						ret = ret ? ret : grb::foldl< descriptors::dense >( out, threshold, min );
					}
				}
	*/

				ret = grb::set( out, 0 );
				assert( ret == SUCCESS );

				ret = ret ? ret : grb::vxm( out, in, ( layers[ 0 ] ), ring );
				assert( ret == SUCCESS );

				ret = ret ? ret : grb::foldl< descriptors::dense >( out, biases[ 1 ], ring.getAdditiveMonoid() );
				assert( ret == SUCCESS );

				for( size_t i = 1; ret == SUCCESS && i < num_layers - 1; ++i ) {

					ret = ret ? ret : grb::foldl< descriptors::dense >( out, 0, relu );
					assert( ret == SUCCESS );

					if( thresholded ) {
						ret = ret ? ret : grb::foldl< descriptors::dense >( out, threshold, min );
						assert( ret == SUCCESS );
					}

					if( ret == SUCCESS ) {
						std::swap( out, temp );
					}

					ret = grb::set( out, 0 );
					assert( ret == SUCCESS );

					ret = ret ? ret : grb::vxm< descriptors::dense >( out, temp, ( layers[ i ] ), ring );
					assert( ret == SUCCESS );

					ret = ret ? ret : grb::foldl< descriptors::dense >( out, biases[ i + 1 ], ring.getAdditiveMonoid() );
					assert( ret == SUCCESS );
				}

				ret = ret ? ret : grb::foldl< descriptors::dense >( out, 0, relu );
				assert( ret == SUCCESS );

				if( thresholded ) {
					ret = ret ? ret : grb::foldl< descriptors::dense >( out, threshold, min );
					assert( ret == SUCCESS );
				}

				return ret;
			}

		} // end namespace ``grb::internal''

		/**
		 * Performs an inference step of a single data element through a Sparse Neural
		 * Network defined by \a num_layers sparse weight matrices and \a num_layers
		 * biases. The initial single data element may be sparse also, such as common
		 * in Graph Neural Networks.
		 *
		 * Inference here is a repeated sequence of application of a sparse linear
		 * layer, addition of a bias factor, and the application of a ReLU.
		 *
		 * We here have a linear algebraic formulation where the ReLU and the bias
		 * application are jointly applied via a max-operator.
		 *
		 * This formalism follows closely the linear algebraic approach to the
		 * related MIT GraphChallenge problem, such as for example described in
		 *
		 *   Combinatorial Tiling for Sparse Neural Networks
		 *   F. Pawlowski, R. H. Bisseling, B. Uçar and A. N. Yzelman
		 *   2020 IEEE High Performance Extreme Computing (HPEC) Conference
		 *
		 * @param[out] out The result of inference through the neural network.
		 * @param[in]  in  The input vector, may be sparse or dense..
		 * @param[in]  layers \a num_layers pointers to sparse linear layers. Each
		 *                    layer here is assumed to be square and of the same
		 *                    size.
		 * @param[in]  biases An array of \a num_layers bias factors.
		 * @param[in]  num_layers The number of layers.
		 * @param[in]  temp A temporary buffer of matching size to each layer.
		 * @param[in]  relu The non-linear ReLU function to apply element-wise.
		 * @param[in]  min  Operator used for thresholding. Maximum feature value
		 *                  is hard-coded to 32, as per the GraphChallenge.
		 * @param[in]  ring The semiring under which to perform the inference.
		 *
		 * Valid descriptors for this algorithm are:
		 *   -# descriptor::no_casting
		 *
		 * \note This algorithm applies the propagation through layers in-place.
		 *       To facilitate this, only square layers are allowed. Non-square
		 *       layers would require the use of different vectors at every
		 *       layer.
		 *
		 * @returns grb::SUCCESS  If the inference was successful
		 * @returns grb::MISMATCH If the input dimensions do not match
		 * @returns grb::ILLEGAL  If a layer was not square
		 */
		template< Descriptor descr = descriptors::no_operation,
			typename IOType,
			typename WeightType,
			typename BiasType,
			class MinMonoid = Monoid< grb::operators::min< IOType >, grb::identities::infinity >,
			class ReluMonoid = Monoid< grb::operators::relu< IOType >, grb::identities::negative_infinity >,
			class Ring = Semiring< grb::operators::add< IOType >, grb::operators::mul< IOType >, grb::identities::zero, grb::identities::one >
		>
		grb::RC sparse_nn_single_inference( grb::Vector< IOType > &out,
			const grb::Vector< IOType > &in,
			const std::vector< grb::Matrix< WeightType > > &layers,
			const std::vector< BiasType > &biases,
			const size_t num_layers,
			grb::Vector< IOType > &temp,
			const ReluMonoid &relu = ReluMonoid(),
			const MinMonoid &min = MinMonoid(),
			const Ring &ring = Ring()
		) {
			static_assert( ! ( descr & descriptors::no_casting ) ||
				( std::is_same< IOType, WeightType >::value && std::is_same< IOType, BiasType >::value ),
				"Input containers have different domains even though the no_casting descriptor was given" );

			return internal::sparse_nn_single_inference< descr, false,
				IOType, WeightType, BiasType, IOType,
				MinMonoid, ReluMonoid, Ring
			> (
				out, in, layers, biases,
				0.0, num_layers, temp,
				relu, min, ring
			);
		}

		/**
		 * Performs an inference step of a single data element through a Sparse Neural
		 * Network defined by \a num_layers sparse weight matrices and \a num_layers
		 * biases. The initial single data element may be sparse also, such as common
		 * in Graph Neural Networks.
		 *
		 * Inference here is a repeated sequence of application of a sparse linear
		 * layer, addition of a bias factor, and the application of a ReLU.
		 *
		 * We here have a linear algebraic formulation where the ReLU and the bias
		 * application are jointly applied via a max-operator.
		 *
		 * This formalism follows closely the linear algebraic approach to the
		 * related MIT GraphChallenge problem, such as for example described in
		 *
		 *   Combinatorial Tiling for Sparse Neural Networks
		 *   F. Pawlowski, R. H. Bisseling, B. Uçar and A. N. Yzelman
		 *   2020 IEEE High Performance Extreme Computing (HPEC) Conference
		 *
		 * @param[out] out The result of inference through the neural network.
		 * @param[in]  in  The input vector, may be sparse or dense..
		 * @param[in]  layers \a num_layers pointers to sparse linear layers. Each
		 *                    layer here is assumed to be square and of the same
		 *                    size.
		 * @param[in]  biases An array of \a num_layers bias factors.
		 * @param[in]  threshold The value used for thresholding.
		 * @param[in]  num_layers The number of layers.
		 * @param[in]  temp A temporary buffer of matching size to each layer.
		 * @param[in]  relu The non-linear ReLU function to apply element-wise.
		 * @param[in]  min  Operator used for thresholding. Maximum feature value
		 *                  is hard-coded to 32, as per the GraphChallenge.
		 * @param[in]  ring The semiring under which to perform the inference.
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
		 * @returns grb::SUCCESS  If the inference was successful
		 * @returns grb::MISMATCH If the input dimensions do not match
		 * @returns grb::ILLEGAL  If a layer was not square
		 */
		template< Descriptor descr = descriptors::no_operation,
			typename IOType,
			typename WeightType,
			typename BiasType,
			typename ThresholdType = IOType,
			class MinMonoid = Monoid< grb::operators::min< IOType >, grb::identities::infinity >,
			class ReluMonoid = Monoid< grb::operators::relu< IOType >, grb::identities::negative_infinity >,
			class Ring = Semiring< grb::operators::add< IOType >, grb::operators::mul< IOType >, grb::identities::zero, grb::identities::one >
		>
		grb::RC sparse_nn_single_inference( grb::Vector< IOType > &out,
			const grb::Vector< IOType > &in,
			const std::vector< grb::Matrix< WeightType > > &layers,
			const std::vector< BiasType > &biases,
			const ThresholdType threshold,
			const size_t num_layers,
			grb::Vector< IOType > &temp,
			const ReluMonoid &relu = ReluMonoid(),
			const MinMonoid &min = MinMonoid(),
			const Ring &ring = Ring()
		) {
			static_assert( ! ( descr & descriptors::no_casting ) ||
				( std::is_same< IOType, WeightType >::value && std::is_same< IOType, BiasType >::value ),
				"Input containers have different domains even though the no_casting descriptor was given" );

			return internal::sparse_nn_single_inference< descr, true,
				IOType, WeightType, BiasType, ThresholdType,
				MinMonoid, ReluMonoid, Ring
			> ( out, in, layers, biases,
				threshold, num_layers, temp,
				relu, min, ring
			);
		}

	} // namespace algorithms

} // end namespace grb

#endif // end _H_GRB_ALGORITHMS_SPARSE_NN_SINGLE_INFERENCE

