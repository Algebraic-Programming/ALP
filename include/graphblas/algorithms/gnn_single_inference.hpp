
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

#ifndef _H_GRB_ALGORITHMS_GNN_SINGLE_INFERENCE
#define _H_GRB_ALGORITHMS_GNN_SINGLE_INFERENCE

#include <graphblas.hpp>

namespace grb {

	namespace algorithms {

		/**
		 * Performs an inference step of a single data element through a Graph Neural
		 * Network defined by \a num_layers sparse weight matrices and \a num_layers
		 * biases.
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
		 *   F. Pawlowski, R. H. Bisseling, B. Uçar, and A. N. Yzelman
		 *   2020 IEEE High Performance Extreme Computing (HPEC) Conference
		 *
		 * @param[out] out The result of inference through the neural network.
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
		 */
		template< Descriptor descr = descriptors::no_operation,
			typename IOType,
			typename WeightType,
			typename BiasType,
			class MinMonoid = Monoid< grb::operators::min< IOType >, grb::identities::infinity >,
			class ReluMonoid = Monoid< grb::operators::relu< IOType >, grb::identities::negative_infinity >,
			class Ring = Semiring< grb::operators::add< IOType >, grb::operators::mul< IOType >, grb::identities::zero, grb::identities::one > >
		grb::RC gnn_single_inference( grb::Vector< IOType > & out,
			const grb::Vector< IOType > & in,
			grb::Matrix< WeightType > ** const layers,
			const BiasType * const biases,
			const size_t num_layers,
			grb::Vector< IOType > & temp,
			const ReluMonoid & relu = ReluMonoid(),
			const MinMonoid & min = MinMonoid(),
			const Ring & ring = Ring() ) {
			static_assert( ! ( descr & descriptors::no_casting ) || ( std::is_same< IOType, WeightType >::value && std::is_same< IOType, BiasType >::value ),
				"Input containers have different domains even though the no_casting descriptor was given" );

			// run-time checks:
			if( num_layers == 0 || layers == NULL || biases == NULL ) {
				return ILLEGAL;
			}
			if( grb::size( in ) != grb::nrows( *( layers[ 0 ] ) ) || grb::size( out ) != grb::ncols( *( layers[ num_layers - 1 ] ) ) || grb::size( out ) != grb::size( temp ) ) {
				return MISMATCH;
			}
			for( size_t i = 1; i < num_layers; ++i ) {
				if( grb::ncols( *( layers[ i - 1 ] ) ) != grb::nrows( *( layers[ i ] ) ) ) {
					return MISMATCH;
				}
			}
			for( size_t i = 0; i < num_layers; ++i ) {
				if( grb::ncols( *( layers[ i ] ) ) != grb::nrows( *( layers[ i ] ) ) ) {
					return ILLEGAL;
				}
			}

			grb::RC ret = SUCCESS;

			// this is a correct implementation that does not unroll the first and the last iterations
			// we do not use it because it requires setting the input vector to the output vector
			// which results in copying data for 2*n elements
			/*
			ret = grb::set( out, in );
			assert( ret == SUCCESS );

			for( size_t i = 1; ret == SUCCESS && i < num_layers ; ++i ) {

			    ret = grb::clear( temp );
			    assert( ret == SUCCESS );

			    if( ret == SUCCESS ) {
			        ret = grb::vxm( temp, out, *(layers[ i - 1 ]), ring );
			        assert( ret == SUCCESS );
			    }

			    if( ret == SUCCESS ) {
			        ret = foldl(out, 32, min);
			        assert( ret == SUCCESS );
			    }

			    std::swap( out, temp );

			    if( ret == SUCCESS ) {
			        ret = foldl( out, biases[ i ], ring.getAdditiveMonoid() );
			        assert( ret == SUCCESS );
			    }
			    if( ret == SUCCESS ) {
			        ret = foldl( out, 0, relu );
			        assert( ret == SUCCESS );
			    }

			} */

			ret = grb::clear( out );
			assert( ret == SUCCESS );

			ret = ret ? ret : grb::vxm( out, in, *( layers[ 0 ] ), ring );
			assert( ret == SUCCESS );

			ret = ret ? ret : foldl( out, 32, min );
			assert( ret == SUCCESS );

			for( size_t i = 1; ret == SUCCESS && i < num_layers - 1; ++i ) {

				ret = ret ? ret : foldl< descriptors::dense >( out, biases[ i ], ring.getAdditiveMonoid() );
				assert( ret == SUCCESS );

				ret = ret ? ret : foldl< descriptors::dense >( out, 0, relu );
				assert( ret == SUCCESS );

				ret = ret ? ret : grb::clear( temp );
				assert( ret == SUCCESS );

				ret = ret ? ret : grb::vxm( temp, out, *( layers[ i ] ), ring );
				assert( ret == SUCCESS );

				if( ret == SUCCESS ) {
					std::swap( out, temp );
				}

				ret = ret ? ret : foldl( out, 32, min );
				assert( ret == SUCCESS );
			}

			ret = ret ? ret : foldl< descriptors::dense >( out, biases[ num_layers - 1 ], ring.getAdditiveMonoid() );
			assert( ret == SUCCESS );

			ret = ret ? ret : foldl< descriptors::dense >( out, 0, relu );
			assert( ret == SUCCESS );

			return ret;
		}

	} // namespace algorithms

} // end namespace grb

#endif // end _H_GRB_ALGORITHMS_GNN_SINGLE_INFERENCE
