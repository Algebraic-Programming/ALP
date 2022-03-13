
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
			 * \internal
			 * Tresholded and non-tresholded sparse/graph neural network inference.
			 *
			 * @tparam thresholded A compile-time parameter controlling whether the
			 *                     inference shall be thresholded.
			 * \endinternal
			 */
			template<
				Descriptor descr,
				bool thresholded, typename ThresholdType,
				typename IOType, typename WeightType, typename BiasType,
				class ReluMonoid, class Ring, class MinMonoid
			>
			grb::RC sparse_nn_single_inference(
				grb::Vector< IOType > &out,
				const grb::Vector< IOType > &in,
				const std::vector< grb::Matrix< WeightType > > &layers,
				const std::vector< BiasType > &biases,
				const ThresholdType threshold,
				grb::Vector< IOType > &temp,
				const ReluMonoid &relu,
				const MinMonoid &min,
				const Ring &ring
			) {
				static_assert( !(descr & descriptors::no_casting) ||
					(
						std::is_same< IOType, WeightType >::value &&
						std::is_same< IOType, BiasType >::value
					), "Input containers have different domains even though the no_casting"
					"descriptor was given"
				);

				const size_t num_layers = layers.size();

				// run-time checks
				{
					const size_t n = grb::size( out );
					if( num_layers == 0 ) {
						return ILLEGAL;
					}
					if( biases.size() != num_layers ) {
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
					assert( n == grb::size( in ) );
					assert( n == grb::size( temp ) );
					if( grb::capacity( out ) != n ) {
						return ILLEGAL;
					}
					if( grb::capacity( temp ) != n ) {
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

				ret = grb::set( out, 0 ); assert( ret == SUCCESS );

				ret = ret ? ret : grb::vxm( out, in, ( layers[ 0 ] ), ring );
				assert( ret == SUCCESS );

				ret = ret ? ret : grb::foldl< descriptors::dense >(
					out, biases[ 1 ], ring.getAdditiveMonoid()
				); assert( ret == SUCCESS );

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

					ret = ret ? ret : grb::vxm< descriptors::dense >(
						out, temp, ( layers[ i ] ), ring
					); assert( ret == SUCCESS );

					ret = ret ? ret : grb::foldl< descriptors::dense >(
						out, biases[ i + 1 ], ring.getAdditiveMonoid()
					); assert( ret == SUCCESS );
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
		template< Descriptor descr = descriptors::no_operation,
			typename IOType,
			typename WeightType,
			typename BiasType,
			class ReluMonoid = Monoid<
				grb::operators::relu< IOType >,
				grb::identities::negative_infinity
			>,
			class Ring = Semiring<
				grb::operators::add< IOType >, grb::operators::mul< IOType >,
				grb::identities::zero, grb::identities::one
			>
		>
		grb::RC sparse_nn_single_inference(
			grb::Vector< IOType > &out,
			const grb::Vector< IOType > &in,
			const std::vector< grb::Matrix< WeightType > > &layers,
			const std::vector< BiasType > &biases,
			grb::Vector< IOType > &temp,
			const ReluMonoid &relu = ReluMonoid(),
			const Ring &ring = Ring()
		) {
			static_assert( !(descr & descriptors::no_casting) ||
				(
					std::is_same< IOType, WeightType >::value &&
					std::is_same< IOType, BiasType >::value
				), "Input containers have different domains even though the no_casting "
				"descriptor was given" );
			Monoid<
				grb::operators::min< IOType >, grb::identities::infinity
			> dummyTresholdMonoid;
			return internal::sparse_nn_single_inference<
				descr, false, double
			> (
				out, in, layers,
				biases, 0.0,
				temp,
				relu, dummyTresholdMonoid, ring
			);
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
		template< Descriptor descr = descriptors::no_operation,
			typename IOType,
			typename WeightType,
			typename BiasType,
			typename ThresholdType = IOType,
			class MinMonoid = Monoid<
				grb::operators::min< IOType >, grb::identities::infinity
			>,
			class ReluMonoid = Monoid<
				grb::operators::relu< IOType >,
				grb::identities::negative_infinity
			>,
			class Ring = Semiring<
				grb::operators::add< IOType >, grb::operators::mul< IOType >,
				grb::identities::zero, grb::identities::one
			>
		>
		grb::RC sparse_nn_single_inference(
			grb::Vector< IOType > &out,
			const grb::Vector< IOType > &in,
			const std::vector< grb::Matrix< WeightType > > &layers,
			const std::vector< BiasType > &biases,
			const ThresholdType threshold,
			grb::Vector< IOType > &temp,
			const ReluMonoid &relu = ReluMonoid(),
			const MinMonoid &min = MinMonoid(),
			const Ring &ring = Ring()
		) {
			static_assert( !(descr & descriptors::no_casting) ||
				(
					std::is_same< IOType, WeightType >::value &&
					std::is_same< IOType, BiasType >::value
				), "Input containers have different domains even though the no_casting "
				"descriptor was given" );
			return internal::sparse_nn_single_inference<
				descr, true
			> (
				out, in, layers,
				biases, threshold,
				temp,
				relu, min, ring
			);
		}

	} // namespace algorithms

} // end namespace grb

#endif // end _H_GRB_ALGORITHMS_SPARSE_NN_SINGLE_INFERENCE

