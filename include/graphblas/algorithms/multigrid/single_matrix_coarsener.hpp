
/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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
 * @file single_matrix_coarsener.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Implementation of a coarsener using the same matrix for both coarsening and prolongation.
 */

#ifndef _H_GRB_ALGORITHMS_HPCG_SINGLE_MATRIX_COARSENER
#define _H_GRB_ALGORITHMS_HPCG_SINGLE_MATRIX_COARSENER

#include <memory>
#include <vector>

#include <graphblas.hpp>

#include "multigrid_data.hpp"

namespace grb {
	namespace algorithms {

		/**
		 * Structure storing the data for the coarsener.
		 */
		template<
			typename IOType,
			typename NonzeroType
		> struct CoarseningData {

			grb::Matrix< NonzeroType > coarsening_matrix; ///< matrix of size #system_size \f$ \times \f$ #finer_size
			///< to coarsen an input vector of size #finer_size into a vector of size #system_size
			grb::Vector< IOType > Ax_finer; ///< finer vector for intermediate computations, of size #finer_size

			/**
			 * Construct a new CoarseningData object by initializing internal data structures.
			 *
			 * @param[in] _finer_size  size of the finer system, i.e. size of external objects \b before coarsening
			 * @param[in] coarser_size size of the current system, i.e. size \b after coarsening
			 */
			CoarseningData(
				size_t _finer_size,
				size_t coarser_size
			) :
				coarsening_matrix( coarser_size, _finer_size ),
				Ax_finer( _finer_size ) {}

			grb::RC init_vectors( IOType zero ) {
				return grb::set( Ax_finer, zero );
			}
		};

		/**
		 * Runner structure, holding the data to coarsen the levels of a multi-grid simulation.
		 *
		 * This coarsener just uses the same matrix to perform the coarsening (via an mxv())
		 * and the prolongation, using it transposed.
		 */
		template<
			class CoarsenerTypes,
			typename TelControllerType,
			Descriptor descr = descriptors::no_operation
		> struct SingleMatrixCoarsener {

			// algebraic types
			using IOType = typename CoarsenerTypes::IOType;
			using NonzeroType = typename CoarsenerTypes::NonzeroType;
			using Ring = typename CoarsenerTypes::Ring;
			using Minus = typename CoarsenerTypes::Minus;

			using MultiGridInputType = MultiGridData< IOType, NonzeroType, TelControllerType >; ///< input data from MG
			using CoarseningDataType = CoarseningData< IOType, NonzeroType >; ///< internal data
			///< with coarsening information

			static_assert( std::is_default_constructible< Ring >::value,
				"cannot construct the Ring with default values" );
			static_assert( std::is_default_constructible< Minus >::value,
				"cannot construct the Minus operator with default values" );

			/**
			 * Data to coarsen each level, from finer to coarser.
			 */
			std::vector< std::unique_ptr< grb::algorithms::CoarseningData< IOType, NonzeroType > > > coarsener_levels;
			Ring ring;
			Minus minus;

			/**
			 * Method required by MultiGridRunner before the recursive call, to coarsen
			 * the residual vector of \p finer (the finer system) into the residual of
			 * \p coarser (the coarser system).
			 */
			inline grb::RC coarsen_residual(
				const MultiGridInputType & finer,
				MultiGridInputType & coarser
			) {
				// first compute the residual
				CoarseningData< IOType, NonzeroType > & coarsener = *coarsener_levels[ finer.level ];
				grb::RC ret = grb::set< descr >( coarsener.Ax_finer, ring.template getZero< IOType >() );
				ret = ret ? ret : grb::mxv< descr >( coarsener.Ax_finer, finer.A, finer.z, ring );

				return ret ? ret : compute_coarsening( finer.r, coarser.r, coarsener );
			}

			/**
			 * Method required by MultiGridRunner after the recursive call, to "prolong" the coarser solution
			 * into the finer solution.
			 */
			inline grb::RC prolong_solution(
				const MultiGridInputType & coarser,
				MultiGridInputType & finer
			) {
				return compute_prolongation( coarser.z, finer.z, *coarsener_levels[ finer.level ] );
			}

		protected:
			/**
			 * computes the coarser residual vector \p CoarseningData.r by coarsening
			 *        \p coarsening_data.Ax_finer - \p r_fine via \p coarsening_data.coarsening_matrix.
			 *
			 * The coarsening information are stored inside \p CoarseningData.
			 *
			 * @param[in] r_fine fine residual vector
			 * @param[out] r_coarse coarse residual vector, the output
			 * @param[in,out] coarsening_data \ref MultiGridData data structure storing the information for coarsening
			 * @return grb::RC::SUCCESS if the algorithm could correctly terminate, the error code of the first
			 *                          unsuccessful operation otherwise
			 */
			grb::RC compute_coarsening(
				const grb::Vector< IOType > & r_fine,
				grb::Vector< IOType > & r_coarse,
				CoarseningData< IOType, NonzeroType > & coarsening_data
			) {
				RC ret = SUCCESS;
				ret = ret ? ret : grb::eWiseApply< descr >( coarsening_data.Ax_finer, r_fine,
					coarsening_data.Ax_finer, minus ); // Ax_finer = r_fine - Ax_finer
				assert( ret == SUCCESS );

				// actual coarsening, from  ncols(*coarsening_data->A) == *coarsening_data->system_size * 8
				// to *coarsening_data->system_size
				ret = ret ? ret : grb::set< descr >( r_coarse, ring.template getZero< IOType >() );
				ret = ret ? ret : grb::mxv< descr >( r_coarse, coarsening_data.coarsening_matrix,
					coarsening_data.Ax_finer, ring ); // r = coarsening_matrix * Ax_finer
				return ret;
			}

			/**
			 * computes the prolongation of the coarser solution \p coarsening_data.z and stores it into
			 * \p z_fine.
			 *
			 * For prolongation, this function uses the matrix \p coarsening_data.coarsening_matrix by transposing it.
			 *
			 * @param[out] z_coarse input solution vector, to be coarsened
			 * @param[out] z_fine the solution vector to store the prolonged solution into
			 * @param[in,out] coarsening_data information for coarsening
			 * @return grb::RC::SUCCESS if the algorithm could correctly terminate, the error code of the first
			 * unsuccessful operation otherwise
			 */
			grb::RC compute_prolongation(
				const grb::Vector< IOType > & z_coarse,
				grb::Vector< IOType > & z_fine, // fine residual
				grb::algorithms::CoarseningData< IOType, NonzeroType > & coarsening_data
			) {
				RC ret = SUCCESS;
				// actual refining, from  *coarsening_data->syztem_size == nrows(*coarsening_data->A) / 8
				// to nrows(z_fine)
				ret = ret ? ret : grb::set< descr >( coarsening_data.Ax_finer,
					ring.template getZero< IOType >() );

				ret = ret ? ret : grb::mxv< descr | grb::descriptors::transpose_matrix >(
					coarsening_data.Ax_finer, coarsening_data.coarsening_matrix, z_coarse, ring );
				assert( ret == SUCCESS );

				ret = ret ? ret : grb::foldl< descr >( z_fine, coarsening_data.Ax_finer,
					ring.getAdditiveMonoid() ); // z_fine += Ax_finer;
				assert( ret == SUCCESS );
				return ret;
			}
		};

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_HPCG_SINGLE_MATRIX_COARSENER
