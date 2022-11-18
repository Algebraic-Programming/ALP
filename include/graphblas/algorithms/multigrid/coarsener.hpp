
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
 * @file hpcg_data.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @brief Implementation of the coarsener of HPCG
 * @date 2022-11-08
 */

#ifndef _H_GRB_ALGORITHMS_HPCG_COARSENER
#define _H_GRB_ALGORITHMS_HPCG_COARSENER

#include <vector>
#include <memory>

#include <graphblas.hpp>

#include "multigrid_data.hpp"

namespace grb {
	namespace algorithms {

		template<
			typename IOType,
			typename NonzeroType
		>
		struct coarsening_data {

			grb::Matrix< NonzeroType > coarsening_matrix; ///< matrix of size #system_size \f$ \times \f$ #finer_size
			///< to coarsen an input vector of size #finer_size into a vector of size #system_size
			grb::Vector< IOType > Ax_finer; ///< finer vector for intermediate computations, of size #finer_size

			/**
			 * @brief Construct a new \c coarsening_data by initializing internal data structures
			 * @param[in] coarser_size size of the current system, i.e. size \b after coarsening
			 * @param[in] _finer_size  size of the finer system, i.e. size of external objects \b before coarsening
			 */
			coarsening_data( size_t _finer_size, size_t coarser_size ) :
				coarsening_matrix( coarser_size, _finer_size ),
				Ax_finer( _finer_size ) {}

			grb::RC zero_temp_vectors() {
				return grb::set( Ax_finer, 0 );
			}
		};

		namespace internal {

			/**
			 * @brief computes the coarser residual vector \p coarsening_data.r by coarsening
			 *        \p coarsening_data.Ax_finer - \p r_fine via \p coarsening_data.coarsening_matrix.
			 *
			 * The coarsening information are stored inside \p coarsening_data.
			 *
			 * @tparam IOType type of result and intermediate vectors used during computation
			 * @tparam NonzeroType type of matrix values
			 * @tparam Ring the ring of algebraic operators zero-values
			 * @tparam Minus the minus operator for subtractions
			 *
			 * @param[in] r_fine fine residual vector
			 * @param[in,out] coarsening_data \ref multigrid_data data structure storing the information for coarsening
			 * @param[in] ring the ring to perform the operations on
			 * @param[in] minus the \f$ - \f$ operator for vector subtractions
			 * @return grb::RC::SUCCESS if the algorithm could correctly terminate, the error code of the first
			 *                          unsuccessful operation otherwise
			 */
			template<
				typename IOType,
				typename NonzeroType,
				class Ring,
				class Minus
			> grb::RC compute_coarsening(
				const grb::Vector< IOType > & r_fine, // fine residual
				grb::Vector< IOType > & r_coarse, // fine residual
				coarsening_data< IOType, NonzeroType > & coarsening_data,
				const Ring & ring,
				const Minus & minus
			) {
				RC ret { SUCCESS };
				// DBG_print_norm( coarsening_data.Ax_finer, "+++ Ax_finer prima" );
				ret = ret ? ret : grb::eWiseApply( coarsening_data.Ax_finer, r_fine, coarsening_data.Ax_finer,
									  minus ); // Ax_finer = r_fine - Ax_finer
				// DBG_print_norm( coarsening_data.Ax_finer, "+++ Ax_finer dopo" );
				assert( ret == SUCCESS );

				// actual coarsening, from  ncols(*coarsening_data->A) == *coarsening_data->system_size * 8
				// to *coarsening_data->system_size
				ret = ret ? ret : grb::set( r_coarse, 0 );
				ret = ret ? ret : grb::mxv< grb::descriptors::dense >( r_coarse, coarsening_data.coarsening_matrix,
					coarsening_data.Ax_finer, ring ); // r = coarsening_matrix * Ax_finer
				// DBG_print_norm( r_coarse, "+++ r_coarse" );
				return ret;
			}

			/**
			 * @brief computes the prolongation of the coarser solution \p coarsening_data.z and stores it into
			 * \p x_fine.
			 *
			 * For prolongation, this function uses the matrix \p coarsening_data.coarsening_matrix by transposing it.
			 *
			 * @tparam IOType type of result and intermediate vectors used during computation
			 * @tparam NonzeroType type of matrix values
			 * @tparam Ring the ring of algebraic operators zero-values
			 *
			 * @param[out] x_fine the solution vector to store the prolonged solution into
			 * @param[in,out] coarsening_data information for coarsening
			 * @param[in] ring the ring to perform the operations on
			 * @return grb::RC::SUCCESS if the algorithm could correctly terminate, the error code of the first
			 * unsuccessful operation otherwise
			 */
			template<
				typename IOType,
				typename NonzeroType,
				class Ring
			> grb::RC compute_prolongation(
				const grb::Vector< IOType > & z_coarse,
				grb::Vector< IOType > & x_fine, // fine residual
				grb::algorithms::coarsening_data< IOType, NonzeroType > & coarsening_data,
				const Ring & ring
			) {
				RC ret { SUCCESS };
				// actual refining, from  *coarsening_data->syztem_size == nrows(*coarsening_data->A) / 8
				// to nrows(x_fine)
				ret = ret ? ret : set( coarsening_data.Ax_finer, 0 );

				ret = ret ? ret : grb::mxv< grb::descriptors::transpose_matrix | grb::descriptors::dense >(
					coarsening_data.Ax_finer, coarsening_data.coarsening_matrix, z_coarse, ring );
				assert( ret == SUCCESS );

				ret = ret ? ret : grb::foldl( x_fine, coarsening_data.Ax_finer, ring.getAdditiveMonoid() ); // x_fine += Ax_finer;
				assert( ret == SUCCESS );
				return ret;
			}

		} // namespace internal

		template<
			typename IOType,
			typename NonzeroType,
			class Ring,
			class Minus
		> struct single_point_coarsener {

			static_assert( std::is_default_constructible< Ring >::value,
				"cannot construct the Ring with default values" );
			static_assert( std::is_default_constructible< Minus >::value,
				"cannot construct the Minus operator with default values" );

			using MultiGridInputType = multigrid_data< IOType, NonzeroType >;

			// default value: override with your own
			std::vector< std::unique_ptr< grb::algorithms::coarsening_data< IOType, NonzeroType > > > coarsener_levels;
			Ring ring;
			Minus minus;


			// single_point_coarsener() = default;

			inline grb::RC coarsen_residual(
				const MultiGridInputType &finer,
				MultiGridInputType &coarser
			) {
				// first compute the residual
				coarsening_data< IOType, NonzeroType > &coarsener = *coarsener_levels[ finer.level ];
				grb::RC ret = grb::set( coarsener.Ax_finer, 0 );
				ret = ret ? ret : grb::mxv< grb::descriptors::dense >( coarsener.Ax_finer, finer.A, finer.z, ring );
				// DBG_print_norm( coarsener.Ax_finer, "temp Axf" );
				return internal::compute_coarsening( finer.r, coarser.r, coarsener, ring, minus );
			}

			inline grb::RC prolong_solution(
				const MultiGridInputType &coarser,
				MultiGridInputType &finer
			) {
				return internal::compute_prolongation( coarser.z, finer.z, *coarsener_levels[ finer.level ], ring );
			}
		};

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_HPCG_COARSENER
