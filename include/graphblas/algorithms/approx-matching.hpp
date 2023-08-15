
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
 * Implements the 3/4 approximation algorithm for weight matching.
 *
 * This is modelled after https://github.com/DavidDieudeBest/ApproximatingMWMGraphBLAS/
 *
 * See also https://studenttheses.uu.nl/bitstream/handle/20.500.12932/44078/Thesis_DaviddeBest.pdf
 *
 * @author A. N. Yzelman
 * @date 15th of August, 2023
 */

#ifndef _H_GRB_ALGORITHMS_34MATCHING
#define _H_GRB_ALGORITHMS_34MATCHING

namespace grb {

	namespace algorithms {

		namespace internal {

			namespace mwm {

				template< typename T >
				RC search1AugsProcedure(
					grb::Matrix< T > &adjacency,
					grb::Matrix< T > &matching,
					grb::Matrix< T > &augmentation,
					const size_t n, const size_t m
				) {
					grb::Vector< T > m_w( n );
					grb::Matrix< T > C( n, n ), AwoM( n, n, m ), G1( n, n ), D1( n, n );
					size_t g1_nvals;
					// grb::RC ret = grb::set( m_w, 0 ); not necessary in ALP
					grb::Monoid< grb::operators::add< T >, grb::identities::zero > plusMonoid;
					grb::RC ret = grb::foldl( m_w, matching, plusMonoid );

					ret = ret ? ret : grb::set<
							grb::descriptors::invert_mask
						>( AwoM, matching, adjacency );
					
					ret = ret ? ret : grb::set( C, AwoM, 0 ); // this is not needed  in C?
					ret = ret ? ret : grb::outer( C, AwoM, m_w, m_w, plusMonoid, plusOp );
					ret = ret ? ret : grb::eWiseApply( G1, AwoM, C, minusOp ); // nonzeroes in AwoM but not in C will not be copied into G1?
					// TODO original code line 53 onwards
				}

				template< typename T >
				RC searchKAugmentations(
					grb::Matrix< T > &adjacency,
					grb::Matrix< T > &matching,
					grb::Matrix< T > &augmentation,
					const size_t k, const size_t n
				) {
					RC ret = grb::SUCCESS;
					if( k == 1 ) {
						ret = search1AugsProcedure( adjacency, matching, augmentation, n );
					} else if( k == 2 ) {
						ret = search2AugsProcedure( adjacency, matching, augmentation, n );
					} else if( k == 3 ) {
						ret = search2AugsProcedure( adjacency, matching, augmentation, n );
					} else {
						std::cerr << "This case should never be triggered\n";
						ret = grb::PANIC;
					}
					return ret;
				}

			} // end namespace grb::algorithms::internal::mwm

		} // end namespace grb::algorithms::internal

		template< typename T > // do all matrices need to have the same element type?
		RC approx34_matching(
			grb::Matrix< T > &adjacency, // const?
			grb::Matrix< T > &matching,
			grb::Matrix< T > &augmentation
		) {
			const size_t n = grb::nrows( adjacency );
			if( n != grb::ncols( adjacency ) ) {
				return MISMATCH;
			}
			if( n != grb::nrows( matching ) || n != grb::nrows( augmentation ) ) {
				return MISMATCH;
			}
			if( n != grb::ncols( matching ) || n != grb::ncols( augmentation ) ) {
				return MISMATCH;
			}
			const size_t m = grb::nnz( adjacency );

			// finish[ 0 ] is probably not used
			bool finish[4] = { true, false, false, false };
			size_t k = 1;

			bool continue_guard = true;
			while( continue_guard ) {
				if( internal::mwm::searchKAugmentations( adjacency, matching, augmentation, k, n, m ) ) {
					finish[ k ] = true;
					(void) ++k;
					if( k == 4 ) { k = 1; }
				} else {
					internal::mwm::flipAugmentations( adjacency, matching, augmentation, mWeight, n, m );
					finish[1] = finish[2] = finish[3] = false;
				}
				continue_guard = !( finish[1] && finish[2] && finish[3] );
			}
			return grb::SUCCESS;
		}
	}

}

#endif

