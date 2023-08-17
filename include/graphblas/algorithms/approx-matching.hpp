
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
					grb::operators::subtract< T > minusOp;
					grb::RC ret = grb::foldl( m_w, matching, plusMonoid );

					ret = ret ? ret : grb::set<
							grb::descriptors::invert_mask
						>( AwoM, matching, adjacency );
					
					//ret = ret ? ret : grb::set( C, AwoM, 0 ); // this is not needed  in C?
					ret = ret ? ret : grb::outer( C, AwoM, m_w, m_w, plusMonoid, plusOp );

					ret = ret ? ret : grb::eWiseApply( G1, AwoM, C, minusOp ); // nonzeroes in AwoM but not in C will not be copied into G1?
					grb::Matrix< bool > tmp( n, n, grb::nnz(G1) ); // ideally best factored out
					grb::Matrix< T > tmp2( n, n ); // default cap is ok
					ret = ret ? ret : grb::eWiseApply( tmp, G1, 0, gtOp );
					ret = ret ? ret : grb::eWiseApply( tmp2, G1, tmp, leftAssignIfOp ); // check this one, ideally I'd use foldl
					std::swap( tmp2, G1 );
					if( ret != grb::SUCCESS ) { return ret; }

					g1_nvals = grb::nnz( G1 );
					if( g1_nvals > 0 ) {
						// select one match (highest value on each row, one value per row, deterministic tie breaking)
						grb::Vector< T > tmp_vec( n ); // ideally factor this out
						grb::Vector< T > tmp_vec2( n ); // ideally factor this out
						grb::Matrix< T > tmp3( n, n, grb::nnz( G1 ) ); // ideally factor out
						ret = grb::foldl( tmp_vec, G1, maxMonoid );
						ret = ret ? ret : grb::diag( tmp2, tmp_vec ); // this function does not yet exist (See branch #228)
						ret = ret ? ret : grb::mxm( tmp3, tmp2, G1,
							orEqualsRing );
						ret = ret ? ret : grb::eWiseLambda( [&tmp3](const size_t i, const size_t j, T& val) { (void) i; val = j; } );
						ret = ret ? ret : grb::clear( tmp_vec );
						ret = ret ? ret : grb::foldl( tmp_vec, tmp3, maxMonoid );
						ret = ret ? ret : grb::set<
								grb::descriptors::use_index
							>( tmp_vec2, 0 ); //or use branch #228
						ret = ret ? ret : grb::zip( D1, tmp_vec2, tmp_vec ); // TODO: D1, augmentation matrices should be pattern matrices?

						// filter conflicts (is this not better named select requited matches?)
						ret = ret ? ret : grb::eWiseApply<
								grb::descriptors::transpose_left
							>( augmentations, D1, D1, anyOrOp );
						// line 68 in orig code not necessary
					} else {
						ret = grb::FAILED;
					}

					// done
					return ret;
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

				template< typename T >
				RC flipAugmentations(
					grb::Matrix< T > &adjacency,
					grb::Matrix< T > &matching,
					grb::Matrix< T > &augmentation,
					double &currentMatchingWeight,
					const size_t n
				) {
					grb::Vector< T > m( n ), a( n ), r( n ); // ideally factor this out
					grb::Matrix< T > buffer( n, n, grb::nnz( matching ) );
					grb::operators::any_or< T > anyOrOp; // if T void is sufficient, then this lcould just be logical_or
					grb::operators::logical_or< T, bool, bool > lorOp;
					grb::Semiring<
						grb::operators::logical_or< bool >,
						grb::operators::logical_and< bool >,
						grb::identities::logical_false,
						grb::identities::logical_true
					> booleanSemiring;
					grb::RC ret = grb::foldl( m, matching, anyOrOp );
					ret = ret ? ret : grb::foldl( a, augmentation, anyOrOp );
					ret = ret ? ret : grb::eWiseApply( r, m, a, anyOrOp );
					ret = ret ? ret : grb::outer(
							R, matching,
							r, 1,
							booleanSemiring
						);
					ret = ret ? ret : grb::foldl<
							grb::descriptors::transpose
						>( R, R, lorOp );

					ret = ret ? ret : grb::set<
							grb::descriptors::invert_mask
						> ( buffer, R, matching );
					std::swap( buffer, matching );
					ret = ret ? ret : grb::eWiseMul(
							matching, R, augmentation, adjacency,
							booleanSemiring
						);

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

