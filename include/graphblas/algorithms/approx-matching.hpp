
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

				RC buildAwoM(
					grb::Matrix< T > &AwoM,
					const grb::Martix< T > &adjacency,
					const grb::Matrix< T > &matching,
				) {
					grb::RC ret = grb::set<
							grb::descriptors::invert_mask
						>( AwoM, matching, adjacency );
					return ret;
				}

				template< typename T >
				RC search1AugsProcedure(
					grb::Matrix< void > &augmentation,
					const grb::Matrix< T > &adjacency,
					const grb::Matrix< T > &matching,
					const size_t n, const size_t m,
					grb::Vector< T > temp,
					grb::Vector< T > m_w,
					grb::Matrix< T > AwoM,
					grb::Matrix< T > C,
					grb::Matrix< T > G1,
					grb::Matrix< void > D1,
					grb::Matrix< bool > maskM,
				) {
					grb::Semiring<
						grb::operators::add< T >, // any_or
						grb::operators::mul< T >, // left_assign
						grb::identities::zero,
						grb::identities::one
					> plusTimes;
					grb::Monoid<
						grb::operators::add< T >,
						grb::identities::zero
					> plusMonoid;
					grb::operators::add< T > plusOp;
					grb::operators::subtract< T > minusOp;
					grb::operators::greater_than< T > gtOp;
					grb::operators::left_assign_if< T, bool, T > leftAssignIfOp;
					if( grb::size( temp ) != n ) { return MISMATCH; }
					if( grb::size( m_w ) != n ) { return MISMATCH; }
					if( grb::ncols( C ) != n || grb::nrows( C ) != n ) { return MISMATCH; }
					if( grb::ncols( G1 ) != n || grb::nrows( G1 ) != n ) { return MISMATCH; }
					if( grb::ncols( AwoM ) != n || grb::nrows( AwoM ) != n ) { return MISMATCH; }
					if( grb::ncols( D1 ) != n || grb::nrows( D1 ) != n ) { return MISMATCH; }
					if( grb::ncols( maskM ) != n || grb::nrows( maskM ) != n ) { return MISMATCH; }
					if( grb::capacity( temp ) < n ) { return ILLEGAL; }
					if( grb::capacity( m_w ) < n ) { return ILLEGAL; }
					if( grb::capacity( C ) < m ) { return ILLEGAL; }
					if( grb::capacity( G1 ) < m ) { return ILLEGAL; }
					if( grb::capacity( AwoM ) < m ) { return ILLEGAL; }
					if( grb::capacity( D1 ) < m ) { return ILLEGAL; }
					if( grb::capacity( maskM ) < m ) { return ILLEGAL; }
					size_t g1_nvals;
					grb::RC ret = grb::set( m_w, 0 ); // we use dense descriptor later
					ret = ret ? ret : grb::set( temp, 1.0 );
					ret = ret ? ret : grb::mxv< grb::descriptors::dense >( m_w, matching, temp, plusTimes );
					ret = ret ? ret : buildAwoM( AwoM, adjacency, matching );

					ret = ret ? ret : grb::set( C, AwoM, 0 ); // note: this is absolutely necessary bc of minus later
					ret = ret ? ret : grb::outer< grb::descriptors::dense >(
						C, AwoM,
						m_w, m_w,
						plusMonoid, plusOp
					);

					ret = ret ? ret : grb::eWiseApply( G1, AwoM, C, minusOp );
					ret = ret ? ret : grb::eWiseApply( maskM, G1, 0, gtOp );
					ret = ret ? ret : grb::eWiseApply( C, G1, maskM, leftAssignIfOp );
					if( ret != grb::SUCCESS ) { return ret; }

					std::swap( C, G1 );

					g1_nvals = grb::nnz( G1 );
					if( g1_nvals > 0 ) {
						// select one match (highest value on each row, one value per row, deterministic tie breaking)
						ret = grb::clear( temp );
						ret = ret ? ret : grb::foldl( temp, G1, maxMonoid );
						ret = ret ? ret : grb::diag( C, temp ); // this function does not yet exist (See branch #228)
						ret = ret ? ret : grb::mxm( maskM, C, G1,
							orEqualsRing );
						ret = ret ? ret : grb::set( C, maskM, G1 );
						ret = ret ? ret : grb::eWiseLambda(
								[&C](const size_t i, const size_t j, T& val) {
									(void) i;
									val = j;
								},
							C );
						ret = ret ? ret : grb::clear( temp );
						ret = ret ? ret : grb::foldl( temp, C, maxMonoid );

						// warning: abuses m_w as a temp vec
						ret = ret ? ret : grb::set<
								grb::descriptors::use_index
							>( m_w, 0 ); //or use branch #228
						ret = ret ? ret : grb::zip( D1, m_w, temp );

						// filter conflicts (select requited matches)
						ret = ret ? ret : grb::eWiseApply<
								grb::descriptors::transpose_left
							>( augmentations, D1, D1, anyOrOp );
					} else {
						ret = grb::FAILED;
					}

					// done
					return ret;
				}

				template< typename T >
				RC search3AugsProcedure(
					grb::Matrix< void > &augmentation,
					const grb::Matrix< T > &adjacency,
					const grb::Matrix< T > &matching,
					const size_t n, const size_t m,
					grb::Vector< T > temp,
					grb::Vector< T > m_w,
					grb::Matrix< T > AwoM,
					grb::Matrix< T > C,
					grb::Matrix< T > G3,
					grb::Matrix< void > D3,
					grb::Matrix< bool > maskM,
				) {
					grb::Semiring<
						grb::operators::add< T >, // any_or
						grb::operators::mul< T >, // left_assign
						grb::identities::zero,
						grb::identities::one
					> plusTimes;
					grb::Monoid<
						grb::operators::add< T >,
						grb::identities::zero
					> plusMonoid;
					grb::operators::add< T > plusOp;
					grb::operators::subtract< T > minusOp;
					grb::operators::greater_than< T > gtOp;
					grb::operators::left_assign_if< T, bool, T > leftAssignIfOp;
					if( grb::size( temp ) != n ) { return MISMATCH; }
					if( grb::size( m_w ) != n ) { return MISMATCH; }
					if( grb::ncols( C ) != n || grb::nrows( C ) != n ) { return MISMATCH; }
					if( grb::ncols( G1 ) != n || grb::nrows( G1 ) != n ) { return MISMATCH; }
					if( grb::ncols( AwoM ) != n || grb::nrows( AwoM ) != n ) { return MISMATCH; }
					if( grb::ncols( D1 ) != n || grb::nrows( D1 ) != n ) { return MISMATCH; }
					if( grb::ncols( maskM ) != n || grb::nrows( maskM ) != n ) { return MISMATCH; }
					if( grb::capacity( temp ) < n ) { return ILLEGAL; }
					if( grb::capacity( m_w ) < n ) { return ILLEGAL; }
					if( grb::capacity( C ) < m ) { return ILLEGAL; }
					if( grb::capacity( G1 ) < m ) { return ILLEGAL; }
					if( grb::capacity( AwoM ) < m ) { return ILLEGAL; }
					if( grb::capacity( D1 ) < m ) { return ILLEGAL; }
					if( grb::capacity( maskM ) < m ) { return ILLEGAL; }
					size_t g1_nvals;
					grb::RC ret = grb::set( m_w, 0 );
					ret = ret ? ret : grb::set( temp, 1.0 );
					ret = ret ? ret : grb::mxv< grb::descriptors::dense >( m_w, matching, temp, plusTimes );
					ret = ret ? ret : buildAwoM( AwoM, adjacency, matching );

					ret = ret ? ret : grb::set( C, AwoM, 0 ); // note: this is absolutely necessary bc of minus later
					ret = ret ? ret : grb::outer< grb::descriptors::dense >(
						C, AwoM,
						m_w, m_w,
						plusMonoid, plusOp
					);

					ret = ret ? ret : grb::eWiseApply( G1, AwoM, C, minusOp );
					ret = ret ? ret : grb::eWiseApply( maskM, G1, 0, gtOp );
					ret = ret ? ret : grb::eWiseApply( C, G1, maskM, leftAssignIfOp );
					if( ret != grb::SUCCESS ) { return ret; }

					std::swap( C, G1 );

					// maskM in the below could be void (TODO)
					ret = ret ? ret : grb::clear( m_w );
					ret = ret ? ret : grb::mxv( m_w, matching, temp, plusTimes ); // maybe one mxv can be enough
					ret = ret ? ret : grb::clean( maskM );
					ret = ret ? ret : grb::outer( maskM, AwoM, m_w, m_w, plusTimes );
					ret = ret ? ret : grb::set( C, maskM, AwoM );
					if( ret != grb::SUCCESS ) { return ret; }

					std::swap( C, AwoM );

					// TODO continue from line 6, alg 8

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
						ret = search1AugsProcedure( adjacency, matching, augmentation, n ); //arguments!
					} else if( k == 2 ) {
						ret = search2AugsProcedure( adjacency, matching, augmentation, n );
					} else if( k == 3 ) {
						ret = search3AugsProcedure( adjacency, matching, augmentation, n );
					} else {
						std::cerr << "This case should never be triggered\n";
						ret = grb::PANIC;
					}
					return ret;
				}

				template< typename T >
				RC flipAugmentations(
					grb::Matrix< T > &matching,
					const grb::Matrix< T > &adjacency,
					const grb::Matrix< void > &augmentation,
					const size_t n
					grb::Vector< T > m,
					grb::Vector< T > a,
					grb::Vector< T > r,
					grb::Matrix< T > buffer
				) {
					if( grb::nrows( matching ) != n || grb::ncols( matching ) != n ) {
						return ILLEGAL;
					}
					if( grb::nrows( adjacency ) != n || grb::ncols( adjacency ) != n ) {
						return ILLEGAL;
					}
					if( grb::nrows( augmentation ) != n || grb::ncols( augmentation ) != n ) {
						return ILLEGAL;
					}
					if( grb::nrows( buffer ) != n || grb::ncols( buffer ) != n ) {
						return ILLEGAL;
					}
					if( grb::size( m ) != n ) { return ILLEGAL; }
					if( grb::size( a ) != n ) { return ILLEGAL; }
					if( grb::size( r ) != n ) { return ILLEGAL; }
					if( grb::capacity( buffer ) < grb::nnz( matching ) ) { return ILLEGAL; }
					if( grb::capacity( m ) < n ) { return ILLEGAL; }
					if( grb::capacity( a ) < n ) { return ILLEGAL; }
					if( grb::capacity( r ) < n ) { return ILLEGAL; }
					grb::operators::any_or< T > anyOrOp; // if T void is sufficient, then this lcould just be logical_or
					grb::operators::logical_or< T, bool, bool > lorOp;
					grb::Semiring<
						grb::operators::logical_or< bool >,
						grb::operators::logical_and< bool >,
						grb::identities::logical_false,
						grb::identities::logical_true
					> booleanSemiring;
					grb::RC ret = grb::foldl( m, matching, anyOrOp );
					ret = ret ? ret : grb::set( r, true );
					ret = ret ? ret : grb::mxv( a, augmentation, r, booleanSemiring );
					ret = ret ? ret : grb::eWiseApply( r, m, a, anyOrOp );
					// now we have the conflicts in r

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
						>( buffer, R, matching );
					if( ret != grb::SUCCESS ) { return ret; }

					std::swap( buffer, matching );
					ret = ret ? ret : grb::eWiseMul(
							matching, augmentation, adjacency,
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

