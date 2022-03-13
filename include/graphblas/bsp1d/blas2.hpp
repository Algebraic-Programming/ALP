
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
 * Implements BLAS-2 operations for the BSP1D backend.
 *
 * @author: A. N. Yzelman
 * @date: 16th of February, 2017.
 */

#ifndef _H_GRB_BSP1D_BLAS2
#define _H_GRB_BSP1D_BLAS2

#include <graphblas/backends.hpp> //BSP1D
#include <graphblas/base/blas2.hpp>
#include <graphblas/bsp1d/config.hpp>
#include <graphblas/collectives.hpp> //collectives
#include <graphblas/descriptors.hpp>
#include <graphblas/distribution.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/reference/blas2.hpp>
#include <graphblas/semiring.hpp>
#include <graphblas/vector.hpp>

#include "matrix.hpp"

#ifdef _DEBUG
 #include "spmd.hpp"
#endif


namespace grb {

	/**
	 * \addtogroup bsp1d
	 * @{
	 */

	namespace internal {

		template< Descriptor descr,
			bool output_masked,
			bool input_masked,
			bool left_handed,
			class Ring,
			typename IOType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename InputType4,
			typename Coords >
		RC bsp1d_mxv( Vector< IOType, BSP1D, Coords > &u,
			const Vector< InputType3, BSP1D, Coords > &u_mask,
			const Matrix< InputType2, BSP1D > &A,
			const Vector< InputType1, BSP1D, Coords > &v,
			const Vector< InputType4, BSP1D, Coords > &v_mask,
			const Ring &ring
		) {
			// transpose must be handled on higher level
			assert( !( descr & descriptors::transpose_matrix ) );
			// dynamic sanity checks
			if( u._n != A._m || v._n != A._n ) {
				return MISMATCH;
			}
			if( output_masked && u_mask._n != A._m ) {
				return MISMATCH;
			}
			if( input_masked && v_mask._n != A._n ) {
				return MISMATCH;
			}

#ifdef _DEBUG
			const auto s = spmd< BSP1D >::pid();
			std::cout << s << ": bsp1d_mxv called with "
				<< descriptors::toString( descr ) << "\nNow synchronising input vector...";
#endif

			// synchronise the input
			RC rc = v.synchronize();

			// synchronise input mask
			if( input_masked && rc == SUCCESS ) {
#ifdef _DEBUG
				std::cout << "\t " << s << ", bsp1d_mxv: synchronising input mask\n";
#endif
				rc = v_mask.synchronize();
			}

#ifdef _DEBUG
			if( output_masked ) {
				std::cout << "\t " << s << ", bsp1d_mxv: output mask has "
					<< internal::getCoordinates( u_mask._local ).nonzeroes()
					<< " nonzeroes and size "
					<< internal::getCoordinates( u_mask._local ).size() << ":";
				for( size_t k = 0;
					k < internal::getCoordinates( u_mask._local ).nonzeroes();
					++k
				) {
					std::cout << " " << internal::getCoordinates( u_mask._local ).index( k );
				}
				std::cout << "\n";
			}
#endif

			// quit on sync errors
			if( rc != SUCCESS ) {
				return rc;
			}

			// delegate to sequential code
			const auto data = internal::grb_BSP1D.cload();
			const size_t offset = internal::Distribution< BSP1D >::local_offset(
				v._n, data.s, data.P
			);

#ifdef _DEBUG
			std::cout << "\t " << s << ", bsp1d_mxv: " << " calling process-local vxm "
				<< "using allgathered input vector at " << &( v._global ) << " with "
				<< nnz( v._global ) << "/" << size( v._global )
				<< " nonzeroes and an output vector currently holding " << nnz( u._local )
				<< " / " << size( u._local ) << " nonzeroes...\n";
#endif
			rc = internal::vxm_generic<
				descr ^ descriptors::transpose_matrix,
				output_masked, input_masked,
				left_handed, true,
				Ring::template One
			> (
				u._local, u_mask._local, v._global, v_mask._global, A._local,
				ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
				[ &offset ]( const size_t i ) {
					return i + offset;
				},
				[ &offset ]( const size_t i ) {
					return i - offset;
				},
				[]( const size_t i ) {
					return i;
				},
				[]( const size_t i ) {
					return i;
				}
			);

#ifdef _DEBUG
			std::cout << s << ": " << " call to internal::vxm_generic completed, output "
				<< "vector now holds " << nnz( u._local ) << "/" << size( u._local )
				<< " updating nonzeroes...\n";
#endif

			// update nnz since we were communicating anyway
			if( rc == SUCCESS ) {
				u._nnz_is_dirty = true;
				rc = u.updateNnz();
			}

#ifdef _DEBUG
			std::cout << s << "; bsp1d_mxv done!\n";
#endif

			return rc;
		}

		template< Descriptor descr,
			bool output_masked, bool input_masked, bool left_handed,
			class Ring,
			typename IOType, typename InputType1, typename InputType2,
			typename InputType3, typename InputType4,
			typename Coords
		>
		RC bsp1d_vxm( Vector< IOType, BSP1D, Coords > &u,
			const Vector< InputType3, BSP1D, Coords > &u_mask,
			const Vector< InputType1, BSP1D, Coords > &v,
			const Vector< InputType4, BSP1D, Coords > &v_mask,
			const Matrix< InputType2, BSP1D > &A,
			const Ring &ring
		) {
			RC rc = SUCCESS;

			// transpose must be handled on higher level
			assert( !( descr & descriptors::transpose_matrix ) );

			// dynamic sanity checks
			if( u._n != A._n || v._n != A._m ) {
				return MISMATCH;
			}
			if( output_masked && u_mask._n != A._n ) {
				return MISMATCH;
			}
			if( input_masked && v_mask._n != A._m ) {
				return MISMATCH;
			}

			const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();

#ifdef _DEBUG
			const auto s = spmd< BSP1D >::pid();
			std::cout << s << ": bsp1d_vxm called with " << descriptors::toString( descr ) << "\n";
			std::cout << "\t" << s << ", unbuffered BSP1D vxm called\n";
			std::cout << "\t" << s << ", bsp1d_vxm, global output vector currently contains " << internal::getCoordinates( u._global ).nonzeroes() << " / "
				  << internal::getCoordinates( u._global ).size() << " nonzeroes. Nnz_is_dirty equals: " << u._nnz_is_dirty << ".\n";
			if( input_masked ) {
				std::cout << "\t" << s << ", input mask has entries at";
				for( size_t i = 0; i < internal::getCoordinates( v_mask._local ).nonzeroes(); ++i ) {
					std::cout << " " << internal::getCoordinates( v_mask._local ).index( i );
				}
				std::cout << "\n";
			}
#endif

			if( output_masked ) {
#ifdef _DEBUG
				std::cout << "\t" << s << ", bsp1d_vxm: synchronising output mask...\n";
#endif
				rc = u_mask.synchronize();
				if( rc != SUCCESS ) {
					return rc;
				}
			}

			const auto &local_coors = internal::getCoordinates( u._local );
#ifdef _DEBUG
			std::cout << "\t" << s <<
				", 0: calling process-local vxm using global output "
				"vector. Local output vector contains " <<
				local_coors.nonzeroes() << " / " << local_coors.size() << ".\n";
#endif
			// prepare global view of u for use. Only local values should be entries.
			const size_t output_offset = internal::Distribution< BSP1D >::local_offset( u._n, data.s, data.P );
			{
				// assuming a `lazy' clear that does not clear value entries(!)
				auto &global_coors = internal::getCoordinates( u._global );
				global_coors.template rebuildGlobalSparsity< false >( local_coors, output_offset );
			}

#ifdef _DEBUG
			std::cout << "\t" << s << ", bsp1d_vxm: global output vector of the local vxm-to-be currently contains "
				<< internal::getCoordinates( u._global ).nonzeroes() << " / "
				<< internal::getCoordinates( u._global ).size() << " nonzeroes. "
				<< "This is the unbuffered variant.\n";
#endif

			// even if the global operation is totally dense, the process-local vxm may generate sparse output
			// thus construct a local descriptor that strips away any dense hint
			constexpr Descriptor local_descr = descr & (~(descriptors::dense));

			// delegate to process-local vxm
			internal::vxm_generic< local_descr, output_masked, input_masked, left_handed, true, Ring::template One >(
				u._global, u_mask._global, v._local, v_mask._local, A._local, ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
				[ &output_offset ]( const size_t i ) {
					return i + output_offset;
				},
				[ &output_offset ]( const size_t i ) {
					return i - output_offset;
				},
				[]( const size_t i ) {
					return i;
				},
				[]( const size_t i ) {
					return i;
				} );
#ifdef _DEBUG
			std::cout << "\t" << s <<
				", bsp1d_vxm: global output vector of the local vxm " "now contains " <<
				internal::getCoordinates( u._global ).nonzeroes() << " / " <<
				internal::getCoordinates( u._global ).size() << " nonzeroes.\n";
#endif

#ifdef _DEBUG
			{
				size_t num = 0;
				const auto * const stack = internal::getCoordinates( u._global ).getStack( num );
				std::cout << "\t" << s << ", bsp1d_vxm: global stack readout of " << num << " output elements...\n";
				for( size_t i = 0; i < num; ++i ) {
					std::cout << "\t\t" << stack[ i ] << "\n";
				}
				std::cout << "\tend global stack readout." << std::endl;
			}
			std::cout << "\t" << s << "bsp1d_vxm: now combining output vector...\n";
#endif

			// allcombine output
			assert( rc == SUCCESS );
			if( rc == SUCCESS ) {
				rc = u.template combine< descr >( ring.getAdditiveOperator() );
				u._nnz_is_dirty = true;
			}

			assert( rc == SUCCESS );

#ifdef _DEBUG
			std::cout << "\t" << s << ", bsp1d_vxm: final output vector now contains " << nnz( u ) << " / " << size( u ) << " nonzeroes.\n";
			std::cout << "\t" << s << ", bsp1d_vxm: done, exit code " << rc << ".\n";
#endif

			// done
			return rc;
		}

	} // namespace internal

	/** \internal Dispatches to bsp1d_vxm or bsp1d_mxv */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2,
		typename Coords >
	RC mxv( Vector< IOType, BSP1D, Coords > & u,
		const Matrix< InputType2, BSP1D > & A,
		const Vector< InputType1, BSP1D, Coords > & v,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const Vector< bool, BSP1D, Coords > mask( 0 );
		// transpose is delegated to vxm
		if( descr & descriptors::transpose_matrix ) {
			return internal::bsp1d_vxm< descr & ~( descriptors::transpose_matrix ), false, false, false >( u, mask, v, mask, A, ring );
		} else {
			return internal::bsp1d_mxv< descr, false, false, false >( u, mask, A, v, mask, ring );
		}
	}

	/** \internal Dispatches to bsp1d_vxm or bsp1d_mxv */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2,
		typename InputType3 = bool,
		typename InputType4 = bool,
		typename Coords >
	RC mxv( Vector< IOType, BSP1D, Coords > & u,
		const Vector< InputType3, BSP1D, Coords > & u_mask,
		const Matrix< InputType2, BSP1D > & A,
		const Vector< InputType1, BSP1D, Coords > & v,
		const Vector< InputType4, BSP1D, Coords > & v_mask,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		// transpose is delegated to vxm
		if( descr & descriptors::transpose_matrix ) {
			return internal::bsp1d_vxm< descr & ~( descriptors::transpose_matrix ), true, true, false >( u, u_mask, v, v_mask, A, ring );
		} else {
			return internal::bsp1d_mxv< descr, true, true, false >( u, u_mask, A, v, v_mask, ring );
		}
	}

	/** \internal Dispatches to bsp1d_vxm or bsp1d_mxv */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2,
		typename InputType3 = bool,
		typename InputType4 = bool,
		typename Coords >
	RC mxv( Vector< IOType, BSP1D, Coords > & u,
		const Vector< InputType3, BSP1D, Coords > & mask,
		const Matrix< InputType2, BSP1D > & A,
		const Vector< InputType1, BSP1D, Coords > & v,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const Vector< bool, BSP1D, Coords > empty_mask( 0 );
		// transpose is delegated to vxm
		if( descr & descriptors::transpose_matrix ) {
			return internal::bsp1d_vxm< descr & ~( descriptors::transpose_matrix ), true, false, false >( u, mask, v, empty_mask, A, ring );
		} else {
			return internal::bsp1d_mxv< descr, true, false, false >( u, mask, A, v, empty_mask, ring );
		}
	}

	/** \internal Dispatches to bsp1d_mxv or bsp1d_vxm */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2,
		typename Coords >
	RC vxm( Vector< IOType, BSP1D, Coords > & u,
		const Vector< InputType1, BSP1D, Coords > & v,
		const Matrix< InputType2, BSP1D > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const Vector< bool, BSP1D, Coords > mask( 0 );
		// transpose is delegated to mxv
		if( descr & descriptors::transpose_matrix ) {
			return internal::bsp1d_mxv< descr & ~( descriptors::transpose_matrix ), false, false, true >( u, mask, A, v, mask, ring );
		} else {
			return internal::bsp1d_vxm< descr, false, false, true >( u, mask, v, mask, A, ring );
		}
	}

	/** \internal Dispatches to bsp1d_vxm or bsp1d_mxv */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2,
		typename InputType3 = bool,
		typename InputType4 = bool,
		typename Coords >
	RC vxm( Vector< IOType, BSP1D, Coords > & u,
		const Vector< InputType3, BSP1D, Coords > & u_mask,
		const Vector< InputType1, BSP1D, Coords > & v,
		const Vector< InputType4, BSP1D, Coords > & v_mask,
		const Matrix< InputType2, BSP1D > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		// transpose is delegated to mxv
		if( descr & descriptors::transpose_matrix ) {
			return internal::bsp1d_mxv< descr & ~( descriptors::transpose_matrix ), true, true, true >( u, u_mask, A, v, v_mask, ring );
		} else {
			return internal::bsp1d_vxm< descr, true, true, true >( u, u_mask, v, v_mask, A, ring );
		}
	}

	/**
	 * This function provides dimension checking and will defer to the below
	 * function for the actual implementation. It also synchronises vectors that
	 * may be dereferenced at non-local positions.
	 *
	 * @see grb::eWiseLambda for the user-level specification.
	 */
	template<
		typename Func,
		typename DataType1, typename DataType2,
		typename Coords, typename... Args
	>
	RC eWiseLambda(
		const Func f,
		const Matrix< DataType1, BSP1D > &A,
		const Vector< DataType2, BSP1D, Coords > &x,
		Args... args
	) {
#ifdef _DEBUG
		std::cout << "In grb::eWiseLambda (BSP1D, matrix, recursive/vararg)\n";
#endif
		// do size checking
		if( size( x ) != nrows( A ) || size( x ) != ncols( A ) ) {
			return MISMATCH;
		}
		// when a vector may be accessed column-wise, make sure they are synchronised
		if( size( x ) == ncols( A ) ) {
			const RC ret = internal::synchronizeVector( x );
			if( ret != SUCCESS ) {
				return ret;
			}
		}
		return eWiseLambda( f, A, args... );
	}

	/**
	 * This function will execute quickly if and only if the matrix nonzeroes are
	 * not modified. If they are, the complexity becomes
	 *     \f$ \mathcal{O}(d_\text{max}\mathit{nnz}) \f$,
	 * with \f$ d_\text{max} \f$ the maximum number of nonzeroes within any single
	 * column of \a A.
	 *
	 * It assumes the copy-assignment and the equals comparison are implemented for
	 * the given data type.
	 */
	template< typename Func, typename DataType1 >
	RC eWiseLambda( const Func f, const Matrix< DataType1, BSP1D > & A ) {
#ifdef _DEBUG
		std::cout << "In grb::eWiseLambda (BSP1D, matrix)\n";
#endif
		const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();
		RC ret = eWiseLambda< internal::Distribution< BSP1D > >( f, internal::getLocal( A ), data.s, data.P );
		collectives< BSP1D >::allreduce< grb::descriptors::no_casting, grb::operators::any_or< RC > >( ret );
		return ret;
	}

	/** @} */

} // namespace grb

#endif // end `_H_GRB_BSP1D_BLAS2'
