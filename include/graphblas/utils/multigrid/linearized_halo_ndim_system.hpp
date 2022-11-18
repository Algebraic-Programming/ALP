
#ifndef _H_GRB_ALGORITHMS_MULTIGRID_LINEARIZED_HALO_NDIM_SYSTEM
#define _H_GRB_ALGORITHMS_MULTIGRID_LINEARIZED_HALO_NDIM_SYSTEM

#include <cstddef>
#include <vector>
#include <array>
#include <cassert>
#include <cstddef>

#include "array_vector_storage.hpp"
#include "linearized_ndim_system.hpp"
#include "linearized_halo_ndim_geometry.hpp"
#include "linearized_halo_ndim_iterator.hpp"

namespace grb {
	namespace utils {
		namespace multigrid {

			// only with ArrayVectorStorage
			template<
				size_t DIMS,
				typename SizeType
			> class LinearizedHaloNDimSystem:
				public LinearizedNDimSystem< SizeType, ArrayVectorStorage< DIMS, SizeType > > {
			public:
				using VectorType = ArrayVectorStorage< DIMS, SizeType >;
				using ConstVectorStorageType = typename VectorType::ConstVectorStorageType;
				using SelfType = LinearizedHaloNDimSystem< DIMS, SizeType >;
				using BaseType = LinearizedNDimSystem< SizeType, VectorType >;
				using Iterator = LinearizedHaloNDimIterator< DIMS, SizeType >;

				LinearizedHaloNDimSystem( ConstVectorStorageType sizes, SizeType halo ):
					BaseType( sizes.cbegin(), sizes.cend() ),
					_halo( halo ) {

					for( SizeType __size : sizes ) {
						if ( __size < 2 * halo + 1 ) {
							throw std::invalid_argument(
								std::string( "the halo (" + std::to_string(halo) +
								std::string( ") goes beyond a system size (" ) +
								std::to_string( __size) + std::string( ")" ) ) );
						}
					}

					this->_system_size = __init_halo_search< SizeType, DIMS >(
							this->get_sizes(),
							_halo, this->_dimension_limits );
					assert( this->_dimension_limits.size() == DIMS );
				}

				LinearizedHaloNDimSystem() = delete;

				LinearizedHaloNDimSystem( const SelfType & ) = default;

				LinearizedHaloNDimSystem( SelfType && ) = delete;

				~LinearizedHaloNDimSystem() noexcept {}

				SelfType & operator=( const SelfType & ) = default;

				SelfType & operator=( SelfType && ) = delete;

				Iterator begin() const {
					return Iterator( *this );
				}

				Iterator end() const {
					return Iterator::make_system_end_iterator( *this );
				}

				size_t halo_system_size() const {
					return this->_system_size;
				}

				size_t base_system_size() const {
					return this->BaseType::system_size();
				}

				size_t halo() const {
					return this->_halo;
				}

				void compute_neighbors_range(
					const VectorType &system_coordinates,
					VectorType &neighbors_start,
					VectorType &neighbors_range) const noexcept {
					__compute_neighbors_range( this->get_sizes(),
						this->_halo,
						system_coordinates,
						neighbors_start,
						neighbors_range
					);
				}

				size_t neighbour_linear_to_element (
					SizeType neighbor,
					VectorType &result) const noexcept {
					return __neighbour_to_system_coords( this->get_sizes(),
					this->_system_size, this->_dimension_limits, this->_halo, neighbor, result );
				}

			private:
				const SizeType _halo;
				std::vector< NDimVector< SizeType, SizeType, DynamicVectorStorage< SizeType > > > _dimension_limits;
				size_t _system_size;
			};

		} // namespace multigrid
	} // namespace utils
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_LINEARIZED_HALO_NDIM_SYSTEM
