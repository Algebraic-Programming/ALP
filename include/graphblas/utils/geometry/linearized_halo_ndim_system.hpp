
#ifndef _LINEARIZED_HALO_NDIM_SYSTEM_H_
#define _LINEARIZED_HALO_NDIM_SYSTEM_H_

#include <cstddef>
#include <vector>
#include <array>
#include <cassert>

#include "array_vector_storage.hpp"
#include "linearized_ndim_system.hpp"
#include "linearized_halo_ndim_geometry.hpp"
#include "linearized_halo_ndim_iterator.hpp"

namespace grb {
	namespace utils {
		namespace geometry {

// only with array_vector_storage
template< typename CoordT, std::size_t DIMS > class linearized_halo_ndim_system:
	public linearized_ndim_system< CoordT, array_vector_storage< CoordT, DIMS > > {
public:

	using iterator = linearized_halo_ndim_iterator< CoordT, DIMS >;
    using const_vector_reference = typename array_vector_storage< CoordT, DIMS >::const_vector_storage;
	using self_t = linearized_halo_ndim_system< CoordT, DIMS >;
	using base_t = linearized_ndim_system< CoordT, array_vector_storage< CoordT, DIMS > >;

    linearized_halo_ndim_system( const_vector_reference sizes, CoordT halo ):
		base_t( sizes.cbegin(), sizes.cend() ),
        _halo( halo ) {

		for( CoordT __size : sizes ) {
			if ( __size < 2 * halo + 1 ) {
				throw std::invalid_argument(
					std::string( "the halo (" + std::to_string(halo) +
					std::string( ") goes beyond a system size (" ) +
					std::to_string( __size) + std::string( ")" ) ) );
			}
		}

        this->_system_size = __init_halo_search< CoordT, DIMS >(
				this->get_sizes(),
				_halo, this->_dimension_limits );
		assert( this->_dimension_limits.size() == DIMS );
    }

    linearized_halo_ndim_system() = delete;

    linearized_halo_ndim_system( const self_t & ) = default;

    linearized_halo_ndim_system( self_t && ) = delete;

    ~linearized_halo_ndim_system() noexcept {}

    self_t & operator=( const self_t & ) = default;

    self_t & operator=( self_t && ) = delete;

	iterator begin() const {
		return iterator( *this );
	}

	iterator end() const {
		return iterator::make_system_end_iterator( *this );
	}

	std::size_t halo_system_size() const {
		return this->_system_size;
	}

	std::size_t base_system_size() const {
		return this->base_t::system_size();
	}

    std::size_t halo() const {
        return this->_halo;
    }

    void compute_neighbors_range(
        const array_vector_storage< CoordT, DIMS >& system_coordinates,
	    array_vector_storage< CoordT, DIMS >& neighbors_start,
	    array_vector_storage< CoordT, DIMS >& neighbors_range) const noexcept {
        __compute_neighbors_range( this->get_sizes(),
            this->_halo,
            system_coordinates,
            neighbors_start,
            neighbors_range
        );
    }

    std::size_t neighbour_linear_to_element (
        CoordT neighbor,
	    array_vector_storage< CoordT, DIMS > & result) const noexcept {
        return __neighbour_to_system_coords( this->get_sizes(),
        this->_system_size, this->_dimension_limits, this->_halo, neighbor, result );
    }

private:

    const CoordT _halo;
    std::vector< ndim_vector< CoordT, CoordT, generic_vector_storage< CoordT > > > _dimension_limits;
    std::size_t _system_size;

};

		} // namespace geometry
	} // namespace utils
} // namespace grb

#endif // _LINEARIZED_HALO_NDIM_SYSTEM_H_
