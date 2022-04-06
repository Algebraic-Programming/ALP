
#ifndef _LINEARIZED_HALO_NDIM_GEOMETRY_H_
#define _LINEARIZED_HALO_NDIM_GEOMETRY_H_

#include <cstddef>
#include <vector>
#include <array>
#include <cassert>
#include <stdexcept>
#include <string>

#include "linearized_ndim_system.hpp"
#include "array_vector_storage.hpp"
#include "generic_vector_storage.hpp"
#include "ndim_vector.hpp"

namespace grb {
	namespace utils {
		namespace geometry {

template< typename CoordT, std::size_t DIMS > void __compute_neighbors_range(
	const array_vector_storage< CoordT, DIMS >& _system_sizes,
	const CoordT halo,
	const array_vector_storage< CoordT, DIMS >& system_coordinates,
	array_vector_storage< CoordT, DIMS >& neighbors_start,
	array_vector_storage< CoordT, DIMS >& neighbors_range ) {

	for( CoordT i{0}; i < DIMS/* - 1*/; i++ ) {
		const CoordT start{ system_coordinates[i] <= halo ? 0 : system_coordinates[i] - halo };
		const CoordT end{ std::min( system_coordinates[i] + halo, _system_sizes[i] - 1 ) };
		neighbors_start[i] = start;
		neighbors_range[i] = end - start + 1;
	}
	/*
	const std::size_t last{ DIMS - 1 };
	const CoordT start{ system_coordinates[ last ] <= halo ? 0 : system_coordinates[ last ] - halo };
	const CoordT end{ system_coordinates[ last ] + halo }; // can extend beyond actual DIMS-dimensional space
	neighbors_start[ last ] = start;
	neighbors_range[ last ] = end - start + 1;
	*/
}






template< typename CoordT, std::size_t DIMS > std::size_t __neighbour_to_system_coords(
	const std::array< CoordT, DIMS > & sizes,
	std::size_t system_size,
	const std::vector< ndim_vector< CoordT, CoordT, generic_vector_storage< CoordT > > > & dimension_neighbors,
	CoordT halo,
	CoordT neighbor,
	array_vector_storage< CoordT, DIMS > & result) {

	if( neighbor > system_size ) {
		throw std::invalid_argument("neighbor number ( " + std::to_string(neighbor)
			+ " ) >= system size ( " + std::to_string( system_size ) + " )");
	}

	array_vector_storage< CoordT, DIMS > halo_coords( DIMS );
#ifdef DBG
	std::size_t * const halo_coords_end{ halo_coords.data() + DIMS };
#endif
	std::fill_n( halo_coords.begin(), DIMS, 0 );

	for( std::size_t _dim{DIMS}; _dim > 0; _dim--) {

		const std::size_t dimension{_dim - 1};
		const std::size_t dimension_size{ sizes[dimension] };
		const ndim_vector< CoordT, CoordT, generic_vector_storage< CoordT > > & neighbors{ dimension_neighbors[dimension] };

		CoordT * const halo_coords_begin{ halo_coords.data() + dimension };

#ifdef DBG
		std::cout << "DIMENSION " << dimension << std::endl << "- setup - neighbour " << neighbor << std::endl;
		std::cout << "\thalo : ";
		print_sequence( halo_coords_begin, halo_coords_end ) << std::endl;
#endif

		std::size_t h{0};
		std::size_t previous_neighs{ 0 };
		*halo_coords_begin = h;
		std::size_t halo_max_neighs{ neighbors.at( halo_coords_begin ) };
		//std::cout << "\tinitial halo_max_neighs " << halo_max_neighs << std::endl;
		while( h < halo && neighbor >= previous_neighs + halo_max_neighs ) {
			h++;
			*halo_coords_begin = h;
			previous_neighs += halo_max_neighs;
			halo_max_neighs = neighbors.at( halo_coords_begin );
		}
#ifdef DBG
		std::cout << "- initial halo - neighbour " << neighbor << std::endl;
		std::cout << "\th " << h << std::endl;
		std::cout << "\thalo : ";
		print_sequence( halo_coords_begin, halo_coords_end ) << std::endl;
		std::cout << "\thalo_max_neighs " << halo_max_neighs << std::endl;
#endif


		if ( h < halo ){
			result[dimension] = h;
			neighbor -= previous_neighs;
#ifdef DBG
			std::cout << "end neighbour " << neighbor << std::endl;
#endif
			continue;
		}
		// saturation occurred
		const std::size_t distance_from_halo{ ( neighbor - previous_neighs ) / halo_max_neighs };
#ifdef DBG
		std::cout << "- before middle elements - neighbour " << neighbor << std::endl;
		std::cout << "\tprevious_neighs " << previous_neighs << std::endl;
		std::cout << "\thalo_max_neighs " << halo_max_neighs << std::endl;
		std::cout << "\tdistance_from_halo " << distance_from_halo << std::endl;
		std::cout << "\tdimension_size " << dimension_size << std::endl;
#endif
		if ( distance_from_halo < dimension_size - 2 * halo ) {
			result[dimension] =  distance_from_halo + halo;
			neighbor -= (previous_neighs + distance_from_halo * halo_max_neighs) ;
#ifdef DBG
			std::cout << "end neighbour " << neighbor << std::endl;
#endif
			continue;
		}
		previous_neighs += ( dimension_size - 2 * halo ) * halo_max_neighs;
#ifdef DBG
		std::cout << "- after middle elements -neighbour " << neighbor << std::endl;
		std::cout << "\tprevious_neighs " << previous_neighs << std::endl;
		std::cout << "\thalo_max_neighs " << halo_max_neighs << std::endl;
#endif

		h = halo - 1;
		*halo_coords_begin = h;
		halo_max_neighs = neighbors.at( halo_coords_begin );
		while( h > 0 && neighbor >= previous_neighs + halo_max_neighs ) {
			h--;
			*halo_coords_begin = h;
			previous_neighs += halo_max_neighs;
			halo_max_neighs = neighbors.at( halo_coords_begin );
		}
		neighbor -= previous_neighs;
#ifdef DBG
		std::cout << "- final halo - neighbour " << neighbor << std::endl;
		std::cout << "\tadding h " << h << " previous_neighs " << previous_neighs << std::endl;
#endif
		// ( dimension_size - 1 ) because coordinates are 0-based and neighbor
		// is "inside" range [ previous_neighs, previous_neighs + halo_max_neighs ]
		result[dimension] = dimension_size - 1 - h;
#ifdef DBG
		std::cout << "end neighbour " << neighbor << std::endl;
#endif
	}

	return neighbor;
}


template< typename CoordT > std::size_t __accumulate_dimension_neighbours(
	const ndim_vector< CoordT, CoordT, generic_vector_storage< CoordT > >& prev_neighs,
    CoordT* coords_buffer,
	std::size_t halo,
    std::size_t local_size ) {
	std::size_t neighs{0};
	std::size_t h{0};
	for( ; h < halo && local_size > 1; h++ ) {
		*coords_buffer = h;

		const std::size_t local_neighs{ prev_neighs.at( coords_buffer ) };
		neighs += 2 * local_neighs; // the 2 sides
		local_size -= 2;
	}
	*coords_buffer = h;
	neighs += local_size * prev_neighs.at( coords_buffer ); // innermost elements
	return neighs;
}

template< typename CoordT > void __populate_halo_neighbors( std::size_t halo,
    ndim_vector< CoordT, CoordT, generic_vector_storage< CoordT > >& container ) {

	using it_type = typename ndim_vector< CoordT, CoordT, generic_vector_storage< CoordT > >::domain_iterator;
	it_type end{ container.domain_end() };
	for( it_type it{ container.domain_begin() }; it != end; ++it ) {
		std::size_t res{1};
		for( std::size_t h: it->get_position() ) res *= (h + 1 + halo);
		container.at( it->get_position() ) = res;
	}
}

template< typename CoordT, std::size_t DIMS > std::size_t __init_halo_search(
    typename linearized_ndim_system< CoordT, array_vector_storage< CoordT, DIMS > >::const_vector_reference sizes,
    std::size_t halo,
	std::vector< ndim_vector< CoordT, CoordT, generic_vector_storage< CoordT > > >& dimension_limits ) {

    using nd_vec = ndim_vector< CoordT, CoordT, generic_vector_storage< CoordT > >;
    using nd_vec_iterator = typename nd_vec::domain_iterator;

	std::vector<std::size_t> halo_sizes( DIMS, halo + 1);
	dimension_limits.emplace_back(halo_sizes);

	// initialize values
	__populate_halo_neighbors< CoordT >( halo, dimension_limits[0] );
	for( std::size_t i{1}; i < DIMS; i++ ) {
		std::vector<std::size_t> halos( DIMS - i, halo + 1 );
		dimension_limits.emplace_back(halos);
	}

    std::array< CoordT, DIMS > prev_coords_buffer; // store at most DIMS values
    CoordT* const prev_coords{ prev_coords_buffer.data() };
	CoordT* const second{ prev_coords + 1 }; // store previous coordinates from second position
	for( std::size_t dimension{1}; dimension < DIMS; dimension++ ) {
		const nd_vec& prev_neighs{dimension_limits[dimension - 1]};
		nd_vec& current_neighs{dimension_limits[dimension]};

		nd_vec_iterator end{ current_neighs.domain_end() };
		for( nd_vec_iterator it{ current_neighs.domain_begin() }; it != end; ++it ) {
			typename nd_vec::const_domain_vector_reference current_halo_coords{ it->get_position() };

			std::copy( it->get_position().cbegin(), it->get_position().cend(), second );
			std::size_t local_size{ sizes[dimension - 1] };
			const std::size_t neighs{ __accumulate_dimension_neighbours(prev_neighs, prev_coords, halo, local_size) };
			current_neighs.at(current_halo_coords) = neighs;
		}
	}
	return __accumulate_dimension_neighbours( dimension_limits[DIMS - 1], prev_coords, halo, sizes.back() );
}

		} // namespace geometry
	} // namespace utils
} // namespace grb

#endif // _LINEARIZED_HALO_NDIM_GEOMETRY_H_
