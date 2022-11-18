
#ifndef _H_GRB_ALGORITHMS_MULTIGRID_LINEARIZED_HALO_NDIM_GEOMETRY
#define _H_GRB_ALGORITHMS_MULTIGRID_LINEARIZED_HALO_NDIM_GEOMETRY

#include <cstddef>
#include <vector>
#include <array>
#include <cassert>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <algorithm>

#include "array_vector_storage.hpp"
#include "dynamic_vector_storage.hpp"
#include "linearized_ndim_system.hpp"
#include "ndim_vector.hpp"

namespace grb {
	namespace utils {
		namespace multigrid {

			template<
				size_t DIMS,
				typename CoordType
			> void __compute_neighbors_range(
				const ArrayVectorStorage< DIMS, CoordType > &_system_sizes,
				const CoordType halo,
				const ArrayVectorStorage< DIMS, CoordType > &system_coordinates,
				ArrayVectorStorage< DIMS, CoordType > &neighbors_start,
				ArrayVectorStorage< DIMS, CoordType > &neighbors_range ) {

				for( CoordType i{0}; i < DIMS/* - 1*/; i++ ) {
					const CoordType start{ system_coordinates[i] <= halo ? 0 : system_coordinates[i] - halo };
					const CoordType end{ std::min( system_coordinates[i] + halo, _system_sizes[i] - 1 ) };
					neighbors_start[i] = start;
					neighbors_range[i] = end - start + 1;
				}
			}

			template<
				size_t DIMS,
				typename CoordType
			> size_t __neighbour_to_system_coords(
				const std::array< CoordType, DIMS > &sizes,
				size_t system_size,
				const std::vector< NDimVector< CoordType, CoordType, DynamicVectorStorage< CoordType > > >
					&dimension_neighbors,
				CoordType halo,
				CoordType neighbor,
				ArrayVectorStorage< DIMS, CoordType > &result
			){
				if( neighbor > system_size ) {
					throw std::invalid_argument("neighbor number ( " + std::to_string(neighbor)
						+ " ) >= system size ( " + std::to_string( system_size ) + " )");
				}
				ArrayVectorStorage< DIMS, CoordType > halo_coords( DIMS );
#ifdef _DEBUG
				size_t * const halo_coords_end{ halo_coords.data() + DIMS };
#endif
				std::fill_n( halo_coords.begin(), DIMS, 0 );

				for( size_t _dim{DIMS}; _dim > 0; _dim--) {

					const size_t dimension{_dim - 1};
					const size_t dimension_size{ sizes[dimension] };
					const NDimVector< CoordType, CoordType, DynamicVectorStorage< CoordType > > & neighbors{ dimension_neighbors[dimension] };

					CoordType * const halo_coords_begin{ halo_coords.data() + dimension };
#ifdef _DEBUG
					std::cout << "DIMENSION " << dimension << std::endl << "- setup - neighbour " << neighbor << std::endl;
					std::cout << "\thalo : ";
					print_sequence( halo_coords_begin, halo_coords_end ) << std::endl;
#endif
					size_t h{0};
					size_t previous_neighs{ 0 };
					*halo_coords_begin = h;
					size_t halo_max_neighs{ neighbors.at( halo_coords_begin ) };
					//std::cout << "\tinitial halo_max_neighs " << halo_max_neighs << std::endl;
					while( h < halo && neighbor >= previous_neighs + halo_max_neighs ) {
						h++;
						*halo_coords_begin = h;
						previous_neighs += halo_max_neighs;
						halo_max_neighs = neighbors.at( halo_coords_begin );
					}
#ifdef _DEBUG
					std::cout << "- initial halo - neighbour " << neighbor << std::endl;
					std::cout << "\th " << h << std::endl;
					std::cout << "\thalo : ";
					print_sequence( halo_coords_begin, halo_coords_end ) << std::endl;
					std::cout << "\thalo_max_neighs " << halo_max_neighs << std::endl;
#endif
					if ( h < halo ){
						result[dimension] = h;
						neighbor -= previous_neighs;
#ifdef _DEBUG
						std::cout << "end neighbour " << neighbor << std::endl;
#endif
						continue;
					}
					// saturation occurred
					const size_t distance_from_halo{ ( neighbor - previous_neighs ) / halo_max_neighs };
#ifdef _DEBUG
					std::cout << "- before middle elements - neighbour " << neighbor << std::endl;
					std::cout << "\tprevious_neighs " << previous_neighs << std::endl;
					std::cout << "\thalo_max_neighs " << halo_max_neighs << std::endl;
					std::cout << "\tdistance_from_halo " << distance_from_halo << std::endl;
					std::cout << "\tdimension_size " << dimension_size << std::endl;
#endif
					if ( distance_from_halo < dimension_size - 2 * halo ) {
						result[dimension] =  distance_from_halo + halo;
						neighbor -= (previous_neighs + distance_from_halo * halo_max_neighs) ;
#ifdef _DEBUG
						std::cout << "end neighbour " << neighbor << std::endl;
#endif
						continue;
					}
					previous_neighs += ( dimension_size - 2 * halo ) * halo_max_neighs;
#ifdef _DEBUG
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
#ifdef _DEBUG
					std::cout << "- final halo - neighbour " << neighbor << std::endl;
					std::cout << "\tadding h " << h << " previous_neighs " << previous_neighs << std::endl;
#endif
					// ( dimension_size - 1 ) because coordinates are 0-based and neighbor
					// is "inside" range [ previous_neighs, previous_neighs + halo_max_neighs ]
					result[dimension] = dimension_size - 1 - h;
#ifdef _DEBUG
					std::cout << "end neighbour " << neighbor << std::endl;
#endif
				}
				return neighbor;
			}


			template< typename CoordType > size_t __accumulate_dimension_neighbours(
				const NDimVector< CoordType, CoordType, DynamicVectorStorage< CoordType > > &prev_neighs,
				CoordType* coords_buffer,
				size_t halo,
				size_t local_size
			) {
				size_t neighs{0};
				size_t h{0};
				for( ; h < halo && local_size > 1; h++ ) {
					*coords_buffer = h;

					const size_t local_neighs{ prev_neighs.at( coords_buffer ) };
					neighs += 2 * local_neighs; // the 2 sides
					local_size -= 2;
				}
				*coords_buffer = h;
				neighs += local_size * prev_neighs.at( coords_buffer ); // innermost elements
				return neighs;
			}

			template< typename CoordType > void __populate_halo_neighbors( size_t halo,
				NDimVector< CoordType, CoordType, DynamicVectorStorage< CoordType > >& container ) {

				using it_type = typename NDimVector< CoordType, CoordType, DynamicVectorStorage< CoordType > >::DomainIterator;
				it_type end{ container.domain_end() };
				for( it_type it{ container.domain_begin() }; it != end; ++it ) {
					size_t res{1};
					for( size_t h: it->get_position() ) res *= (h + 1 + halo);
					container.at( it->get_position() ) = res;
				}
			}

			template<
				typename CoordType,
				size_t DIMS
			> size_t __init_halo_search(
				typename LinearizedNDimSystem< CoordType, ArrayVectorStorage< DIMS, CoordType > >::ConstVectorReference
					sizes,
				size_t halo,
				std::vector< NDimVector< CoordType, CoordType, DynamicVectorStorage< CoordType > > >& dimension_limits
			) {
				using nd_vec = NDimVector< CoordType, CoordType, DynamicVectorStorage< CoordType > >;
				using nd_vec_iterator = typename nd_vec::DomainIterator;

				std::vector<size_t> halo_sizes( DIMS, halo + 1);
				dimension_limits.emplace_back(halo_sizes);
				// initialize values
				__populate_halo_neighbors< CoordType >( halo, dimension_limits[0] );
				for( size_t i{1}; i < DIMS; i++ ) {
					std::vector<size_t> halos( DIMS - i, halo + 1 );
					dimension_limits.emplace_back(halos);
				}

				std::array< CoordType, DIMS > prev_coords_buffer; // store at most DIMS values
				CoordType* const prev_coords{ prev_coords_buffer.data() };
				CoordType* const second{ prev_coords + 1 }; // store previous coordinates from second position
				for( size_t dimension{1}; dimension < DIMS; dimension++ ) {
					const nd_vec& prev_neighs{dimension_limits[dimension - 1]};
					nd_vec& current_neighs{dimension_limits[dimension]};

					nd_vec_iterator end{ current_neighs.domain_end() };
					for( nd_vec_iterator it{ current_neighs.domain_begin() }; it != end; ++it ) {
						typename nd_vec::ConstDomainVectorReference current_halo_coords{ it->get_position() };

						std::copy( it->get_position().cbegin(), it->get_position().cend(), second );
						size_t local_size{ sizes[dimension - 1] };
						const size_t neighs{ __accumulate_dimension_neighbours(prev_neighs, prev_coords, halo, local_size) };
						current_neighs.at(current_halo_coords) = neighs;
					}
				}
				return __accumulate_dimension_neighbours( dimension_limits[DIMS - 1], prev_coords, halo, sizes.back() );
			}

		} // namespace multigrid
	} // namespace utils
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_LINEARIZED_HALO_NDIM_GEOMETRY
