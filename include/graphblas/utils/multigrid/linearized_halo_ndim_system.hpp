
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
 * @file linearized_halo_ndim_system.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Definition of LinearizedHaloNDimSystem.
 */

#ifndef _H_GRB_ALGORITHMS_MULTIGRID_LINEARIZED_HALO_NDIM_SYSTEM
#define _H_GRB_ALGORITHMS_MULTIGRID_LINEARIZED_HALO_NDIM_SYSTEM

#include <array>
#include <cassert>
#include <cstddef>
#include <vector>
#ifdef _DEBUG
#include <iostream>
#endif

#include "array_vector_storage.hpp"
#include "dynamic_vector_storage.hpp"
#include "linearized_halo_ndim_iterator.hpp"
#include "linearized_ndim_system.hpp"
#include "ndim_vector.hpp"

namespace grb {
	namespace utils {
		namespace multigrid {

			/**
			 * Structure to represent an N-dimensional space (or \a system) of given sizes and to
			 * iterate on both the \a elements of the N-dimensional system and the N-dimensional
			 * \a neighbors of each element within a given \p halo. This facility takes into account
			 * the various cases where the element is at the corner, edge or face of the N-dimensional
			 * system, to which different neighbors correspond. Both elements and their neighbors are
			 * vectors in the N-dimensional system and as such described via both N-dimensional coordinates
			 * and a linear coordinate.
			 *
			 * This structure returns the number of elements of the underlying N-dimensional system
			 * (the \a base system) via #base_system_size() and the total sum of neighbors of all
			 * system elements via #halo_system_size().
			 *
			 * The peculiar feature of this structure is the method #neighbour_linear_to_element(), to translate
			 * a neighbor index (i.e. a value from \a 0 to #halo_system_size(), uniquely identifying an element
			 * as neighbor of an element) to the N-dimensional coordinates of the corresponding elements in a time
			 * that is constant with respect to the input value (it depends on \p DIMS and the halo size).
			 * This facility allows the iterators of a LinearizedNDimSystem to be random-access: when advancing
			 * an iterator by an \a offset via the \a += method, the logic:
			 *
			 * - increments the index of the current neighbor (stored inside the iterator) by \a offset, thus
			 *  computing the index of the destination neighbor (constant time)
			 * - translates the index of the destination neighbor to its base element's coordinates via
			 *  #neighbour_linear_to_element() (constant time)
			 *
			 * The same method also returns the index of the destination neighbor within the sub-space of the base
			 * element's neighbors: hence, the logic can compute in constant time the destination base element
			 * and its destination neighbor. The constant time of this translation is achieved by pre-computing
			 * the number of neighbors for each element along each dimension: for example, inner elements in
			 * a 3D mesh with halo 1 have 27 neighbors. Thus, it suffices in principle to divide the neighbor
			 * index by 27 to compute the base element of a neighbor. Care must be taken for elements at the
			 * sides of each dimension: for example, a corner element on a face has 8 neighbors, while a corner
			 * element in an iternal slab (a 2D "plane" in a 3D mesh) has 12 neighbors. The pre-computed
			 * information and the logic also account for this.
			 *
			 * @tparam DIMS number of dimensions of the system
			 * @tparam SizeType type storing the system sizes and offsets
			 */
			template<
				size_t DIMS,
				typename SizeType
			> class LinearizedHaloNDimSystem :
				public LinearizedNDimSystem< SizeType, ArrayVectorStorage< DIMS, SizeType > > {
			public:
				using VectorType = ArrayVectorStorage< DIMS, SizeType >;
				using ConstVectorStorageType = typename VectorType::ConstVectorStorageType;
				using SelfType = LinearizedHaloNDimSystem< DIMS, SizeType >;
				using BaseType = LinearizedNDimSystem< SizeType, VectorType >;
				using Iterator = LinearizedHaloNDimIterator< DIMS, SizeType >;

				/**
				 * Construct a new LinearizedHaloNDimSystem object with given sizes and halo.
				 *
				 * The size of \p sizes must be exactly \p DIMS. Each size must be so that there is at least
				 * en element in the system with full halo neighors, i.e. for each size \a s
				 * <em>s >= 2 * halo + 1</em> (otherwise an exception is thrown).
				 */
				LinearizedHaloNDimSystem(
					ConstVectorStorageType sizes,
					SizeType halo
				) :
					BaseType( sizes.cbegin(), sizes.cend() ),
					_halo( halo )
				{
					for( SizeType __size : sizes ) {
						if( __size < halo + 1 ) {
							throw std::invalid_argument(
								std::string( "the halo (" + std::to_string( halo )
								+ std::string( ") goes beyond a system size (" )
								+ std::to_string( __size ) + std::string( ")" ) ) );
						}
					}

					this->_system_size = init_neigh_to_base_search(
						this->get_sizes(), _halo, this->_dimension_limits );
					assert( this->_dimension_limits.size() == DIMS );
				}

				LinearizedHaloNDimSystem() = delete;

				LinearizedHaloNDimSystem( const SelfType & ) = default;

				LinearizedHaloNDimSystem( SelfType && ) = delete;

				~LinearizedHaloNDimSystem() noexcept {}

				SelfType & operator=( const SelfType & ) = default;

				SelfType & operator=( SelfType && ) = delete;

				/**
				 * Builds an iterator from the beginning of the system, i.e. from vector \a [0,0,...,0].
				 * The iterator iterates on each neighbor and allows iterating on each element and on
				 * its neighbors.
				 */
				Iterator begin() const {
					return Iterator( *this );
				}

				/**
				 * Build an iterator marking the end of the system; it should not be accessed.
				 */
				Iterator end() const {
					return Iterator::make_system_end_iterator( *this );
				}

				/**
				 * Returns the size of the entire system, i.e. the number of neighbors of all elements.
				 */
				size_t halo_system_size() const {
					return this->_system_size;
				}

				/**
				 * Returns the size of the base system, i.e. number of elements (not considering neighbors).
				 */
				size_t base_system_size() const {
					return this->BaseType::system_size();
				}

				/**
				 * Returns the halo size.
				 */
				size_t halo() const {
					return this->_halo;
				}

				/**
				 * Computes the first neighbor and the size of the N-dimensional range of neighbors
				 * around the given element's coordinates for the system \c this.
				 *
				 * @param[in] element_coordinates coordinates of the element to iterate around
				 * @param[out] neighbors_start first neighbor around \p element_coordinates to iterate from
				 * @param[out] neighbors_range vector of halos around \p element_coordinates;
				 * if \p element_coordinates is an inner point, all values equal #halo(), they are smaller
				 * otherwise (on corner, edge, or face).
				 */
				void compute_neighbors_range(
					const VectorType & element_coordinates,
					VectorType & neighbors_start,
					VectorType & neighbors_range
				) const noexcept {
					compute_first_neigh_and_range( this->get_sizes(),
						this->_halo, element_coordinates, neighbors_start, neighbors_range );
				}

				/**
				 * Maps the linear index \p neighbor_linear of a neighbor to the vector \p base_element_vector
				 * of the corresponding element \p neighbor_linear is neighbor of, and returns the neighbor's
				 * number within the sub-space of \p base_element_vector 's neighbors.
				 *
				 * @param[in] neighbor_linear linear coordinate of input neighbor
				 * @param[out] base_element_vector vector of coordinates that identify which element
				 *  \p neighbor_linear is neighbor of
				 * @return size_t the neighbor number w.r.t. to the corresponding element: if \a e is the system
				 * element \p neighbor_linear is neighbor of and \a e has \a n neighbors, then the return value
				 * \a 0<=i<n is the the index of \p neighbor_linear among \a e's neighbors, computed w.r.t. the
				 * iteration order.
				 */
				size_t neighbour_linear_to_element(
					SizeType neighbor_linear,
					VectorType & base_element_vector
				) const noexcept {
					return map_neigh_to_base_and_index( this->get_sizes(), this->_system_size,
						this->_dimension_limits, this->_halo, neighbor_linear, base_element_vector );
				}

			private:
				const SizeType _halo;
				std::vector< NDimVector< SizeType, SizeType,
					DynamicVectorStorage< SizeType > > > _dimension_limits;
				size_t _system_size;

				/**
				 * Computes the total number of neighbors along a certain dimension and configuration by accumulating
				 * the neighbors along the smaller dimensions.
				 *
				 * The logic uses this buffer to iterate over the configurations of
				 * the previous dimension. Example: to compute in 3D the neighbors of an inner row of a face
				 * (configuration <em>[0,1,0]</em>, dimension 1 - y), the logic needs the neighbors of
				 * en edge element and of an element internal to a face of the mesh, corresponding to
				 * the configurations <em>[0,1,0]</em> and <em>[1,1,0]</em>, respectively. Hence, the caller
				 * must initialize a buffer with the values <em>[X,1,0]</em> (\a X meaning don't care) and pass
				 * as \p coords_buffer the pointer to the first position (the \a X ), where this function
				 * will write all possible values <em>[0, \p halo )</em> to access the number of neighbors
				 * of the configurations of the previous dimension via \p prev_neighs and accumulate them.
				 *
				 * @param[in] prev_neighs neighbors in the configurations of the previous dimension
				 * @param[in,out] coords_buffer pointer to the first position of the configuration buffer
				 *  for this dimension
				 * @param[in] halo halo size
				 * @param[in] local_size size (i.e., number of elements) along the current dimension,
				 *  including the edges
				 * @return size_t the total number of neighbors for this configuration and this dimension
				 */
				static size_t accumulate_dimension_neighbours(
					const NDimVector< SizeType, SizeType, DynamicVectorStorage< SizeType > > & prev_neighs,
					SizeType * coords_buffer,
					size_t halo,
					size_t local_size
				) {
					size_t neighs = 0;
					size_t h = 0;
					for( ; h < halo && local_size > 1; h++ ) {
						*coords_buffer = h;

						const size_t local_neighs = prev_neighs.at( coords_buffer );
						neighs += 2 * local_neighs; // the 2 sides
						local_size -= 2;
					}
					*coords_buffer = h;
					neighs += local_size * prev_neighs.at( coords_buffer ); // innermost elements
					return neighs;
				}

				/**
				 * Computes the number of neighbors for each configuration along dimension 0:
				 * corner, edge, face, inner element.
				 *
				 * Example: in a 3D system with <em>\p halo = 1</em>, the configurations along dimension 0 are 8:
				 * 1. z axis - face:
				 *   1. y axis - top row: corner element (8 neighbors), edge element (12 neighbors)
				 *   2  y axis - inner row: edge element (12 neighbors), face inner element (18 neighbors)
				 * 2. z axis - inner slab:
				 *   1. y axis - top row: edge element (12 neighbors), face inner element (18 neighbors)
				 *   2  y axis - inner row: face inner element (18 neighbors), inner element (27 neighbors)
				 *
				 * @param[in] halo halo size
				 * @param[out] config_neighbors the storage object for each configuration
				 */
				static void compute_dim0_neighbors(
					size_t halo,
					NDimVector< SizeType, SizeType, DynamicVectorStorage< SizeType > > & config_neighbors
				) {
					using it_type = typename NDimVector< SizeType, SizeType,
						DynamicVectorStorage< SizeType > >::DomainIterator;
					it_type end = config_neighbors.domain_end();
					for( it_type it = config_neighbors.domain_begin(); it != end; ++it ) {
						size_t res = 1;
						for( size_t h : it->get_position() )
							res *= ( h + 1 + halo );
						config_neighbors.at( it->get_position() ) = res;
					}
				}

				/**
				 * Initializes the search space of neighbors for the <neighbor linear> -> <base vector> translation.
				 *
				 * This function populates an std::vector<> with the number of neighors for each dimension
				 * and each configuration (corner, edge, face, inner).
				 * Along each dimension \a d, it stores an \a n -dimensional vector
				 * NDimVector<SizeType,SizeType,DynamicVectorStorage< SizeType>> (<em>n = 2 ^ d</em>) with all
				 * possible numbers of neighbors along that dimension, depending on the position of the element
				 * (corner, edge, face, inner volume); for example, for 3 dimensions:
				 *  - dimension 2 (z axis) moves along "slabs" of a 3D systems, where the total number of neighbors
				 *   depends on whether the slab is a face of the mesh of an internal slab (2 possible configurations:
				 *   face slabs or inner slabs)
				 *  - dimension 1 (y axis) moves along "rows" within each slab, whose total number of neighbors
				 *	  depends on whether the row is at the extreme sides (top or bottom of the face) or inside;
				 *   in turn, each type of slab has different geometry (face slabs comprise mesh corners, edges and
				 * 	 faces, while inner slabs comprise edges, faces and inner elements), thus resulting in
				 *   2*2 different configurations of dimension-1 total neighbors
				 *  - dimension 0 (x axis) moves along "column" elements within each row, where the first (or last)
				 *   column has a different number of neighbors than the inner ones; here again are two configuration
				 *   for each dimension-1 configuration, leading to a total of 8 dimension-1 configurations
				 * Within each dimension \a d, each configuration (as per the above explanation) can be identified
				 * via a vector of <em>N - d</em> coordinates; to limit the data storage, every dimension stores the
				 * total number of neighbors only at the first side and inside, since the second side  is identical
				 * to the first one: for example, along the z axis the first and last slab (those on the two extremes)
				 * have the same size, and one only is stored. Therefore, with <em>halo = 1</em> a vector identifying
				 * a configuration is composed only of 0s and 1s. For example, the vector <em>[0,1,0]</em> identifies:
				 * - rightmost 0 (z axis): first (or last) slab, i.e. face slab
				 * - (middle) 1 (y axis): inner row
				 * - leftmost 0 (x axis): first (or last) element, i.e. on the edge of the mesh
				 * In a 3D space with <em>halo = 1</em>, this element has 12 neighbors (it is on the edge of a face).
				 *
				 * @paragraph[in] vector of sizes sizes of the N-dimensional system
				 * @param[in] halo halo size
				 * @param[out] dimension_limits the std::vector<> with the neighbors information for each dimension
				 *  and each configuration
				 * @return size_t the number of neighbors of the entire system
				 */
				static size_t init_neigh_to_base_search(
					typename LinearizedNDimSystem< SizeType, ArrayVectorStorage< DIMS, SizeType >
						>::ConstVectorReference sizes,
					size_t halo,
					std::vector< NDimVector< SizeType, SizeType, DynamicVectorStorage< SizeType > > > & dimension_limits
				) {
					using nd_vec = NDimVector< SizeType, SizeType, DynamicVectorStorage< SizeType > >;
					using nd_vec_iterator = typename nd_vec::DomainIterator;

					std::vector< size_t > halo_sizes( DIMS, halo + 1 );
					dimension_limits.emplace_back( halo_sizes );
					// initialize values
					compute_dim0_neighbors( halo, dimension_limits[ 0 ] );
					for( size_t i = 1; i < DIMS; i++ ) {
						std::vector< size_t > halos( DIMS - i, halo + 1 );
						dimension_limits.emplace_back( halos );
					}

					std::array< SizeType, DIMS > prev_coords_buffer; // store at most DIMS values
					SizeType * const prev_coords = prev_coords_buffer.data();
					SizeType * const second = prev_coords + 1; // store previous coordinates from second position
					for( size_t dimension = 1; dimension < DIMS; dimension++ ) {
						const nd_vec & prev_neighs { dimension_limits[ dimension - 1 ] };
						nd_vec & current_neighs { dimension_limits[ dimension ] };

						nd_vec_iterator end = current_neighs.domain_end();
						for( nd_vec_iterator it = current_neighs.domain_begin(); it != end; ++it ) {
							typename nd_vec::ConstDomainVectorReference current_halo_coords = it->get_position();

							std::copy( it->get_position().cbegin(), it->get_position().cend(), second );
							size_t local_size = sizes[ dimension - 1 ];
							const size_t neighs = accumulate_dimension_neighbours( prev_neighs,
								prev_coords, halo, local_size );
							current_neighs.at( current_halo_coords ) = neighs;
						}
					}
					return accumulate_dimension_neighbours( dimension_limits[ DIMS - 1 ],
						prev_coords, halo, sizes.back() );
				}

				/**
				 * For the given system (with sizes \p _system_sizes), the given halo size \p halo,
				 * the given element's coordinates \p element_coordinates, computes the coordinates
				 * of the first neighbor of \p element_coordinates into \p neighbors_start (within the main system)
				 * and the range of neighbors of \p element_coordinates, i.e. the sub-space of neighbors of
				 * \p element_coordinates; hence, \p neighbors_range stores at most <em>2 *<\em> \p halo
				 * <em> + 1</em> per coordinate.
				 *
				 * @param[in] _system_sizes sizes of the N-dimensional system
				 * @param[in] halo halo size
				 * @param[in] element_coordinates coordinates of the considered element
				 * @param[out] neighbors_start stores the (absolute) coordinates of the first neighbor
				 *  of \p element_coordinates
				 * @param[out] neighbors_range stores the range of neighbors around \p element_coordinates
				 */
				static void compute_first_neigh_and_range(
					const ArrayVectorStorage< DIMS, SizeType > & _system_sizes,
					const SizeType halo,
					const ArrayVectorStorage< DIMS, SizeType > & element_coordinates,
					ArrayVectorStorage< DIMS, SizeType > & neighbors_start,
					ArrayVectorStorage< DIMS, SizeType > & neighbors_range
				) {
					for( SizeType i = 0; i < DIMS /* - 1*/; i++ ) {
						const SizeType start = element_coordinates[ i ] <= halo ? 0 :
							element_coordinates[ i ] - halo;
						const SizeType end = std::min( element_coordinates[ i ] + halo, _system_sizes[ i ] - 1 );
						neighbors_start[ i ] = start;
						neighbors_range[ i ] = end - start + 1;
					}
				}

#ifdef _DEBUG
				template< typename IterType >
				static std::ostream & print_sequence( IterType begin, IterType end ) {
					for( ; begin != end; ++begin ) {
						std::cout << *begin << ' ';
					}
					return std::cout;
				}
#endif

				/**
				 * Maps a neighbor's linear coordinate \p neighbor_linear to the element \p element_vector it is
				 * neighbor of and also returns the neighbor index of \p neighbor_linear within the sub-space
				 * of \p element_vector's neighbors.
				 *
				 * @param[in] sizes main system sizes along all dimensions
				 * @param[in] system_size total size of the neighbors system, i.e. the total number of neighbors
				 * @param[in] neighbors_per_dimension along each dimension \a d, it stores an \a n -dimensional vector
				 *  NDimVector<SizeType,SizeType,DynamicVectorStorage< SizeType>> (<em>n = 2 ^ d</em>) with all
				 *  possible numbers of neighbors along that dimension, depending on the position of the element
				 *  (corner, edge, face, inner volume)
				 * @param[in] halo halo size
				 * @param[in] neighbor_linear linear coordinate of the neighbor
				 * @param[out] element_vector coordinates vector representing the element \p neighbor_linear is
				 *  neighbor of
				 * @return size_t the index of the neighbor within the element's neighbors
				 */
				static size_t map_neigh_to_base_and_index(
					const std::array< SizeType, DIMS > & sizes,
					size_t system_size,
					const std::vector< NDimVector< SizeType, SizeType, DynamicVectorStorage< SizeType > >
						> & neighbors_per_dimension,
					SizeType halo,
					SizeType neighbor_linear,
					ArrayVectorStorage< DIMS, SizeType > & element_vector
				) {
					if( neighbor_linear > system_size ) {
						throw std::invalid_argument( "neighbor number ( " + std::to_string( neighbor_linear )
							+ " ) >= system size ( " + std::to_string( system_size ) + " )" );
					}
					ArrayVectorStorage< DIMS, SizeType > configuration( DIMS );
#ifdef _DEBUG
					size_t * const halo_coords_end = configuration.data() + DIMS;
#endif
					std::fill_n( configuration.begin(), DIMS, 0 );

					for( size_t _dim = DIMS; _dim > 0; _dim-- ) {
						// each iteration looks for the base element along a dimension via the number of neighbors
						// each element has: once previous_neighs reaches neighbor_linear, the corresponding
						// base element is found; if the control reaches the end, this means it must explore
						// the following dimension to find the base element: this is why dimensions are explored
						// starting from the highest, because moving along a higher dimension means "skipping"
						// more neighbors; then the search "zooms in"to a smaller dimension to find the base element

						// start from highest dimension
						const size_t dimension = _dim - 1;
						// how many elements along this dimension
						const size_t dimension_size = sizes[ dimension ];
						// configurations of neighbors along this dimension
						// (e.g., corner, edge; or edge, inner element)
						const NDimVector< SizeType, SizeType, DynamicVectorStorage< SizeType > > & neighbors =
							neighbors_per_dimension[ dimension ];

						// coordinate to modify to identify each configuration
						SizeType * const halo_coords_begin = configuration.data() + dimension;
#ifdef _DEBUG
						std::cout << "DIMENSION " << dimension << std::endl << "- setup - neighbour "
							<< neighbor_linear << std::endl << "\thalo : ";
						print_sequence( halo_coords_begin, halo_coords_end ) << std::endl;
#endif
						size_t h = 0; // configuration type along this dimension
						size_t previous_neighs = 0;
						*halo_coords_begin = h;
						// account for neighbors in the first elements along the dimension, within halo distance:
						// these elements have a number of neighbors that depends on the distance h
						// and on the configuration
						size_t halo_max_neighs = neighbors.at( halo_coords_begin );
						while( h < halo && neighbor_linear >= previous_neighs + halo_max_neighs ) {
							h++;
							*halo_coords_begin = h;
							previous_neighs += halo_max_neighs;
							halo_max_neighs = neighbors.at( halo_coords_begin );
						}
#ifdef _DEBUG
						std::cout << "- initial halo - neighbour " << neighbor_linear
							<< std::endl << "\th " << h << std::endl << "\thalo : ";
						print_sequence( halo_coords_begin, halo_coords_end ) << std::endl;
						std::cout << "\thalo_max_neighs " << halo_max_neighs << std::endl;
#endif
						if( h < halo ) {
							// we have already counted enough neighbors: neighbor_linear is thus a neighbor
							// of one of the first (< halo) elements along this dimension: go to next dimension
							element_vector[ dimension ] = h;
							neighbor_linear -= previous_neighs;
#ifdef _DEBUG
							std::cout << "end neighbour " << neighbor_linear << std::endl;
#endif
							continue;
						}
						// saturation occurred: the base element is beyond the halo: go on with the search

						// inner elements have the same number of neighbors halo_max_neighs: compute
						// the base element via division
						const size_t distance_from_halo = ( neighbor_linear - previous_neighs ) / halo_max_neighs;
#ifdef _DEBUG
						std::cout << "- before middle elements - neighbour " << neighbor_linear << std::endl
								  << "\tprevious_neighs " << previous_neighs << std::endl
								  << "\thalo_max_neighs " << halo_max_neighs << std::endl
								  << "\tdistance_from_halo " << distance_from_halo << std::endl
								  << "\tdimension_size " << dimension_size << std::endl;
#endif
						if( distance_from_halo < dimension_size - 2 * halo ) {
							// the base element is one of the internal elements along this dimension:
							// hence return its diatance from the halo + the halo itself (= distance from
							// beginning of the space)
							element_vector[ dimension ] = distance_from_halo + halo;
							neighbor_linear -= ( previous_neighs + distance_from_halo * halo_max_neighs );
#ifdef _DEBUG
							std::cout << "end neighbour " << neighbor_linear << std::endl;
#endif
							continue;
						}
						// base element is even beyond inner elements, it might be among the elements at the end,
						// which also have different numbers of neighbors (specular to initial elements)
						previous_neighs += ( dimension_size - 2 * halo ) * halo_max_neighs;
#ifdef _DEBUG
						std::cout << "- after middle elements -neighbour " << neighbor_linear << std::endl;
						std::cout << "\tprevious_neighs " << previous_neighs << std::endl;
						std::cout << "\thalo_max_neighs " << halo_max_neighs << std::endl;
#endif
						// look for base the element at the end of the dimension: specular search to beginning,
						// just with h decreasing
						h = halo - 1;
						*halo_coords_begin = h;
						halo_max_neighs = neighbors.at( halo_coords_begin );
						while( h > 0 && neighbor_linear >= previous_neighs + halo_max_neighs ) {
							h--;
							*halo_coords_begin = h;
							previous_neighs += halo_max_neighs;
							halo_max_neighs = neighbors.at( halo_coords_begin );
						}
						neighbor_linear -= previous_neighs;
#ifdef _DEBUG
						std::cout << "- final halo - neighbour " << neighbor_linear << std::endl;
						std::cout << "\tadding h " << h << " previous_neighs " << previous_neighs << std::endl;
#endif
						// ( dimension_size - 1 ) because coordinates are 0-based and neighbor
						// is "inside" range [ previous_neighs, previous_neighs + halo_max_neighs ]
						element_vector[ dimension ] = dimension_size - 1 - h;
#ifdef _DEBUG
						std::cout << "end neighbour " << neighbor_linear << std::endl;
#endif
					}
					return neighbor_linear;
				}
			};

		} // namespace multigrid
	}     // namespace utils
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_LINEARIZED_HALO_NDIM_SYSTEM
