
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
 * @file linearized_halo_ndim_iterator.cpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Definition of LinearizedHaloNDimSystem.
 */

#ifndef _H_GRB_ALGORITHMS_MULTIGRID_LINEARIZED_HALO_NDIM_ITERATOR
#define _H_GRB_ALGORITHMS_MULTIGRID_LINEARIZED_HALO_NDIM_ITERATOR

#include <cstddef>
#include <vector>
#include <iterator>
#include <limits>
#include <cstddef>

#include <graphblas/utils/iterators/utils.hpp>

#include "linearized_ndim_system.hpp"
#include "array_vector_storage.hpp"
#include "linearized_ndim_iterator.hpp"

namespace grb {
	namespace utils {
		namespace multigrid {

			// forward declaration
			template<
				size_t DIMS,
				typename SizeType
			> class LinearizedHaloNDimSystem;

			/**
			 * Class to iterate over the \b neighbors of a system with halo: by advancing the iterator,
			 * the user can traverse all neighbors of all elements one-by-one, in order, for example, to
			 * emit all possible copies element-neighbor.
			 *
			 * Example: for a 2-dimensional 3 x 3 system with halo 1, with elements numbered as in
			 *
			 * 0 1 2
			 * 3 4 5
			 * 6 7 8
			 *
			 * the emitted couples <element-neighbor> are:
			 *
			 * 0-0, 0-1, 0-3, 0-4; 1-0, 1-1, 1-2, 1-3, 1-4, 1-5; 2-1, 2-2, 2-4, 2-5;
			 * 3-0, 3-1, 3-3, 3-4; 4-0, 4-1, 4-2, 4-3, 4-4, 4-5, 4-6, 4-7, 4-8; and so on.
			 *
			 * It implements two interfaces for iteration. The first is a standard STL-like
			 * interface meeting the random-access requirements, with operators \a ++, \a *, \a ->,
			 * \a +=, \a -, \a ==; these facilities iterate over \b all neighbors of the underlying system,
			 * automatically updating the corresponding element the neighbor is associated to.
			 * The second interface is a custom (Java-like) one that allows to iterate separately over elements
			 * and their neighbors: the user can query whether more elements exist, move to the next element,
			 * iterate over the neighbors of the current element, query whether more neighbors exist for the
			 * current element.
			 *
			 * The state of this structure essentially contains:
			 *
			 * 1. a const-pointer to a LinearizedHaloNDimSystem<DIMS,SizeType> object, storing the geometry
			 * information of the N-dimensional system.
			 * 2. the iterator to the current element (which in turn provides the element's vector
			 *  and linear coordinates)
			 * 3. the vector coordinate of the current neighbor
			 * 4. the linear coordinate of the current neighbor
			 * 5. information about the current element's neighbors space:
			 *   1. the N-dimensional sub-space of neighbors w.r.t. the current element: this
			 *    LinearizedHaloNDimSystem<DIMS,SizeType> object stores the sizes of the neighbors's sub-space
			 *    centered around the current element (at most <em>2 * halo + 1</em> per dimension, if the current
			 *    element is an inner one); hence, it computes coordinates and provides iterators that are
			 *    \b relative to the current element
			 *   2. vector coordinates of the first neighbor of the current element, in the main system
			 *    (i.e. \b not relative); this allows computing any neighbor as the sum of this vector
			 *    plus its relative coordinates in the neighbors' sub-space
			 *   3. iterator to the current neighbor, built out of the relative sub-space, to actually iterate
			 *    over the current element's neighbors
			 *   4. iterator to the last neighbor of the current element, to stop the iteration over neighbors
			 *    and advance to the next element.
			 *
			 * The above-mentioned methods to advance the iterator \c this (over neighbors or elements)
			 * take care of updating these structures properly, keeping the state \b always coherent.
			 *
			 * @tparam DIMS syztem number of dimensions
			 * @tparam SizeType type of coordinates and of sizes (must be large enough to describe the size
			 * of the system along each direction)
			 */
			template<
				size_t DIMS,
				typename SizeType
			> class LinearizedHaloNDimIterator {

				using SystemType = LinearizedHaloNDimSystem< DIMS, SizeType >;
				using VectorType = ArrayVectorStorage< DIMS, SizeType >;
				using VectorIteratorType = LinearizedNDimIterator< SizeType, VectorType >;

			public:
				using ConstVectorReference = typename VectorIteratorType::ConstVectorReference;
				using SelfType = LinearizedHaloNDimIterator< DIMS, SizeType >;

				/**
				 * Structure holding the information about a neighbor in a system: its linear
				 * and vector coordinates and the element it is neighbor of (in the form of both
				 * linear and vectoor coordinate).
				 */
				struct HaloNDimElement {
				private:

					// for linearization
					const SystemType* _system;

					// for iteration
					VectorIteratorType _element_iter; // coordinates iterator

					VectorType _neighbor; //the current neighbor
					SizeType _position;

				public:
					friend SelfType;

					HaloNDimElement() = delete;

					HaloNDimElement( const HaloNDimElement& ) = default;

					HaloNDimElement( HaloNDimElement&& ) = delete;

					HaloNDimElement( const SystemType& system ) noexcept :
						_system( &system ),
						_element_iter( system ),
						_neighbor( DIMS ),
						_position( 0 )
					{
						std::fill_n( this->_neighbor.begin(), DIMS, 0 );
					}

					HaloNDimElement& operator=( const HaloNDimElement& ) = default;

					/**
					 * Get the element as vector coordinates.
					 */
					ConstVectorReference get_element() const {
						return this->_element_iter->get_position();
					}

					/**
					 * Get the element as linear coordinates.
					 */
					size_t get_element_linear() const {
						return this->_system->ndim_to_linear( this->_element_iter->get_position() );
					}

					/**
					 * Get the neighbor as vector coordinates.
					 */
					ConstVectorReference get_neighbor() const {
						return this->_neighbor;
					}

					/**
					 * Get the neighbor as linear coordinates.
					 */
					size_t get_neighbor_linear() const {
						return this->_system->ndim_to_linear( this->_neighbor );
					}

					/**
					 * Get the (unique) neighbor number in the system.
					 */
					SizeType get_position() const {
						return this->_position;
					}
				};

				// interface for std::random_access_iterator
				using iterator_category = std::random_access_iterator_tag;
				using value_type = HaloNDimElement;
				using pointer = const HaloNDimElement*;
				using reference = const HaloNDimElement&;
				using difference_type = signed long;

				LinearizedHaloNDimIterator() = delete;

				/**
				 * Construct a new LinearizedHaloNDimIterator object from the underlying system
				 * \p system (whose geometry information is used to iterate). The constructed object
				 * points to the first neighbor of the first element, i.e. the one with vector coordinates
				 * \a [0,0,...,0].
				 *
				 * IF \p system is not valid anymore, then also \c this is not.
				 */
				LinearizedHaloNDimIterator( const SystemType& system ) noexcept :
					_point( system ),
					_neighbors_subspace( DIMS, system.halo() + 1 ),
					_neighbors_start( DIMS ),
					_neighbor_iter( this->_neighbors_subspace ),
					_neighbor_end( VectorIteratorType::make_system_end_iterator( this->_neighbors_subspace ) )
				{
					std::fill_n( this->_neighbors_start.begin(), DIMS, 0 );
				}

				LinearizedHaloNDimIterator( const SelfType & ) = default;

				SelfType & operator=( const SelfType & ) = default;

				bool operator!=( const SelfType &other ) const {
					return this->_point._position != other._point._position; // use linear coordinate
				}

				reference operator*() const {
					return this->_point;
				}

				pointer operator->() const {
					return &(this->_point);
				}

				/**
				 * Tells whether the current element has more neighbor available (on which the user
				 * has not iterated yet).
				 */
				bool has_more_neighbours() const {
					return this->_neighbor_iter != this->_neighbor_end;
				}

				/**
				 * Moves \c this to point to the next neighbor (if any, exception otherwise).
				 *
				 * Does \b not advance the element, which should be done manually via #next_element().
				 */
				void next_neighbour() {
					if( !has_more_neighbours() ) {
						throw std::out_of_range("the current element has no more neighbors");
					}
					++(this->_neighbor_iter);
					this->on_neighbor_iter_update();
					this->_point._position++;
				}

				/**
				 * Tells whether the system has more elements.
				 */
				bool has_more_elements() const {
					return this->_point.get_element_linear() != (this->_point._system)->base_system_size();
				}

				/**
				 * Moves \c this to point to the next element, setting the neighbor as the first one.
				 */
				void next_element() {
					if( !has_more_elements() ) {
						throw std::out_of_range("the system has no more elements");
					}
					size_t num_neighbours = this->_neighbors_subspace.system_size();
					size_t neighbour_position_offset =
						this->_neighbors_subspace.ndim_to_linear( this->_neighbor_iter->get_position() );
					++(this->_point._element_iter);
					this->on_element_advance();
					this->_point._position -= neighbour_position_offset;
					this->_point._position += num_neighbours;
				}

				/**
				 * Moves \c this to point to the next neighbor, also advancing the element if needed.
				 */
				SelfType & operator++() noexcept {
					++(this->_neighbor_iter);
					if( !has_more_neighbours() ) {
						++(this->_point._element_iter);
						this->on_element_advance();

					} else {
						this->on_neighbor_iter_update();
					}
					this->_point._position++;
					return *this;
				}

				/**
				 * Moves \c this ahead of \p offste neighbors, also advancing the element if necessary.
				 */
				SelfType & operator+=( size_t offset ) {
					if( offset == 1UL ) {
						return this->operator++();
					}
					const size_t final_position = this->_point._position + offset;
					if( final_position > this->_point._system->halo_system_size() ) {
						throw std::range_error( "neighbor linear value beyond system" );
					}
					VectorType final_element( DIMS );
					size_t neighbor_index = (this->_point._system->neighbour_linear_to_element( final_position, final_element ));

					this->_point._element_iter = VectorIteratorType( *this->_point._system, final_element.cbegin() );
					this->_point._position = final_position;

					this->on_element_update();
					this->_neighbors_subspace.linear_to_ndim( neighbor_index, final_element );

					this->_neighbor_iter = VectorIteratorType( this->_neighbors_subspace, final_element.cbegin() );
					this->_neighbor_end = VectorIteratorType::make_system_end_iterator( this->_neighbors_subspace );
					this->on_neighbor_iter_update();

					return *this;
				}

				/**
				 * Returns the difference between \c this and \p other in the linear space of neighbors,
				 * i.e. how many times \p other must be advanced in order to point to the same neighbor of \c this.
				 *
				 * It throws if the result cannot be stored as a difference_type variable.
				 */
				difference_type operator-( const SelfType &other ) const {
					return grb::utils::compute_signed_distance< difference_type, SizeType >(
						_point.get_position(), other._point.get_position() );
				}

				/**
				 * Utility to build an iterator to the end of the system \p system.
				 *
				 * The implementation depends on the logic of operator++.
				 */
				static SelfType make_system_end_iterator( const SystemType& system ) {
					SelfType result( system );
					// go to the very first point outside of space
					result._point._element_iter = VectorIteratorType::make_system_end_iterator( system );
					result.on_element_advance();
					result._point._position = system.halo_system_size();
					return result;
				}

			private:
				HaloNDimElement _point;
				LinearizedNDimSystem< SizeType, VectorType > _neighbors_subspace;
				VectorType _neighbors_start;
				VectorIteratorType _neighbor_iter; // iterator in the sub-space of neighbors (0-based)
				VectorIteratorType _neighbor_end;

				/**
				 * To be called when the iterator pointing to the neighbor is updated in order to update
				 * the actual neighbor's coordinates.
				 */
				inline void on_neighbor_iter_update() {
					for( size_t i = 0; i < DIMS; i++ ) {
						this->_point._neighbor[i] = this->_neighbors_start[i]
							+ this->_neighbor_iter->get_position()[i];
					}
				}

				/**
				 * To be called after the iterator pointing to the element is updated in order to
				 * reset the information about the neighbor.
				 */
				void on_element_update() {
					// reset everything
					VectorType neighbors_range( DIMS );
					this->_point._system->compute_neighbors_range(
						this->_point._element_iter->get_position(),
						this->_neighbors_start,
						neighbors_range
					);
					// re-target _neighbors_subspace
					this->_neighbors_subspace.retarget( neighbors_range );
				}

				/**
				 * To be called after the iterator pointing to the element is updated in order to update
				 * all information about the neighbor, like iterator, sorrounding halo and coordinates.
				 */
				void on_element_advance() {
					this->on_element_update();

					this->_neighbor_iter = VectorIteratorType( this->_neighbors_subspace );
					this->_neighbor_end = VectorIteratorType::make_system_end_iterator( this->_neighbors_subspace );

					this->on_neighbor_iter_update();
				}
			};

		} // namespace multigrid
	} // namespace utils
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_LINEARIZED_HALO_NDIM_ITERATOR
