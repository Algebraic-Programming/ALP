
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

#ifndef _H_GRB_ALGORITHMS_GEOMETRY_LINEARIZED_HALO_NDIM_ITERATOR
#define _H_GRB_ALGORITHMS_GEOMETRY_LINEARIZED_HALO_NDIM_ITERATOR

#include <cstddef>
#include <vector>
#include <utility>
#include <iterator>
#include <limits>
#include <cstddef>

#include "linearized_ndim_system.hpp"
#include "array_vector_storage.hpp"
#include "linearized_ndim_iterator.hpp"

namespace grb {
	namespace utils {
		namespace geometry {

			// forward declaration
			template<
				typename SizeType,
				size_t DIMS
			> class LinearizedHaloNDimSystem;

			template<
				typename SizeType,
				size_t DIMS
			> class LinearizedHaloNDimIterator {

				using SystemType = LinearizedHaloNDimSystem< SizeType, DIMS >;
				using VectorType = ArrayVectorStorage< SizeType, DIMS >;
				using VectorIteratorType = LinearizedNDimIterator< SizeType, VectorType >;

			public:
				//using VectorType = typename VectorIteratorType::VectorType;
				using ConstVectorReference = typename VectorIteratorType::ConstVectorReference;
				using SelfType = LinearizedHaloNDimIterator< SizeType, DIMS >;

				struct HaloNDimElement {
				private:

					// for linearization
					const SystemType* _system;

					// for iteration
					VectorIteratorType _element_iter; // coordinates iterator

					//VectorType* _element;
					//size_t _coordinates_linear;
					VectorType _neighbor; //the current neighbor
					//size_t _neighbor_linear;
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

					//HaloNDimElement& operator=( HaloNDimElement&& ) = delete;

					ConstVectorReference get_element() const {
						return this->_element_iter->get_position();
					}

					size_t get_element_linear() const {
						return this->_system->ndim_to_linear( this->_element_iter->get_position() );
					}

					ConstVectorReference get_neighbor() const {
						return this->_neighbor;
					}

					size_t get_neighbor_linear() const {
						return this->_system->ndim_to_linear( this->_neighbor );
					}

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

			private:
				HaloNDimElement _point;
				LinearizedNDimSystem< SizeType, VectorType > _neighbors_linearizer;
				VectorIteratorType _neighbor_iter; // iterator in the sub-space of neighbors (0-based)
				VectorType _neighbors_start;
				VectorIteratorType _neighbor_end;

				inline void __update_neighbor() {
					for( size_t i{0}; i < DIMS; i++ ) {
						//(this->_point)._neighbor[i] = this->_neighbors_start[i] + (*(this->_neighbor_iter))[i];
						this->_point._neighbor[i] = this->_neighbors_start[i] + this->_neighbor_iter->get_position()[i];
					}
				}

				/*
				void __update_neighbor_linear() {
					(this->_point)._neighbor_linear =
						this->_system.ndim_to_linear( this->_point._neighbor );
				}
				*/

				inline void on_neighbor_iter_update() {
					this->__update_neighbor();
					//this->__update_neighbor_linear();
				}

				/*
				void __update_coordinates_linear() {
					(this->_point)._coordinates_linear =
						this->_system.ndim_to_linear( *this->_element_iter );
				}
				*/

				void on_element_update() {
					//this->__update_coordinates_linear();
					// reset everything
					VectorType neighbors_range( DIMS );
					this->_point._system->compute_neighbors_range(
						//*(this->_point._element_iter),
						this->_point._element_iter->get_position(),
						this->_neighbors_start,
						neighbors_range
					);
					/*
					std::cout << "\t=== start ";
					print( this->_neighbors_start ) << " range ";
					print( neighbors_range )  << std::endl;
					*/
					// re-target _neighbors_linearizer
					this->_neighbors_linearizer.retarget( neighbors_range );
				}

				void on_element_advance() {
					this->on_element_update();

					this->_neighbor_iter = VectorIteratorType( this->_neighbors_linearizer );
					this->_neighbor_end = VectorIteratorType::make_system_end_iterator( this->_neighbors_linearizer );

					this->on_neighbor_iter_update();
				}

			public:

				LinearizedHaloNDimIterator() = delete;

				LinearizedHaloNDimIterator( const SystemType& system ) noexcept :
					_point( system ),
					_neighbors_linearizer( DIMS, system.halo() + 1 ),
					_neighbor_iter( this->_neighbors_linearizer ),
					_neighbors_start( DIMS ),
					_neighbor_end( VectorIteratorType::make_system_end_iterator( this->_neighbors_linearizer ) )
				{
					std::fill_n( this->_neighbors_start.begin(), DIMS, 0 );
				}


				/*
				LinearizedHaloNDimIterator( const LinearizedHaloNDimIterator< SizeType, DIMS >& original ) noexcept:
					_coordinates_linearizer( original._coordinates_linearizer ),
					_halo( original._halo ),
					_dimension_limits( original._dimension_limits ),
					_neighbors_linearizer( original._neighbors_linearizer ),
					_element_iter( original._element_iter ),
					_neighbor_iter( original._neighbor_iter ),
					_neighbor_end( original._neighbor_end ),
					_neighbors_start( original._neighbors_start ),
					_point( original._point ) {}
				*/

				LinearizedHaloNDimIterator( const SelfType & ) = default;

				//LinearizedHaloNDimIterator( SelfType &&original ) = delete;

				/*
				LinearizedHaloNDimIterator< SizeType, DIMS >& operator=(
					const LinearizedHaloNDimIterator< SizeType, DIMS >& original ) noexcept {
					this->_coordinates_linearizer = original._coordinates_linearizer;
					this->_halo = original._halo;
					this->_dimension_limits = original._dimension_limits;
					this->_neighbors_linearizer = original._neighbors_linearizer;
					this->_element_iter = original._element_iter;
					this->_coordinates_linear = original._coordinates_linear;
					this->_neighbor_iter = original._neighbor_iter;
					this->_neighbor_end = original._neighbor_end;
					this->_neighbor = original._neighbor;
					this->_neighbors_start = original._neighbors_start;
					this->_neighbor_linear = original._neighbor_linear;
				}
				*/

				SelfType & operator=( const SelfType & ) = default;

				//SelfType & operator=( SelfType && ) = delete;

				bool operator!=( const SelfType &other ) const {
					//return (this->_point)._coordinates_linear != (other._point)._coordinates_linear
					//	|| (this->_point)._neighbor_linear != (other._point)._neighbor_linear;
					return this->_point._position != other._point._position; // use linear coordinate
				}

				reference operator*() const {
					return this->_point;
				}

				pointer operator->() const {
					return &(this->_point);
				}

				bool has_more_neighbours() const {
					return this->_neighbor_iter != this->_neighbor_end;
				}

				void next_neighbour() {
					/*
					std::cout << "sizes: " << this->_neighbors_linearizer.get_sizes()
						<< " offset " << this->_neighbor_iter->get_position() << " -> "
						<< this->_neighbors_linearizer.ndim_to_linear_offset( this->_neighbor_iter->get_position() )
						<< std::endl;
					*/
					++(this->_neighbor_iter);
					this->on_neighbor_iter_update();
					this->_point._position++;
				}

				bool has_more_elements() const {
					return this->_point.get_element_linear() != (this->_point._system)->base_system_size();
				}

				void next_element() {
					size_t num_neighbours = this->_neighbors_linearizer.system_size();
					size_t neighbour_position_offset =
						this->_neighbors_linearizer.ndim_to_linear_offset( this->_neighbor_iter->get_position() );
					// std::cout << " num_neighbours " << num_neighbours << " offset " << neighbour_position_offset << std::endl;
					++(this->_point._element_iter);
					this->on_element_advance();
					// this->_point._position++;
					this->_point._position -= neighbour_position_offset;
					this->_point._position += num_neighbours;
				}

				SelfType & operator++() noexcept {
					++(this->_neighbor_iter);
					if( !has_more_neighbours() ) {
						++(this->_point._element_iter);
						//this->_coordinates_linear = this->_coordinates_linearizer.ndim_to_linear( this->_element_iter );
						this->on_element_advance();

					} else {
						this->on_neighbor_iter_update();
					}
					this->_point._position++;
					return *this;
				}

				SelfType & operator+=( size_t offset ) {
					if( offset == 1UL ) {
						return this->operator++();
					}
					const size_t final_position { this->_point._position + offset };
					if( final_position > this->_point._system->halo_system_size() ) {
						throw std::range_error( "neighbor linear value beyond system" );
					}
					VectorType final_element( DIMS );
					size_t neighbor_index{ (this->_point._system->neighbour_linear_to_element( final_position, final_element )) };

					// std::cout << "\t=== element " << offset << " -- ";
					// std::cout << final_element[0] << " " << final_element[0] << std::endl;

					this->_point._element_iter = VectorIteratorType( *this->_point._system, final_element.cbegin() );
					//this->_point._element = &( *this->_element_iter );
					this->_point._position = final_position;

					this->on_element_update();
					this->_neighbors_linearizer.linear_to_ndim( neighbor_index, final_element );

					this->_neighbor_iter = VectorIteratorType( this->_neighbors_linearizer, final_element.cbegin() );
					this->_neighbor_end = VectorIteratorType::make_system_end_iterator( this->_neighbors_linearizer );
					this->on_neighbor_iter_update();

					return *this;
				}

				difference_type operator-( const SelfType &other ) const {
					/*
					if( _point.get_position() < a_point.get_position() ) {
						throw std::invalid_argument( "first iterator is in a lower position than second" );
					}
					*/
					size_t a_pos{ _point.get_position() }, b_pos{ other._point.get_position() };
					// std::cout << "diff " << a_pos << " - " << b_pos << std::endl;
					size_t lowest{ std::min( a_pos, b_pos ) }, highest{ std::max( a_pos, b_pos )};
					using diff_t = typename LinearizedHaloNDimIterator< SizeType, DIMS >::difference_type;

					if( highest - lowest > static_cast< size_t >(
						std::numeric_limits< diff_t >::max() ) ) {
						throw std::invalid_argument( "iterators are too distant" );
					}

					return ( static_cast< diff_t >( a_pos - b_pos ) );
				}




				// implementation depending on logic in operator++
				static SelfType make_system_end_iterator( const SystemType& system ) {
					SelfType result( system );

					/*
					std::cout << "result 0: element ";
					print(result->get_element()) << " neighbor ";
					print(result->get_neighbor())  << std::endl;
					*/

					// go to the very first point outside of space
					result._point._element_iter = VectorIteratorType::make_system_end_iterator( system );
					/*
					std::cout << "result 1: element ";
					print(result->get_element()) << " neighbor ";
					print(result->get_neighbor())  << std::endl;
					*/

					result.on_element_advance();
					result._point._position = system.halo_system_size();
					//std::cout << "got sys size " << system.halo_system_size() << std::endl;

					return result;
				}

			};

			/*
			template< typename SizeType, size_t DIMS > LinearizedHaloNDimIterator< SizeType, DIMS >
				operator+( const LinearizedHaloNDimIterator< SizeType, DIMS >& original, size_t increment ) {
				LinearizedHaloNDimIterator< SizeType, DIMS > res( original );
				return ( res += increment );
			}
			*/


		} // namespace geometry
	} // namespace utils
} // namespace grb

#endif // _H_GRB_ALGORITHMS_GEOMETRY_LINEARIZED_HALO_NDIM_ITERATOR
