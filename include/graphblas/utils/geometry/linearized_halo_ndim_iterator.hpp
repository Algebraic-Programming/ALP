
#ifndef _LINEARIZED_HALO_NDIM_ITERATOR_H_
#define _LINEARIZED_HALO_NDIM_ITERATOR_H_

#include <cstddef>
#include <vector>
#include <utility>
#include <iterator>
#include <limits>

#include "linearized_ndim_system.hpp"
#include "array_vector_storage.hpp"
#include "linearized_ndim_iterator.hpp"

namespace grb {
	namespace utils {
		namespace geometry {

// forward declaration
template< typename CoordT, std::size_t DIMS > class linearized_halo_ndim_system;

template< typename CoordT, std::size_t DIMS > class linearized_halo_ndim_iterator {

	using system_t = linearized_halo_ndim_system< CoordT, DIMS >;
	using vector_t = array_vector_storage< CoordT, DIMS >;
	using vector_iter = linearized_ndim_iterator< CoordT, vector_t >;
public:

	//using vector_t = typename vector_iter::vector_t;
	using const_vector_reference = typename vector_iter::const_vector_reference;



	struct halo_ndim_point {
	private:

		// for linearization
		const system_t* _system;

		// for iteration
		vector_iter _element_iter; // coordinates iterator

		//vector_t* _element;
		//std::size_t _coordinates_linear;
		vector_t _neighbor; //the actual neighbor
		//std::size_t _neighbor_linear;
		CoordT _position;

	public:

		friend linearized_halo_ndim_iterator< CoordT, DIMS>;

		halo_ndim_point() = delete;

		halo_ndim_point( const halo_ndim_point& ) = default;

		halo_ndim_point( halo_ndim_point&& ) = delete;

		halo_ndim_point( const system_t& system ) noexcept :
			_system( &system ),
			_element_iter( system ),
			_neighbor( DIMS ),
			_position( 0 )
		{
			std::fill_n( this->_neighbor.begin(), DIMS, 0 );
		}

		halo_ndim_point& operator=( const halo_ndim_point& ) = default;

		//halo_ndim_point& operator=( halo_ndim_point&& ) = delete;

		const_vector_reference get_element() const {
			return this->_element_iter->get_position();
		}

		std::size_t get_element_linear() const {
			return this->_system->ndim_to_linear( this->_element_iter->get_position() );
		}

		const_vector_reference get_neighbor() const {
			return this->_neighbor;
		}

		std::size_t get_neighbor_linear() const {
			return this->_system->ndim_to_linear( this->_neighbor );
		}

		CoordT get_position() const {
			return this->_position;
		}
	};






	using const_point_reference = const struct halo_ndim_point&;
	using const_point_pointer = const struct halo_ndim_point*;

	// interface for std::random_access_iterator
	using iterator_category = std::random_access_iterator_tag;
	using value_type = halo_ndim_point;
	using pointer = const halo_ndim_point*;
	using reference = const halo_ndim_point&;
	using difference_type = signed long;

private:

	halo_ndim_point _point;
	linearized_ndim_system< CoordT, vector_t > _neighbors_linearizer;
	vector_iter _neighbor_iter; // iterator in the sub-space of neighbors (0-based)
	vector_t _neighbors_start;
	vector_iter _neighbor_end;

	inline void __update_neighbor() {
		for( std::size_t i{0}; i < DIMS; i++ ) {
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
		vector_t neighbors_range( DIMS );
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

		this->_neighbor_iter = vector_iter( this->_neighbors_linearizer );
		this->_neighbor_end = vector_iter::make_system_end_iterator( this->_neighbors_linearizer );

		this->on_neighbor_iter_update();
	}

public:

	linearized_halo_ndim_iterator() = delete;

	linearized_halo_ndim_iterator( const system_t& system ) noexcept :
		_point( system ),
		_neighbors_linearizer( DIMS, system.halo() + 1 ),
		_neighbor_iter( this->_neighbors_linearizer ),
		_neighbors_start( DIMS ),
		_neighbor_end( vector_iter::make_system_end_iterator( this->_neighbors_linearizer ) )
	{
		std::fill_n( this->_neighbors_start.begin(), DIMS, 0 );
	}


	/*
	linearized_halo_ndim_iterator( const linearized_halo_ndim_iterator< CoordT, DIMS >& original ) noexcept:
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

	linearized_halo_ndim_iterator( const linearized_halo_ndim_iterator< CoordT, DIMS >& ) = default;

	//linearized_halo_ndim_iterator( linearized_halo_ndim_iterator< CoordT, DIMS >&& original ) = delete;

	/*
	linearized_halo_ndim_iterator< CoordT, DIMS >& operator=(
		const linearized_halo_ndim_iterator< CoordT, DIMS >& original ) noexcept {
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

	linearized_halo_ndim_iterator< CoordT, DIMS >& operator=( const linearized_halo_ndim_iterator< CoordT, DIMS >& ) = default;

	//linearized_halo_ndim_iterator< CoordT, DIMS >& operator=( linearized_halo_ndim_iterator< CoordT, DIMS >&& ) = delete;

	bool operator!=( const linearized_halo_ndim_iterator< CoordT, DIMS >& other ) const {
		//return (this->_point)._coordinates_linear != (other._point)._coordinates_linear
		//	|| (this->_point)._neighbor_linear != (other._point)._neighbor_linear;
		return this->_point._position != other._point._position; // use linear coordinate
	}

	const_point_reference operator*() const {
		return this->_point;
	}

	const_point_pointer operator->() const {
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
		std::size_t num_neighbours = this->_neighbors_linearizer.system_size();
		std::size_t neighbour_position_offset =
			this->_neighbors_linearizer.ndim_to_linear_offset( this->_neighbor_iter->get_position() );
		// std::cout << " num_neighbours " << num_neighbours << " offset " << neighbour_position_offset << std::endl;
		++(this->_point._element_iter);
		this->on_element_advance();
		// this->_point._position++;
		this->_point._position -= neighbour_position_offset;
		this->_point._position += num_neighbours;
	}

	linearized_halo_ndim_iterator< CoordT, DIMS >& operator++() noexcept {
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



	linearized_halo_ndim_iterator< CoordT, DIMS >& operator+=( std::size_t offset ) {
		if( offset == 1UL ) {
			return this->operator++();
		}
		const std::size_t final_position { this->_point._position + offset };
		if( final_position > this->_point._system->halo_system_size() ) {
			throw std::range_error( "neighbor linear value beyond system" );
		}
		vector_t final_element( DIMS );
		std::size_t neighbor_index{ (this->_point._system->neighbour_linear_to_element( final_position, final_element )) };

		// std::cout << "\t=== element " << offset << " -- ";
		// std::cout << final_element[0] << " " << final_element[0] << std::endl;

		this->_point._element_iter = vector_iter( *this->_point._system, final_element.cbegin() );
		//this->_point._element = &( *this->_element_iter );
		this->_point._position = final_position;

		this->on_element_update();
		this->_neighbors_linearizer.linear_to_ndim( neighbor_index, final_element );

		this->_neighbor_iter = vector_iter( this->_neighbors_linearizer, final_element.cbegin() );
		this->_neighbor_end = vector_iter::make_system_end_iterator( this->_neighbors_linearizer );
		this->on_neighbor_iter_update();

		return *this;
	}

	difference_type operator-( const linearized_halo_ndim_iterator< CoordT, DIMS >& other ) const {
		/*
		if( _point.get_position() < a_point.get_position() ) {
			throw std::invalid_argument( "first iterator is in a lower position than second" );
		}
		*/
		std::size_t a_pos{ _point.get_position() }, b_pos{ other._point.get_position() };
		// std::cout << "diff " << a_pos << " - " << b_pos << std::endl;
		std::size_t lowest{ std::min( a_pos, b_pos ) }, highest{ std::max( a_pos, b_pos )};
		using diff_t = typename linearized_halo_ndim_iterator< CoordT, DIMS >::difference_type;

		if( highest - lowest > static_cast< std::size_t >(
			std::numeric_limits< diff_t >::max() ) ) {
			throw std::invalid_argument( "iterators are too distant" );
		}

		return ( static_cast< diff_t >( a_pos - b_pos ) );
	}




	// implementation depending on logic in operator++
	static linearized_halo_ndim_iterator< CoordT, DIMS > make_system_end_iterator(
		const system_t& system
	) {
		linearized_halo_ndim_iterator< CoordT, DIMS > result( system );

		/*
		std::cout << "result 0: element ";
		print(result->get_element()) << " neighbor ";
		print(result->get_neighbor())  << std::endl;
		*/

		// go to the very first point outside of space
		result._point._element_iter = vector_iter::make_system_end_iterator( system );
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
template< typename CoordT, std::size_t DIMS > linearized_halo_ndim_iterator< CoordT, DIMS >
	operator+( const linearized_halo_ndim_iterator< CoordT, DIMS >& original, std::size_t increment ) {
	linearized_halo_ndim_iterator< CoordT, DIMS > res( original );
	return ( res += increment );
}
*/


		} // namespace geometry
	} // namespace utils
} // namespace grb

#endif // _LINEARIZED_HALO_NDIM_ITERATOR_H_
