
#ifndef _NDIM_ITERATOR_H_
#define _NDIM_ITERATOR_H_

#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <limits>

#include "array_vector_storage.hpp"


namespace grb {
	namespace utils {
		namespace geometry {

// forward declaration for default
template< typename T, typename StorageT > class linearized_ndim_system;

template< typename T, typename StorageT > class linearized_ndim_iterator {
public:

	using storage_t = StorageT;
	using lin_t = linearized_ndim_system< T, storage_t >;
	using const_vector_reference = const storage_t&;
	using self_t = linearized_ndim_iterator< T, StorageT >;

	struct ndim_point {
	private:

		const lin_t* system; // pointer because of copy assignment
		storage_t coords;

	public:

		friend self_t;

		ndim_point() = delete;

		ndim_point( const ndim_point& ) = default;

		ndim_point( ndim_point&& ) = delete;

		ndim_point( const lin_t& _system ) noexcept :
			system( &_system ),
			coords( _system.dimensions() )
		{
			std::fill_n( this->coords.begin(), _system.dimensions(), 0 );
		}

		ndim_point& operator=( const ndim_point& ) = default;

		inline const_vector_reference get_position() const {
			return coords;
		}

		std::size_t get_linear_position() const {
			return system->ndim_to_linear( coords );
		}
	};


	// interface for std::random_access_iterator
	using iterator_category = std::random_access_iterator_tag;
	using value_type = ndim_point;
	using pointer = const value_type*;
	using reference = const value_type&;
	using difference_type = signed long;

	linearized_ndim_iterator( const lin_t &_system ) noexcept :
		_p( _system )
	{}

	template< typename IterT > linearized_ndim_iterator( const lin_t &_system, IterT begin ) noexcept :
		_p( _system )
	{
		std::copy_n( begin, _system.dimensions(), this->_p.coords.begin() );
	}

	linearized_ndim_iterator() = delete;

	linearized_ndim_iterator( const self_t& original ):
		_p( original._p ) {}

	self_t& operator=( const self_t& original ) = default;

	//linearized_ndim_iterator( self_t&& original ) = delete;

	//self_t operator=( self_t&& ) = delete;

	~linearized_ndim_iterator() {}

    self_t & operator++() noexcept {
		bool rewind{ true };
		// rewind only the first N-1 coordinates
		for( std::size_t i { 0 }; i < this->_p.system->dimensions() - 1 && rewind; i++ ) {
			T& coord = this->_p.coords[ i ];
			// must rewind dimension if we wrap-around
			/*
			T new_coord = ( coord + 1 ) % this->_p.system->get_sizes()[ i ];
			rewind = new_coord < coord;
			coord = new_coord;
			*/
			T plus = coord + 1;
			rewind = plus >= this->_p.system->get_sizes()[ i ];
			coord = rewind ? 0 : plus;
		}
		// if we still have to rewind, increment the last coordinate, which is unbounded
		if( rewind ) {
			this->_p.coords[ this->_p.system->dimensions() - 1 ]++;
		}
		return *this;
	}

    self_t & operator+=( std::size_t offset ) {
		std::size_t linear{ _p.get_linear_position() + offset };
		if( linear > _p.system->system_size() ) {
			throw std::invalid_argument("increment is too large");
		}
		_p.system->linear_to_ndim( linear, _p.coords );
		return *this;
	}

	difference_type operator-( const self_t &other ) const {
		std::size_t a_pos{ _p.get_linear_position() },
			b_pos{ other._p.get_linear_position() };
		std::size_t lowest{ std::min( a_pos, b_pos ) }, highest{ std::max( a_pos, b_pos )};

		if( highest - lowest > static_cast< std::size_t >(
			std::numeric_limits< difference_type >::max() ) ) {
			throw std::invalid_argument( "iterators are too distant" );
		}

		return ( static_cast< difference_type >( a_pos - b_pos ) );
	}

	reference operator*() const {
        return this->_p;
    }

	pointer operator->() const {
		return &( this->_p );
	}

    bool operator!=( const self_t &o ) const {
		const std::size_t dims{ this->_p.system->dimensions() };
		if( dims != o._p.system->dimensions() ) {
			throw std::invalid_argument("system sizes do not match");
		}
        bool equal{ true };
		for( std::size_t i{0}; i < dims && equal; i++) {
			equal &= ( this->_p.coords[i] == o._p.coords[i] );
		}
		return !equal;
    }

	// implementation depending on logic in operator++
	static self_t
		make_system_end_iterator( const lin_t &_system ) {
		// fill with 0s
		self_t iter( _system );
		std::size_t last{ iter->system->dimensions() - 1 };
		// store last size in last position
		iter._p.coords[ last ] = iter->system->get_sizes()[ last ];
		return iter;
	}

private:
	ndim_point _p;

};

		} // namespace geometry
	} // namespace utils
} // namespace grb

#endif // _NDIM_ITERATOR_H_
