
#ifndef _NDIM_SYSTEM_LINEARIZER_H_
#define _NDIM_SYSTEM_LINEARIZER_H_

#include <cstddef>
#include <algorithm>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cassert>
#include <string>

#include "ndim_system.hpp"
#include "linearized_ndim_iterator.hpp"
#include "array_vector_storage.hpp"


namespace grb {
	namespace utils {
		namespace geometry {

template< typename IterIn, typename IterOut >
	std::size_t __compute_offsets( IterIn in_begin, IterIn in_end, IterOut out_begin ) {
	std::size_t prod{1};
	for( ; in_begin != in_end; ++in_begin, ++out_begin ) {
		*out_begin = prod;
		prod *= *in_begin;
	}
	return prod;
}

// container for system sizes, doing only ndim <--> linear translation
template< typename T, typename StorageT > class linearized_ndim_system:
	public ndim_system< T, StorageT > {
public:

	using base_t = ndim_system< T, StorageT >;
	using storage_t = StorageT;
	using self_t = linearized_ndim_system< T, StorageT >;

	using vector_reference = typename base_t::vector_reference;
	using const_vector_reference = typename base_t::const_vector_reference;
	using vector_storage = typename storage_t::vector_storage;
	using const_vector_storage = typename storage_t::const_vector_storage;
	using iterator = linearized_ndim_iterator< T, storage_t >;

	template< typename IterT > linearized_ndim_system( IterT begin, IterT end) noexcept :
		base_t( begin, end ),
		offsets( std::distance( begin, end ) )
	{
		this->_system_size = __compute_offsets( begin, end, this->offsets.begin() ) ;
	}

	linearized_ndim_system() = delete;

	linearized_ndim_system( const self_t &original ) = default;


	linearized_ndim_system( self_t &&original ) noexcept:
		base_t( std::move(original) ), offsets( std::move( original.offsets ) ),
		_system_size( original._system_size ) {
			original._system_size = 0;
	}

	linearized_ndim_system( const std::vector<std::size_t> & _sizes ) noexcept :
		linearized_ndim_system( _sizes.cbegin(), _sizes.cend() ) {}

	linearized_ndim_system( std::size_t _dimensions, std::size_t max_value ) noexcept :
		base_t( _dimensions, max_value ),
		offsets( _dimensions ),
		_system_size( _dimensions )
	{
		T v{1};
		for( std::size_t i{0}; i < _dimensions; i++ ) {
			this->offsets[i] = v;
			v *= max_value;
		}
		this->_system_size = v;
	}

	~linearized_ndim_system() {}

	self_t& operator=( const self_t & ) = default;

	//linearized_ndim_system& operator=( linearized_ndim_system &&original ) = delete;

	inline std::size_t system_size() const {
		return this->_system_size;
	}

	inline const_vector_reference get_offsets() const {
		return this->offsets;
	}

	void linear_to_ndim(std::size_t linear, vector_reference output ) const {
		if( linear > this->_system_size ) {
			throw std::range_error( "linear value beyond system" );
		}
		for( std::size_t _i{ this->offsets.dimensions() }; _i > 0; _i-- ) {
			const std::size_t dim{ _i - 1 };
			const std::size_t coord{ linear / this->offsets[dim] };
			output[dim] = coord;
			linear -= ( coord * this->offsets[dim] );
		}
		assert( linear == 0 );
	}

	std::size_t ndim_to_linear_check( const_vector_reference ndim_vector) const {
		return this->ndim_to_linear_check( ndim_vector.storage() );
	}

	std::size_t ndim_to_linear_check( const_vector_storage ndim_vector ) const {
        std::size_t linear { 0 };
        for( std::size_t i { 0 }; i < this->dimensions(); i++ ) {
			if( ndim_vector[i] >= this->get_sizes()[i] ) {
				throw std::invalid_argument( "input vector beyond system sizes" );
			}
        }
        return ndim_to_linear( ndim_vector );
	}

	std::size_t ndim_to_linear( const_vector_reference ndim_vector) const {
		return this->ndim_to_linear( ndim_vector.storage() );
	}

	std::size_t ndim_to_linear( const_vector_storage ndim_vector ) const {
        std::size_t linear { 0 };
        for( std::size_t i { 0 }; i < this->dimensions(); i++ ) {
            linear += this->offsets[i] * ndim_vector[i];
        }
        return linear;
	}

	std::size_t ndim_to_linear_offset( const_vector_storage ndim_vector ) const {
        std::size_t linear { 0 };
		std::size_t steps{ 1 };
        for( std::size_t i { 0 }; i < this->dimensions(); i++ ) {
            linear += steps * ndim_vector[i];
			steps *= this->_sizes[i];
        }
        return linear;
	}

	// must be same dimensionality
	void retarget( const_vector_reference _new_sizes ) {
		if( _new_sizes.dimensions() != this->_sizes.dimensions() ) {
			throw std::invalid_argument("new system must have same dimensions as previous: new "
				+ std::to_string( _new_sizes.dimensions() ) + ", old "
				+ std::to_string( this->_sizes.dimensions() ) );
		}
		this->_sizes = _new_sizes; // copy
		this->_system_size = __compute_offsets( _new_sizes.begin(), _new_sizes.end(), this->offsets.begin() ) ;
	}

	iterator begin() const {
		return iterator( *this );
	}

	iterator end() const {
		return iterator::make_system_end_iterator( *this );
	}

private:
	storage_t offsets;
	std::size_t _system_size;

};


		} // namespace geometry
	} // namespace utils
} // namespace grb

#endif // _NDIM_SYSTEM_LINEARIZER_H_
