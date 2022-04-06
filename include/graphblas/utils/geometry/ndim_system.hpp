
#ifndef _NDIM_SYSTEM_H_
#define _NDIM_SYSTEM_H_

#include <cstddef>
#include <algorithm>
#include <vector>
#include <utility>

#include "array_vector_storage.hpp"


namespace grb {
	namespace utils {
		namespace geometry {

template< typename T, typename StorageT > class ndim_system {

public:
	using storage_t = StorageT;
	using vector_reference = storage_t&;
	using const_vector_reference = const storage_t&;
	using self_t = ndim_system< T, StorageT >;

	template< typename IterT > ndim_system( IterT begin, IterT end) noexcept :
		_sizes( std::distance( begin, end ) )
	{
		std::copy( begin, end, this->_sizes.begin() );
	}

	ndim_system() = delete;

	ndim_system( const self_t & ) = default;

	ndim_system( const std::vector<std::size_t> & _sizes ) noexcept :
		self_t( _sizes.cbegin(), _sizes.cend() ) {}

	ndim_system( std::size_t _dimensions, std::size_t max_value ) noexcept :
		_sizes( _dimensions )
	{
		std::fill_n( this->_sizes.begin(), _dimensions, max_value );
	}

	ndim_system( self_t &&original ) noexcept: _sizes( std::move( original._sizes ) ) {}

	~ndim_system() {}

	self_t & operator=( const self_t &original ) = default;

	//self_t & operator=( self_t &&original ) = delete;

	inline std::size_t dimensions() const noexcept {
		return _sizes.dimensions();
	}

	inline const_vector_reference get_sizes() const noexcept {
		return this->_sizes;
	}

protected:

	storage_t _sizes;
};

		} // namespace geometry
	} // namespace utils
} // namespace grb

#endif
