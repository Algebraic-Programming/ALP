
#ifndef _ARRAY_VECTOR_STORAGE_H_
#define _ARRAY_VECTOR_STORAGE_H_

#include <array>
#include <stdexcept>
#include <algorithm>

namespace grb {
	namespace utils {
		namespace geometry {

template< typename T, std::size_t DIMS > class array_vector_storage: public std::array< T, DIMS > {

public:

	using vector_storage = std::array< T, DIMS >&;
	using const_vector_storage = const std::array< T, DIMS >&;

	array_vector_storage( std::size_t _dimensions ) {
		static_assert( DIMS > 0, "cannot allocate 0-sized array" );
		if( _dimensions != DIMS ) {
			throw std::invalid_argument("given dimensions must match the type dimensions");
		}
	}

	array_vector_storage() = delete;

	// only copy constructor/assignment, since there's no external storage
	array_vector_storage( const array_vector_storage< T, DIMS >& o ) noexcept {
		std::copy_n( o.cbegin(), DIMS, this->begin() );
	}

	/*
	array_vector_storage( array_vector_storage< T >&& o ) {
		std::copy_n( o._storage.cbegin(), DIMS, this->_storage.cbegin() );
	}
	*/

	array_vector_storage< T, DIMS >& operator=( const array_vector_storage< T, DIMS > &original ) noexcept {
		std::copy_n( original.begin(), DIMS, this->begin() );
		return *this;
	}

	//array_vector_storage< T, DIMS >& operator=( array_vector_storage< T, DIMS > &&original ) = delete;

	~array_vector_storage() {}

	constexpr std::size_t dimensions() const {
		return DIMS;
	}

	inline vector_storage storage() {
		return *this;
	}

	inline const_vector_storage storage() const {
		return *this;
	}

};

		} // namespace geometry
	} // namespace utils
} // namespace grb

#endif // _ARRAY_VECTOR_STORAGE_H_
