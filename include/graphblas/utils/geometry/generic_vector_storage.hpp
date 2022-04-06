
#ifndef _GENERIC_VECTOR_STORAGE_H_
#define _GENERIC_VECTOR_STORAGE_H_

#include <cstddef>
#include <algorithm>

namespace grb {
	namespace utils {
		namespace geometry {

template< typename T > class generic_vector_storage {

	std::size_t _dimensions;
	T* _storage;

	void clean() {
		if( this->_storage != nullptr ) {
			delete[] this->_storage;
		}
	}

public:

	using reference = T&;
	using const_reference = const T&;
	using iterator = T*;
	using const_iterator = const T*;
	using pointer = T*;
	using const_pointer = const T*;
	using vector_storage = T*;
	using const_vector_storage = T*;

	generic_vector_storage( std::size_t __dimensions ):
		_dimensions( __dimensions ) {
		if( __dimensions == 0 ) {
			throw std::invalid_argument("dimensions cannot be 0");
		}
		this->_storage = new T[ __dimensions ];
	}

	generic_vector_storage() = delete;

	generic_vector_storage( const generic_vector_storage< T >& o ):
		_dimensions( o._dimensions ), _storage( new T[ o._dimensions ] ) {
		std::copy_n( o._storage, o._dimensions, this->_storage );
	}

	generic_vector_storage( generic_vector_storage< T >&& o ) = delete;

	generic_vector_storage< T >& operator=( const generic_vector_storage< T > &original ) {
		if( original._dimensions != this->_dimensions ) {
			this->clean();
			this->_storage = new T[ original._dimensions];
		}
		this->_dimensions = original._dimensions;
		std::copy_n( original._storage, original._dimensions, this->_storage );
		return *this;
	}

	generic_vector_storage< T >& operator=( generic_vector_storage< T > &&original ) = delete;

	~generic_vector_storage() {
		this->clean();
	}

	std::size_t dimensions() const {
		return this->_dimensions;
	}

	inline iterator begin() {
		return this->_storage;
	}

	inline iterator end() {
		return this->_storage + this->_dimensions;
	}

	inline const_iterator begin() const {
		return this->_storage;
	}

	inline const_iterator end() const {
		return this->_storage + this->_dimensions;
	}

	inline const_iterator cbegin() const {
		return this->_storage;
	}

	inline const_iterator cend() const {
		return this->_storage + this->_dimensions;
	}

	inline vector_storage storage() {
		return this->_storage;
	}

	inline const_vector_storage storage() const {
		return this->_storage;
	}

	inline reference operator[]( std::size_t pos ) {
		return *( this->_storage + pos);
	}

	inline const_reference operator[]( std::size_t pos ) const {
		return *( this->_storage + pos );
	}

};

		} // namespace geometry
	} // namespace utils
} // namespace grb

#endif // _GENERIC_VECTOR_STORAGE_H_
