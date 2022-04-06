
#ifndef _NDIM_VECTOR_H_
#define _NDIM_VECTOR_H_

#include <utility>
#include <vector>
#include <array>
#include <stdexcept>
#include <cassert>
#include <iterator>
#include <type_traits>

#include "linearized_ndim_system.hpp"

namespace grb {
	namespace utils {
		namespace geometry {

template< typename OutT, typename CoordsT, typename StorageT  > class ndim_vector {

public:

	using const_domain_vector_reference =
		typename linearized_ndim_system< CoordsT, StorageT >::const_vector_reference;
	using domain_vector_storage = typename StorageT::const_vector_storage;
	using domain_iterator = typename linearized_ndim_system< CoordsT, StorageT >::iterator;

private:

	const linearized_ndim_system< CoordsT, StorageT > _linearizer;
	OutT* data;

	inline std::size_t get_coordinate( domain_vector_storage coordinates ) const {
		return this->_linearizer.ndim_to_linear( coordinates );
	}

	inline std::size_t get_coordinate( domain_iterator coordinates ) const {
		return this->_linearizer.ndim_to_linear( coordinates );
	}

    void clean_mem() {
        if ( this->data == nullptr ) {
            delete[] this->data;
        }
    }

public:

	ndim_vector() = delete;

	template< typename IterT > ndim_vector( IterT begin, IterT end): _linearizer( begin, end ) {
		static_assert( std::is_default_constructible< OutT >::value,
			"the stored type is not default constructible" );
		this->data = new OutT[ _linearizer.system_size() ];
	}

	ndim_vector( const std::vector<std::size_t> & _sizes ):
		ndim_vector( _sizes.cbegin(), _sizes.cend() ) {}

	// ndim_vector( const ndim_vector< OutT, CoordsT, StorageT >& original ):
	// 	_linearizer( original._linearizer ) {
    //     this->data = new std::size_t[ original.data_size() ];
	// 	std::copy_n( original.data, original.data_size(), this->data );
    // }
	ndim_vector( const ndim_vector< OutT, CoordsT, StorageT >& original ) = delete;


	ndim_vector( ndim_vector< OutT, CoordsT, StorageT >&& original ) noexcept:
		_linearizer( std::move( original._linearizer ) ) {
        this->data = original.data;
        original.data = nullptr;
    }
	// ndim_vector( ndim_vector< OutT, CoordsT, StorageT >&& original ) = delete;

	ndim_vector< OutT, CoordsT, StorageT >& operator=(
			const ndim_vector< OutT, CoordsT, StorageT > &original ) = delete;

	ndim_vector< OutT, CoordsT, StorageT >& operator=(
			ndim_vector< OutT, CoordsT, StorageT > &&original ) = delete;

    ~ndim_vector() {
        this->clean_mem();
    }

	std::size_t dimensions() const {
		return this->_linearizer.dimensions();
	}

	std::size_t data_size() const {
		return this->_linearizer.system_size();
	}

	inline OutT& at( const_domain_vector_reference coordinates ) {
		return this->data[ this->get_coordinate( coordinates.storage() ) ];
	}

	inline const OutT& at( const_domain_vector_reference coordinates ) const {
		return this->data[ this->get_coordinate( coordinates.storage() ) ];
	}

	inline OutT& at( domain_vector_storage coordinates ) {
		return this->data[ this->get_coordinate( coordinates ) ];
	}

	inline const OutT& at( domain_vector_storage coordinates ) const {
		return this->data[ this->get_coordinate( coordinates ) ];
	}

	domain_iterator domain_begin() const {
		return this->_linearizer.begin();
	}

	domain_iterator domain_end() const {
		return this->_linearizer.end();
	}
};

		} // namespace geometry
	} // namespace utils
} // namespace grb

#endif // _NDIM_VECTOR_H_
