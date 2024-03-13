
#ifndef _H_GRB_UTILS_READ_DENSE_VECTOR
#define _H_GRB_UTILS_READ_DENSE_VECTOR

#include <fstream>
#include <complex>
#include <stdexcept>
#include <string>
#include <cctype>

/**
 * Attempts to read in a value from a given file into a given memory
 * location.
 *
 * @tparam T The datatype of the value
 *
 * @param[in]  in  The input file
 * @param[out] out Where to store the read value.
 *
 * @returns 0 on success and 1 on failure.
 *
 * If the function fails, \a out shall not be assigned.
 *
 * \internal This is the overload for reading T data.
 */
template< typename T >
int data_fscanf( std::ifstream &in, T * const out ) {
	return !(in >> *out);
}

/**
 * Attempts to read in a complex value from a given file into a given memory
 * location.
 *
 * @tparam T The data type to be used in the complex value
 *
 * @param[in]  in  The input file
 * @param[out] out Where to store the read value.
 *
 * @returns 0 on success and 1 on failure.
 *
 * If the function fails, \a out shall not be assigned.
 *
 * \internal This is the overload for reading complex data.
 */
template< typename T >
int data_fscanf( std::ifstream &in, std::complex< T > * const out ) {
	T x, y;
	if( in >> x >> y ) {
		*out = std::complex< T >( x, y );
		return 0;
	} else {
		return 1;
	}
}

/**
 * Reads the values stored on different lines of text file \a filename into array \a dst;
 * it reads exactly \a dst_size values, and throws if less values are available or more
 * are present.
 *
 * This function assumes each line contains one and one value only.
 * 
 * @tparam T type of values to read and store
 * @param filename name of input file
 * @param dst pointer to destination array
 * @param dst_size number of values to read
 */
template< typename T >
void read_dense_vector_to_array(
	const char * const filename,
	T * dst,
	size_t dst_size
) {
	if( filename == nullptr ) {
		throw std::runtime_error( "filename is nullptr" );
	}
	if( dst == nullptr ) {
		throw std::runtime_error( "destination vector is nullptr" );
	}
	if( dst_size == 0UL ) {
		throw std::runtime_error( "destination vector size is 0" );
	}
	std::ifstream in( filename);

	if( not in.is_open() ) {
		throw std::runtime_error( "Could not open the file \"" + std::string( filename ) + "\"");
	}

	int rc = 0;
	size_t i = 0;
	for( ; i < dst_size && rc == 0; i++ ) {
		rc = data_fscanf( in, dst + i );
	}
	if( rc != 0 || i < dst_size ) {
		throw std::runtime_error( "file \"" + std::string( filename ) +
			"\" looks incomplete from line " + std::to_string( i - 1 ) );
	}
	while( not in.eof() ) {
		if( std::isalnum( in.get() ) ) {
			throw std::runtime_error( "file \"" + std::string( filename ) +
				"\" has more than " + std::to_string( dst_size ) + " lines" );
		}
	}
}

#endif // _H_GRB_UTILS_READ_DENSE_VECTOR
