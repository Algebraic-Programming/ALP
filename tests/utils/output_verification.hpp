
/*
 *   Copyright 2021 Huawei Technologies Co., Ltd.
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
 * @file output_verification.hpp
 * @author Aristeidis Mastoras (aristeidis.mastoras@huawei.com)
 * @author Albert-Jan N. Yzelman (albertjan.yzelman@huawei.com)
 * @date 2022-01-07
 */

#ifndef _H_GRB_UTILS_OUTPUT_VERIFICATION
#define _H_GRB_UTILS_OUTPUT_VERIFICATION

#include <graphblas.hpp>

#include <limits>
#include <cmath>
#include <complex>

#include <assert.h>



/**
 * Attempts to read in a T value from a given file into a given memory
 * location.
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
template< typename fileType, typename T >
int data_fscanf( fileType& in, T * const out ) {
	return !(in >> *out);
};

/**
 * Attempts to read in a complex value from a given file into a given memory
 * location.
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
template< typename fileType, typename T >
int data_fscanf( fileType& in, std::complex< T > * const out ) {
	T x, y;
	if(in >> x >> y){
		*out = std::complex< T >( x, y );
		return 0;
	}else{
		return 1;
	}
};

/**
 * Verifies a dense vector against a ground-truth output vector.
 *
 * Performs verifications using the inf-norm and the 2-norm.
 *
 * @tparam T The type of elements in the dense ALP vector.
 * @tparam B The backend of the given dense vector.
 *
 * @param[in] output_vector  The dense output vector to check.
 * @param[in] truth_filename Path to the file storing the ground-truth vector.
 * @param[in] c1             Relative tolerance to apply for 2-norm
 *                           verification.
 * @param[in] c2             Relative tolerance to apply for inf-norm
 *                           verification.
 *
 * @returns  0 if verification succeeded
 * @returns 10 if the ground truth file could not be opened
 * @returns 20 on memory allocation errors while reading the ground truth file
 * @returns 30 on I/O errors on the ground truth file
 * @returns 40 on memory allocation errors for verification buffers
 * @returns 50 if the \a output_vector was not dense
 * @returns 51 if inf-norm verification failed
 * @returns 52 if both 50 and 51 apply
 * @returns 53 if the computation of the inf-norm failed
 * @returns 54 if both 53 and 50 apply
 * @returns 55 if both 53 and 51 apply
 * @returns 56 if all of 53, 51, and 50 apply
 * @returns 57 if the computation of the 2-norm failed
 * @returns 58 if both 57 and 50 apply
 * @returns 59 if both 57 and 51 apply
 * @returns 60 if all of 57, 51, and 50 apply
 * @returns 61 if both 57 and 53 apply
 * @returns 62 if all of 57, 53, and 50 apply
 * @returns 63 if all of 57, 53, and 51 apply
 * @returns 64 if all of 57, 53, 51, and 50 apply
 * @returns 65 if 2-norm verification failed
 * ...
 * @returns 80 if all of 65, 57, 53, 51, and 50 apply
 *
 * \note Please note that error codes 0, 10, 20, 30, 40, 50, 51, 53, 57, and 65
 *       correspond to individual errors this function can detect. Errors from
 *       number 50 (inclusive) onwards can be detected simultaneously, which
 *       leads to the other error codes that are not all exhaustively
 *       enumerated in the above. The mixed error codes are systematic by
 *       power-of-two offsets.
 */
template< typename T, enum grb::Backend B >
int vector_verification(
	const grb::PinnedVector< T, B > &output_vector,
	const char * const truth_filename,
	const double c1, const double c2
) {
	assert( truth_filename != nullptr );
	assert( c1 > 0 ); assert( c1 < 1 );
	assert( c2 > 0 ); assert( c2 < 1 );
	const constexpr T one = static_cast< T >( 1 );

	// open verification file
	std::ifstream in;
	in.open( truth_filename);

	if( !in.is_open() ) {
		std::cerr << "Could not open the file \"" << truth_filename << "\"."
			<< std::endl;
		return 10;
	}

	// read the truth output vector from the input verification file
	const size_t n = output_vector.size();
	T * const truth = new T[ n ];
	if( truth == nullptr ) {
		std::cerr << "Could not allocate necessary buffer" << std::endl;
		return 20;
	}

	for( size_t i = 0; i < n; i++ ) {
		const int rc = data_fscanf( in, truth + i );
		if( rc != 0 ) {
			std::cerr << "The verification file looks incomplete. " << "Line i = " << i
				<< ", data = " << truth[ i ] << ", rc = " << rc << std::endl;
			delete [] truth;
			return 30;
		}
	}

	// close verification file
	in.close();

	// compute magnitudes
	double magnitude2 = 0;
	double magnitudeInf = 0;
	for( size_t i = 0; i < n; ++i ) {
		magnitude2 +=  std::norm( truth[ i ] );
		magnitudeInf = std::abs( truth[ i ] ) > magnitudeInf ?
			std::abs( truth[ i ] ) :
			magnitudeInf;
	}
	// we assume the ground truth should have a properly computable 2-norm
	assert( magnitude2 >= 0 );
	magnitude2 = sqrt( magnitude2 );

	// convert the Pinned Vector into raw data
	T * const raw_output_vector = new T[ n ];
	bool * const written_to = new bool[ n ];
	if( raw_output_vector == nullptr || written_to == nullptr ) {
		std::cerr << "Could not allocate necessary buffers" << std::endl;
		delete [] truth;
		return 40;
	}
	for( size_t i = 0; i < n; i++ ) {
		written_to[ i ] = false;
	}

	for( size_t k = 0; k < output_vector.nonzeroes(); k++ ) {
		const T &value = output_vector.getNonzeroValue( k, one );
		const size_t index = output_vector.getNonzeroIndex( k );
		assert( index < n );
		assert( !written_to[ index ] );
		raw_output_vector[ index ] = value;
		written_to[ index ] = true;
	}

	// detect accidental zeroes
	int ret = 0;
	bool all_ok = true;
	for( size_t i = 0; i < n; ++i ) {
		if( !written_to[ i ] ) {
			std::cerr << "Output vector index " << i << " does not exist\n";
			all_ok = false;
		}
	}
	if( !all_ok ) {
		std::cerr << "Output vector verification failed; "
			<< "one or more output entries were not written\n";
		ret = 1;
	}

	// compute the norm-inf
	constexpr const double eps = std::numeric_limits< double >::epsilon();
	double norm_inf = fabs( raw_output_vector[ 0 ] - truth[ 0 ] );
	size_t norm_inf_at = 0;
	bool atLeastOneFailed = false;
	// starting the loop from i = 0 entails some redundant computation in
	// curInfNorm, but prevents a bunch of code duplication for checking the
	// output at i = 0. We prefer no code duplication.
	for( size_t i = 0; i < n; i++ ) {
		const double curInfNorm = fabs( raw_output_vector[ i ] - truth[ i ] );
		// if any of the variables involved in the condition below is NaN or -NaN
		// the condition evaluated by the function isless will be false and then
		// the whole condition of the if-statement will be evaluated to true
		// making the verification to fail as expected
		if( !isless( curInfNorm, c2 * magnitudeInf + eps ) ) {
			std::cerr << "Output vector failed inf-norm verification at index "
				<< i << ":\n"
				<< "\tmeasured absolute error at this index: " << curInfNorm << "\n"
				<< "\tthe inf-norm of the truth vector is " << magnitudeInf
				<< ", requested relative tolerance is " << c2 << "\n"
				<< "\tHence " << curInfNorm << " <= " << (c2 * magnitudeInf + eps)
				<< " fails at this index\n";
			atLeastOneFailed = true;
		}
		if( curInfNorm > norm_inf ) {
			norm_inf = curInfNorm;
			norm_inf_at = i;
		}
	}
	if( atLeastOneFailed ) {
		ret += 2;
	}
	assert( norm_inf_at < n );

	// isgreaterequal is used to ensure that the condition norm_inf >= 0
	// will be evaluated to false when norm_inf is equal to NaN or -NaN
	if( !isgreaterequal( norm_inf, 0 ) ) {
		std::cerr << "Output vector failed inf-norm verification:\n"
			<< "\tinf-norm is neither positive nor zero -- "
			<< "it reads " << norm_inf << " instead\n";
		ret += 4;
	}

	// compute the norm-2
	double norm2 = 0;
	for( size_t i = 0; i < n; i++ ) {
		norm2 += std::norm( raw_output_vector[ i ] - truth[ i ] );
	}

	// isgreaterequal is used to ensure that the condition norm2 >= 0
	// will be evaluated to false when norm2 is equal to NaN or -NaN
	if( isgreaterequal( norm2, 0 ) ) {
		norm2 = sqrt( norm2 );
	} else {
		std::cerr << "Output vector failed 2-norm verification:\n"
			<< "\tsquare of the 2-norm is neither positive nor zero -- "
			<< "it reads " << norm2 << " instead\n";
		ret += 8;
	}

	// free local buffers
	assert( truth != nullptr );
	assert( written_to != nullptr );
	assert( raw_output_vector != nullptr );
	delete [] truth;
	delete [] written_to;
	delete [] raw_output_vector;

	// perform check and return
	if( !isless( norm2, c1 * magnitude2 + n * eps ) ) {
		std::cerr << "Output vector failed 2-norm verification:\n"
			<< "\t2-norm is " << norm2 << ".\n"
			<< "\t2-norm is larger than the specified relative tolerance of "
			<< c1 << ".\n"
			<< "\t2-norm magnitude of the truth vector is " << magnitude2 << ", hence "
			<< norm2 << " <= " << (c1 * magnitude2 + n * eps) << " failed\n";
		ret += 16;
	} else {
		std::cerr << "Info: output vector passed 2-norm verification\n"
			<< "\t2-norm is " << norm2 << " which is smaller or equal to the effective "
			<< "relative tolerance of " << (c1 * magnitude2 + n * eps) << "\n";
	}
	if( !isless( norm_inf, c2 * magnitudeInf + eps ) ) {
		std::cerr << "Output vector failed inf-norm verification:\n"
		<< "\tinf-norm is " << norm_inf << " at index " << norm_inf_at << "\n"
		<< "\tinf-norm is larger than the specified relative tolerance of "
		<< c2 << "\n"
		<< "\tinf-norm of the truth vector is " << magnitudeInf << ", hence "
		<< norm_inf << " <= " << (c2 * magnitudeInf + eps) << " failed\n";
		// If this branch triggered, ret += 2 and/or ret +4 must have already been
		// triggered.
		assert( ret > 0 );
		// Hence we need not assign an additional error code here.
	} else {
		std::cerr << "Info: output vector passed inf-norm verification\n"
			<< "\tinf-norm is " << norm_inf << " which is smaller or equal to the "
			<< "effective relative tolerance of " << (c2 * magnitudeInf + eps) << "\n";
	}

	// apply error code offset (if there was an error)
	if( ret > 0 ) {
		ret += 49;
	}

	// done
	return ret;
}

#endif // _H_GRB_UTILS_OUTPUT_VERIFICATION

