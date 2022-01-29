
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
 * @date 2022-01-07
 */

#ifndef _H_GRB_UTILS_OUTPUT_VERIFICATION
#define _H_GRB_UTILS_OUTPUT_VERIFICATION

#include <graphblas.hpp>

#include <limits>

#include <assert.h>


template< typename T, enum grb::Backend B >
int vector_verification( const grb::PinnedVector< T, B > &output_vector, char *truth_filename, double c1, double c2 ) {
	assert( truth_filename != nullptr );
	assert( c1 > 0 ); assert( c1 < 1 );
	assert( c2 > 0 ); assert( c2 < 1 );

	// open verification file
	FILE *in = fopen( truth_filename, "r" );

	if( in == nullptr ) {
		std::cerr << "Could not open the file \"" << truth_filename << "\"." << std::endl;
		return 10;
	}

	// read the truth output vector from the input verification file
	size_t n = output_vector.length();
	double * const truth = new double[ n ];
	if( truth == nullptr ) {
		std::cerr << "Could not allocate necessary buffer" << std::endl;
		return 20;
	}
	for( size_t i = 0; i < n; i++ ) {
		if( fscanf( in, "%lf", &(truth[ i ]) ) != 1 ) {
			std::cerr << "The verification file looks incomplete." << std::endl;
			delete [] truth;
			return 30;
		}
	}

	// close verification file
	if( fclose( in ) != 0 ) {
		std::cerr << "I/O warning: closing verification file failed." << std::endl;
	}

	// compute magnitudes
	double magnitude2 = 0;
	double magnitudeInf = 0;
	for( size_t i = 0; i < n; ++i ) {
		magnitude2 += truth[ i ] * truth[ i ];
		magnitudeInf = fabs( truth[ i ] ) > magnitudeInf ?
			fabs( truth[ i ] ) :
			magnitudeInf;
	}

	// convert the Pinned Vector into raw data
	assert( output_vector.length() == n );
	double * const raw_output_vector = new double[ n ];
	bool * const written_to = new bool[ n ];
	if( raw_output_vector == nullptr || written_to == nullptr ) {
		std::cerr << "Could not allocate necessary buffers" << std::endl;
		delete [] truth;
		return 40;
	}
	for( size_t i = 0; i < n; i++ ) {
		written_to[ i ] = false;
	}
	for( size_t k = 0; k < n; k++ ) {
		const size_t i = output_vector.index( k );
		assert( i < n );
		assert( !written_to[ i ] );
		raw_output_vector[ i ] = output_vector.mask( k ) ? output_vector[ k ] : 0;
		written_to[ i ] = true;
	}
#ifndef NDEBUG
	{
		bool all_ok = true;
		for( size_t i = 0; i < n; ++i ) {
			if( !written_to[ i ] ) { all_ok = false; }
		}
		assert( all_ok );
	}
#endif

	// compute the norm-2
	double norm2 = 0;
	for( size_t i = 0; i < n; i++ ) {
		norm2 += ( raw_output_vector[ i ] - truth[ i ] ) * ( raw_output_vector[ i ] - truth[ i ] );
	}
	assert( norm2 >= 0 );
	norm2 = sqrt( norm2 );

	// compute the norm-inf
	int ret = 0;
	constexpr const double eps = std::numeric_limits< double >::epsilon();
	double norm_inf = fabs( raw_output_vector[ 0 ] - truth[ 0 ] );
	size_t norm_inf_at = 0;
	for( size_t i = 1; i < n; i++ ) {
		const double curInfNorm = fabs( raw_output_vector[ i ] - truth[ i ] );
		if( curInfNorm > c2 * magnitudeInf + eps ) {
			std::cerr << "Output vector failed inf-norm verification at index " << i << ":\n"
				<< "\tmeasured absolute error at this index: " << curInfNorm << "\n"
				<< "\tthe inf-norm of the truth vector is " << magnitudeInf
				<< ", requested relative tolerance is " << c2 << "\n"
				<< "\tHence " << curInfNorm << " <= " << (c2 * magnitudeInf + eps) << " fails at this index\n";
			ret = 50;
		}
		if( curInfNorm > norm_inf ) {
			norm_inf = curInfNorm;
			norm_inf_at = i;
		}
	}
	assert( norm_inf >= 0 );
	assert( norm_inf_at < n );

	// free local buffers
	assert( truth != nullptr );
	assert( raw_output_vector != nullptr );
	delete [] truth;
	delete [] written_to;
	delete [] raw_output_vector;

	// perform check and return
	if( norm2 > c1 * magnitude2 + n * eps ) {
		std::cerr << "Output vector failed 2-norm verification:\n"
			<< "\t2-norm is " << norm2 << ".\n"
			<< "\t2-norm is larger than the specified relative tolerance of " << c1 << ".\n"
			<< "\t2-norm magnitude of the truth vector is " << magnitude2 << ", hence "
			<< norm2 << " <= " << (c1 * magnitude2 + n * eps) << " failed.\n";
		ret = ret == 50 ? 70 : 60;
	} else {
		std::cerr << "Info: output vector passed 2-norm verification\n";
	}
	if( norm_inf > c2 * magnitudeInf + eps ) {
		std::cerr << "Output vector failed inf-norm verification:\n"
		<< "\tinf-norm is " << norm_inf << " at index " << norm_inf_at << "\n"
		<< "\tinf-norm is larger than the specified relative tolerance of " << c2 << "\n"
		<< "\tinf-norm of the truth vector is " << magnitudeInf << ", hence "
		<< norm_inf << " <= " << (c2 * magnitudeInf + eps) << " failed\n";
	} else {
		std::cerr << "Info: output vector passed inf-norm verification\n";
	}
	return ret;
}

#endif // _H_GRB_UTILS_OUTPUT_VERIFICATION

