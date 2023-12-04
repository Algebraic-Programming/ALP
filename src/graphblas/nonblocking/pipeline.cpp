
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
 * @file
 *
 * Provides a pipeline for nonblocking execution.
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#include <graphblas/config.hpp>
#include <graphblas/backends.hpp>

#include <graphblas/nonblocking/pipeline.hpp>
#include <graphblas/nonblocking/analytic_model.hpp>

// #define _LOCAL_DEBUG


using namespace grb::internal;

Pipeline::Pipeline() {
	constexpr const size_t initial_container_cap =
		config::PIPELINE::max_containers;
	constexpr const size_t initial_stage_cap = config::PIPELINE::max_depth;
	constexpr const size_t initial_tile_cap = config::PIPELINE::max_tiles;

	// an initially empty pipeline does not contain any primitive
	contains_out_of_place_primitive = false;

	// the value 0 for the size of containers indicates
	// either an empty pipeline or a pipeline of empty containers
	containers_size = 0;
	size_of_data_type = 0;

	// reserve sufficient memory to avoid dynamic memory allocation at run-time
	stages.reserve( initial_stage_cap );
	opcodes.reserve( initial_stage_cap );
	lower_bound.reserve( initial_tile_cap );
	upper_bound.reserve( initial_tile_cap );
	input_output_intersection.reserve( initial_container_cap );

	// the below looped-insert-then-clear simulates a reserve and can be reasonably
	// expected to work for an optimised STL implementation. However, it would be
	// nicer if we had our own set container that supports a guaranteed reserve(),
	// as this simulation might not always work as expected.
	for( size_t i = 0; i < initial_container_cap; ++i ) {
		void * const dummy = reinterpret_cast< void * >( i );
		Coordinates< nonblocking > * const dumCoor =
			reinterpret_cast< Coordinates< nonblocking > * >( i );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
		const Coordinates< nonblocking > * const dumCCoor =
			reinterpret_cast< const Coordinates< nonblocking > * >( i );
#endif
		accessed_coordinates.insert( dumCoor );
		input_vectors.insert( dummy );
		output_vectors.insert( dummy );
		vxm_input_vectors.insert( dummy );
		input_matrices.insert( dummy );
		out_of_place_output_coordinates.insert( dumCoor );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
		already_dense_coordinates.insert( dumCCoor );
#endif
		dense_descr_coordinates.insert( dumCoor );
	}
	accessed_coordinates.clear();
	input_vectors.clear();
	output_vectors.clear();
	vxm_input_vectors.clear();
	input_matrices.clear();
	out_of_place_output_coordinates.clear();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
	already_dense_coordinates.clear();
#endif
	dense_descr_coordinates.clear();

	no_warning_emitted_yet = true;
}

Pipeline::Pipeline( const Pipeline &pipeline ) :
	stages( pipeline.stages ), opcodes( pipeline.opcodes),
	accessed_coordinates( pipeline.accessed_coordinates ),
	input_vectors( pipeline.input_vectors ),
	output_vectors( pipeline.output_vectors ),
	vxm_input_vectors( pipeline.vxm_input_vectors ),
	input_matrices( pipeline.input_matrices ),
	out_of_place_output_coordinates( pipeline.out_of_place_output_coordinates ),
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
	already_dense_coordinates( pipeline.already_dense_coordinates ),
#endif
	dense_descr_coordinates( pipeline.dense_descr_coordinates ),
	no_warning_emitted_yet( pipeline.no_warning_emitted_yet )
{
	contains_out_of_place_primitive = pipeline.contains_out_of_place_primitive;
	containers_size = pipeline.containers_size;
	size_of_data_type = pipeline.size_of_data_type;

	// no action is requred regarding the input_output_intersection that is only
	// used temporarily during the pipeline execution
}

Pipeline::Pipeline( Pipeline &&pipeline ) noexcept :
	stages( std::move( pipeline.stages ) ),
	opcodes( std::move( pipeline.opcodes ) ),
	accessed_coordinates( std::move( pipeline.accessed_coordinates ) ),
	input_vectors( std::move( pipeline.input_vectors ) ),
	output_vectors( std::move( pipeline.output_vectors ) ),
	vxm_input_vectors( std::move( pipeline.vxm_input_vectors ) ),
	input_matrices( std::move( pipeline.input_matrices ) ),
	out_of_place_output_coordinates(
		std::move( pipeline.out_of_place_output_coordinates ) ),
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
	already_dense_coordinates( std::move( pipeline.already_dense_coordinates ) ),
#endif
	dense_descr_coordinates( std::move( pipeline.dense_descr_coordinates ) ),
	no_warning_emitted_yet( pipeline.no_warning_emitted_yet )
{
	contains_out_of_place_primitive = pipeline.contains_out_of_place_primitive;
	containers_size = pipeline.containers_size;
	size_of_data_type = pipeline.size_of_data_type;

	// no action is requred regarding the input_output_intersection that is only
	// used temporarily during the pipeline execution

	pipeline.contains_out_of_place_primitive = false;
	pipeline.containers_size = 0;
	pipeline.size_of_data_type = 0;
}

Pipeline &Pipeline::operator=( const Pipeline &pipeline ) {
	contains_out_of_place_primitive = pipeline.contains_out_of_place_primitive;
	containers_size = pipeline.containers_size;
	size_of_data_type = pipeline.size_of_data_type;

	stages = pipeline.stages;
	opcodes = pipeline.opcodes;
	accessed_coordinates = pipeline.accessed_coordinates;
	input_vectors = pipeline.input_vectors;
	output_vectors = pipeline.output_vectors;
	vxm_input_vectors = pipeline.vxm_input_vectors;
	input_matrices = pipeline.input_matrices;
	out_of_place_output_coordinates = pipeline.out_of_place_output_coordinates;
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
	already_dense_coordinates = pipeline.already_dense_coordinates;
#endif
	dense_descr_coordinates = pipeline.dense_descr_coordinates;
	no_warning_emitted_yet = pipeline.no_warning_emitted_yet;

	// no action is requred regarding the input_output_intersection that is only
	// used temporarily during the pipeline execution

	return *this;
}

Pipeline &Pipeline::operator=( Pipeline &&pipeline ) {
	contains_out_of_place_primitive = pipeline.contains_out_of_place_primitive;
	containers_size = pipeline.containers_size;
	size_of_data_type = pipeline.size_of_data_type;

	stages = std::move( pipeline.stages );
	opcodes = std::move( pipeline.opcodes );

	accessed_coordinates = std::move( pipeline.accessed_coordinates );
	input_vectors = std::move( pipeline.input_vectors );
	output_vectors = std::move( pipeline.output_vectors );
	vxm_input_vectors = std::move( pipeline.vxm_input_vectors );
	input_matrices = std::move( pipeline.input_matrices );
	out_of_place_output_coordinates =
		std::move( pipeline.out_of_place_output_coordinates );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
	already_dense_coordinates = std::move( pipeline.already_dense_coordinates );
#endif
	dense_descr_coordinates = std::move( pipeline.dense_descr_coordinates );
	no_warning_emitted_yet = pipeline.no_warning_emitted_yet;

	// no action is requred regarding the input_output_intersection that is only
	// used temporarily during the pipeline execution

	pipeline.contains_out_of_place_primitive = false;
	pipeline.containers_size = 0;
	pipeline.size_of_data_type = 0;

	return *this;
}

void Pipeline::warnIfExceeded() {
	if( no_warning_emitted_yet && config::PIPELINE::warn_if_exceeded ) {
		if( stages.size() > config::PIPELINE::max_depth ||
			opcodes.size() > config::PIPELINE::max_depth
		) {
			std::cerr << "Warning: the number of pipeline stages has been increased "
				<< "past the initial reserved number of stages\n";
		}
		if( accessed_coordinates.size() > config::PIPELINE::max_containers ||
			input_vectors.size() > config::PIPELINE::max_containers ||
			output_vectors.size() > config::PIPELINE::max_containers ||
			vxm_input_vectors.size() > config::PIPELINE::max_containers ||
			input_matrices.size() > config::PIPELINE::max_containers ||
			out_of_place_output_coordinates.size() > config::PIPELINE::max_containers ||
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			already_dense_coordinates.size() > config::PIPELINE::max_containers ||
#endif
			dense_descr_coordinates.size() > config::PIPELINE::max_containers
		) {
			std::cerr << "Warning: the number of pipeline containers has increased past "
				<< "the initial number of reserved containers.\n";
		}
		if( lower_bound.size() > config::PIPELINE::max_tiles ||
			upper_bound.size() > config::PIPELINE::max_tiles ||
			input_output_intersection.size() > config::PIPELINE::max_tiles
		) {
			std::cerr << "Warning: the number of pipeline tiles has increased past the "
				<< "initial number of reserved tiles.\n";
		}
		no_warning_emitted_yet = false;
	}
}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
bool Pipeline::allAlreadyDenseVectors() const {
	return all_already_dense_vectors;
}
#endif

bool Pipeline::empty() const {
	return stages.empty();
}

typename std::vector< Pipeline::stage_type >::iterator Pipeline::pbegin() {
	return stages.begin();
}

typename std::vector< Pipeline::stage_type >::iterator Pipeline::pend() {
	return stages.end();
}

typename std::set< Coordinates< grb::nonblocking > * >::iterator
Pipeline::vbegin() {
	return accessed_coordinates.begin();
}

typename std::set< Coordinates< grb::nonblocking > * >::iterator
Pipeline::vend() {
	return accessed_coordinates.end();
}

size_t Pipeline::accessedCoordinatesSize() const {
	return accessed_coordinates.size();
}

size_t Pipeline::getNumStages() const {
	return stages.size();
}

size_t Pipeline::getContainersSize() const {
	return containers_size;
}

void Pipeline::addStage(
		const Pipeline::stage_type &&func, const Opcode opcode,
		const size_t n, const size_t data_type_size,
		const bool dense_descr, const bool dense_mask,
		void * const output_vector_ptr, void * const output_aux_vector_ptr,
		Coordinates< nonblocking > * const coor_output_ptr,
		Coordinates< nonblocking > * const coor_output_aux_ptr,
		const void * const input_a_ptr, const void * const input_b_ptr,
		const void * const input_c_ptr, const void * const input_d_ptr,
		const Coordinates< nonblocking > * const coor_a_ptr,
		const Coordinates< nonblocking > * const coor_b_ptr,
		const Coordinates< nonblocking > * const coor_c_ptr,
		const Coordinates< nonblocking > * const coor_d_ptr,
		const void * const input_matrix
) {
	assert( stages.size() != 0 || containers_size == 0);

	if( stages.size() == 0 ) {
		containers_size = n;
	}

	// the size of containers and the data type should match
	assert( containers_size == n );

	//TODO (internal issue 617): does the size of data matches for all containers?

	// pipelines may consist of primitives that operate on different data types,
	// e.g., double and bool the analytic model should take into account the
	// different data types and make a proper estimation an easy and perhaps
	// temporary fix is to use the maximum size of the data types involved in a
	// pipeline
	if( data_type_size > size_of_data_type ) {
		size_of_data_type = data_type_size;
	}

	stages.push_back( std::move( func ) );
	opcodes.push_back( opcode );

	if( output_vector_ptr != nullptr ) {
		output_vectors.insert( output_vector_ptr );
	}

	if( output_aux_vector_ptr != nullptr ) {
		output_vectors.insert( output_aux_vector_ptr );
	}

	// special treatment for an SpMV operation as the input must not be overwritten
	// by another stage of the pipeline
	if( opcode == Opcode::BLAS2_VXM_GENERIC ) {

		if( input_a_ptr != nullptr ) {
			input_vectors.insert( input_a_ptr );
			vxm_input_vectors.insert( input_a_ptr );
		}

		if( input_b_ptr != nullptr ) {
			input_vectors.insert( input_b_ptr );
			vxm_input_vectors.insert( input_b_ptr );
		}

		if( input_c_ptr != nullptr ) {
			input_vectors.insert( input_c_ptr );
			vxm_input_vectors.insert( input_c_ptr );
		}

		if( input_d_ptr != nullptr ) {
			input_vectors.insert( input_d_ptr );
			vxm_input_vectors.insert( input_d_ptr );
		}

		// in the current implementation that supports level-1 and level-2 operations
		// a pointer to an input matrix may be passed only by an SpMV operation
		// TODO once level-3 operations are supported, the following code should be
		//      moved
		if( input_matrix != nullptr ) {
			input_matrices.insert( input_matrix );
		}
	} else {
		if( input_a_ptr != nullptr ) {
			input_vectors.insert( input_a_ptr );
		}

		if( input_b_ptr != nullptr ) {
			input_vectors.insert( input_b_ptr );
		}

		if( input_c_ptr != nullptr ) {
			input_vectors.insert( input_c_ptr );
		}

		if( input_d_ptr != nullptr ) {
			input_vectors.insert( input_d_ptr );
		}
	}

	// update all the sets of the pipeline by adding the entries of the new stage
	if( coor_a_ptr != nullptr ) {
		if( dense_descr ) {
			dense_descr_coordinates.insert(
				const_cast< Coordinates< nonblocking > * >( coor_a_ptr ) );
		} else {
			accessed_coordinates.insert(
				const_cast< Coordinates< nonblocking > * >( coor_a_ptr ) );
		}
	}

	if( coor_b_ptr != nullptr ) {
		if( dense_descr ) {
			dense_descr_coordinates.insert(
				const_cast< internal::Coordinates< nonblocking > * >( coor_b_ptr ) );
		} else {
			accessed_coordinates.insert(
				const_cast< internal::Coordinates< nonblocking > * >( coor_b_ptr ) );
		}
	}

	if( coor_c_ptr != nullptr ) {
		if( dense_descr ) {
			dense_descr_coordinates.insert(
				const_cast< internal::Coordinates<nonblocking > * >( coor_c_ptr ) );
		} else {
			accessed_coordinates.insert(
				const_cast< internal::Coordinates< nonblocking > * >( coor_c_ptr ) );
		}
	}

	if( coor_d_ptr != nullptr ) {
		if( dense_descr ) {
			dense_descr_coordinates.insert(
				const_cast< internal::Coordinates< nonblocking > * >( coor_d_ptr ) );
		} else {
			accessed_coordinates.insert(
				const_cast< internal::Coordinates< nonblocking > * >( coor_d_ptr ) );
		}
	}

	// keep track of out-of-place operations that may make a dense vector sparse
	// such operations disable potential optimizations for already dense vectors
	if( opcode == Opcode::BLAS1_EWISEAPPLY ||
		opcode == Opcode::BLAS1_MASKED_EWISEAPPLY ||
		opcode == Opcode::IO_SET_MASKED_SCALAR ||
		opcode == Opcode::IO_SET_VECTOR ||
		opcode == Opcode::IO_SET_MASKED_VECTOR
	) {
		// the output of these specific primitives cannot be nullptr

		if( dense_descr ) {
			dense_descr_coordinates.insert( coor_output_ptr );
		}

		// when the dense descriptor is not provided or the operation is masked
		// there is no guarantee that an already dense vector will remain dense
		// therefore, the pipeline is marked to disable the already dense optimization
		if( !dense_descr || ( !dense_mask && (
			opcode == Opcode::BLAS1_MASKED_EWISEAPPLY ||
			opcode == Opcode::IO_SET_MASKED_SCALAR ||
			opcode == Opcode::IO_SET_MASKED_VECTOR
		) ) ) {
			contains_out_of_place_primitive = true;
			out_of_place_output_coordinates.insert( coor_output_ptr );
			accessed_coordinates.insert( coor_output_ptr );
		}

		// TODO: once UNZIP is complete
		// the second output is always nullptr for the out-of-place primitives that
		// are handled here
		// however, once we have the complete implementation of unzip (which handles
		// sparsity) then need to consider the second output here
	} else {

		// check the first output
		if( coor_output_ptr != nullptr ) {
			if( dense_descr ) {
				dense_descr_coordinates.insert( coor_output_ptr );
			} else {
				accessed_coordinates.insert( coor_output_ptr );
			}
		}

		// check the second output
		if( coor_output_aux_ptr != nullptr ) {
			if( dense_descr ) {
				dense_descr_coordinates.insert( coor_output_aux_ptr );
			} else {
				accessed_coordinates.insert( coor_output_aux_ptr );
			}
		}
	}

	warnIfExceeded();
}

void Pipeline::addeWiseLambdaStage(
	const Pipeline::stage_type &&func, const Opcode opcode,
	const size_t n, const size_t data_type_size,
	const bool dense_descr,
	std::vector< const void * > all_vectors_ptr,
	const Coordinates< nonblocking > * const coor_a_ptr
) {
	(void) data_type_size;

	assert( stages.size() != 0 || containers_size == 0);

	if( stages.size() == 0 ) {
		containers_size = n;
	}

	// the analytic model takes into account the size of data used by an
	// eWiseLambda primitive
	if( data_type_size > size_of_data_type ) {
		size_of_data_type = data_type_size;
	}

	assert( containers_size == n );

	stages.push_back( std::move( func ) );
	opcodes.push_back( opcode );

	// add all vectors accessed by eWiseLambda as output vectors
	for( std::vector< const void *>::iterator it =
		all_vectors_ptr.begin(); it != all_vectors_ptr.end(); ++it
	) {
		output_vectors.insert( *it );
	}

	// add the coordinates for the single vector
	if( coor_a_ptr != nullptr ) {
		if( dense_descr ) {
			dense_descr_coordinates.insert(
				const_cast< Coordinates< nonblocking > * >( coor_a_ptr ) );
		} else {
			accessed_coordinates.insert(
				const_cast< Coordinates< nonblocking > * >( coor_a_ptr ) );
		}
	}

	warnIfExceeded();
}

bool Pipeline::accessesInputVector( const void * const vector ) const {
	return input_vectors.find( vector ) != input_vectors.end();
}

bool Pipeline::accessesOutputVector( const void * const vector ) const {
	return output_vectors.find( vector ) != output_vectors.end();
}

bool Pipeline::accessesVector( const void * const vector ) const {
	return (input_vectors.find( vector ) != input_vectors.end()) ||
		(output_vectors.find( vector ) != output_vectors.end());
}

bool Pipeline::accessesMatrix( const void * const matrix ) const {
	return ( input_matrices.find( matrix ) != input_matrices.end() );
}

bool Pipeline::overwritesVXMInputVectors(
	const void * const output_vector_ptr
) const {
	return vxm_input_vectors.find( output_vector_ptr ) != vxm_input_vectors.end();
}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
bool Pipeline::emptyAlreadyDenseVectors() const {
	return already_dense_coordinates.empty();
}

bool Pipeline::containsAlreadyDenseVector(
	const Coordinates< nonblocking > * const vector_ptr
) const {
	return already_dense_coordinates.find( vector_ptr ) !=
		already_dense_coordinates.end();
}

void Pipeline::markMaybeSparseVector(
	const Coordinates< nonblocking > * const vector_ptr
) {
	// the vector should be marked sparse only if it has not already been marked
	if( already_dense_coordinates.find( vector_ptr ) !=
		already_dense_coordinates.end()
	) {
		// when this method is invoked by an out-of-place primitive
		// disable a potentially enabled dense descriptor
		all_already_dense_vectors = false;
		// and remove the coordinates from the set
		already_dense_coordinates.erase( vector_ptr );
	}
}
#endif

void Pipeline::markMaybeSparseDenseDescriptorVerification(
	Coordinates< nonblocking > * const vector_ptr
) {
	if( dense_descr_coordinates.find( vector_ptr ) !=
		dense_descr_coordinates.end()
	) {
		dense_descr_coordinates.erase( vector_ptr );
	}
}

bool Pipeline::outOfPlaceOutput(
	const internal::Coordinates< nonblocking > * const vector_ptr
) {
	if( out_of_place_output_coordinates.find( vector_ptr ) !=
		out_of_place_output_coordinates.end()
	) {
		return true;
	}

	return false;
}

void Pipeline::merge( Pipeline &pipeline ) {
	// if any of the pipelines contains an out-of-place primitive, the merged
	// pipeline contains as well
	if( pipeline.contains_out_of_place_primitive ) {
		contains_out_of_place_primitive = true;
	}

	// the size of the data accessed in the pipeline is updated based on the
	// maximum of the two merged pipelines
	if( pipeline.size_of_data_type > size_of_data_type ) {
		size_of_data_type = pipeline.size_of_data_type;
	}

	assert( containers_size == pipeline.containers_size );

	// add all the stages into the pipeline by maintaining the relative order
	for(
		std::vector< stage_type >::iterator st = pipeline.stages.begin();
		st != pipeline.stages.end(); st++
	) {
		stages.push_back( std::move( *st ) );
	}

	// add all the opcodes into the pipeline by maintaining the relative order
	for(
		std::vector< Opcode >::iterator ot = pipeline.opcodes.begin();
		ot != pipeline.opcodes.end(); ot++
	) {
		opcodes.push_back( *ot );
	}

	// update all the sets of the pipeline by adding the entries of the new stage
	accessed_coordinates.insert(
		pipeline.accessed_coordinates.begin(), pipeline.accessed_coordinates.end() );

	input_vectors.insert( pipeline.input_vectors.begin(), pipeline.input_vectors.end() );

	output_vectors.insert(
		pipeline.output_vectors.begin(), pipeline.output_vectors.end() );

	vxm_input_vectors.insert(
		pipeline.vxm_input_vectors.begin(), pipeline.vxm_input_vectors.end() );

	input_matrices.insert(
		pipeline.input_matrices.begin(), pipeline.input_matrices.end() );

	out_of_place_output_coordinates.insert(
		pipeline.out_of_place_output_coordinates.begin(),
		pipeline.out_of_place_output_coordinates.end()
	);
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
	already_dense_coordinates.insert( pipeline.already_dense_coordinates.begin(),
		pipeline.already_dense_coordinates.end() );
#endif
	dense_descr_coordinates.insert( pipeline.dense_descr_coordinates.begin(),
		pipeline.dense_descr_coordinates.end() );

	// clear all the sets of the pipeline to mark it as inactive
	pipeline.contains_out_of_place_primitive = false;
	pipeline.containers_size = 0;
	pipeline.size_of_data_type = 0;

	pipeline.stages.clear();
	pipeline.opcodes.clear();
	pipeline.accessed_coordinates.clear();
	pipeline.input_vectors.clear();
	pipeline.output_vectors.clear();
	pipeline.vxm_input_vectors.clear();
	pipeline.input_matrices.clear();
	pipeline.out_of_place_output_coordinates.clear();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
	pipeline.already_dense_coordinates.clear();
#endif
	pipeline.dense_descr_coordinates.clear();

	// no action is requred regarding the input_output_intersection that is only
	// used temporarily during the pipeline execution

	warnIfExceeded();
}

void Pipeline::clear() {
	// after executing the pipeline, the size of vectors should be reset to
	// indicate an empty pipeline
	contains_out_of_place_primitive = false;
	containers_size = 0;
	size_of_data_type = 0;

	stages.clear();
	opcodes.clear();
	accessed_coordinates.clear();
	input_vectors.clear();
	output_vectors.clear();
	vxm_input_vectors.clear();
	input_matrices.clear();
	input_output_intersection.clear();
	out_of_place_output_coordinates.clear();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
	already_dense_coordinates.clear();
#endif
	dense_descr_coordinates.clear();
}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
void Pipeline::buildAlreadyDenseVectors() {
	all_already_dense_vectors = true;

	// the intersection of the sets "dense_descr_coordinates" and
	// "accessed_coordinates" is usually empty except for the output of an
	// out-of-place operation that is always added in the set of
	// "accessed_coordinates" regardless the dense descriptor
	for(
		std::set< Coordinates< nonblocking > * >::iterator it =
			dense_descr_coordinates.begin();
		it != dense_descr_coordinates.end(); ++it
	) {
		if( ( *it )->isDense() ) {
			already_dense_coordinates.insert( *it );
		} else {
			all_already_dense_vectors = false;
		}
	}

	for(
		std::set< Coordinates< nonblocking > * >::iterator it =
			accessed_coordinates.begin();
		it != accessed_coordinates.end(); ++it
	) {
		if( ( *it )->isDense() ) {
			already_dense_coordinates.insert( *it );
		} else {
			all_already_dense_vectors = false;
		}
	}
}
#endif

grb::RC Pipeline::verifyDenseDescriptor() {
#ifdef _NONBLOCKING_DEBUG
	std::cout << "dense descriptor verification using "
		<< dense_descr_coordinates.size()
		<< " accessed vector(s) in the executed pipeline" << std::endl;
#endif

	// the coordinates for all vectors that are accessed in a primitive with the
	// dense descriptor should be dense after the execution of the pipeline
	//
	// otherwise, the dense descriptor was used illegally for vectors that were
	// not dense
	//
	// for all primitives with the dense descriptor, the coordinates of all
	// vectors are added in the set "dense_descr_coordinates" for which the local
	// coordinates are not buit, not accessed, and not updated
	//
	// therefore, a sparse vector can become dense only if the same vector is used
	// in different primitives, i.e., with and without the dense descriptor
	//
	// this is a case that the dense descriptor may be used illegally, but the
	// following code cannot catch it
	for(
		std::set< Coordinates<nonblocking > * >::iterator it =
			dense_descr_coordinates.begin();
		it != dense_descr_coordinates.end(); ++it
	) {
		if( !( *it )->isDense() ) {
			return ILLEGAL;
		}
	}

	return SUCCESS;
}

grb::RC Pipeline::execution() {
	RC ret = SUCCESS;

	// if the pipeline is empty, nothing needs to be executed
	if( pbegin() == pend() ) {
		return ret;
	}

	// if the pipeline operates on empty vectors, nothings needs to be executed
	// all operations stored in the pipeline are cleared and the function returns
	// immediately
	if( containers_size == 0 ) {
		clear();
		return ret;
	}

	// compute the intersection of the input and output vectors that should be
	// subtracted from the number of accessed vectors
	std::set_intersection(
		input_vectors.begin(), input_vectors.end(),
		output_vectors.begin(), output_vectors.end(),
		std::back_inserter( input_output_intersection )
	);

	const size_t num_accessed_vectors = input_vectors.size() +
		output_vectors.size() - input_output_intersection.size();

	assert( num_accessed_vectors > 0 );

	// make use of the analytic model to estimate a proper number of threads and a
	// tile size
	AnalyticModel am( size_of_data_type, containers_size, num_accessed_vectors );

	const size_t nthreads = am.getNumThreads();
	const size_t tile_size = am.getTileSize();
	const size_t num_tiles = am.getNumTiles();

#ifdef _NONBLOCKING_DEBUG
	std::cout << std::endl << "Analytic Model: threads(" << nthreads
		<< "), tile_size(" << tile_size << "), num_tiles(" << num_tiles
		<< ")" << std::endl;
#endif

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
	// build the set of already dense vectors that will be used for optimizations
	// by each primitive during the execution of the pipeline
	buildAlreadyDenseVectors();
#else
	all_already_dense_vectors = true;

	for(
		std::set< internal::Coordinates< nonblocking > * >::iterator vt =
			vbegin();
		vt != vend(); ++vt
	) {
		if( (**vt).isDense() == false ) {
			all_already_dense_vectors = false;
		}
	}
#endif

#ifdef _NONBLOCKING_DEBUG
	std::cout << std::endl << "Pipeline execution: stages(" << stages.size()
		<< "), accessed vectors(" << num_accessed_vectors
		<< "), accessed coordinates(" << accessed_coordinates.size()
		<< "), input vectors(" << input_vectors.size()
		<< "), output vectors(" << output_vectors.size()
		<< "), size of vectors(" << containers_size
		<< "), threads(" << nthreads
		<< "), tile size(" << tile_size
		<< ")" << std::endl;
#endif

	lower_bound.resize( num_tiles );
	upper_bound.resize( num_tiles );
	for( std::set< internal::Coordinates< nonblocking > * >::iterator vt = vbegin(); vt != vend(); ++vt ) {
		auto* coords = *vt;
		coords->_debug_is_counting_sort_done = false;
	}

	// if all vectors are already dense and there is no out-of-place operation to
	// make them sparse we avoid paying the overhead for updating the coordinates
	// for the output vectors
	if( all_already_dense_vectors && !contains_out_of_place_primitive ) {
		// each thread should receive an identifier during the execution of the loop

#ifndef GRB_ALREADY_DENSE_OPTIMIZATION
		for(
			std::set< internal::Coordinates< nonblocking > * >::iterator vt = vbegin();
			vt != vend(); ++vt
		) {
			if ( (**vt).size() != getContainersSize() ) {
				continue;
			}

			(**vt).localCoordinatesInit( am );
		}
#endif

		{ // Initialise the lower and upper bounds
			#pragma omp parallel for schedule(dynamic, config::CACHE_LINE_SIZE::value()) num_threads(nthreads)
			for( size_t tile_id = 0; tile_id < num_tiles; ++tile_id ) {

				config::OMP::localRange(
						lower_bound[tile_id], upper_bound[tile_id],
						0, containers_size, tile_size, tile_id, num_tiles
				);
				assert(lower_bound[tile_id] <= upper_bound[tile_id]);
			}
		}


#if defined(_DEBUG) || defined(_LOCAL_DEBUG)
			fprintf( stderr, "Pipeline::execution(2): check if any of the coordinates will use the search-variant of asyncSubsetInit:\n" );
#endif
#ifndef GRB_ALREADY_DENSE_OPTIMIZATION

		for(
			std::set< internal::Coordinates< nonblocking > * >::iterator vt = vbegin();
			vt != vend(); ++vt
		) {
			if( (**vt).size() != getContainersSize() ) { continue; }

			(**vt).asyncSubsetInit( num_tiles, lower_bound, upper_bound );
		}
#endif
		#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
		for( size_t tile_id = 0; tile_id < num_tiles; ++tile_id ) {

//			// compute the lower and upper bounds
//			config::OMP::localRange(
//				lower_bound[ tile_id ], upper_bound[ tile_id ],
//				0, containers_size, tile_size, tile_id, num_tiles
//			);
//			assert( lower_bound[ tile_id ] <= upper_bound[ tile_id ] );


			RC local_ret = SUCCESS;
			for( std::vector< stage_type >::iterator pt = pbegin();
				pt != pend(); ++pt
			) {
				local_ret = local_ret
					? local_ret
					: (*pt)( *this, lower_bound[ tile_id ], upper_bound[ tile_id ] );
			}
			if( local_ret != SUCCESS ) {
				ret = local_ret;
			}
		}
	} else {

		bool initialized_coordinates = false;

		for(
			std::set< internal::Coordinates< nonblocking > * >::iterator vt = vbegin();
			vt != vend(); ++vt
		) {

			if ( (**vt).size() != getContainersSize() ) {
				continue;
			}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			if( (**vt).isDense() && (
				!contains_out_of_place_primitive || !outOfPlaceOutput( *vt )
			) ) {
				continue;
			}
#endif
			(**vt).localCoordinatesInit( am );
		}

		{ // Initialise the lower and upper bounds
			#pragma omp parallel for schedule(dynamic, config::CACHE_LINE_SIZE::value()) num_threads(nthreads)
			for( size_t tile_id = 0; tile_id < num_tiles; ++tile_id ) {
				config::OMP::localRange(
						lower_bound[tile_id], upper_bound[tile_id],
						0, containers_size, tile_size, tile_id, num_tiles
				);
				assert(lower_bound[tile_id] <= upper_bound[tile_id]);
			}
		}

		{
#if defined(_DEBUG) || defined(_LOCAL_DEBUG)
			fprintf( stderr, "Pipeline::execution(2): check if any of the coordinates will use the search-variant of asyncSubsetInit:\n" );
#endif
			for(
				std::set< internal::Coordinates< nonblocking > * >::iterator vt = vbegin();
				vt != vend(); ++vt
				) {
				if ( (**vt).size() != getContainersSize() ) {
					continue;
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				if( (**vt).isDense() && (
					!contains_out_of_place_primitive || !outOfPlaceOutput( *vt )
				) ) {
					continue;
				}
#endif

				(**vt).asyncSubsetInit( num_tiles, lower_bound, upper_bound );
				initialized_coordinates = true;
			}
		}


		// even if only one vector is sparse, we cannot reuse memory because the first
		// two arguments that we pass to the lambda functions determine whether we
		// reuse memory or not and they cannot vary for different vectors
		#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
		for( size_t tile_id = 0; tile_id < num_tiles; ++tile_id ) {

			RC local_ret = SUCCESS;
			for( std::vector< stage_type >::iterator pt = pbegin();
				pt != pend(); ++pt
			) {
				local_ret = local_ret
					? local_ret
					: (*pt)( *this, lower_bound[ tile_id ], upper_bound[ tile_id ] );
			}
			if( local_ret != SUCCESS ) {
				ret = local_ret;
			}
		}

		if( initialized_coordinates ) {
			bool new_nnz = false;

			// compute the prefix sums for each vector and store them in the last part of
			// _buffer
			// the computation for different vectors may run in parallel but is not
			// preferred to avoid high overhead
			for(
				std::set< internal::Coordinates< nonblocking > * >::iterator vt = vbegin();
				vt != vend(); ++vt
			) {

				// skip as done for the initialization
				if ( (**vt).size() != getContainersSize() ) {
					continue;
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				// skip as done for the initialization
				if( (**vt).isDense() && (
					!contains_out_of_place_primitive || !outOfPlaceOutput( *vt )
				) ) {
					continue;
				}
#endif

				if( (**vt).newNonZeroes() ) {
					new_nnz = true;
					(**vt).prefixSumComputation();
				}
			}

			if( new_nnz ) {
				#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
				for( size_t tile_id = 0; tile_id < num_tiles; ++tile_id ) {
					for(
						std::set< internal::Coordinates< nonblocking > * >::iterator vt =
							vbegin();
						vt != vend(); ++vt
					) {

						// skip as done for the initialization
						if ( (**vt).size() != getContainersSize() ) {
							continue;
						}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
						// skip as done for the initialization
						if( (**vt).isDense() && (
							!contains_out_of_place_primitive || !outOfPlaceOutput( *vt )
						) ) {
							continue;
						}
#endif

						if( (**vt).newNonZeroes() ) {
							(**vt).joinSubset( lower_bound[ tile_id ], upper_bound[ tile_id ] );
						}
					}
				}
			}
		}
	}

	// verify that the dense descriptor was legally used
	ret = ret ? ret : verifyDenseDescriptor();

#ifdef _NONBLOCKING_DEBUG
	if( ret == ILLEGAL ) {
		std::cerr << "error in pipeline execution: the dense descriptor was "
			<< "illegally used" << std::endl;
	}
#endif

	clear();

	return ret;
}

