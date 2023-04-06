
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
 * Implements lazy evaluation.
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#include <graphblas/config.hpp>
#include <graphblas/backends.hpp>

#include <graphblas/nonblocking/lazy_evaluation.hpp>


using namespace grb::internal;

namespace grb {

	namespace internal {

		LazyEvaluation le;
	}

}

LazyEvaluation::LazyEvaluation() : warn_if_exceeded( true ) {
	// 32 elements should be sufficient to avoid dynamic memory allocation for the
	// pipelines built at run-time
	pipelines.resize( config::PIPELINE::max_pipelines );
	shared_data_pipelines.resize( config::PIPELINE::max_pipelines );
}

void LazyEvaluation::checkIfExceeded() noexcept {
	if( warn_if_exceeded && config::PIPELINE::warn_if_exceeded ) {
		if( pipelines.size() > config::PIPELINE::max_pipelines ) {
			std::cerr << "Warning: the number of pipelines has exceeded the configured "
				<< "initial capacity.\n";
		}
		warn_if_exceeded = false;
	}
}

grb::RC LazyEvaluation::addStage(
	const Pipeline::stage_type &&func, Opcode opcode,
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
	RC ret = SUCCESS;

	// ensure that nothing is left from previous stages
	shared_data_pipelines.clear();

	if( opcode == Opcode::BLAS2_VXM_GENERIC ) {
		// one output, one input, and maybe another two inputs
		// TODO: the matrix is not currently added as input and thus not taken into
		//       account for data dependence analysis

		// search for pipelines with shared data
		for(
			std::vector< Pipeline >::iterator pt = pipelines.begin();
			pt != pipelines.end(); pt++
		) {

			if( ( *pt ).empty() ) {
				continue;
			}

			bool shared_data_found = false;
			bool pipeline_executed = false;

			if( (*pt).accessesInputVector( output_vector_ptr ) ) {
				if( ( *pt ).overwritesVXMInputVectors( output_vector_ptr ) ) {
					ret = ret ? ret : ( *pt ).execution();
					pipeline_executed = true;
				} else {
					shared_data_found = true;
				}
			} else if( (*pt).accessesOutputVector( output_vector_ptr ) ) {
				shared_data_found = true;
			}

			// it doesn't matter if any shared data found already
			// it's still possibe that the pipeline has to be executed to avoid
			// overwriting the input vectors of SpMV
			if( !pipeline_executed ) {

				// first we check for shared data with the write-access vectors for
				// efficiency and only later we check for read-only vectors that don't
				// enforce pipeline execution
				if( ( *pt ).accessesOutputVector( input_a_ptr ) ) {
					ret = ret ? ret : ( *pt ).execution();
					pipeline_executed = true;
				} else if( !shared_data_found &&
					( *pt ).accessesInputVector( input_a_ptr )
				) {
					shared_data_found = true;
				}

				if( !pipeline_executed ) {
					if( input_b_ptr != nullptr ) {
						if( ( *pt ).accessesOutputVector( input_b_ptr ) ) {
							ret = ret ? ret : ( *pt ).execution();
							pipeline_executed = true;
						} else if( !shared_data_found &&
							( *pt ).accessesInputVector( input_b_ptr )
						) {
							shared_data_found = true;
						}
					}

					if( !pipeline_executed ) {
						if( input_c_ptr != nullptr ) {
							if( ( *pt ).accessesOutputVector( input_c_ptr ) ) {
								ret = ret ? ret : ( *pt ).execution();
								pipeline_executed = true;
							} else if( !shared_data_found &&
								( *pt ).accessesInputVector( input_c_ptr )
							) {
								shared_data_found = true;
							}
						}
					}
				}
			}

			if( !pipeline_executed && shared_data_found ) {
				shared_data_pipelines.push_back( pt );
			}
		}
	}  else {
		for(
			std::vector< Pipeline >::iterator pt = pipelines.begin();
			pt != pipelines.end(); pt++
		) {

			if( ( *pt ).empty() ) {
				continue;
			}

			bool shared_data_found = false;
			bool pipeline_executed = false;

			if( output_vector_ptr != nullptr ) {
				if( (*pt).accessesInputVector( output_vector_ptr ) ) {
					if( ( *pt ).overwritesVXMInputVectors( output_vector_ptr ) ) {
						ret = ret ? ret : ( *pt ).execution();
						pipeline_executed = true;
					} else {
						shared_data_found = true;
					}
				} else if( (*pt).accessesOutputVector( output_vector_ptr ) ) {
					shared_data_found = true;
				}
			}

			if( !pipeline_executed ) {

				if( opcode == Opcode::BLAS1_UNZIP ) {
					// it doesn't matter if have already found shared data
					// it's still necessary to execute the pipeline if the second output of
					// unzip overwrites any of the the input vectors of SpMV

					// check the second output
					if( (*pt).accessesInputVector( output_aux_vector_ptr ) ) {
						if( ( *pt ).overwritesVXMInputVectors( output_aux_vector_ptr ) ) {
							ret = ret ? ret : ( *pt ).execution();
							pipeline_executed = true;
						} else {
							shared_data_found = true;
						}
					} else if( (*pt).accessesOutputVector( output_aux_vector_ptr ) ) {
						shared_data_found = true;
					}
				}

				if( !pipeline_executed ) {
					if( !shared_data_found ) {
						if( ( input_a_ptr != nullptr && (*pt).accessesVector( input_a_ptr ) ) ||
							( input_b_ptr != nullptr && (*pt).accessesVector( input_b_ptr ) ) ||
							( input_c_ptr != nullptr && (*pt).accessesVector( input_c_ptr ) ) ||
							( input_d_ptr != nullptr && (*pt).accessesVector( input_d_ptr ) )
						) {
							shared_data_found = true;
						}
					}

					if( shared_data_found ) {
						shared_data_pipelines.push_back( pt );
					}
				}
			}
		}
	}

#ifdef _DEBUG
	if( !(
		opcode == Opcode::IO_SET_SCALAR || opcode == Opcode::IO_SET_MASKED_SCALAR ||
		opcode == Opcode::IO_SET_VECTOR || opcode == Opcode::IO_SET_MASKED_VECTOR ||
		opcode == Opcode::BLAS1_FOLD_VECTOR_SCALAR_GENERIC ||
		opcode == Opcode::BLAS1_FOLD_SCALAR_VECTOR_GENERIC ||
		opcode == Opcode::BLAS1_FOLD_MASKED_SCALAR_VECTOR_GENERIC ||
		opcode == Opcode::BLAS1_FOLD_VECTOR_VECTOR_GENERIC ||
		opcode == Opcode::BLAS1_FOLD_MASKED_VECTOR_VECTOR_GENERIC ||
		opcode == Opcode::BLAS1_EWISEAPPLY ||
		opcode == Opcode::BLAS1_MASKED_EWISEAPPLY ||
		opcode == Opcode::BLAS1_EWISEMULADD_DISPATCH ||
		opcode == Opcode::BLAS1_DOT_GENERIC ||
		opcode == Opcode::BLAS1_EWISELAMBDA || opcode == Opcode::BLAS1_EWISEMAP ||
		opcode == Opcode::BLAS1_ZIP || opcode == Opcode::BLAS1_UNZIP ||
		opcode == Opcode::BLAS2_VXM_GENERIC
	) ) {
		std::cerr << "error:Data Dependence Analysis has not been implemented for "
			<< "the operation with code " << static_cast< unsigned int >( opcode )
			<< std::endl;
		exit( 1 );
	}

	for(
		std::vector< std::vector< Pipeline >::iterator >::iterator st =
			shared_data_pipelines.begin();
		st != shared_data_pipelines.end(); st++
	) {
		if( (*(*st)).getContainersSize() != n ) {
			std::cerr << "error:Data Dependence Analysis detected data-dependent "
				<< "operations on vectors of different size" << std::endl;
			exit( 1 );
		}
	}
#endif

	// after executing all the pipelines with which the current stage shares data
	// and these data dependences do not allow this stage to be inserted into such
	// a pipeline we know that we can now merge all the remaining pipelines with
	// which the current stage shares data an then add the current stage at the end
	// of the new pipeline for efficiency, we consider the three following cases
	if( shared_data_pipelines.empty() ) {
		// if none of the current pipelines shares any data, the stage is added in a
		// new pipeline

		Pipeline *empty_pipeline = nullptr;

		for(
			std::vector< Pipeline >::iterator pt = pipelines.begin();
			pt != pipelines.end(); pt++
		) {

			if( ( *pt ).empty() ) {
				empty_pipeline = &( *pt );
				break;
			}
		}

		if( empty_pipeline != nullptr ) {
			( *empty_pipeline).addStage(
				std::move( func ), opcode,
				n, data_type_size, dense_descr, dense_mask,
				output_vector_ptr, output_aux_vector_ptr,
				coor_output_ptr, coor_output_aux_ptr,
				input_a_ptr, input_b_ptr, input_c_ptr, input_d_ptr,
				coor_a_ptr, coor_b_ptr, coor_c_ptr, coor_d_ptr,
				input_matrix
			);

			// we always execute the pipeline when a scalar is returned
			if( output_vector_ptr == nullptr ) {
				ret = ret ? ret : ( *empty_pipeline ).execution();
			}
		} else {
			Pipeline pipeline;

			pipeline.addStage(
				std::move( func ), opcode,
				n, data_type_size, dense_descr, dense_mask,
				output_vector_ptr, output_aux_vector_ptr,
				coor_output_ptr, coor_output_aux_ptr,
				input_a_ptr, input_b_ptr, input_c_ptr, input_d_ptr,
				coor_a_ptr, coor_b_ptr, coor_c_ptr, coor_d_ptr,
				input_matrix
			);

			// we always execute the pipeline when a scalar is returned
			if( output_vector_ptr == nullptr ) {
				ret = ret ? ret : pipeline.execution();
			} else {
				pipelines.push_back( std::move( pipeline ) );
				// pipelines.emplace_back( Pipeline() );
			}
		}
	} else if ( shared_data_pipelines.size() == 1 ) {

		std::vector< Pipeline >::iterator ptr = ( *(shared_data_pipelines.begin()) );

		// the stage is added in the current pipeline which may be empty if it
		// overwrites the input of SpMV
		// it is not necessary to deallocate/release this pipeline
		( *ptr ).addStage(
			std::move( func ), opcode,
			n, data_type_size, dense_descr, dense_mask,
			output_vector_ptr, output_aux_vector_ptr,
			coor_output_ptr, coor_output_aux_ptr,
			input_a_ptr, input_b_ptr, input_c_ptr, input_d_ptr,
			coor_a_ptr, coor_b_ptr, coor_c_ptr, coor_d_ptr,
			input_matrix
		);

		// we always execute the pipeline when a scalar is returned
		if( output_vector_ptr == nullptr ) {
			ret = ret ? ret : ( *ptr ).execution();
		}
	} else {

		// all pipelines with which the current pipelines shares data will be merged
		// under the first pipeline
		std::vector< Pipeline >::iterator union_pipeline =
			( *(shared_data_pipelines.begin()) );

		for(
			std::vector< std::vector< Pipeline >::iterator >::iterator st =
				++shared_data_pipelines.begin();
			st != shared_data_pipelines.end(); st++
		) {
			( *union_pipeline ).merge( *( *st ) );
		}

		// the stage is added in the merged pipeline
		// it is not necessary to deallocate/release this pipeline
		( *union_pipeline ).addStage(
			std::move( func ), opcode,
			n, data_type_size, dense_descr, dense_mask,
			output_vector_ptr, output_aux_vector_ptr,
			coor_output_ptr, coor_output_aux_ptr,
			input_a_ptr, input_b_ptr, input_c_ptr, input_d_ptr,
			coor_a_ptr, coor_b_ptr, coor_c_ptr, coor_d_ptr,
			input_matrix
		);

		// we always execute the pipeline when a scalar is returned
		if( output_vector_ptr == nullptr ) {
			ret = ret ? ret : ( *union_pipeline ).execution();
		}
	}

	checkIfExceeded();

	return ret;
}

grb::RC LazyEvaluation::addeWiseLambdaStage(
	const Pipeline::stage_type &&func, Opcode opcode,
	const size_t n, const size_t data_type_size,
	const bool dense_descr,
	std::vector< const void * > all_vectors_ptr,
	const Coordinates< nonblocking > * const coor_a_ptr
) {
	RC ret = SUCCESS;

	// ensure that nothing is left from previous stages
	shared_data_pipelines.clear();

	for(
		std::vector< Pipeline >::iterator pt = pipelines.begin();
		pt != pipelines.end(); pt++
	) {
		if( ( *pt ).empty() ) {
			continue;
		}

		// processes all output vectors of eWiseLambda
		for(
			std::vector< const void *>::iterator it = all_vectors_ptr.begin();
			it != all_vectors_ptr.end(); ++it
		) {
			if( (*pt).accessesInputVector( *it ) ) {
				if( ( *pt ).overwritesVXMInputVectors( *it ) ) {
					( *pt ).execution();
				} else {
				shared_data_pipelines.push_back( pt );
				}
			} else if( (*pt).accessesOutputVector( *it ) ) {
				shared_data_pipelines.push_back( pt );
			}
		}
	}

#ifdef _DEBUG
	for(
		std::vector< std::vector< Pipeline >::iterator >::iterator st =
			shared_data_pipelines.begin();
		st != shared_data_pipelines.end(); st++
	) {
		if( (*(*st)).getContainersSize() != n ) {
			std::cerr << "error:Data Dependence Analysis detected data-dependent "
				<< "operations on vectors of different size" << std::endl;
			exit( 1 );
		}
	}
#endif

	if( shared_data_pipelines.empty() ) {
		// if none of the current pipelines shares any data, the stage is added in a
		// new pipeline

		Pipeline *empty_pipeline = nullptr;

		for(
			std::vector< Pipeline >::iterator pt = pipelines.begin();
			pt != pipelines.end(); pt++
		) {
			if( ( *pt ).empty() ) {
				empty_pipeline = &( *pt );
				break;
			}
		}

		if( empty_pipeline != nullptr ) {
			(*empty_pipeline).addeWiseLambdaStage(
				std::move( func ), opcode,
				n, data_type_size,
				dense_descr,
				all_vectors_ptr, coor_a_ptr
			);
		} else {
			Pipeline pipeline;
			pipeline.addeWiseLambdaStage(
				std::move( func ), opcode,
				n, data_type_size,
				dense_descr,
				all_vectors_ptr, coor_a_ptr
			);
			pipelines.push_back( std::move( pipeline ) );
			// pipelines.emplace_back( Pipeline() );
		}
	} else if ( shared_data_pipelines.size() == 1 ) {

		std::vector< Pipeline >::iterator ptr =
			( *(shared_data_pipelines.begin()) );

		// the stage is added in the current pipeline which may be empty if it
		// overwrites the input of SpMV
		// it is not necessary to deallocate/release this pipeline
		( *ptr ).addeWiseLambdaStage(
			std::move( func ), opcode,
			n, data_type_size,
			dense_descr,
			all_vectors_ptr, coor_a_ptr
		);
	} else {

		// all pipelines with which the current pipelines shares data will be merged
		// under the first pipeline
		std::vector< Pipeline >::iterator union_pipeline =
			( *(shared_data_pipelines.begin()) );

		for(
			std::vector< std::vector< Pipeline >::iterator >::iterator st =
				++shared_data_pipelines.begin();
			st != shared_data_pipelines.end(); st++
		) {
			( *union_pipeline ).merge( *( *st ) );
		}

		// the stage is added in the merged pipeline
		// it is not necessary to deallocate/release this pipeline
		( *union_pipeline ).addeWiseLambdaStage(
			std::move( func ), opcode,
			n, data_type_size,
			dense_descr,
			all_vectors_ptr, coor_a_ptr
		);
	}

	checkIfExceeded();

	return ret;
}

grb::RC LazyEvaluation::execution( const void * const container )
{
	RC rc = SUCCESS;

	// search for pipelines with shared data
	for(
		std::vector< Pipeline >::iterator pt = pipelines.begin();
		pt != pipelines.end(); pt++
	) {

		if( ( *pt ).empty() ) {
			continue;
		}

		// a single pipeline is executed, and in the case of returning an error, it
		// is handled correctly
		if( (*pt).accessesVector( container ) || (*pt).accessesMatrix( container ) ) {
			rc = (*pt).execution();
			break;
		}
	}

	return rc;
}

grb::RC LazyEvaluation::execution()
{
	RC rc = SUCCESS;

	// execute all pipelines
	for(
		std::vector< Pipeline >::iterator pt = pipelines.begin();
		pt != pipelines.end(); pt++
	) {

		if( ( *pt ).empty() ) {
			continue;
		}

		rc = (*pt).execution();
		if( rc != SUCCESS ) {
			return rc;
		}
	}

	return rc;
}

