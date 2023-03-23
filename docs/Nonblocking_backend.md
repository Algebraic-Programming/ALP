
<pre>
  Copyright 2021 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
</pre>


# Design and implementation of the nonblocking backend

The [C API specification](https://graphblas.org/docs/GraphBLAS_API_C_v1.3.0.pdf) of [GraphBLAS](https://graphblas.org) defines two execution modes: blocking execution and nonblocking execution. In the blocking mode, the invocation of an operation implies that the computation is completed and the result is written to memory when the function returns. The nonblocking execution allows an operation to return although the result has not been computed yet. Therefore, the nonblocking execution may delay the execution of some operations to perform optimisations. Lazy evaluation is the key idea in nonblocking execution, and computations are performed only when they are required for the sound execution of a program.

For the description of the full design and experimental results for nonblocking execution in ALP/GraphBLAS, please read the following publications.

* A. Mastoras, S. Anagnostidis, and A. N. Yzelman, "Design and Implementation for Nonblocking Execution in GraphBLAS: Tradeoffs and Performance," ACM Trans. Archit. Code Optim. 20, 1, Article 6 (March 2023), 23 pages, [https://doi.org/10.1145/3561652](https://doi.org/10.1145/3561652)
* A. Mastoras, S. Anagnostidis, and A. N. Yzelman, "Nonblocking execution in GraphBLAS," 2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), 2022, pp. 230-233, doi: [10.1109/IPDPSW55747.2022.00051](10.1109/IPDPSW55747.2022.00051).

ALP/GraphBLAS provides the `nonblocking` backend that performs multi-threaded nonblocking execution on shared-memory systems. The implementation of the `nonblocking` backend relies on that of the `reference` and `reference_omp` backends that perform sequential and multi-threaded blocking execution, respectively.


## Overview of the sources files

The source files for the `nonblocking` backend are maintained under the `src/graphblas/nonblocking` directory, and the header files are maintained under `include/graphblas/nonblocking`. Most of these files exist for the `reference` backend, and the `nonblocking` backend uses some additional files. In particular, the full list of the source files for the `nonblocking` backend are the following:

* `analytic_model.cpp`
* `init.cpp` (relies on `reference/init.cpp`)
* `io.cpp`
* `lazy_evaluation.cpp`
* `pipeline.cpp`

from which the `analytic_model.cpp`, `lazy_evaluation.cpp`, and `pipeline.cpp` exist only for the `nonblocking` backend, and they are the main source files for the implementation of the nonblocking execution. The `init.cpp` file invokes the corresponding functions of the `reference` backend. The header files of the `nonblocking` backend include:

* `alloc.hpp` (delegates to `reference/alloc.hpp`)
* `analytic_model.hpp`
* `benchmark.hpp` (delegates to `reference/benchmark.hpp`)
* `blas1.hpp`
* `blas2.hpp`
* `blas3.hpp`
* `boolean_dispathcer_blas1.hpp`
* `boolean_dispathcer_blas2.hpp`
* `boolean_dispathcer_io.hpp`
* `collectives.hpp` (delegates to `reference/collectives.hpp`)
* `config.hpp`
* `coordinates.hpp`
* `exec.hpp` (delegates to `reference/exec.hpp`)
* `forward.hpp`
* `init.hpp`
* `io.hpp`
* `lazy_evaluation.hpp`
* `matrix.hpp`
* `pinnedVector.hpp`
* `pipeline.hpp`
* `properties.hpp`
* `spmd.hpp` (delegates to `reference/spmd.hpp`)
* `vector.hpp` (relies on `reference/vector.hpp`)
* `vector_wrapper.hpp`

from which the `analytic_model.hpp`, `boolean_dispathcer_blas1.hpp`, `boolean_dispathcer_blas2.hpp`, `boolean_dispathcer_io.hpp`, `lazy_evaluation.hpp`, `pipeline.hpp`, and `vector_wrapper.hpp` are used only for the `nonblocking` backend.
The current implementation supports nonblocking execution only for level-1 and level-2 operations defined in the following files:

* `nonblocking/io.hpp`
* `nonblocking/blas1.hpp`
* `nonblocking/blas2.hpp`

and thus most of the code for the nonblocking execution is found in these three files. The level-3 operations defined in `blas3.hpp` and some defined in `blas2.hpp` incur blocking behaviour. If a program invokes these primitives while compiled using the nonblocking backend, a warning will be emitted to the standard error stream. Please check regularly for future releases that enable native nonblocking execution for these remaining primitives.


## Lazy evaluation

Lazy evaluation enables the loop fusion and loop tiling optimisations in a pure library implementation such as required by ALP/GraphBLAS. Dynamic data dependence analysis identifies operations that share data, and these operations are added as stages of the same pipeline. Operations grouped into the same pipeline may be executed in parallel and reuse data in cache. The design for nonblocking execution is fully dynamic, since the optimisations are performed at run-time and the pipelines may include operations of arbitrary control-flow. The nonblocking execution is fully automatic, since the performance parameters, i.e., the number of threads and the tile size, are selected based on an analytic model (defined in `analytic_model.cpp`).

To illustrate lazy evaluation for the nonblocking backend, we use the `grb::set` operation that initialises all the elements of the output vector `x` with the value of an input scalar `val`. The code below shows the implementation of `grb::set` for the `reference` and `reference_omp` backends found in `reference/io.hpp`.

```cpp
template<
	Descriptor descr = descriptors::no_operation,
	typename DataType, typename T,
	typename Coords
>
RC set(
	Vector< DataType, reference, Coords > &x,
	const T val,
	...
) {
	...

	const size_t n = size( x );
	if( (descr & descriptors::dense) && nnz( x ) < n ) {
		return ILLEGAL;
	}

	const DataType toCopy = static_cast< DataType >( val );

	if( !(descr & descriptors::dense) ) {
		internal::getCoordinates( x ).assignAll();
	}
	DataType * const raw = internal::getRaw( x );

#ifdef _H_GRB_REFERENCE_OMP_IO
	#pragma omp parallel
	{
		size_t start, end;
		config::OMP::localRange( start, end, 0, n );
#else
		const size_t start = 0;
		const size_t end = n;
#endif
		for( size_t i = start; i < end; ++ i ) {
			raw[ i ] = internal::template ValueOrIndex< descr, DataType, DataType >::getFromScalar( toCopy, i );
		}
#ifdef _H_GRB_REFERENCE_OMP_IO
	}
#endif

	assert( internal::getCoordinates( x ).nonzeroes() ==
		internal::getCoordinates( x ).size() );

	return SUCCESS;
}
```

A typical operation of ALP/GraphBLAS includes a main for loop that iterates over all the elements (or only the nonzeroes) of the containers to perform the required computation. One additional step is to check if the `dense` descriptor is correctly used, i.e., none of the input and output vectors is sparse, and otherwise the error code `grb::ILLEGAL` is returned. It is also necessary to properly assign the coordinates of the output vector. In the case of the `grb::set` operation, the raw data of the output vector are initialised with the value of the input scalar within the body of the main loop. The check for the correct usage of the `dense` descriptor is performed before the main loop, and all the coordinates of the output vector are assigned by invoking `assignAll`. That is, the initialisation of the coordinates is performed in one step, since the output vector will be dense after the completion of this operation. If the `dense` descriptor is given by the user, the vector is supposed to be already dense, and thus the invocation of `assignAll` is omitted.

To implement lazy evaluation in the ALP/GraphBLAS library implementation, the code of an operation is not necessarily executed when the corresponding function is invoked. Instead, the loop is added into a lambda function that corresponds to a stage of a pipeline, and the lambda function is stored and executed later. Lambda functions are an implementation decision that meshes well with template-based programming in ALP/GraphBLAS. The code below shows the implementation of the `grb::set` operation discussed above for the corresponding nonblocking implementation defined in `nonblocking/io.hpp`.

```cpp
template<
	Descriptor descr = descriptors::no_operation,
	typename DataType, typename T,
	typename Coords
>
RC set(
	Vector< DataType, nonblocking, Coords > &x, const T val,
	...
) {
	...

	RC ret = SUCCESS;

	const DataType toCopy = static_cast< DataType >( val );
	DataType * const raw = internal::getRaw( x );
	const size_t n = internal::getCoordinates( x ).size();

	constexpr const bool dense_descr = descr & descriptors::dense;

	internal::Pipeline::stage_type func = [&x, toCopy, raw] (
			internal::Pipeline &pipeline, size_t active_chunk_id, size_t max_num_chunks, size_t lower_bound, size_t upper_bound
		) {
			(void) active_chunk_id;
			(void) max_num_chunks;

			const bool already_dense_vectors = dense_descr || pipeline.allAlreadyDenseVectors();

			if( !already_dense_vectors ) {
				bool already_dense_output = pipeline.containsAlreadyDenseContainer( &internal::getCoordinates( x ) );
				if( !already_dense_output ) {
					Coords local_x = internal::getCoordinates( x ).asyncSubset( active_chunk_id, max_num_chunks, lower_bound, upper_bound );

					local_x.local_assignAllNotAlreadyAssigned();
					assert( local_x.nonzeroes() == local_x.size() );

					internal::getCoordinates( x ).asyncJoinSubset( local_x, active_chunk_id, max_num_chunks );
				}
			}

			for( size_t i = lower_bound; i < upper_bound; i++ ) {
				raw[ i ] = internal::template ValueOrIndex< descr, DataType, DataType >::getFromScalar( toCopy, i );
			}

			return SUCCESS;
		};

	ret = ret ? ret : internal::le.addStage(
			std::move( func ), internal::Opcode::IO_SET_SCALAR,
			n, sizeof( DataType ), dense_descr, true,
			&x, nullptr,
			&internal::getCoordinates( x ), nullptr,
			nullptr, nullptr, nullptr, nullptr,
			nullptr, nullptr, nullptr, nullptr
		);

	return ret;
}
```

The implementation of `grb::set` for the `nonblocking` backend is very similar to that of the `reference` and `reference_omp` backends. In particular, a lambda function is defined for the execution of a subset of consecutive iterations of the initial loop determined by the `lower_bound` and `upper_bound` parameters. Therefore, the main loop iterates from `lower_bound` to `upper_bound` to initialise the raw data of the output vector. The main difference between the `nonblocking` backend and the `reference` backend is the way the coordinates are handled. First, it is impossible to check if the `dense` descriptor is correctly given in the beginning of an operation, because the computation may not be completed yet due to lazy evaluation and the number of nonzeroes of a vector may not be up to date. Therefore, the check for the `dense` descriptor must be moved into the lambda function. However, the coordinates used by the `nonblocking` backend require a different mechanism than that used by the `reference` backend. The design of the coordinates mechanism for the `nonblocking` backend is presented in the next section.


## Handling sparse vectors

Vectors in ALP/GraphBLAS may be either sparse or dense. In the case of dense vectors, each operation accesses all the elements as shown above with the example of `grb::set`. However, to efficiently handle sparsity, it is necessary to maintain the coordinates of the nonzeroes, such that ALP/GraphBLAS operations access only the nonzeroes. Hence, each vector includes a so-called Sparse Accumulator (SPA), consisting of the following data to handle sparsity:

* an unsigned integer `_cap` that stores the size of the vector;
* an unsigned integer `_n` that stores the number of nonzeroes in the vector;
* a boolean array, `_assigned`, of size`_cap` that indicates if the element of a coordinate is a nonzero; and
* an unsigned integer array, `_stack`, that represents a stack and stores the coordinates of the assigned elements.

A vector is dense when the number of nonzeroes is equal to the size of the vector, i.e., `_n = _cap`.
The stack and the `_assigned` array are used only when accessing a sparse vector.
For an empty vector, `_n = 0`, all the elements of `_assigned` are initialised to `false`, and the stack is empty.
The assignment of the i-th element of a vector implies that:
```cpp
_stack[_n] = i;
_assigned[i] = true;
_n++
```
Therefore, the coordinates of the nonzeroes are not sorted; they are pushed to the stack in an arbitrary order. Iterating over the nonzeroes of a sparse vector is done via the stack, and thus access to the elements may happen in any order.

The internal representation of a vector is sufficient to correctly and efficiently handle sparse vectors for sequential execution. However, this is not the case for multi-threaded execution, since simultaneous assignments of vector elements may cause data races. Protecting the stack and the counter of nonzeroes with a global lock is a trivial solution that leads to significant performance degradation. Therefore, it is necessary to design a different mechanism that is tailored to the needs of the nonblocking execution and exploits any information about accesses of elements by different threads.


## Local coordinates mechanism

The local coordinates mechanism is used for efficient handling of sparse vectors in parallel nonblocking execution and is implemented in `coordinates.hpp`. The local coordinates mechanism consists of a set of local views for the coordinates stored in the global stack. Each local view includes the coordinates of the nonzeroes for a tile of iterations, and each thread access its own local coordinates and any update to the sparsity structure of a vector is performed in the local view. The local coordinates mechanism requires initialisation of the local views before the execution of the pipeline and update of the global stack with the new nonzeroes after the execution of the pipeline.

The local coordinates mechanism requires some additional data for each tile of a vector:

* an unsigned integer array that stores the number of nonzeroes for each local view, which are read from the global stack during initialisation;
* an unsigned integer array that stores the number of nonzeroes that were assigned to each local view during the execution of a pipeline;
* a set of unsigned integer arrays that represent local stacks and store the local coordinates, i.e., each array corresponds to a different local view.

The local coordinates mechanism relies on five main functions defined in `nonblocking/coordinates.hpp`. The local views are initialised via `asyncSubsetInit`. Each operation reads the state of the local view with `asyncSubset`, and it updates the state with `asyncJoinSubset` once the computation is completed. The invocation of `joinSubset` pushes the local coordinates to the global stack. None of these functions uses locks, and to avoid data races, `joinSubset` updates the global stack based on the prefix-sum computation for the number of new nonzeroes performed by `prefixSumComputation`.

To illustrate the usage of the local coordinates mechanism in the `nonblocking` backend, we use the in-place `grb::foldl` operation shown below, which receives one output vector, one input vector and an operator.

```cpp
template<
	Descriptor descr = descriptors::no_operation, class OP,
	typename IOType, typename InputType, typename Coords
>
RC foldl(
	Vector< IOType, nonblocking, Coords > &x,
	const Vector< InputType, nonblocking, Coords > &y,
	const OP &op = OP(),
	...
) {
	const size_t n = size( x );

	...

	RC ret = SUCCESS;

	constexpr const bool dense_descr = descr & descriptors::dense;

	internal::Pipeline::stage_type func = [&x, &y, &op, phase] (
			internal::Pipeline &pipeline,
			const size_t active_chunk_id, const size_t max_num_chunks,
			const size_t lower_bound, const size_t upper_bound
		) {
			RC rc = SUCCESS;

			const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;
			const Coords * const local_null_mask = nullptr;

			Coords local_x, local_y;
			const size_t local_n = upper_bound - lower_bound;
			size_t local_x_nz, local_y_nz;
			bool sparse = false;

			const bool already_dense_vectors = dense_descr || pipeline.allAlreadyDenseVectors();

			bool already_dense_output = true;
			bool already_dense_input = true;

			if( !already_dense_vectors ) {
				already_dense_output = pipeline.containsAlreadyDenseContainer( &internal::getCoordinates( x ) );
				if( !already_dense_output ) {
					local_x = internal::getCoordinates( x ).asyncSubset( active_chunk_id, max_num_chunks, lower_bound, upper_bound );
					local_x_nz = local_x.nonzeroes();
					if( local_x_nz < local_n ) {
						sparse = true;
					}
				}

				already_dense_input = pipeline.containsAlreadyDenseContainer( &internal::getCoordinates( y ) );
				if( !already_dense_input ) {
					local_y = internal::getCoordinates( y ).asyncSubset( active_chunk_id, max_num_chunks, lower_bound, upper_bound );
					local_y_nz = local_y.nonzeroes();
					if( local_y_nz < local_n ) {
						sparse = true;
					}
				}
			}

			if( sparse ) {
				// performs the computation for the sparse case
				...
			} else {
				// performs the computation for the dense case
				...
			}

			if( !already_dense_output ) {
				internal::getCoordinates( x ).asyncJoinSubset( local_x, active_chunk_id, max_num_chunks );
			}

			return rc;
		};

	ret = ret ? ret : internal::le.addStage(
			std::move( func ), internal::Opcode::BLAS1_FOLD_VECTOR_VECTOR_GENERIC,
			n, sizeof( IOType ), dense_descr, true,
			&x, nullptr,
			&internal::getCoordinates( x ), nullptr,
			&y, nullptr, nullptr, nullptr,
			&internal::getCoordinates( y ), nullptr, nullptr, nullptr
		);

	return ret;
}
```

The state of the local view is read for each vector accessed in an operation by invoking `asyncSubset`. The sparsity structure may be updated only for the output vector, and thus `asyncJoinSubset` is invoked only for the output vector to update the number of new nonzeroes. Operations consider the dense and the sparse case, and the executed path is determined at run-time based on the sparsity structure of the local coordinates. To avoid the overhead of initialising the local views, the `nonblocking` backend performs compile-time and runtime optimisations discussed in the next section. Therefore, `asyncSubset` and `asyncJoinSubset` are conditionally invoked depending on whether the corresponding vectors are already dense.


## Optimisations for dense vectors

To improve the performance of nonblocking execution, it is crucial to avoid the usage of the local views when the vectors are dense. It is possible to determine whether a vector is dense based on compile-time information from descriptors and runtime analysis. The first one implies zero runtime overhead, but the descriptors must be provided by the user.

There exist two main differences between the compile-time information from descriptors and the runtime analysis.
First, descriptors may apply to all vectors of an operation, whereas the runtime analysis applies to each individual vector of an operation. Second, descriptors refer to the vectors of a specific operation, whereas the runtime analysis refers to the state of a vector before the execution of a pipeline.

### Compile-time descriptors

The ALP/GraphBLAS implementation provides a set of descriptors defined in `include/graphblas/descriptors.hpp`, and they may be combined using bit-wise operators.
A descriptor is passed to an operation and indicates some information about some or all of the output and input containers, e.g., vectors and matrices.
Three of these descriptors are the following:

* `dense` to indicate that all input and output vectors are structurally dense before the invocation;
* `structural` that ignores the values of the mask and uses only its structure, i.e., the i-th element evaluates to true if any value is assigned to it; and
* `invert_mask` that inverts the mask.

The `dense` and `structural` descriptors may affect both correctness and performance, and `invert_mask` affect only the correctness of an operation. These three descriptors may be used to perform optimisations for the local coordinates mechanism. In particular, if the dense descriptor is provided, it implies that all the vectors accessed in an operation are dense before the invocation. Therefore, an operation can safely iterate over all the elements of the vectors without using neither the global nor the local coordinates.

One exception is an out-of-place operation that receives a mask, since the dense descriptor itself does not guarantee that all the elements of a dense mask evaluate to true. Therefore, a dense output vector may become sparse once the computation is completed. That is, the output vector becomes empty in the beginning of the operation, and then each of its coordinates may be assigned depending on whether the corresponding element of the mask evaluates to true or not. Reading the elements of a mask does not require usage of the local coordinates when the dense descriptor is given. However, to avoid the usage of the local coordinates for the output vector of an out-of-place operation that receives a mask, both the `structural` and the `invert_mask` descriptors should be given in addition to the `dense` descriptor.

### Runtime analysis

The runtime analysis for dense vectors relies on a simple property of ALP/GraphBLAS. A vector that is already dense before the execution of a pipeline cannot become sparse during the execution of the pipeline unless the pipeline contains an out-of-place operation, i.e., `grb::set`, `grb::eWiseApply`, or `grb::clear` that makes the vector empty. The current design for nonblocking execution in ALP/GraphBLAS allows pipelines that include an out-of-place operation but does not allow pipelines that include the `grb::clear` operation.

The nonblocking execution relies on the runtime analysis to determine whether a vector is already dense before the execution of a pipeline, only when the `dense` descriptor is not given by the user. For each already dense vector of a pipeline, neither the global nor the local coordinates are used unless the vector is the output of an out-of-place operation. Therefore, the overhead of the local coordinates mechanism is completely avoided.

### Implementation of the optimisation

To illustrate the implementation of the compile-time and runtime optimisations for dense vectors, we use one example of an in-place and one example of an out-of-place operation.
The runtime analysis relies on the `allAlreadyDenseVectors` function that returns `true` when all the vectors accessed in a pipeline are already dense, and on `containsAlreadyDenseContainer` that returns `true` when a specific vector accessed in a pipeline is already dense.

#### In-place operations

In the case of an in-place operation, we use the example of the `grb::foldl` operation discussed earlier.
The code below is included in the lambda function of `grb::foldl`.

```cpp
const bool already_dense_vectors = dense_descr || pipeline.allAlreadyDenseVectors();

bool already_dense_output = true;
bool already_dense_input = true;

if( !already_dense_vectors ) {
	already_dense_output = pipeline.containsAlreadyDenseContainer( &internal::getCoordinates( x ) );
	if( !already_dense_output ) {
		local_x = internal::getCoordinates( x ).asyncSubset( active_chunk_id, max_num_chunks, lower_bound, upper_bound );
		local_x_nz = local_x.nonzeroes();
		if( local_x_nz < local_n ) {
			sparse = true;
		}
	}

	already_dense_input = pipeline.containsAlreadyDenseContainer( &internal::getCoordinates( y ) );
	if( !already_dense_input ) {
		local_y = internal::getCoordinates( y ).asyncSubset( active_chunk_id, max_num_chunks, lower_bound, upper_bound );
		local_y_nz = local_y.nonzeroes();
		if( local_y_nz < local_n ) {
			sparse = true;
		}
	}
}

...

if( !already_dense_output ) {
	internal::getCoordinates( x ).asyncJoinSubset( local_x, active_chunk_id, max_num_chunks );
}
```

The variable `already_dense_vectors` indicates whether all the vectors accessed in this operation are already dense based on compile-time or runtime information.
In addition, one variable is declared for each vector to indicate whether a vector is already dense, i.e., the variables `already_dense_output` and `already_dense_input` are initialised to `true`, assuming that the vectors are already dense.
If `already_dense_vectors` is evaluated to true, the state of the local views is not read and the assumption for already dense vectors is correct.
Otherwise, it is necessary to check if each vector accessed in the operation is already dense, and if this is not the case, the state of the local view is read by invoking `asyncSubset`.
The update of the state for the local view is performed once the computation is completed via `asyncJoinSubset` only when the output vector is not already dense.

#### Out-of-place operations

For the implementation of the optimisation for dense vectors of an out-of-place operation, we use the example of the `grb::eWiseApply` operation defined in `blas1.hpp`.
There exist four main scenarios we need to consider, depending on whether the output vector for a tile needs to become empty, dense, or both empty and dense, and whether the operation receives a mask.

##### Out-of-place operation with a potentially sparse output vector

In the case that the input consists of three vectors, the output vector will have an a-priori unknown sparsity structure.
Therefore, unless all vectors are already dense, it is necessary to initialise the state of the output vector via `asyncSubset` and clear the coordinates of each local view by invoking `local_clear`.
In contrast to an in-place operation, the decision about reading and updating the state of the output vector does not depend on whether the output vector is already dense,
since an already dense output vector may become sparse depending on the sparsity structure of the input vectors.

Since the current design for nonblocking execution does not allow the number of nonzeroes to decrease, it is necessary to reset the global counter of nonzeroes by invoking `reset_global_nnz_counter`.
The `local_clear` function updates properly the number of new nonzeroes that should be written later to the global stack by `joinSubset`, i.e., all the nonzeroes of the local view are considered as new.
In addition, the output vector is marked as potentially sparse by invoking `markMaybeSparseContainer`.
Both of these functions are invoked only by the thread that executes the first tile, i.e., when `lower_bound = 0`.

```cpp
template<
	Descriptor descr = descriptors::no_operation, class Monoid,
	typename OutputType, typename InputType1, typename InputType2,
	typename Coords
>
RC eWiseApply(
	Vector< OutputType, nonblocking, Coords > &z,
	const Vector< InputType1, nonblocking, Coords > &x,
	const Vector< InputType2, nonblocking, Coords > &y,
	const Monoid &monoid = Monoid(),
	...
) {
	const size_t n = internal::getCoordinates( z ).size();

	...

	RC ret = SUCCESS;

	constexpr const bool dense_descr = descr & descriptors::dense;

	internal::Pipeline::stage_type func = [&z, &x, &y, &monoid, phase] (
			internal::Pipeline &pipeline,
			const size_t active_chunk_id, const size_t max_num_chunks,
			const size_t lower_bound, const size_t upper_bound
		) {
			RC rc = SUCCESS;

			const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;
			const Coords * const local_null_mask = nullptr;

			Coords local_x, local_y, local_z;

			const bool already_dense_vectors = dense_descr || pipeline.allAlreadyDenseVectors();

			bool already_dense_input_x = true;
			bool already_dense_input_y = true;

			if( !already_dense_vectors ) {
				local_z = internal::getCoordinates( z ).asyncSubset( active_chunk_id, max_num_chunks, lower_bound, upper_bound );

				already_dense_input_x = pipeline.containsAlreadyDenseContainer( &internal::getCoordinates( x ) );
				if( !already_dense_input_x ) {
					local_x = internal::getCoordinates( x ).asyncSubset( active_chunk_id, max_num_chunks, lower_bound, upper_bound );
				}

				already_dense_input_y = pipeline.containsAlreadyDenseContainer( &internal::getCoordinates( y ) );
				if( !already_dense_input_y ) {
					local_y = internal::getCoordinates( y ).asyncSubset( active_chunk_id, max_num_chunks, lower_bound, upper_bound );
				}
			}

			const internal::Wrapper< false, InputType1, Coords > x_wrapper( x );
			const internal::Wrapper< false, InputType2, Coords > y_wrapper( y );

			const auto op = monoid.getOperator();

			if( !already_dense_vectors ) {
				local_z.local_clear();
				if( lower_bound == 0 ) {
					internal::getCoordinates( z ).reset_global_nnz_counter();
					pipeline.markMaybeSparseContainer( &internal::getCoordinates( z ) );
				}
			}

			// performs the computation
			...

			if( !already_dense_vectors ) {
				internal::getCoordinates( z ).asyncJoinSubset( local_z, active_chunk_id, max_num_chunks );
			}

			return rc;
		};

	ret = ret ? ret : internal::le.addStage(
			std::move( func ), internal::Opcode::BLAS1_EWISEAPPLY,
			n, sizeof( OutputType ), dense_descr, true,
			&z, nullptr,
			&internal::getCoordinates( z ), nullptr,
			&x, &y, nullptr, nullptr,
			&internal::getCoordinates( x ), &internal::getCoordinates( y ), nullptr, nullptr
		);

	return ret;
}
```

##### Out-of-place operation with a dense output vector

In the case that the input consists of a scalar and a monoid, it is guaranteed that the output vector will be dense.
Therefore, the only criterion to avoid the usage of the local views is whether the output vector is already dense.
If the output vector is not already dense, then the state of the local view is read, all the not assigned coordinates are assigned by invoking `local_assignAllNotAlreadyAssigned`, and the state is updated via `asyncJoinSubset`.

```cpp

template<
	Descriptor descr = descriptors::no_operation, class Monoid,
	typename OutputType, typename InputType1, typename InputType2,
	typename Coords
>
RC eWiseApply(
	Vector< OutputType, nonblocking, Coords > &z,
	const InputType1 alpha,
	const Vector< InputType2, nonblocking, Coords > &y,
	const Monoid &monoid = Monoid(),
	...
) {
	const size_t n = internal::getCoordinates( z ).size();

	...

	RC ret = SUCCESS;

	constexpr const bool dense_descr = descr & descriptors::dense;

	internal::Pipeline::stage_type func = [&z, alpha, &y, &monoid] (
			internal::Pipeline &pipeline,
			const size_t active_chunk_id, const size_t max_num_chunks,
			const size_t lower_bound, const size_t upper_bound
		) {
			RC rc = SUCCESS;

			Coords local_x, local_y, local_z;

			const bool already_dense_vectors = dense_descr || pipeline.allAlreadyDenseVectors();

			bool already_dense_output = true;
			bool already_dense_input_y = true;

			already_dense_output = pipeline.containsAlreadyDenseContainer( &internal::getCoordinates( z ) );
			if( !already_dense_output ) {
				local_z = internal::getCoordinates( z ).asyncSubset( active_chunk_id, max_num_chunks, lower_bound, upper_bound );
			}

			if( !already_dense_vectors ) {
				already_dense_input_y = pipeline.containsAlreadyDenseContainer( &internal::getCoordinates( y ) );
				if( !already_dense_input_y ) {
					local_y = internal::getCoordinates( y ).asyncSubset( active_chunk_id, max_num_chunks, lower_bound, upper_bound );
				}
			}

			const internal::Wrapper< true, InputType1, Coords > x_wrapper( alpha );
			const internal::Wrapper< false, InputType2, Coords > y_wrapper( y );

			const auto &op = monoid.getOperator();

			if( !already_dense_output ) {
				local_z.local_assignAllNotAlreadyAssigned();
			}

			// performs the computation
			...

			if( !already_dense_output ) {
				internal::getCoordinates( z ).asyncJoinSubset( local_z, active_chunk_id, max_num_chunks );
			}

			return rc;
		};

	ret = ret ? ret : internal::le.addStage(
			std::move( func ), internal::Opcode::BLAS1_EWISEAPPLY,
			n, sizeof( OutputType ), dense_descr, true,
			&z, nullptr,
			&internal::getCoordinates( z ), nullptr,
			&y, nullptr, nullptr, nullptr,
			&internal::getCoordinates( y ), nullptr, nullptr, nullptr
		);

	return ret;
}
```

##### Out-of-place operation with an output vector that consists of some potentially sparse tiles and some dense tiles

In the case that the input consists of an operator instead of a monoid, the output vector may become sparse after the computation unless all vectors are already dense.
Therefore, the global counter of nonzeroes is reset, and the decision about clearing the local coordinates or assigning all of them is made separately for each local view.
The vector is marked as potentially sparse when the local coordinates are cleared for at least one of the tiles.

```cpp
template<
	Descriptor descr = descriptors::no_operation, class OP,
	typename OutputType, typename InputType1, typename InputType2,
	typename Coords
>
RC eWiseApply(
	Vector< OutputType, nonblocking, Coords > &z,
	const InputType1 alpha,
	const Vector< InputType2, nonblocking, Coords > &y,
	const OP &op = OP(),
	...
) {
	const size_t n = internal::getCoordinates( z ).size();

	...

	RC ret = SUCCESS;

	constexpr const bool dense_descr = descr & descriptors::dense;

	internal::Pipeline::stage_type func = [&z, alpha, &y, &op] (
			internal::Pipeline &pipeline,
			const size_t active_chunk_id, const size_t max_num_chunks,
			const size_t lower_bound, const size_t upper_bound
		) {
			RC rc = SUCCESS;

			const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;
			const Coords * const local_null_mask = nullptr;

			Coords local_mask, local_x, local_y, local_z;
			const size_t local_n = upper_bound - lower_bound;
			size_t local_y_nz = local_n;

			const bool already_dense_vectors = dense_descr || pipeline.allAlreadyDenseVectors();

			bool already_dense_input_y = true;

			if( !already_dense_vectors ) {
				local_z = internal::getCoordinates( z ).asyncSubset( active_chunk_id, max_num_chunks, lower_bound, upper_bound );

				already_dense_input_y = pipeline.containsAlreadyDenseContainer( &internal::getCoordinates( y ) );
				if( !already_dense_input_y ) {
					local_y = internal::getCoordinates( y ).asyncSubset( active_chunk_id, max_num_chunks, lower_bound, upper_bound );
					local_y_nz = local_y.nonzeroes();
				}
			}

			const internal::Wrapper< true, InputType1, Coords > x_wrapper( alpha );
			const internal::Wrapper< false, InputType2, Coords > y_wrapper( y );

			if( !already_dense_vectors ) {
				if( lower_bound == 0 ) {
					internal::getCoordinates( z ).reset_global_nnz_counter();
				}
			}

			if( (descr & descriptors::dense) || local_y_nz == local_n ) {
				if( !already_dense_vectors ) {
					local_z.local_assignAll( );
				}

				// performs the computation for the dense case
				...
			} else {
				if( !already_dense_vectors ) {
					local_z.local_clear();
					pipeline.markMaybeSparseContainer( &internal::getCoordinates( z ) );
				}

				// performs the computation for the sparse case
				...
			}

			if( !already_dense_vectors ) {
				internal::getCoordinates( z ).asyncJoinSubset( local_z, active_chunk_id, max_num_chunks );
			}

			return rc;
		};

	ret = ret ? ret : internal::le.addStage(
			std::move( func ), internal::Opcode::BLAS1_EWISEAPPLY,
			n, sizeof( OutputType ), dense_descr, true,
			&z, nullptr,
			&internal::getCoordinates( z ), nullptr,
			&y, nullptr, nullptr, nullptr,
			&internal::getCoordinates( y ), nullptr, nullptr, nullptr
		);

	return ret;
}
```

##### Out-of-place operation that receives a mask

In the case that an out-of-place operation receives a mask, a second variable, `mask_is_dense`, is used to indicate whether the mask is dense based on compile-time information from descriptors or the runtime analysis for already dense vectors.
Then, all the decisions about the output vector are made based on this variable.
In addition, the function `markMaybeSparseDenseDescriptorVerification` is invoked to mark the output vector as potentially sparse when the `dense` descriptor is provided and the elements of the mask may be evaluated to `false` as explained in the section about the dense descriptor verification.

```cpp
template<
	Descriptor descr = descriptors::no_operation, class Monoid,
	typename OutputType, typename MaskType,
	typename InputType1, typename InputType2,
	typename Coords
>
RC eWiseApply(
	Vector< OutputType, nonblocking, Coords > &z,
	const Vector< MaskType, nonblocking, Coords > &mask,
	const InputType1 alpha,
	const Vector< InputType2, nonblocking, Coords > &y,
	const Monoid &monoid = Monoid(),
	...
) {
	const size_t n = internal::getCoordinates( z ).size();

	...

	RC ret = SUCCESS;

	constexpr const bool dense_descr = descr & descriptors::dense;
	constexpr const bool dense_mask = dense_descr && (descr & descriptors::structural) && !(descr & descriptors::invert_mask);

	internal::Pipeline::stage_type func = [&z, &mask, alpha, &y, &monoid] (
			internal::Pipeline &pipeline,
			const size_t active_chunk_id, const size_t max_num_chunks,
			const size_t lower_bound, const size_t upper_bound
		) {
			RC rc = SUCCESS;

			Coords local_mask, local_x, local_y, local_z;
			const size_t local_n = upper_bound - lower_bound;

			const bool already_dense_vectors = dense_descr || pipeline.allAlreadyDenseVectors();

			const bool mask_is_dense = (descr & descriptors::structural) &&
				!(descr & descriptors::invert_mask) && already_dense_vectors;

			bool already_dense_mask = true;
			bool already_dense_input_y = true;

			if( !mask_is_dense ) {
				local_z = internal::getCoordinates( z ).asyncSubset( active_chunk_id, max_num_chunks, lower_bound, upper_bound );
				if( dense_descr && local_z.nonzeroes() < local_n ) {
					return ILLEGAL;
				}
			}

			if( !already_dense_vectors ) {
				already_dense_mask = pipeline.containsAlreadyDenseContainer( &internal::getCoordinates( mask ) );
				if( !already_dense_mask ) {
					local_mask = internal::getCoordinates( mask ).asyncSubset( active_chunk_id, max_num_chunks, lower_bound, upper_bound );
				}

				already_dense_input_y = pipeline.containsAlreadyDenseContainer( &internal::getCoordinates( y ) );
				if( !already_dense_input_y ) {
					local_y = internal::getCoordinates( y ).asyncSubset( active_chunk_id, max_num_chunks, lower_bound, upper_bound );
				}
			}

			const internal::Wrapper< true, InputType1, Coords > x_wrapper( alpha );
			const internal::Wrapper< false, InputType2, Coords > y_wrapper( y );

			const InputType2 right_identity = monoid.template getIdentity< InputType2 >();
			const auto &op = monoid.getOperator();

			if( !mask_is_dense ) {
				local_z.local_clear();
				if( lower_bound == 0 ) {
					internal::getCoordinates( z ).reset_global_nnz_counter();
					pipeline.markMaybeSparseContainer( &internal::getCoordinates( z ) );
					if( dense_descr ) {
						pipeline.markMaybeSparseDenseDescriptorVerification( &internal::getCoordinates( z ) );
					}
				}
			}

			// performs the computation
			...

			if( !mask_is_dense ) {
				internal::getCoordinates( z ).asyncJoinSubset( local_z, active_chunk_id, max_num_chunks );
			}

			return rc;
		};

	ret = ret ? ret : internal::le.addStage(
			std::move( func ), internal::Opcode::BLAS1_MASKED_EWISEAPPLY,
			n, sizeof( OutputType ), dense_descr, dense_mask,
			&z, nullptr,
			&internal::getCoordinates( z ), nullptr,
			&y, &mask, nullptr, nullptr,
			&internal::getCoordinates( y ), &internal::getCoordinates( mask ), nullptr, nullptr
		);

	return ret;
}
```


## Pipeline execution

The nonblocking execution in ALP/GraphBLAS expresses operations as a linear sequence of stages that form a pipeline. The execution of a pipeline is performed when the computation is necessary for the sound execution of the program. Opaqueness guarantees that lazy evaluation is safe when the output of an operation is a container, i.e., a vector or a matrix. The current version of ALP/GraphBLAS does not implement scalars as opaque data types according to the [version 1.3.0](https://graphblas.org/docs/GraphBLAS_API_C_v1.3.0.pdf) of the C API specification. Opaque scalars were introduced later in the [version 2.0.0](https://graphblas.org/docs/GraphBLAS_API_C_v2.0.0.pdf) and may further improve the performance of nonblocking execution.

A pipeline must be executed in the following cases:

* the user explicitly extracts data from a container by using the ALP/GraphBLAS API, e.g., when reading the elements of a vector by using iterators;

* the user invokes the constructor of a container;

* memory is deallocated due to a destructor invocation;

* the invoked operation returns a scalar, e.g., the `grb::dot` operation, in particular, the operation is first added into the pipeline, and then the pipeline is executed immediately before returning the scalar;

* when a sparse matrix–vector multiplication (SpMV) operation is added into a pipeline with another operation that overwrites the input vector to the SpMV;

* when the user explicitly forces the execution of a pipeline via a call to `grb::wait`.

Although level-3 operations are not yet implemented for nonblocking execution, a sparse matrix–sparse matrix multiplication (SpMSpM) operation implies the same constraint with SpMV, i.e., the SpMSpM operation cannot be fused together with another operation that overwrites any of the SpMSpM input matrices.

When a new stage is added to a pipeline, the pipeline execution is performed within the `addStage` function of `lazy_evaluation.cpp`, which implements the dynamic data dependence analysis and identifies any shared data between operations. The pipeline execution due to explicit invocation of iterators or constructors or memory deallocation is performed in `vector.hpp`. The execution of a pipeline caused by `grb::wait` is implemented in `io.hpp`.

The code for the pipeline execution is found in the `execution` method of `pipeline.cpp`. The execution is performed in four main steps, three of which may be omitted when the pipeline does not include any out-of-place operation and all accessed vectors are dense. Simplified code for the execution of the four main steps is shown below.

```cpp
bool initialized_coordinates = false;

#pragma omp parallel for private(vt, pt) schedule(dynamic) num_threads(nthreads)
for( size_t tile_id = 0; tile_id < tiles; ++tile_id ) {
	...
	for( vt = vbegin(); vt != vend(); ++vt ) {
		...
		(**vt).asyncSubsetInit( tile_id, tiles, lower_bound, upper_bound );
		initialized_coordinates = true;
	}
}

#pragma omp parallel for private(vt, pt) schedule(dynamic) num_threads(nthreads)
for( size_t tile_id = 0; tile_id < tiles; ++tile_id ) {
	...
	RC local_ret = SUCCESS;
	for( pt = pbegin(); pt != pend(); ++pt ) {
		local_ret = local_ret ? local_ret : (*pt)( *this, tile_id, tiles, lower_bound, upper_bound );
	}
	if( local_ret != SUCCESS ) {
		ret = local_ret;
	}
}

if( initialized_coordinates ) {
	bool new_nnz = false;

	for( vt = vbegin(); vt != vend(); ++vt ) {
		...
		if( (**vt).newNonZeroes( tiles ) ) {
			new_nnz = true;
			(**vt).prefixSumComputation( tiles );
		}
	}

	if( new_nnz ) {
		#pragma omp parallel for private(vt) schedule(dynamic) num_threads(nthreads)
		for( size_t tile_id = 0; tile_id < tiles; ++tile_id ) {
			...
			for( vt = vbegin(); vt != vend(); ++vt ) {
				...
				if( (**vt).newNonZeroes( tiles ) ) {
					(**vt).joinSubset( tile_id, tiles, lower_bound, upper_bound );
				}
			}
		}
	}
}
```
The local views of each vector accessed in the pipeline are initialised via `asyncSubsetInit`, and then the pipeline is executed. Once the execution is completed, the local views may contain a number of new nonzeroes that must be pushed to the global stack by `joinSubset`. Before this step, it is necessary to perform the prefix-sum computation for the number of new nonzeroes of each local view by invoking `prefixSumComputation`. All these steps may be executed in parallel for different tiles of the vectors as shown with the OpenMP directives, except for the prefix-sum computation that is parallelised internally. The scheduling policy used for OpenMP is dynamic to handle load imbalance, and the performance parameters, i.e., the number of threads and the tile size used in the lambda functions, are automatically selected by the analytic model (see `analytic_model.cpp`).


## Analytic performance model

The analytic performance model used for nonblocking execution consists of the `getPerformanceParameters` function defined in `analytic_model.cpp`, and this function is invoked before the pipeline execution within the `execution` method in `pipeline.cpp`. The analytic model makes an estimation about the number of threads and the tile size that lead to good performance for the execution of a given pipeline, and the estimation is based on various parameters such as the number of vectors accessed in the pipeline, the data type of the vectors, and the size of the vectors. Two additional parameters of special importance are the size of the L1 cache and the number of cores available in the system, since the selected tile size must allow data fit in L1 cache and there should be sufficient work to utilise as many cores as possible.

The analytic model relies on two environment variables:

* `OMP_NUM_THREADS`
* `GRB_NONBLOCKING_TILE_SIZE`

for the number of threads used by OpenMP and the tile size used by the nonblocking backend, respectively. The number of threads determined by the environment variable is an upper bound for the number of threads that may be selected by the analytic model. If the environment variable for the tile size is set, a fixed tile size is used for all executed pipelines. Otherwise, the analytic model automatically selects a proper tile size, depending on the parameters of the executed pipeline.

The initialisation for the number of threads used by OpenMP and the manual tile size is performed in `init.cpp`, and the data of the analytic model are handled by the `ANALYTIC_MODEL` and `IMPLEMENTATION` classes of `config.hpp`.


## Dense descriptor verification

The correct usage of the `dense` descriptor, for the blocking execution, is checked in the beginning of each ALP/GraphBLAS operation.
If there exists at least one input or output vector that is not dense, then the `grb::ILLEGAL` error code is returned as shown in the example below.

```cpp
const size_t n = size( x );
if( (descr & descriptors::dense) && nnz( x ) < n ) {
	return ILLEGAL;
}
```

For the nonblocking execution, checking the correct usage of the `dense` descriptor requires a different process, since the number of nonzeroes in the vectors may not be up to date due to lazy evaluation.
In particular, the check is moved within the lambda function defined for each operation, and the check for the sparsity structure is based on the local views.
However, the optimisation employed by the nonblocking execution for already dense vectors implies that the local views are not always available.
Therefore, it is not always possible to perform the check for correct usage of the `dense` descriptor within the lambda function of an operation.

The verification process for correct usage of the `dense` descriptor relies on the following property:

*A vector that should be dense when an operation is invoked, should remain dense after the execution of the pipeline, unless this vector is the output of an out-of-place operation that receives a mask with elements that may be evaluated to `false`*.

Therefore, the `nonblocking` backend delays the check and performs the verification for correct usage of the `dense` descriptor after the pipeline execution.
To keep track of the vectors that should be dense after the execution of the pipeline, the addition of a lambda function as a stage of a pipeline is accompanied by a boolean variable, called `dense_descr`, that indicates if the `dense` descriptor is given for this operation.
In the case of an out-of-place operation that receives a mask, e.g., `grb::eWiseApply` discussed earlier, the output vector may be marked as potentially sparse when the `dense` descriptor is provided, by invoking `markMaybeSparseDenseDescriptorVerification` as shown in the example of `grb::eWiseApply` above.
In this case, the dense descriptor verification is disabled for the output vector of this specific operation.

This solution is efficient and catches most cases of an illegal `dense` descriptor.
However, it cannot catch an illegal usage of the `dense` descriptor for an operation that receives a sparse vector, which becomes dense during the execution of the pipeline, since it is impossible to detect that the vector was not dense earlier.

