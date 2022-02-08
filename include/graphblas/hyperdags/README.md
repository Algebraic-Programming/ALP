
This backend gathers meta-data while user programs execute. The actual compute
logic is executed by a compile-time selected secondary backend, which by default
is the `reference` backend. The meta-data will be used to generate, at program
exit, a HyperDAG representation of the executed computation. We foresee two
possible HyperDAG representations:

 1. a coarse-grain representation where vertices correspond to a) source
    containers (vectors or matrices-- not scalars), b) output containers, or
    c) ALP/GraphBLAS primitives (such as grb::mxv or grb::dot). Hyperedges
    capture which vertices act as source to operations or outputs in other
    vertices. Each hyperedge has exactly one source vertex only.

 2. a fine-grain representation where source vertices correspond to nonzeroes
    in a source container, not the container as a whole, and likewise for output
    vertices that correspond to individual elements of output containers. Also
    there are now many fine-grained operation vertices that are executed by a
    single ALP/GraphBLAS primitive. For example, a call to grb::vxm will emit
    two hyperedges for every nonzero in the sparse input matrix.

Usage
=====

To use the HyperDAG generation backend, follow the following steps. Note that
steps 1-5 are common to building the general ALP/GraphBLAS template library.
Steps 6 & 7 showcase the HyperDAG generation using representation no. 1 on the
tests/unit/dot.cpp unit test.

1. `cd /path/to/ALP/GraphBLAS/root/directory`

2. `./configure --prefix=/path/to/install/directory`

3. `cd build`

4. `make -j && make -j install`

5. `source /path/to/install/directory/bin/setenv`

6. `grbcxx -b hyperdags -g -O0 -Wall -o dot_hyperdag ../tests/unit/dot.cpp`

7. `grbrun -b hyperdags ./dot_hyperdag`

After these steps, something like the following will be produced:

```
This is functional test ./dot_hyperdag
Info: grb::init (hyperdags) called.
Info: grb::init (reference) called.
Info: grb::finalize (hyperdags) called.
	 dumping HyperDAG to stdout
%%MatrixMarket matrix coordinate pattern general
%	 Source vertices:
%		 0: container initialised by a call to set no. 0
%		 1: container initialised by a call to set no. 1
%		 2: input scalar no. 0
%		 6: input scalar no. 1
...more comment lines follow...
%		 212: input scalar no. 103
%		 213: user-initialised container no. 0
%		 214: user-initialised container no. 1
214 216 428
0 2
0 3
1 0
1 3
2 1
2 3
...more pins follow...
213 214
213 215
Info: grb::finalize (reference) called.
Test OK
```

This output contains the HyperDAG corresponding to the code in the given source
file, `tests/unit/dot.cpp`. Let us examine it. First, ALP/GraphBLAS will always
print info (and warning) statements to the standard error stream. These are:

```
$ grbrun -b hyperdags ./dot_hyperdag 1> /dev/null
Info: grb::init (hyperdags) called.
Info: grb::init (reference) called.
Info: grb::finalize (hyperdags) called.
	 dumping HyperDAG to stdout
Info: grb::finalize (reference) called.
```

These statements indicate which backends are used and when they are
initialised, respectively, finalised. The info messages indicate that the
hyperdags backend is used, which, in turn, employs the standard sequential
reference backend for the actual computations. The second to last message
reports that as part of finalising the hyperdags backend, it dumps the
HyperDAG constructed during computations to the stdandard output stream
(stdout).

The output to stdout starts with

```
%%MatrixMarket matrix coordinate pattern general
```

This indicates the HyperDAG is stored using a MatrixMarket format. As the name
implies, this format stores sparse matrices, so we need a definition of how the
sparse matrix is mapped back to a HyperDAG. Here, rows correspond to hyperedges
while columns correspond to vertices.

In the MatrixMarket format, comments are allowed and should start with a `%`.
The hyperdags backend presently prints which vertices are sources as comment
lines. Later, also information on the operation and output vertices may be
added.

After the comments follow the so-called header line:

```
214 216 428
```

This indicates that there 214 hyperedges, 216 vertices, and 428 pins in the
output HyperDAG. What then follows is one line for each of the pins, printed
as a pair of hypergraph and vertex IDs.

For example, the first two pins contain:

```
0 2
0 3
```

These operate on vertices 2 and 3, which the comments note are an input scalar
and a non-source vertex, respectively. The corresponding first statements of
`tests/unit/dot.cpp` are as follows. It stands to reason that vertex 2 thus
corresponds to the scalar `out` in the below code, while vertex 3 corresponds
to the scalar output of the `grb::dot`.

```
	double out = 2.55;
	grb::Vector< double > left( n );
	grb::Vector< double > right( n );
	grb::set( left, 1.5 );
	grb::set( right, -1.0 );
	grb::dot( out, left, right, ring );
```

If this reading is correct, then there should also be two hyperedges connecting
`left` and `right` to vertex 3, the output of `grb::dot`. Indeed the next four
pins are

```
1 0
1 3
2 1
2 3
```

which indeed correspond to two hyperedges connecting `left` and `right` to the
output of `grb::dot`. Do note that thus far the HyperDAG is in fact just a DAG,
given every hyperedge has exectly two pins.


Extending the HyperDAGs backend
===============================

We now briefly visit the implementation of the HyperDAGs backend. The
implementation of the `hyperdags` `grb::dot` is as follows:

```
template<
	Descriptor descr = descriptors::no_operation,
	class AddMonoid, class AnyOp,
	typename OutputType, typename InputType1, typename InputType2,
	typename Coords
>
RC dot( OutputType &z,
	const Vector< InputType1, hyperdags, Coords > &x,
	const Vector< InputType2, hyperdags, Coords > &y,
	const AddMonoid &addMonoid = AddMonoid(),
	const AnyOp &anyOp = AnyOp(),
	const typename std::enable_if<
		!grb::is_object< OutputType >::value &&
		!grb::is_object< InputType1 >::value &&
		!grb::is_object< InputType2 >::value &&
		grb::is_monoid< AddMonoid >::value &&
		grb::is_operator< AnyOp >::value,
	void >::type * const = nullptr
) {
...
```

The signature of the `grb::dot` follows the specification that is found in
`include/graphblas/reference/blas1.hpp`-- if we need to add a new primitive,
the first step is to simply copy the signature from the reference backend and
then change any container template arguments that read `reference` into
`hyperdags`. This makes sure that the compiler will select the implementation
we are providing here whenever it needs to generate code for a dot-product using
the hyperdags backend.

The source file continues:
```
	// always force input scalar to be a new source
	internal::hyperdags::generator.addSource(
		internal::hyperdags::SCALAR,
		&z
	);
	...
```

Here, we recognise that `z` is an input to the algorithm and needs to be
registered as a source vertex. Recall that by the `grb::dot` specification,
`z` is indeed computed in-place: `z += < x, y >`.

The source continues with registering the sources and destinations (outputs) of
the dot-operation itself:

```
	std::array< const void *, 3 > sources{ &z, &x, &y };
	std::array< const void *, 1 > destinations{ &z };
	...
```

With that done, we finally record the operation, as follows:

```
	internal::hyperdags::generator.addOperation(
		internal::hyperdags::DOT,
		sources.begin(), sources.end(),
		destinations.begin(), destinations.end()
	);
	...
```

Here, the `addOperation` needs to know the type of operation (`DOT`), what its
sources are (given here by iterator pairs to the `sources` array), and what its
destinations are (ditto).

The attentive reader will realise that so far no computation has occurred yet--
we so far only recorded sources and the intended operation. So we finish up
with actually performing the requested computation, relying fully on the
reference backend instead of reimplementing things all over again:

```
	return dot( z,
		internal::getVector(x), internal::getVector(y),
		addMonoid, anyOp
	);
}
```

Here, the `internal::getVector` wrapper function retrieves a reference backend
version of the input vector, and passes that on to the reference backend.


Registering new operation and source types
==========================================

Following the above, one may want to register a new type of operation vertex or
source vertex. For this, see `include/graphblas/hyperdags/hyperdags.hpp` and,
in the case of source vertices, look for the following enum:

```
enum SourceVertexType {
	SCALAR,
	CONTAINER,
	SET
};

const constexpr size_t numSourceVertexTypes = 3;

const constexpr enum SourceVertexType
	allSourceVertexTypes[ numSourceVertexTypes ] =
{
	SCALAR,
	CONTAINER,
	SET
};
```

A new type of source vertex should:

1. be added to the enum. While not copied here, every type is conjoined with
   documentation describing unambiguously where such a source vertex could
   come from / how and when they are generated;

2. increment numSourceVertexTypes; and, finally

3. add the new enum entry to the allSourceVertexTypes array.

This is all that is required-- the implementation will, using these three
structures, automatically generate the data structures required for each type
when the hyperdags backend is initialised.

To add new operation vertex types, the same recipe should be followed, but then
using the `OperationVertexType` enum and the `numOperationVertexTypes` counter
and the `allOperationVertexTypes` array.


TODOs
=====

1. Implement more standard ALP/GraphBLAS operations.

2. Instead of building `std::array`s for `sources` and `destinations` by
   recording pointers, use the new `grb::getID` function for ALP vectors and
   matrices instead. For scalars `z`, indices of type `uintptr_t` must be
   derived by converting them from pointers as follows:
   `const uintptr_t z_id = reinterpret_cast< uintptr_t >( &z );`

3. Implement support for matrices in the `hyperdags` backend-- currently, only
   vector containers are supported.

