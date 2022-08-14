
This file keeps track of all active compiler warning suppressions and updates
with every release. Listed are the code, which suppression is used, and the
rationale for why the suppression is OK to use and the compiler warning is safe
to ignore.

1. `include/graphblas/reference/compressed_storage.hpp`, copyFrom:
```
GRB_UTIL_IGNORE_CLASS_MEMACCESS // by the ALP spec, D can only be POD types.
                                // In this case raw memory copies are OK.
(void) std::memcpy( values + k,
	other.values + k,
	(loop_end - k) * sizeof( D )
);
GRB_UTIL_RESTORE_WARNINGS
```

2. `include/graphblas/reference/blas1.hpp`, dot_generic:
```
for( size_t k = 0; k < AnyOp::blocksize; ++k ) {
	zz[ k ] = addMonoid.template getIdentity< typename AnyOp::D3 >();
}
for( size_t k = 0; k < AnyOp::blocksize; ++k ) {
	if( mask[ k ] ) {
		GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED        // yy and xx cannot be used
		                                           // uninitialised or mask
		apply( zz[ k ], xx[ k ], yy[ k ], anyOp ); // would be false while zz
		GRB_UTIL_RESTORE_WARNINGS                  // init is just above
	}
}
```

3. `include/graphblas/reference/blas1.hpp`, sparse_apply_generic:
```
if( masked ) {
	if( mask[ i ] ) {
		GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED // if masked && mask[ i ], then
		z_p[ offsets[ i ] ] = z_b[ i ];     // z_b[ i ] was set from x or y in
		GRB_UTIL_RESTORE_WARNINGS           // the above
	}
} else {
	GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED // the only way the below could write
	                                    // an uninitialised value is if the
	                                    // static_assert at the top of this
	z_p[ offsets[ i ] ] = z_b[ i ];     // function had triggered. See also
	GRB_UTIL_RESTORE_WARNINGS           // internal issue #321.
}
```

4. `include/graphblas/base/internalops.hpp`, multiple sources:
- mul::apply, add::apply, add::foldl, equal::apply, not_equal::apply.

These are indirectly caused by the following calls:
- `include/graphblas/blas0.hpp`, apply;
- `include/graphblas/reference/blas1.hpp`, dot_generic, masked_apply_generic,
  and sparse_apply_generic.

These are all OK to suppress since the reads are masked.

5. `include/graphblas/reference/blas1.hpp`, fold_from_vector_to_scalar_generic:
```
GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED // the below code ensures to set local
IOType local;                       // whenever our local block is
GRB_UTIL_RESTORE_WARNINGS           // non-empty
if( end > 0 ) {
	if( i < end ) {
		local = static_cast< IOType >( internal::getRaw( to_fold )[ i ] );
	} else {
		local = static_cast< IOType >( internal::getRaw( to_fold )[ 0 ] );
	}
}
```
and
```
if( root == s ) {
	// then I should be non-empty
	assert( !empty );
	// set global value to locally computed value
	GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED // one is only root if the local
	global = local;                     // chunk is non-empty, in which case
	GRB_UTIL_RESTORE_WARNINGS           // local will be initialised (above)
	}
```

6. `include/graphblas/reference/blas1.hpp`, masked_apply_generic:
```
if( mask_b[ t ] ) {
	// ...
	GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED  // z_b is computed from x_b and
	*( z_p + indices[ t ] ) = z_b[ t ];  // y_b, which are both initialised
	GRB_UTIL_RESTORE_WARNINGS            // if mask_b is true
```

