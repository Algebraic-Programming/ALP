#!/bin/bash

# Usage: ./generate_synthetic.sh /path/to/synthetic
SYNTH_DIR="$1"
ALPDIR="$2"
BANDSIZE=1
SIZES=(256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432)

mkdir -p "$SYNTH_DIR"

for N in "${SIZES[@]}"; do
    MATRIX_FILE="$SYNTH_DIR/banded_diag_${N}x${N}_band_${BANDSIZE}.mtx"
    if [ ! -f "$MATRIX_FILE" ]; then
        echo "Generating $MATRIX_FILE"
        python3 $ALPDIR/include/graphblas/cost/mtx_generator.py $N $BANDSIZE $SYNTH_DIR
    else
        echo "Matrix $MATRIX_FILE already exists"
    fi
done