#!/bin/bash

# Usage: ./benchmark_cg_controller.sh
# Set these flags to control which matrices to use
RUN_REAL=0
RUN_SYNTHETIC=1

MATRIX_DIR="/scratch/panastasiadis/matrices"
SYNTH_DIR="$MATRIX_DIR/synthetic"
MM_DIR="$MATRIX_DIR/MM_suite"
MODEL_NAME="d_4_GS"
ALLOC_POLICY="close"
DATADIR="/scratch/panastasiadis/${MODEL_NAME}"
ALPDIR="/home/panastasiadis/ALP"
THREAD_COUNTS=(96) # 48 24 12 8 4 2 1

mkdir -p "$SYNTH_DIR"
mkdir -p "$MM_DIR"
mkdir -p "$DATADIR"

if [ "$RUN_REAL" -eq 1 ]; then
    bash ${ALPDIR}/include/graphblas/cost/download_MM.sh "$MM_DIR"
fi

if [ "$RUN_SYNTHETIC" -eq 1 ]; then
    bash ${ALPDIR}/include/graphblas/cost/generate_synthetic.sh "$SYNTH_DIR" "$ALPDIR"
fi

for THREADS in "${THREAD_COUNTS[@]}"; do
    THREAD_DIR="$DATADIR/t${THREADS}"
    mkdir -p $THREAD_DIR/outputs
    mkdir -p $THREAD_DIR/results/plots
    mkdir -p $THREAD_DIR/results/analysis

    export OMP_NUM_THREADS=$THREADS
    if [ "$ALLOC_POLICY" == "close" ]; then
        export GOMP_CPU_AFFINITY="$(seq -s' ' 0 $((THREADS-1)))"
    fi

    if [ "$RUN_REAL" -eq 1 ]; then
        for MM_MTX in $MM_DIR/*.mtx; do
            MM_NAME=$(basename "$MM_MTX" .mtx)
            mkdir -p $THREAD_DIR/results/analysis/real
            mkdir -p $THREAD_DIR/outputs/real
            OUTPUT_FILE="$THREAD_DIR/outputs/real/${MM_NAME}_output.log"
            RESULT_FILE="$THREAD_DIR/results/analysis/real/${MM_NAME}_analysis.log"
            if [ ! -f "$OUTPUT_FILE" ]; then
                echo "$ALPDIR/build/tests/smoke/conjugate_gradient_reference_omp "$MM_MTX" direct 1 1 > "$OUTPUT_FILE" 2>&1"
                $ALPDIR/build/tests/smoke/conjugate_gradient_reference_omp "$MM_MTX" direct 1 1 > "$OUTPUT_FILE" 2>&1
            fi
            if [ ! -f "$RESULT_FILE" ]; then
                echo "python3 $ALPDIR/include/graphblas/cost/parse_benchmark_logs.py "$OUTPUT_FILE" > "$RESULT_FILE""
                python3 $ALPDIR/include/graphblas/cost/parse_benchmark_logs.py "$OUTPUT_FILE" > "$RESULT_FILE"
            fi
        done

        echo "python3 $ALPDIR/include/graphblas/cost/benchmark_ploter.py --results-dir $DATADIR --threads $THREADS" --filegroup-name real
        python3 $ALPDIR/include/graphblas/cost/benchmark_ploter.py.py --results-dir $DATADIR --threads $THREADS --filegroup-name real
    fi

    if [ "$RUN_SYNTHETIC" -eq 1 ]; then
        BANDSIZE=1
        SIZES=(256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304)
        for N in "${SIZES[@]}"; do
            mkdir -p $THREAD_DIR/results/analysis/synthetic
            mkdir -p $THREAD_DIR/outputs/synthetic
            MATRIX_FILE="$SYNTH_DIR/banded_diag_${N}x${N}_band_${BANDSIZE}.mtx"
            OUTPUT_FILE="$THREAD_DIR/outputs/synthetic/banded_diag_${N}x${N}_band_${BANDSIZE}_output.log"
            RESULT_FILE="$THREAD_DIR/results/analysis/synthetic/banded_diag_${N}x${N}_band_${BANDSIZE}_analysis.log"
            if [ ! -f "$OUTPUT_FILE" ]; then
                echo "$ALPDIR/build/tests/smoke/conjugate_gradient_reference_omp "$MATRIX_FILE" direct 1 1 > "$OUTPUT_FILE" 2>&1"
                $ALPDIR/build/tests/smoke/conjugate_gradient_reference_omp "$MATRIX_FILE" direct 1 1 > "$OUTPUT_FILE" 2>&1
            fi
            if [ ! -f "$RESULT_FILE" ]; then
                echo "python3 $ALPDIR/include/graphblas/cost/parse_benchmark_logs.py "$OUTPUT_FILE" > "$RESULT_FILE""
                python3 $ALPDIR/include/graphblas/cost/parse_benchmark_logs.py "$OUTPUT_FILE" > "$RESULT_FILE"
            fi
        done
        rm -rf /scratch/panastasiadis/d_4_GS/t96/results/plots/
        rm -rf /home/panastasiadis/ALP/build/d_4_GS/t96/results/plots/
        echo "python3 $ALPDIR/include/graphblas/cost/benchmark_ploter.py --results-dir $DATADIR --threads $THREADS" --filegroup-name synthetic
        python3 $ALPDIR/include/graphblas/cost/benchmark_ploter.py --results-dir $DATADIR --threads $THREADS --filegroup-name synthetic
    fi
    mkdir -p $ALPDIR/build/${MODEL_NAME}
    cp -r $THREAD_DIR $ALPDIR/build/${MODEL_NAME}
done

echo "All thread count benchmarks complete! Results are in the '$DATADIR' directory"