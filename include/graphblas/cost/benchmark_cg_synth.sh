#!/bin/bash
# filepath: /home/panastasiadis/ALP/include/graphblas/cost/benchmark_cg_threads.sh

#SBATCH --job-name=graphblas_cg_thread_benchmark    # Job name
#SBATCH --output=benchmark_cg_threads_%j.out         # Standard output file
#SBATCH --error=benchmark_cg_threads_%j.err          # Standard error file
#SBATCH --time=12:00:00                             # Time limit
#SBATCH --exclusive                                 # Exclusive node allocation
#SBATCH --nodes=1                                   # Request 1 node
#SBATCH --ntasks=1                                  # Run a single task
#SBATCH --partition=ARM                             # Specify partition/queue name

# Set to 1 to force re-running all steps even if files already exist
FORCE_REPEAT=1

# Directory with stored/to store matrices
MATRIX_DIR="/scratch/panastasiadis/matrices"
mkdir -p $MATRIX_DIR/synthetic
mkdir -p $MATRIX_DIR/MM_suite

MODEL_NAME="d_4_GS"
# Directories for this specific benchmark
DATADIR="/scratch/panastasiadis/${MODEL_NAME}"

ALPDIR="/home/panastasiadis/ALP"

# Define thread counts to test
THREAD_COUNTS=(96 48 24 12 8 4 2 1)  # Adjust as needed for your system
# Define range of problem sizes to test
SIZES=(256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304) #

BANDSIZE=1

for THREADS in "${THREAD_COUNTS[@]}"; do
    THREAD_DIR="$DATADIR/t${THREADS}"
    mkdir -p $THREAD_DIR/outputs
    mkdir -p $THREAD_DIR/results/plots
    mkdir -p $THREAD_DIR/results/analysis

    echo "========================================================"
    echo "=== Running benchmarks with $THREADS threads ============"
    echo "========================================================"

    for N in "${SIZES[@]}"; do
        MATRIX_FILE="$MATRIX_DIR/synthetic/banded_diag_${N}x${N}_band_${BANDSIZE}.mtx"
        if [ ! -f "$MATRIX_FILE" ] || [ "$FORCE_REPEAT" -eq 1 ]; then
            echo "Generating matrix: $MATRIX_FILE"
            python3 $ALPDIR/include/graphblas/cost/mtx_generator.py $N $BANDSIZE $MATRIX_DIR/synthetic
        else
            echo "Matrix file $MATRIX_FILE already exists, skipping generation"
        fi

        export OMP_NUM_THREADS=$THREADS
        export GOMP_CPU_AFFINITY="$(seq -s' ' 0 $((THREADS-1)))"
        echo "=== Processing matrix of size $N x $N with $THREADS threads (GOMP_CPU_AFFINITY: $GOMP_CPU_AFFINITY) ==="

        OUTPUT_FILE="$THREAD_DIR/outputs/banded_diag_${N}x${N}_band_${BANDSIZE}_output.log"
        if [ ! -f "$OUTPUT_FILE" ] || [ "$FORCE_REPEAT" -eq 1 ]; then
            echo "Running conjugate gradient solver with $THREADS threads, output to: $OUTPUT_FILE"
            $ALPDIR/build/tests/smoke/conjugate_gradient_reference_omp $MATRIX_FILE direct 1 1 > $OUTPUT_FILE 2>&1
        else
            echo "Output file $OUTPUT_FILE already exists, skipping conjugate gradient solver"
        fi

        RESULT_FILE="$THREAD_DIR/results/analysis/banded_diag_${N}x${N}_band_${BANDSIZE}_analysis.log"
        if [ ! -f "$RESULT_FILE" ] || [ "$FORCE_REPEAT" -eq 1 ]; then
            echo "Parsing results to: $RESULT_FILE"
            python3 $ALPDIR/include/graphblas/cost/cost_and_time_parser_plus_DEBUG.py $OUTPUT_FILE > $RESULT_FILE
        else
            echo "Result file $RESULT_FILE already exists, skipping analysis"
        fi

        echo "Completed analysis for size $N with $THREADS threads"
        echo "----------------------------------------"
    done

    echo "Generating plots for $THREADS threads"
    python3 $ALPDIR/include/graphblas/cost/cost_model_plot_cg_analysis.py --results-dir $THREAD_DIR/results --threads $THREADS
    cp -r $THREAD_DIR/results $DATADIR/results/plots/t$THREADS
    echo "Benchmark complete for $THREADS threads!"
    echo "========================================================"
done

echo "All thread count benchmarks complete! Results are in the '$DATADIR' directory"