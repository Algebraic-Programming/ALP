#!/bin/bash
# filepath: /home/panastasiadis/ALP/include/graphblas/cost/benchmark_cg_threads.sh

#SBATCH --job-name=graphblas_cg_thread_benchmark    # Job name
#SBATCH --output=benchmark_cg_threads_%j.out         # Standard output file
#SBATCH --error=benchmark_cg_threads_%j.err          # Standard error file
#SBATCH --time=12:00:00                             # Time limit (48 hours)
#SBATCH --exclusive                                 # Exclusive node allocation
#SBATCH --nodes=1                                   # Request 1 node
#SBATCH --ntasks=1                                  # Run a single task
#SBATCH --partition=ARM                             # Specify partition/queue name

# Set to 1 to force re-running all steps even if files already exist
FORCE_REPEAT=0

# Create output directories if they don't exist
mkdir -p matrices
mkdir -p outputs
mkdir -p results
mkdir -p results/plots

# Define thread counts to test
THREAD_COUNTS=(92 48 24 1 2 4 8 12)  # Adjust as needed for your system

# Define range of problem sizes to test
# You can adjust these values as needed
SIZES=(256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304) #
BANDSIZE=1

ALPDIR="/home/panastasiadis/ALP"

# First, generate all matrices (this is thread-independent)
for N in "${SIZES[@]}"; do
    MATRIX_FILE="matrices/banded_diag_${N}x${N}_band_${BANDSIZE}.mtx"
    if [ ! -f "$MATRIX_FILE" ] || [ "$FORCE_REPEAT" -eq 1 ]; then
        echo "Generating matrix: $MATRIX_FILE"
        python3 $ALPDIR/include/graphblas/cost/mtx_generator.py $N $BANDSIZE matrices
    else
        echo "Matrix file $MATRIX_FILE already exists, skipping generation"
    fi
done

# Now run benchmarks for each thread count
for THREADS in "${THREAD_COUNTS[@]}"; do
    echo "========================================================"
    echo "=== Running benchmarks with $THREADS threads ============"
    echo "========================================================"
    
    # Set OpenMP environment for this thread count
    export OMP_NUM_THREADS=$THREADS
    export OMP_PROC_BIND=true
    export OMP_PLACES={0:$THREADS}  # Adjust placement according to thread count
    
    # Create thread-specific plots directory
    mkdir -p results/plots/t$THREADS
    
    # Loop through each problem size
    for N in "${SIZES[@]}"; do
        echo "=== Processing matrix of size $N x $N with $THREADS threads ==="
        
        # Run conjugate gradient solver
        OUTPUT_FILE="outputs/banded_diag_${N}x${N}_band_${BANDSIZE}_threads-${THREADS}_output.log"
        if [ ! -f "$OUTPUT_FILE" ] || [ "$FORCE_REPEAT" -eq 1 ]; then
            echo "Running conjugate gradient solver with $THREADS threads, output to: $OUTPUT_FILE"
            $ALPDIR/build/tests/smoke/conjugate_gradient_reference_omp $MATRIX_FILE direct 1 1 > $OUTPUT_FILE 2>&1
        else
            echo "Output file $OUTPUT_FILE already exists, skipping conjugate gradient solver"
        fi
        
        # Parse the output
        RESULT_FILE="results/banded_diag_${N}x${N}_band_${BANDSIZE}_threads-${THREADS}_analysis.log"
        #if [ ! -f "$RESULT_FILE" ] || [ "$FORCE_REPEAT" -eq 1 ]; then
            echo "Parsing results to: $RESULT_FILE"
            python3 $ALPDIR/include/graphblas/cost/cost_and_time_parser_plus_DEBUG.py $OUTPUT_FILE > $RESULT_FILE
        #else
        #    echo "Result file $RESULT_FILE already exists, skipping analysis"
        #fi
        
        echo "Completed analysis for size $N with $THREADS threads"
        echo "----------------------------------------"
    done
    
    # Generate plots for this thread count
    echo "Generating plots for $THREADS threads"
    python3 $ALPDIR/include/graphblas/cost/cost_model_plot_cg_analysis.py
    
    echo "Benchmark complete for $THREADS threads!"
    echo "========================================================"
done

echo "All thread count benchmarks complete! Results are in the 'results' directory"