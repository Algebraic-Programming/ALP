#!/bin/bash
# filepath: /home/panastasiadis/ALP/include/graphblas/cost/benchmark_cg.sh

#SBATCH --job-name=graphblas_cg_benchmark    # Job name
#SBATCH --output=benchmark_cg_%j.out         # Standard output file (%j expands to jobId)
#SBATCH --error=benchmark_cg_%j.err          # Standard error file (%j expands to jobId)
#SBATCH --time=24:00:00                      # Time limit (24 hours)
#SBATCH --exclusive                           # Exclusive node allocation
#SBATCH --nodes=1                             # Request 1 node
#SBATCH --ntasks=1                            # Run a single task
#SBATCH --cpus-per-task=1                    # Request 24 CPU cores per task (matches OMP_NUM_THREADS=24)
#SBATCH --partition=ARM                       # Specify partition/queue name (change as needed)

# Set to 1 to force re-running all steps even if files already exist
FORCE_REPEAT=0

# Create output directories if they don't exists
mkdir -p matrices
mkdir -p outputs
mkdir -p results
mkdir -p results/plots

export OMP_NUM_THREADS=1  # Set number of threads for OpenMP
export OMP_PROC_BIND=true
export OMP_PLACES={0:1}

ALPDIR="/home/panastasiadis/ALP"
# Define range of problem sizes to test
# You can adjust these values as needed
SIZES=(256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 ) #524288 1048576 2097152 4194304 8388608
BANDSIZE=1
# Loop through each problem size
for N in "${SIZES[@]}"; do
    echo "=== Processing matrix of size $N x $N ==="
    
    # 1. Create band-diagonal matrix with band size 1
    MATRIX_FILE="matrices/banded_diag_${N}x${N}_band_${BANDSIZE}.mtx"
    if [ ! -f "$MATRIX_FILE" ] || [ "$FORCE_REPEAT" -eq 1 ]; then
        echo "Generating matrix: $MATRIX_FILE"
        python3 $ALPDIR/include/graphblas/cost/mtx_generator.py $N $BANDSIZE matrices
    else
        echo "Matrix file $MATRIX_FILE already exists, skipping generation"
    fi
    
    # 2. Run conjugate gradient solver
    OUTPUT_FILE="outputs/banded_diag_${N}x${N}_band_${BANDSIZE}_threads-${OMP_NUM_THREADS}_output.log"
    if [ ! -f "$OUTPUT_FILE" ] || [ "$FORCE_REPEAT" -eq 1 ]; then
        echo "Running conjugate gradient solver, output to: $OUTPUT_FILE"
        $ALPDIR/build/tests/smoke/conjugate_gradient_reference_omp $MATRIX_FILE direct 1 1 > $OUTPUT_FILE 2>&1
    else
        echo "Output file $OUTPUT_FILE already exists, skipping conjugate gradient solver"
    fi
    
    # 3. Parse the output
    RESULT_FILE="results/banded_diag_${N}x${N}_band_${BANDSIZE}_threads-${OMP_NUM_THREADS}_analysis.log"
    #if [ ! -f "$RESULT_FILE" ] || [ "$FORCE_REPEAT" -eq 1 ]; then
        echo "Parsing results to: $RESULT_FILE"
        python3 $ALPDIR/include/graphblas/cost/cost_and_time_parser_plus_DEBUG.py $OUTPUT_FILE > $RESULT_FILE
    #else
    #    echo "Result file $RESULT_FILE already exists, skipping analysis"
    #fi
    
    echo "Completed analysis for size $N"
    echo "----------------------------------------"
done

python3 $ALPDIR/include/graphblas/cost/cost_model_plot_cg_analysis.py
echo "Benchmark complete! Results are in the 'results' directory"