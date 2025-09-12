#!/bin/bash
# filepath: /home/panastasiadis/ALP/include/graphblas/cost/benchmark_cg.sh

# Set to 1 to force re-running all steps even if files already exist
FORCE_REPEAT=0

# Create output directories if they don't exist
mkdir -p matrices
mkdir -p outputs
mkdir -p results
mkdir -p results/plots

ALPDIR="/home/panastasiadis/ALP"
# Define range of problem sizes to test
# You can adjust these values as needed
SIZES=(256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288) # 1048576)
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
    OUTPUT_FILE="outputs/banded_diag_${N}x${N}_band_${BANDSIZE}_output.log"
    if [ ! -f "$OUTPUT_FILE" ] || [ "$FORCE_REPEAT" -eq 1 ]; then
        echo "Running conjugate gradient solver, output to: $OUTPUT_FILE"
        $ALPDIR/build_nocostprints/tests/smoke/conjugate_gradient_reference_omp $MATRIX_FILE direct 1 1 > $OUTPUT_FILE 2>&1
    else
        echo "Output file $OUTPUT_FILE already exists, skipping conjugate gradient solver"
    fi
    
    # 3. Parse the output
    RESULT_FILE="results/banded_diag_${N}x${N}_band_${BANDSIZE}_analysis.log"
    if [ ! -f "$RESULT_FILE" ] || [ "$FORCE_REPEAT" -eq 1 ]; then
        echo "Parsing results to: $RESULT_FILE"
        python3 $ALPDIR/include/graphblas/cost/cost_and_time_parser_plus_DEBUG.py $OUTPUT_FILE > $RESULT_FILE
    else
        echo "Result file $RESULT_FILE already exists, skipping analysis"
    fi
    
    echo "Completed analysis for size $N"
    echo "----------------------------------------"
done
python3 $ALPDIR/include/graphblas/cost/cost_model_plot_cg_analysis.py

echo "Benchmark complete! Results are in the 'results' directory"