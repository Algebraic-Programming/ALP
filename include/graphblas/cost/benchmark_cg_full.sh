#!/bin/bash

# Usage: ./benchmark_cg_controller.sh
# Set these flags to control which matrices to use
RUN_REAL=1
RUN_SYNTHETIC=1

# Function to get NUMA topology information
get_numa_topology() {
    if command -v lscpu &> /dev/null; then
        # Use lscpu to get NUMA node count
        local numa_nodes=$(lscpu | grep "NUMA node(s):" | awk '{print $3}')
        echo "${numa_nodes:-1}"
    elif [ -d "/sys/devices/system/node" ]; then
        # Count NUMA nodes from /sys filesystem
        local numa_nodes=$(ls -1d /sys/devices/system/node/node* 2>/dev/null | wc -l)
        echo "${numa_nodes:-1}"
    else
        # Fallback: assume 1 NUMA node
        echo "1"
    fi
}

# Function to get cores per NUMA node
get_cores_per_numa_node() {
    local num_numa_nodes=$1
    if command -v lscpu &> /dev/null; then
        # Get total number of cores and divide by NUMA nodes
        local total_cores=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
        echo $((total_cores / num_numa_nodes))
    else
        # Fallback: use nproc
        local total_cores=$(nproc --all)
        echo $((total_cores / num_numa_nodes))
    fi
}

# Function to calculate spread CPU affinity
calculate_spread_cpu_affinity() {
    local num_threads=$1
    local thread_offset=${2:-0}  # Default offset is 0 if not provided
    
    # Get NUMA topology
    local num_numa_nodes=$(get_numa_topology)
    local cores_per_numa=$(get_cores_per_numa_node $num_numa_nodes)
    
    echo "Spread policy: $num_numa_nodes NUMA nodes, $cores_per_numa cores per node" >&2
    
    # Calculate CPU affinity for spread policy
    local cpu_affinity=()
    local threads_per_node=$((num_threads / num_numa_nodes))
    
    for ((i=0; i<num_threads; i++)); do
        local numa_node
        local thread_in_node
        
        # Determine which NUMA node this thread should use
        if [ $i -lt $((threads_per_node * num_numa_nodes)) ]; then
            # Evenly distributed threads
            numa_node=$((i / threads_per_node))
            thread_in_node=$((i % threads_per_node))
        else
            # Remaining threads go to the first NUMA nodes
            numa_node=$((i - threads_per_node * num_numa_nodes))
            thread_in_node=$threads_per_node
        fi
        
        local cpu_id=$((thread_offset + numa_node * cores_per_numa + thread_in_node))
        cpu_affinity+=($cpu_id)
    done
    
    echo "Spread CPU affinity: ${cpu_affinity[*]}" >&2
    echo "${cpu_affinity[*]}"
}

MATRIX_DIR="/scratch/panastasiadis/matrices"
SYNTH_DIR="$MATRIX_DIR/synthetic"
MM_DIR="$MATRIX_DIR/MM_suite"
MODEL_NAME="d_4_GS_close"
ALLOC_POLICY="close"
DATADIR="/scratch/panastasiadis/${MODEL_NAME}"
ALPDIR="/home/panastasiadis/ALP"
BUILD_DIR="${ALPDIR}/build_nocost"
THREAD_COUNTS=(96 64 48 32 24 16 12 8 4 2 1)

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
    elif [ "$ALLOC_POLICY" == "spread" ]; then
        # Calculate spread CPU affinity using NUMA-aware distribution
        cpu_affinity=$(calculate_spread_cpu_affinity $THREADS 0)
        export GOMP_CPU_AFFINITY="${cpu_affinity// / }"
        echo "Set GOMP_CPU_AFFINITY to: $GOMP_CPU_AFFINITY"
    fi

    if [ "$RUN_REAL" -eq 1 ]; then
        for MM_MTX in $MM_DIR/*.mtx; do
            MM_NAME=$(basename "$MM_MTX" .mtx)
            mkdir -p $THREAD_DIR/results/analysis/real
            mkdir -p $THREAD_DIR/outputs/real
            OUTPUT_FILE="$THREAD_DIR/outputs/real/${MM_NAME}_output.log"
            RESULT_FILE="$THREAD_DIR/results/analysis/real/${MM_NAME}_analysis.log"
            if [ ! -f "$OUTPUT_FILE" ]; then
                echo "$BUILD_DIR/tests/smoke/conjugate_gradient_reference_omp "$MM_MTX" direct 1 1 > "$OUTPUT_FILE" 2>&1"
                $BUILD_DIR/tests/smoke/conjugate_gradient_reference_omp "$MM_MTX" direct 1 1 > "$OUTPUT_FILE" 2>&1
            fi
            if [ ! -f "$RESULT_FILE" ]; then
                echo "python3 $ALPDIR/include/graphblas/cost/parse_benchmark_logs.py "$OUTPUT_FILE" > "$RESULT_FILE""
                python3 $ALPDIR/include/graphblas/cost/parse_benchmark_logs.py "$OUTPUT_FILE" > "$RESULT_FILE"
            fi
        done

        echo "python3 $ALPDIR/include/graphblas/cost/benchmark_ploter.py --results-dir $DATADIR --threads $THREADS" --filegroup-name real
        python3 $ALPDIR/include/graphblas/cost/benchmark_ploter.py --results-dir $DATADIR --threads $THREADS --filegroup-name real
    fi

    if [ "$RUN_SYNTHETIC" -eq 1 ]; then
        BANDSIZE=1
        SIZES=(256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432)
        for N in "${SIZES[@]}"; do
            mkdir -p $THREAD_DIR/results/analysis/synthetic
            mkdir -p $THREAD_DIR/outputs/synthetic
            MATRIX_FILE="$SYNTH_DIR/banded_diag_${N}x${N}_band_${BANDSIZE}.mtx"
            OUTPUT_FILE="$THREAD_DIR/outputs/synthetic/banded_diag_${N}x${N}_band_${BANDSIZE}_output.log"
            RESULT_FILE="$THREAD_DIR/results/analysis/synthetic/banded_diag_${N}x${N}_band_${BANDSIZE}_analysis.log"
            if [ ! -f "$OUTPUT_FILE" ]; then
                echo "$BUILD_DIR/tests/smoke/conjugate_gradient_reference_omp "$MATRIX_FILE" direct 1 1 > "$OUTPUT_FILE" 2>&1"
                $BUILD_DIR/tests/smoke/conjugate_gradient_reference_omp "$MATRIX_FILE" direct 1 1 > "$OUTPUT_FILE" 2>&1
            fi
            if [ ! -f "$RESULT_FILE" ]; then
                echo "python3 $ALPDIR/include/graphblas/cost/parse_benchmark_logs.py "$OUTPUT_FILE" > "$RESULT_FILE""
                python3 $ALPDIR/include/graphblas/cost/parse_benchmark_logs.py "$OUTPUT_FILE" > "$RESULT_FILE"
            fi
        done
        echo "python3 $ALPDIR/include/graphblas/cost/benchmark_ploter.py --results-dir $DATADIR --threads $THREADS" --filegroup-name synthetic
        python3 $ALPDIR/include/graphblas/cost/benchmark_ploter.py --results-dir $DATADIR --threads $THREADS --filegroup-name synthetic
    fi
    mkdir -p $BUILD_DIR/${MODEL_NAME}
    cp -r $THREAD_DIR $BUILD_DIR/${MODEL_NAME}
done

echo "All thread count benchmarks complete! Results are in the '$DATADIR' directory"