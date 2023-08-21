#!/bin/bash

if [ -z "${DATASETS_PATH}" || ! -d "${DATASETS_PATH}" ]; then
    echo "Please provide the correct path to the datasets directory"
    exit 1
fi

# Create output directories
mkdir -p "${HYPERDAGS_OUTPUT_FOLDER}/limited_iterations"
mkdir -p "${HYPERDAGS_OUTPUT_FOLDER}/until_convergence"


# Build tests (if needed)
make build_tests_category_smoke -j$(nproc)


## Limited iterations tests
CURRENT_OUT_DIR="${HYPERDAGS_OUTPUT_FOLDER}/limited_iterations"

HYPERDAGS_OUTPUT_PATH="${CURRENT_OUT_DIR}/bicgstab.mtx" \
    ${TEST_BIN_DIR}/bicgstab_hyperdags ${DATASETS_PATH}/gyro_m.mtx direct 1 1

HYPERDAGS_OUTPUT_PATH="${CURRENT_OUT_DIR}/conjugate_gradient.mtx" \
    ${TEST_BIN_DIR}/conjugate_gradient_hyperdags ${DATASETS_PATH}/gyro_m.mtx direct 1 1

HYPERDAGS_OUTPUT_PATH="${CURRENT_OUT_DIR}/pregel.mtx" \
    ${TEST_BIN_DIR}/pregel_connected_components_hyperdags ${DATASETS_PATH}/west0497.mtx direct 1 1

HYPERDAGS_OUTPUT_PATH="${CURRENT_OUT_DIR}/simple_pagerank.mtx" \
    ${TEST_BIN_DIR}/simple_pagerank_hyperdags ${DATASETS_PATH}/west0497.mtx direct 1 1


## Until convergence tests
CURRENT_OUT_DIR="${HYPERDAGS_OUTPUT_FOLDER}/until_convergence"

HYPERDAGS_OUTPUT_PATH="${CURRENT_OUT_DIR}/bicgstab_gyro_m.mtx" \
    ${TEST_BIN_DIR}/bicgstab_hyperdags ${DATASETS_PATH}/gyro_m.mtx direct 1 1

HYPERDAGS_OUTPUT_PATH="${CURRENT_OUT_DIR}/conjugate_gradient_gyro_m.mtx" \
    ${TEST_BIN_DIR}/conjugate_gradient_hyperdags ${DATASETS_PATH}/gyro_m.mtx direct 1 1

HYPERDAGS_OUTPUT_PATH="${CURRENT_OUT_DIR}/hpcg.mtx" \
    ${TEST_BIN_DIR}/hpcg

HYPERDAGS_OUTPUT_PATH="${CURRENT_OUT_DIR}/k-NN_3_gyro_m.mtx" \
    ${TEST_BIN_DIR}/knn_hyperdags 3 ${DATASETS_PATH}/gyro_m.mtx direct 1 1

HYPERDAGS_OUTPUT_PATH="${CURRENT_OUT_DIR}/k-means.mtx" \
    ${TEST_BIN_DIR}/kmeans_hyperdags

HYPERDAGS_OUTPUT_PATH="${CURRENT_OUT_DIR}/label_propagation_256.mtx" \
    ${TEST_BIN_DIR}/labeltest_hyperdags 256

HYPERDAGS_OUTPUT_PATH="${CURRENT_OUT_DIR}/label_propagation_4096.mtx" \
    ${TEST_BIN_DIR}/labeltest_hyperdags 4096

HYPERDAGS_OUTPUT_PATH="${CURRENT_OUT_DIR}/pregel_connected_components_gyro_m.mtx" \
    ${TEST_BIN_DIR}/pregel_connected_components_hyperdags ${DATASETS_PATH}/gyro_m.mtx direct 1 1

HYPERDAGS_OUTPUT_PATH="${CURRENT_OUT_DIR}/simple_pagerank_gyro_m.mtx" \
    ${TEST_BIN_DIR}/simple_pagerank_hyperdags ${DATASETS_PATH}/gyro_m.mtx direct 1 1

if [ -f "${DATASETS_PATH}/wikipedia-20051105.mtx" ]; then
    HYPERDAGS_OUTPUT_PATH="${CURRENT_OUT_DIR}/simple_pagerank_wikipedia-20051105.mtx" \
    ${TEST_BIN_DIR}/simple_pagerank_hyperdags ${DATASETS_PATH}/wikipedia-20051105.mtx direct 1 1
else
    echo "Skipping simple_pagerank_wikipedia-20051105.mtx test"
fi

if [ ! -z "${GNN_DATASET_PATH}" && -d "${GNN_DATASET_PATH}" ]; then
    HYPERDAGS_OUTPUT_PATH="${CURRENT_OUT_DIR}/graphchallenge_nn_single_inference_1024neurons_120layers.mtx" \
        ${TEST_BIN_DIR}/graphchallenge_nn_single_inference_hyperdags ${GNN_DATASET_PATH} 1024 120 294 1 32 indirect 1 1
else
    echo "Skipping graphchallenge_nn_single_inference_1024neurons_120layers.mtx test"
fi