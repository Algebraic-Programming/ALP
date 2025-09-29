#!/bin/bash

# Usage: ./download_MM.sh /path/to/MM_suite
MM_DIR="$1"
mkdir -p "$MM_DIR/downloads"

MM_MATRIX_LIST=("gyro_m" "vanbody" "G2_circuit" "bundle_adj" "apache2" "Emilia_923" "ecology2" "Serena" "G3_circuit" "Queen_4147")
MM_GROUP_LIST=("Oberwolfach" "GHS_psdef" "AMD" "Mazaheri" "GHS_psdef" "Janna" "McRae" "Janna" "AMD" "Janna")

for idx in "${!MM_MATRIX_LIST[@]}"; do
    M="${MM_MATRIX_LIST[$idx]}"
    GROUP="${MM_GROUP_LIST[$idx]}"
    finalfile="${MM_DIR}/${M}.mtx"
    if [ -f "$finalfile" ]; then
        echo "Skipping $M (already exists)"
        continue
    fi
    if [ -z "$GROUP" ]; then
        echo "No group specified for $M"
        continue
    fi
    wget -q -O "$MM_DIR/downloads/${M}.tar.gz" "https://suitesparse-collection-website.herokuapp.com/MM/${GROUP}/${M}.tar.gz"
    if [ ! -s "$MM_DIR/downloads/${M}.tar.gz" ]; then
        echo "Download failed for $M"
        continue
    fi
    outdir="$MM_DIR/downloads/extracted_${M}"
    mkdir -p "$outdir"
    tar -xzvf "$MM_DIR/downloads/${M}.tar.gz" -C "$outdir"
    mfile=$(find "$outdir" -type f -name "*.mtx" | head -n 1)
    if [ -n "$mfile" ]; then
        cp "$mfile" "$finalfile"
        rm -r "$outdir"
        rm -r "$MM_DIR/downloads/${M}.tar.gz"
        echo "Stored $finalfile"
    else
        echo "No .mtx found for $M"
    fi
done