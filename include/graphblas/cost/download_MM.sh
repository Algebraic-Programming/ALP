RUNDIR=/scratch/panastasiadis
MATRIX_LIST=("gyro_m" "vanbody" "G2_circuit" "bundle_adj" "apache2" "Emilia_923" "ecology2" "Serena" "G3_circuit" "Queen_4147")
GROUP_LIST=("Oberwolfach" "GHS_psdef" "AMD" "Mazaheri" "GHS_psdef" "Janna" "McRae" "Janna" "AMD" "Janna") # <-- Fill with the group for each matrix, e.g. GROUP_LIST[0] for MATRIX_LIST[0]
# Directories
mkdir -p ${RUNDIR}/all_mtx/downloads
mkdir -p ${RUNDIR}/all_mtx
echo "Running on dir $RUNDIR/all_mtx"
for idx in "${!MATRIX_LIST[@]}"; do
    M="${MATRIX_LIST[$idx]}"
    GROUP="${GROUP_LIST[$idx]}"
    finalfile="${M}.mtx"
    if [ -f "$finalfile" ]; then
        echo "  Skipping $M (already exists)"
        continue
    fi
    if [ -z "$GROUP" ]; then
        echo "✗ No group specified for $M in GROUP_LIST"
        continue
    fi
    echo "  Using group: $GROUP for matrix: $M"
    echo "  Downloading $M..."
    wget -q -O "downloads/${M}.tar.gz" "https://suitesparse-collection-website.herokuapp.com/MW/${GROUP}/${M}.tar.gz"
    if [ ! -s "downloads/${M}.tar.gz" ]; then
        echo "✗ Download failed or file is empty for $M (${GROUP})"
        continue
    fi
    outdir="downloads/extracted_${M}"
    mkdir -p "$outdir"
    tar -xzvf downloads/${M}.tar.gz -C "$outdir"
    mfile=$(find "$outdir" -type f -name "*.mtx" | head -n 1)
    if [ -n "$mfile" ]; then
        cp "$mfile" "$finalfile"
        rm -r "$outdir"
        echo "✔ Stored locally as $finalfile"
    else
        echo "▲ No .mtx found for $M"
    fi
done