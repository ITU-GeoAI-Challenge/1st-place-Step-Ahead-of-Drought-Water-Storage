#!/usr/bin/env bash
# Rebuild the final submission from raw data: every base member is retrained at the commit
# that produced it, then the 19 blend steps run in order.
#
# The two convolutional members read an earlier blend for the rows they do not predict, so
# they run between blend steps:
#   fieldcnn     needs blend_d2_fx
#   fielddirect3 needs blend_cnn
#
# Runtime: 280 minutes measured on 16 CPU cores and one RTX 3070. The two MLP bags take most of it.
# Output: submission_model/output/scratch/final_dart.csv
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=submission_model/output
WORK=$OUT/scratch
REB=$OUT/rebuilt
mkdir -p "$WORK"

# 1. base members with no dependency on a blend (era5_v5 runs before probe_persist)
$PY -W ignore submission_model/rebuild_members.py \
    eof climatology_nb final_v4 era5_v5 probe_persist recursive lineartree dart covonly_big mlpbag mlpbag8 \
    --inputs "$REB"

# 2. blend steps up to the re-anchor patch
# The solve2 steps apply the weights solved on the scored files (see rebuild_final.py docstring).
$PY -W ignore submission_model/rebuild_final.py --members "$REB" --work "$WORK" --scored-weights --until blend_d2_fx

# 3. first convolutional member, then the next blend steps
$PY -W ignore submission_model/rebuild_members.py fieldcnn --inputs "$WORK"
$PY -W ignore submission_model/rebuild_final.py --members "$REB" --work "$WORK" --scored-weights --resume --until blend_cnn

# 4. second convolutional member, then the rest of the chain
$PY -W ignore submission_model/rebuild_members.py fielddirect3 --inputs "$WORK"
$PY -W ignore submission_model/rebuild_final.py --members "$REB" --work "$WORK" --scored-weights --resume

echo "final file: $WORK/final_dart.csv"
