#!/usr/bin/env bash

CONFIG=${CONFIG:-ogbl_ddi_cfg}
GRID=${GRID:-ogbl_ddi_grid}
REPEAT=${REPEAT:-10}
MAX_JOBS=${MAX_JOBS:-8}
SLEEP=${SLEEP:-1}
MAIN=${MAIN:-main}
METRIC=${METRIC:-ogbl_ddi_hits_20}

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python configs_gen.py --config configs/ogb/ogbl/${CONFIG}.yaml \
  --grid grids/ogb/ogbl/${GRID}.txt \
  --out_dir configs/ogb/ogbl

# run batch of configs
# Args: config_dir, num of repeats, max jobs running, sleep time
bash parallel.sh configs/ogb/ogbl/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
bash parallel.sh configs/ogb/ogbl/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
bash parallel.sh configs/ogb/ogbl/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN

# aggregate results for the batch
python agg_batch.py --dir results/ogb/ogbl/${CONFIG}_grid_${GRID} --metric $METRIC
