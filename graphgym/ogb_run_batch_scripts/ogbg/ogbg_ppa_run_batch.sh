#!/usr/bin/env bash

CONFIG=${CONFIG:-ogbg_ppa_cfg}
GRID=${GRID:-ogbg_ppa_grid}
REPEAT=${REPEAT:-3}
MAX_JOBS=${MAX_JOBS:-8}
SLEEP=${SLEEP:-1}
MAIN=${MAIN:-main}
METRIC=${METRIC:-ogbg_ppa_accuracy}

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python configs_gen.py --config configs/ogb/ogbg/${CONFIG}.yaml \
  --grid grids/ogb/ogbg/${GRID}.txt \
  --out_dir configs/ogb/ogbg

# run batch of configs
# Args: config_dir, num of repeats, max jobs running, sleep time
bash parallel.sh configs/ogb/ogbg/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
bash parallel.sh configs/ogb/ogbg/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
bash parallel.sh configs/ogb/ogbg/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN

# aggregate results for the batch
python agg_batch.py --dir results/ogb/ogbg/${CONFIG}_grid_${GRID} --metric $METRIC
