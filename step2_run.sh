#!/bin/bash
export PYTHONUNBUFFERED=1
# module load anaconda/2021.11 # N32EA14P
module load anaconda/2022.10 # N40R4
module load cuda/12.1
source activate BraVO

# Create a state file to control the collection process
STATE_FILE="state_${BATCH_JOB_ID}.log"
/usr/bin/touch ${STATE_FILE}

# Execute the example script
python main.py --task t --tower_name caption
# python main.py --task t --tower_name image

# Stop the GPU collection process
echo "over" >> "${STATE_FILE}"