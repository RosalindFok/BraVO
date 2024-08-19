#!/bin/bash
export PYTHONUNBUFFERED=1
module load anaconda/2021.11
module load cuda/12.1
source activate BraVO

# Create a state file to control the collection process
STATE_FILE="state_${BATCH_JOB_ID}.log"
/usr/bin/touch ${STATE_FILE}

# Execute the example script
# python main.py --task analyze 
python main.py --task train 
python main.py --task test 
python main.py --task generate 

# Stop the GPU collection process
echo "over" >> "${STATE_FILE}"