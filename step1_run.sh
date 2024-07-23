#!/bin/bash
export PYTHONUNBUFFERED=1
module load anaconda/2021.11
module load cuda/12.1
source activate BraVO

# Create a state file to control the collection process
STATE_FILE="state_${BATCH_JOB_ID}.log"
/usr/bin/touch ${STATE_FILE}

# Collecting data in the background, gathering GPU data every 1 second interval.
# Data collected will be outputted to the local log_[Job ID]/gpu.log file.
X_LOG_DIR="log_${SLURM_JOB_ID}"
X_GPU_LOG="${X_LOG_DIR}/gpu.log"
mkdir "${X_LOG_DIR}"
/usr/bin/touch ${X_GPU_LOG}
function gpus_collection(){
   sleep 15
   process=`ps -ef | grep python | grep $USER | grep -v "grep" | wc -l`
   while [[ "${process}" > "0" ]]; do
      sleep 1
      nvidia-smi >> "${X_GPU_LOG}" 2>&1
      echo "process num:${process}" >> "${X_GPU_LOG}" 2>&1
      process=`ps -ef | grep python | grep $USER | grep -v "grep" | wc -l`
   done
}
gpus_collection &

# Execute the example script
python make_nsd_data.py

# Stop the GPU collection process
echo "over" >> "${STATE_FILE}"