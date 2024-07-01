#!/bin/bash
#DSUB -A root.bingxing2.gpuuser486
#DSUB -q root.default
#DSUB -l wuhanG5500
#DSUB --job_type cosched
#DSUB -R 'cpu=6;gpu=1;mem=45000'
#DSUB -N 1
#DSUB -e %J.out
#DSUB -o %J.out
export PYTHONUNBUFFERED=1
module load anaconda/2021.11
module load cuda/11.8
source activate BraVO

# Create a state file to control the collection process
STATE_FILE="state_${BATCH_JOB_ID}.log"
/usr/bin/touch ${STATE_FILE}

# Collect GPU data in the background, one sample per second
# The collected data will be output to the local gpu_job_id.log file
function gpus_collection() {
  while [[ `cat "${STATE_FILE}" | grep "over" | wc -l` == "0" ]]; do
    /usr/bin/sleep 1
    /usr/bin/nvidia-smi >> "gpu_${BATCH_JOB_ID}.log"
  done
}
gpus_collection &

# Execute the example script
python main.py

# Stop the GPU collection process
echo "over" >> "${STATE_FILE}"