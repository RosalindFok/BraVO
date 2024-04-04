#!/bin/bash
#DSUB -A root.bingxing2.gpuuser486
#DSUB -q root.default
#DSUB -l wuhanG5500
#DSUB --job_type cosched
#DSUB -R 'cpu=6;gpu=1;mem=45000'
#DSUB -N 1
#DSUB -e %J.out
#DSUB -o %J.out
export export PYTHONUNBUFFERED=1
module load anaconda/2021.11 
module load cuda/11.8
source activate BandCLIP
python -m image2caption.img2cap