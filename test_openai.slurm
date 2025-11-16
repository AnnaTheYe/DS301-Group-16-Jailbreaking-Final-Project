#!/bin/bash
#SBATCH --job-name=test_openai
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4

singularity exec --nv \
  --bind /scratch/azy2006 \
  --overlay /scratch/azy2006/pytorch-example/my_pytorch.ext3:rw \
  /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
  /bin/bash -c "
  
source /ext3/env.sh
conda activate
python test_openai.py

"
