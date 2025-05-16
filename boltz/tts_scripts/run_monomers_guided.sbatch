#!/bin/bash

#SBATCH --job-name=score_ipa
#SBATCH --output=score_ipa_%j.log
#SBATCH --error=score_ipa_%j.err 
#SBATCH --time=2-00:00:00     
#SBATCH --ntasks=1                       
#SBATCH --gres=gpu:volta:1

source /state/partition1/llgrid/pkg/anaconda/python-ML-2025a/etc/profile.d/conda.sh
conda activate boltz1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python /home/gridsan/jroney/solab/af3/FoldScaling/boltz/src/boltz/main.py predict /home/gridsan/jroney/solab/af3/FoldScaling/data/monomers_fasta --out_dir /home/gridsan/jroney/solab/af3/FoldScaling/data/monomer_predictions_fk --diffusion_samples 32 --confidence_fk --recycling_steps 0
