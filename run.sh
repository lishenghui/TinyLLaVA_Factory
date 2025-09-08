#!/bin/bash
#SBATCH -A NAISS2024-22-1358 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:4
#SBATCH --job-name=paaa
#SBATCH --tasks-per-node=1
#SBATCH --exclude=alvis3-08
#SBATCH --time=3:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err


EXE='apptainer exec --nv /mimer/NOBACKUP/groups/scalablefl/containers/fm.sif bash -c'
$EXE "source ~/.bashrc && export WANDB_MODE=offline && conda activate tinyllava && bash ./scripts/train/train_phi.sh"
# $EXE "source ~/.bashrc && export WANDB_MODE=offline && conda activate tinyllava && bash ./scripts/train/lora/train_phi_lora.sh"
# $EXE "source ~/.bashrc && export WANDB_MODE=offline && bash ./scripts/exp/exp_2nd_stage_finetune/full_2nd_stage.sh"
# $EXE "source ~/.bashrc && export WANDB_MODE=offline && bash ./scripts/exp/exp_1st_stage_pretrain/pretrain_phi2.sh"

# EXE='apptainer exec --nv /mimer/NOBACKUP/groups/scalablefl/containers/torch.sif bash -c "source ~/.bashrc && python"'
# $EXE ./scripts/exp/exp_1st_stage_pretrain/pretrain_phi2.sh
