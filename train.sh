#!/bin/bash
#SBATCH -J train_dist_job
#SBATCH -o log.o%j
#SBATCH -t 48:00:00
#SBATCH -N 1 -n 9
#SBATCH --mem-per-cpu=8GB
#SBATCH --gpus=8
#SBATCH -A tsekos
 
module add python/3.9
module add PyTorch/2.5.1-foss-2023a-CUDA-12.1.1
module add torchvision/0.20.0-foss-2023a-CUDA-12.1.1
# module add torchvision
# conda init bash
# conda activate /project/tsekos/carter_baselines
pip install opencv_python
pip install matplotlib
pip install nibabel
pip install patchify
pip install timm 
pip install -r requirement.txt 
 
# python train.py $1 $2 $3 $4 $5
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_psnr.py --opt options/swinir/train_swinir_sr_classical.json  --dist True
python main_train_psnr_brats.py --opt options/swinir/train_swinir_denoising_gray_brats.json  --dist False
# python tt.py