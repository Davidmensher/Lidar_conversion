#! /bin/sh

#SBATCH --job-name=awesome_david
#SBATCH --output=awesome_david.out
#SBATCH --error=awesome_david.err
#SBATCH --partition=studentbatch
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

python3 main.py -a resnet50 -d upproj -m rgbd -s 100 --data lidardepthv2 -b 6
