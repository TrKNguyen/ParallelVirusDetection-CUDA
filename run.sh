#!/bin/bash

## Change this to a job name you want
#SBATCH --job-name=lab3_job

## Change based on length of job and `sinfo` partitions available
#SBATCH --partition=gpu

## Request for a specific type of node
## Commented out for now, change if you need one
##SBATCH --constraint xgpe

## gpu:1 ==> any gpu. For e.g., gpu:a100-40:1 gets you one of the A100 GPU shared instances
#SBATCH --gres=gpu:1

## Must change this based on how long job will take. We are just expecting 30 seconds for now
#SBATCH --time=00:10:30

## Probably no need to change anything here
#SBATCH --ntasks=1

## May want to change this depending on how much host memory you need
## #SBATCH --mem-per-cpu=10G

sigfile="signatures_big.fasta"
samfile="sample_big.fastq"

echo "Job is running on $(hostname), started at $(date)"

# ./gen_sig <num_signatures> <min_length> <max_length> <n_ratio>
# ./gen_sig 1000 100 300 0.1 > $sigfile
#./gen_sig 1000 3000 10000 0.1 > $sigfile

#./gen_sample <fasta_file> <num_no_virus> <num_with_virus> <min_viruses> <max_viruses> <min_length>
# <max_length> <min_phred> <max_phred> <n_ratio>
# ./gen_sample $sigfile 2000 20 1 2 100 200 10 30 0.1 > $samfile
#./gen_sample $sigfile  2000 20 1 2 100000 2000000 10 30 0.1 > $samfile

# Get some output about GPU status
nvidia-smi 

# Set the nvidia compiler directory
NVCC=/usr/local/cuda/bin/nvcc

# Check that it exists and print some version info
[[ -f $NVCC ]] || { echo "ERROR: NVCC Compiler not found at $NVCC, exiting..."; exit 1; }
echo "NVCC info: $($NVCC --version)"

# Actually compile the code
echo -e "\n====> Compiling...\n"
make

# run matcher ./matcher <sample.fastq> <signatures.fasta>
echo -e "\n====> Running...\n"
./matcher $samfile $sigfile

echo -e "\n====> Finished running.\n"

echo -e "\nJob completed at $(date)"
