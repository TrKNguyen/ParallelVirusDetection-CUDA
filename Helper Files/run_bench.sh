#!/bin/bash

## Change this to a job name you want
#SBATCH --job-name=gpu_job

## Change based on length of job and `sinfo` partitions available
#SBATCH --partition=gpu

## Request for a specific type of node
## Commented out for now, change if you need one
#SBATCH --constraint xgph

## gpu:1 ==> any gpu. For e.g., gpu:a100-40:1 gets you one of the A100 GPU shared instances
#SBATCH --gres=gpu:a100-40:1

## Must change this based on how long job will take. We are just expecting 30 seconds for now
#SBATCH --time=00:15:00

## Probably no need to change anything here
#SBATCH --ntasks=1

## May want to change this depending on how much host memory you need
## #SBATCH --mem-per-cpu=10G

## Just useful logfile names
#SBATCH --output=job-gpu_%j.slurmlog
#SBATCH --error=job-gpu_%j.slurmlog


echo "Job is running on $(hostname), started at $(date)"

bench_exec="bench-a100"
fasta_file="signatures_big.fasta"
fastq_file="sample_big.fastq"

rm matcher
make matcher

./matcher $fastq_file $fasta_file | sort > my-out.txt
./$bench_exec $fastq_file $fasta_file | sort > their-out.txt
diff my-out.txt their-out.txt

echo "Job ended at $(date)"
