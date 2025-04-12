#!/bin/bash

echo "run_matcher.sh started..."

gpus="h100-96"
constraint="xgpi"

srun_command="srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus $gpus --constraint $constraint"

fasta_file="signatures.fasta"
fastq_file="sample.fastq"

$srun_command make
$srun_command nsys nvprof ./matcher $fastq_file $fasta_file

echo "run_matcher.sh ended!"
