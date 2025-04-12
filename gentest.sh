#!/bin/bash
#SBATCH --job-name=gen_tests
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --output=gen_tests_%j.out
#SBATCH --error=gen_tests_%j.err

module load cuda/11.4

# Create a directory to store test input files in test_inputs/test
TEST_DIR="test_inputs/virus"
mkdir -p "$TEST_DIR"

# Signature configurations: "num_signatures min_length max_length n_ratio"
declare -a SIG_CONFIGS=(
  "1000 3000 10000 0.1"
)

# Sample configurations: "num_no_virus num_with_virus min_viruses max_viruses min_length max_length min_phred max_phred n_ratio"
declare -a SAMPLE_CONFIGS=(
  "1000 10 100 200 100000 200000 10 30 0.1"
  "1000 10 10 20 100000 200000 10 30 0.1"
  "1000 10 50 70 100000 200000 10 30 0.1"
)

sig_count=1
for sigParams in "${SIG_CONFIGS[@]}"; do
  sigfile="${TEST_DIR}/sig${sig_count}.fasta"
  echo "Generating signature file (sig${sig_count}): ${sigParams}"
  ./gen_sig ${sigParams} > "${sigfile}"
  
  sample_count=1
  for sampParams in "${SAMPLE_CONFIGS[@]}"; do
    samfile="${TEST_DIR}/sam${sig_count}${sample_count}.fastq"
    echo "Generating sample file (sam${sig_count}${sample_count}) with parameters: ${sampParams}"
    ./gen_sample "${sigfile}" ${sampParams} > "${samfile}"
    sample_count=$((sample_count+1))
  done
  sig_count=$((sig_count+1))
done

echo "Test input files generated in ${TEST_DIR}"
