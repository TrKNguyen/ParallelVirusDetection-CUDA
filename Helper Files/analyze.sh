#!/bin/bash
#SBATCH --job-name=matcher_run
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --output=matcher_run_%j.out
#SBATCH --error=matcher_run_%j.err

module load cuda/11.4

# Directory where test files are stored (adjusted to the new naming scheme)
TEST_DIR="test_inputs/virus"
# Directory to store all profiling outputs
PROFILE_DIR="profiles"
mkdir -p "$PROFILE_DIR"
# Combined log file for all profiling output
PROFILE_LOG="profile_results.log"
echo "Matcher Profiling Runs" | tee "$PROFILE_LOG"

# Loop over generated signature files (e.g., sig1.fasta, sig2.fasta, etc.)
for sigfile in ${TEST_DIR}/sig*.fasta; do
  baseSig=$(basename "$sigfile" .fasta)  # e.g., sig1
  echo "-------------------------------------------------" | tee -a "$PROFILE_LOG"
  echo "Running matcher for signature file: ${sigfile}" | tee -a "$PROFILE_LOG"
  
  # For each signature file, run over matching sample files.
  # New naming scheme: sample files are named as samXY.fastq, where X corresponds to the signature number.
  for samfile in ${TEST_DIR}/sam${baseSig:3}*.fastq; do
    echo "Running matcher on sample=${samfile} and signature=${sigfile}" | tee -a "$PROFILE_LOG"
    
    # Run matcher normally, redirecting its output to /dev/null
    ./matcher "${samfile}" "${sigfile}" > /dev/null 2>&1
    
    # Profile with Nsight Compute (kernel-level profiling)
    ncu_out="${PROFILE_DIR}/ncu_$(basename ${samfile} .fastq)_${baseSig}"
    echo "Profiling with Nsight Compute, output: ${ncu_out}.ncu-rep" | tee -a "$PROFILE_LOG"
    ncu --target-processes all --metrics achieved_occupancy,sm_efficiency --clock-control none -o "${ncu_out}" ./matcher "${samfile}" "${sigfile}" 2>&1 | tee -a "$PROFILE_LOG"
    
    # Profile with Nsight Systems (timeline profiling)
    nsys_out="${PROFILE_DIR}/nsys_$(basename ${samfile} .fastq)_${baseSig}"
    echo "Profiling with Nsight Systems, output: ${nsys_out}.qdrep" | tee -a "$PROFILE_LOG"
    nsys profile --trace=cuda,osrt --stats=true -o "${nsys_out}" ./matcher "${samfile}" "${sigfile}" 2>&1 | tee -a "$PROFILE_LOG"
    
    echo "Completed run for sample file: ${samfile} with signature file: ${sigfile}" | tee -a "$PROFILE_LOG"
    echo "-------------------------------------------------" | tee -a "$PROFILE_LOG"
  done
done

echo "All tests completed." | tee -a "$PROFILE_LOG"
cat "$PROFILE_LOG"
