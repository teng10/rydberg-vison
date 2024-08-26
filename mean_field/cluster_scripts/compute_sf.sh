#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 2-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue #shared #  # Partition to submit to
#SBATCH --mail-user=y_teng@g.harvard.edu #Email for notifications
#SBATCH --mail-type=END #This command would send an email when the job ends.
#SBATCH --mem=50000
#SBATCH -c 1
#SBATCH --array=0-17 # 18 different parameter settings
# enumerating in parameter {bond_dimension}, {onsite_z_field} index.
# e.g. 2 x 3=6 This enumerates 6 parameter setting
#SBATCH -o /n/home11/yteng/experiments/Vison/logsVison/%A_%a.out # Standard out
#SBATCH -e /n/home11/yteng/experiments/Vison/logsVison/%A_%a.err # Standard err
module load python/3.10.9-fasrc01
# source activate rydberg-vison
source ../../venv/bin/activate
package_path="/n/home11/yteng/rydberg-vison/"
cd ${package_path}
FILEPATH="/n/home11/yteng/experiments/Vison"
data_dir="$FILEPATH/dataVison/%CURRENT_DATE/"
mkdir -p ${data_dir}
echo "Data saving directory is ${data_dir}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"

python -m mean_field.run_structure_factor \
--ham_config=mean_field/ham_configs/${1:-'ising'}_config.py \
--ham_config.job_id=${SLURM_ARRAY_JOB_ID} \
--ham_config.task_id=${SLURM_ARRAY_TASK_ID} \
--ham_config.output.data_dir=${data_dir} \
--ham_config.sweep_name=${2:-"sweep_t_m"} # $2-SWEEP_NAME

echo "job finished"
> "$FILEPATH/logsVison/${SLURM_ARRAY_JOB_ID}_log.txt"
