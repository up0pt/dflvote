#!/bin/bash
#SBATCH --output=/home/members/nakadam/dflvote/jobs/job%j.out  # where to store the output (%j is the JOBID)
#SBATCH --error=/home/members/nakadam/dflvote/jobs/job%j.err  # where to store error messages
#SBATCH --mem=160G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodelist=heart

PROJECT_ROOT=/home/members/nakadam/dflvote

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

cd $PROJECT_ROOT
pwd

# Sweep parameters
# WARNING CHANGE FOLDER NAME!!!!
folder_names=("exp7_cl32_average_dist")

num_clients_arr=(32)
num_attackers_limit=(16)
datasets=("MNIST")
epochs=(400)
alphas=(0.5)
lrs=(0.001)
datasizes=(100)
dist_arr=([[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15],[16,17,18,19,20,21,22,23],[24,25,26,27,28,29,30,31]])


# For each num_clients value, generate groups via Python
mapfile -t groups_arr < <(python3 - << 'EOF'
import json
from src.generate_division import random_dividions, generate_divisions
n = 32  # use num_clients value here
groups = random_dividions(n)
groups += generate_divisions(n)
print(*[json.dumps(g) for g in groups], sep="\n")
EOF
)
for folder_n in "${folder_names[@]}"; do
  for num_clients in "${num_clients_arr[@]}"; do
    for num_attackers in "${num_attackers_limit[@]}"; do
      for groups in "${groups_arr[@]}"; do
        for dist in "${dist_arr[@]}"; do
          for dataset in "${datasets[@]}"; do
            for epoch in "${epochs[@]}"; do
              for lr in "${lrs[@]}"; do
                for alpha in "${alphas[@]}"; do
                  for datasize in "${datasizes[@]}"; do
                    echo ">>> num_clients=$num_clients, num_attackers_limit=$num_attackers, dist_groups=$dist, groups=$groups, dataset=$dataset, epoch=$epoch, lr=$lr, alpha=$alpha, datasize=$datasize"
                    uv run src/main.py \
                        --num_clients "$num_clients" \
                        --num_attackers_limit "$num_attackers" \
                        --groups "$groups" \
                        --dataset "$dataset" \
                        --dists_groups "$dist" \
                        --epoch "$epoch" \
                        --lr "$lr" \
                        --alpha "$alpha" \
                        --num_train_data "$datasize" \
                        --folder_name "$folder_n"
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "Finished at: $(date)"
exit 0
