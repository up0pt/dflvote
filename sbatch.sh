#!/bin/bash
#SBATCH --output=/home/members/nakadam/dflvote/jobs/job%j.out  # where to store the output (%j is the JOBID)
#SBATCH --error=/home/members/nakadam/dflvote/jobs/job%j.err  # where to store error messages
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=liver

PROJECT_ROOT=/home/members/nakadam/dflvote

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

cd $PROJECT_ROOT
pwd

# Sweep parameters
num_clients_arr=(32)
num_attackers_arr=(0 1 2 4 6 8)
datasets=("MNIST")
epochs=(40)
alphas=(10)
lrs=(0.001)
datasizes=(100 500 1000 6000)

# For each num_clients value, generate groups via Python
mapfile -t groups_arr < <(python3 - << 'EOF'
import json
from src.generate_division import generate_divisions
n = 16  # use num_clients value here
groups = generate_divisions(n)
print(*[json.dumps(g) for g in groups], sep="\n")
EOF
)

for num_clients in "${num_clients_arr[@]}"; do
  for num_attackers in "${num_attackers_arr[@]}"; do
    for groups in "${groups_arr[@]}"; do
      for dataset in "${datasets[@]}"; do
        for epoch in "${epochs[@]}"; do
          for lr in "${lrs[@]}"; do
            for alpha in "${alphas[@]}"; do
              for datasize in "${datasizes[@]}"; do
                echo ">>> num_clients=$num_clients, num_attackers=$num_attackers, groups=$groups, dataset=$dataset, epoch=$epoch, lr=$lr, alpha=$alpha, datasize=$datasize"
                uv run src/main.py \
                    --num_clients "$num_clients" \
                    --num_attackers "$num_attackers" \
                    --groups "$groups" \
                    --dataset "$dataset" \
                    --epoch "$epoch" \
                    --lr "$lr" \
                    --alpha "$alpha" \
                    --num_train_data "$datasize"
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
