#!/bin/bash
#SBATCH --output=/home/members/nakadam/dflvote/jobs/job%j.out  # where to store the output (%j is the JOBID)
#SBATCH --error=/home/members/nakadam/dflvote/jobs/job%j.err  # where to store error messages
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=liver



PROJECT_ROOT=/home/members/nakadam/backdoor

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

cd $PROJECT_ROOT
pwd
# nvidia-smi  # show GPU status
# Define arrays based on src/main.py arguments
num_clients_arr=(8)
num_attackers_arr=(0 1 2 3 4)
groups_arr=(
    "[[0, 1, 2, 3, 4, 5, 6, 7]]"
    "[[0, 1, 2, 3], [4, 5, 6, 7]]"
    "[[0, 1, 4, 5], [2, 3, 6, 7]]"
    "[[0, 1], [2, 3], [4, 5], [6, 7]]"
    "[[0, 4], [1, 5], [2, 6], [3, 7]]"
    "[[0], [1], [2], [3], [4], [5], [6], [7]]"
)
datasets=("MNIST")
epochs=(40)
alpha=(0.1 1 10 100)
lrs=(0.001)
seeds=(123)

# attackers_arr, dists_arr, is_targeted_arr が未定義なので、仮に定義します
# attackers_arr=("")  # 必要に応じて値を設定
# dists_arr=("")      # 必要に応じて値を設定
# is_targeted_arr=("") # 必要に応じて値を設定

for num_clients in "${num_clients_arr[@]}"
do
  for num_attackers in "${num_attackers_arr[@]}"
  do
    for groups in "${groups_arr[@]}"
    do
      for dataset in "${datasets[@]}"
      do
        for epoch in "${epochs[@]}"
        do
          for lr in "${lrs[@]}"
          do
            for seed in "${seeds[@]}"
            do
                echo ">>> num_clients=$num_clients, num_attackers=$num_attackers, groups=$groups, dataset=$dataset, epoch=$epoch, lr=$lr, seed=$seed"
                uv run src/main.py \
                    --num_clients "$num_clients" \
                    --num_attackers "$num_attackers" \
                    --groups "$groups" \
                    --dataset "$dataset" \
                    --epoch "$epoch" \
                    --lr "$lr" \
                    --seed "$seed"
            done
          done
        done
      done
    done
  done
done

# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0
