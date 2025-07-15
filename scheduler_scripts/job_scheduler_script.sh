#!/bin/bash

# Run both Python scripts in parallel
MINICONDA_PATH="/home/ayubi/conda_setup/miniconda3"
REPO_PATH="/home/ayubi/issaclab_setup/IsaacLabDRAIL"
GIT_USERNAME="idgitiopia"
GIT_PASSWORD="TODO"

python $REPO_PATH/scheduler_scripts/fetch_and_run_job.py --miniconda-path $MINICONDA_PATH --repo-path $REPO_PATH --git-username $GIT_USERNAME --git-password $GIT_PASSWORD --enable-git-pull &
python $REPO_PATH/scheduler_scripts/allocate_resource.py &

# Wait for both processes to complete
wait