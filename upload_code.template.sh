#!/bin/bash

# Add files to exclude from the upload.
EXCLUDES=(
  # Delete from target if present, and don't upload
  --exclude '*.pyc'
  --exclude '__pycache__'

  # Don't delete from target, and don't upload
  --exclude='outputs/'
  --filter='protect outputs/'

  --exclude='wandb/'
  --filter='protect wandb/'

  --exclude='logs/'
  --filter='protect logs/'

  --exclude='*.npy'
  --filter='protect *.npy'

  --exclude='*.mp4'
  --filter='protect *.mp4'
)

# These run in parallel. Add more target machines as needed.
rsync -avz --delete --delete-excluded "${EXCLUDES[@]}" /home/user0/workspace/IsaacLabDRAIL/ user1@123.123.123.123:/home/user1/workspace/IsaacLabDRAIL/ && echo "Upload to user1 machine finished" &
rsync -avz --delete --delete-excluded "${EXCLUDES[@]}" /home/user0/workspace/IsaacLabDRAIL/ user2@123.123.123.123:/home/user2/workspace/IsaacLabDRAIL/ && echo "Upload to user2 machine finished" &
rsync -avz --delete --delete-excluded "${EXCLUDES[@]}" /home/user0/workspace/IsaacLabDRAIL/ user3@123.123.123.123:/home/user3/workspace/IsaacLabDRAIL/ && echo "Upload to user3 machine finished" &

# Wait until upload is complete in all machines.
wait
