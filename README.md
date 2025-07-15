# TIAMAT TASK

This Checkpoint implements a proof of concept for Map then Navigate system for DARPA TIAMAT program. 

- We assume that all objects of interest are visibile from starting location of the robot.
- Task description will come from language input.
- The goal is to reach an object as specified by the task.

## Demo Video

https://drive.google.com/file/d/17_dWeyZcf70ps6EvN03f7MBk2hR7cUZX/view


**Method Description:**

- Go2 does an initial 360 degree scan of its environment, capturing RGBD images.
- The images are then processed:

  - We use Meta's SAM to create bounding boxes of detected objects.
  - We convert the pixel locations of each detected object along with the depth value at the location
    into world coordinates.
  - We send the images to Claude 3.5 Haiku using an API call to create descriptions of the objects.
  - We map each object to its description, and the world coordinates.
- Once processed, we are able to enter a natural language command to the program, and Go2 will navigate to
  the location. we use a LLM to figure out which object id/description pair matches  best with natural language command. 

Examples:

- "Go to the red pyramid"
- "Navigate to the pinkmost object in the room"
- "Find the blue cube"

### Codebase Setup FOR TIAMAT FSM TASK

<details>
<summary><strong>Set up Conda environment</strong></summary>

1. Run the setup script:

   ```bash
   ./setup_conda_env.sh
   ```

   You’ll be prompted to name your environment.
2. Activate it:

   ```bash
   conda activate <your_conda_env>
   ```
3. Pull files from Git LFS

   ```
   git-lfs pull
   ```
4. Install rsl_rl and drail_extension

   ```
   cd drail_extensions
   pip install -e . 
   ```

   ```
   cd drail_learning/rsl_rl_ashton
   pip install -e .
   ```
5. Set Anthropic API key in environment variable

   ```
   export ANTHROPIC_API_KEY='your-api-key-here'


   ```

</details>

### Run Command

To run the simulation with pretrained policy and natural language navigation:

```
python tiamat_fsm_task_scripts/pose_wrapper.py --task ashton-Go2-Velocity-FSM-Play --checkpoint drail_extensions/drail_extensions/research_ashton/resources/pretrained_policy/go2_velocity/e_spot_r4/model_best.pt --device cpu --real-time --enable_cameras

```
