# Language-Grounded FSM Navigation

## Overview
The tiamat_fsm_task_scripts package provides a pose_wrapper.py script that:
- Reads robot state (position, orientation, RGBD input)
- Contains a pretrained Go2 robot policy for velocity control (x, y planar velocity and z-axis angular velocity)
- Currently implements a basic FSM to navigate to random target poses

The new task is to develop a language-guided FSM that:
1. Takes natural language commands as input instead of target positions
2. Implements a pipeline to convert language commands into appropriate velocity controls
3. Successfully navigates the robot to complete the specified language task


#### Some Examples of Language Grounded Commands are
- #### Object-Based Commands
    - `"Go to the green box"`
    - `"Navigate to the red sphere"`
    - `"Move to the largest object"`

- #### Spatial Relationship Commands  
    - `"Go to the closest object"`
    - `"Move to the object on the left"`
    - `"Navigate to the furthest blue item"`

- #### Semantic/Contextual Commands
    - `"Go to the object that seems most interesting for Pythagoras"`
    - `"Move to something a mathematician would like"`
    - `"Navigate to the most unusual object"`



## Proposed Solution Architecture

The solution consists of a Finite State Machine (FSM) with the following phases:

### Phase 1: Environment Scanning
- Execute a complete 360° rotation while:
  - Capturing RGBD image frames at regular intervals
  - Recording robot heading at each capture point

### Phase 2: Scene Understanding & Target Selection
2.1. Object Detection & Description
   - Process RGBD images to detect and localize objects
   - Generate rich object descriptions using:
     - Dense captioning models for detailed visual features
     - Vision-Language Models (VLM) for semantic understanding
   - Create a structured mapping of:
     - Object IDs
     - 3D positions
     - Visual/semantic descriptions

2.2 Language-Guided Target Selection  
   - Feed object descriptions and language command to LLM
   - LLM analyzes semantic relationships to select target
   - Output target object ID and corresponding position

### Phase 3: Navigation Control
- Use selected target position to:
  - Generate appropriate velocity commands
  - Navigate robot to target location
  - Monitor completion status

# Deliverables

## Core Requirements
1. **Repository Setup**
   - Fork the provided repository to create your solution
   - Submit the fork URL to the evaluation team
   - All solution code should be committed to your fork

2. **Documentation**
   - Provide clear setup instructions in README.md including:
     - Step-by-step installation guide
     - Complete list of dependencies and versions
     - Instructions for running the system
     - Examples of supported commands

## Bonus Objectives 
1. **Containerization**
   - Create a Dockerfile for containerized deployment
   - Document container build and run instructions
   - Include container-specific environment setup

2. **Extended Analysis**
   - Document challenges and limitations encountered with:
     - Pre-trained LLM integration
     - Vision-language model performance
   - Propose creative solutions and improvements
   - Include experimental results and observations



#### FSM Input/Output Specification

**Input:**
```python
def language_grounded_fsm_wrapper(depth_image, rgb_image, robot_position, 
                                 robot_heading, robot_quat, language_command):
    """
    Args:
        depth_image (torch.Tensor): Shape (num_envs, H, W) - depth camera data
        rgb_image (torch.Tensor): Shape (num_envs, 3, H, W) - RGB camera data  
        robot_position (torch.Tensor): Shape (num_envs, 3) - world frame position
        robot_heading (torch.Tensor): Shape (num_envs,) - heading angle
        robot_quat (torch.Tensor): Shape (num_envs, 4) - quaternion orientation
        language_command (str): Natural language navigation instruction
    
    Returns:
        vel_commands (torch.Tensor): Shape (num_envs, 4) - velocity commands
            Column 0: x_velocity (linear velocity along x-axis)
            Column 1: y_velocity (linear velocity along y-axis) 
            Column 2: z_angular_velocity (angular velocity around z-axis)
            Column 3: stand_bit (1 if target reached, 0 otherwise)
    """
```

**Output Format:**
```python
vel_commands = torch.tensor([
    [x_vel, y_vel, z_ang_vel, stand_bit],  # Environment 1
    [x_vel, y_vel, z_ang_vel, stand_bit],  # Environment 2
    [x_vel, y_vel, z_ang_vel, stand_bit],  # Environment 3
    [x_vel, y_vel, z_ang_vel, stand_bit]   # Environment 4
], device=env.unwrapped.device)
```

### Integration Points

The new function should integrate seamlessly with the existing codebase:

```python
# Current implementation (pose_wrapper.py line ~180)
vel_cmds = direct_fsm_pose_wrapper(depth_image, rgb_image, robot_position, 
                                  robot_heading, robot_quat, target_pose)

# Target implementation  
vel_cmds = language_grounded_fsm_wrapper(depth_image, rgb_image, robot_position,
                                        robot_heading, robot_quat, language_command)
```

