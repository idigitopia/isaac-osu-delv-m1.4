
def direct_fsm_pose_wrapper_v2(robot_position, robot_heading, robot_quat, target_xyz_position):
    """Finite State Machine (FSM) wrapper for the pose command.
    Target pose is a 2D pose in the world frame (x, y, z, heading).
    Returns the velocity commands for the robot in the body frame.
    
    Output Format:
    vel_commands = torch.zeros((env.num_envs, 4), device=env.unwrapped.device)
    
    Matrix Structure:
    Column 0: x_velocity (linear velocity along x-axis)
    Column 1: y_velocity (linear velocity along y-axis)
    Column 2: z_angular_velocity (angular velocity around z-axis)
    Column 3: stand_bit (1 if target reached, 0 otherwise)
    
    Example Tensor:
    vel_commands = torch.tensor([
        [x_vel, y_vel, z_ang_vel, stand_bit],  # Environment 1
        [x_vel, y_vel, z_ang_vel, stand_bit],  # Environment 2
        [x_vel, y_vel, z_ang_vel, stand_bit],  # Environment 3
        [x_vel, y_vel, z_ang_vel, stand_bit]   # Environment 4
    ], device=env.unwrapped.device)
    """

    num_envs = 1
    # target_xyz_position = target_pose[:, :3]

    # Get the current pose and heading of the robot in world frame
    current_robot_xyz_position = robot_position
    current_heading = robot_heading # angle in world frame
    current_quat = robot_quat # quaternion in world frame

    # Calculate the position and heading commands in the body frame
    localized_target_position = math_utils.quat_rotate_inverse(math_utils.yaw_quat(current_quat), target_xyz_position - current_robot_xyz_position)
    dist_to_target = torch.linalg.norm(localized_target_position[:, :2], dim=1)
    target_heading_delta = math_utils.wrap_to_pi(0 - current_heading)

    # # Check if the target has been reached
    target_reached_mask = (dist_to_target <= 0.1) & (target_heading_delta.abs() <= 0.1) # stand bit

    # # Initialize and Set the velocity commands ###
    vel_commands = torch.zeros((num_envs, 4), device="cpu")


    
    # # Clamp the vector delta to get linear velocity command.
    # vel_commands[~target_reached_mask, :2] = torch.clamp(localized_target_position[~target_reached_mask, :2], -1.0, 1.0)
    # # Clamp the heading delta to get angular velocity command.
    # vel_commands[~target_reached_mask, 2] = torch.clamp(target_heading_delta[~target_reached_mask], -1.0, 1.0)
    # # State 4: Stop (if reached target stop)
    # vel_commands[target_reached_mask, 3] = 1.0  # Set the stand-bit to 1
    # #################################################################################

    # Print the current pose and target pose
    # print(f"Target pose: {target_xyz_position} , 0 | Current pose: {current_robot_xyz_position[:, :2]}, {current_heading}", end="\r")
    # print(f"Vel commands: {vel_commands}")

        #################################################################################
    # Velocity Command Calculation Block (Two-Stage: Rotate then Move)
    #################################################################################

    # Define thresholds
    ANGLE_THRESHOLD = 0.3         # radians: how close to goal-facing to allow walking
    MAX_SPEED = 1.8               # max forward speed
    ANGULAR_MAX = 1.5             # max angular velocity

    # Compute rotation alignment mask
    aligned_mask = (torch.abs(target_heading_delta) < ANGLE_THRESHOLD) & ~target_reached_mask

    # Stage 1: Rotate in place until aligned
    vel_commands[~target_reached_mask, :3] = 0.0  # zero everything first
    vel_commands[~target_reached_mask, 2] = torch.clamp(
        target_heading_delta[~target_reached_mask], -ANGULAR_MAX, ANGULAR_MAX
    )

    # Stage 2: Once aligned, move forward at full speed
    if torch.any(aligned_mask):
        dir_vec = localized_target_position[aligned_mask, :2]
        norm = torch.norm(dir_vec, dim=1, keepdim=True) + 1e-6
        forward_vel = dir_vec / norm * MAX_SPEED
        vel_commands[aligned_mask, :2] = forward_vel

    # Stop if reached target
    vel_commands[target_reached_mask, 3] = 1.0

    #################################################################################


    return vel_commands
