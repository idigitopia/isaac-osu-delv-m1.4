import os

import yaml

joint_config_file_path = os.path.join(os.path.dirname(__file__), "joint_config.yaml")
with open(joint_config_file_path) as file:
    data = yaml.safe_load(file)

default_joint_pos = {}
default_joint_damping = {}
default_joint_armature = {}
default_spring_stiffness = {}
default_output_torque_limit = {}

joint_names = data.keys()
for name in joint_names:
    if "position" in data[name]:
        default_joint_pos[name] = data[name]["position"]
    if "damping" in data[name]:
        default_joint_damping[name] = data[name]["damping"]
    if "armature" in data[name]:
        default_joint_armature[name] = data[name]["armature"]
    if "stiffness" in data[name]:
        default_spring_stiffness[name] = data[name]["stiffness"]
    if "output_torque_limit" in data[name]:
        default_output_torque_limit[name] = data[name]["output_torque_limit"]

left_leg_joint_names = [
    "left_leg_hip_roll_joint",
    "left_leg_hip_yaw_joint",
    "left_leg_hip_pitch_joint",
    "left_leg_knee_joint",
    "left_leg_toe_a_joint",
    "left_leg_toe_b_joint",
]
right_leg_joint_names = [
    "right_leg_hip_roll_joint",
    "right_leg_hip_yaw_joint",
    "right_leg_hip_pitch_joint",
    "right_leg_knee_joint",
    "right_leg_toe_a_joint",
    "right_leg_toe_b_joint",
]
left_arm_joint_names = [
    "left_arm_shoulder_roll_joint",
    "left_arm_shoulder_pitch_joint",
    "left_arm_shoulder_yaw_joint",
    "left_arm_elbow_joint",
]
right_arm_joint_names = [
    "right_arm_shoulder_roll_joint",
    "right_arm_shoulder_pitch_joint",
    "right_arm_shoulder_yaw_joint",
    "right_arm_elbow_joint",
]
left_encoders = [
    "left_leg_shin_joint",
    "left_leg_tarsus_joint",
    "left_leg_heel_spring_joint",
    "left_leg_toe_pitch_joint",
    "left_leg_toe_roll_joint",
]
right_encoders = [
    "right_leg_shin_joint",
    "right_leg_tarsus_joint",
    "right_leg_heel_spring_joint",
    "right_leg_toe_pitch_joint",
    "right_leg_toe_roll_joint",
]
