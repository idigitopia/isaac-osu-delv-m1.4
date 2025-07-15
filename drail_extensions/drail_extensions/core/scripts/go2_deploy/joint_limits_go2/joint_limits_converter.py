import math

import yaml


def radians_to_degrees(value):
    return {key: {k: round(math.degrees(v), 6) for k, v in val.items()} for key, val in value.items()}


# MULTIPLIER = 1.25 #(works on hw)
MULTIPLIER = 1.0
OUTPUT_FORMAT = "radian"  # "radian" or "degree"


def convert_yaml_to_degrees(input_file, output_file):
    with open(input_file) as file:
        data = yaml.safe_load(file)

    joint_order = [
        "front_right_hip",
        "front_right_thigh",
        "front_right_calf",
        "front_left_hip",
        "front_left_thigh",
        "front_left_calf",
        "rear_right_hip",
        "rear_right_thigh",
        "rear_right_calf",
        "rear_left_hip",
        "rear_left_thigh",
        "rear_left_calf",
    ]

    new_data = {"joint_metadata": {}}
    if "joint_limits" in data:
        for joint, limits in data["joint_limits"].items():
            for key, value in limits.items():
                if key != "all":
                    continue
                if OUTPUT_FORMAT == "radian":
                    new_data["joint_metadata"][joint] = {
                        "min": limits["all"]["min"] * MULTIPLIER,
                        "max": limits["all"]["max"] * MULTIPLIER,
                    }
                else:
                    new_data["joint_metadata"][joint] = {
                        "min": round(math.degrees(limits["all"]["min"]), 6) * MULTIPLIER,
                        "max": round(math.degrees(limits["all"]["max"]), 6) * MULTIPLIER,
                    }

    new_data["joint_limit_max"] = []
    new_data["joint_limit_min"] = []
    for joint in joint_order:
        new_data["joint_limit_max"].append(new_data["joint_metadata"][joint]["max"])
        new_data["joint_limit_min"].append(new_data["joint_metadata"][joint]["min"])

    with open(output_file, "w") as file:
        yaml.dump(new_data, file, default_flow_style=False)

    print(f"Converted YAML file saved to {output_file}")


# Example usage
convert_yaml_to_degrees("joint_limits_raw.yaml", "joint_limits_processed.yaml")
