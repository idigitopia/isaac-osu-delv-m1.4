import pandas as pd
import base64
import anthropic
import os



client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

def image_path(index):
    return f"tiamat_fsm_task_scripts/data/bounded_imgs/bounded_imgs_{index:03d}.png"


def encode_image_to_base64(image_path):
    """Convert image to base64 string for API call"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_response(image_path):
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare the message for Claude
    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": """Identify each 3D object inside of the red bounding boxes. You can ignore the boxes, just focus on what's inside of them. 

                        CRITICAL: Your response must contain ONLY the color and shape for each object in the exact format [color] [shape], separated by commas. Do not include any other words, explanations, introductions, or text whatsoever.
                        IT is very important that you order to objects. always describe from left to the right.
                        Required format: [color] [shape], [color] [shape]
                        Example: red cube, blue sphere

                        RESPOND WITH NOTHING BUT THE COLORS AND SHAPES."""
                    }
                ]
            }
        ]
    )
    
    return message.content[0].text.strip().lower()

def reorder_description(description, centroids_x):
    import numpy as np

    description = f"{description}."
    centroids_x = f"{centroids_x}"
    
    # One-liner ranking function
    get_ranking = lambda lst: np.argsort(np.argsort(lst))

    # centroids_x is a string of comma separated floats
    centroids_x_ranking = get_ranking([float(i) for i in centroids_x.split(",")])

    # description is a string of comma separated colors and shapes
    ordered_description = []
    for i in range(min(len(centroids_x_ranking), len(description.split(",")))):
        try:
            ordered_description.append(description.split(",")[centroids_x_ranking[i]])
        except:
            print("label and output mismatch for image")
            continue
    ordered_description = ", ".join(ordered_description)

    return ordered_description

def VLM_preds():
    csv_path = "tiamat_fsm_task_scripts/data/scan_data/scan_metadata_with_objects.csv"
    df = pd.read_csv(csv_path)

    descriptions = []
    for index in df['image_index']:
        try:
            path = image_path(index)
            description = get_response(path)
            reordered_description = reorder_description(description, df.iloc[index]["centroids_x"])
            descriptions.append(reordered_description)
        except:
            print(f"Error for image {index}")
            import pdb; pdb.set_trace()
        print(f"Image {index}: {reordered_description}")

    df['object_descriptions'] = descriptions    
    df.to_csv("tiamat_fsm_task_scripts/data/scan_data/scan_metadata_with_objects.csv", index=False)
    return csv_path


if __name__ == "__main__":
    VLM_preds()