import json
import os
import anthropic
import re


def get_navigation_instruction(json_filepath: str, command: str) -> list:
    """
    Match a natural language command to an object in the JSON file and return navigation coordinates.
    
    Args:
        json_filepath: Path to the JSON file containing object descriptions and navigation data
        command: Natural language command (e.g., 'go to the red cube')
        
    Returns:
        Navigation array with 6 elements: [x, y, z, heading, object_id, image_id]
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If no matching object is found or API error occurs
        
    Note:
        Requires ANTHROPIC_API_KEY environment variable to be set
    """
    
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Read and stringify the JSON
    try:
        with open(json_filepath, 'r') as f:
            json_data = json.load(f)
            json_text = json.dumps(json_data, indent=2)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_filepath}")
    
    # Create the prompt with JSON as text
    prompt = f"""You are given a JSON file containing object descriptions and their corresponding navigation coordinates. Your task is to match a natural language command to the most appropriate object and return its navigation instruction.


            

            PROCESS:
            1. Parse the command to identify the target object (color + shape)
            2. Search the JSON data for the closest matching description
            3. Extract the navigation array for that object

            OUTPUT FORMAT: Return ONLY the navigation array with exactly 5 elements: [x, y, z, object_id, image_id]

            MATCHING RULES:
            - Prioritize exact color and shape matches
            - If multiple matches exist, return the first one found
            - If no exact match exists, find the closest match by color or shape
            - Ignore case sensitivity

            EXAMPLE:
            Command: 'go to the red cube'
            Response: [3.78, 1.77, 0.27, "merged_1", 1]

            CRITICAL: Return ONLY the array. No explanations, no additional text, no formatting beyond the array.

            INPUT: A natural language command: "{command}"
            
            JSON DATA:
            ```json
            {json_text}
            ```
            """

    # Get response from Claude
    try:
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        print(f"Response: {response_text}")
        
        
        navigation_array = json.loads(response_text)

        
        
        return navigation_array
        
    except Exception as e:
        raise ValueError(f"Error processing command with Claude API: {str(e)}")


## DEBUGGING CODE #########################################################
# if __name__ == "__main__":
#     json_filepath = "tiamat_fsm_task_scripts/data/scan_data/object_navigation_map.json"
#     command = "go to the purple cube"
    
#     try:
#         navigation = get_navigation_instruction(json_filepath, command)
#         print(f"Command: '{command}'")
#         print(f"Navigation: {navigation}")
#     except Exception as e:
#         print(f"Error: {e}")
#############################################################################