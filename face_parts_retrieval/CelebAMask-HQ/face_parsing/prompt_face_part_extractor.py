import os
import base64
from PIL import Image
from io import BytesIO
import subprocess
import argparse
import google.generativeai as genai
import shutil







def extract_prompt(model_name="gemini-1.5-pro", prompt=None, image_path=None, overwrited_prompt=None):

    # image_test ="/kaggle/input/celebamaskhq/CelebAMask-HQ/CelebA-HQ-img/10.jpg"
    # label_test = "NOT BALD AND SMILLING"   
    # label_test = "I want to have smilling face and thick hair"
    # Initialize the model
    model = genai.GenerativeModel(model_name=model_name)

    # Open the image from the local path
    image = Image.open(image_path)

    # Convert the image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Create the prompt
    if overwrited_prompt is None:
        prompt = f"""
        Given the my image, suggest which parts of the following list SHOULD CHANGE to have **${prompt}** Image: ['background','skin','nose', 'eye_glasses', 'left_eye', 'right_eye', 'left_brow', 'right_brow', 'left_ear', 'right_ear', 'mouth', 'upper_lip', 'lower_lip', 'hair', 'hat', 'ear_ring', 'necklace', 'neck', 'cloth',]. Your answer must FAITHFUL AND BE ONLY a list separated by comma. Maximun 5 parts
        """
    else:
        prompt = overwrited_prompt

    # Generate the content using the model
    response = model.generate_content([{'mime_type':'image/jpg', 'data': image_base64}, prompt])

    # Output the response text
    chosen_labels = [label.strip() for label in response.text.split(',')] 

    return chosen_labels

def get_api_key():
    """Get API key from environment variable or command line argument"""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        api_key = input("Please enter your Google API key: ").strip()
        if not api_key:
            raise ValueError("API key is required. Set GOOGLE_API_KEY environment variable or provide it when prompted.")
    return api_key

def get_image_path():
    """Get image path from command line argument"""
    parser = argparse.ArgumentParser(description='Extract prompt for face parsing')
    parser.add_argument('--image_path', type=str, required=True, 
                       help='Path to the input image')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Prompt describing the desired change')
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    
    return args.image_path, args.prompt

def get_prompt_args():
    """Get both image_path and prompt from command line arguments"""
    parser = argparse.ArgumentParser(description='Extract prompt for face parsing')
    parser.add_argument('--image_path', type=str, required=True, 
                       help='Path to the input image')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Prompt describing the desired change')
    parser.add_argument('--model_name', type=str, default='gemini-1.5-pro',
                       help='Model name to use (default: gemini-1.5-pro)')
    parser.add_argument('--overwrited_prompt', type=str, default=None,
                       help='Custom prompt to overwrite the default prompt template')
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    
    return args.image_path, args.prompt, args.model_name, args.overwrited_prompt

if __name__ == "__main__":
    # Get API key and configure genai
    api_key = get_api_key()
    genai.configure(api_key=api_key)

    # Get image path and prompt from command line arguments
    image_path, prompt, model_name, overwrited_prompt = get_prompt_args()
    img_index = image_path.split('/')[-1].split('.')[0]
    
    chosen_labels = extract_prompt(
        model_name=model_name, 
        prompt=prompt, 
        image_path=image_path, 
        overwrited_prompt=overwrited_prompt
    )
    print(chosen_labels)
    chosen_labels_str = ' '.join(chosen_labels)


    test_img_dir = './Data_preprocessing/test_img'
    # Remove all images in ./Data_preprocessing/test_img
    for file in os.listdir(test_img_dir):
        os.remove(os.path.join(test_img_dir, file))
        
    # Copy the image to ./Data_preprocessing/test_img/0.jpg
    shutil.copy(image_path, os.path.join(test_img_dir, f'0.jpg'))

    # Prepare the command to run main.py with the chosen_labels
    command = f"python -u main.py --batch_size 1 --imsize 512 --version parsenet --train False --test_size 1 --chosen_labels {chosen_labels_str} --test_image_path {test_img_dir} --output_mask_name {img_index}_combined_mask.jpg"

    # Execute the command
    # Run the command and stream the output
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Continuously read and print the output in real-time
    for line in process.stdout:
        print(line, end="")  # Print the line from stdout

    # Wait for the process to complete
    process.wait()
        
    # Write the chosen_labels to a file in     /test_results
    with open(f'./test_results/{img_index}_chosen_labels.txt', 'w') as f:
        f.write(chosen_labels_str)
    
    # Check for any errors from stderr
    stderr_output = process.stderr.read()
    if stderr_output:
        print(f"Errors: {stderr_output}")



