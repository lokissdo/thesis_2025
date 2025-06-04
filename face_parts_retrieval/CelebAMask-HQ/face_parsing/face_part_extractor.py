import os
import base64
from PIL import Image
from io import BytesIO

# Initialize the model
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Open the image from the local path
image = Image.open(image_test)

# Convert the image to base64
buffered = BytesIO()
image.save(buffered, format="JPEG")
image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

# Create the prompt
prompt = f"""
Given the my image, suggest which parts of the following list SHOULD CHANGE to have **${label_test}** Image: ['background','skin','nose', 'eye_glasses', 'left_eye', 'right_eye', 'left_brow', 'right_brow', 'left_ear', 'right_ear', 'mouth', 'upper_lip', 'lower_lip', 'hair', 'hat', 'ear_ring', 'necklace', 'neck', 'cloth',]. Your answer must FAITHFUL AND BE ONLY a list separated by comma. Maximun 5 parts
"""

# Generate the content using the model
response = model.generate_content([{'mime_type':'image/jpg', 'data': image_base64}, prompt])

# Output the response text
chosen_labels = [label.strip() for label in response.text.split(',')] 
print(chosen_labels)


import subprocess
chosen_labels_str = ' '.join(chosen_labels)

# Prepare the command to run main.py with the chosen_labels
command = f"python -u main.py --batch_size 1 --imsize 512 --version parsenet --train False --test_size 1 --chosen_labels {chosen_labels_str}"

# Execute the command
# Run the command and stream the output
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Continuously read and print the output in real-time
for line in process.stdout:
    print(line, end="")  # Print the line from stdout

# Wait for the process to complete
process.wait()

# Check for any errors from stderr
stderr_output = process.stderr.read()
if stderr_output:
    print(f"Errors: {stderr_output}")