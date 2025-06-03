import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# Path to the folder containing images
folder_path = '/kaggle/working/CelebAMask-HQ/face_parsing/test_results'

# List all image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Number of images to display
num_images = len(image_files)
columns = 5  # Number of columns in the grid
rows = math.ceil(num_images / columns)  # Calculate the required number of rows

# Set up the figure for displaying images
plt.figure(figsize=(15, rows * 3))

for i, img_file in enumerate(image_files):
    img_path = os.path.join(folder_path, img_file)
    img = mpimg.imread(img_path)
    
    # Add a subplot for each image
    plt.subplot(rows, columns, i + 1)
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.title(img_file, fontsize=8)  # Optional: Show the file name as title

# Adjust layout
plt.tight_layout()
plt.show()




import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
img = mpimg.imread("/kaggle/working/CelebAMask-HQ/face_parsing/test_results/0_mouth.jpg")
    
# Add a subplot for each image
plt.subplot(1, 1, 1)
plt.imshow(img)
plt.axis('off')  # Hide axes

img = mpimg.imread("/kaggle/working/CelebAMask-HQ/face_parsing/test_results/0_combined_mask.jpg")
    
# Add a subplot for each image
plt.subplot(1, 1, 1)
plt.imshow(img)
plt.axis('off')  # Hide axes
