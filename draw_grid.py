import random
from glob import glob
root = "mixup"

images = glob(f"{root}/**/*.png")
random.shuffle(images)
# Choose 1000 images
images = images[:1000]
# Make a grid of images that is suitable for a 16x9 screen
from PIL import Image
import math

# Define the aspect ratio for A4 paper
a4_ratio = 1.6

# Calculate the height and width of each image in the grid
image_width = 32  # Assuming each image is 32 pixels wide
image_height = int(image_width * a4_ratio)

# Calculate the number of rows and columns in the grid
num_rows = math.ceil(math.sqrt(len(images) / a4_ratio))
num_cols = math.ceil(len(images) / num_rows)
# Calculate the dimensions of the canvas
canvas_width = num_cols * image_width
canvas_height = num_rows * image_height
canvas = Image.new("RGB", (canvas_width, canvas_height))

# Paste the images onto the canvas in a grid pattern
for i, image_path in enumerate(images):
    row = i // num_cols
    col = i % num_cols
    image = Image.open(image_path)
    image = image.resize((image_width, image_height), Image.ANTIALIAS)
    canvas.paste(image, (col * image_width, row * image_height))

# Display or save the resulting grid
canvas.save("image_grid.png")  # Save the grid as an image file

