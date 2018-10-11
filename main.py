from PIL import Image
from ipfml import image_processing

path = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection/img_train/final/appartAopt_00850.png'
img = Image.open(path)
blocks = image_processing.divide_in_blocks(img, (80, 80))
print(len(blocks))

blocks[60].show()