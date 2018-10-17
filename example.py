from PIL import Image
from ipfml import image_processing

path = './images/test_img.png'
img = Image.open(path)
blocks = image_processing.divide_in_blocks(img, (10, 10))
print(len(blocks))